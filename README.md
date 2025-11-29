# Diffusion‑based Self‑Supervised Pretraining

This repository implements a **denoising diffusion model** to pretrain a U‑Net on multi‑channel satellite chips *without labels*.  
The model learns to **predict the noise** that was added to an input image at a random diffusion timestep. After pretraining, the U‑Net weights can be fine‑tuned on downstream tasks (e.g., semantic segmentation).

This README walks you through:

1. How the diffusion setup works (noise schedule, time conditioning, loss)
2. What the `unet_att_d` architecture looks like and what changed for SSL
3. How data loading, normalization and augmentation are done
4. How the optimizer, LR scheduler and losses are configured
5. Explanation of the `ssl_train.yaml` configuration
6. Pre-training dataset
7. How to run pre-training

---

## 1. High‑level idea

For each image `x₀` (4‑channel, 224×224 Planet NICIFI image chip):

1. Sample a diffusion timestep `t ~ Uniform({0…T‑1})`.
2. Sample Gaussian noise `ε ~ N(0, I)`.
3. Use a **fixed forward process** to create a noisy image `x_t = q_sample(x₀, t, ᾱ, ε)`.
4. Feed `(x_t, t)` into the U‑Net and predict the noise `ε̂ = model(x_t, t)`.
5. Train the model to **match the original noise**: `loss(ε̂, ε)`.


We never use labels here; the supervision signal is the noise we injected ourselves.

---

## 2. Model: `unet_att_d` (U‑Net with attention and time conditioning)

File: `src.models.unet_att_d.py`

### 2.1 Architecture overview

- Input: `in_channels` (default **4** channels from config)
- Resolution: works cleanly with **224×224** chips (with 6 down/up blocks).
- Encoder path (contracting):
  - 6 encoder blocks: `encoder_1 … encoder_6`
  - Each is a `ConvBlock` = stacked conv + BN + ReLU (+ optional dropout).
  - After each encoder block (except the last), a 2×2 max‑pool downsamples spatially.
- Bottleneck:
  - Last encoder `encoder_6` produces high‑dim features (e.g. 4096 channels with the provided `filter_config`).
- Decoder path (expansive):
  - 5 decoder blocks: `decoder_1 … decoder_5` with `UpconvBlock` (transposed conv / DUC) followed by `ConvBlock` with skip connections.
- Skip‑connection attention (optional):
  - If `use_skipAtt: true` in the config, each skip uses an `AdditiveAttentionBlock` gate instead of a raw concatenation. 

### 2.2 Diffusion‑specific changes vs. the original segmentation U‑Net

Compared to the original supervised segmentation version, the SSL/diffusion version has several key changes:

1. **Time‑step embedding**
   - The model now takes a timestep `t` as an extra input: `forward(self, inputs, t)`.
   - `t` is embedded using the sinusoidal function `get_timestep_embedding(t, time_embedding_dim)` from `scheduler.py`.
   - A small MLP (`self.time_mlp`) maps this embedding to the bottleneck channel dimension (`filter_config[5]`), then it is **added to the bottleneck features**:
     ```python
     t_emb = get_timestep_embedding(t, self.time_mlp[0].in_features)
     t_emb = self.time_mlp(t_emb).view(-1, e6.shape[1], 1, 1)
     e6 = e6 + t_emb
     ```
   - This conditions the U‑Net on “how noisy” the current input is.

2. **Noise‑prediction head instead of classifier**
   - The original segmentation model used a classifier head with `n_classes` output channels.
   - For diffusion SSL, the segmentation head is commented out and replaced by:
     ```python
     self.output_conv = nn.Conv2d(filter_config[0], in_channels, kernel_size=1)
     ```
   - This means the network outputs **one feature map per input channel** (e.g. 4) that is interpreted as the **predicted noise** `ε̂` instead of class logits.

3. **Loss is computed against the original noise, not labels**
   - The training loop (see below) passes the **sampled noise** `ε` as the target to the loss function.
   - No label tensor is used in the SSL diffusion setting; the dataset only returns images (`ImageDataSSL`).

Together, these changes turn a standard U‑Net into a **denoising diffusion backbone** that learns from unlabeled imagery.

---

## 3. Diffusion noise schedule

File: `diffusion.scheduler.py` 

### 3.1 Beta schedule: `make_beta_schedule`

```python
def make_beta_schedule(timesteps, beta_start=1e-6, beta_end=0.02, scheduler_type="cosine"):
    # scheduler_type: "linear" or "cosine"
```

- **`timesteps`**: total number of diffusion steps `T` (e.g. 1000).
- **`beta_start`, `beta_end`**:
  - minimum and maximum noise levels.
- **`scheduler_type`**:
  - `"linear"` – linearly interpolates betas from `beta_start` to `beta_end`.
  - `"cosine"` – cosine schedule from the IDDPM paper (Nichol & Dhariwal 2021).

The noise schedule used in training is controlled from the YAML config under `diffusion.noise_schedule` (see Section 7). In your current config this is set to `"linear"`.

### 3.2 Alpha schedule: `compute_alpha_schedule`

```python
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
```

- `alphas[t]` tells you how much of the original signal remains at step `t`.
- `alpha_bars[t] = Π(s = 0 → t) alpha[s]` is used in the closed‑form sampling of `x_t`.

### 3.3 Forward diffusion: `q_sample`

```python
def q_sample(x0, t, alpha_bars, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_alpha_bar = alpha_bars[t].sqrt().view(-1, 1, 1, 1)
    sqrt_one_minus = (1 - alpha_bars[t]).sqrt().view(-1, 1, 1, 1)
    return sqrt_alpha_bar * x0 + sqrt_one_minus * noise
```

- **Input**: clean image `x0`, timestep indices `t` (vector of shape `[B]`), and `alpha_bars`.
- **Output**: noisy image `x_t`.
- If you pass an explicit `noise` tensor, that same tensor can be reused as the **ground‑truth noise** target for training.

### 3.4 Timestep embeddings: `get_timestep_embedding`

- Standard sinusoidal timestep embedding:
  - Uses sine and cosine at varying frequencies.
  - Supports any embedding dimension `dim`; if `dim` is odd, it pads with one extra channel.

This embedding is fed into the U‑Net bottleneck via `time_mlp` as described above.

---

## 4. Data loading, normalization and augmentation

### 4.1 Dataset: `ImageDataSSL`

File: `src.datatorch.py`

`ImageDataSSL` is a lightweight `Dataset` for **self‑supervised diffusion training**:

- Expects a **catalog CSV** (see `train_csv_name` in the config) with at least:
  - one or more image path columns (default `["image"]`),
  - a `usage` column indicating `train` or `validate`.
- Key args (driven by `Train_Validate` config):
  - `data_path` – folder where chips live.
  - `data_size` – target chip size (e.g. 224).
  - `buffer` – padding buffer around chips.
  - `apply_normalization`, `normal_strategy`, `stat_procedure`, `global_stats` – passed to `process_img`.
  - `nodata`, `clip_val`, `downfactor` – controlling masking and resampling.

`__getitem__` does:

1. Look up the row by index and collect relative image paths.
2. Build full paths with `os.path.join(data_path, p)`.
3. Compute a read window with `get_buffered_window`.
4. Call `process_img(...)`, which runs normalization and optional clipping/masking.
5. Convert the `(H, W, C)` numpy array to a PyTorch tensor `(C, H, W)`.

It **returns only the image tensor**, not any labels, which fits the SSL diffusion scenario.

### 4.2 Normalization strategies

File: `src.normalization.py`

Controlled by:

- `apply_normalization: true/false`
- `normal_strategy: "min_max" | "z_value"`
- `stat_procedure: "gpb" | "gab" | "lpb" | "lab"`

Intuition:

- **`min_max`**: rescales bands to `[0, 1]` based on either global or local min/max, depending on `stat_procedure`.
- **`z_value`**: turns bands into z‑scores `(x - mean)/std` using global or local stats.
- **`stat_procedure`** picks how the mean/std or min/max are computed:
  - `"gpb"` – global per‑band stats (from `global_stats`).
  - `"gab"` – global “all‑bands” stats (one mean/std for all bands).
  - `"lpb"` – local per‑band, per‑image stats.
  - `"lab"` – local all‑bands stats (single mean/std over the full chip). In your config you use `normal_strategy: "min_max"` with `stat_procedure: "lab"`.

### 4.3 Data augmentation

File: `src.augmentation.py` (used indirectly via `process_img`)

Controlled by the following keys in `Train_Validate`:

- `scale_factor: [min_scale, max_scale]`
  - Random isotropic rescaling in this interval.
- `crop_strategy: "center" | "random" | ...`
  - Where the crop is taken after scaling; `"center"` keeps chips aligned.
- `rotation_degree: list of degrees`
  - Randomly rotates the chip by one of the listed angles (e.g. `[-180, -90, 90, 180]`).
- `sigma_range: [min_sigma, max_sigma]`
  - Std‑dev for Gaussian noise augmentation.
- `br_range: [min_delta, max_delta]`
  - Random brightness shift.
- `contrast_range: [low, high]`
  - Random contrast stretch.
- `bshift_gamma_range: [min_gamma, max_gamma]`
  - Brightness/gamma correction factor range.
- `patch_shift: true/false`
  - If `true`, applies a small random spatial shift to simulate patch misalignment.

These augmentations are applied **before** the diffusion corruption, so the model sees diverse clean images `x₀`.

---

## 5. Optimizer, LR scheduler and loss

### 5.1 Optimizer

File: `src.optimizer.py`

The helper `get_optimizer(config)` builds both an optimizer and an LR scheduler using the `Train_Validate` block.

Supported optimizers include `"adam"`, `"adamw"`, `"sgd"`, `"rmsprop"`, `"adagrad"`, `"radam"`, `"adabelief"`, `"sam"`, etc.  
In your **SSL config**, you are using:

```yaml
optimizer: "adamw"
learning_rate_init: 2e-4
weight_decay: 1e-4
betas: [0.9, 0.999]
eps: 1e-8
```

So practically it is **AdamW only** for this run (even though other options exist in the helper). If you set `optimizer: "sam"`, the code will wrap the base optimizer with a **SAM** (Sharpness‑Aware Minimization) wrapper, but that is not used in `ssl_train.yaml`.

Key optimizer config fields:

- `optimizer` – string name, e.g. `"adamw"`.
- `learning_rate_init` – initial learning rate (`lr`).
- `weight_decay` – L2 regularization (`weight_decay`).
- `betas` – only used for Adam/AdamW family.
- `eps` – numerical stability term for Adam/AdamW.
- `momentum`, `rho`, `adaptive` – only relevant if you switch to SAM or SGD‑like optimizers.

### 5.2 Learning‑rate schedulers

Still in `src.optimizer.py`, `get_scheduler` maps `scheduler` from the config to a PyTorch scheduler:

Config options (string in `Train_Validate.scheduler`):

- `"cosinewarm"` → `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`
  - Uses:
    - `t_0` – number of epochs before the first restart.
    - `t_mult` – factor to increase the cycle length after each restart.
    - `eta_min` – minimum LR.
- `"cosine"` → `torch.optim.lr_scheduler.CosineAnnealingLR`
  - Uses:
    - `T_max` – maximum number of iterations/epochs before LR hits `eta_min`.
    - `eta_min` – minimum LR.
- `"step"` → `torch.optim.lr_scheduler.StepLR`
  - Uses:
    - `step_size` – step interval in epochs.
    - `gamma` – multiplicative decay factor at each step.
- `"multistep"` → `torch.optim.lr_scheduler.MultiStepLR`
  - Uses:
    - `milestones` – list of epochs where LR decays.
    - `gamma` – multiplicative factor.
- `"plateau"` → `torch.optim.lr_scheduler.ReduceLROnPlateau`
  - Uses:
    - `patience` – how many epochs of no improvement before LR is reduced.
    - `factor` – multiplicative factor when reducing LR.
    - `min_lr` – lower bound on LR.
- `"poly"` → `torch.optim.lr_scheduler.PolynomialLR`
  - Uses:
    - `max_decay_steps` – roughly the number of steps until reaching `min_learning_rate`.
    - `min_learning_rate` – final LR.
    - `power` – curvature of the decay (1.0 ≈ linear).
- `"onecycle"` → `torch.optim.lr_scheduler.OneCycleLR`
  - Uses:
    - `max_lr` – `learning_rate_init` from config.
    - `pct_start`, `div_factor`, `final_div_factor` – shape of the one‑cycle schedule.
- `"none"` or empty → no scheduler (constant LR).

In your current config you have:

```yaml
scheduler: "cosinewarm"
t_0: 30
t_mult: 2
eta_min: 1e-5
```

So LR will follow a **cosine‑annealing with warm restarts**, restarting first at epoch 30 and then with progressively longer cycles.

### 5.3 Loss functions for noise prediction

File: `src.loss.py`

`get_loss_fn(name, alpha, beta, gamma)` returns the loss function used to compare predicted noise `ε̂` and true noise `ε`:

Available options:

- `"mse"` – Mean Squared Error (`F.mse_loss`)
- `"l1"` – L1 / MAE (`F.l1_loss`)
- `"charbonnier"` – Charbonnier loss:
  \\( \sqrt{(ε̂ - ε)^2 + \epsilon^2} \\) averaged over all pixels.
- `"ssim"` – (1 − SSIM) over channels; good for perceptual structure.
- `"perceptual"` – VGG‑based perceptual L1 loss.
- `"hybrid_light"` – custom mix: `α·MSE + β·SSIM + γ·KL` between global normal distributions of pred/target.

In `ssl_train.yaml` you currently set:

```yaml
loss_fn: "mse"
loss_weights:
  alpha: 0.9
  beta: 0.1
  gamma: 0.0
```

- For pure `"mse"`, `alpha/beta/gamma` are not used.
- They become relevant if you switch to `"hybrid_light"` to balance the components.

**Important in SSL context:** the *target* is the **true noise** you sampled and passed to `q_sample`, not a semantic label.

A typical training step looks like (pseudo‑code):

```python
x0 = batch_img                     # from ImageDataSSL
t  = random_timesteps(batch_size)  # 0 … T-1
eps = torch.randn_like(x0)         # ground-truth noise

x_t = q_sample(x0, t, alpha_bars, noise=eps)
eps_hat = model(x_t, t)

loss = criterion(eps_hat, eps)
loss.backward()
optimizer.step()
scheduler.step()
```

---

## 6. `ssl_train.yaml` – configuration

File: `config.ssl_train.yaml`

The YAML has a top‑level block `Train_Validate:` which drives the entire SSL diffusion training.

Below we explain each group of options.

### 6.1 Paths / data

```yaml
data_path: "TBD"
log_dir: "TBD"
train_csv_name: "TBD/catalog.csv"
```

- **`data_path`** – root directory containing the image chips referenced in `catalog.csv`.
- **`log_dir`** – where training logs, checkpoints and any debug files are saved.
- **`train_csv_name`** – full path to the catalog CSV. Must include paths (relative to `data_path`) and a `usage` column marking `train` or `validate` rows.

### 6.2 Dataset / normalization

```yaml
data_size: 224
buffer: 0
buffer_comp: 0
global_stats: null
catalog_index: null
trans: null
parallel: null

img_path_cols: ["image"]
apply_normalization: true
normal_strategy: "min_max"
stat_procedure: "lab"
downfactor: 32
clip_val: 0
nodata: []
```

- **`data_size`** – final chip size that the model expects (height = width = 224).
- **`buffer`** – extra pixels around each chip when reading from raster (0 = none).
- **`buffer_comp`** – buffer used when making composites; not used in basic SSL but kept for consistency.
- **`global_stats`** – path or object with global statistics (if you use `gpb`/`gab` strategies). `null` means compute per‑image stats only.
- **`catalog_index`**, **`trans`**, **`parallel`** – hooks for more advanced pipelines; currently unused (`null`).

- **`img_path_cols`** – list of catalog column names pointing to image file paths. For single‑image chips, this can just be `["image"]`.
- **`apply_normalization`** – turn normalization on/off.
- **`normal_strategy`** – `"min_max"` or `"z_value"` (see Section 4.2).
- **`stat_procedure`** – `"lab"` in your config (local all‑band stats).
- **`downfactor`** – downscaling factor used internally when computing statistics.
- **`clip_val`** – if > 0, pixel values are clipped to `[-clip_val, clip_val]` (after normalization).
- **`nodata`** – list of nodata values per band; empty list means no explicit nodata masking.

### 6.3 Data augmentation

```yaml
scale_factor: [0.75, 1.5]
crop_strategy: "center"
rotation_degree: [-180, -90, 90, 180]
sigma_range: [0.03, 0.07]
br_range: [-0.02, 0.02]
contrast_range: [0.9, 1.2]
bshift_gamma_range: [0.2, 2.0]
patch_shift: true
```

- **`scale_factor`** – random zoom in/out range.
- **`crop_strategy`** – where to crop after scaling (`"center"` keeps chip centered).
- **`rotation_degree`** – list of allowed rotation angles (degrees); one is chosen at random.
- **`sigma_range`** – Gaussian noise sigma range.
- **`br_range`** – brightness shift interval.
- **`contrast_range`** – random contrast factor interval.
- **`bshift_gamma_range`** – gamma correction range.
- **`patch_shift`** – if `true`, randomly shifts patches slightly to mimic misalignment.

### 6.4 Batching

```yaml
train_batch: 40
validate_batch: 20
```

- **`train_batch`** – batch size for training.
- **`validate_batch`** – batch size for validation/eval.

### 6.5 Diffusion noise schedule (forward process)

```yaml
diffusion:
  timesteps: 1000
  noise_schedule: "linear"     # or "cosine"
  beta_start: 1e-6
  beta_end: 0.02
```

- **`timesteps`** – total number of diffusion steps `T`. Used when creating betas via `make_beta_schedule`.
- **`noise_schedule`** – `"linear"` or `"cosine"` (passed as `scheduler_type` to `make_beta_schedule`).
- **`beta_start`, `beta_end`** – minimum and maximum beta values.

These control how quickly signal is destroyed across timesteps during pretraining.

### 6.6 Model

```yaml
model: "unet_att_d"
channels: 4
save_dir: "TBD"

model_kwargs:
  filter_config: [64, 256, 512, 1024, 2048, 4096]
  block_num: [2, 2, 2, 2, 2, 2]
  dropout_rate: 0.1
  dropout_type: "traditional"
  upmode: "deconv_2"
  use_skipAtt: true
  time_embedding_dim: 128
```

- **`model`** – model class to instantiate (`unet_att_d` from `unet_att_d.py`).
- **`channels`** – number of input/output channels (e.g. 4 for your satellite stack).
- **`save_dir`** – where checkpoints and final weights are stored.

`model_kwargs` are passed as keyword arguments to `unet_att_d`:

- **`filter_config`** – channel widths at each encoder/decoder level.
- **`block_num`** – number of conv layers per block at each level.
- **`dropout_rate`** – dropout probability inside `ConvBlock`s.
- **`dropout_type`** – `"traditional"` (nn.Dropout) or `"spatial"` (nn.Dropout2d).
- **`upmode`** – upsampling type in `UpconvBlock` (`"deconv_2"` = overlapping transposed conv).
- **`use_skipAtt`** – whether to use attention gates on skip connections.
- **`time_embedding_dim`** – dimension of the sinusoidal timestep embedding before the MLP.

### 6.7 Training loop

```yaml
epoch: 100
```

- **`epoch`** – number of training epochs.

The actual per‑epoch training loop should:
- iterate over `ImageDataSSL` dataloader,
- sample timesteps `t` and noise `ε`,
- call `q_sample` and `model(x_t, t)`,
- compute the loss and update the optimizer/scheduler.

(See pseudo‑code in Section 5.3.)

### 6.8 Optimizer

```yaml
optimizer: "adamw"
learning_rate_init: 2e-4
weight_decay: 1e-4
betas: [0.9, 0.999]
eps: 1e-8
# momentum/rho/adaptive are not used for AdamW (SAM-only), so omitted
```

As described in Section 5.1, this builds an AdamW optimizer with those hyper‑parameters.

### 6.9 LR scheduler

```yaml
scheduler: "cosinewarm"        # options: cosinewarm | cosine | step | multistep | plateau | poly | onecycle
t_0: 30
t_mult: 2
eta_min: 1e-5
```

- **`scheduler`** – which LR schedule strategy to use (Section 5.2).
- **`t_0`**, **`t_mult`**, **`eta_min`** – arguments for `CosineAnnealingWarmRestarts` when `scheduler: "cosinewarm"`.

Commented examples in the YAML show how you would configure other schedulers (`cosine`, `step`, `plateau`, `poly`, etc.) if you switch types.

### 6.10 Loss

```yaml
loss_fn: "mse"                  # mse | l1 | charbonnier | ssim | perceptual
loss_weights:
  alpha: 0.9
  beta: 0.1
  gamma: 0.0
```

- **`loss_fn`** – name passed to `get_loss_fn` in `loss.py`.
- **`loss_weights.alpha/beta/gamma`** – weighting factors, mainly used by `"hybrid_light"`. For `"mse"` they can be left as defaults.

### 6.11 Checkpointing / misc

```yaml
checkpoint_interval: 1
warmup_period: 5
```

- **`checkpoint_interval`** – save a model checkpoint every N epochs.
- **`warmup_period`** – number of warmup epochs you plan to treat differently (e.g. LR warmup, logging); exact behavior depends on the training script.

---
## 6. Pre-training dataset

This diffusion setup is designed for **unlabeled, multi-channel satellite chips**. You only need images; no masks or class labels are required.

### 6.1. What the model expects

- **Chip size**: by default, `224 × 224` pixels (controlled by `data_size` in `ssl_train.yaml`).
- **Channels**: `channels` in the config (e.g., 4) —
- **File format**: the code assumes raster files readable by `rasterio` / GDAL (e.g., `.tif`).

Each chip is read on-the-fly from disk; there is **no requirement** that you pre-stack them into giant arrays.

### 6.2. Dataset for tutorial

100k 224 by 224 Planet NICIFI images (RGB, NIR)

### 6.3. Catalog CSV (`train_csv_name`)

The dataset is indexed by a **catalog CSV**, referenced by `Train_Validate.train_csv_name` in `ssl_train.yaml`.  
At minimum, this CSV must contain:

- an `image` column (or more, matching `img_path_cols`) with **relative paths** to the chip files, and  
- a `usage` column with values `train` or `validate`.

Example `catalog.csv`:

```csv
image,usage
tile_000001.tif,train
tile_000002.tif,train
tile_000101.tif,validate
tile_000102.tif,validate
```

If your chips live under `data_path = /path/to/chips`, then the files above should exist as:

```text
/path/to/chips/tile_000001.tif
/path/to/chips/tile_000002.tif
...
```

### 6.4. Recommended directory structure

A typical layout might look like:

```text
project_root/
├─ configs/
│  └─ ssl_train.yaml
├─ data/
│  ├─ pretrain_data_random_1/
│  │  ├─ catalog.csv
│  │  ├─ tile_000001.tif
│  │  ├─ tile_000002.tif
│  │  └─ ...
├─ src/
│  ├─ datatorch.py
│  ├─ unet_att_d.py
│  ├─ scheduler.py
│  ├─ optimizer.py
│  ├─ loss.py
│  └─ ...
└─ work_dir/
   └─ exp_pretrain_random_900k_5/
      ├─ logs/
      ├─ checkpoints/
      └─ tensorboard/
```

Adjust paths in `ssl_train.yaml` to match your own layout.

---

## 7. How to run pre-training

- Docker image
- python train.py --config configs/ssl_train.yaml


Once pretraining finishes, you can load the U-Net backbone weights into a supervised model and fine-tune on labeled downstream tasks.
