import os
import click
import yaml
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
import time
from torchvision.utils import save_image
from src.loss import get_loss_fn
from src.optimizer import get_optimizer
from diffusion.scheduler import make_beta_schedule, compute_alpha_schedule, q_sample
from src.models.unet_att_d import unet_att_d
from src.datatorch import ImageDataSSL
from src.utils import make_reproducible
from src.tools.logger import *
from tqdm import tqdm

def _set_lr(optim, lr: float):
    for g in optim.param_groups:
        g["lr"] = float(lr)
        
        
@click.command()
@click.option('--config', help='Path to YAML config file.')
def main(config):
    with open(config, "r") as config_file:
        params = yaml.safe_load(config_file)

    params_train = params["Train_Validate"]

    # Loss config
    loss_weights = params_train.get("loss_weights", {})
    alpha = loss_weights.get('alpha', 1.0)
    beta = loss_weights.get('beta', 0.1)
    gamma = loss_weights.get('gamma', 0.01)

    loss_fn = get_loss_fn(params_train["loss_fn"], alpha, beta, gamma)
    print(f"Using loss function: {params_train['loss_fn']}")

    # Reproducibility & device
    make_reproducible(seed=12)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Diffusion noise schedule (from Train_Validate.diffusion)
    diff_cfg = params_train.get("diffusion", {})
    timesteps = int(diff_cfg.get("timesteps", 1000))
    noise_sched = diff_cfg.get("noise_schedule", "cosine")
    betas = make_beta_schedule(
        timesteps,
        beta_start=float(diff_cfg.get("beta_start", 1e-6)),
        beta_end=float(diff_cfg.get("beta_end", 0.02)),
        scheduler_type=noise_sched
    ).to(device)
    alphas, alpha_bars = compute_alpha_schedule(betas)

    # Data

    catalog = pd.read_csv(params_train["train_csv_name"]).reset_index(drop=True)

    print("Creating training dataset...")
    train_dataset = ImageDataSSL(
        data_path=params_train["data_path"],
        log_dir=params_train["log_dir"],
        catalog=catalog,
        data_size=params_train["data_size"],
        buffer=params_train["buffer"],
        buffer_comp=params_train["buffer_comp"],
        usage="train",
        img_path_cols=params_train["img_path_cols"],
        apply_normalization=params_train["apply_normalization"],
        normal_strategy=params_train["normal_strategy"],
        stat_procedure=params_train["stat_procedure"],
        global_stats=params_train["global_stats"],
        trans=params_train["trans"],
        parallel=params_train["parallel"],
        downfactor=params_train["downfactor"],
        clip_val=params_train["clip_val"],
        nodata=params_train["nodata"]
    )
    train_loader = DataLoader(train_dataset, batch_size=params_train["train_batch"],
                              shuffle=True, num_workers=8, pin_memory=True)

    print("Creating validation dataset...")
    val_dataset = ImageDataSSL(
        data_path=params_train["data_path"],
        log_dir=params_train["log_dir"],
        catalog=catalog,
        data_size=params_train["data_size"],
        buffer=params_train["buffer"],
        buffer_comp=params_train["buffer_comp"],
        usage="validate",
        img_path_cols=params_train["img_path_cols"],
        apply_normalization=params_train["apply_normalization"],
        normal_strategy=params_train["normal_strategy"],
        stat_procedure=params_train["stat_procedure"],
        global_stats=params_train["global_stats"],
        trans=None,
        parallel=params_train["parallel"],
        downfactor=params_train["downfactor"],
        clip_val=params_train["clip_val"],
        nodata=params_train["nodata"]
    )
    val_loader = DataLoader(val_dataset, batch_size=params_train["validate_batch"],
                            shuffle=False, num_workers=8, pin_memory=True)

    # Model
    print("Instantiating UNet with timestep input...")
    cfg = params_train["model_kwargs"]
    in_channels = params_train["channels"]

    model = unet_att_d(
        in_channels=in_channels,
        filter_config=cfg["filter_config"],
        block_num=cfg["block_num"],
        dropout_rate=cfg.get("dropout_rate", 0),
        dropout_type=cfg.get("dropout_type", "traditional"),
        upmode=cfg.get("upmode", "deconv_2"),
        use_skipAtt=cfg.get("use_skipAtt", False),
        time_embedding_dim=cfg.get("time_embedding_dim", 128)
    )

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    # Optimizer + LR scheduler (from Train_Validate)
    # optimizer.py expects keys like:
    #   optimizer: adamw
    #   scheduler: cosinewarm | cosine | step | multistep | plateau | poly |
    #   onecycle
    optimizer, scheduler = get_optimizer(model, params_train)
    num_epochs = int(params_train["epoch"])
    # Warmup config (linear warmup from base_lr/10 -> base_lr)
    base_lr = float(params_train.get("learning_rate_init", 1e-3))
    warmup_epochs = int(params_train.get("warmup_period", 0))
    warmup_start_lr = base_lr / 10.0


    # Book-keeping
    best_val_loss = float('inf')
    best_epoch = 0
    save_dir = params_train.get("save_dir", "./checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "training_log.csv")
    with open(log_file, "w") as f:
        f.write("epoch,train_loss,val_loss,learning_rate,best_epoch_so_far,epoch_time_sec\n")

    print("Starting diffusion-based self-supervised training...")

    # Train/validate loop

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0.0
        # warmup
        if warmup_epochs > 0 and epoch < warmup_epochs:
            warm_lr = warmup_start_lr + (base_lr - warmup_start_lr) * ((epoch + 1) / warmup_epochs)
            _set_lr(optimizer, warm_lr)

        print(f"\n---> Epoch {epoch + 1}/{num_epochs} started")
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")

        for i, x0 in pbar:
            x0 = x0.to(device)
            B = x0.size(0)
            t = torch.randint(0, timesteps, (B,), device=device)
            noise = torch.randn_like(x0)
            x_t = q_sample(x0, t, alpha_bars, noise)
            pred_noise = model(x_t, t)
            loss = loss_fn(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % 10 == 0:
                print(f"[Epoch {epoch+1} | Batch {i}] Train Loss: {loss.item():.4f}")

            pbar.set_postfix({"batch_loss": loss.item()})

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        print(f"\n Validation for Epoch {epoch + 1}")
        val_pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Val {epoch+1}")

        with torch.no_grad():
            for i, x0 in val_pbar:
                x0 = x0.to(device)
                B = x0.size(0)
                t = torch.randint(0, timesteps, (B,), device=device)
                noise = torch.randn_like(x0)
                x_t = q_sample(x0, t, alpha_bars, noise)
                pred_noise = model(x_t, t)
                loss = loss_fn(pred_noise, noise)
                val_loss += loss.item()

                if i % 10 == 0:
                    print(f"[Val Epoch {epoch+1} | Batch {i}] Val Loss: {loss.item():.4f}")

                val_pbar.set_postfix({"val_loss": loss.item()})

        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        # Step LR scheduler
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        log_line = (
            f"[Epoch {epoch + 1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f} | "
            f"Best Epoch: {best_epoch} | Time: {epoch_time:.2f}s"
        )
        print(log_line)

        with open(log_file, "a") as f:
            f.write(f"{epoch + 1},{avg_train_loss:.4f},{avg_val_loss:.4f},{current_lr:.6f},{best_epoch},{epoch_time:.2f}\n")

        # Save checkpoints
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))

            # Visualize best sample of this epoch
            val_batch = next(iter(val_loader)).to(device)
            B = val_batch.size(0)
            t = torch.randint(0, timesteps, (B,), device=device)
            noise = torch.randn_like(val_batch)
            x_t = q_sample(val_batch, t, alpha_bars, noise)
            pred_noise = model.module(x_t, t) if isinstance(model, torch.nn.DataParallel) else model(x_t, t)

            loss_per_sample = F.mse_loss(pred_noise, noise, reduction='none').view(B, -1).mean(dim=1)
            best_idx = torch.argmin(loss_per_sample)
            best_t = t[best_idx].item()
            best_loss = loss_per_sample[best_idx].item()
            print(f"[VAL:Best] Epoch {epoch+1} - Best sample idx={best_idx}, timestep={best_t}, loss={best_loss:.4f}")

            alpha_bar_t = alpha_bars[best_t]
            sqrt_alpha_bar = alpha_bar_t.sqrt()
            sqrt_one_minus = (1 - alpha_bar_t).sqrt()
            x0_best = val_batch[best_idx:best_idx+1]
            x_t_best = x_t[best_idx:best_idx+1]
            pred_noise_best = pred_noise[best_idx:best_idx+1]
            recon_x0 = (x_t_best - sqrt_one_minus * pred_noise_best) / sqrt_alpha_bar
            recon_x0 = torch.clamp(recon_x0, 0.0, 1.0)

            def to_numpy_img(x):
                return x.detach().cpu().clamp(0, 1)

            base_name = f"epoch{best_epoch}_best_sample{best_idx}_t{best_t}"
            save_sample_dir = os.path.join(save_dir, "best_samples")
            os.makedirs(save_sample_dir, exist_ok=True)
            save_image(to_numpy_img(x0_best), os.path.join(save_sample_dir, f"{base_name}_x0.png"))
            save_image(to_numpy_img(x_t_best), os.path.join(save_sample_dir, f"{base_name}_xt.png"))
            save_image(to_numpy_img(pred_noise_best), os.path.join(save_sample_dir, f"{base_name}_prednoise.png"))
            save_image(to_numpy_img(recon_x0), os.path.join(save_sample_dir, f"{base_name}_reconstructed.png"))

        torch.save(model.state_dict(), os.path.join(save_dir, "latest_model.pth"))

    print("Training finished.")


if __name__ == '__main__':
    main()
