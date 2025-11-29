import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch import optim
from torch.nn import init
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter

import sys
import os
from pathlib import Path
from datetime import datetime

from .train import *
from .validate import *
from .evaluate import *
from .predict import *
from .models import *
from .optimizer import *
from .sync_batchnorm import convert_model
from .sync_batchnorm import DataParallelWithCallback
from IPython.core.debugger import set_trace


class PolynomialLR(_LRScheduler):
    """
    Polynomial learning rate decay until step reach to max_decay_step
    
    Args:
        optimizer : Optimizer 
            Wrapped optimizer.
        max_decay_steps (int): 
            after this step, we stop decreasing learning rate
        min_learning_rate : float 
            scheduler stopping learning rate decay, value of learning rate must 
            be this value
        power : float 
            The power of the polynomial.
    """

    def __init__(self, optimizer, max_decay_steps, min_learning_rate=1e-5, 
                 power=1.0):

        if max_decay_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')

        self.max_decay_steps = max_decay_steps
        self.min_learning_rate = min_learning_rate
        self.power = power
        self.last_step = 0

        super().__init__(optimizer)

    def get_lr(self):
        if self.last_step > self.max_decay_steps:
            return [self.min_learning_rate for _ in self.base_lrs]

        return [(base_lr - self.min_learning_rate) *
                ((1 - self.last_step / self.max_decay_steps) ** self.power) +
                self.min_learning_rate for base_lr in self.base_lrs]

    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1

        if self.last_step <= self.max_decay_steps:
            decay_lrs = [
                (base_lr - self.min_learning_rate) *
                ((1 - self.last_step / self.max_decay_steps) ** self.power) +
                self.min_learning_rate for base_lr in self.base_lrs
            ]

            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr


def get_optimizer(optimizer, model, params, lr, momentum, weight_decay=0.0):
    """
    Get an instance of the specified optimizer with the given parameters.

    Args:
        optimizer (str): The name of the optimizer. Options: 
                              "sgd", "nesterov", "adam", "amsgrad" and "sam".
        model(nn.Module): Initialized model.
        params (iterable): The parameters to optimize.
        lr (float): The learning rate.
        momentum (float): The momentum factor for optimizers that support it.
        weight_decay (float, optional): weight decay coefficient. Default: 0.

    Returns:
        torch.optim.Optimizer: An instance of the specified optimizer with the 
        given parameters.
    """
    optimizer = optimizer.lower()
    
    if optimizer == 'sgd':
        return torch.optim.SGD(params, lr, momentum=momentum)
    elif optimizer == 'nesterov':
        return torch.optim.SGD(params, lr, momentum=momentum, nesterov=True)
    elif optimizer == 'adam':
        return torch.optim.Adam(params, lr)
    elif optimizer == "adamw":
        return torch.optim.AdamW(params, lr, weight_decay=weight_decay)
    elif optimizer == 'amsgrad':
        return torch.optim.Adam(params, lr, amsgrad=True)
    elif optimizer == 'sam':
        base_optimizer = optim.SGD
        return SAM(model.parameters(), base_optimizer, lr=lr, 
                   momentum=momentum)
    else:
        raise ValueError(
            f"{optimizer} currently not supported, please customize your \
            optimizer in compiler.py"
        )

def init_weights(model, init_type="normal", gain=0.02):
    """ 
    Recursively initialize the network weights.

    Args:
        model (torch.nn.Module): The initialized model.
        init_type (str): The initialization type.
        gain (float): The scaling factor for the initialized weights.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            if init_type == "normal":
                init.normal_(m.weight, mean=0.0, std=gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight, a=0, mode="fan_out")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight, gain=gain)
            else:
                raise NotImplementedError(f"initialization method {init_type} is not implemented.")

            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias, val=0.0)

        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1.0)
            init.constant_(m.bias, 0.0)

    print(f"Initialized network with {init_type}.")


class ModelCompiler:
    def __init__(self, model, working_dir, out_dir, buffer, class_mapping, gpu_devices=[0], 
                 use_sync_bn=False, model_init_type="kaiming", params_init=None, freeze_params=None):
        """
        Compiler of specified model
        
        Args:
            model (ordered Dict): initialized model either vanilla or pre-trained depending on
                                  the argument 'params_init'.
            working_dir (str): General Directory to store output from any experiment.
            out_dir (str): specific output directory for the current experiment.
            buffer (int): distance to sample edges not considered in optimization
            class_mapping (dict): A dictionary mapping class indices to class names.
            gpu_devices (list): indices of gpu devices to use
            use_sync_bn (binary): If True, applies synchronized BN on distributed models.
            params_init (str or None) Path to the saved model parameters. If set to 'None', 
                          a vanilla model will be initialized.
            freeze_params (list): list of integers that show the index of layers in a pre-trained
                                    model (on the source domain) that we want to freeze for fine-tuning
                                    the model on the target domain used in the model-based transfer learning.
        """

        self.working_dir = working_dir
        self.out_dir = out_dir
        self.buffer = buffer
        self.class_mapping = class_mapping
        self.num_classes = len(self.class_mapping)
        self.gpu_devices = gpu_devices
        self.use_sync_bn = use_sync_bn
        self.model_init_type = model_init_type
        self.params_init = params_init
        self.checkpoint_dirpath = None
        
        self.model = model
        self.model_name = self.model.__class__.__name__
        
        if self.params_init:
            self.load_params(self.params_init, freeze_params)
            print("--------- Pre-trained model compiled successfully ---------")
        else:
            #init_weights(self.model, self.model_init_type, gain=0.01)
            print("--------- Vanilla Model compiled successfully ---------")
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if self.device.type == "cuda":
            print("----------GPU available----------")
            self.gpu = True
            if self.gpu_devices:
                torch.cuda.set_device(self.gpu_devices[0])
                if len(self.gpu_devices) > 1:
                    if self.use_sync_bn:
                        self.model = convert_model(self.model)
                        self.model = DataParallelWithCallback(self.model, device_ids=self.gpu_devices)
                    else:
                        self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_devices)
            self.model = self.model.to(self.device)
        else:
            print('----------No GPU available, using CPU instead----------')
            self.gpu = False            
        
        num_params = sum([p.numel() for p in self.model.parameters() 
                          if p.requires_grad])
        print("total number of trainable parameters: {:2.1f}M"\
              .format(num_params / 1000000))

    
    def load_params(self, dir_params, freeze_params):

        in_params = torch.load(dir_params)

        ## overwrite model entries with new parameters
        model_dict = self.model.state_dict()

        if "module" in list(in_params.keys())[0]:
            in_params_filter = {k[7:]: v.cpu() for k, v in in_params.items() 
                               if k[7:] in model_dict}
        else:
            in_params_filter = {k: v.cpu() for k, v in in_params.items() 
                               if k in model_dict}
        model_dict.update(in_params_filter)
        # load new state dict
        self.model.load_state_dict(model_dict)

        # free some layers
        if freeze_params is not None:
            for i, p in enumerate(self.model.parameters()):
                if i in freeze_params:
                    p.requires_grad = False


    def fit(self, train_dataset, val_dataset, epochs, optimizer_name, lr_init, 
            lr_policy, criterion, momentum=None, resume=False, checkpoint_interval=20, 
            resume_epoch=None, early_stopping_patience=10, min_delta=0.001, 
            warmup_period=10, **kwargs):
        """)
        Train the model on the provided datasets.

        Args:
            trainDataset: The loaded training dataset.
            valDataset: The loaded validation dataset.
            epochs (int): The number of epochs to train.
            optimizer_name (str): The name of the optimizer to use.
            lr_init (float): The initial learning rate.
            lr_policy (str): The learning rate policy.
            criterion: The loss criterion.
            momentum (float, optional): The momentum factor for the optimizer 
                                        (default: None).
            checkpoint_interval (int): How often to save a checkpoint during training. 
                                       Default to 20.
            resume (bool, optional): Whether to resume training from a checkpoint 
                                     (default: False).
            resume_epoch (int, optional): The epoch from which to resume training 
                                          (default: None).
            early_stopping_patience (int): The number of epochs to wait for an improvement 
                                           in the validation loss before stopping the training 
                                           early (default: 10).
            min_delta (float): The minimum change in the validation loss to qualify as 
                               an improvement (default: 0.001).
            warmup_period (int): Numer of epochs at the initial training phase before enabling 
                                 the early stopping consideration.
            **kwargs: Additional arguments specific to certain learning rate policies.

        Returns:
            None
        """

        working_dir = self.working_dir
        out_dir = self.out_dir
        
        model_dir = "{}/{}/{}_ep{}".format(working_dir, out_dir, self.model_name, epochs)
        
        if not os.path.exists(Path(working_dir) / out_dir / model_dir):
            os.makedirs(Path(working_dir) / out_dir / model_dir)
        
        self.checkpoint_dir = Path(working_dir) / out_dir / model_dir / "chkpt"
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        
        os.chdir(Path(working_dir) / out_dir / model_dir)

        print('----------------------- Start training -----------------------')
        start = datetime.now()

        # Tensorboard writer setting
        writer = SummaryWriter('../')

        lr = lr_init
        # lr_decay = lr_decay if isinstance(lr_decay,tuple) else (lr_decay,1)
        weight_decay = kwargs.get("weight_decay", 0.0)
        optimizer = get_optimizer(optimizer_name, 
                                  self.model, 
                                  filter(lambda p: p.requires_grad, self.model.parameters()), 
                                  lr, 
                                  momentum, 
                                  weight_decay)

        # initialize different learning rate scheduler
        lr_policy = lr_policy.lower()
        if lr_policy == "StepLR".lower():
            step_size = kwargs.get("step_size", 3)
            gamma = kwargs.get("gamma", 0.98)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                        step_size=step_size, 
                                                        gamma=gamma)

        elif lr_policy == "MultiStepLR".lower():
            milestones = kwargs.get("milestones", [15, 25, 35, 50, 70, 90, 120, 150, 200])
            gamma = kwargs.get("gamma", 0.5)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                             milestones=milestones, 
                                                             gamma=gamma)

        elif lr_policy == "ReduceLROnPlateau".lower():
            mode = kwargs.get('mode', 'min')
            factor = kwargs.get('factor', 0.8)
            patience = kwargs.get('patience', 3)
            threshold = kwargs.get('threshold', 0.0001)
            threshold_mode = kwargs.get('threshold_mode', 'rel')
            min_lr = kwargs.get('min_lr', 3e-6)
            verbose = kwargs.get('verbose', True)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                   mode=mode, 
                                                                   factor=factor, 
                                                                   patience=patience, 
                                                                   threshold=threshold, 
                                                                   threshold_mode=threshold_mode, 
                                                                   min_lr=min_lr, 
                                                                   verbose=verbose)
        
        elif lr_policy == "PolynomialLR".lower():
            max_decay_steps = kwargs.get('max_decay_steps', 100)
            min_learning_rate = kwargs.get('min_learning_rate', 1e-5)
            power = kwargs.get('power', 0.8)
            scheduler = PolynomialLR(optimizer, 
                                     max_decay_steps=max_decay_steps, 
                                     min_learning_rate=min_learning_rate,
                                     power=power)

        elif lr_policy == "CyclicLR".lower():
            base_lr = kwargs.get('base_lr', 3e-5)
            max_lr = kwargs.get('max_lr', 0.01)
            step_size_up = kwargs.get('step_size_up', 1100)
            mode = kwargs.get('mode', 'triangular')
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 
                                                          base_lr=base_lr, 
                                                          max_lr=max_lr, 
                                                          step_size_up=step_size_up, 
                                                          mode=mode)
        else:
            scheduler = None

        train_loss = []
        val_loss = []
        
        if resume:
            model_state_file = os.path.join(self.checkpoint_dir,
                                            f"{resume_epoch}_checkpoint.pth.tar")
            
            # Resume the model from the specified checkpoint in the config file.
            if os.path.exists(model_state_file):
                checkpoint = torch.load(model_state_file)
                print(f"Checkpoint file loaded from {model_state_file}")
                resume_epoch = checkpoint["epoch"]
                scheduler.load_state_dict(checkpoint["scheduler"])
                self.model.load_state_dict(checkpoint["state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                train_loss = checkpoint["train_loss"]
                val_loss = checkpoint["val_loss"]
                best_val_loss = checkpoint.get("best_val_loss", float('inf'))
                epochs_no_improve = checkpoint.get("epochs_no_improve", 0)
                early_stop = checkpoint.get("early_stop", False)
            else:
                raise ValueError(f"{model_state_file} does not exist")

        if resume:
            iterable = range(resume_epoch, epochs)
        else:
            iterable = range(epochs)
            # early stopping parameters
            best_val_loss = float('inf')
            epochs_no_improve = 0
            early_stop = False

        for t in iterable:

            print(f"[{t + 1}/{epochs}]")

            # start fitting
            start_epoch = datetime.now()
            train_one_epoch(train_dataset, self.model, criterion, optimizer, scheduler,
                  self.num_classes, device=self.device, train_loss=train_loss)
            validate_one_epoch(val_dataset, self.model, criterion, self.buffer, 
                     device=self.device, val_loss=val_loss)

            # Early Stopping Check
            if t > warmup_period:
                current_val_loss = val_loss[t]
                if current_val_loss < best_val_loss - min_delta:
                    best_val_loss = current_val_loss
                    epochs_no_improve = 0
                    torch.save(self.model.state_dict(), 
                               os.path.join(self.checkpoint_dir, 'best_model.pth'))
                else:
                    epochs_no_improve += 1
            
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {t+1}")
                early_stop = True
                break
            
            # Update the scheduler
            if lr_policy in ["StepLR".lower(), "MultiStepLR".lower()]:
                scheduler.step()
                print(f"LR: {scheduler.get_last_lr()}")

            if lr_policy == "ReduceLROnPlateau".lower():
                scheduler.step(val_loss[t])

            if lr_policy == "PolynomialLR".lower():
                scheduler.step(t)
                print(f"LR: {optimizer.param_groups[0]['lr']}")

            # time spent on single iteration
            print('time:', (datetime.now() - start_epoch).seconds)

            # Adjust index and logger to resume status and save checkpoits in 
            # defined intervals.
            # index = t-resume_epoch if resume else t

            writer.add_scalars(
                "Loss",
                {"train_loss": train_loss[t],
                 "val_loss": val_loss[t]},
                t + 1
            )

            if (t + 1) % checkpoint_interval == 0:
                torch.save(
                    {
                        "epoch": t + 1,
                        "state_dict": self.model.module.state_dict() if len(self.gpu_devices)>1 else \
                            self.model.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "best_val_loss": best_val_loss,
                        "epochs_no_improve": epochs_no_improve,
                        "early_stop": early_stop 
                    }, os.path.join(
                        self.checkpoint_dir,
                        f"{t + 1}_checkpoint.pth.tar")
                )

        writer.close()

        print(
            f"-------------------------- Training finished in \
                {(datetime.now() - start).seconds}s --------------------------")


    def accuracy_evaluation(self, eval_dataset, filename=None):
        """
        Evaluate the accuracy of the model on the provided evaluation dataset.

        Args:
            eval_dataset (DataLoader): The evaluation dataset.
            filename (str or pathlib object): The filename or path to save the evaluation 
                                              results in the output CSV.
        """

        default_out_dir = Path(self.working_dir) / self.out_dir

        if not os.path.exists(default_out_dir):
            os.makedirs(default_out_dir)
        
        if filename is None:
            output_path = str(default_out_dir / "metrics.csv")
        else:
            if os.path.dirname(filename):
                file_dir = Path(filename).parent
                if not file_dir.exists():
                    os.makedirs(file_dir)
                output_path = filename
            else:
                #output_path = str(default_out_dir / filename)
                output_path = os.path.join(default_out_dir, filename)
                
        print("---------------- Start evaluation ----------------")

        start = datetime.now()

        evaluate(self.model, eval_dataset, self.num_classes, self.class_mapping, 
                 self.device, self.buffer, output_path)

        duration_in_sec = (datetime.now() - start).seconds
        print(f"--------- Evaluation finished in {duration_in_sec}s ---------")


    def inference(self, pred_dataset, out_prefix, pred_buffer, mc_samples=10,
                  shrink_buffer=0, hardening_threshold=70, filename=""):
        """
        Performs prediction on a given dataset and writes the output tif files to
        the specified directory.
        
        This method supports both regular and Monte Carlo (MC) predictions. 
        For MC predictions, it outputs the mean prediction, variance, and 
        mutual information for each non-background class. The method ensures 
        the output tif files are saved with correct spatial references.
        
        Args:
            pred_dataset (torch.utils.data.DataLoader): A DataLoader containing the 
                                                        dataset on which to perform 
                                                        prediction.
            out_prefix (str): The output directory path where the prediction results 
                              will be saved.
            pred_buffer (int): The buffer size to be used for prediction. 
            mc_samples (int, optional): The number of Monte Carlo samples to generate. 
                                        If 0, regular prediction is performed. 
                                        Defaults to 10.
            shrink_buffer (int, optional): The size to shrink the prediction buffer. 
                                           Defaults to 0.
            hardening_threshold (int): Threshold to harden the probability score
            filename (str, optional): The filename suffix for the saved prediction files. 
                                      Defaults to an empty string.
        
        """
        
        print('---------------------- Start prediction ----------------------')
        start = datetime.now()

        _, meta, tile, year = pred_dataset
        #name_score = f"{year}_c{tile[0]}_r{tile[1]}{filename}.tif"
        name_score = "{}_{}_{}.tif".format(year, tile,filename)     #f"{year}_c{tile[0]}_r{tile[1]}{filename}.tif"
        #print(name_score)
        #exit()

        meta.update({
            'dtype': 'int8'
        })
        meta_uncertanity = meta.copy()
        meta_uncertanity.update({
            "dtype": "float64"
        })

        prefix_score = os.path.join(out_prefix, "prob_score")
        os.makedirs(prefix_score, exist_ok=True)
        
        new_buffer = pred_buffer - shrink_buffer

        prefix_hardened = os.path.join(out_prefix, "Hardened")
        os.makedirs(prefix_hardened, exist_ok=True)
        
        if mc_samples > 0:
            scores = mc_predict(pred_dataset, self.model, mc_samples, 
                                pred_buffer, gpu=self.gpu, num_classes=self.num_classes,
                                shrink_pixel=shrink_buffer)
            # write score of each non-background classes into s3
            nclass = len(scores)
            prefix_var = os.path.join(out_prefix, "Variance")
            os.makedirs(prefix_var, exist_ok=True)
            prefix_entropy = os.path.join(out_prefix, "Entropy_MI")
            os.makedirs(prefix_entropy, exist_ok=True)
            # subtracting one as we want to ingnore generating results for 
            # boundary class to increase the speed and save space.
            for n in range(nclass - 1):
                canvas = scores[n][
                    :, new_buffer: meta['height'] + new_buffer,
                    new_buffer: meta['width'] + new_buffer
                ]

                mean_pred = np.mean(canvas, axis=0)
                mean_pred = np.rint(mean_pred)

                mean_pred = mean_pred.astype(meta['dtype'])

                with rasterio.open(os.path.join(prefix_score, f"{n + 1}_{name_score}"), 
                                    "w", **meta) as dst:
                    dst.write(mean_pred, indexes=1)

                hardened_score = np.where(mean_pred > hardening_threshold, mean_pred, 0)
                with rasterio.open(os.path.join(prefix_hardened, f"hardened_{n + 1}_{name_score}"), 
                                   "w", **meta) as dst:
                    dst.write(hardened_score, indexes=1)
                
                var_pred = np.var(canvas, axis=0)
                var_pred = var_pred.astype(meta_uncertanity['dtype'])
                with rasterio.open(os.path.join(prefix_var, f"{n + 1}_{name_score}"),
                                    "w", **meta_uncertanity) as dst:
                    dst.write(var_pred, indexes=1)

                epsilon = sys.float_info.min
                entropy = -(mean_pred * np.log(mean_pred + epsilon))
                mutual_info = entropy - np.mean(
                    -canvas * np.log(canvas + epsilon), axis=0
                )
                mutual_info = mutual_info.astype(meta_uncertanity['dtype'])
                with rasterio.open(os.path.join(prefix_entropy, f"{n + 1}_{name_score}"), 
                                    "w", **meta_uncertanity) as dst:
                    dst.write(mutual_info, indexes=1)

        else:
            scores = predict(pred_dataset, self.model, pred_buffer, 
                                gpu=self.gpu, num_classes=self.num_classes,
                                shrink_pixel=shrink_buffer)
            # write score of each non-background classes
            nclass = len(scores)
            for n in range(nclass):
                if n ==1: 
                    continue
                canvas = scores[n][new_buffer: meta['height'] + new_buffer, 
                                    new_buffer: meta['width'] + new_buffer]
                canvas = canvas.astype(meta['dtype'])

                with rasterio.open(os.path.join(prefix_score, f"{n + 1}_{name_score}"), 
                                    "w", **meta) as dst:
                    dst.write(canvas, indexes=1)
                
                hardened_score = np.where(canvas > hardening_threshold, canvas, 0)
  
                with rasterio.open(os.path.join(prefix_hardened, f"hardened_{n + 1}_{name_score}"), 
                                   "w", **meta) as dst:
                    dst.write(hardened_score, indexes=1)

        # print('----------------- Prediction finished in {}s -----------------' \
        #       .format((datetime.now() - start).seconds))


    def save(self, save_object="params", filename=""):
        """
        Saves the state of the model or its parameters to disk.
        
        This method allows for the saving of either the entire model or just its parameters. 
        The method also allows for specifying a custom filename for the saved file. If no 
        filename is provided, the model's name is used as the default filename.
        
        Args:
            save_object (str, optional): Determines what to save. If set to "params", only the 
                                         model's parameters are saved. If set to "model", the 
                                         entire model is saved. Defaults to "params".
            filename (str, optional): The name of the file to save the model or parameters to. 
                                      If not provided, the model's name is used as the filename. 
                                      Defaults to an empty string.
        Note:
            The method prints a confirmation message upon successful saving of the model's 
            parameters or the model itself. The saved file will be located in the model's 
            checkpoint directory.
        """

        if not filename:
            filename = self.model_name
        
        if save_object == "params":
            if len(self.gpu_devices) > 1:
                torch.save(self.model.module.state_dict(),
                           os.path.join(self.checkpoint_dir, "{}_final_state.pth".format(filename)))
            else:
                torch.save(self.model.state_dict(),
                           os.path.join(self.checkpoint_dir, "{}_final_state.pth".format(filename)))

            print("--------------------- Model parameters is saved to disk ---------------------")

        elif save_object == "model":
            torch.save(self.model,
                       os.path.join(self.checkpoint_dir, "{}_final_state.pth".format(filename)))

        else:
            raise ValueError("Improper object type.")
    