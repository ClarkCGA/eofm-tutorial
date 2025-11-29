from torch.autograd import Variable
from torch.nn.modules.batchnorm import _BatchNorm
from .optimizer import SAM
from .utils import *
from IPython.core.debugger import set_trace


def train_one_epoch(train_data, model, criterion, optimizer, scheduler, num_classes, 
                    device, train_loss=[]):
    """
    Train model for a single epoch.

    Args:
        train_data (DataLoader): Batch grouped data
        model (torch.nn): Model to train
        criterion (str): Function to caculate loss
        oprimizer (str): Function for optimzation
        scheduler (str): Update policy for learning rate decay.
        num_classes (int): number of output classes based on the classification scheme
        device (str): Either 'cuda' or 'cpu'.
        train_loss (empty list): To record average loss for each epoch
    """

    def disable_running_stats(model):
        """
        Disables the running statistics for Batch Normalization layers in the model.
        This function can be useful during certain training steps, 
        for example, in the first step of Sharpness-Aware Minimization (SAM) optimizer.
        """
        def _disable(module):
            if isinstance(module, _BatchNorm):
                module.backup_momentum = module.momentum
                module.momentum = 0

        model.apply(_disable)

    def enable_running_stats(model):
        """
        Enables the running statistics for Batch Normalization layers in the model.
        This function is typically used in pair with the 'disable_running_stats' function
        to re-enable batch normalization updates after they have been temporarily disabled,
        for example, in the second step of Sharpness-Aware Minimization (SAM) optimizer.
        """
        def _enable(module):
            if (isinstance(module, _BatchNorm)
                and hasattr(module, "backup_momentum")):
                module.momentum = module.backup_momentum

        model.apply(_enable)
    
    
    model.train()

    # mini batch iteration
    epoch_loss = 0
    num_train_batches = len(train_data)
    
    # get optimizer name
    optimizer_name = optimizer.__class__.__name__
    print(f"Using {optimizer_name}")

    #set_trace()
    for img, label, mask in train_data:

        # forward
        img = img.to(device)
        label = label.to(device)
    
        label = label * mask.to(device)
        mask_multi_channel = torch.stack([mask] * num_classes, dim=1).to(device)

        # Conditional to enable SAM optimizer to run
        if optimizer_name == "SAM":
            enable_running_stats(model)  # Enable running stats

            def closure():
                predictions = model(img) * mask_multi_channel
                loss = criterion(predictions, label)
                loss.mean().backward()
                return loss

            model_out = model(img) * mask_multi_channel
            # if np.isnan(model_out.detach().cpu().numpy()).any():
            #     print("NaN values are detected in model output")
            loss = criterion(model_out, label)
            loss.mean().backward()
            optimizer.first_step(zero_grad=True)

            if any(p.grad is not None for p in model.parameters()):
                optimizer.step(closure)

            disable_running_stats(model)  # Disable running stats

            # second forward-backward step
            def closure2():
                predictions = model(img) * mask_multi_channel
                loss = criterion(predictions, label)
                loss.mean().backward()
                return loss

            loss2 = criterion(model(img) * mask_multi_channel, label)
            loss2.mean().backward()
            optimizer.second_step(zero_grad=True)

            if any(p.grad is not None for p in model.parameters()):
                optimizer.step(closure2)

            epoch_loss += loss.item()

        else:
            out = model(img) * mask_multi_channel
            loss = criterion(out, label)
            epoch_loss += loss.item()

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # avoid calling to config.yaml
        isCyclicLR = False
        if type(scheduler) == torch.optim.lr_scheduler.CyclicLR:
            scheduler.step()
            isCyclicLR = True

    print(f'train loss:{epoch_loss / num_train_batches}')
    if isCyclicLR:
        print(f"LR: {scheduler.get_last_lr()}")

    if train_loss != None:
        train_loss.append(float(epoch_loss / num_train_batches))
