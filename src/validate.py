import torch
from torch.autograd import Variable

def validate_one_epoch(val_data, model, criterion, buffer, device, val_loss=[]):
    """
        Evaluate the performance of the trained model after each epoch on
        a separate validation dataset.

        Params:
            val_data (torch.DataLoader) Batch grouped data
            model (torch.nn): Trained model for validation
            criterion (str): Function to calculate loss
            buffer (int): Buffer added to the targeted grid when creating dataset. 
                    This allows loss to calculate at non-buffered region
            device (str): Either 'cuda' or 'cpu'.
            valLoss (empty list): To record average loss for each epoch
            
    """

    model.eval()

    # mini batch iteration
    epoch_loss = 0
    num_val_batches = len(val_data)

    with torch.no_grad():
        
        for img_batch, lbl_batch in val_data:

            img_batch = img_batch.to(device)
            lbl_batch = lbl_batch.to(device)

            pred = model(img_batch)

            if buffer > 0:
                loss = criterion(pred[:, :, buffer:-buffer, buffer:-buffer],
                                lbl_batch[:, buffer:-buffer, buffer:-buffer])
            else:
                loss = criterion(pred,
                                lbl_batch)
                 
            epoch_loss += loss.item()

    print('validation loss: {}'.format(epoch_loss / num_val_batches))

    if val_loss != None:
        val_loss.append(float(epoch_loss / num_val_batches))
