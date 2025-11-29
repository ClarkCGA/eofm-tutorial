import torch
from torch import nn
from .basicBlocks import ConvBlock, UpconvBlock, AdditiveAttentionBlock, basicBlock, bottleNeck
from .backBones import VGG_encoder, Resnet_encoder
from .decoders import VGG_decoder, ResNet_decoder 
from IPython.core.debugger import set_trace


class unet(nn.Module):
    """
    A modular U-Net model for image segmentation tasks.
    
    The modular design allows for flexibility in choosing different encoder and decoder architectures. 
    It supports popular architectures like VGG and ResNet, and can be easily extended to others.
    
    Args:
        n_classes (int): The number of classes for the segmentation task.
        in_channels (int): The number of input channels in the data.
        encoder_config (dict): A dictionary containing configuration for the encoder, including the number 
                               of stages, stage widths, block types, etc.
        decoder_config (dict): A dictionary containing configuration for the decoder, including the number 
                               of stages, stage widths, block types, etc.
        activation_func (str, optional): The name of the activation function to be used in the network. 
                                         Defaults to "relu".
        dropout_rate (float, optional): The dropout rate to be used in the network. Defaults to 0.
    
    Returns:
        torch.Tensor: The output tensor representing the segmented image.
    """
    def __init__(self, n_classes, in_channels, encoder_config, decoder_config, dropout_rate=0, 
                 dropout_type="spatial", activation_func="relu"):
        super(unet, self).__init__()
        
        # Define a dictionary to map block type names to class references.
        # This is prefered than using eval function.
        func_mapping = {
            "basicBlock": basicBlock,
            "bottleNeck": bottleNeck,
            "relu": nn.ReLU(inplace=True)
        }
        
        if "vgg" in encoder_config["name"].lower():
            
            assert len(encoder_config["block_num"]) == len(encoder_config["stage_width"]),\
              "The length of 'block_num' and 'stage_width' lists in 'encoder_config' must be equal."
            
            self.activation_func = func_mapping[activation_func]
            
            self.encoder = VGG_encoder(in_channels=in_channels,
                                       filter_config=encoder_config["stage_width"], 
                                       block_size=encoder_config["block_num"],
                                       activation_func=self.activation_func,
                                       drop_rate=dropout_rate,
                                       dropout_type=dropout_type)
            
            self.decoder = VGG_decoder(filter_config=encoder_config["stage_width"], 
                                       block_size=decoder_config["block_num"],
                                       activation_func=self.activation_func,
                                       drop_rate=dropout_rate, 
                                       dropout_type=dropout_type, 
                                       upmode=decoder_config["upmode"],
                                       use_skipAtt=decoder_config["use_skipAtt"])
        
        elif "resnet" in encoder_config["name"].lower():
            encoder_block_type = func_mapping[encoder_config["block_type"]]
            
            self.encoder = Resnet_encoder(block=encoder_block_type,
                                          in_channels=in_channels, 
                                          layers=encoder_config["block_num"], 
                                          stage_width=encoder_config["stage_width"], 
                                          block_version=encoder_config["block_version"], 
                                          drop_rate=dropout_rate,
                                          dropout_type=dropout_type, 
                                          reduction=encoder_config["reduction"])
            
            self.decoder = ResNet_decoder(filter_config=encoder_config["stage_width"], 
                                          block_size=decoder_config["block_num"],
                                          encoder_name=encoder_config["name"],
                                          drop_rate=dropout_rate,
                                          dropout_type=dropout_type, 
                                          upmode=decoder_config["upmode"],
                                          use_skipAtt=decoder_config["use_skipAtt"],
                                          block_version=encoder_config["block_version"],
                                          reduction=encoder_config["reduction"])
        else:
            raise ValueError(f"'{model_family}' is not an accepted model family.") 
             
        # Final classifier layer
        self.classifier = nn.Conv2d(encoder_config["stage_width"][0], n_classes, kernel_size=1, 
                                    stride=1, padding=0)  

    def forward(self, inputs):

        features = self.encoder(inputs)
        output = self.decoder(features)
        output = self.classifier(output)

        return output
    