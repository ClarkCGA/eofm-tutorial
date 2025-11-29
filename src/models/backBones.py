import torch
from torch import nn
from .basicBlocks import *
from IPython.core.debugger import set_trace

class VGG_encoder(nn.Module):
    """
    A VGG-style encoder module for CNNs.
    This encoder is a sequence of convolutional blocks, each potentially consisting of 
    multiple convolutional layers, followed by batch normalization, activation function, 
    and optional dropout. The encoder supports variable number of stages with configurable 
    number of convolutional layers in each stage. Max pooling is applied between stages to 
    reduce the spatial dimensions.
    
    Args:
        in_channels (int): The number of input channels for the first convolutional block.
        filter_config (list of int): A list specifying the number of output channels for each stage.
        block_size (list of int): A list indicating the number of convolutional layers in each stage. 
                                  The length of this list should match that of 'filter_config'.
        activation_func (callable): The activation function to be used after each convolutional layer.
        drop_rate (float, optional): The dropout rate to apply at the end of each convolutional block. 
                                     Defaults to 0 (no dropout).
        dropout_type (str): decides on the type of dropout to be used.
    
    Returns:
        list of torch.Tensor: A list of features produced at each stage of the encoder.
    """
    def __init__(self, in_channels, filter_config, block_size, activation_func, 
                 drop_rate=0, dropout_type='spatial'):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_stages = len(filter_config)
        
        if len(filter_config) != len(block_size):
            raise ValueError(f"The length of filter_config: {filter_config} and block_size: {block_size} must be the same")
        
        for i in range(self.num_stages):
            in_ch = self.in_channels if i == 0 else filter_config[i - 1]
            out_ch = filter_config[i]
            first_block = True if i == 0 else False
            setattr(self, f"encoder_{i+1}", ConvBlock(in_ch, out_ch, activation_func,
                                                      num_conv_layers=block_size[i], 
                                                      drop_rate=drop_rate, 
                                                      dropout_type=dropout_type, 
                                                      first_block=first_block))
            
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            
    def forward(self, inputs):
        features = []
        e = inputs
        
        for i in range(self.num_stages):
            
            en_conv = getattr(self, f"encoder_{i+1}")
            e = en_conv(e)
            
            if i < self.num_stages - 1:
                features.append(e)
                e = self.pool(e)
            else:
                features.append(e)
                
        return features


class Resnet_encoder(nn.Module):
    """
    A ResNet model constructor that can be used to create various ResNet architectures.

    Args:
        block (nn.Module): The block type to use (basicBlock or bottleNeck).
        in_channels (int): Number of input channels.
        layers (list): A list containing the number of layers in each stage.
        stage_width (list): A list containing the channel dimensions for each stage.
        firstKernel (int): Kernel size for the first convolutional layer.
        firstPadding (int): Padding for the first convolutional layer.
        block_version (str): Version of the block ('v1' or 'v2').
        activation_func (str): Activation function to be used in the blocks.
        drop_rate (float): Dropout rate to be used in the blocks.
        reduction (int or None): Reduction factor for the SELayer. None if SELayer is not used.

    Attributes:
        conv1 (nn.Sequential): The initial convolutional layer.
        pool1 (nn.MaxPool2d): The initial pooling layer.
        stage1, stage2, stage3, stage4 (nn.Sequential): Stages of the network.
    """
    def __init__(self, block, in_channels, layers, stage_width, block_version='v1', 
                 drop_rate=0, dropout_type='spatial', reduction=None):
        super(Resnet_encoder, self).__init__()

        self.block_version = block_version
        self.drop_rate = drop_rate
        self.dropout_type = dropout_type
        self.reduction = reduction
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, stage_width[0]//2, 3, padding=1),\
                                   nn.BatchNorm2d(stage_width[0]//2),\
                                   nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(stage_width[0]//2, stage_width[0], 3, padding=1),\
                                   nn.BatchNorm2d(stage_width[0]),\
                                   nn.ReLU(True))
        
        self.conv3 = nn.Sequential(nn.Conv2d(stage_width[0], stage_width[1]//2, 3, padding=1),\
                                   nn.BatchNorm2d(stage_width[1]//2),\
                                   nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(stage_width[1]//2, stage_width[1], 3, padding=1),\
                                   nn.BatchNorm2d(stage_width[1]),\
                                   nn.ReLU(True))
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        if drop_rate > 0:
            if self.dropout_type == 'spatial':
                self.drop_lyr = nn.Dropout2d(drop_rate)
            elif self.dropout_type == 'traditional':
                self.drop_lyr = nn.Dropout(drop_rate)
            else:
                raise ValueError("dropout_type must be 'spatial' or 'traditional'.")

        self.stage1 = self.makeStage(block, stage_width[2], layers[0], firstStage= True)  # 1/4
        self.stage2 = self.makeStage(block, stage_width[3], layers[1])  # 1/8
        self.stage3 = self.makeStage(block, stage_width[4], layers[2])
        self.stage4 = self.makeStage(block, stage_width[5], layers[3])

    
    def makeStage(self, block, width, blocks, firstStage=False):
        """
        Constructs a stage in the ResNet architecture.

        Args:
            block (nn.Module): The block type to use (basicBlock or bottleNeck).
            width (int): The channel dimension for this stage.
            blocks (int): Number of blocks in this stage.
            firstStage (bool): Whether this is the first stage (affects input channel size).

        Returns:
            nn.Sequential: A sequential container of blocks forming a stage in ResNet.
        """
        layers = []

        out_ch = int(width * block.expansion)
        #print(out_ch)

        if firstStage:
            in_ch = width  # 64
        else:
            in_ch = int(width * block.expansion / 2)
            #in_ch = width * block.expansion


        for i in range(blocks):
            if i == 0:
                conv = block(in_ch, out_ch, block_version=self.block_version, drop_rate=self.drop_rate, 
                             dropout_type=self.dropout_type, reduction=self.reduction)
            else:
                # conv = block(out_ch * block.expansion, out_ch, block_version=self.block_version, activation_func=self.activation_func, 
                #              drop_rate=self.drop_rate, reduction=self.reduction)
                conv = block(out_ch, out_ch, block_version=self.block_version, drop_rate=self.drop_rate, 
                             dropout_type=self.dropout_type, reduction=self.reduction)
            layers.append(conv)

        return(nn.Sequential(*layers))

    def forward(self, x):
        #set_trace()
        e1 = self.conv2(self.conv1(x))
        p1 = self.pool(e1)
        if hasattr(self, "drop_lyr"):
            p1 = self.drop_lyr(p1)
        
        e2 = self.conv4(self.conv3(p1))
        p2 = self.pool(e2)
        if hasattr(self, "drop_lyr"):
            p2 = self.drop_lyr(p2)
        
        e3 = self.stage1(p2)
        p3 = self.pool(e3)
        
        e4 = self.stage2(p3)
        p4 = self.pool(e4)
        
        e5 = self.stage3(p4)
        p5 = self.pool(e5)
        
        e6 = self.stage4(p5)
        
        features = [e1, e2, e3, e4, e5, e6]

        return features
