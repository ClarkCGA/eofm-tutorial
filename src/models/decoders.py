import torch
from torch import nn
from .basicBlocks import ConvBlock, UpconvBlock, AdditiveAttentionBlock, basicBlock, bottleNeck, ResidualDoubleConv, Conv_bn_activation
from IPython.core.debugger import set_trace


class VGG_decoder(nn.Module):
    """
    A VGG-style decoder module for CNNs.
    
    This decoder is designed to work in tandem with it's corresponding encoder (VGG_encoder). 
    It sequentially processes the features from the encoder using a series of up-convolutional 
    blocks and convolutional blocks, optionally integrating skip connections with attention mechanisms. 
    The decoder will reconstruct the spatial dimensions of the input and map each pixel to the 
    defined label space while allowing the network to learn to focus on salient features using 
    attention, if enabled.
    
    Args:
        filter_config (list of int): A list specifying the number of channels for each stage in 
                                     the decoder.
        block_size (list of int): A list indicating the number of convolutional layers in each 
                                  stage of the decoder. The length should match that of 'filter_config'.
        activation_func (callable): The activation function to be used after each convolutional layer.
        drop_rate (float, optional): The dropout rate to apply at the end of each convolutional block. 
                                     Defaults to 0 (no dropout).
        upmode (str, optional): The mode of up-sampling to use. Options typically include 'deconv_2', etc.
                                Defaults to 'deconv_2'.
        use_skipAtt (bool, optional): Flag to indicate whether to use additive attention blocks for skip 
                                      connections. Defaults to False.
    Returns:
        torch.Tensor: The output tensor reconstructed from the encoded features and is the mapping of 
                      the original input image to the label space.
                                     
    """
    def __init__(self, filter_config, block_size, activation_func, drop_rate=0, dropout_type='spatial', 
                 upmode="deconv_2", use_skipAtt=False):
        super(VGG_decoder, self).__init__()
        
        self.use_skipAtt = use_skipAtt
        self.num_stages = len(filter_config)

        for i in range(1, self.num_stages):
            
            setattr(self, f"decoderUpsample_{i}", UpconvBlock(filter_config[self.num_stages-i], 
                                                              filter_config[(self.num_stages-1)-i],
                                                              upmode=upmode))
            setattr(self, f"decoderConv_{i}", ConvBlock(filter_config[(self.num_stages-1)-i] * 2, 
                                                        filter_config[(self.num_stages-1)-i],
                                                        activation_func=activation_func,
                                                        num_conv_layers=block_size[(self.num_stages-1)-i], 
                                                        drop_rate=drop_rate, dropout_type=dropout_type))
            # Attention blocks (if used)
            if self.use_skipAtt:
                if i == self.num_stages - 1:
                    F_inter = int(filter_config[(self.num_stages-1)-i] / 2)
                else:
                    F_inter = filter_config[(self.num_stages-1)-i]
                                  
                setattr(self, f"Att_{i}", AdditiveAttentionBlock(F_g=filter_config[(self.num_stages-1)-i], 
                                                                 F_x=filter_config[(self.num_stages-1)-i], 
                                                                F_inter=filter_config[(self.num_stages-2)-i]))
    
    def forward(self, features):
        e1, e2, e3, e4, e5, e6 = features
        d = e6
        
        for i in range(1, self.num_stages):
            decoder_upsample = getattr(self, f"decoderUpsample_{i}")
            d = decoder_upsample(d)
            #print(f"Shape of d after decoder_upsample:", d.shape)
            encoder_feature = locals()[f'e{(self.num_stages)-i}']
            #print(f"encoder_feature:e{(self.num_stages-i)}")
            
            if self.use_skipAtt:
                att = getattr(self, f"Att_{i}")
                skip_connection = torch.cat((att(g=d, x=encoder_feature), d), dim=1)
            else:
                skip_connection = torch.cat((encoder_feature, d), dim=1)
            #print(f"Shape of skip_connection:", skip_connection.shape)
            decoder_conv = getattr(self, f"decoderConv_{i}")
            d = decoder_conv(skip_connection)
            #print(f"Shape of d after decoder_conv:", d.shape)

        return d


class ResNet_decoder(nn.Module):
    """
    A Rsidual-style decoder module for CNNs.
    
    This decoder complements a ResNet-style encoder and is designed to reconstruct the spatial dimensions 
    of the input from the encoded features. It employs a series of up-sampling and residual convolutional 
    blocks, optionally integrating skip connections with attention mechanisms for enhanced feature integration.
    The decoder can adapt its architecture slightly depending on the type of ResNet encoder used (e.g., ResNet18,34, ...).
    
    Args:
        filter_config (list of int): A list specifying the number of channels for each stage in the decoder.
        block_size (list of int): A list indicating the number of convolutional layers in each stage of the decoder.
                                  The length should match that of `filter_config`.
        activation_func (callable): The activation function to be used after each convolutional layer.
        encoder_name (str): The name of the ResNet encoder model being complemented by this decoder (e.g., 'resnet18').
        block_version (str, optional): The version of the residual block to use, e.g., 'v1' or 'v2'. Defaults to 'v1'.
        drop_rate (float, optional): The dropout rate to apply at the end of each convolutional block. Defaults to 0.
        upmode (str, optional): The mode of up-sampling to use. Defaults to 'deconv_2'.
        use_skipAtt (bool, optional): Flag to indicate whether to use additive attention blocks for skip connections. 
                                      Defaults to False.
        reduction (float, optional): The reduction factor to apply in the bottleneck of the residual blocks which 
                                     applies squeeze and excitation atention to the block. Defaults to None.
    
    Returns:
        torch.Tensor: The output tensor reconstructed from the encoded features and is the mapping of the original 
                      input image to the label space.
    
    Note:
        To maintain a manageable number of parameters, especially for larger ResNet encoder variants with an expansion 
        factor greater than 1, a 1x1 convolution is applied on the skip conection. This step effectively reduces the 
        dimensionality of features from each stage of the ResNet architecture, except the initial stem subnetwork. 
        This approach balances the model complexity while preserving essential feature information.
    """
    def __init__(self, filter_config, block_size, encoder_name, block_version="v1", drop_rate=0, 
                 dropout_type="spatial", upmode="deconv_2", use_skipAtt=False, reduction=None):
        super(ResNet_decoder, self).__init__()
        
        self.use_skipAtt = use_skipAtt
        self.encoder_name = encoder_name
        
        self.decoder_upsample_1 = UpconvBlock(filter_config[5] * 4, filter_config[4], upmode=upmode)
        self.decoder_resConv_1 = ResidualDoubleConv(filter_config[4] * 2, filter_config[4],
                                                    drop_rate=drop_rate,
                                                    dropout_type=dropout_type,
                                                    block_version=block_version, 
                                                    reduction=reduction)
        
        self.decoder_upsample_2 = UpconvBlock(filter_config[4], filter_config[3], upmode=upmode)
        self.decoder_resConv_2 = ResidualDoubleConv(filter_config[3] * 2, filter_config[3],
                                                    drop_rate=drop_rate,
                                                    dropout_type=dropout_type,
                                                    block_version=block_version, 
                                                    reduction=reduction)
        
        self.decoder_upsample_3 = UpconvBlock(filter_config[3], filter_config[2], upmode=upmode)
        self.decoder_resConv_3 = ResidualDoubleConv(filter_config[2] * 2, filter_config[2], 
                                                    drop_rate=drop_rate,
                                                    dropout_type=dropout_type,
                                                    block_version=block_version, 
                                                    reduction=reduction)
        
        self.decoder_upsample_4 = UpconvBlock(filter_config[2], filter_config[1], upmode=upmode)
        self.decoder_resConv_4 = ResidualDoubleConv(filter_config[1] * 2, filter_config[1], 
                                                    drop_rate=drop_rate,
                                                    dropout_type=dropout_type,
                                                    block_version=block_version, 
                                                    reduction=reduction)
        
        self.decoder_upsample_5 = UpconvBlock(filter_config[1], filter_config[0], upmode=upmode)
        self.decoder_resConv_5 = ResidualDoubleConv(filter_config[0] * 2, filter_config[0], 
                                                    drop_rate=drop_rate,
                                                    dropout_type=dropout_type,
                                                    block_version=block_version, 
                                                    reduction=reduction)

        if self.encoder_name not in ["resnet18", "resnet34"]:
            self.decoder_conv_1 = Conv_bn_activation(filter_config[4] * 4, filter_config[4], 
                                                     kernel_size=1, apply_activation=False)
            self.decoder_conv_2 = Conv_bn_activation(filter_config[3] * 4, filter_config[3],
                                                     kernel_size=1, apply_activation=False)
            self.decoder_conv_3 = Conv_bn_activation(filter_config[2] * 4, filter_config[2], 
                                                     kernel_size=1, apply_activation=False)

        
        if self.use_skipAtt:
            self.Att1 = AdditiveAttentionBlock(
                F_g=filter_config[4], F_x=filter_config[4], 
                F_inter=filter_config[3]
            )
            self.Att2 = AdditiveAttentionBlock(
                F_g=filter_config[3], F_x=filter_config[3], 
                F_inter=filter_config[2]
            )
            self.Att3 = AdditiveAttentionBlock(
                F_g=filter_config[2], F_x=filter_config[2], 
                F_inter=filter_config[1]
            )
            self.Att4 = AdditiveAttentionBlock(
                F_g=filter_config[1], F_x=filter_config[1], 
                F_inter=filter_config[0]
            )
            self.Att5 = AdditiveAttentionBlock(
                F_g=filter_config[0], F_x=filter_config[0],
                F_inter=int(filter_config[0] / 2)
            )


    def forward(self, features):
        #set_trace()
        e1, e2, e3, e4, e5, e6 = features
        
        d6 = self.decoder_upsample_1(e6)
        
        if self.encoder_name not in ["resnet18", "resnet34"]:
            e5 = self.decoder_conv_1(e5)
        
        if self.use_skipAtt:
            skip1 = torch.cat((self.Att_6(g=d6, x=e5), d6), dim=1)
        else:
            skip1 = torch.cat((e5, d6), dim=1)
        
        d6_proper = self.decoder_resConv_1(skip1)
        
        d5 = self.decoder_upsample_2(d6_proper)
        
        if self.encoder_name not in ["resnet18", "resnet34"]:
            e4 = self.decoder_conv_2(e4)
        
        if self.use_skipAtt:
            skip2 = torch.cat((self.Att_6(g=d5, x=e4), d5), dim=1)
        else:
            skip2 = torch.cat((e4, d5), dim=1)
        
        d5_proper = self.decoder_resConv_2(skip2)
        
        d4 = self.decoder_upsample_3(d5_proper)
        
        if self.encoder_name not in ["resnet18", "resnet34"]:
            e3 = self.decoder_conv_3(e3)
        
        if self.use_skipAtt:
            skip3 = torch.cat((self.Att_6(g=d4, x=e3), d4), dim=1)
        else:
            skip3 = torch.cat((e3, d4), dim=1)
        
        d4_proper = self.decoder_resConv_3(skip3)
        
        d3 = self.decoder_upsample_4(d4_proper)
        
        if self.use_skipAtt:
            skip4 = torch.cat((self.Att_6(g=d3, x=e2), d3), dim=1)
        else:
            skip4 = torch.cat((e2, d3), dim=1)
        
        d3_proper = self.decoder_resConv_4(skip4)
        
        d2 = self.decoder_upsample_5(d3_proper)
        if self.use_skipAtt:
            skip5 = torch.cat((self.Att_6(g=d2, x=e1), d2), dim=1)
        else:
            skip5 = torch.cat((e1, d2), dim=1)
        
        d2_proper = self.decoder_resConv_5(skip5)

        return d2_proper


