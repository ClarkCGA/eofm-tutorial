import torch
from torch import nn
import torch.nn.functional as F
from IPython.core.debugger import set_trace


class Conv_bn_activation(nn.Module):
    """
    A convolutional block that combines a convolutional layer, batch normalization, 
    and an optional activation function.

    This block can operate in two versions: 'v1' applies batch normalization after 
    convolution, while 'v2' applies it before.

    Args:
        in_ch (int): Number of channels in the input image.
        out_ch (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the 
                                          input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to 
                                output channels. Default: 1
        apply_activation (bool, optional): Flag to apply an activation function. 
                                           Default: True
        version (str, optional): Version of the block - 'v1' for conv-bn-activation, 
                                 'v2' for bn-activation-conv. Default: 'v1'

        Returns:
            Tensor: The output tensor.
    """
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1, 
                 groups=1, apply_activation=True, version='v1'):
        super(Conv_bn_activation, self).__init__()
        
        self.version = version
        self.apply_activation = apply_activation
        
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, 
                              dilation=dilation, groups=groups)

        if self.version == 'v1':
            self.bn = nn.BatchNorm2d(out_ch)
        elif self.version == 'v2':
            self.bn = nn.BatchNorm2d(in_ch)
        else:
            raise ValueError("Invalid version specified. Choose 'v1' or 'v2'.")

        if self.apply_activation:
            self.relu = nn.ReLU()

    def forward(self, x):
        if self.version == 'v1':
            x = self.conv(x)
            x = self.bn(x)
            if self.apply_activation:
                x = self.relu(x)
        elif self.version == 'v2':
            x = self.bn(x)
            if self.apply_activation:
                x = self.relu(x)
            x = self.conv(x)
            
        return x


# class ConvBlock(nn.Module):
#     """
#     This module creates a user-defined number of conv+BN+ReLU layers.
    
#     Args:
#         in_channels (int): Number of input channels or features.
#         out_channels (int): Number of output channels or features.
#         kernel_size (int or tuple): Size of the convolutional kernel. 
#                                     Default: 3.
#         stride (int or tuple): Stride of the convolution. Decides how the 
#                                kernel moves along spatial dimensions. 
#                                Default: 1.
#         padding (int or tuple): Zero-padding added to both sides of the input. 
#                                 Default: 1.
#         dilation (int or tuple): Dilation rate for enlarging the receptive field 
#                                  of the kernel. Default: 1.
#         num_conv_layers (int): Number of convolutional layers, each followed by 
#                                batch normalization and activation, to be included 
#                                in the block. Default: 2.
#         drop_rate (float): Dropout rate to be applied at the end of the block. 
#                            If greater than 0, a dropout layer is added for 
#                            regularization. Default: 0.
#         activation_func (str): The activation function to be used. It is dynamically 
#                                evaluated from the string, allowing flexibility in the 
#                                choice of function. Default: "nn.ReLU(inplace=True)".
#         first_block (bool): If set to True, the output of the first convolutional layer 
#                             is saved. This allows accessing the activation maps of the 
#                             first conv layer later. Default: False.
#     """
    
#     def __init__(self, in_channels, out_channels, activation_func, kernel_size=3, stride=1, 
#                  padding=1, dilation=1, num_conv_layers=2, drop_rate=0., first_block=False):
#         super(ConvBlock, self).__init__()
#         self.first_block = first_block
        

#         layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
#                             stride=stride, padding=padding, dilation=dilation, 
#                             bias=False),
#                   nn.BatchNorm2d(out_channels),
#                   activation_func, ]

#         if num_conv_layers > 1:
#             if drop_rate > 0:
#                 layers += [
#                     nn.Conv2d(
#                         out_channels, out_channels, kernel_size=kernel_size, 
#                         stride=stride, padding=padding, dilation=dilation, 
#                         bias=False
#                     ), nn.BatchNorm2d(out_channels), activation_func, 
#                     nn.Dropout2d(drop_rate), 
#                 ] * (num_conv_layers - 1)
#             else:
#                 layers += [
#                     nn.Conv2d(
#                         out_channels, out_channels, kernel_size=kernel_size, 
#                         stride=stride, padding=padding, dilation=dilation, 
#                         bias=False
#                     ), nn.BatchNorm2d(out_channels), activation_func, 
#                 ] * (num_conv_layers - 1)

#         self.block = nn.Sequential(*layers)

#     def forward(self, inputs):
#         if self.first_block:
#             first_conv = self.block[0](inputs)
#             first_bn = self.block[1](first_conv)
#             first_activation = self.block[2](first_bn)
#             # Store the output after the first convolutional layer
#             self.first_conv_output = first_activation
#             outputs = self.block[3:](first_activation)
#         else:
#             outputs = self.block(inputs)
        
#         return outputs

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation_func, kernel_size=3, stride=1, 
                 padding=1, dilation=1, num_conv_layers=2, drop_rate=0., dropout_type='spatial',
                 first_block=False):
        """
        This module creates a user-defined number of conv+BN+ReLU layers.

        Args:
            in_channels (int): Number of input channels or features.
            out_channels (int): Number of output channels or features.
            kernel_size (int or tuple): Size of the convolutional kernel. 
                                        Default: 3.
            stride (int or tuple): Stride of the convolution. Decides how the 
                                   kernel moves along spatial dimensions. 
                                   Default: 1.
            padding (int or tuple): Zero-padding added to both sides of the input. 
                                    Default: 1.
            dilation (int or tuple): Dilation rate for enlarging the receptive field 
                                     of the kernel. Default: 1.
            num_conv_layers (int): Number of convolutional layers, each followed by 
                                   batch normalization and activation, to be included 
                                   in the block. Default: 2.
            drop_rate (float): Dropout rate to be applied at the end of the block. 
                               If greater than 0, a dropout layer is added for 
                               regularization. Default: 0.
            dropout_type (str): decides on the type of dropout to be used.
            activation_func (str): The activation function to be used. It is dynamically 
                                   evaluated from the string, allowing flexibility in the 
                                   choice of function. Default: "nn.ReLU(inplace=True)".
            first_block (bool): If set to True, the output of the first convolutional layer 
                                is saved. This allows accessing the activation maps of the 
                                first conv layer later. Default: False.
        """
        super(ConvBlock, self).__init__()
        self.first_block = first_block
        
        # Choose the dropout layer based on dropout_type
        if dropout_type == 'spatial':
            dropout_layer = nn.Dropout2d(drop_rate)
        elif dropout_type == 'traditional':
            dropout_layer = nn.Dropout(drop_rate)
        else:
            raise ValueError("dropout_type must be 'spatial' or 'traditional'.")

        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                            stride=stride, padding=padding, dilation=dilation, 
                            bias=False),
                  nn.BatchNorm2d(out_channels),
                  activation_func]

        if num_conv_layers > 1:
            for _ in range(1, num_conv_layers):
                layers += [
                    nn.Conv2d(
                        out_channels, out_channels, kernel_size=kernel_size, 
                        stride=stride, padding=padding, dilation=dilation, 
                        bias=False
                    ), nn.BatchNorm2d(out_channels), activation_func
                ]
            
            if drop_rate > 0:
                layers.append(dropout_layer)

        self.block = nn.Sequential(*layers)

    def forward(self, inputs):
        if self.first_block:
            first_conv = self.block[0](inputs)
            first_bn = self.block[1](first_conv)
            first_activation = self.block[2](first_bn)
            # Store the output after the first convolutional layer
            self.first_conv_output = first_activation
            outputs = self.block[3:](first_activation)
        else:
            outputs = self.block(inputs)
        
        return outputs


class basicBlock(nn.Module):
    
    expansion = 1

    def __init__(self, in_ch, out_ch, dilation=1, block_version='v1', 
                 drop_rate=0., dropout_type='spatial', reduction=None, *kwargs):
        """
        A basic block for a ResNet architectures V1 or V2 with optional 
        Squeeze-and-Excitation (SE) layer and dropout.

        This block implements a sequence of convolutions, batch normalization, 
        and an activation function where the ordering id decided by the choice 
        of 'version'. 
        It optionally includes an SE layer for channel recalibration and a 
        dropout layer for regularization.

        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            dilation (int, optional): Dilation rate for convolutions. Default: 1.
            block_version (str, optional): Version of the block ('v1' or 'v2'). 
                                           Default: 'v1'. 
            drop_rate (float, optional): Dropout rate. If greater than 0, a dropout layer 
                                         is added. Default: 0.
            dropout_type (str): decides on the type of dropout to be used.
            reduction (int, optional): Reduction ratio for the SE layer. If provided, an SE layer 
                                       is added. Default: None.
        """
        super(basicBlock, self).__init__()
        
        # Indicates if this block is the first convolutional block
        self.firstBlock = (in_ch != out_ch)

        if self.firstBlock:
            # for dimension matching in case of the first block
            self.conv0 = Conv_bn_activation(in_ch, out_ch, kernel_size=1, apply_activation=False, 
                                            activation_func=activation_func, version=block_version)

        # 1st 3x3 Conv
        self.conv1 = Conv_bn_activation(in_ch, out_ch, activation_func=activation_func, kernel_size=3, 
                                        padding=dilation, dilation=dilation, version=block_version)
        # 2nd 3x3 Conv
        self.conv2 = Conv_bn_activation(out_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation,
                                        apply_activation=False, activation_func=activation_func, 
                                        version=block_version)
        
        self.relu = nn.ReLU()
        
        if drop_rate > 0:
            if dropout_type == 'spatial':
                self.drop_lyr = nn.Dropout2d(drop_rate)
            elif dropout_type == 'traditional':
                self.drop_lyr = nn.Dropout(drop_rate)
            else:
                raise ValueError("dropout_type must be 'spatial' or 'traditional'.")
        
        if reduction is not None:
            if not isinstance(reduction, int) or reduction < 1:
                raise ValueError("Reduction value for SELayer must be an integer greater than or equal to 1.")
            self.se = SELayer(out_ch, reduction)

    def forward(self, x):
        res = self.conv1(x)
        res = self.conv2(res)
        
        if hasattr(self, "se"):
            res = self.se(res)
        
        if self.firstBlock:
            x = self.conv0(x)

        if hasattr(self, "drop_lyr"):
            out = self.drop_lyr(res + x)
        
        out = self.relu(out)
        
        return out


class bottleNeck(nn.Module):
    
    expansion = 4

    def __init__(self, in_ch, out_ch, dilation=1, groups=1, base_width=64, 
                 block_version='v1', drop_rate=0., dropout_type='spatial', reduction=None):
        """
        A bottleneck block for a ResNet architectures V1 or V2 with optional 
        Squeeze-and-Excitation (SE) layer and dropout.

        This block is a deeper version of the basic block, using three layers 
        of convolutions (1x1, 3x3, 1x1) with the 3x3 layer at the bottleneck. 
        It optionally includes an SE layer for channel recalibration and a 
        dropout layer for regularization.

        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels. 
            dilation (int, optional): Dilation rate for the 3x3 convolution. 
                                      Default: 1.
            groups (int, optional): Number of blocked connections from input channels 
                                    to output channels. Default: 1.
            base_width (int, optional): Base width for the bottleneck structure. 
                                        Default: 64.
            block_version (str, optional): Version of the block ('v1' or 'v2'). 
                                           Default: 'v1'.
            drop_rate (float, optional): Dropout rate. If greater than 0, a dropout layer 
                                         is added. Default: 0.
            dropout_type (str): decides on the type of dropout to be used.
            reduction (int, optional): Reduction ratio for the SE layer. If provided, an 
                                       SE layer is added. Default: None.
        """
        super(bottleNeck, self).__init__()

        self.firstBlock = (in_ch != out_ch)
        
        #width = int(out_ch * (base_width / 64.0)) * groups
        width = int(out_ch / (self.expansion * groups * base_width / 64))

        # Downsample in first 1x1 convolution if needed
        if self.firstBlock:
            self.conv0 = Conv_bn_activation(in_ch, out_ch, kernel_size=1, apply_activation=False, 
                                            version=block_version)

        # 1x1 conv
        self.conv1 = Conv_bn_activation(in_ch, width, kernel_size=1, version=block_version)
        # 3x3 conv
        self.conv2 = Conv_bn_activation(width, width, kernel_size=3, stride=1, padding=dilation, 
                                        dilation=dilation, groups=groups, version=block_version)
        # 1x1 conv
        self.conv3 = Conv_bn_activation(width, out_ch, kernel_size=1, stride=1, apply_activation=False,  
                                        version=block_version)

        self.relu = nn.ReLU()
        if drop_rate > 0:
            if dropout_type == 'spatial':
                self.drop_lyr = nn.Dropout2d(drop_rate)
            elif dropout_type == 'traditional':
                self.drop_lyr = nn.Dropout(drop_rate)
            else:
                raise ValueError("dropout_type must be 'spatial' or 'traditional'.")
        
        if reduction is not None:
            if not isinstance(reduction, int) or reduction < 1:
                raise ValueError("Reduction value for SELayer must be an integer greater than or equal to 1.")
            self.se = SELayer(out_ch, reduction)

    def forward(self, x):
        res = self.conv1(x)
        res = self.conv2(res)
        res = self.conv3(res)
        
        if hasattr(self, "se"):
            res = self.se(res)

        if self.firstBlock:
            x = self.conv0(x)

        if hasattr(self, "drop_lyr"):
            out = self.drop_lyr(res + x)
        
        out = self.relu(out)

        return out


class ResidualDoubleConv(nn.Module):
    
    def __init__(self, in_ch, out_ch, block_version='v1', drop_rate=0., dropout_type='spatial',
                 reduction=None):
        """
        Residual block with two Conv_bn_activation layers, optionally followed by a 
        Squeeze-and-Excitation layer and dropout. It includes a skip connection that 
        adds the input to the output for forming a residual block.

        Args:
            in_ch (int): Number of channels in the input image.
            out_ch (int): Number of channels produced by the convolution.
            activation_func (str, optional): Activation function to use. 
            block_version (str, optional): Version of the Conv_bn_activation block 
                                           ('v1' or 'v2'). Default: 'v1'.
            drop_rate (float, optional): Dropout rate. If greater than 0, a dropout 
                                         layer is added. Default: 0.
            dropout_type (str): decides on the type of dropout to be used.
            reduction (int, optional): Reduction ratio for the SE layer. If provided, 
                                       an SE layer is added. Default: None.
        """
        super(ResidualDoubleConv, self).__init__()

        self.conv1 = Conv_bn_activation(in_ch, out_ch, kernel_size=3, padding=1, version=block_version)
        self.conv2 = Conv_bn_activation(out_ch, out_ch, kernel_size=3, padding=1, version=block_version)
        
        # Residual connection: match dimensions if necessary
        if in_ch != out_ch:
            self.residual_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
            self.residual_bn = nn.BatchNorm2d(out_ch)

        if reduction is not None:
            if not isinstance(reduction, int) or reduction < 1:
                raise ValueError("Reduction value for SELayer must be an integer greater than or equal to 1.")
            self.se = SELayer(out_ch, reduction)

        if drop_rate > 0:
            if dropout_type == 'spatial':
                self.drop_lyr = nn.Dropout2d(drop_rate)
            elif dropout_type == 'traditional':
                self.drop_lyr = nn.Dropout(drop_rate)
            else:
                raise ValueError("dropout_type must be 'spatial' or 'traditional'.")

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.conv2(x)

        if hasattr(self, "se"):
            x = self.se(x)

        # Apply the residual connection
        if hasattr(self, "residual_conv"):
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x = x + residual

        if hasattr(self, "drop_lyr"):
            x = self.drop_lyr(x)

        x = self.relu(x)

        return x


class SELayer(nn.Module):
    """
    The Squeeze-and-Excitation (SE) layer, which adaptively recalibrates channel-wise 
    feature responses.

    The SE layer computes channel-wise weights by squeezing global spatial information 
    into a channel descriptor using global average pooling and then excites the channels 
    with these weights using a simple gating mechanism.

    Args:
        in_ch (int): Number of input channels.
        reduction (int): Reduction ratio for the intermediate channel dimension in the SE block.

    Source code: 
           https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    """

    def __init__(self, in_ch, reduction):
        super(SELayer, self).__init__()

        # aggregates global spatial information
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # compute channel-wise weights
        self.fc = nn.Sequential(
            nn.Linear(in_ch, in_ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // reduction, in_ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        squeezed = self.avg_pool(x).view(b, c)
        weights = self.fc(squeezed).view(b, c, 1, 1)
        return x * weights.expand_as(x)


class SpatialSELayer(nn.Module):
    """
    Implementation of the Spatial Squeeze and Excitation (sSE) block, which performs channel-wise 
    feature recalibration by squeezing spatially and exciting channel-wise as described in:
    "Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, 
    MICCAI 2018"
    
    Args:
        num_channels (int): The number of input channels.
    
    Returns:
        torch.Tensor: The output tensor after applying spatial SE.
    
    Source code:
        https://github.com/ai-med/squeeze_and_excitation/blob/master/squeeze_and_excitation/squeeze_and_excitation.py
    """

    def __init__(self, num_channels):
        super(SpatialSELayer, self).__init__()
        
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        Args:
            input_tensor (torch.Tensor): The input tensor with shape (batch_size, num_channels, H, W).
            weights (torch.Tensor, optional): Tensor containing weights for few-shot learning with the
            shape: (num_channels, 1, H, W). If not provided, the layer's own convolution weight will be used.
        """

        if weights is not None:
            weights = torch.mean(weights, dim=0)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        
        # Should have the shape: (batch_size, 1, H, W)
        attention_map = self.sigmoid(out)

        return input_tensor * attention_map


class UpconvBlock(nn.Module):
    r"""
    An upsampling block used in the decoder or expansive path.
    
    The block can perform fixed linear upsampling, transposed convolution 
    with or without overlap, or Dense Upscaling Convolution (DUC) for more 
    refined upsampling.

    Args:
        in_channels (int): Number of input channels or features.
        out_channels (int): Number of output channels or features after upsampling.
        upmode (str): Upsampling type. Options include:
            - "fixed": Linear upsampling with a scale factor of two using bi-linear interpolation.
            - "deconv_1": Non-overlapping transposed convolution for upsampling.
            - "deconv_2": Overlapping transposed convolution for upsampling.
            - "DUC": Dense Upscaling Convolution, combining convolution, batch normalization, 
                     activation, and pixel shuffle for upsampling.
                     This approach is beneficial for more refined upscaling with increased detail.
    """

    def __init__(self, in_channels, out_channels, upmode="deconv_1"):
        super(UpconvBlock, self).__init__()

        if upmode == "fixed":
            layers = [
                nn.Upsample(scale_factor=2, mode="bilinear", 
                            align_corners=True), 
            ]
            layers += [
                nn.BatchNorm2d(in_channels), 
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, 
                          padding=0, bias=False), 
            ]

        elif upmode == "deconv_1":
            layers = [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, 
                                   stride=2, padding=0, dilation=1), 
            ]

        elif upmode == "deconv_2":
            layers = [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, 
                                   stride=2, padding=1, dilation=1), 
            ]

        # Dense Upscaling Convolution
        elif upmode == "DUC":
            up_factor = 2
            upsample_dim = (up_factor ** 2) * out_channels
            layers = [
                nn.Conv2d(in_channels, upsample_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(upsample_dim),
                nn.ReLU(inplace=True),
                nn.PixelShuffle(up_factor), 
            ]

        else:
            raise ValueError("Provided upsampling mode is not recognized.")

        self.block = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.block(inputs)


class AdditiveAttentionBlock(nn.Module):
    """
    An Additive Attention Block to enhance fusion of feature through skip connections.
    This block is beneficial in scenarios where feature maps from different levels of 
    the network (e.g., encoder and decoder paths in a U-Net architecture) need to be 
    effectively combined.
    additive attention gate (AG) to merge feature maps extracted at multiple 
    scales through skip connection.
    
    Args:
        F_g (int): Number of feature maps from the higher resolution in the encoder path.
        F_x (int): Number of feature maps in layer "x" in the decoder path.
        F_inter (int): Number of feature maps after summation, determining the number of learnable 
                       multidimensional attention coefficients.
    """

    def __init__(self, F_g, F_x, F_inter):
        super(AdditiveAttentionBlock, self).__init__()

        # for processing the encoder path's feature maps.
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_inter, kernel_size=1, stride=1, padding=0, 
                      bias=True),
            nn.BatchNorm2d(F_inter)
        )
        # for processing the decoder path's feature maps
        self.W_x = nn.Sequential(
            nn.Conv2d(F_x, F_inter, kernel_size=1, stride=1, padding=0, 
                      bias=True),
            nn.BatchNorm2d(F_inter)
        )

        # generating the attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(F_inter, 1, kernel_size=1, stride=1, padding=0, 
                      bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # set_trace()
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        merge = self.relu(g1 + x1)
        psi = self.psi(merge)

        return x * psi