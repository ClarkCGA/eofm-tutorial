import torch
from torch import nn
from diffusion.scheduler import get_timestep_embedding

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 dilation=1, num_conv_layers=2, drop_rate=0., dropout_type='spatial'):
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
        """
        super(ConvBlock, self).__init__()
        
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
                  nn.ReLU(inplace=True)]

        if num_conv_layers > 1:
            for _ in range(1, num_conv_layers):
                layers += [
                    nn.Conv2d(
                        out_channels, out_channels, kernel_size=kernel_size, 
                        stride=stride, padding=padding, dilation=dilation, 
                        bias=False
                    ), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
                ]
            
            if drop_rate > 0:
                layers.append(dropout_layer)

        self.block = nn.Sequential(*layers)

    def forward(self, inputs):
        outputs = self.block(inputs)
        
        return outputs


class UpconvBlock(nn.Module):
    r"""
    Decoder layer decodes the features along the expansive path.
    Args:
        in_channels (int) -- number of input features.
        out_channels (int) -- number of output features.
        upmode (str) -- Upsampling type. If "fixed" then a linear upsampling with scale factor
                        of two will be applied using bi-linear as interpolation method.
                        If deconv_1 is chosen then a non-overlapping transposed convolution will
                        be applied to upsample the feature maps. If deconv_1 is chosen then an
                        overlapping transposed convolution will be applied to upsample the feature maps.
    """

    def __init__(self, in_channels, out_channels, upmode="deconv_1"):
        super(UpconvBlock, self).__init__()

        if upmode == "fixed":
            layers = [nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True), ]
            layers += [nn.BatchNorm2d(in_channels),
                       nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False), ]

        elif upmode == "deconv_1":
            layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, dilation=1), ]

        elif upmode == "deconv_2":
            layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, dilation=1), ]
        
        #Dense Upscaling Convolution
        elif upmode == "DUC":
            up_factor = 2
            upsample_dim = (up_factor ** 2) * out_channels
            layers = [nn.Conv2d(in_channels, upsample_dim, kernel_size=3, padding=1),  
                      nn.BatchNorm2d(upsample_dim),
                      nn.ReLU(inplace=True),
                      nn.PixelShuffle(up_factor),]

        else:
            raise ValueError("Provided upsampling mode is not recognized.")

        self.block = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.block(inputs)


class AdditiveAttentionBlock(nn.Module):
    r"""
    additive attention gate (AG) to merge feature maps extracted at multiple scales through skip connection.
    Args:
        f_g (int) -- number of feature maps collected from the higher resolution in encoder path.
        f_x (int) -- number of feature maps in layer "x" in the decoder.
        f_inter (int) -- number of feature maps after summation equal to the number of
                       learnable multidimensional attention coefficients.
    Note: Unlike the original paper we upsample
    """

    def __init__(self, F_g, F_x, F_inter):
        super(AdditiveAttentionBlock, self).__init__()

        # Decoder
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_inter, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_inter)
        )
        # Encoder
        self.W_x = nn.Sequential(
            nn.Conv2d(F_x, F_inter, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_inter)
        )

        # Fused
        self.psi = nn.Sequential(
            nn.Conv2d(F_inter, 1, kernel_size=1, stride=1, padding=0, bias=True),
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


class unet_att_d(nn.Module):
    def __init__(self, in_channels, filter_config, block_num, dropout_rate=0, 
                 dropout_type='traditional', upmode="deconv_2", use_skipAtt=False, 
                 time_embedding_dim=112):
        super(unet_att_d, self).__init__()

        self.in_channels = in_channels
        self.use_skipAtt = use_skipAtt


        assert len(filter_config) == len(block_num) == 6
        
        # Time embedding MLP to match bottleneck channels for diffusion (2048 here)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, filter_config[5]),
            nn.ReLU()
        )

        # Contraction Path
        self.encoder_1 = ConvBlock(self.in_channels, filter_config[0], num_conv_layers=block_num[0],
                                   drop_rate=dropout_rate, dropout_type=dropout_type)  # filter_config[0]x224x224
        self.encoder_2 = ConvBlock(filter_config[0], filter_config[1], num_conv_layers=block_num[1],
                                   drop_rate=dropout_rate, dropout_type=dropout_type)  # filter_config[1]x112x112
        self.encoder_3 = ConvBlock(filter_config[1], filter_config[2], num_conv_layers=block_num[2],
                                   drop_rate=dropout_rate, dropout_type=dropout_type)  # filter_config[2]x56x56
        self.encoder_4 = ConvBlock(filter_config[2], filter_config[3], num_conv_layers=block_num[3],
                                   drop_rate=dropout_rate, dropout_type=dropout_type)  # filter_config[3]x28x28
        self.encoder_5 = ConvBlock(filter_config[3], filter_config[4], num_conv_layers=block_num[4],
                                   drop_rate=dropout_rate, dropout_type=dropout_type)  # filter_config[4]x14x14
        self.encoder_6 = ConvBlock(filter_config[4], filter_config[5], num_conv_layers=block_num[5],
                                   drop_rate=dropout_rate, dropout_type=dropout_type)  # filter_config[5]x7x7
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Expansion Path
        self.decoder_1 = UpconvBlock(filter_config[5], filter_config[4], upmode=upmode)  # filter_config[4]x14x14
        self.conv1 = ConvBlock(filter_config[4] * 2, filter_config[4], num_conv_layers=block_num[4], 
                               drop_rate=dropout_rate, dropout_type=dropout_type)

        self.decoder_2 = UpconvBlock(filter_config[4], filter_config[3], upmode=upmode)  # filter_config[3]x28x28
        self.conv2 = ConvBlock(filter_config[3] * 2, filter_config[3], num_conv_layers=block_num[3], 
                               drop_rate=dropout_rate, dropout_type=dropout_type)

        self.decoder_3 = UpconvBlock(filter_config[3], filter_config[2], upmode=upmode)  # filter_config[2]x56x56
        self.conv3 = ConvBlock(filter_config[2] * 2, filter_config[2], num_conv_layers=block_num[2], 
                               drop_rate=dropout_rate, dropout_type=dropout_type)

        self.decoder_4 = UpconvBlock(filter_config[2], filter_config[1], upmode=upmode)  # filter_config[1]x112x112
        self.conv4 = ConvBlock(filter_config[1] * 2, filter_config[1], num_conv_layers=block_num[1], 
                               drop_rate=dropout_rate, dropout_type=dropout_type)

        self.decoder_5 = UpconvBlock(filter_config[1], filter_config[0], upmode=upmode)  # filter_config[0]x224x224
        self.conv5 = ConvBlock(filter_config[0] * 2, filter_config[0], num_conv_layers=block_num[0], 
                               drop_rate=dropout_rate, dropout_type=dropout_type)

        if self.use_skipAtt:
            self.Att1 = AdditiveAttentionBlock(F_g=filter_config[4], F_x=filter_config[4], F_inter=filter_config[3])
            self.Att2 = AdditiveAttentionBlock(F_g=filter_config[3], F_x=filter_config[3], F_inter=filter_config[2])
            self.Att3 = AdditiveAttentionBlock(F_g=filter_config[2], F_x=filter_config[2], F_inter=filter_config[1])
            self.Att4 = AdditiveAttentionBlock(F_g=filter_config[1], F_x=filter_config[1], F_inter=filter_config[0])
            self.Att5 = AdditiveAttentionBlock(F_g=filter_config[0], F_x=filter_config[0],
                                               F_inter=int(filter_config[0] / 2))

        #self.classifier = nn.Conv2d(filter_config[0], n_classes, kernel_size=1, stride=1, padding=0)  # classNumx224x224
        self.output_conv = nn.Conv2d(filter_config[0], in_channels, kernel_size=1)  # Predict noise matching input

    #def forward(self, inputs):
    def forward(self, inputs, t): # Adding timestep for diffusion
        # set_trace()
        e1 = self.encoder_1(inputs)  
        p1 = self.pool(e1)  
        e2 = self.encoder_2(p1)  
        p2 = self.pool(e2)  

        e3 = self.encoder_3(p2) 
        p3 = self.pool(e3)  

        e4 = self.encoder_4(p3)  
        p4 = self.pool(e4)  

        e5 = self.encoder_5(p4)  
        p5 = self.pool(e5)  

        e6 = self.encoder_6(p5)  

        # Adding timestep embedding to bottleneck
        t_emb = get_timestep_embedding(t, self.time_mlp[0].in_features).to(inputs.device)
        t_emb = self.time_mlp(t_emb).view(-1, e6.shape[1], 1, 1)
        e6 = e6 + t_emb

        d6 = self.decoder_1(e6)  
        

        if self.use_skipAtt:
            x5 = self.Att1(g=d6, x=e5)  
            skip1 = torch.cat((x5, d6), dim=1)  
        else:
            skip1 = torch.cat((e5, d6), dim=1) 

        d6_proper = self.conv1(skip1)  

        d5 = self.decoder_2(d6_proper)  

        if self.use_skipAtt:
            x4 = self.Att2(g=d5, x=e4)  
            skip2 = torch.cat((x4, d5), dim=1)  
        else:
            skip2 = torch.cat((e4, d5), dim=1)  

        d5_proper = self.conv2(skip2)  

        d4 = self.decoder_3(d5_proper)  

        if self.use_skipAtt:
            x3 = self.Att3(g=d4, x=e3)  
            skip3 = torch.cat((x3, d4), dim=1)  
        else:
            skip3 = torch.cat((e3, d4), dim=1)  

        d4_proper = self.conv3(skip3) 

        d3 = self.decoder_4(d4_proper)  

        if self.use_skipAtt:
            x2 = self.Att4(g=d3, x=e2)  
            skip4 = torch.cat((x2, d3), dim=1)  
        else:
            skip4 = torch.cat((e2, d3), dim=1)  

        d3_proper = self.conv4(skip4)  

        d2 = self.decoder_5(d3_proper)  

        if self.use_skipAtt:
            x1 = self.Att5(g=d2, x=e1)  
            skip5 = torch.cat((x1, d2), dim=1)  
        else:
            skip5 = torch.cat((e1, d2), dim=1)  

        d2_proper = self.conv5(skip5)  

        #d1 = self.classifier(d2_proper)  
        d1 = self.output_conv(d2_proper)   # Predicted noise (same shape as input)
        return d1
