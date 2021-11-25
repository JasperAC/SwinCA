import torch
import torch.nn as nn
from architecture.ResidualFeat import Res2Net
from architecture.netunit import *
from architecture.network_swinir import *

import pdb

_NORM_BONE = False


class Swin_Net(nn.Module):
    r"""
        Args:
            img_size (int | tuple(int)): Input image size. Default 256
            patch_size (int | tuple(int)): Patch size. Default: 1
            in_chans (int): Number of input image channels. Default: 3
            embed_dim (int): Patch embedding dimension. Default: 96
            depths (tuple(int)): Depth of each Swin Transformer layer.
            num_heads (tuple(int)): Number of attention heads in different layers.
            window_size (int): Window size. Default: 7
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
            qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
            qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
            drop_rate (float): Dropout rate. Default: 0
            attn_drop_rate (float): Attention dropout rate. Default: 0
            drop_path_rate (float): Stochastic depth rate. Default: 0.1
            norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
            ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
            patch_norm (bool): If True, add normalization after patch embedding. Default: True
            use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
            upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
            img_range: Image range. 1. or 255.
            upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
            resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
        """

    def __init__(self, img_size=256, patch_size=1, in_chans=28, out_chans=28,
                 embed_dim=112, depths=(2, 2, 2, 2), num_heads=(4, 4, 4, 4),
                 window_size=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False , resi_connection='1conv',
                 **kwargs):
        super(Swin_Net, self).__init__()

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        # self.conv_first = nn.Sequential(
        #     nn.Conv2d(in_chans, in_chans, 1),
        #     nn.BatchNorm2d(in_chans),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(in_chans, in_chans, 3, 1, 1),
        #     nn.BatchNorm2d(in_chans),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(in_chans, embed_dim, 1),
        #     nn.BatchNorm2d(embed_dim)
        # )
        # self.conv_res = nn.Sequential(
        #     nn.Conv2d(in_chans, embed_dim, 1),
        #     nn.BatchNorm2d(embed_dim)
        # )
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection
                         )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        #####################################################################################################
        ################################### 3, Reconstruction stage #########################################

        # self.tconv_up4 = Decoder_Triblock(embed_dim, embed_dim//2)
        # self.tconv_up3 = Decoder_Triblock(embed_dim//2, embed_dim//4)
        # self.tconv_up2 = Decoder_Triblock(256, 128)
        # self.tconv_up1 = Decoder_Triblock(128, 64)
        self.reconstruction = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim//2, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            conv_block(embed_dim//2, embed_dim//4),
            nn.Conv2d(embed_dim//4, out_chans, 1),
            nn.Sigmoid()
        )
        self.apply(self._init_weights)
        #
        #
        # self.layer1 = nn.Sequential(
        #     nn.ConvTranspose2d(inChannel, outChannel, kernel_size=2, stride=2),
        #     nn.LeakyReLU(inplace=True)
        # )
        # if flag_res:
        #     self.layer2 = Res2Net(int(outChannel * 2), int(outChannel / 2))
        # else:
        #     self.layer2 = conv_block(outChannel * 2, outChannel * 2, nKernal, flag_norm=_NORM_BONE)
        # self.layer3 = conv_block(outChannel * 2, outChannel, nKernal, flag_norm=_NORM_BONE)
        #
        #
        # self.conv_last = nn.Conv2d(embed_dim//4, out_chans, 1)
        # self.afn_last = nn.Sigmoid()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}


    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}


    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)


        for layer in self.layers:
            x = layer(x, x_size)


        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        # x1 = self.conv_first(x)
        # x2 = self.conv_res(x)
        # x = x1+x2
        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x)) + x

        x = self.reconstruction(x)

        return x


# class Encoder_Triblock(nn.Module):
#     def __init__(self,inChannel,outChannel,flag_res=True,nKernal=3,nPool=2,flag_Pool=True):
#         super(Encoder_Triblock, self).__init__()
#
#         self.layer1 = conv_block(inChannel,outChannel,nKernal,flag_norm=_NORM_BONE)
#         if flag_res:
#             self.layer2 = Res2Net(outChannel,int(outChannel/4))
#         else:
#             self.layer2 = conv_block(outChannel,outChannel,nKernal,flag_norm=_NORM_BONE)
#
#         self.pool = nn.MaxPool2d(nPool) if flag_Pool else None
#     def forward(self,x):
#         feat = self.layer1(x)
#         feat = self.layer2(feat)
#
#         feat_pool = self.pool(feat) if self.pool is not None else feat
#         return feat_pool,feat
#
# class Decoder_Triblock(nn.Module):
#     def __init__(self,inChannel,outChannel,flag_res=True,nKernal=3,nPool=2,flag_Pool=True):
#         super(Decoder_Triblock, self).__init__()
#
#         self.layer1 = nn.Sequential(
#             nn.ConvTranspose2d(inChannel, outChannel, kernel_size=2, stride=2),
#             nn.LeakyReLU(inplace=True)
#         )
#         if flag_res:
#             self.layer2 = Res2Net(int(outChannel*2),int(outChannel/2))
#         else:
#             self.layer2 = conv_block(outChannel*2,outChannel*2,nKernal,flag_norm=_NORM_BONE)
#         self.layer3 = conv_block(outChannel*2,outChannel,nKernal,flag_norm=_NORM_BONE)
#
#     def forward(self,feat):
#         feat = self.layer1(feat)
#         # diffY = feat_enc.size()[2] - feat_dec.size()[2]
#         # diffX = feat_enc.size()[3] - feat_dec.size()[3]
#         # if diffY != 0 or diffX != 0:
#         #     print('Padding for size mismatch ( Enc:', feat_enc.size(), 'Dec:', feat_dec.size(),')')
#         #     feat_dec = F.pad(feat_dec, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
#         # feat = torch.cat([feat_dec,feat_enc],dim=1)
#         feat = self.layer2(feat)
#         feat = self.layer3(feat)
#         return feat