from functools import partial
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath





class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        attention_map = attn

        # print("Attention Weights:", attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # print("Attention Weights Shape:", x.shape)
        x = self.proj(x)
        x = self.proj_drop(x) #B 180 22 256 1 180 512

        return x,attn


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        out, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x,attn

class ST(nn.Module):
    def __init__(self,class_num,num_frame=180, num_joints=22, in_chans=128, embed_dim_ratio=256, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None):

        super().__init__()
        self.class_num = class_num
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * 2   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = num_joints * 3     #### output dimension is num_joints * 3

        ### spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        # self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints + 1 , embed_dim_ratio))
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints , embed_dim_ratio))
        self.Spatial_cls_token = nn.Parameter(torch.randn(1, 1, embed_dim_ratio))

        ### temporal_pos_embedding
        self.Temporal_patch_to_embedding = nn.Linear(embed_dim_ratio, embed_dim)
        # self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame + 1 , embed_dim))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))					# nn.Parameter()�I�f`�p
        self.pos_drop = nn.Dropout(p=drop_rate)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        ####### A easy way to implement weighted mean
        pool = 'cls'

        assert pool in {'cls', 'mean'}

        self.pool = pool
        self.to_latent = nn.Identity()
        self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame, out_channels=1, kernel_size=1)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, class_num)
        )

        self.fcn = nn.Conv1d(512,  class_num, kernel_size=1)
        # conv_init(self.fcn)

    # @get_local('attention_str_map')
    # def Spatial_forward_features(self, x):
    #     # print("x.start",x.shape)
    #     b, _, f, p = x.shape  ##### b is batch size, f is number of frames, p is number of joints
    #     x = rearrange(x, 'b c f p  -> (b f) p  c', )
    #     x = self.Spatial_patch_to_embedding(x)
    #     # print("x1",x.shape) #2880 22 256
    #     # print("self.Spatial_pos_embed", self.Spatial_pos_embed.shape) # 1 22 256
    #     x += self.Spatial_pos_embed
    #     # print("x2", x.shape)
    #     x = self.pos_drop(x)
    #
    #     for blk in self.Spatial_blocks:
    #         x = blk(x)
    #
    #     attention_str_map = x
    #     # print("str:",x.shape)   # 180 22 256
    #     b, p, c = x.shape
    #
    #     x = rearrange(x, 'b  p  c  -> b  c  p', )
    #
    #     x = F.avg_pool2d(x, kernel_size=(1, p))  # 2880 22 256
    #     # print("mean",x.shape)  180 256 1
    #     # print("x.spatial",x.shape)
    #     x = rearrange(x, '(b f) c p -> b f (c p)', f=f)
    #     # print("x.spatial",x.shape)
    #     return x
    #
    # @get_local('attention_ttr_map')
    # # @get_local('ttr_feature')
    # def forward_features(self, x):
    #     # print("x_start",x.shape)   #32 180 256
    #     b = x.shape[0]
    #     n = x.shape[1]
    #     x = self.Temporal_patch_to_embedding(x)
    #
    #     x += self.Temporal_pos_embed
    #     # print("x4", x.shape)
    #     x = self.pos_drop(x)
    #     for blk in self.blocks:
    #         x = blk(x)
    #
    #     attention_ttr_map = x
    #
    #     b, f, c = x.shape
    #
    #     x = rearrange(x, 'b  f  c  -> b  c  f', )
    #
    #     # print("mean",x.shape) # 1 512 180
    #     # x = F.avg_pool2d(x, kernel_size=(1, f))
    #     x = F.max_pool2d(x, kernel_size=(1, f))
    #     ttr_feature = x
    #     # print("x_mid", x.shape)
    #     # 32 512 1
    #     x = x.mean(dim=2)
    #
    #     x = self.mlp_head(x)
    #
    #     return x, ttr_feature
    #
    # @get_local('feature')
    # def forward(self, x):
    #
    #     x = self.Spatial_forward_features(x)
    #
    #     pred, ttr_feature = self.forward_features(x)
    #     feature = ttr_feature
    #     return pred

    #/////8888888888888888888
    # @get_local('attention_str_map')
    def Spatial_forward_features(self, x):
        # print("x.start",x.shape)
        b, _, f, p = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(x, 'b c f p  -> (b f) p  c', )
        x = self.Spatial_patch_to_embedding(x)
        # print("x1",x.shape) #2880 22 256
        # print("self.Spatial_pos_embed", self.Spatial_pos_embed.shape) # 1 22 256
        x += self.Spatial_pos_embed
        # print("x2", x.shape)
        x = self.pos_drop(x)

        attn_weights_all = []
        for blk in self.Spatial_blocks:
            x_out ,attn= blk(x)
            attn_weights_all.append(attn)
        attention_str_map = x_out
        # print("str:",x.shape)   # 180 22 256
        b, p, c = x_out.shape

        x_out = rearrange(x_out, 'b  p  c  -> b  c  p', )
        new_p = int(p / 2)
        x_out = F.avg_pool2d(x_out, kernel_size=(1, new_p))  # 2880 22 256
        # print("mean",x.shape)  180 256 1
        # print("x.spatial",x.shape)
        x_out = rearrange(x_out, '(b f) c p -> b f c p', f=f)

        return x_out,attn_weights_all


    # @get_local('ttr_feature')
    def forward_features(self, x):
        # print("x_start",x.shape)   #32 180 256
        p = x.shape[3]
        x = rearrange(x, 'b f c p  -> (b p) f  c', )
        x = self.Temporal_patch_to_embedding(x)

        x += self.Temporal_pos_embed
        # print("x4", x.shape)
        x = self.pos_drop(x)
        attn_weights_all = []
        for blk in self.blocks:
            x_out, attn = blk(x)
            attn_weights_all.append(attn)

        attention_ttr_map = x_out

        b, f, c = x_out.shape

        x_out = rearrange(x_out, 'b  f  c  -> b  c  f', )

        # print("mean",x.shape) # 1 512 180
        # x = F.avg_pool2d(x, kernel_size=(1, f))
        x_out = F.max_pool2d(x_out, kernel_size=(1, f))

        x_out = x_out.squeeze(2)

        x_out = rearrange(x_out, '(b p)  c  -> b  c  p', p = p)
        ttr_feature = x_out
        # print("x_mid", x.shape)
        # 32 512 1
        x_out = x_out.mean(dim=2)

        x_out = self.mlp_head(x_out)
        # print("x", x.shape)
        return x_out, attn_weights_all


    def forward(self, x):

        # x = self.Spatial_forward_features(x)
        #
        # pred, ttr_feature = self.forward_features(x)

        x, spatial_attns = self.Spatial_forward_features(x)
        pred, temporal_attns = self.forward_features(x)
        return pred, spatial_attns, temporal_attns