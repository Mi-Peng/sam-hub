
from functools import partial   
import torch
import torch.nn as nn

from utils.configurable import configurable
from models.build import MODELS_REGISTRY
# Vision Transformer


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0 or not training:
        return x 
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None) -> None:
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)



class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super(PatchEmbed, self).__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size= patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_channels=in_c, 
            out_channels=embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1]

        x = self.proj(x) # [B,C_in,H,W] -> [B,embed_dim,grid_size,grid_size]
        x = x.flatten(2) # [B,embed_dim,grid_size,grid_size] -> [B,embed_dim,num_patches]
        x = x.transpose(1, 2) # [B,num_patches,embed_dim]
        return self.norm(x)


class Attention(nn.Module):
    def __init__(self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop_ratio=0.,
        proj_drop_ratio=0.
    ):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        B, N, D = x.shape # [B, num_patches+1, embed_dim]
        
        qkv = self.qkv(x) # [B, num_patches+1, embed_dim] -> [B, num_patches+1, embed_dim*3]
        qkv = qkv.reshape(B, N, 3, self.num_heads, D // self.num_heads) # [B, num_patches+1, embed_dim*3] -> [B, num_patches+1, 3, num_heads, embed_dim//num_heads]
        qkv = qkv.permute(2, 0, 3, 1, 4) # [B, num_patches+1, 3, num_heads, embed_dim//num_heads] -> [3, B, num_heads, num_patches+1, embed_dim//num_heads]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # q,k        : [B, num_heads, num_patches+1, embed_dim//num_heads]
        # k.transpose: [B, num_heads, embed_dim//num_heads, num_patches+1]
        attention = (q @ k.transpose(-2,-1)) * self.scale # attention: [B, num_heads, embed_dim//num_heads]
        attention = attention.softmax(dim=-1) # softmax(q * k / \sqrt{d})
        attention = self.attn_drop(attention)

        # attention: [B, num_heads, num_patches+1, num_patches+1]
        # v:         [B, num_heads, num_patches+1, embed_dim//num_heads]
        # transpose: [B, num_heads, num_patches+1, embed_dim//num_heads] -> [B, num_patches+1, num_heads, embed_dim//num_heads]
        # reshape: [B, num_patches+1, num_heads, embed_dim//num_heads] -> [B, num_patches+1, embed_dim]
        x = (attention @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x) # Out = W^o * V
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(MLP, self).__init__()
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

class Block(nn.Module):
    def __init__(self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop_ratio=0.,
        attn_drop_ratio=0.,
        drop_path_ratio=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm
    ):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_ratio=attn_drop_ratio,
            proj_drop_ratio=drop_ratio
        )
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop_ratio,
        )
    
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self,
        img_size=224,
        patch_size=16,
        in_c=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        qkv_scale=None,
        drop_ratio=0.,
        attn_drop_ratio=0.,
        drop_path_ratio=0.,
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None
    ):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_c=in_c,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) # [B, 1, D]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(drop_ratio)

        drop_ratio_depth = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qkv_scale,
                drop_ratio=drop_ratio,
                attn_drop_ratio=attn_drop_ratio,
                drop_path_ratio=drop_ratio_depth[i],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # classifier:
        self.classifier = nn.Linear(self.num_features, num_classes)

        # weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        x = self.patch_embed(x) # [B, C, H, W] -> [B, N, D]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) # [1, 1, 768] -> [B, 1, 768]
        x = torch.cat([cls_token, x], dim=1) # [B, N, D] -> [B, N+1, D]
        
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        out = self.classifier(x)
        return out


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def _cfg_to_vit(cfg):
    return {
        "num_classes": cfg.data.dataset.n_classes,
    }


@MODELS_REGISTRY.register()
@configurable(from_config=_cfg_to_vit)
def vit_tiny_patch16_224(num_classes):
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        num_classes=num_classes
    )
    return model

@MODELS_REGISTRY.register()
@configurable(from_config=_cfg_to_vit)
def vit_small_patch16_224(num_classes):
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_classes=num_classes
    )
    return model

@MODELS_REGISTRY.register()
@configurable(from_config=_cfg_to_vit)
def vit_base_patch16_224(num_classes):
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_classes=num_classes
    )
    return model