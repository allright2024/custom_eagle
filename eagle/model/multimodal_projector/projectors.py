"""Custom Honeybee projectors based on Conv and MLP, including C-Abstractor.
"""
from functools import partial

from typing import List
import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import LayerNorm, LayerNorm2d
from timm.models.regnet import RegStage
from transformers.modeling_outputs import BaseModelOutput
import torch.nn.functional as F

from .configuration import HoneybeeVisualProjectorConfig, CabsConfig


def build_pos_embeds(
    config: CabsConfig, num_input_tokens: List[int], vision_hidden_size: List[int]
):
    # pos emb
    if config.pos_emb:
        pos_emb1 = torch.nn.Parameter(torch.zeros(1, num_input_tokens[0], vision_hidden_size[0])).to("cuda")
        pos_emb2 = torch.nn.Parameter(torch.zeros(1, num_input_tokens[1], vision_hidden_size[1])).to("cuda")
        pos_emb3 = torch.nn.Parameter(torch.zeros(1, num_input_tokens[2], vision_hidden_size[2])).to("cuda")

        # pos_emb = torch.cat((pos_emb1, pos_emb2, pos_emb3), dim=1).to(torch.bfloat16)
        nn.init.trunc_normal_(pos_emb1, mean=0.0, std=0.02)
        nn.init.trunc_normal_(pos_emb2, mean=0.0, std=0.02)
        nn.init.trunc_normal_(pos_emb3, mean=0.0, std=0.02)

        pos_emb = torch.zeros((1, 4828, 1152)).to("cuda")
        pos_emb[:, :2048, :768] = pos_emb1.detach()
        pos_emb[:, 2048:2048+732, :1152] = pos_emb2.detach()
        pos_emb[:, 2048+732:, :768] = pos_emb3.detach()
        # pos_emb = pos_emb1
        pos_emb = pos_emb.to("cuda")
        # pos_emb = [pos_emb1, pos_emb2, pos_emb3]
    else:
        pos_emb = None

    return pos_emb


def build_eos_tokens(config: CabsConfig, output_hidden_size: int):
    # think tokens
    num_eos_tokens = config.num_eos_tokens
    if num_eos_tokens:
        eos_tokens = torch.nn.Parameter(torch.randn(1, num_eos_tokens, output_hidden_size))
        nn.init.trunc_normal_(eos_tokens, mean=0.0, std=config.initializer_range)
    else:
        eos_tokens = None

    return eos_tokens


def build_prenorm(config: CabsConfig):
    if getattr(config, "prenorm", False):
        prenorm = [LayerNorm(config.encoder_hidden_size[0]), LayerNorm(config.encoder_hidden_size[1]), LayerNorm(config.encoder_hidden_size[2])]
    else:
        prenorm = None
    return prenorm


def build_mlp(depth: int, hidden_size: int, output_hidden_size: int):
    layers = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        layers.append(nn.SiLU())
        layers.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*layers)


class Projector(nn.Module):
    """Base projector class"""

    def __init__(
        self,
        config: CabsConfig,
        num_input_tokens: List[int],
    ):
        super().__init__()
        self.config = config
        self.num_input_tokens = num_input_tokens

        # think tokens
        self.eos_tokens = build_eos_tokens(config, config.output_hidden_size)

        # pos emb
        self.pos_emb = build_pos_embeds(config, num_input_tokens, config.encoder_hidden_size)

        self.prenorm = build_prenorm(config)

        self.build_net()

    def build_net(self):
        raise NotImplementedError()

    def _forward(self, x):
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, encoder_hidden_size) tensor from the visual backbone (CLIP visual encoder),
                including cls token.
        """
        # 현재 None으로 설정
        if self.prenorm is not None:
            x = self.prenorm(x)
        
        x = x.to("cuda")

        if self.pos_emb is not None:
            x[:, :4828, :] += self.pos_emb

        x = self._forward(x)  # (B, L, output_hidden_size)

        B = x.size(0)
        if self.eos_tokens is not None:
            x = torch.cat([x, self.eos_tokens.expand(B, -1, -1)], dim=1)

        output = BaseModelOutput(last_hidden_state=x)
        return output
    
    def _load_from_state_dict(self, state_dict, *args, **kwargs):
        # update old ckpt compatible with current code
        # 이 부분은 state_dict에 없어서 못 불러오는 중..
        # pos_emb = state_dict["abstractor.pos_emb"]
        # if pos_emb.size(1) == self.pos_emb.size(1) + 1:
        #     # remove obsolete first pos emb (for cls token originally)
        #     state_dict["abstractor.pos_emb"] = pos_emb[:, 1:]

        super()._load_from_state_dict(state_dict, *args, **kwargs)


class MLPProjector(Projector):
    def build_net(self):
        encoder_hidden_size = self.config.encoder_hidden_size
        output_hidden_size = self.config.output_hidden_size
        depth = self.config.depth

        self.net = build_mlp(depth, encoder_hidden_size, output_hidden_size)
        
    def _forward(self, x):
        return self.net(x)


class ConvProjector(Projector):
    def _forward(self, x):
        x = x.to(torch.bfloat16) # Float16
        # self.net = self.net.float() # Float32 오류 있음
        # self.readout = self.readout.float() # Float32
        

        # x: [B, L, dim]
        # padded_x = torch.full((x.shape[0], 4900, x.shape[2]), -100, dtype=torch.bfloat16).to("cuda")
        if x.shape[1] < 4900: # 다른 resize technique를 써서, 4828이 나올때, 4900으로 padding해주기 
            padding_size = 4900 - x.shape[1]
            x = F.pad(x, (0, 0, 0, padding_size))
        x = rearrange(x, "b (h w) d -> b d h w", h=70, w=70)

        x = self.net(x)
        x = rearrange(x, "b d h w -> b (h w) d")
        x = self.readout(x)
        return x


class CAbstractor(ConvProjector):
    """C-Abstractor based on RegBlock"""
    def build_net(self):
        encoder_hidden_size = self.config.encoder_hidden_size
        hidden_size = self.config.hidden_size
        output_hidden_size = self.config.output_hidden_size
        depth = self.config.depth
        mlp_depth = self.config.mlp_depth

        n_queries = self.config.num_query_tokens
        assert (n_queries ** 0.5).is_integer(), "n_queries must be square number"
        hw = int(n_queries ** 0.5)

        RegBlock = partial(
            RegStage,
            stride=1,
            dilation=1,
            act_layer=nn.SiLU,
            norm_layer=LayerNorm2d,
        )

        s1_p = RegBlock(
            depth,
            encoder_hidden_size[1],
            hidden_size,
        )
        sampler = nn.AdaptiveAvgPool2d((hw, hw))
        s2 = RegBlock(
            depth,
            hidden_size,
            hidden_size,
        )

        if depth:
            self.net = nn.Sequential(s1_p, sampler, s2)
            self.readout = build_mlp(mlp_depth, hidden_size, output_hidden_size)
        else:
            self.net = sampler
            self.readout = build_mlp(mlp_depth, encoder_hidden_size, output_hidden_size)
