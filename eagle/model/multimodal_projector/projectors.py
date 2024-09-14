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
    config: CabsConfig, num_input_tokens: List[int], vision_hidden_size: int
):
    # pos emb
    if config.pos_emb:
        pos_emb1 = torch.nn.Parameter(torch.zeros(1, num_input_tokens[0], vision_hidden_size))
        pos_emb2 = torch.nn.Parameter(torch.zeros(1, num_input_tokens[1], vision_hidden_size))
        pos_emb3 = torch.nn.Parameter(torch.zeros(1, num_input_tokens[2], vision_hidden_size))

        pos_emb = torch.cat((pos_emb1, pos_emb2, pos_emb3), dim=1).to(torch.bfloat16)
        nn.init.trunc_normal_(pos_emb, mean=0.0, std=0.02)
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
        prenorm = LayerNorm(config.encoder_hidden_size)
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
        if self.prenorm is not None:
            x = self.prenorm(x)

        if self.pos_emb is not None:
            x += self.pos_emb

        x = self._forward(x)  # (B, L, output_hidden_size)

        B = x.size(0)
        if self.eos_tokens is not None:
            x = torch.cat([x, self.eos_tokens.expand(B, -1, -1)], dim=1)

        output = BaseModelOutput(last_hidden_state=x)
        return output
    
    def _load_from_state_dict(self, state_dict, *args, **kwargs):
        # update old ckpt compatible with current code
        pos_emb = state_dict["abstractor.pos_emb"]
        if pos_emb.size(1) == self.pos_emb.size(1) + 1:
            # remove obsolete first pos emb (for cls token originally)
            state_dict["abstractor.pos_emb"] = pos_emb[:, 1:]

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
        # x: [B, L, dim]
        hw = int(x.size(1) ** 0.5)
        x1 = x[:, :2048, :]
        x2 = x[:, 2048:2048+732, :]
        x3 = x[:, 2048+732:, :]

        x1 = F.interpolate(x1.unsqueeze(1), size=(2025, 1152), mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2.unsqueeze(1), size=(729, 1152), mode='bilinear', align_corners=False)
        x3 = F.interpolate(x3.unsqueeze(1), size=(2025, 1152), mode='bilinear', align_corners=False)


        x1 = x1.squeeze(1).to("cuda")
        x2 = x2.squeeze(1).to("cuda")
        x3 = x3.squeeze(1).to("cuda")

        x1 = rearrange(x1, "b (h w) d -> b d h w", h = 45, w = 45)
        x1 = self.net(x1)

        x2 = rearrange(x2, "b (h w) d -> b d h w", h = 27, w = 27)
        x2 = self.net(x2)

        x3 = rearrange(x3, "b (h w) d -> b d h w", h = 45, w = 45)
        x3 = self.net(x3)

        x1 = rearrange(x1, "b d h w -> b (h w) d")
        x2 = rearrange(x2, "b d h w -> b (h w) d")
        x3 = rearrange(x3, "b d h w -> b (h w) d")

        x1 = self.readout(x1)
        x2 = self.readout(x2)
        x3 = self.readout(x3)

        x = torch.cat((x1, x2, x3), dim=1)
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

        s1 = RegBlock(
            depth,
            encoder_hidden_size,
            hidden_size,
        )
        sampler = nn.AdaptiveAvgPool2d((hw, hw))
        s2 = RegBlock(
            depth,
            hidden_size,
            hidden_size,
        )

        if depth:
            self.net = nn.Sequential(s1, sampler, s2)
            self.readout = build_mlp(mlp_depth, hidden_size, output_hidden_size)
        else:
            self.net = sampler
            self.readout = build_mlp(mlp_depth, encoder_hidden_size, output_hidden_size)
