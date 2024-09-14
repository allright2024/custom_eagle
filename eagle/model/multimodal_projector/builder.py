import torch
import torch.nn as nn
import re

from .projectors import CAbstractor
from .dabs import DAbstractor
from .configuration import CabsConfig

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, fpn_input_dim=[], **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    
    if projector_type == "cabstractor":
        num_input_tokens = [2048, 732, 2048] # 하드코딩(이미지 인코더 특성에 따라 정해주는 것)
        
        config = CabsConfig(output_hidden_size = 2048, 
                            depth = 3, 
                            mlp_depth = 2, 
                            num_query_tokens = 144, 
                            hidden_size = 1024, 
                            encoder_hidden_size = 1152) # honeybee에서 가져온 정보
        return CAbstractor(config, num_input_tokens=num_input_tokens)
    
    if projector_type == "dabstractor":
        return DAbstractor()

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
