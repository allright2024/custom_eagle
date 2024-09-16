# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .convnext_encoder import ConvNextVisionTower
from .hr_clip_encoder import HRCLIPVisionTower
from .vision_models.eva_vit import EVAVITVisionTower
from .sam_encoder import SAMVisionTower
from .pix2struct_encoder import Pix2StructLargeVisionTower
from .donut_encoder import DonutVisionTower
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from .siglip_encoder import SigLipVisionTower
from copy import deepcopy
import random
import math

class MultiBackboneChannelConcatenationVisionTower(nn.Module):
    def __init__(self,
                 vision_tower,
                 args,
                 grid_size=32):
        
        super().__init__()

        self.is_loaded = False
        self.grid_size = grid_size
        self.num_tokens = self.grid_size ** 2
        
        vision_tower_name_list = vision_tower.split(";")
        self.input_image_size = 1024 # hardcode
        self.load_vision_towers(vision_tower_name_list, args)

      
    def load_vision_towers(self, vision_tower_name_list, args):
        self.vision_towers = nn.ModuleList()
        
        for name in vision_tower_name_list:
            if name == "mPLUG/TinyChart-3B-768-siglip":
                siglip_vision_tower = SigLipVisionTower("mPLUG/TinyChart-3B-768-siglip")
                siglip_vision_tower.load_model()
                self.vision_towers.append(siglip_vision_tower)
            if name == 'khhuang/chart-to-table':
                donut_args = deepcopy(args)
                donut_args.input_image_size = 1024
                donut_args.de_normalize=False
                donut_args.freeze_vision = False
                donut_vision_tower = DonutVisionTower('khhuang/chart-to-table', donut_args)     
                donut_vision_tower.load_model()
                self.vision_towers.append(donut_vision_tower)
            elif name == 'google/deplot':
                pix_args = deepcopy(args)
                pix_args.input_image_size = 1024
                pix_args.freeze_vision = False
                pix_args.do_resize = True
                pix_args.de_normalize = True
                deplot_vision_tower = Pix2StructLargeVisionTower("google/deplot", pix_args)     
                deplot_vision_tower.load_model()
                self.vision_towers.append(deplot_vision_tower)
            elif name == "google/matcha-chart2text-pew":
                pix_args = deepcopy(args)
                pix_args.input_image_size = 1024
                pix_args.freeze_vision = False
                pix_args.do_resize = True
                pix_args.de_normalize = True
                deplot_vision_tower = Pix2StructLargeVisionTower("google/matcha-chart2text-pew", pix_args)     
                deplot_vision_tower.load_model()
                self.vision_towers.append(deplot_vision_tower)
            elif name == "sam-1024":
                sam_args = deepcopy(args)
                sam_args.freeze_vision = False
                sam_args.input_image_size = 1024
                sam_args.add_pixel_shuffle = True
                sam_vision_tower = SAMVisionTower("SAM-L", sam_args)
                sam_vision_tower.load_model()
                self.vision_towers.append(sam_vision_tower)

            elif name == 'google/pix2struct-large':
                pix_args = deepcopy(args)
                #pix_args.freeze_vision = True
                pix_args.input_image_size = 1024
                pix_args.freeze_vision = False
                pix_args.do_resize = True
                pix_args.de_normalize = True
                pix_vision_tower = Pix2StructLargeVisionTower("google/pix2struct-large", pix_args)     
                pix_vision_tower.load_model()
                self.vision_towers.append(pix_vision_tower)

            elif name == 'clip-448': # 그대로 사용
                clip_args = deepcopy(args)
                clip_args.input_image_size = 336 # actually 448, will have no effect
                clip_args.freeze_vision = False
                clip_vision_tower = HRCLIPVisionTower("openai/clip-vit-large-patch14-336", clip_args)     
                clip_vision_tower.load_model()
                self.vision_towers.append(clip_vision_tower)
        
        # a hardcode here, so we always use convnext in the vision encoder mixture
        self.image_processor = siglip_vision_tower.image_processor
        self.is_loaded = True

    def load_model(self):
        assert self.is_loaded, "All the vision encoders should be loaded during initialization!"

    def forward(self, x):
        features = []
        
        for vision_tower in self.vision_towers:
            try:
                from PIL import Image
                def expand2square(pil_imgs, background_color):
                    arr_img = []
                    for pil_img in pil_imgs:
                        width, height = pil_img.size
                        if width == height:
                            arr_img.append(pil_img)
                        elif width > height:
                            result = Image.new(pil_img.mode, (width, width), background_color)
                            result.paste(pil_img, (0, (width - height) // 2))
                            arr_img.append(result)
                        else:
                            result = Image.new(pil_img.mode, (height, height), background_color)
                            result.paste(pil_img, ((height - width) // 2, 0))
                            arr_img.append(result)
                    return arr_img
                
                squared_x = expand2square(x, tuple(int(t*255) for t in [0.5, 0.5, 0.5]))
                processed_image = vision_tower.image_processor.preprocess(squared_x, return_tensors='pt')['pixel_values']
                feature = vision_tower(processed_image)
            except:
                processed_image = vision_tower.image_processor(x, return_tensors='pt')
                feature = vision_tower(**processed_image)
            features.append(feature)
        
        return features
        
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.clip_vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.clip_vision_tower.parameters()).device

    @property
    def config(self):
        assert NotImplementedError
        pass

    @property
    def hidden_size(self):
        return sum([_.hidden_size for _ in self.vision_towers])

    @property
    def num_patches(self):
        return self.num_tokens
