from typing import Iterable
from typing import Tuple, Dict

from codriving import CODRIVING_REGISTRY

import torch.nn.functional as F
import torch.nn as nn
import torch
import logging

from .planning_end2end import WaypointPlanner_e2e
from common.torch_helper import load_checkpoint as load_planning_model_checkpoint
_logger = logging.getLogger(__name__)
import os

from einops import rearrange

class Conv3D(nn.Module):
    def __init__(self, in_channel : int, out_channel : int, kernel_size : int, stride : int, padding : int):
        super(Conv3D, self).__init__()
        self.conv3d = nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn3d = nn.BatchNorm3d(out_channel)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # input x: (batch, seq, c, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, c, seq_len, h, w)
        x = F.relu(self.bn3d(self.conv3d(x)))
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, seq_len, c, h, w)

        return x
    

class MLP(nn.Module):
	def __init__(self, in_feat : int, out_feat : int, hid_feat : Iterable[int]=(1024, 512), activation=None, dropout=-1):
		super(MLP, self).__init__()
		dims = (in_feat, ) + hid_feat + (out_feat, )

		self.layers = nn.ModuleList()
		for i in range(len(dims) - 1):
			self.layers.append(nn.Linear(dims[i], dims[i + 1]))

		self.activation = activation if activation is not None else lambda x: x
		self.dropout = nn.Dropout(dropout) if dropout != -1 else lambda x: x

	def forward(self, x : torch.Tensor) -> torch.Tensor:
		for i in range(len(self.layers)):
			x = self.activation(x)
			x = self.dropout(x)
			x = self.layers[i](x)
		return x


@CODRIVING_REGISTRY.register
class WaypointPlanner_e2e_cmd(nn.Module):
    """
    WaypointPlanner with BEV feature, navigation commands, occupancy map and road map as inputs
    """
    def __init__(self,feature_dir=128):
        super().__init__()
        height_feat_size = 6
        self.conv_pre_1 = nn.Conv2d(height_feat_size, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(32)
        self.bn_pre_2 = nn.BatchNorm2d(32)

        ####### for extral feature
        self.conv_pre_1_f = nn.Conv2d(feature_dir, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2_f = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1_f = nn.BatchNorm2d(32)
        self.bn_pre_2_f = nn.BatchNorm2d(32)

        self.conv_pre_1_f2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2_f2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1_f2 = nn.BatchNorm2d(32)
        self.bn_pre_2_f2 = nn.BatchNorm2d(32)
        #######

        self.conv3d_1 = Conv3D(64, 64, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv3d_2 = Conv3D(128, 128, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.bn3_1 = nn.BatchNorm2d(256)

        self.decoder_640 = MLP(256+128+128+128, 20, hid_feat=(1025, 512))

        self.target_encoder = MLP(2, 128, hid_feat=(16, 64))
        self.cmd_direction_encoder = MLP(6, 128, hid_feat=(48, 64))
        self.cmd_speed_encoder = MLP(4, 128, hid_feat=(32, 64))

    def reset_parameters(self):
        pass
    

    def forward(self, input_data : Dict) -> torch.Tensor:
        """Forward method for WaypointPlanner

        Args:
            input_data: input data to forward

                required keys:

                - occupancy: rasterized map from perception results
                - target: target point to go to

        Return:
            torch.Tensor: predicted waypoints
        """
        occupancy = input_data["occupancy"]  # B,T,C,H,W = ([2, 5, 6, 192, 96])
        batch, seq, c, h, w = occupancy.size()

        x = occupancy.view(-1, c, h, w)  # batch*seq, c, h, w ([10, 6, 192, 96])
        x = F.relu(self.bn_pre_1(self.conv_pre_1(x))) # ([batch*seq, 32, 192, 96]) ([10, 32, 192, 96])
        x = F.relu(self.bn_pre_2(self.conv_pre_2(x))) # ([batch*seq, 32, 192, 96])

        ########### feature embedding ##############
        features_list = input_data['feature_warpped_list']
        feature = torch.cat(features_list, dim=0)

        batch, seq, c2, h, w = feature.size()

        xf = feature.view(-1, c2, h, w)  # batch*seq, c, h, w ([10, 128, 192, 96])
        xf = F.relu(self.bn_pre_1_f(self.conv_pre_1_f(xf))) # ([batch*seq, 32, 192, 96]) ([10, 32, 192, 96])
        xf = F.relu(self.bn_pre_2_f(self.conv_pre_2_f(xf))) # ([batch*seq, 32, 192, 96])

        # concatenate feature and embedd again
        x_enhanced = torch.cat((x,xf), dim=1)  # batch*seq, c, h, w ([10, 64, 192, 96])
        x_enhanced = F.relu(self.bn_pre_1_f2(self.conv_pre_1_f2(x_enhanced))) # ([batch*seq, 32, 192, 96]) ([10, 32, 192, 96])
        x_enhanced = F.relu(self.bn_pre_2_f2(self.conv_pre_2_f2(x_enhanced))) # ([batch*seq, 32, 192, 96])



        # -------------------------------- Encoder Path --------------------------------
        # -- STC block 1
        # x_1 = F.relu(self.bn1_1(self.conv1_1(x))) # ([10, 64, 96, 48])
        x_1 = F.relu(self.bn1_1(self.conv1_1(x_enhanced)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1))) # ([10, 64, 96, 48])


        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()  # (batch, seq, c, h, w)  ([2, 5, 64, 96, 48])
        x_1 = self.conv3d_1(x_1)  # ([2, 3, 64, 96, 48])
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()  # (batch * seq, c, h, w) ([6, 64, 96, 48])

        # -- STC block 2
        x_2 = F.relu(self.bn2_1(self.conv2_1(x_1))) # ([6, 128, 48, 24])
        x_2 = F.relu(self.bn2_2(self.conv2_2(x_2))) # ([6, 128, 48, 24])

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)).contiguous()  # (batch, seq, c, h, w) ([2, 3, 128, 48, 24])
        x_2 = self.conv3d_2(x_2)  #  ([2, 1, 128, 48, 24])
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()  # (batch * seq, c, h, w), seq = 1 ([2, 128, 48, 24])

        x_3 = F.relu(self.bn3_1(self.conv3_1(x_2)))  #  ([2, 256, 24, 12])

        feature = x_3.mean(dim=(2, 3))  # NOTE: adopt the mean pooling!  ([2, 256])
        feature_target = self.target_encoder(input_data['target'])  #  ([2, 128]) 2-16-64-128
        feature_cmd_dir = self.cmd_direction_encoder(input_data['cmd_direction']) # ([2, 128]) 6-48-64-128
        feature_cmd_speed = self.cmd_speed_encoder(input_data['cmd_speed']) # ([2, 128]) 4-32-64-128
        future_waypoints = self.decoder_640(torch.cat((feature, feature_target, feature_cmd_dir, feature_cmd_speed), dim=1)).contiguous().view(batch, 10, 2)  #  ([2, 10, 2]) 640-1025-512-20
        output_data = dict(future_waypoints=future_waypoints)

        return output_data



@CODRIVING_REGISTRY.register
class WaypointPlanner_e2e_cmd_attn(nn.Module):
    """
    WaypointPlanner with BEV feature, navigation commands, occupancy map and road map as inputs
    """
    def __init__(self,feature_dir=128):
        super().__init__()
        self.t = 0
        ####### occupancy
        height_feat_size = 6
        self.conv_pre_1 = nn.Conv2d(height_feat_size, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(32)
        self.bn_pre_2 = nn.BatchNorm2d(32)

        ####### bev feature
        self.conv_pre_1_f = nn.Conv2d(feature_dir, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2_f = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1_f = nn.BatchNorm2d(32)
        self.bn_pre_2_f = nn.BatchNorm2d(32)

        ####### feature
        self.conv_pre_1_f2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2_f2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1_f2 = nn.BatchNorm2d(32)
        self.bn_pre_2_f2 = nn.BatchNorm2d(32)

        ####### STC
        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.conv3d_1 = Conv3D(64, 64, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.conv3d_2 = Conv3D(128, 128, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.bn3_2 = nn.BatchNorm2d(256)

        #######
        embed_dims = 256
        self.direction_embedding = nn.Embedding(6, embed_dims)
        self.speed_embedding = nn.Embedding(4, embed_dims)
        self.target_encoder = MLP(2, embed_dims, hid_feat=(16, 64))

        self.mlp_fuser = nn.Sequential(
                nn.Linear(embed_dims*3, embed_dims),
                nn.LayerNorm(embed_dims),
                nn.ReLU(inplace=True),
            )

        attn_module_layer = nn.TransformerDecoderLayer(embed_dims, 8, dim_feedforward=embed_dims*2, dropout=0.1, batch_first=False)
        self.attn_module = nn.TransformerDecoder(attn_module_layer, 3)
        
        self.reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 10*2),
        )

    def reset_parameters(self):
        pass

    def forward(self, input_data : Dict) -> torch.Tensor:
        """Forward method for WaypointPlanner

        Args:
            input_data: input data to forward

                required keys:

                - occupancy: rasterized map from perception results
                - target: target point to go to

        Return:
            torch.Tensor: predicted waypoints
        """
        # import matplotlib.pyplot as plt
        # import torch
        # import random
        # import os
        # def save_random_name_image(tensor_data, colormap, output_dir, flag):
        #     """
        #     Save tensor data as an image with a random file name and no extra content.
            
        #     Args:
        #         tensor_data (torch.Tensor): Tensor of shape (C, H, W) to visualize.
        #         colormap (str): Colormap for visualization ('gray' for occupancy, 'viridis' for feature).
        #         output_dir (str): Directory to save images.
        #     """
        #     # Create output directory if it doesn't exist
        #     os.makedirs(output_dir, exist_ok=True)
            
        #     # Generate random file name
        #     random_filename = str(self.t) + f"{flag}.jpg"
        #     # Convert tensor to numpy for visualization
            
        #     img_data = tensor_data.cpu().numpy()
                
            
        #     # Plot and save the image without any extra content
        #     try:
        #         plt.imshow(img_data, cmap=colormap)
        #     except:
        #         img_data = tensor_data.mean(dim=0).cpu().numpy()
        #         plt.imshow(img_data, cmap=colormap)
        #     plt.axis('off')
        #     plt.savefig(os.path.join(output_dir, random_filename), bbox_inches='tight', pad_inches=0)
        #     plt.close()

        # # Example usage with occupancy and feature tensors
        # def visualize_and_save(input_data, output_dir):
        #     # Occupancy: reshape for visualization (B*T, C, H, W)
        #     occupancy = input_data["occupancy"].view(-1, *input_data["occupancy"].size()[2:])
            
        #     for i in range(occupancy.size(0)):
        #         save_random_name_image(occupancy[i][0], 'gray', output_dir, "occupancy")  # Visualize first channel of each occupancy map

        #     # Feature: concatenate and reshape for visualization (N, C, H, W)
        #     features_list = input_data['feature_warpped_list']
        #     feature = torch.cat(features_list, dim=0)
            
        #     for i in range(feature.size(0)):
        #         save_random_name_image(feature[i][0], 'gray', output_dir, 'feature')  # Visualize first channel of each feature map
        #     self.t += 1
        occupancy = input_data["occupancy"]  # B,T,C,H,W = ([2, 5, 6, 192, 96])
        batch, seq, c, h, w = occupancy.size()

        x = occupancy.view(-1, c, h, w)  # batch*seq, c, h, w ([10, 6, 192, 96])
        x = F.relu(self.bn_pre_1(self.conv_pre_1(x))) # ([batch*seq, 32, 192, 96]) ([10, 32, 192, 96])
        x = F.relu(self.bn_pre_2(self.conv_pre_2(x))) # ([batch*seq, 32, 192, 96])

        ########### feature embedding ##############
        features_list = input_data['feature_warpped_list']
        feature = torch.cat(features_list, dim=0)
        

        # visualize_and_save(input_data, 'tmp/vis_fea_occ')

        batch, seq, c2, h, w = feature.size()

        xf = feature.view(-1, c2, h, w)  # batch*seq, c, h, w ([10, 128, 192, 96])
        xf = F.relu(self.bn_pre_1_f(self.conv_pre_1_f(xf))) # ([batch*seq, 32, 192, 96]) ([10, 32, 192, 96])
        xf = F.relu(self.bn_pre_2_f(self.conv_pre_2_f(xf))) # ([batch*seq, 32, 192, 96])

        # concatenate feature and embedd again
        x_enhanced = torch.cat((x,xf), dim=1)  # batch*seq, c, h, w ([10, 64, 192, 96])
        x_enhanced = F.relu(self.bn_pre_1_f2(self.conv_pre_1_f2(x_enhanced))) # ([batch*seq, 32, 192, 96]) ([10, 32, 192, 96])
        x_enhanced = F.relu(self.bn_pre_2_f2(self.conv_pre_2_f2(x_enhanced))) # ([batch*seq, 32, 192, 96])

        # -- STC block 1
        x_1 = F.relu(self.bn1_1(self.conv1_1(x_enhanced))) # ([10, 64, 192, 96])
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1))) # ([10, 64, 192, 96])
        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()  # (batch, seq, c, h, w)  ([2, 5, 64, 192, 96])
        x_1 = self.conv3d_1(x_1)  # ([2, 3, 64, 192, 96])
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()  # (batch * seq, c, h, w) ([6, 64, 192, 96])

        # -- STC block 2
        x_2 = F.relu(self.bn2_1(self.conv2_1(x_1))) # ([6, 128, 96, 48])
        x_2 = F.relu(self.bn2_2(self.conv2_2(x_2))) # ([6, 128, 96, 48])
        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)).contiguous()  # (batch, seq, c, h, w) ([2, 3, 128, 96, 48])
        x_2 = self.conv3d_2(x_2)  #  ([2, 1, 128, 96, 48])
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()  # (batch * seq, c, h, w), seq = 1 ([2, 128, 96, 48])

        x_3 = F.relu(self.bn3_1(self.conv3_1(x_2)))  #  ([2, 256, 96, 48])
        x_3 = F.relu(self.bn3_2(self.conv3_2(x_3)))  #  ([2, 256, 96, 48])
        x_3 = rearrange(x_3, 'b c h w -> (h w) b c') #  ([96*48, 2, 256])
        
        direction_ind = torch.argmax(input_data['cmd_direction'], dim=1)
        dir_embed = self.direction_embedding.weight[direction_ind] # ([2, 256])
        speed_ind = torch.argmax(input_data['cmd_speed'], dim=1)
        speed_embed = self.speed_embedding.weight[speed_ind]   # ([2, 256])
        feature_target = self.target_encoder(input_data['target'])  # ([2, 256])

        cmd_feature = torch.cat([dir_embed, speed_embed, feature_target], dim=1)
        cmd_feature = self.mlp_fuser(cmd_feature).unsqueeze(0)   # ([1, 2, 256])

        plan_query = self.attn_module(cmd_feature, x_3)  # ([2, 256, 1])

        future_waypoints = self.reg_branch(plan_query.squeeze(-1)).contiguous().view(batch, 10, 2)  #  ([2, 10, 2])
        future_waypoints[...,:2] = torch.cumsum(future_waypoints[...,:2], dim=1)

        output_data = dict(future_waypoints=future_waypoints)

        return output_data



@CODRIVING_REGISTRY.register
class WaypointPlanner_e2e_cmd_attn_128(nn.Module):
    def __init__(self,feature_dir=128):
        super().__init__()
        ####### occupancy
        height_feat_size = 6
        self.conv_pre_1 = nn.Conv2d(height_feat_size, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(32)
        self.bn_pre_2 = nn.BatchNorm2d(32)

        ####### bev feature
        self.conv_pre_1_f = nn.Conv2d(feature_dir, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2_f = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1_f = nn.BatchNorm2d(32)
        self.bn_pre_2_f = nn.BatchNorm2d(32)

        ####### feature
        self.conv_pre_1_f2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2_f2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1_f2 = nn.BatchNorm2d(64)
        self.bn_pre_2_f2 = nn.BatchNorm2d(64)

        ####### STC
        # self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.bn1_1 = nn.BatchNorm2d(64)
        # self.bn1_2 = nn.BatchNorm2d(64)

        self.conv3d_1 = Conv3D(64, 64, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.conv3d_2 = Conv3D(128, 128, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv3_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.bn3_2 = nn.BatchNorm2d(128)

        #######
        embed_dims = 128
        self.direction_embedding = nn.Embedding(6, embed_dims)
        self.speed_embedding = nn.Embedding(4, embed_dims)
        self.target_encoder = MLP(2, embed_dims, hid_feat=(16, 64))

        self.mlp_fuser = nn.Sequential(
                nn.Linear(embed_dims*3, embed_dims),
                nn.LayerNorm(embed_dims),
                nn.ReLU(inplace=True),
            )

        attn_module_layer = nn.TransformerDecoderLayer(embed_dims, 8, dim_feedforward=embed_dims*2, dropout=0.1, batch_first=False)
        self.attn_module = nn.TransformerDecoder(attn_module_layer, 3)
        
        self.reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 10*2),
        )

    def reset_parameters(self):
        pass

    def forward(self, input_data : Dict) -> torch.Tensor:
        occupancy = input_data["occupancy"]  # B,T,C,H,W = ([2, 5, 6, 192, 96])
        batch, seq, c, h, w = occupancy.size()

        x = occupancy.view(-1, c, h, w)  # batch*seq, c, h, w ([10, 6, 192, 96])
        x = F.relu(self.bn_pre_1(self.conv_pre_1(x))) # ([batch*seq, 32, 192, 96]) ([10, 32, 192, 96])
        x = F.relu(self.bn_pre_2(self.conv_pre_2(x))) # ([batch*seq, 32, 192, 96])

        ########### feature embedding ##############
        features_list = input_data['feature_warpped_list']
        feature = torch.cat(features_list, dim=0)

        batch, seq, c2, h, w = feature.size()

        xf = feature.view(-1, c2, h, w)  # batch*seq, c, h, w ([10, 128, 192, 96])
        xf = F.relu(self.bn_pre_1_f(self.conv_pre_1_f(xf))) # ([batch*seq, 32, 192, 96]) ([10, 32, 192, 96])
        xf = F.relu(self.bn_pre_2_f(self.conv_pre_2_f(xf))) # ([batch*seq, 32, 192, 96])

        # concatenate feature and embedd again
        x_enhanced = torch.cat((x,xf), dim=1)  # batch*seq, c, h, w ([10, 64, 192, 96])
        x_enhanced = F.relu(self.bn_pre_1_f2(self.conv_pre_1_f2(x_enhanced))) # ([batch*seq, 64, 192, 96])
        x_enhanced = F.relu(self.bn_pre_2_f2(self.conv_pre_2_f2(x_enhanced))) # ([batch*seq, 64, 192, 96])

        # -- STC block 1
        # x_1 = F.relu(self.bn1_1(self.conv1_1(x_enhanced))) # ([10, 64, 192, 96])
        # x_1 = F.relu(self.bn1_2(self.conv1_2(x_1))) # ([10, 64, 192, 96])
        x_1 = x_enhanced
        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()  # (batch, seq, c, h, w)  ([2, 5, 64, 192, 96])
        x_1 = self.conv3d_1(x_1)  # ([2, 3, 64, 192, 96])
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()  # (batch * seq, c, h, w) ([6, 64, 192, 96])

        # -- STC block 2
        x_2 = F.relu(self.bn2_1(self.conv2_1(x_1))) # ([6, 128, 192, 96])
        x_2 = F.relu(self.bn2_2(self.conv2_2(x_2))) # ([6, 128, 192, 96])
        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)).contiguous()  # (batch, seq, c, h, w) ([2, 3, 128, 96, 48])
        x_2 = self.conv3d_2(x_2)  #  ([2, 1, 128, 96, 48])
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()  # (batch * seq, c, h, w), seq = 1 ([2, 128, 96, 48])

        x_3 = F.relu(self.bn3_1(self.conv3_1(x_2)))  #  ([2, 128, 192, 96])
        x_3 = F.relu(self.bn3_2(self.conv3_2(x_3)))  #  ([2, 128, 192, 96])
        x_3 = rearrange(x_3, 'b c h w -> (h w) b c') #  ([192*96, 2, 128])
        
        direction_ind = torch.argmax(input_data['cmd_direction'], dim=1)
        dir_embed = self.direction_embedding.weight[direction_ind] # ([2, 128])
        speed_ind = torch.argmax(input_data['cmd_speed'], dim=1)
        speed_embed = self.speed_embedding.weight[speed_ind]   # ([2, 128])
        feature_target = self.target_encoder(input_data['target'])  # ([2, 128])

        cmd_feature = torch.cat([dir_embed, speed_embed, feature_target], dim=1)
        cmd_feature = self.mlp_fuser(cmd_feature).unsqueeze(0)   # ([1, 2, 128])

        plan_query = self.attn_module(cmd_feature, x_3)  # ([2, 128, 1])

        future_waypoints = self.reg_branch(plan_query.squeeze(-1)).contiguous().view(batch, 10, 2)  #  ([2, 10, 2])
        future_waypoints[...,:2] = torch.cumsum(future_waypoints[...,:2], dim=1)

        output_data = dict(future_waypoints=future_waypoints)

        return output_data
    
@CODRIVING_REGISTRY.register
class WaypointPlanner_e2e_cmd_attn_wo_feature(nn.Module):
    """
    WaypointPlanner with BEV feature, navigation commands, occupancy map and road map as inputs
    """
    def __init__(self,feature_dir=128):
        super().__init__()
        ####### occupancy
        height_feat_size = 6
        self.conv_pre_1 = nn.Conv2d(height_feat_size, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(32)
        self.bn_pre_2 = nn.BatchNorm2d(32)

        # ####### bev feature
        # self.conv_pre_1_f = nn.Conv2d(feature_dir, 32, kernel_size=3, stride=1, padding=1)
        # self.conv_pre_2_f = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        # self.bn_pre_1_f = nn.BatchNorm2d(32)
        # self.bn_pre_2_f = nn.BatchNorm2d(32)

        ####### feature
        # self.conv_pre_1_f2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        # self.conv_pre_2_f2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        # self.bn_pre_1_f2 = nn.BatchNorm2d(32)
        # self.bn_pre_2_f2 = nn.BatchNorm2d(32)

        ####### STC
        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.conv3d_1 = Conv3D(64, 64, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.conv3d_2 = Conv3D(128, 128, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.bn3_2 = nn.BatchNorm2d(256)

        #######
        embed_dims = 256
        self.direction_embedding = nn.Embedding(6, embed_dims)
        self.speed_embedding = nn.Embedding(4, embed_dims)
        self.target_encoder = MLP(2, embed_dims, hid_feat=(16, 64))

        self.mlp_fuser = nn.Sequential(
                nn.Linear(embed_dims*3, embed_dims),
                nn.LayerNorm(embed_dims),
                nn.ReLU(inplace=True),
            )

        attn_module_layer = nn.TransformerDecoderLayer(embed_dims, 8, dim_feedforward=embed_dims*2, dropout=0.1, batch_first=False)
        self.attn_module = nn.TransformerDecoder(attn_module_layer, 3)
        
        self.reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 10*2),
        )

    def reset_parameters(self):
        pass

    def forward(self, input_data : Dict) -> torch.Tensor:
        """Forward method for WaypointPlanner

        Args:
            input_data: input data to forward

                required keys:

                - occupancy: rasterized map from perception results
                - target: target point to go to

        Return:
            torch.Tensor: predicted waypoints
        """
        occupancy = input_data["occupancy"]  # B,T,C,H,W = ([2, 5, 6, 192, 96])
        batch, seq, c, h, w = occupancy.size()

        x = occupancy.view(-1, c, h, w)  # batch*seq, c, h, w ([10, 6, 192, 96])
        x = F.relu(self.bn_pre_1(self.conv_pre_1(x))) # ([batch*seq, 32, 192, 96]) ([10, 32, 192, 96])
        x = F.relu(self.bn_pre_2(self.conv_pre_2(x))) # ([batch*seq, 32, 192, 96])

        # ########### feature embedding ##############
        # features_list = input_data['feature_warpped_list']
        # feature = torch.cat(features_list, dim=0)

        # # batch, seq, c2, h, w = feature.size()

        # xf = feature.view(-1, c2, h, w)  # batch*seq, c, h, w ([10, 128, 192, 96])
        # xf = F.relu(self.bn_pre_1_f(self.conv_pre_1_f(xf))) # ([batch*seq, 32, 192, 96]) ([10, 32, 192, 96])
        # xf = F.relu(self.bn_pre_2_f(self.conv_pre_2_f(xf))) # ([batch*seq, 32, 192, 96])

        # concatenate feature and embedd again
        # x_enhanced = torch.cat((x,xf), dim=1)  # batch*seq, c, h, w ([10, 64, 192, 96])
        # x_enhanced = F.relu(self.bn_pre_1_f2(self.conv_pre_1_f2(x_enhanced))) # ([batch*seq, 32, 192, 96]) ([10, 32, 192, 96])
        # x_enhanced = F.relu(self.bn_pre_2_f2(self.conv_pre_2_f2(x_enhanced))) # ([batch*seq, 32, 192, 96])

        # -- STC block 1
        x_1 = F.relu(self.bn1_1(self.conv1_1(x))) # ([10, 64, 192, 96])
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1))) # ([10, 64, 192, 96])
        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()  # (batch, seq, c, h, w)  ([2, 5, 64, 192, 96])
        x_1 = self.conv3d_1(x_1)  # ([2, 3, 64, 192, 96])
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()  # (batch * seq, c, h, w) ([6, 64, 192, 96])

        # -- STC block 2
        x_2 = F.relu(self.bn2_1(self.conv2_1(x_1))) # ([6, 128, 96, 48])
        x_2 = F.relu(self.bn2_2(self.conv2_2(x_2))) # ([6, 128, 96, 48])
        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)).contiguous()  # (batch, seq, c, h, w) ([2, 3, 128, 96, 48])
        x_2 = self.conv3d_2(x_2)  #  ([2, 1, 128, 96, 48])
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()  # (batch * seq, c, h, w), seq = 1 ([2, 128, 96, 48])

        x_3 = F.relu(self.bn3_1(self.conv3_1(x_2)))  #  ([2, 256, 96, 48])
        x_3 = F.relu(self.bn3_2(self.conv3_2(x_3)))  #  ([2, 256, 96, 48])
        x_3 = rearrange(x_3, 'b c h w -> (h w) b c') #  ([96*48, 2, 256])
        
        direction_ind = torch.argmax(input_data['cmd_direction'], dim=1)
        dir_embed = self.direction_embedding.weight[direction_ind] # ([2, 256])
        speed_ind = torch.argmax(input_data['cmd_speed'], dim=1)
        speed_embed = self.speed_embedding.weight[speed_ind]   # ([2, 256])
        feature_target = self.target_encoder(input_data['target'])  # ([2, 256])

        cmd_feature = torch.cat([dir_embed, speed_embed, feature_target], dim=1)
        cmd_feature = self.mlp_fuser(cmd_feature).unsqueeze(0)   # ([1, 2, 256])

        plan_query = self.attn_module(cmd_feature, x_3)  # ([2, 256, 1])

        future_waypoints = self.reg_branch(plan_query.squeeze(-1)).contiguous().view(batch, 10, 2)  #  ([2, 10, 2])
        future_waypoints[...,:2] = torch.cumsum(future_waypoints[...,:2], dim=1)

        output_data = dict(future_waypoints=future_waypoints)

        return output_data

# class ResNetBlock(nn.Module):
#     def __init__(self, dim):
#         super(ResNetBlock, self).__init__()
#         self.block = nn.Sequential(
#             nn.Linear(dim, dim),
#             nn.ReLU(),
#             nn.Linear(dim, dim)
#         )

#     def forward(self, x):
#         return x + self.block(x)

# bash scripts/train_planner_e2e.sh 3 1 /GPFS/data/gjliu-1/Auto-driving/OpenCOODv2/opencood/logs/closed_loop/where2comm covlm_cmd_add_model None log ./ckpt/add_model_1016
@CODRIVING_REGISTRY.register
class WaypointPlannerExtend(nn.Module):
    def __init__(self, ckpt_path, feature_dir=128, gru_hidden_size=256):
        super(WaypointPlannerExtend, self).__init__()      

        
        self.waypoint_planner = WaypointPlanner_e2e(feature_dir=feature_dir)
        checkpoint = torch.load(ckpt_path)
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(checkpoint['model_state_dict'], 'module.')        
        self.waypoint_planner.load_state_dict(checkpoint['model_state_dict'], strict=True)
        
        for param in self.waypoint_planner.parameters():
            param.requires_grad = False
            
        self.gru = nn.GRU(input_size=2, hidden_size=gru_hidden_size, num_layers=3, batch_first=True)
        embed_dims = 256
        self.direction_embedding = nn.Embedding(6, embed_dims)
        self.speed_embedding = nn.Embedding(4, embed_dims)
        self.target_encoder = MLP(2, embed_dims, hid_feat=(16, 64))

        self.mlp_fuser = nn.Sequential(
            nn.Linear(embed_dims * 3, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True)
        )

        
        conv_layers = []
        in_channels = 1

        for i in range(3):
            out_channels = 16 * (i + 1)
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
            conv_layers.append(nn.BatchNorm1d(out_channels))
            conv_layers.append(nn.ReLU())
            in_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)

        self.flatten = nn.Flatten()
        
        self.mlp_decoder = nn.Sequential(
            nn.Linear(48*512, 128),
            nn.ReLU(),
            nn.Linear(128, 10 * 2)
        )
        

    def forward(self, input_data):
        waypoint_output = self.waypoint_planner(input_data)
        # import ipdb; ipdb.set_trace()
        waypoints = waypoint_output['future_waypoints']
        
        gru_output, _ = self.gru(waypoints)

        cmd_feature = self.get_cmd_feature(input_data)

        gru_output_last = gru_output[:, -1, :] 

        combined_features = torch.cat([cmd_feature, gru_output_last], dim=-1)  # (b, 2*256)
        combined_features = combined_features.unsqueeze(1)  # (b, 1, 2*256)
        conv_features = self.flatten(self.conv(combined_features))  # (b, 48*512)

        finetuned_waypoints = self.mlp_decoder(conv_features).view(-1, 10, 2)

        output_data = dict(future_waypoints=finetuned_waypoints)
        return output_data

    def get_cmd_feature(self, input_data):
        # import ipdb; ipdb.set_trace()
        direction_ind = torch.argmax(input_data['cmd_direction'], dim=1)
        dir_embed = self.direction_embedding.weight[direction_ind] # ([b, 256])
        speed_ind = torch.argmax(input_data['cmd_speed'], dim=1)
        speed_embed = self.speed_embedding.weight[speed_ind]   # ([b, 256])
        feature_target = self.target_encoder(input_data['target'])  # ([b, 256])

        cmd_feature = torch.cat([dir_embed, speed_embed, feature_target], dim=1)
        cmd_feature = self.mlp_fuser(cmd_feature)   # ([b, 256])
        return cmd_feature

@CODRIVING_REGISTRY.register
class WaypointPlanner_e2e_intention_target(nn.Module):
    """
    WaypointPlanner with BEV feature, navigation commands, occupancy map and road map as inputs
    """
    def __init__(self,feature_dir=128):
        super().__init__()

        #######
        embed_dims = 256
        self.direction_embedding = nn.Embedding(6, embed_dims)
        self.speed_embedding = nn.Embedding(4, embed_dims)
        self.target_encoder = MLP(2, embed_dims, hid_feat=(16, 64))

        self.mlp_fuser = nn.Sequential(
                nn.Linear(embed_dims*3, embed_dims),
                nn.LayerNorm(embed_dims),
                nn.ReLU(inplace=True),
            )
        
        self.mlp_encoder = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
        )
        
        self.reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 10*2),
        )

    def reset_parameters(self):
        pass

    def forward(self, input_data : Dict) -> torch.Tensor:
        """Forward method for WaypointPlanner

        Args:
            input_data: input data to forward

                required keys:

                - occupancy: rasterized map from perception results
                - target: target point to go to

        Return:
            torch.Tensor: predicted waypoints
        """
        # import ipdb; ipdb.set_trace()
        occupancy = input_data["occupancy"]  # B,T,C,H,W = ([2, 5, 6, 192, 96])
        batch = occupancy.size()[0]
        
        direction_ind = torch.argmax(input_data['cmd_direction'], dim=1)
        dir_embed = self.direction_embedding.weight[direction_ind] # ([b, 256])
        speed_ind = torch.argmax(input_data['cmd_speed'], dim=1)
        speed_embed = self.speed_embedding.weight[speed_ind]   # ([b, 256])
        feature_target = self.target_encoder(input_data['target'])  # ([b, 256])

        cmd_feature = torch.cat([dir_embed, speed_embed, feature_target], dim=1)
        cmd_feature = self.mlp_fuser(cmd_feature)   # ([b, 256])

        deep_feature = self.mlp_encoder(cmd_feature) # ([b, 256])
        future_waypoints = self.reg_branch(deep_feature).view(batch, 10, 2)  #  ([2, 10, 2])
        future_waypoints[...,:2] = torch.cumsum(future_waypoints[...,:2], dim=1)

        output_data = dict(future_waypoints=future_waypoints)

        return output_data
    


@CODRIVING_REGISTRY.register
class WaypointPlanner_e2e_cmd_attn_wo_occupancy(nn.Module):
    """
    WaypointPlanner with BEV feature, navigation commands, occupancy map and road map as inputs
    """
    def __init__(self,feature_dir=128):
        super().__init__()
        # ####### occupancy
        # height_feat_size = 6
        # self.conv_pre_1 = nn.Conv2d(height_feat_size, 32, kernel_size=3, stride=1, padding=1)
        # self.conv_pre_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        # self.bn_pre_1 = nn.BatchNorm2d(32)
        # self.bn_pre_2 = nn.BatchNorm2d(32)

        ####### bev feature
        self.conv_pre_1_f = nn.Conv2d(feature_dir, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2_f = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1_f = nn.BatchNorm2d(32)
        self.bn_pre_2_f = nn.BatchNorm2d(32)

        # ####### feature
        # self.conv_pre_1_f2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        # self.conv_pre_2_f2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        # self.bn_pre_1_f2 = nn.BatchNorm2d(32)
        # self.bn_pre_2_f2 = nn.BatchNorm2d(32)

        ####### STC
        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.conv3d_1 = Conv3D(64, 64, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.conv3d_2 = Conv3D(128, 128, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.bn3_2 = nn.BatchNorm2d(256)

        #######
        embed_dims = 256
        self.direction_embedding = nn.Embedding(6, embed_dims)
        self.speed_embedding = nn.Embedding(4, embed_dims)
        self.target_encoder = MLP(2, embed_dims, hid_feat=(16, 64))

        self.mlp_fuser = nn.Sequential(
                nn.Linear(embed_dims*3, embed_dims),
                nn.LayerNorm(embed_dims),
                nn.ReLU(inplace=True),
            )

        attn_module_layer = nn.TransformerDecoderLayer(embed_dims, 8, dim_feedforward=embed_dims*2, dropout=0.1, batch_first=False)
        self.attn_module = nn.TransformerDecoder(attn_module_layer, 3)
        
        self.reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 10*2),
        )

    def reset_parameters(self):
        pass

    def forward(self, input_data : Dict) -> torch.Tensor:
        """Forward method for WaypointPlanner

        Args:
            input_data: input data to forward

                required keys:

                - occupancy: rasterized map from perception results
                - target: target point to go to

        Return:
            torch.Tensor: predicted waypoints
        """
        # occupancy = input_data["occupancy"]  # B,T,C,H,W = ([2, 5, 6, 192, 96])
        # batch, seq, c, h, w = occupancy.size()

        # x = occupancy.view(-1, c, h, w)  # batch*seq, c, h, w ([10, 6, 192, 96])
        # x = F.relu(self.bn_pre_1(self.conv_pre_1(x))) # ([batch*seq, 32, 192, 96]) ([10, 32, 192, 96])
        # x = F.relu(self.bn_pre_2(self.conv_pre_2(x))) # ([batch*seq, 32, 192, 96])

        ########### feature embedding ##############
        features_list = input_data['feature_warpped_list']
        feature = torch.cat(features_list, dim=0)

        batch, seq, c2, h, w = feature.size()

        xf = feature.view(-1, c2, h, w)  # batch*seq, c, h, w ([10, 128, 192, 96])
        xf = F.relu(self.bn_pre_1_f(self.conv_pre_1_f(xf))) # ([batch*seq, 32, 192, 96]) ([10, 32, 192, 96])
        xf = F.relu(self.bn_pre_2_f(self.conv_pre_2_f(xf))) # ([batch*seq, 32, 192, 96])

        # concatenate feature and embedd again
        # x_enhanced = torch.cat((x,xf), dim=1)  # batch*seq, c, h, w ([10, 64, 192, 96])
        # x_enhanced = F.relu(self.bn_pre_1_f2(self.conv_pre_1_f2(x_enhanced))) # ([batch*seq, 32, 192, 96]) ([10, 32, 192, 96])
        # x_enhanced = F.relu(self.bn_pre_2_f2(self.conv_pre_2_f2(x_enhanced))) # ([batch*seq, 32, 192, 96])

        # -- STC block 1
        x_1 = F.relu(self.bn1_1(self.conv1_1(xf))) # ([10, 64, 192, 96])
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1))) # ([10, 64, 192, 96])
        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()  # (batch, seq, c, h, w)  ([2, 5, 64, 192, 96])
        x_1 = self.conv3d_1(x_1)  # ([2, 3, 64, 192, 96])
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()  # (batch * seq, c, h, w) ([6, 64, 192, 96])

        # -- STC block 2
        x_2 = F.relu(self.bn2_1(self.conv2_1(x_1))) # ([6, 128, 96, 48])
        x_2 = F.relu(self.bn2_2(self.conv2_2(x_2))) # ([6, 128, 96, 48])
        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)).contiguous()  # (batch, seq, c, h, w) ([2, 3, 128, 96, 48])
        x_2 = self.conv3d_2(x_2)  #  ([2, 1, 128, 96, 48])
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()  # (batch * seq, c, h, w), seq = 1 ([2, 128, 96, 48])

        x_3 = F.relu(self.bn3_1(self.conv3_1(x_2)))  #  ([2, 256, 96, 48])
        x_3 = F.relu(self.bn3_2(self.conv3_2(x_3)))  #  ([2, 256, 96, 48])
        x_3 = rearrange(x_3, 'b c h w -> (h w) b c') #  ([96*48, 2, 256])
        
        direction_ind = torch.argmax(input_data['cmd_direction'], dim=1)
        dir_embed = self.direction_embedding.weight[direction_ind] # ([2, 256])
        speed_ind = torch.argmax(input_data['cmd_speed'], dim=1)
        speed_embed = self.speed_embedding.weight[speed_ind]   # ([2, 256])
        feature_target = self.target_encoder(input_data['target'])  # ([2, 256])

        cmd_feature = torch.cat([dir_embed, speed_embed, feature_target], dim=1)
        cmd_feature = self.mlp_fuser(cmd_feature).unsqueeze(0)   # ([1, 2, 256])

        plan_query = self.attn_module(cmd_feature, x_3)  # ([2, 256, 1])

        future_waypoints = self.reg_branch(plan_query.squeeze(-1)).contiguous().view(batch, 10, 2)  #  ([2, 10, 2])
        future_waypoints[...,:2] = torch.cumsum(future_waypoints[...,:2], dim=1)

        output_data = dict(future_waypoints=future_waypoints)

        return output_data


    
@CODRIVING_REGISTRY.register
class WaypointPlanner_e2e_cmd_attn_fix(nn.Module):
    """
    WaypointPlanner with BEV feature, navigation commands, occupancy map and road map as inputs
    """
    def __init__(self,feature_dir=128):
        super().__init__()
        ####### occupancy
        height_feat_size = 6
        self.conv_pre_1 = nn.Conv2d(height_feat_size, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(32)
        self.bn_pre_2 = nn.BatchNorm2d(32)

        ####### bev feature
        self.conv_pre_1_f = nn.Conv2d(feature_dir, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2_f = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1_f = nn.BatchNorm2d(32)
        self.bn_pre_2_f = nn.BatchNorm2d(32)

        ####### feature
        self.conv_pre_1_f2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2_f2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1_f2 = nn.BatchNorm2d(32)
        self.bn_pre_2_f2 = nn.BatchNorm2d(32)

        ####### STC
        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.conv3d_1 = Conv3D(64, 64, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.conv3d_2 = Conv3D(128, 128, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.bn3_2 = nn.BatchNorm2d(256)

        #######
        embed_dims = 256
        self.direction_embedding = nn.Embedding(6, embed_dims)
        self.speed_embedding = nn.Embedding(4, embed_dims)
        self.target_encoder = MLP(2, embed_dims, hid_feat=(16, 64))
        self.quary_embedding = nn.Embedding(1, embed_dims)

        self.mlp_fuser = nn.Sequential(
                nn.Linear(embed_dims*3, embed_dims),
                nn.LayerNorm(embed_dims),
                nn.ReLU(inplace=True),
            )
        self.attn_modules = nn.ModuleList([
            nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    embed_dims, 8, dim_feedforward=embed_dims * 2, dropout=0.1, batch_first=False
                ), 
                1
            ) for _ in range(6)
        ])
        
        self.reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 10*2),
        )

    def reset_parameters(self):
        pass

    def forward(self, input_data : Dict) -> torch.Tensor:
        """Forward method for WaypointPlanner

        Args:
            input_data: input data to forward

                required keys:

                - occupancy: rasterized map from perception results
                - target: target point to go to

        Return:
            torch.Tensor: predicted waypoints
        """
        
        occupancy = input_data["occupancy"]  # B,T,C,H,W = ([2, 5, 6, 192, 96])
        batch, seq, c, h, w = occupancy.size()

        x = occupancy.view(-1, c, h, w)  # batch*seq, c, h, w ([10, 6, 192, 96])
        x = F.relu(self.bn_pre_1(self.conv_pre_1(x))) # ([batch*seq, 32, 192, 96]) ([10, 32, 192, 96])
        x = F.relu(self.bn_pre_2(self.conv_pre_2(x))) # ([batch*seq, 32, 192, 96])

        ########### feature embedding ##############
        features_list = input_data['feature_warpped_list']
        feature = torch.cat(features_list, dim=0)

        batch, seq, c2, h, w = feature.size()

        xf = feature.view(-1, c2, h, w)  # batch*seq, c, h, w ([10, 128, 192, 96])
        xf = F.relu(self.bn_pre_1_f(self.conv_pre_1_f(xf))) # ([batch*seq, 32, 192, 96]) ([10, 32, 192, 96])
        xf = F.relu(self.bn_pre_2_f(self.conv_pre_2_f(xf))) # ([batch*seq, 32, 192, 96])

        # concatenate feature and embedd again
        x_enhanced = torch.cat((x,xf), dim=1)  # batch*seq, c, h, w ([10, 64, 192, 96])
        x_enhanced = F.relu(self.bn_pre_1_f2(self.conv_pre_1_f2(x_enhanced))) # ([batch*seq, 32, 192, 96]) ([10, 32, 192, 96])
        x_enhanced = F.relu(self.bn_pre_2_f2(self.conv_pre_2_f2(x_enhanced))) # ([batch*seq, 32, 192, 96])

        # -- STC block 1
        x_1 = F.relu(self.bn1_1(self.conv1_1(x_enhanced))) # ([10, 64, 192, 96])
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1))) # ([10, 64, 192, 96])
        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()  # (batch, seq, c, h, w)  ([2, 5, 64, 192, 96])
        x_1 = self.conv3d_1(x_1)  # ([2, 3, 64, 192, 96])
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()  # (batch * seq, c, h, w) ([6, 64, 192, 96])

        # -- STC block 2
        x_2 = F.relu(self.bn2_1(self.conv2_1(x_1))) # ([6, 128, 96, 48])
        x_2 = F.relu(self.bn2_2(self.conv2_2(x_2))) # ([6, 128, 96, 48])
        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)).contiguous()  # (batch, seq, c, h, w) ([2, 3, 128, 96, 48])
        x_2 = self.conv3d_2(x_2)  #  ([2, 1, 128, 96, 48])
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()  # (batch * seq, c, h, w), seq = 1 ([2, 128, 96, 48])

        x_3 = F.relu(self.bn3_1(self.conv3_1(x_2)))  #  ([2, 256, 96, 48])
        x_3 = F.relu(self.bn3_2(self.conv3_2(x_3)))  #  ([2, 256, 96, 48])
        x_3 = rearrange(x_3, 'b c h w -> (h w) b c') #  ([96*48, 2, 256])
        
        direction_ind = torch.argmax(input_data['cmd_direction'], dim=1)
        dir_embed = self.direction_embedding.weight[direction_ind] # ([2, 256])
        speed_ind = torch.argmax(input_data['cmd_speed'], dim=1)
        speed_embed = self.speed_embedding.weight[speed_ind]   # ([2, 256])
        feature_target = self.target_encoder(input_data['target'])  # ([2, 256])

        cmd_feature = torch.cat([dir_embed, speed_embed, feature_target], dim=1)
        cmd_feature = self.mlp_fuser(cmd_feature).unsqueeze(0)   # ([1, 2, 256])
        plan_query = self.quary_embedding.weight.unsqueeze(0).expand(-1,batch,-1)  # ([1, 2, 256])

        # plan_query = self.attn_module(cmd_feature, x_3)  # ([2, 256, 1])
        # import ipdb; ipdb.set_trace()
        # print(plan_query.device, cmd_feature.device, x_3.device)
        for i in range(len(self.attn_modules)):
            if i % 2 == 0:
                plan_query = self.attn_modules[i](plan_query, cmd_feature)
            else:
                plan_query = self.attn_modules[i](plan_query, x_3)
        plan_query = plan_query.squeeze(0)
        future_waypoints = self.reg_branch(plan_query).contiguous().view(batch, 10, 2)  #  ([2, 10, 2])
        future_waypoints[...,:2] = torch.cumsum(future_waypoints[...,:2], dim=1)

        output_data = dict(future_waypoints=future_waypoints)

        return output_data
    

@CODRIVING_REGISTRY.register
class WaypointPlanner_e2e_cmd_attn_fix_20points(nn.Module):
    """
    WaypointPlanner with BEV feature, navigation commands, occupancy map and road map as inputs
    """
    def __init__(self,feature_dir=128):
        super().__init__()
        ####### occupancy
        height_feat_size = 6
        self.conv_pre_1 = nn.Conv2d(height_feat_size, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(32)
        self.bn_pre_2 = nn.BatchNorm2d(32)

        ####### bev feature
        self.conv_pre_1_f = nn.Conv2d(feature_dir, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2_f = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1_f = nn.BatchNorm2d(32)
        self.bn_pre_2_f = nn.BatchNorm2d(32)

        ####### feature
        self.conv_pre_1_f2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2_f2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1_f2 = nn.BatchNorm2d(32)
        self.bn_pre_2_f2 = nn.BatchNorm2d(32)

        ####### STC
        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.conv3d_1 = Conv3D(64, 64, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.conv3d_2 = Conv3D(128, 128, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.bn3_2 = nn.BatchNorm2d(256)

        #######
        embed_dims = 256
        self.direction_embedding = nn.Embedding(6, embed_dims)
        self.speed_embedding = nn.Embedding(4, embed_dims)
        self.target_encoder = MLP(2, embed_dims, hid_feat=(16, 64))
        self.quary_embedding = nn.Embedding(1, embed_dims)

        self.mlp_fuser = nn.Sequential(
                nn.Linear(embed_dims*3, embed_dims),
                nn.LayerNorm(embed_dims),
                nn.ReLU(inplace=True),
            )
        self.attn_modules = nn.ModuleList([
            nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    embed_dims, 8, dim_feedforward=embed_dims * 2, dropout=0.1, batch_first=False
                ), 
                1
            ) for _ in range(6)
        ])
        
        self.reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 20*2),
        )

    def reset_parameters(self):
        pass

    def forward(self, input_data : Dict) -> torch.Tensor:
        """Forward method for WaypointPlanner

        Args:
            input_data: input data to forward

                required keys:

                - occupancy: rasterized map from perception results
                - target: target point to go to

        Return:
            torch.Tensor: predicted waypoints
        """
        
        occupancy = input_data["occupancy"]  # B,T,C,H,W = ([2, 5, 6, 192, 96])
        batch, seq, c, h, w = occupancy.size()

        x = occupancy.view(-1, c, h, w)  # batch*seq, c, h, w ([10, 6, 192, 96])
        x = F.relu(self.bn_pre_1(self.conv_pre_1(x))) # ([batch*seq, 32, 192, 96]) ([10, 32, 192, 96])
        x = F.relu(self.bn_pre_2(self.conv_pre_2(x))) # ([batch*seq, 32, 192, 96])

        ########### feature embedding ##############
        features_list = input_data['feature_warpped_list']
        feature = torch.cat(features_list, dim=0)

        batch, seq, c2, h, w = feature.size()

        xf = feature.view(-1, c2, h, w)  # batch*seq, c, h, w ([10, 128, 192, 96])
        xf = F.relu(self.bn_pre_1_f(self.conv_pre_1_f(xf))) # ([batch*seq, 32, 192, 96]) ([10, 32, 192, 96])
        xf = F.relu(self.bn_pre_2_f(self.conv_pre_2_f(xf))) # ([batch*seq, 32, 192, 96])

        # concatenate feature and embedd again
        x_enhanced = torch.cat((x,xf), dim=1)  # batch*seq, c, h, w ([10, 64, 192, 96])
        x_enhanced = F.relu(self.bn_pre_1_f2(self.conv_pre_1_f2(x_enhanced))) # ([batch*seq, 32, 192, 96]) ([10, 32, 192, 96])
        x_enhanced = F.relu(self.bn_pre_2_f2(self.conv_pre_2_f2(x_enhanced))) # ([batch*seq, 32, 192, 96])

        # -- STC block 1
        x_1 = F.relu(self.bn1_1(self.conv1_1(x_enhanced))) # ([10, 64, 192, 96])
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1))) # ([10, 64, 192, 96])
        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()  # (batch, seq, c, h, w)  ([2, 5, 64, 192, 96])
        x_1 = self.conv3d_1(x_1)  # ([2, 3, 64, 192, 96])
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()  # (batch * seq, c, h, w) ([6, 64, 192, 96])

        # -- STC block 2
        x_2 = F.relu(self.bn2_1(self.conv2_1(x_1))) # ([6, 128, 96, 48])
        x_2 = F.relu(self.bn2_2(self.conv2_2(x_2))) # ([6, 128, 96, 48])
        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)).contiguous()  # (batch, seq, c, h, w) ([2, 3, 128, 96, 48])
        x_2 = self.conv3d_2(x_2)  #  ([2, 1, 128, 96, 48])
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()  # (batch * seq, c, h, w), seq = 1 ([2, 128, 96, 48])

        x_3 = F.relu(self.bn3_1(self.conv3_1(x_2)))  #  ([2, 256, 96, 48])
        x_3 = F.relu(self.bn3_2(self.conv3_2(x_3)))  #  ([2, 256, 96, 48])
        x_3 = rearrange(x_3, 'b c h w -> (h w) b c') #  ([96*48, 2, 256])
        
        direction_ind = torch.argmax(input_data['cmd_direction'], dim=1)
        dir_embed = self.direction_embedding.weight[direction_ind] # ([2, 256])
        speed_ind = torch.argmax(input_data['cmd_speed'], dim=1)
        speed_embed = self.speed_embedding.weight[speed_ind]   # ([2, 256])
        feature_target = self.target_encoder(input_data['target'])  # ([2, 256])

        cmd_feature = torch.cat([dir_embed, speed_embed, feature_target], dim=1)
        cmd_feature = self.mlp_fuser(cmd_feature).unsqueeze(0)   # ([1, 2, 256])
        plan_query = self.quary_embedding.weight.unsqueeze(0).expand(-1,batch,-1)  # ([1, 2, 256])

        # plan_query = self.attn_module(cmd_feature, x_3)  # ([2, 256, 1])
        # import ipdb; ipdb.set_trace()
        # print(plan_query.device, cmd_feature.device, x_3.device)
        for i in range(len(self.attn_modules)):
            if i % 2 == 0:
                plan_query = self.attn_modules[i](plan_query, cmd_feature)
            else:
                plan_query = self.attn_modules[i](plan_query, x_3)
        plan_query = plan_query.squeeze(0)
        future_waypoints = self.reg_branch(plan_query).contiguous().view(batch, 20, 2)  #  ([2, 20, 2])
        future_waypoints[...,:2] = torch.cumsum(future_waypoints[...,:2], dim=1)

        output_data = dict(future_waypoints=future_waypoints)

        return output_data

@CODRIVING_REGISTRY.register
class WaypointPlanner_e2e_cmd_attn_fix_20points_3target(nn.Module):
    """
    WaypointPlanner with BEV feature, navigation commands, occupancy map and road map as inputs
    """
    def __init__(self,feature_dir=128):
        super().__init__()
        ####### occupancy
        height_feat_size = 6
        self.conv_pre_1 = nn.Conv2d(height_feat_size, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(32)
        self.bn_pre_2 = nn.BatchNorm2d(32)

        ####### bev feature
        self.conv_pre_1_f = nn.Conv2d(feature_dir, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2_f = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1_f = nn.BatchNorm2d(32)
        self.bn_pre_2_f = nn.BatchNorm2d(32)

        ####### feature
        self.conv_pre_1_f2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2_f2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1_f2 = nn.BatchNorm2d(32)
        self.bn_pre_2_f2 = nn.BatchNorm2d(32)

        ####### STC
        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.conv3d_1 = Conv3D(64, 64, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.conv3d_2 = Conv3D(128, 128, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.bn3_2 = nn.BatchNorm2d(256)

        #######
        embed_dims = 256
        self.direction_embedding = nn.Embedding(6, embed_dims)
        self.speed_embedding = nn.Embedding(4, embed_dims)
        self.target_encoder = MLP(2*3, embed_dims, hid_feat=(16, 64))
        self.quary_embedding = nn.Embedding(1, embed_dims)

        self.mlp_fuser = nn.Sequential(
                nn.Linear(embed_dims*3, embed_dims),
                nn.LayerNorm(embed_dims),
                nn.ReLU(inplace=True),
            )
        self.attn_modules = nn.ModuleList([
            nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    embed_dims, 8, dim_feedforward=embed_dims * 2, dropout=0.1, batch_first=False
                ), 
                1
            ) for _ in range(6)
        ])
        
        self.reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 20*2),
        )

    def reset_parameters(self):
        pass

    def forward(self, input_data : Dict) -> torch.Tensor:
        """Forward method for WaypointPlanner

        Args:
            input_data: input data to forward

                required keys:

                - occupancy: rasterized map from perception results
                - target: target point to go to

        Return:
            torch.Tensor: predicted waypoints
        """
        
        occupancy = input_data["occupancy"]  # B,T,C,H,W = ([2, 5, 6, 192, 96])
        batch, seq, c, h, w = occupancy.size()

        x = occupancy.view(-1, c, h, w)  # batch*seq, c, h, w ([10, 6, 192, 96])
        x = F.relu(self.bn_pre_1(self.conv_pre_1(x))) # ([batch*seq, 32, 192, 96]) ([10, 32, 192, 96])
        x = F.relu(self.bn_pre_2(self.conv_pre_2(x))) # ([batch*seq, 32, 192, 96])

        ########### feature embedding ##############
        features_list = input_data['feature_warpped_list']
        feature = torch.cat(features_list, dim=0)

        batch, seq, c2, h, w = feature.size()

        xf = feature.view(-1, c2, h, w)  # batch*seq, c, h, w ([10, 128, 192, 96])
        xf = F.relu(self.bn_pre_1_f(self.conv_pre_1_f(xf))) # ([batch*seq, 32, 192, 96]) ([10, 32, 192, 96])
        xf = F.relu(self.bn_pre_2_f(self.conv_pre_2_f(xf))) # ([batch*seq, 32, 192, 96])

        # concatenate feature and embedd again
        x_enhanced = torch.cat((x,xf), dim=1)  # batch*seq, c, h, w ([10, 64, 192, 96])
        x_enhanced = F.relu(self.bn_pre_1_f2(self.conv_pre_1_f2(x_enhanced))) # ([batch*seq, 32, 192, 96]) ([10, 32, 192, 96])
        x_enhanced = F.relu(self.bn_pre_2_f2(self.conv_pre_2_f2(x_enhanced))) # ([batch*seq, 32, 192, 96])

        # -- STC block 1
        x_1 = F.relu(self.bn1_1(self.conv1_1(x_enhanced))) # ([10, 64, 192, 96])
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1))) # ([10, 64, 192, 96])
        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()  # (batch, seq, c, h, w)  ([2, 5, 64, 192, 96])
        x_1 = self.conv3d_1(x_1)  # ([2, 3, 64, 192, 96])
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()  # (batch * seq, c, h, w) ([6, 64, 192, 96])

        # -- STC block 2
        x_2 = F.relu(self.bn2_1(self.conv2_1(x_1))) # ([6, 128, 96, 48])
        x_2 = F.relu(self.bn2_2(self.conv2_2(x_2))) # ([6, 128, 96, 48])
        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)).contiguous()  # (batch, seq, c, h, w) ([2, 3, 128, 96, 48])
        x_2 = self.conv3d_2(x_2)  #  ([2, 1, 128, 96, 48])
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()  # (batch * seq, c, h, w), seq = 1 ([2, 128, 96, 48])

        x_3 = F.relu(self.bn3_1(self.conv3_1(x_2)))  #  ([2, 256, 96, 48])
        x_3 = F.relu(self.bn3_2(self.conv3_2(x_3)))  #  ([2, 256, 96, 48])
        x_3 = rearrange(x_3, 'b c h w -> (h w) b c') #  ([96*48, 2, 256])
        
        direction_ind = torch.argmax(input_data['cmd_direction'], dim=1)
        dir_embed = self.direction_embedding.weight[direction_ind] # ([2, 256])
        speed_ind = torch.argmax(input_data['cmd_speed'], dim=1)
        speed_embed = self.speed_embedding.weight[speed_ind]   # ([2, 256])
        feature_target = self.target_encoder(input_data['target'])  # ([2, 256])

        cmd_feature = torch.cat([dir_embed, speed_embed, feature_target], dim=1)
        cmd_feature = self.mlp_fuser(cmd_feature).unsqueeze(0)   # ([1, 2, 256])
        plan_query = self.quary_embedding.weight.unsqueeze(0).expand(-1,batch,-1)  # ([1, 2, 256])

        # plan_query = self.attn_module(cmd_feature, x_3)  # ([2, 256, 1])
        # import ipdb; ipdb.set_trace()
        # print(plan_query.device, cmd_feature.device, x_3.device)
        for i in range(len(self.attn_modules)):
            if i % 2 == 0:
                plan_query = self.attn_modules[i](plan_query, cmd_feature)
            else:
                plan_query = self.attn_modules[i](plan_query, x_3)
        plan_query = plan_query.squeeze(0)
        future_waypoints = self.reg_branch(plan_query).contiguous().view(batch, 20, 2)  #  ([2, 20, 2])
        future_waypoints[...,:2] = torch.cumsum(future_waypoints[...,:2], dim=1)

        output_data = dict(future_waypoints=future_waypoints)

        return output_data