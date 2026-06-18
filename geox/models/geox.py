import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from geox.models.geox_aggregator import GeoXAggregator
from geox.heads.camera_head import CameraHead
from geox.heads.dpt_head import DPTHead
import pdb


class GeoX(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14,
                embed_dim=384,
                # embed_dim=768,
                cam_drop_prob=0.1,
                enable_camera=True, enable_depth=True, enable_point=True,
                enable_mask_adapter=False,
                mask_adapter_ratio=0.25):
        super().__init__()

        self.aggregator = GeoXAggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
                                         pose_hidden_dim=9, cam_drop_prob=cam_drop_prob,
                                         enable_mask_adapter=enable_mask_adapter,
                                         mask_adapter_ratio=mask_adapter_ratio)
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        # self.camera_token_proj = nn.Linear(2 * embed_dim, 2048) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1", intermediate_layer_idx=[5, 7, 9, 11]) if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1", intermediate_layer_idx=[5, 7, 9, 11]) if enable_depth else None
    
    def inference(self,
        images: torch.Tensor,
        extrinsics: torch.Tensor = None,
        intrinsics: torch.Tensor = None,
        camera_gt_index: list = None,
        ):

        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator.inference(
            images=images,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            camera_gt_index=camera_gt_index,
        )
        
        B, S, C_in, H, W = images.shape
        predictions = {}

        with torch.amp.autocast('cuda', enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf


        predictions["images"] = images  # store the images for visualization during inference

        return predictions
