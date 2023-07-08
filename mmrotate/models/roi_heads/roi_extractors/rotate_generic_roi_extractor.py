"""
/*
 * @Author: nihao
 * @Email: nihao@baidu.com
 * @Date: 2023-05-17 19:25:24
 * @Last Modified by: nihao
 * @Last Modified time: 2023-05-17 19:40:59
 * @Description: Description
 */
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch import Tensor

from mmcv import ops
from mmcv.runner import force_fp32
from mmcv.utils import to_2tuple
from mmcv.cnn.bricks import build_plugin_layer
from mmdet.models.roi_heads.roi_extractors.base_roi_extractor import \
    BaseRoIExtractor

from ...builder import ROTATED_ROI_EXTRACTORS


@ROTATED_ROI_EXTRACTORS.register_module()
class RotatedGenericRoIExtractor(BaseRoIExtractor):
    """
    """
    def __init__(self,
                 aggregation,
                 pre_cfg,
                 post_cfg,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        assert aggregation in ['sum', 'concat']

        self.aggregation = aggregation
        self.with_post = post_cfg is not None
        self.with_pre = pre_cfg is not None
        # build pre/post processing modules
        if self.with_post:
            self.post_module = build_plugin_layer(post_cfg, '_post_module')[1]
        if self.with_pre:
            self.pre_module = build_plugin_layer(pre_cfg, '_pre_module')[1]

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self,
                feats: Tuple[Tensor],
                rois: Tensor,
                roi_scale_factor: Optional[float] = None) -> Tensor:
        """Extractor ROI feats.

        Args:
            feats (Tuple[Tensor]): Multi-scale features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.
            roi_scale_factor (Optional[float]): RoI scale factor.
                Defaults to None.

        Returns:
            Tensor: RoI feature.
        """
        from mmrotate import digit_version, mmcv_version
        if isinstance(self.roi_layers[0], ops.RiRoIAlignRotated
                      ) or mmcv_version == digit_version('1.4.5'):
            out_size = nn.modules.utils._pair(self.roi_layers[0].out_size)
        else:
            out_size = self.roi_layers[0].output_size
            
        num_levels = len(feats)
        roi_feats = feats[0].new_zeros(
                rois.size(0), self.out_channels, *out_size)

        if roi_feats.shape[0] == 0:
            return roi_feats

        if num_levels == 1:
            return self.roi_layers[0](feats[0], rois)

        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)

        # mark the starting channels for concat mode
        start_channels = 0
        for i in range(num_levels):
            roi_feats_t = self.roi_layers[i](feats[i], rois)
            end_channels = start_channels + roi_feats_t.size(1)
            if self.with_pre:
                # apply pre-processing to a RoI extracted from each layer
                roi_feats_t = self.pre_module(roi_feats_t)
            if self.aggregation == 'sum':
                # and sum them all
                roi_feats += roi_feats_t
            else:
                # and concat them along channel dimension
                roi_feats[:, start_channels:end_channels] = roi_feats_t
            # update channels starting position
            start_channels = end_channels
        # check if concat channels match at the end
        if self.aggregation == 'concat':
            assert start_channels == self.out_channels
        
        if self.with_post:
            # apply post-processing before return the result
            roi_feats = self.post_module(roi_feats)
        return roi_feats
        

    def roi_rescale(self, rois, scale_factor):
        """Scale RoI coordinates by scale factor.

        Args:
            rois (torch.Tensor): RoI (Region of Interest), shape (n, 6)
            scale_factor (float): Scale factor that RoI will be multiplied by.

        Returns:
            torch.Tensor: Scaled RoI.
        """
        if scale_factor is None:
            return rois
        h_scale_factor, w_scale_factor = to_2tuple(scale_factor)
        new_rois = rois.clone()
        new_rois[:, 3] = w_scale_factor * new_rois[:, 3]
        new_rois[:, 4] = h_scale_factor * new_rois[:, 4]
        return new_rois
