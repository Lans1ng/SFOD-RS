# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection
"""
semi-supervised two stage detector
"""
import torch

from mmrotate.core import rbbox2result
from mmrotate.models.detectors import RotatedTwoStageDetector
from .rotated_semi_base import SemiBaseDetector


class SemiTwoStageDetector(SemiBaseDetector, RotatedTwoStageDetector):
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 # ema model
                 ema_config=None,
                 ema_ckpt=None,
                 classes=None
                 ):
        SemiBaseDetector.__init__(self, ema_config=ema_config, ema_ckpt=ema_ckpt, classes=classes)
        RotatedTwoStageDetector.__init__(self, backbone=backbone, rpn_head=rpn_head, roi_head=roi_head,
                                  train_cfg=train_cfg, test_cfg=test_cfg, neck=neck, pretrained=pretrained)

    @torch.no_grad()
    def inference_unlabeled(self, img, img_metas, rescale=True, return_feat=False):
        ema_model = self.ema_model.module
        bbox_results = ema_model.simple_test(img, img_metas, with_cga = True, rescale = rescale)
        # except:
        #     bbox_results = ema_model.simple_test(img, img_metas, rescale = rescale)
        return bbox_results
            
    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)
