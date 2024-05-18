# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmrotate.models.builder import ROTATED_DETECTORS
from mmrotate.models.detectors.two_stage import RotatedTwoStageDetector
import torch.optim as optim
from .cga import TestMixins

@ROTATED_DETECTORS.register_module()
class OrientedRCNN_CGA(RotatedTwoStageDetector, TestMixins):
    """Implementation of `Oriented R-CNN for Object Detection.`__

    __ https://openaccess.thecvf.com/content/ICCV2021/papers/Xie_Oriented_R-CNN_for_Object_Detection_ICCV_2021_paper.pdf  # noqa: E501, E261.
    """

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(OrientedRCNN_CGA, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        
    def simple_test(self, img, img_metas, with_cga = False, proposals = None, rescale = False): 
        bbox_results = super().simple_test(img, img_metas, proposals, rescale)
        if with_cga:
            return self.refine_test(bbox_results, img_metas)
        else:
            return bbox_results

