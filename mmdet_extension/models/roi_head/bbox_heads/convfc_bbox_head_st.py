# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection
import torch

from mmdet.core import multi_apply
from mmdet.models.roi_heads.bbox_heads import Shared2FCBBoxHead
from mmdet.models.builder import HEADS


@HEADS.register_module()
class Shared2FCBBoxHeadST(Shared2FCBBoxHead):
    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x
        bbox_feats = x
#         print(feats.shape)
        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred, bbox_feats
        
    """
    pos: only do classification
    ig: only do regression
    """
    def _get_target_single_st(
            self, pos_bboxes, pos_gt_bboxes, pos_gt_labels,   # positive
            ig_bboxes, ig_gt_bboxes, ig_gt_labels,  # ignore
            neg_bboxes, cfg):
        num_pos = pos_bboxes.size(0)
        num_ig = ig_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg + num_ig

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples,),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 0  # not do box regression
        if num_ig > 0:
            # labels[num_pos:num_ig + num_pos] = ig_gt_labels
            label_weights[num_pos:num_ig + num_pos] = 1  # do classification as background
            if not self.reg_decoded_bbox:
                ig_bbox_targets = self.bbox_coder.encode(
                    ig_bboxes, ig_gt_bboxes)
            else:
                ig_bbox_targets = ig_gt_bboxes
            bbox_targets[num_pos:num_pos + num_ig, :] = ig_bbox_targets
            bbox_weights[num_pos:num_pos + num_ig, :] = 1

        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    def get_targets_st(self,
                       sampling_results,
                       gt_bboxes,
                       gt_labels,
                       rcnn_train_cfg,
                       concat=True
                       ):
        # positive
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        # ignore
        ig_bboxes_list = [res.ig_bboxes for res in sampling_results]
        ig_gt_bboxes_list = [res.ig_gt_bboxes for res in sampling_results]
        ig_gt_labels_list = [res.ig_gt_labels for res in sampling_results]
        # negative
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single_st,
            pos_bboxes_list, pos_gt_bboxes_list, pos_gt_labels_list,
            ig_bboxes_list, ig_gt_bboxes_list, ig_gt_labels_list,
            neg_bboxes_list, cfg=rcnn_train_cfg)
        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights
