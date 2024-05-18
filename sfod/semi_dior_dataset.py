# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection
import copy
import random
from torch.utils.data import Dataset
from mmrotate.datasets.builder import ROTATED_DATASETS
from mmrotate.datasets.builder import ROTATED_PIPELINES
from .dior import DIORDataset

import collections
from mmcv.utils import build_from_cfg

class Compose:
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, ROTATED_PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """

        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            str_ = t.__repr__()
            if 'Compose(' in str_:
                str_ = str_.replace('\n', '\n    ')
            format_string += '\n'
            format_string += f'    {str_}'
        format_string += '\n)'
        return format_string

@ROTATED_DATASETS.register_module()
class SemiDIORDataset(Dataset):

    def __init__(self,
                 ann_file,
                 ann_file_u,
                 ann_subdir,
                 pipeline,
                 pipeline_u_share,
                 pipeline_u,
                 pipeline_u_1,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 data_root_u=None,
                 img_prefix_u='',
                 seg_prefix_u=None,
                 proposal_file_u=None,
                 classes=None,
                 filter_empty_gt=True,
                 ):
        super().__init__()
                     
        self.dior_labeled = DIORDataset(
            ann_file, pipeline, ann_subdir=ann_subdir, data_root=data_root, img_prefix=img_prefix,
            proposal_file=proposal_file, test_mode=False,
            filter_empty_gt=filter_empty_gt, classes=classes)
        self.dior_unlabeled = DIORDataset(
            ann_file_u, pipeline_u_share, ann_subdir=ann_subdir, data_root=data_root_u, img_prefix=img_prefix_u,
            proposal_file=proposal_file_u, test_mode=False,
            filter_empty_gt=False, classes=classes)
        self.CLASSES = classes
        self.pipeline_u = Compose(pipeline_u)
        self.pipeline_u_1 = Compose(pipeline_u_1) if pipeline_u_1 else None

        self.flag = self.dior_unlabeled.flag  # not used

    def __len__(self):
        return len(self.dior_labeled)

    def __getitem__(self, idx):
        idx_label = random.randint(0, len(self.dior_labeled) - 1)
        results = self.dior_labeled[idx_label]
        results_u = self.dior_unlabeled[idx]
        if self.pipeline_u_1:
            results_u_1 = copy.deepcopy(results_u)
            results_u_1 = self.pipeline_u_1(results_u_1)
            results.update({f'{key}_unlabeled_1': val for key, val in results_u_1.items()})
        results_u = self.pipeline_u(results_u)
        results.update({f'{key}_unlabeled': val for key, val in results_u.items()})
        return results
