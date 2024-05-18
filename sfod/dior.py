# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import xml.etree.ElementTree as ET
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv import print_log
from mmdet.datasets import CustomDataset
from PIL import Image

from mmrotate.core import eval_rbbox_map, poly2obb_np
from mmrotate.datasets.builder import ROTATED_DATASETS


@ROTATED_DATASETS.register_module()
class DIORDataset(CustomDataset):
    """DIOR dataset for detection.

    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        img_subdir (str): Subdir where images are stored.
            Defaults to ``JPEGImages-trainval``.
        ann_subdir (str): Subdir where annotations are.
            Defaults to ``Annotations/Oriented Bounding Boxes/``.
        version (str, optional): Angle representations. Defaults to ``oc``.
        xmltype : Choose obb or hbb as ground truth. Defaults to ``obb``.
    """

    CLASSES = ('airplane', 'airport', 'baseballfield', 'basketballcourt',
               'bridge', 'chimney', 'expressway-service-area',
               'expressway-toll-station', 'dam', 'golffield',
               'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
               'storagetank', 'tenniscourt', 'trainstation', 'vehicle',
               'windmill')

    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
               (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
               (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
               (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157)]

    def __init__(self,
                 ann_file,
                 pipeline,
                 img_subdir='JPEGImages-trainval',
                 ann_subdir='Annotations',
                 version='oc',
                 xmltype='obb',
                 **kwargs):
        assert xmltype in ['hbb', 'obb']
        self.xmltype = xmltype
        self.img_subdir = img_subdir
        self.ann_subdir = ann_subdir
        self.version = version
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        super(DIORDataset, self).__init__(ann_file, pipeline, **kwargs)

    def load_annotations(self, ann_file):
        """Load annotation from XML style ann_file.

        Args:
            ann_file (str): Path of Imageset file.

        Returns:
            list[dict]: Annotation info from XML file.
        """

        data_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        for img_id in img_ids:
            data_info = {}

            filename = osp.join(self.img_subdir, f'{img_id}.jpg')
            data_info['filename'] = f'{img_id}.jpg'
#             xml_path = osp.join(self.img_prefix, self.ann_subdir,
#                                 f'{img_id}.xml')
            xml_path = osp.join(self.ann_subdir,
                                f'{img_id}.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()

            size = root.find('size')
            if size is not None:
                width = int(size.find('width').text)
                height = int(size.find('height').text)

            if width is None or height is None:
                img_path = osp.join(self.img_prefix, filename)
                img = Image.open(img_path)
                width, height = img.size
            data_info['width'] = width
            data_info['height'] = height
            data_info['ann'] = {}
            gt_bboxes = []
            gt_labels = []
            gt_polygons = []
            gt_bboxes_ignore = []
            gt_labels_ignore = []
            gt_polygons_ignore = []

            for obj in root.findall('object'):
                cls = obj.find('name').text.lower()
                label = self.cat2label[cls]
                if label is None:
                    continue

                if self.xmltype == 'obb':
                    bnd_box = obj.find('robndbox')
                    polygon = np.array([
                        float(bnd_box.find('x_left_top').text),
                        float(bnd_box.find('y_left_top').text),
                        float(bnd_box.find('x_right_top').text),
                        float(bnd_box.find('y_right_top').text),
                        float(bnd_box.find('x_right_bottom').text),
                        float(bnd_box.find('y_right_bottom').text),
                        float(bnd_box.find('x_left_bottom').text),
                        float(bnd_box.find('y_left_bottom').text),
                    ]).astype(np.float32)
                else:
                    bnd_box = obj.find('bndbox')
                    if bnd_box is None:
                        continue
                    polygon = np.array([
                        float(bnd_box.find('xmin').text),
                        float(bnd_box.find('ymin').text),
                        float(bnd_box.find('xmax').text),
                        float(bnd_box.find('ymin').text),
                        float(bnd_box.find('xmax').text),
                        float(bnd_box.find('ymax').text),
                        float(bnd_box.find('xmin').text),
                        float(bnd_box.find('ymax').text)
                    ]).astype(np.float32)

                bbox = poly2obb_np(polygon, self.version)
                if bbox is not None:
                    gt_bboxes.append(np.array(bbox, dtype=np.float32))
                    gt_labels.append(label)
                    gt_polygons.append(polygon)

            if gt_bboxes:
                data_info['ann']['bboxes'] = np.array(
                    gt_bboxes, dtype=np.float32)
                data_info['ann']['labels'] = np.array(
                    gt_labels, dtype=np.int64)
                data_info['ann']['polygons'] = np.array(
                    gt_polygons, dtype=np.float32)
            else:
                data_info['ann']['bboxes'] = np.zeros((0, 5), dtype=np.float32)
                data_info['ann']['labels'] = np.array([], dtype=np.int64)
                data_info['ann']['polygons'] = np.zeros((0, 8),
                                                        dtype=np.float32)

            if gt_polygons_ignore:
                data_info['ann']['bboxes_ignore'] = np.array(
                    gt_bboxes_ignore, dtype=np.float32)
                data_info['ann']['labels_ignore'] = np.array(
                    gt_labels_ignore, dtype=np.int64)
                data_info['ann']['polygons_ignore'] = np.array(
                    gt_polygons_ignore, dtype=np.float32)
            else:
                data_info['ann']['bboxes_ignore'] = np.zeros((0, 5),
                                                             dtype=np.float32)
                data_info['ann']['labels_ignore'] = np.array([],
                                                             dtype=np.int64)
                data_info['ann']['polygons_ignore'] = np.zeros(
                    (0, 8), dtype=np.float32)

            data_infos.append(data_info)
        return data_infos

    def _filter_imgs(self):
        """Filter images without ground truths."""
        valid_inds = []
        for i, data_info in enumerate(self.data_infos):
            if data_info['ann']['labels'].size > 0:
                valid_inds.append(i)
        return valid_inds

    def evaluate(
            self,
            results,
            metric='mAP',
            logger=None,
            proposal_nums=(100, 300, 1000),
            # iou_thr=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
            iou_thr=[0.5],
            scale_ranges=None,
            use_07_metric=True,
            nproc=4):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
            use_07_metric (bool): Whether to use the voc07 metric.
            nproc (int): Processes used for computing TP and FP.
                Default: 4.
        """
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')

        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, _ = eval_rbbox_map(
                    results,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    use_07_metric=use_07_metric,
                    dataset=self.CLASSES,
                    logger=logger,
                    nproc=nproc)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            eval_results.move_to_end('mAP', last=False)
        elif metric == 'recall':
            raise NotImplementedError

        return eval_results
                
@ROTATED_DATASETS.register_module()
class DIORDatasetWithIdx(DIORDataset):
    def __init__(
        self,
        with_idx=True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.with_idx=with_idx
        return
    
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        if self.with_idx:
            if type(data) == dict:
                data['idx'] = idx
            else:
                for i in data:
                    i['idx'] = idx
        return data


@ROTATED_DATASETS.register_module()
class DIORDatasetSubset(DIORDatasetWithIdx):
    def __init__(
        self,
        indices,
        with_idx=True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.indices = indices
        self.with_idx = with_idx
        self.img_ids = np.array(mmcv.list_from_file(self.ann_file))[self.indices]
        return
    
    def __getitem__(self, idx):
        data = super().__getitem__(self.indices[idx])
        if self.with_idx is False:
            if type(data) == dict:
                if 'idx' in data:
                    del data['idx']
            else:
                for i in data:
                    if 'idx' in data:
                        del i['idx']
        return data

    def __len__(self):
        return len(self.indices)

    def get_ann_info_imgid(self, img_id):
        """Get annotation from XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        xml_path = osp.join(self.img_prefix, 'Annotations', f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()

        gt_bboxes = []
        gt_labels = []
        gt_polygons = []
        gt_bboxes_ignore = []
        gt_labels_ignore = []
        gt_polygons_ignore = []

        for obj in root.findall('object'):
            cls = obj.find('name').text.lower()
            label = self.cat2label[cls]
            if label is None:
                continue

            if self.xmltype == 'obb':
                bnd_box = obj.find('robndbox')
                polygon = np.array([
                    float(bnd_box.find('x_left_top').text),
                    float(bnd_box.find('y_left_top').text),
                    float(bnd_box.find('x_right_top').text),
                    float(bnd_box.find('y_right_top').text),
                    float(bnd_box.find('x_right_bottom').text),
                    float(bnd_box.find('y_right_bottom').text),
                    float(bnd_box.find('x_left_bottom').text),
                    float(bnd_box.find('y_left_bottom').text),
                ]).astype(np.float32)
            else:
                bnd_box = obj.find('bndbox')
                if bnd_box is None:
                    continue
                polygon = np.array([
                    float(bnd_box.find('xmin').text),
                    float(bnd_box.find('ymin').text),
                    float(bnd_box.find('xmax').text),
                    float(bnd_box.find('ymin').text),
                    float(bnd_box.find('xmax').text),
                    float(bnd_box.find('ymax').text),
                    float(bnd_box.find('xmin').text),
                    float(bnd_box.find('ymax').text)
                ]).astype(np.float32)

            bbox = poly2obb_np(polygon, self.version)
            if bbox is not None:
                gt_bboxes.append(np.array(bbox, dtype=np.float32))
                gt_labels.append(label)
                gt_polygons.append(polygon)

        if gt_bboxes:
            data_info['ann']['bboxes'] = np.array(
                gt_bboxes, dtype=np.float32)
            data_info['ann']['labels'] = np.array(
                gt_labels, dtype=np.int64)
            data_info['ann']['polygons'] = np.array(
                gt_polygons, dtype=np.float32)
        else:
            data_info['ann']['bboxes'] = np.zeros((0, 5), dtype=np.float32)
            data_info['ann']['labels'] = np.array([], dtype=np.int64)
            data_info['ann']['polygons'] = np.zeros((0, 8),
                                                    dtype=np.float32)

        if gt_polygons_ignore:
            data_info['ann']['bboxes_ignore'] = np.array(
                gt_bboxes_ignore, dtype=np.float32)
            data_info['ann']['labels_ignore'] = np.array(
                gt_labels_ignore, dtype=np.int64)
            data_info['ann']['polygons_ignore'] = np.array(
                gt_polygons_ignore, dtype=np.float32)
        else:
            data_info['ann']['bboxes_ignore'] = np.zeros((0, 5),
                                                         dtype=np.float32)
            data_info['ann']['labels_ignore'] = np.array([],
                                                         dtype=np.int64)
            data_info['ann']['polygons_ignore'] = np.zeros(
                (0, 8), dtype=np.float32)


        return data_info

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate in VOC protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'mAP', 'recall'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple], optional): Scale ranges for evaluating
                mAP. If not specified, all bounding boxes would be included in
                evaluation. Default: None.

        Returns:
            dict[str, float]: AP/recall metrics.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')

        annotations = [self.get_ann_info_imgid(i) for i in self.img_ids]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            if self.year == 2007:
                ds_name = 'voc07'
            else:
                ds_name = self.CLASSES
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                # Follow the official implementation,
                # http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar
                # we should use the legacy coordinate system in mmdet 1.x,
                # which means w, h should be computed as 'x2 - x1 + 1` and
                # `y2 - y1 + 1`
                mean_ap, eval_results_cat = eval_map(
                    results,
                    annotations,
                    scale_ranges=None,
                    iou_thr=iou_thr,
                    dataset=ds_name,
                    logger=logger,
                    use_legacy_coordinate=True)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)

                categorical_info = {'AP': [], 'Count': []}
                # import ipdb;ipdb.set_trace()
                for eval_result in eval_results_cat:
                    if eval_result['ap'] == 'nan':
                        categorical_info['AP'].append(0.)
                    else:
                        categorical_info['AP'].append(eval_result['ap'])
                    categorical_info['Count'].append(eval_result['num_gts'])
                eval_results['categorical_info'] = categorical_info

            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(
                gt_bboxes,
                results,
                proposal_nums,
                iou_thrs,
                logger=logger,
                use_legacy_coordinate=True)
            for i, num in enumerate(proposal_nums):
                for j, iou_thr in enumerate(iou_thrs):
                    eval_results[f'recall@{num}@{iou_thr}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]

        return eval_results