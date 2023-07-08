# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta

import torch
from mmcv.runner import BaseModule
from mmdet.core import bbox2roi, multiclass_nms
from mmrotate.core.bbox import bbox_mapping, bbox_mapping_back

from mmrotate.core import build_assigner, build_sampler, obb2xyxy, rbbox2result, rbbox2roi,multiclass_nms_rotated, aug_multiclass_nms_rotated
from ..builder import (ROTATED_HEADS, build_head, build_roi_extractor,
                       build_shared_head)


def merge_aug_bboxes(aug_bboxes, aug_scores, img_metas, rcnn_test_cfg):
    """Merge augmented detection bboxes and scores.

    Args:
        aug_bboxes (list[Tensor]): shape (n, 4*#class)
        aug_scores (list[Tensor] or None): shape (n, #class)
        img_shapes (list[Tensor]): shape (3, ).
        rcnn_test_cfg (dict): rcnn test config.

    Returns:
        tuple: (bboxes, scores)
    """
    recovered_bboxes = []
    for bboxes, img_info in zip(aug_bboxes, img_metas):
        img_shape = img_info[0]['img_shape']
        scale_factor = img_info[0]['scale_factor']
        flip = img_info[0]['flip']
        flip_direction = img_info[0]['flip_direction']
        bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip,
                                   flip_direction)
        recovered_bboxes.append(bboxes)
    bboxes = torch.stack(recovered_bboxes).mean(dim=0)
    if aug_scores is None:
        return bboxes
    else:
        scores = torch.stack(aug_scores).mean(dim=0)
        return bboxes, scores

@ROTATED_HEADS.register_module()
class RotatedStandardRoIHead(BaseModule, metaclass=ABCMeta):
    """Simplest base rotated roi head including one bbox head.

    Args:
        bbox_roi_extractor (dict, optional): Config of ``bbox_roi_extractor``.
        bbox_head (dict, optional): Config of ``bbox_head``.
        shared_head (dict, optional): Config of ``shared_head``.
        train_cfg (dict, optional): Config of train.
        test_cfg (dict, optional): Config of test.
        pretrained (str, optional): Path of pretrained weight.
        init_cfg (dict, optional): Config of initialization.
        version (str, optional): Angle representations. Defaults to 'oc'.
    """

    def __init__(self,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 version='oc'):

        super(RotatedStandardRoIHead, self).__init__(init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.version = version

        if shared_head is not None:
            shared_head.pretrained = pretrained
            self.shared_head = build_shared_head(shared_head)

        if bbox_head is not None:
            self.init_bbox_head(bbox_roi_extractor, bbox_head)

        self.init_assigner_sampler()

        self.with_bbox = True if bbox_head is not None else False
        self.with_shared_head = True if shared_head is not None else False

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``.

        Args:
            bbox_roi_extractor (dict): Config of ``bbox_roi_extractor``.
            bbox_head (dict): Config of ``bbox_head``.
        """
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def forward_dummy(self, x, proposals):
        """Dummy forward function.

        Args:
            x (list[Tensors]): list of multi-level img features.
            proposals (list[Tensors]): list of region proposals.

        Returns:
            list[Tensors]: list of region of interest.
        """
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        return outs

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task. Always
                set to None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """
        # assign gts and sample proposals
        if self.with_bbox:

            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                gt_hbboxes = obb2xyxy(gt_bboxes[i], self.version)
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_hbboxes, gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_hbboxes,
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])

                if gt_bboxes[i].numel() == 0:
                    sampling_result.pos_gt_bboxes = gt_bboxes[i].new(
                        (0, gt_bboxes[0].size(-1))).zero_()
                else:
                    sampling_result.pos_gt_bboxes = \
                        gt_bboxes[i][sampling_result.pos_assigned_gt_inds, :]

                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        return losses

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing.

        Args:
            x (list[Tensor]): list of multi-level img features.
            rois (list[Tensors]): list of region of interests.

        Returns:
            dict[str, Tensor]: a dictionary of bbox_results.
        """
        # import pdb; pdb.set_trace()
        # rois[0].shape = 6
        # x[:self.bbox_roi_extractor.num_inputs][0].shape = [1, 256, 192, 256]
        # 1: 1, 256, 96, 128
        # 2: 1, 256, 48, 64
        # 3: 1, 256, 24, 3
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training.

        Args:
            x (list[Tensor]): list of multi-level img features.
            sampling_results (list[Tensor]): list of sampling results.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.

        Returns:
            dict[str, Tensor]: a dictionary of bbox_results.
        """
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                rescale=False):
        """Async test without augmentation.

        Args:
            x (list[Tensor]): list of multi-level img features.
            proposal_list (list[Tensors]): list of region proposals.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            dict[str, Tensor]: a dictionary of bbox_results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = rbbox2result(det_bboxes, det_labels,
                                    self.bbox_head.num_classes)
        return bbox_results

    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation.

        Args:
            x (list[Tensor]): list of multi-level img features.
            proposal_list (list[Tensors]): list of region proposals.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            dict[str, Tensor]: a dictionary of bbox_results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        # img_metas len() == 1: 
        # [{'filename': '/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/eval/images/train00001.jpg', 'ori_filename': 'train00001.jpg', 'ori_shape': (700, 955, 3), 'img_shape': (751, 1024, 3), 'pad_shape': (768, 1024, 3), 'scale_factor': array([1.0723, 1.0729, 1.0723, 1.0729], dtype=float32), 'flip': False, 'flip_direction': None, 'img_norm_cfg': {'mean': array([123.675, 116.28 , 103.53 ], dtype=float32), 'std': array([58.395, 57.12 , 57.375], dtype=float32), 'to_rgb': True}, 'batch_input_shape': (768, 1024)}]
        # x: len() == 5, tuple
        # proposal_list: len() == 1, len(proposal_list[0]) == 2000

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        
        # print("img_metas: ", img_metas)
        # print("det_bboxes: ", det_bboxes)

        bbox_results = [
            rbbox2result(det_bboxes[i], det_labels[i],
                         self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        return bbox_results

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations."""

        # img_metas len() == 4
        # x: len() == 4, list

        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)
        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        
        bbox_results = [
            rbbox2result(_det_bboxes[i], det_labels[i],
                         self.bbox_head.num_classes)
            for i in range(len(_det_bboxes))
        ]

        return bbox_results
        # raise NotImplementedError

    def aug_test_bboxes(self, feats, img_metas, proposal_list, rcnn_test_cfg):
        """Test det bboxes with test time augmentation."""
        aug_bboxes = []
        aug_labels = []
        i = 0
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']
            # TODO more flexible
            # proposals = [bbox_mapping(proposal_list[0][:, :5], img_shape,
            #                          scale_factor, flip, flip_direction)]
            # import pdb; pdb.set_trace()
            proposals = [bbox_mapping(proposal_list[i][:], img_shape,
                                     scale_factor, flip, flip_direction)]
            i += 1
            rois = rbbox2roi(proposals)
            bbox_results = self._bbox_forward(x, rois)

            ###########
            img_shapes = tuple(meta['img_shape'] for meta in img_meta)
            scale_factors = tuple(meta['scale_factor'] for meta in img_meta)

            # print("scale_factors: ", scale_factors)
            # print("img_shapes: ", img_shapes)

            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple(len(p) for p in proposals)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)

            # some detector with_reg is False, bbox_pred will be None
            if bbox_pred is not None:
                # the bbox prediction of some detectors like SABL is not Tensor
                if isinstance(bbox_pred, torch.Tensor):
                    bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
                else:
                    bbox_pred = self.bbox_head.bbox_pred_split(
                        bbox_pred, num_proposals_per_img)
            else:
                bbox_pred = (None, ) * len(proposals)

            # apply bbox post-processing to each image individually
            for i in range(len(proposals)):
                det_bbox, det_label = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=True,
                    cfg=rcnn_test_cfg)
                
                # det_bbox = bbox_mapping_back(det_bbox, img_shape, scale_factor, flip,
                #                        flip_direction)

                aug_bboxes.append(det_bbox) # len == 1, (n, 6)
                aug_labels.append(det_label) # len == 1, (n, 1)
                # import pdb; pdb.set_trace()

            # aug_bboxes.append(det_bboxes_single_aug)
            # aug_labels.append(det_labels_single_aug)
            ############
        
        merged_bboxes = torch.cat(aug_bboxes, 0)
        merged_labels = torch.cat(aug_labels, 0)

        det_bboxes, det_labels = aug_multiclass_nms_rotated(merged_bboxes,
                                                    merged_labels,
                                                    rcnn_test_cfg.score_thr,
                                                    rcnn_test_cfg.nms,
                                                    rcnn_test_cfg.max_per_img,
                                                    self.bbox_head.num_classes)

        return [det_bboxes], [det_labels]

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains \
                the boxes of the corresponding image in a batch, each \
                tensor has the shape (num_boxes, 5) and last dimension \
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor \
                in the second list is the labels with shape (num_boxes, ). \
                The length of both lists should be equal to batch_size.
        """

        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            batch_size = len(proposals)
            det_bbox = rois.new_zeros(0, 5)
            det_label = rois.new_zeros((0, ), dtype=torch.long)
            if rcnn_test_cfg is None:
                det_bbox = det_bbox[:, :4]
                det_label = rois.new_zeros(
                    (0, self.bbox_head.fc_cls.out_features))
            # There is no proposal in the whole batch
            return [det_bbox] * batch_size, [det_label] * batch_size

        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                # There is no proposal in the single image
                det_bbox = rois[i].new_zeros(0, 5)
                det_label = rois[i].new_zeros((0, ), dtype=torch.long)
                if rcnn_test_cfg is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros(
                        (0, self.bbox_head.fc_cls.out_features))

            else:
                det_bbox, det_label = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels
