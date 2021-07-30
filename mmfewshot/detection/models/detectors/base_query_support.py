import copy
from abc import abstractmethod

from mmcv.runner import auto_fp16
from mmdet.models.builder import (DETECTORS, build_backbone, build_head,
                                  build_neck)
from mmdet.models.detectors import BaseDetector


@DETECTORS.register_module()
class BaseQuerySupportDetector(BaseDetector):
    """Base class for two-stage detectors in query-support fashion.

    Query-support detectors typically consisting of a region
    proposal network and a task-specific regression head. There are
    two pipelines for query and support data respectively.

    Args:
        backbone (dict): Config of the backbone for query data.
        neck (dict | None): Config of the neck for query data and
            probably for support data. Default: None.
        support_backbone (dict | None): Config of the backbone for
            support data only. If None, support and query data will
            share same backbone. Default: None.
        support_neck (dict | None): Config of the neck for support
            data only. Default: None.
        rpn_head (dict | None): Config of rpn_head. Default: None.
        roi_head (dict | None): Config of roi_head. Default: None.
        train_cfg (dict | None): Training config. Useless in CenterNet,
            but we keep this variable for SingleStageDetector. Default: None.
        test_cfg (dict | None): Testing config of CenterNet. Default: None.
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 support_backbone=None,
                 support_neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(BaseQuerySupportDetector, self).__init__(init_cfg)
        backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck) if neck is not None else None
        # if `support_backbone` is None, then support and query pipeline will
        # share same backbone.
        self.support_backbone = build_backbone(support_backbone) \
            if support_backbone is not None else self.backbone
        # support neck only forward support data.
        self.support_neck = build_neck(support_neck) \
            if support_neck is not None else None
        assert roi_head is not None, 'missing config of roi_head'
        # when rpn with aggregation neck, the input of rpn will consist of
        # query and support data. otherwise the input of rpn only
        # has query data.
        self.with_rpn = False
        self.rpn_with_support = False
        if rpn_head is not None:
            self.with_rpn = True
            if rpn_head.get('aggregation_layer', None) is not None:
                self.rpn_with_support = True
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = copy.deepcopy(rpn_head)
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_query_feat(self, img):
        """Extract features of query data.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            list[Tensor]: Features of support images, each item with shape
                 (N, C, H, W).
        """
        feats = self.backbone(img)
        if self.with_neck:
            feats = self.neck(feats)
        return feats

    def extract_feat(self, img):
        """Extract features of query data.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            list[Tensor]: Features of query images.
        """
        return self.extract_query_feat(img)

    @abstractmethod
    def extract_support_feat(self, img):
        """Extract features of support data."""
        raise NotImplementedError

    @auto_fp16(apply_to=('img', ))
    def forward(self,
                query_data=None,
                support_data=None,
                img=None,
                img_metas=None,
                mode='train',
                **kwargs):
        """Calls one of (:func:`forward_train`, :func:`forward_test` and
        :func:`forward_model_init`) according to the `mode`. The inputs
        of forward function would change with the `mode`.

        - When `mode` is 'train', the input will be query and support data
        for training.

        - When `mode` is 'model_init', the input will be support template
        data at least including (img, img_metas).

        - When `mode` is 'test', the input will be test data at least
        including (img, img_metas).

        Args:
            query_data (dict): Used for :func:`forward_train`. Dict of
                query data and data info where each dict has: `img`,
                `img_metas`, `gt_bboxes`, `gt_labels`, `gt_bboxes_ignore`.
                Default: None.
            support_data (dict): Used for :func:`forward_train`. Dict of
                support data and data info dict where each dict has: `img`,
                `img_metas`, `gt_bboxes`, `gt_labels`, `gt_bboxes_ignore`.
                Default: None.
            img (list[Tensor]): Used for func:`forward_test` or
                :func:`forward_model_init`. List of tensors of shape
                (1, C, H, W). Typically these should be mean centered
                and std scaled. Default: None.
            img_metas (list[dict]): Used for func:`forward_test` or
                :func:`forward_model_init`.  List of image info dict
                where each dict has: `img_shape`, `scale_factor`, `flip`,
                and may also contain `filename`, `ori_shape`, `pad_shape`,
                and `img_norm_cfg`. For details on the values of these keys,
                see :class:`mmdet.datasets.pipelines.Collect`. Default: None.
            mode (str): Indicate which function to call. Options are 'train',
                'model_init' and 'test'. Default: 'train'.
        """
        if mode == 'train':
            return self.forward_train(query_data, support_data, **kwargs)
        elif mode == 'model_init':
            return self.forward_model_init(img, img_metas, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, img_metas, **kwargs)
        else:
            raise ValueError(
                f'invalid forward mode {mode}, '
                f'only support `train`, `model_init` and `test` now')

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN. For most of query-support detectors, the
        batch size denote the batch size of query data.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        # For most of query-support detectors, the batch size denote the
        # batch size of query data.
        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data['query_data']['img_metas']))

        return outputs

    def val_step(self, data, optimizer=None):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        # For most of query-support detectors, the batch size denote the
        # batch size of query data.
        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data['query_data']['img_metas']))

        return outputs

    def forward_train(self,
                      query_data,
                      support_data,
                      proposals=None,
                      **kwargs):
        """
        Args:
            query_data (dict): In most cases, dict of query data contains:
                `img`, `img_metas`, `gt_bboxes`, `gt_labels`,
                `gt_bboxes_ignore`.
            support_data (dict):  In most cases, dict of support data contains:
                `img`, `img_metas`, `gt_bboxes`, `gt_labels`,
                `gt_bboxes_ignore`.
            proposals (list): Override rpn proposals with custom proposals.
                Use when `with_rpn` is False. Default: None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        query_img = query_data['img']
        support_img = support_data['img']
        query_feats = self.extract_query_feat(query_img)
        support_feats = self.extract_support_feat(support_img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            if self.rpn_with_support:
                rpn_losses, proposal_list = self.rpn_head.forward_train(
                    query_feats,
                    support_feats,
                    query_img_metas=query_data['img_metas'],
                    query_gt_bboxes=query_data['gt_bboxes'],
                    query_gt_labels=None,
                    query_gt_bboxes_ignore=query_data.get(
                        'gt_bboxes_ignore', None),
                    support_img_metas=support_data['img_metas'],
                    support_gt_bboxes=support_data['gt_bboxes'],
                    support_gt_labels=support_data['gt_labels'],
                    support_gt_bboxes_ignore=support_data.get(
                        'gt_bboxes_ignore', None),
                    proposal_cfg=proposal_cfg)
            else:
                rpn_losses, proposal_list = self.rpn_head.forward_train(
                    query_feats,
                    copy.deepcopy(query_data['img_metas']),
                    copy.deepcopy(query_data['gt_bboxes']),
                    gt_labels=None,
                    gt_bboxes_ignore=copy.deepcopy(
                        query_data.get('gt_bboxes_ignore', None)),
                    proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(
            query_feats,
            support_feats,
            proposals=proposal_list,
            query_img_metas=query_data['img_metas'],
            query_gt_bboxes=query_data['gt_bboxes'],
            query_gt_labels=query_data['gt_labels'],
            query_gt_bboxes_ignore=query_data.get('gt_bboxes_ignore', None),
            support_img_metas=support_data['img_metas'],
            support_gt_bboxes=support_data['gt_bboxes'],
            support_gt_labels=support_data['gt_labels'],
            support_gt_bboxes_ignore=support_data.get('gt_bboxes_ignore',
                                                      None),
            **kwargs)
        losses.update(roi_losses)

        return losses

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        raise NotImplementedError

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        raise NotImplementedError

    async def async_simple_test(self, **kwargs):
        """Async test without augmentation."""
        raise NotImplementedError

    def aug_test(self, **kwargs):
        """Test with augmentation."""
        raise NotImplementedError

    @abstractmethod
    def forward_model_init(self,
                           img,
                           img_metas,
                           gt_bboxes=None,
                           gt_labels=None,
                           **kwargs):
        """extract and save support features for model initialization."""
        raise NotImplementedError

    @abstractmethod
    def model_init(self, **kwargs):
        """process the saved support features for model initialization."""
        raise NotImplementedError
