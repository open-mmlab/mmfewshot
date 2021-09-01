import copy

from mmcls.models.builder import CLASSIFIERS, build_head

from .base import FewShotBaseClassifier


@CLASSIFIERS.register_module()
class FinetuneBaseClassifier(FewShotBaseClassifier):
    """Base class for fine-tune based classifier.

    Args:
        head (dict): Config of classification head for training.
        meta_test_head (dict): Config of classification head for meta testing.
            the meta_test_head only will be built and run in meta testing.
            Default: None.
    """

    def __init__(self, head, meta_test_head=None, *args, **kwargs):
        assert meta_test_head is not None
        super(FinetuneBaseClassifier, self).__init__(
            head=head, *args, **kwargs)
        # meta_test_head only will be built and used in meta testing
        self.meta_test_head_cfg = meta_test_head
        self.meta_test_head = None

    def forward(self, img=None, feats=None, mode='train', **kwargs):
        """Calls one of (:func:`forward_train`, :func:`forward_query`,
        :func:`forward_support` and :func:`extract_feat`) according to
        the `mode`. The inputs of forward function would change with the
        `mode`.

        - When `mode` is any of 'train', 'support' and 'query',
          the input will be either images or extracted features.
        - When `mode` is 'extract_feat', the input will be images.

        Args:
            img (Tensor): Used for func:`forward_query` or
                :func:`forward_support`. With shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
                Default: None.
            feats (Tensor): Used for func:`forward_query` or
                :func:`forward_support`. With shape (N, C, H, W) or (N, C).
                Default: None.
            mode (str): Indicate which function to call. Options are 'train',
                'support', 'query' and 'extract_feat'. Default: 'train'.
        """
        assert (img is not None) or (feats is not None)
        if mode == 'train':
            return self.forward_train(img=img, feats=feats, **kwargs)
        elif mode == 'query':
            return self.forward_query(img=img, feats=feats, **kwargs)
        elif mode == 'support':
            return self.forward_support(img=img, feats=feats, **kwargs)
        elif mode == 'extract_feat':
            assert img is not None
            return self.extract_feat(img=img)
        else:
            raise ValueError()

    def forward_train(self, gt_label, img=None, feats=None, **kwargs):
        """Forward computation during training.

        Input can be either images or extracted features.

        Args:
            img (Tensor | None): With shape (N, C, H, W). Default: None.
            feats (Tensor | None): With shape (N, C). Default: None.
            gt_label (Tensor): It should be of shape (N, 1) encoding the
                ground-truth label of input images.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        assert (img is not None) or (feats is not None)
        if feats is None:
            x = self.extract_feat(img)
        else:
            x = feats
        losses = dict()
        loss = self.head.forward_train(x, gt_label)
        losses.update(loss)

        return losses

    def forward_support(self, gt_label, img=None, feats=None, **kwargs):
        """Forward support data in meta testing.

        Input can be either images or extracted features.

        Args:
            img (Tensor | None): With shape (N, C, H, W). Default: None.
            feats (Tensor | None): With shape (N, C).
            gt_label (Tensor): It should be of shape (N, 1) encoding the
                ground-truth label of input images.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """

        assert (img is not None) or (feats is not None)
        if feats is None:
            x = self.extract_feat(img)
        else:
            x = feats
        losses = dict()
        loss = self.meta_test_head.forward_support(x, gt_label, **kwargs)
        losses.update(loss)

        return losses

    def forward_query(self, img=None, feats=None, **kwargs):
        """Forward query data in meta testing.

        Input can be either images or extracted features.

        Args:
            img (Tensor | None): With shape (N, C, H, W). Default: None.
            feats (Tensor | None): With shape (N, C). Default: None.

        Returns:
            list[np.ndarray]: A list of predicted results.
        """
        assert (img is not None) or (feats is not None)
        if feats is None:
            x = self.extract_feat(img)
        else:
            x = feats
        return self.meta_test_head.forward_query(x, **kwargs)

    def before_meta_test(self, meta_test_cfg, **kwargs):
        """Used in meta testing.

        This function will be called before the meta testing.
        """
        # For each test task the model will be copied and reset.
        # When using extracted features to accelerate meta testing,
        # the unused backbone will be removed to avoid copying
        # useless parameters.
        if meta_test_cfg.get('fast_test', False):
            self.backbone = None
        else:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        meta_test_head = copy.deepcopy(self.meta_test_head_cfg)
        # create a head for meta testing only, since the number
        # of classes is different from the training one.
        self.meta_test_head = build_head(meta_test_head)
        self.meta_test_cfg = meta_test_cfg

    def before_forward_support(self, **kwargs):
        """Used in meta testing.

        This function will be called before model forward support data during
        meta testing.
        """
        assert self.meta_test_head is not None
        self.meta_test_head.before_forward_support()
        self.meta_test_head.to(self.device)

    def before_forward_query(self, **kwargs):
        """Used in meta testing.

        This function will be called before model forward query data during
        meta testing.
        """
        assert self.meta_test_head is not None
        self.meta_test_head.before_forward_query()
