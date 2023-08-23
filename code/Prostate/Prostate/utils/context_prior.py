from mmcv.cnn.bricks.norm import build_norm_layer
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale
from torch import nn
from torch.nn.modules.utils import _pair
import numpy as np

from ..builder import HEADS, build_loss
from .decode_head import BaseDecodeHead


class CPHead(BaseDecodeHead):
    """Context Prior for Scene Segmentation.
    This head is the implementation of `CPNet
    <https://arxiv.org/abs/2004.01547>`_.
    """

    def __init__(self,
                 prior_channels,
                 prior_size,
                 groups=1,
                 loss_prior_decode=dict(type='AffinityLoss', loss_weight=1.0),
                 **kwargs):
        super(CPHead, self).__init__(**kwargs)
        self.prior_channels = prior_channels
        self.prior_size = [prior_size,prior_size,prior_size]

        # 2개의 ch -> hwd개
        self.prior_conv= nn.Sequential(
            nn.Conv3d(self.prior_channels, np.prod(self.prior_size), 1, padding=0,stride=1),
            nn.BatchNorm3d(np.prod(self.prior_size)))

        self.loss_prior_decode = build_loss(loss_prior_decode)

    def forward(self, inputs):
        """Forward function."""

        batch_size, channels, height, width, dimension = inputs.size()
        assert self.prior_size[0] == height and self.prior_size[1] == width and self.prior_size[1] == dimension

        value = inputs      # bs,ch,h,w,d

        context_prior_map = self.prior_conv(value)       # ch -> hwd : bs,ch,h,w,d -> bs,hwd ,h,w,d : 왜 굳이 2ch -> H*W로 늘리는걸까?
        context_prior_map = context_prior_map.view(batch_size,
                                                   np.prod(self.prior_size),-1)  # bs,hwd ,h,w,d -> bs, hwd, hwd
        context_prior_map = context_prior_map.permute(0, 2, 1)  # bs, hwd(ch), hwd -> bs, hwd, hwd(ch)
        context_prior_map = torch.sigmoid(context_prior_map)        # final context_prior map : GT가 0~1사이 이므로 , sigmoid 사용


        '''
        value = value.view(batch_size, self.prior_channels, -1)     # bs,ch,h,w,d -> bs,ch,hwd
        value = value.permute(0, 2, 1)                              # bs,ch,hwd -> bs,hwd, ch

        intra_context = torch.bmm(context_prior_map, value)     # bs, hwd, hwd(ch) *  bs,hwd,ch  = bs,hwd, ch
        intra_context = intra_context.div(np.prod(self.prior_size))     # why divide ?
        intra_context = intra_context.permute(0, 2, 1).contiguous()     # bs,hwd, ch -> bs,ch, hwd
        intra_context = intra_context.view(batch_size, self.prior_channels,self.prior_size[0],self.prior_size[1],self.prior_size[2]) # bs, ch, h,w,d
        '''
        return context_prior_map


    '''
    def _construct_ideal_affinity_matrix(self, label, label_size):
        scaled_labels = F.interpolate(
            label.float(), size=label_size, mode="nearest")     # 작았던 label의 크기 크게
        scaled_labels = scaled_labels.squeeze_().long()
        scaled_labels[scaled_labels == 255] = self.num_classes      #255인 값을 class수의 값으로 (255 -> 1), background는 num_classes에 포함X
        one_hot_labels = F.one_hot(scaled_labels, self.num_classes + 1) # one hot encoding으로 바꿔줌 , back까지 합쳐서 고려하는 듯
        one_hot_labels = one_hot_labels.view(
            one_hot_labels.size(0), -1, self.num_classes + 1).float()       # bs, hw, ch
        ideal_affinity_matrix = torch.bmm(one_hot_labels,
                                          one_hot_labels.permute(0, 2, 1))       # bs, hw, ch * bs, ch, hw = bs, hw, hw
        return ideal_affinity_matrix
    '''


    def losses(self, seg_logit, seg_label):
        """Compute ``seg``, ``prior_map`` loss."""
        seg_logit, context_prior_map = seg_logit        # segmentation map, context_prior_map
        logit_size = seg_logit.shape[2:]    # h,w,d
        loss = dict()
        loss.update(super(CPHead, self).losses(seg_logit, seg_label))    #segmentation loss
        prior_loss = self.loss_prior_decode(
            context_prior_map,
            self._construct_ideal_affinity_matrix(seg_label, logit_size))       # afiinity loss
        loss['loss_prior'] = prior_loss
        return loss

class AffinityLoss(nn.Module):
    """CrossEntropyLoss.
    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float], optional): Weight of each class.
            Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(AffinityLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.cls_criterion = binary_cross_entropy

    def binary_cross_entropy(pred,
                             label,
                             use_sigmoid=False,
                             weight=None,
                             reduction='mean',
                             avg_factor=None,
                             class_weight=None):
        """Calculate the binary CrossEntropy loss.
        Args:
            pred (torch.Tensor): The prediction with shape (N, 1).
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            reduction (str, optional): The method used to reduce the loss.
                Options are "none", "mean" and "sum".
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            class_weight (list[float], optional): The weight for each class.
        Returns:
            torch.Tensor: The calculated loss
        """
        if pred.dim() != label.dim():
            label, weight = _expand_onehot_labels(label, weight, pred.size(-1))

        # weighted element-wise losses
        if weight is not None:
            weight = weight.float()
        if use_sigmoid:
            loss = F.binary_cross_entropy_with_logits(
                pred, label.float(), weight=class_weight, reduction='none')
        else:
            loss = F.binary_cross_entropy(
                pred, label.float(), weight=class_weight, reduction='none')
        # do the reduction for the weighted loss
        loss = weight_reduce_loss(
            loss, weight, reduction=reduction, avg_factor=avg_factor)

        return loss

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        unary_term = self.cls_criterion(
            cls_score,
            label,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)

        diagonal_matrix = (1 - torch.eye(label.size(1))).to(label.get_device())
        vtarget = diagonal_matrix * label

        recall_part = torch.sum(cls_score * vtarget, dim=2)
        denominator = torch.sum(vtarget, dim=2)
        denominator = denominator.masked_fill_(~(denominator > 0), 1)
        recall_part = recall_part.div_(denominator)
        recall_label = torch.ones_like(recall_part)
        recall_loss = self.cls_criterion(
            recall_part,
            recall_label,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)

        spec_part = torch.sum((1 - cls_score) * (1 - label), dim=2)
        denominator = torch.sum(1 - label, dim=2)
        denominator = denominator.masked_fill_(~(denominator > 0), 1)
        spec_part = spec_part.div_(denominator)
        spec_label = torch.ones_like(spec_part)
        spec_loss = self.cls_criterion(
            spec_part,
            spec_label,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)

        precision_part = torch.sum(cls_score * vtarget, dim=2)
        denominator = torch.sum(cls_score, dim=2)
        denominator = denominator.masked_fill_(~(denominator > 0), 1)
        precision_part = precision_part.div_(denominator)
        precision_label = torch.ones_like(precision_part)
        precision_loss = self.cls_criterion(
            precision_part,
            precision_label,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)

        global_term = recall_loss + spec_loss + precision_loss

        loss_cls = self.loss_weight * (unary_term + global_term)
        return loss_cls