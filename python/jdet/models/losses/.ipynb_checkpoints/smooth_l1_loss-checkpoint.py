import jittor as jt
from jdet.utils.registry import LOSSES
from jittor import nn

def balanced_l1_loss(pred,
                     target,
                     weight=None,
                     alpha=0.5,
                     gamma=1.5,
                     avg_factor=None,
                     reduction='mean'):
  
    diff = jt.abs(pred - target)
    b = jt.exp(gamma / alpha) - 1
    flag_in = (diff < 1.0).float()
    loss_in = alpha / b * (b * diff + 1) * jt.log(b * diff + 1) - alpha * diff
    loss_out = gamma * diff + (alpha * jt.log(b + 1) - gamma)
    loss = flag_in * loss_in + (1 - flag_in) * loss_out

    if weight is not None:
        if weight.ndim == 1:
            weight = weight[:, None]
        loss *= weight

    if avg_factor is None:
        avg_factor = max(loss.shape[0], 1)

    if reduction == 'mean':
        loss = loss.sum() / avg_factor
    elif reduction == 'sum':
        loss = loss.sum()
    return loss
# ------------------------------------------------------------


def smooth_l1_loss(pred, target, weight=None, beta=1., avg_factor=None, reduction="mean"):
    diff = jt.abs(pred - target)
    if beta != 0.:
        flag = (diff < beta).float()
        loss = flag * 0.5 * diff.sqr() / beta + (1 - flag) * (diff - 0.5 * beta)
    else:
        loss = diff
    if weight is not None:
        if weight.ndim == 1:
            weight = weight[:, None]
        loss *= weight
    if avg_factor is None:
        avg_factor = max(loss.shape[0], 1)
    if reduction == "mean":
        loss = loss.sum() / avg_factor
    elif reduction == "sum":
        loss = loss.sum()
    return loss


@LOSSES.register_module()
class SmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def execute(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss_bbox = self.loss_weight * smooth_l1_loss(
            pred, target, weight, beta=self.beta,
            reduction=reduction, avg_factor=avg_factor)
        return loss_bbox


@LOSSES.register_module()
class BalancedL1Loss(nn.Module):
    def __init__(self, alpha=0.5, gamma=1.5, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def execute(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss_bbox = self.loss_weight * balanced_l1_loss(
            pred, target, weight,
            alpha=self.alpha, gamma=self.gamma,
            reduction=reduction, avg_factor=avg_factor)
        return loss_bbox


# import jittor as jt 
# from jdet.utils.registry import LOSSES
# from jittor import nn

# def smooth_l1_loss(pred,target,weight=None,beta=1.,avg_factor=None,reduction="mean"):
#     diff = jt.abs(pred-target)
#     if beta!=0.:
#         flag = (diff<beta).float()
#         loss = flag*0.5* diff.sqr()/beta + (1-flag)*(diff - 0.5 * beta)
#     else:
#         loss = diff 

#     if weight is not None:
#         if weight.ndim==1:
#             weight = weight[:,None]
#         loss *= weight

#     if avg_factor is None:
#         avg_factor = max(loss.shape[0],1)

#     if reduction == "mean":
#         loss = loss.sum()/avg_factor
#     elif reduction == "sum":
#         loss = loss.sum()

#     return loss 


# @LOSSES.register_module()
# class SmoothL1Loss(nn.Module):

#     def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
#         super(SmoothL1Loss, self).__init__()
#         self.beta = beta
#         self.reduction = reduction
#         self.loss_weight = loss_weight

#     def execute(self,
#                 pred,
#                 target,
#                 weight=None,
#                 avg_factor=None,
#                 reduction_override=None):
#         assert reduction_override in (None, 'none', 'mean', 'sum')
#         reduction = (
#             reduction_override if reduction_override else self.reduction)
#         loss_bbox = self.loss_weight * smooth_l1_loss(
#             pred,
#             target,
#             weight,
#             beta=self.beta,
#             reduction=reduction,
#             avg_factor=avg_factor)
#         return loss_bbox