import jittor as jt 
from jittor import nn
import warnings

from jdet.utils.registry import NECKS
from jdet.models.utils.modules import ConvModule
from jdet.models.utils.weight_init import xavier_init
from jdet.models.utils.weight_init import constant_init
import pdb


@NECKS.register_module()
class FPN(nn.Module):
 

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform'),
                 upsample_div_factor=1,
                 use_balanced=False,        # 是否启用 BalancedFPN
                 refine_type='NonLocal',  ):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.upsample_cfg = upsample_cfg.copy()
        self.upsample_div_factor = upsample_div_factor
        self.use_balanced = use_balanced
        if self.use_balanced:
            if refine_type == 'NonLocal':
                self.refine = NonLocal2d(out_channels)
            elif refine_type == 'Conv':
                self.refine = ConvModule(out_channels, out_channels, 3, padding=1)
            else:
                self.refine = nn.Identity()
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
                self.fpn_convs.append(extra_fpn_conv)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def execute(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += nn.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += nn.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)
            laterals[i - 1] /= self.upsample_div_factor
        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(nn.pool(outs[-1], 1, stride=2,op="maximum"))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](nn.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        if self.use_balanced:
            target_size = outs[2].shape[2:]   # (H4, W4)
        
            # 1. rescale 所有层到同一尺寸
            resized = [
                nn.interpolate(feat, size=target_size, mode='nearest')
                for feat in outs
            ]
            balanced = jt.stack(resized, dim=0).mean(dim=0)
            balanced = self.refine(balanced)
        
            # 2. 再 strengthen 回各层
            outs = [
                feat + nn.interpolate(balanced, size=feat.shape[2:], mode='nearest')
                for feat in outs
            ]
        # =====================================
        return tuple(outs)




class NonLocal2d(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super().__init__()
        inter_channels = max(in_channels // reduc_ratio, 1)
        self.theta = nn.Conv2d(in_channels, inter_channels, 1)
        self.phi   = nn.Conv2d(in_channels, inter_channels, 1)
        self.g     = nn.Conv2d(in_channels, inter_channels, 1)
        self.out_conv = nn.Conv2d(inter_channels, in_channels, 1)
        self.norm = nn.BatchNorm2d(in_channels)
        constant_init(self.norm, 0)   

    def execute(self, x):
        n, c, h, w = x.shape
        theta = self.theta(x).reshape(n, -1, h*w)          # N×C'×HW
        phi   = self.phi(x).reshape(n, -1, h*w).permute(0,2,1)  # N×HW×C'
        g     = self.g(x).reshape(n, -1, h*w)              # N×C'×HW
        attn = nn.softmax(jt.matmul(theta, phi), dim=-1)   # N×C'×C'
        y = jt.matmul(attn, g).reshape(n, -1, h, w)        # N×C'×H×W
        y = self.out_conv(y)
        return self.norm(y + x)














# import jittor as jt 
# from jittor import nn
# import warnings

# from jdet.utils.registry import NECKS
# from jdet.models.utils.modules import ConvModule
# from jdet.models.utils.weight_init import xavier_init

# @NECKS.register_module()
# class FPN(nn.Module):
#     r"""Feature Pyramid Network.
#     This is an implementation of paper `Feature Pyramid Networks for Object
#     Detection <https://arxiv.org/abs/1612.03144>`_.
#     Args:
#         in_channels (List[int]): Number of input channels per scale.
#         out_channels (int): Number of output channels (used at each scale)
#         num_outs (int): Number of output scales.
#         start_level (int): Index of the start input backbone level used to
#             build the feature pyramid. Default: 0.
#         end_level (int): Index of the end input backbone level (exclusive) to
#             build the feature pyramid. Default: -1, which means the last level.
#         add_extra_convs (bool | str): If bool, it decides whether to add conv
#             layers on top of the original feature maps. Default to False.
#             If True, its actual mode is specified by `extra_convs_on_inputs`.
#             If str, it specifies the source feature map of the extra convs.
#             Only the following options are allowed
#             - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
#             - 'on_lateral':  Last feature map after lateral convs.
#             - 'on_output': The last output feature map after fpn convs.
#         extra_convs_on_inputs (bool, deprecated): Whether to apply extra convs
#             on the original feature from the backbone. If True,
#             it is equivalent to `add_extra_convs='on_input'`. If False, it is
#             equivalent to set `add_extra_convs='on_output'`. Default to True.
#         relu_before_extra_convs (bool): Whether to apply relu before the extra
#             conv. Default: False.
#         no_norm_on_lateral (bool): Whether to apply norm on lateral.
#             Default: False.
#         conv_cfg (dict): Config dict for convolution layer. Default: None.
#         norm_cfg (dict): Config dict for normalization layer. Default: None.
#         act_cfg (str): Config dict for activation layer in ConvModule.
#             Default: None.
#         upsample_cfg (dict): Config dict for interpolate layer.
#             Default: `dict(mode='nearest')`
#         init_cfg (dict or list[dict], optional): Initialization config dict.
#     Example:
#         >>> in_channels = [2, 3, 5, 7]
#         >>> scales = [340, 170, 84, 43]
#         >>> inputs = [jt.randn(1, c, s, s)
#         ...           for c, s in zip(in_channels, scales)]
#         >>> self = FPN(in_channels, 11, len(in_channels)).eval()
#         >>> outputs = self.forward(inputs)
#         >>> for i in range(len(outputs)):
#         ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
#         outputs[0].shape = ([1, 11, 340, 340])
#         outputs[1].shape = ([1, 11, 170, 170])
#         outputs[2].shape = ([1, 11, 84, 84])
#         outputs[3].shape = ([1, 11, 43, 43])
#     """

#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  num_outs,
#                  start_level=0,
#                  end_level=-1,
#                  add_extra_convs=False,
#                  extra_convs_on_inputs=True,
#                  relu_before_extra_convs=False,
#                  no_norm_on_lateral=False,
#                  conv_cfg=None,
#                  norm_cfg=None,
#                  act_cfg=None,
#                  upsample_cfg=dict(mode='nearest'),
#                  init_cfg=dict(
#                      type='Xavier', layer='Conv2d', distribution='uniform'),
#                  upsample_div_factor=1):
#         super(FPN, self).__init__()
#         assert isinstance(in_channels, list)
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.num_ins = len(in_channels)
#         self.num_outs = num_outs
#         self.relu_before_extra_convs = relu_before_extra_convs
#         self.no_norm_on_lateral = no_norm_on_lateral
#         self.upsample_cfg = upsample_cfg.copy()
#         self.upsample_div_factor = upsample_div_factor
#         if end_level == -1:
#             self.backbone_end_level = self.num_ins
#             assert num_outs >= self.num_ins - start_level
#         else:
#             # if end_level < inputs, no extra level is allowed
#             self.backbone_end_level = end_level
#             assert end_level <= len(in_channels)
#             assert num_outs == end_level - start_level
#         self.start_level = start_level
#         self.end_level = end_level
#         self.add_extra_convs = add_extra_convs
#         assert isinstance(add_extra_convs, (str, bool))
#         if isinstance(add_extra_convs, str):
#             assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')

#         self.lateral_convs = nn.ModuleList()
#         self.fpn_convs = nn.ModuleList()

#         for i in range(self.start_level, self.backbone_end_level):
#             l_conv = ConvModule(
#                 in_channels[i],
#                 out_channels,
#                 1,
#                 conv_cfg=conv_cfg,
#                 norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
#                 act_cfg=act_cfg)
#             fpn_conv = ConvModule(
#                 out_channels,
#                 out_channels,
#                 3,
#                 padding=1,
#                 conv_cfg=conv_cfg,
#                 norm_cfg=norm_cfg,
#                 act_cfg=act_cfg)

#             self.lateral_convs.append(l_conv)
#             self.fpn_convs.append(fpn_conv)

#         # add extra conv layers (e.g., RetinaNet)
#         extra_levels = num_outs - self.backbone_end_level + self.start_level
#         if self.add_extra_convs and extra_levels >= 1:
#             for i in range(extra_levels):
#                 if i == 0 and self.add_extra_convs == 'on_input':
#                     in_channels = self.in_channels[self.backbone_end_level - 1]
#                 else:
#                     in_channels = out_channels
#                 extra_fpn_conv = ConvModule(
#                     in_channels,
#                     out_channels,
#                     3,
#                     stride=2,
#                     padding=1,
#                     conv_cfg=conv_cfg,
#                     norm_cfg=norm_cfg,
#                     act_cfg=act_cfg)
#                 self.fpn_convs.append(extra_fpn_conv)
#         self.init_weights()

#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 xavier_init(m, distribution='uniform')

#     def execute(self, inputs):
#         """Forward function."""
#         assert len(inputs) == len(self.in_channels)

#         # build laterals
#         laterals = [
#             lateral_conv(inputs[i + self.start_level])
#             for i, lateral_conv in enumerate(self.lateral_convs)
#         ]

#         # build top-down path
#         used_backbone_levels = len(laterals)
#         for i in range(used_backbone_levels - 1, 0, -1):
#             # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
#             #  it cannot co-exist with `size` in `F.interpolate`.
#             if 'scale_factor' in self.upsample_cfg:
#                 laterals[i - 1] += nn.interpolate(laterals[i],
#                                                  **self.upsample_cfg)
#             else:
#                 prev_shape = laterals[i - 1].shape[2:]
#                 laterals[i - 1] += nn.interpolate(
#                     laterals[i], size=prev_shape, **self.upsample_cfg)
#             laterals[i - 1] /= self.upsample_div_factor
#         # build outputs
#         # part 1: from original levels
#         outs = [
#             self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
#         ]
#         # part 2: add extra levels
#         if self.num_outs > len(outs):
#             # use max pool to get more levels on top of outputs
#             # (e.g., Faster R-CNN, Mask R-CNN)
#             if not self.add_extra_convs:
#                 for i in range(self.num_outs - used_backbone_levels):
#                     outs.append(nn.pool(outs[-1], 1, stride=2,op="maximum"))
#             # add conv layers on top of original feature maps (RetinaNet)
#             else:
#                 if self.add_extra_convs == 'on_input':
#                     extra_source = inputs[self.backbone_end_level - 1]
#                 elif self.add_extra_convs == 'on_lateral':
#                     extra_source = laterals[-1]
#                 elif self.add_extra_convs == 'on_output':
#                     extra_source = outs[-1]
#                 else:
#                     raise NotImplementedError
#                 outs.append(self.fpn_convs[used_backbone_levels](extra_source))
#                 for i in range(used_backbone_levels + 1, self.num_outs):
#                     if self.relu_before_extra_convs:
#                         outs.append(self.fpn_convs[i](nn.relu(outs[-1])))
#                     else:
#                         outs.append(self.fpn_convs[i](outs[-1]))
#         return tuple(outs)
