import numpy as np
from torch.nn import functional as F
from torch import nn
import torch
from torchvision.ops import nms
from model import tools

class RegionProposalNetwork(nn.Module):
    """
    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    """

    def __init__(
            self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32], feat_stride=16):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_base = self.generate_anchor_base(
            anchor_scales=anchor_scales, ratios=ratios)
        self.feat_stride = feat_stride

        n_anchor = self.anchor_base.shape[0]

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)  # conv sliding layer
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)  # classification layer
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)  # Regression layer

        self.proposal_layer = tools.ProposalCreator(self)
        # initialize weight
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv1.bias.data.zero_()
        self.score.weight.data.normal_(0, 0.01)
        self.score.bias.data.zero_()
        self.loc.weight.data.normal_(0, 0.01)
        self.loc.bias.data.zero_()

    def forward(self, feature_map, img_size,scale=1.):

        n, _, hh, ww = feature_map.shape  # batch_siz, 512, H/16,W/16
        anchor = self.generate_anchors(np.array(self.anchor_base), self.feat_stride, hh, ww)

        n_anchor = anchor.shape[0] // (hh * ww)  # 9

        h = F.relu(self.conv1(feature_map))
        rpn_locs = self.loc(h)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)  # from （n，hh，ww，9*4）to（n，hh*ww*9，4）

        rpn_scores = self.score(h)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()  # to（n，hh，ww，9*2）
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()  # 得到前景的分类概率
        rpn_fg_scores = rpn_fg_scores.view(n, -1)  # 得到所有anchor的前景分类概率
        rpn_scores = rpn_scores.view(n, -1, 2)  # 得到每一张feature map上所有anchor的网络输出值

        rois = list()
        roi_indices = list()
        for i in range(n):  # batch_size
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor, img_size,scale=scale
            )
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)
        return rpn_locs, rpn_scores, rois, roi_indices, anchor

    @staticmethod
    def generate_anchor_base(sub_sample=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
        y = sub_sample / 2
        x = sub_sample / 2
        anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)

        for i in range(len(ratios)):
            for j in range(len(anchor_scales)):
                h = sub_sample * anchor_scales[j] * np.sqrt(ratios[i])
                w = sub_sample * anchor_scales[j] * np.sqrt(1. / ratios[i])

                index = i * len(anchor_scales) + j

                anchor_base[index, 0] = y - h / 2.
                anchor_base[index, 1] = x - w / 2.
                anchor_base[index, 2] = y + h / 2.
                anchor_base[index, 3] = x + w / 2.
        return anchor_base

    @staticmethod
    def generate_anchors(anchor_base, feat_stride, height, width):

        shift_y = np.arange(0, height * feat_stride, feat_stride)
        shift_x = np.arange(0, width * feat_stride, feat_stride)
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                          shift_y.ravel(), shift_x.ravel()), axis=1)

        A = anchor_base.shape[0]
        K = shift.shape[0]
        anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((1, K, 4)).transpose((1, 0, 2))
        anchor = anchor.reshape((K * A, 4)).astype(np.float32)
        return anchor


    def control_proposals_num(self,loc, score, anchor, img_size,scale):

        if self.training:
            n_pre_nms = 12000
            n_post_nms = 2000
        else:
            n_pre_nms = 6000
            n_post_nms = 300
        nms_thresh = 0.7
        min_size = 16

        # Convert anchors into proposal via bbox transformations.
        # roi = loc2bbox(anchor, loc)
        roi = tools.loc_to_bbox(anchor, loc)

        # Clip predicted boxes to image.
        roi[:, slice(0, 4, 2)] = np.clip(
            roi[:, slice(0, 4, 2)], 0, img_size[0])
        roi[:, slice(1, 4, 2)] = np.clip(
            roi[:, slice(1, 4, 2)], 0, img_size[1])

        # Remove predicted boxes with either height or width < threshold.
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]
        keep = np.where((hs >= min_size) & (ws >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]

        # Sort all (proposal, score) pairs by score from highest to lowest.
        # Take top pre_nms_topN (e.g. 6000).
        order = score.ravel().argsort()[::-1]
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]

        keep = nms(
            torch.from_numpy(roi).cuda(),torch.from_numpy(score).cuda(),nms_thresh)
        if n_post_nms > 0:
            keep = keep[:n_post_nms]
        roi = roi[keep.cpu().numpy()]
        return roi


