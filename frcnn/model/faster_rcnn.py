
import torch
import numpy as np

from torchvision.ops import nms
from torch import nn
from torch.nn import functional as F
from model import tools
from model import feature_extractor,region_proposal_network,roi_head,tools

def nograd(f):
    def new_f(*args,**kwargs):
        with torch.no_grad():
           return f(*args,**kwargs)
    return new_f

class FasterRCNN(nn.Module):
    feat_stride = 16  # downsample 16x for output of conv5 in vgg16
    def __init__(self, n_class=5,
                loc_normalize_mean = (0., 0., 0., 0.),
                loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
    ):
        super(FasterRCNN, self).__init__()

        self.n_class = n_class

        self.extractor,self.classifier = feature_extractor.get_feature_extractor()
        self.rpn = region_proposal_network.RegionProposalNetwork()
        self.head = roi_head.RoIHead(n_class=n_class,
                        roi_size=7,
                        spatial_scale=(1. / self.feat_stride),
                        classifier=self.classifier )

        self.proposal_target_creator=tools.ProposalTargetCreator()
        self.anchor_target_creator=tools.AnchorTargetCreator()        # mean and std
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std

        self.nms_thresh = 0.3
        self.score_thresh = 0.05

        self.rpn_sigma = 3
        self.roi_sigma = 1
    # @property
    # def n_class(self):
    #     # Total number of classes including the background.
    #     return self.n_class

    def forward(self, x,bboxes,labels,scale=1.):
        _, _, H, W = x.shape
        img_size = (H, W)

        features = self.extractor(x)

        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.rpn(features, img_size,scale)

        # Since batch size is one, convert variables to singular form
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        # Sample RoIs and forward
        # it's fine to break the computation graph of rois,
        # consider them as constant input
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            tools.tonumpy(bbox),
            tools.tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std)
        # NOTE it's all zero because now it only support for batch=1 now
        sample_roi_index = torch.zeros(len(sample_roi))
        roi_cls_loc, roi_score = self.head(
            features,
            sample_roi,
            sample_roi_index)

        # ------------------ RPN losses -------------------#
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            tools.tonumpy(bbox),
            anchor,
            img_size)
        gt_rpn_label = tools.totensor(gt_rpn_label).long()
        gt_rpn_loc = tools.totensor(gt_rpn_loc)
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc,
            gt_rpn_loc,
            gt_rpn_label.data,
            self.rpn_sigma)

        # NOTE: default value of ignore_index is -100 ...
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = tools.tonumpy(rpn_score)[tools.tonumpy(gt_rpn_label) > -1]

        # ------------------ ROI losses (fast rcnn loss) -------------------#
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[torch.arange(0, n_sample).long().cuda(), \
                              tools.totensor(gt_roi_label).long()]
        gt_roi_label = tools.totensor(gt_roi_label).long()
        gt_roi_loc = tools.totensor(gt_roi_loc)

        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)

        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())


        total_loss = rpn_loc_loss+ rpn_cls_loss+ roi_loc_loss+ roi_cls_loss

        losses=dict(rpn_loc_loss=rpn_loc_loss,rpn_cls_loss=rpn_cls_loss,
                  roi_loc_loss=roi_loc_loss,roi_cls_loss=roi_cls_loss,
                  total_loss=total_loss )


        return losses



    def _suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = nms(cls_bbox_l, prob_l,self.nms_thresh)
            # import ipdb;ipdb.set_trace()
            # keep = cp.asnumpy(keep)
            bbox.append(cls_bbox_l[keep].cpu().numpy())
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep].cpu().numpy())
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score

    @nograd
    def predict(self, imgs,sizes=None):
        self.eval()

        bboxes = list()
        labels = list()
        scores = list()
        for img, size in zip(imgs, sizes):
            img = tools.totensor(img[None]).float()
            scale = float(img.shape[3] / size[1])
            h = self.extractor(img)
            rpn_locs, rpn_scores, rois, roi_indices, anchor = \
                self.rpn(h, img.shape[2:], scale)
            roi_cls_locs, roi_scores = self.head(
                h, rois, roi_indices)

            # We are assuming that batch size is 1.
            roi_score = roi_scores.data
            roi_cls_loc = roi_cls_locs.data
            roi = tools.totensor(rois) / scale

            # Convert predictions to bounding boxes in image coordinates.
            # Bounding boxes are scaled to the scale of the input images.
            mean = tools.totensor(np.array(self.loc_normalize_mean)). \
                repeat(self.n_class)[None]
            std = tools.totensor(np.array(self.loc_normalize_std)). \
                repeat(self.n_class)[None]

            roi_cls_loc = (roi_cls_loc * std + mean)
            roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
            cls_bbox = tools.loc_to_bbox(tools.tonumpy(roi).reshape((-1, 4)),
                                tools.tonumpy(roi_cls_loc).reshape((-1, 4)))
            cls_bbox = tools.totensor(cls_bbox)
            cls_bbox = cls_bbox.view(-1, self.n_class * 4)
            # clip bounding box
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=float(size[0]))
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=float(size[1]))

            prob = (F.softmax(tools.totensor(roi_score), dim=1))

            bbox, label, score = self._suppress(cls_bbox, prob)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)
        self.train()
        return bboxes, labels, scores



def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = torch.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation,
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float()) # ignore gt_label==-1 for rpn_loss
    return loc_loss