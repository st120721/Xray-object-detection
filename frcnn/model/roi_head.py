import torch
from torch import nn
from torchvision.models import vgg16
from torchvision.ops import RoIPool
from model import tools

class RoIHead(nn.Module):

    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super(RoIHead, self).__init__()

        self.classifier = classifier

        # self.classifier = nn.Sequential(*[nn.Linear(25088, 4096), nn.Linear(4096, 4096)])

        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        self.cls_loc.weight.data.normal_(0, 0.01)
        self.cls_loc.bias.data.zero_()
        self.score.weight.data.normal_(0, 0.01)
        self.score.bias.data.zero_()

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPool((self.roi_size, self.roi_size), self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        roi_indices = tools.totensor(roi_indices).float()
        rois = tools.totensor(rois).float()
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
        # yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = xy_indices_and_rois.contiguous()

        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        fc = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc)
        roi_scores = self.score(fc)
        return roi_cls_locs, roi_scores