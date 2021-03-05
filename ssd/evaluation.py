import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from ssd.model.xray_dataloder import XrayDataset
from ssd.model.ssd_model import *
from ssd.model import tools

model_dir = "..//results"
train_losses=torch.load( os.path.join(model_dir, 'train_losses.pth' ))
val_losses=torch.load( os.path.join(model_dir, 'val_losses.pth' ))
val_mAPs=torch.load( os.path.join(model_dir, 'val_mAPs.pth' ))


# path_pretrained_state_dict="..//best_vgg_state_dict.pth"  # this line is used to load retrained model
model = SSD300(4,path_pretrained_state_dict=False)
state_dict = torch.load(os.path.join(model_dir, 'state_dict.pth' ))
model.load_state_dict(state_dict)

plt.plot(np.arange(len(val_mAPs))+1,np.array(val_mAPs))
plt.xlabel('epoch')
plt.ylabel('mAP')
plt.title('mean Average Precision(mAP)')
plt.savefig(os.path.join(model_dir, 'map_100.png' ))
plt.show()

plt.plot(np.arange(len(train_losses))+1,train_losses,label="train losses")
plt.plot(np.arange(len(val_losses))+1,val_losses,label="validation losses")
plt.legend()
plt.title("Losses of SSD with self-retrained VGG16 over 50 epoch")
plt.savefig(os.path.join(model_dir,'train_losses.png' ))
plt.show()

data_path="../dataset"
batch_size=1

def collate_fn(batch):
    return (zip(*batch))

test_dataset = XrayDataset(data_path, "test")
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

min_score=0.2
max_overlap=0.5
top_k=200
det_boxes = list()
det_labels = list()
det_scores = list()
true_boxes = list()
true_labels = list()
model.eval()

for idx, (img, bboxes, labels) in enumerate(test_dataloader):
    images = torch.stack(img, 0).to(device)
    bboxes = [torch.tensor(b).to(device)/300 for b in bboxes]
    labels = [torch.tensor(l).to(device) for l in labels]
    with torch.no_grad():
        predicted_locs, predicted_scores = model(images)

        det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score, max_overlap=max_overlap, top_k=top_k)
        det_boxes.extend(det_boxes_batch)
        det_labels.extend(det_labels_batch)
        det_scores.extend(det_scores_batch)
        true_boxes.extend(bboxes)
        true_labels.extend(labels)

test_APs, test_mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels)

print(test_APs)
print('\nMean Average Precision (mAP): %.3f' % test_mAP)
