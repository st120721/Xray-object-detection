import os

import json

from model import faster_rcnn,tools
from xray_dataloader import Xaydataset
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch


data_path="dataset"
train_dataset = Xaydataset(data_path, "train")
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True)
val_dataset = Xaydataset(data_path, "val")
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)
#sample = next(iter(train_dataloader))

num_classes = 5

model = faster_rcnn.FasterRCNN(n_class=num_classes)
model = model.to("cuda")

params = [p for p in model.parameters() if p.requires_grad]


lr=0.0003
lr=0.001
optimizer = torch.optim.Adam(params, lr=lr)



def train(model, optimizer, sample):
    model.train()
    img=sample["img"]
    bbox=sample["boxes"]
    label=sample["label"]
    # print(tools.totensor(img))
    # print(tools.totensor(bbox))
    # print(label.cuda())
    img, bbox, label = img.cuda().float(), bbox.cuda(), label.cuda()

    optimizer.zero_grad()
    losses = model(img, bbox, label)
    losses["total_loss"].backward()
    optimizer.step()

    return losses

def eval(dataloader, model):
    model.eval()
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for idx, sample in enumerate(dataloader):
        img = sample["img"]
        bbox = sample["boxes"]
        label = sample["label"]
        size= sample["size"]
        pred_bboxes_, pred_labels_, pred_scores_ = model.predict(img, [size])
        gt_bboxes += list(bbox)
        gt_labels += list(label)
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_

    result = tools.eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


def main():
    tmp_path = './checkpoint.pth'
    save_stride = 1000
    num_epochs = 400
    model_dir="models"
    best_map = 0
    total_train_loss = []
    total_eval_result = []
    for epoch in range(num_epochs):
        print("---------------------epoch: {}---------------------".format(epoch+1))
        if epoch > 0:
            checkpoint = torch.load(tmp_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        train_loss=[]
        print("train phase")
        for idx, sample in enumerate(train_dataloader):

            losses=train(model, optimizer, sample)

            train_loss.append(losses)

        total_train_loss.append(train_loss)
        checkpoint = {
            'model': faster_rcnn.FasterRCNN(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }

        torch.save(checkpoint, tmp_path)
        if (epoch + 1) % save_stride == 0:
            torch.save(checkpoint, os.path.join(model_dir, 'frcnn_ver2_{}.pth'.format(epoch + 1)))
        torch.save(checkpoint, os.path.join(model_dir, 'frcnn_recent.pth'))



        print("validation phase")
        eval_result=eval(val_dataloader,model)
        total_eval_result.append(eval_result)
        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            print("best_map: ",best_map )
            torch.save(checkpoint, os.path.join(model_dir, 'frcnn_ver2_best.pth'))

    torch.save(total_train_loss, 'total_train_loss.pt')
    torch.save(total_eval_result, 'total_eval_result.pt')


if __name__ == '__main__':


    main()

