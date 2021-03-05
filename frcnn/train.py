import os
from torch.utils.data import Dataset, DataLoader
import torch
from frcnn.model import faster_rcnn,tools
from frcnn.model.xray_dataloder import XrayDataset

data_path = "../dataset"
train_dataset = XrayDataset(data_path, "train")
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True)
val_dataset = XrayDataset(data_path, "val")
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)

num_classes = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = faster_rcnn.FasterRCNN(n_class=num_classes)
model = model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
lr=0.003
optimizer = torch.optim.Adam(params, lr=lr)

def train(model, optimizer, sample):
    model.train()
    img=sample["img"]
    bbox=sample["boxes"]
    label=sample["label"]
    img, bbox, label = img.cuda().float(), bbox.cuda(), label.cuda()

    optimizer.zero_grad()
    losses = model(img, bbox, label)
    losses["total_loss"].backward()
    optimizer.step()

    return losses

def eval(dataloader, model):
    model.eval()
    det_boxes, det_labels, det_scores = list(), list(), list()
    true_boxes, true_labels, difficulties = list(), list(), list()
    for idx, sample in enumerate(dataloader):
        img = sample["img"]
        bbox = sample["boxes"]
        label = sample["label"]
        size= sample["size"]
        det_boxes_batch, det_labels_batch, det_scores_batch = model.predict(img, [size])

        bboxes = [b.to(device)for b in bbox]
        labels = [l.to(device) for l in label]
        diff = [torch.tensor([0]).to(device) for l in label]

        det_boxes.extend(det_boxes_batch)
        det_labels.extend(det_labels_batch)
        det_scores.extend(det_scores_batch)
        true_boxes.extend(bboxes)
        true_labels.extend(labels)
        difficulties.extend(diff)


    val_APs, val_mAP = tools.calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, difficulties)

    return val_mAP


def main():
    tmp_path = './checkpoint.pth'

    num_epochs = 50
    model_dir = '..\\results'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
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
        #torch.save(checkpoint, os.path.join(model_dir, 'frcnn_recent.pth'))

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

