import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from ssd.model.xray_dataloder import XrayDataset
from ssd.model.ssd_model import *
from ssd.model import tools

data_path = "../dataset"
batch_size = 16
n_classes = 4
# path_pretrained_state_dict = torch.load("model/best_state_dict.pth")


def collate_fn(batch):
    return (zip(*batch))


train_dataset = XrayDataset(data_path, "train")
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                              collate_fn=collate_fn)

val_dataset = XrayDataset(data_path, "val")
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# path_pretrained_state_dict = "../best_vgg_state_dict.pth"
model = SSD300(n_classes=n_classes)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2)

criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)


def train(model, criterion, optimizer, images, bboxes, labels):
    model.train()

    images = torch.stack(images, 0).to(device)
    bboxes = [torch.tensor(b).to(device) / 300 for b in bboxes]
    labels = [torch.tensor(l).to(device) for l in labels]

    pred_locs, pred_scores = model(images)

    loss = criterion(pred_locs, pred_scores, bboxes, labels)

    # clear gradient and perform backprop
    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    return loss


def validate(model, criterion, images, bboxes, labels):
    model.eval()

    with torch.no_grad():
        images = torch.stack(images, 0).to(device)
        bboxes = [torch.tensor(b).to(device) / 300 for b in bboxes]
        labels = [torch.tensor(l).to(device) for l in labels]

        pred_locs, pred_scores = model(images)
        det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(pred_locs, pred_scores,
                                                                                   min_score=0.01, max_overlap=0.45,
                                                                                   top_k=200)
        loss = criterion(pred_locs, pred_scores, bboxes, labels)

    return loss, det_boxes_batch, det_labels_batch, det_scores_batch


if __name__ == '__main__':
    max_epoch = 50
    max_val_mAP = 0

    model_dir = '..\\results'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    train_losses = []
    val_losses = []
    val_mAPs = []
    APs = []
    for epoch in (range(max_epoch)):
        train_loss = 0.0

        # Load the saved MODEL AND OPTIMIZER after evaluation.
        if epoch > 0:
            state_dict = torch.load(os.path.join(model_dir, 'state_dict.pth'))
            optimizer_state_dict = torch.load(os.path.join(model_dir, 'optimizer_state_dict.pth'))
            model.load_state_dict = state_dict
            optimizer.load_state_dict = optimizer_state_dict

        print()
        print("--------------epoch {}-----------------".format(epoch + 1))
        for idx, (images, bboxes, labels) in enumerate(train_dataloader):
            curr_loss = train(model, criterion, optimizer, images, bboxes, labels)
            train_loss += curr_loss / len(train_dataloader)

        print("train_loss: ", train_loss)

        torch.save(model.state_dict(), os.path.join(model_dir, 'state_dict.pth'))
        torch.save(optimizer.state_dict(), os.path.join(model_dir, 'optimizer_state_dict.pth'))

        val_loss = 0.0
        det_boxes = list()
        det_labels = list()
        det_scores = list()
        true_boxes = list()
        true_labels = list()
        
        for idx, (images, bboxes, labels) in enumerate(val_dataloader):
            curr_loss, det_boxes_batch, det_labels_batch, det_scores_batch = validate(model, criterion, images,
                                                                                      bboxes, labels)
            val_loss += curr_loss / len(val_dataloader)

            bboxes = [torch.tensor(b).to(device) / 300 for b in bboxes]
            labels = [torch.tensor(l).to(device) for l in labels]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(bboxes)
            true_labels.extend(labels)

        val_APs, val_mAP = tools.calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels)
        print("val_loss: ", val_loss)
        print("val_mAP", val_mAP)
        print(val_APs)
        if max_val_mAP < val_mAP:
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_state_dict.pth'))
            max_val_mAP = val_mAP
            print(f'new min val loss: {max_val_mAP}')

        train_losses.append(train_loss.cpu().detach().numpy())
        val_losses.append(val_loss.cpu().detach().numpy())
        val_mAPs.append(val_mAP)
        APs.append(val_APs)
        scheduler.step(val_loss)

    torch.save(train_losses, os.path.join(model_dir, 'train_losses.pth'))
    torch.save(val_losses, os.path.join(model_dir, 'val_losses.pth'))
    torch.save(val_mAPs, os.path.join(model_dir, 'val_mAPs.pth'))
    torch.save(APs, os.path.join(model_dir, 'APs.pth'))
