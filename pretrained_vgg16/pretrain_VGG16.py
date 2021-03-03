# -*- coding: utf-8 -*-
"""

"""

import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# data_path = 'data_vgg'
# model_dir = 'data_vgg\models'


class XayDataset(Dataset):
    def __init__(self, data_path, is_training):
        self.data_path = data_path
        self.train_path = os.path.join(data_path, 'train')
        self.val_path = os.path.join(data_path, 'val')
        self.is_training = is_training
        if self.is_training:
            self.target_path = self.train_path
        else:
            self.target_path = self.val_path
        self.img_list = os.listdir(self.target_path)

        self.tensor_transform = torchvision.transforms.ToTensor()
        self.normalize_transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])
        self.random_crop = torchvision.transforms.RandomCrop(size=170)
        self.random_flip = torchvision.transforms.RandomHorizontalFlip(p=0.5)
        self.resize = torchvision.transforms.Resize(size=(224, 224))
        self.train_transform = torchvision.transforms.Compose([self.tensor_transform, self.random_flip,
                                                               self.resize,
                                                               self.normalize_transform])
        self.validate_transform = torchvision.transforms.Compose([self.tensor_transform, self.resize,
                                                                  self.normalize_transform])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        fp = os.path.join(self.target_path, self.img_list[idx])
        img = Image.open(fp)
        class_name = self.img_list[idx].split(".")[1]
        #class_to_idx = {"Gun": 0, "Knife": 1, "Wrench": 2}
        class_to_idx = {"background":0,"Gun": 1, "Knife": 2, "Wrench": 3}

        label = class_to_idx[class_name]

        if self.is_training:
            input = self.train_transform(img)
        else:
            input = self.validate_transform(img)

        sample = dict()
        sample['input'] = input
        sample['target'] = label
        sample['class_name'] = class_name
        return sample


batch_size = 16

train_dataset = XayDataset(data_path, True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,pin_memory=True)
val_dataset = XayDataset(data_path, False)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, pin_memory=True )

device = 'cuda'


class VGG_16(nn.Module):
    def __init__(self, num_class=4):
        super(VGG_16, self).__init__()

        self.net = torchvision.models.vgg16(pretrained=False)
        self.net.classifier[6] = nn.Linear(in_features=4096, out_features=num_class, bias=True)

    def forward(self, img):
        out = self.net(img)
        return out


model = VGG_16()

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.0005)


def train(model, optimizer, sample):
    model.train()

    criterion = nn.CrossEntropyLoss()

    optimizer.zero_grad()

    input = sample['input'].float().to(device)
    target = sample['target'].long().to(device)

    pred = model(input)
    pred_loss = criterion(pred, target)

    y_pred = pred.argmax(1)
    num_true = torch.sum(((y_pred - target) == 0))

    pred_loss.backward()

    optimizer.step()
    return pred_loss.item(), num_true


def validate(model, sample):
    model.eval()

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        input = sample['input'].float().to(device)
        target = sample['target'].long().to(device)

        pred = model(input)
        pred_loss = criterion(pred, target)

        y_pred = pred.argmax(1)
        num_true = torch.sum(((y_pred - target) == 0))

    return pred_loss.item(), num_true

if __name__ == '__main__':
    max_epoch = 100
    save_stride = 10
    # tmp_path =  os.path.join(model_dir, 'checkpoint.pth')
    max_accu = 0
    best_epoch=0
    for epoch in range(max_epoch):

        train_loss = 0.0


        if epoch > 0:
            state_dict=torch.load(os.path.join(model_dir, 'state_dict.pth'))
            optimizer_state_dict = torch.load(os.path.join(model_dir, 'optimizer_state_dict.pth'))
            model.load_state_dict = state_dict
            optimizer.load_state_dict=optimizer_state_dict


        print()
        print("--------------epoch {}-----------------".format(epoch+1))
        train_true = 0
        for idx, sample in enumerate(train_dataloader):
            curr_loss, num_true = train(model, optimizer, sample)
            train_loss += curr_loss / len(train_dataloader)
            train_true += num_true
        print("train_loss: ", train_loss)
        print("train_acc: ", train_true / len(train_dataset))


        torch.save(model.state_dict(), os.path.join(model_dir, 'state_dict.pth'))
        torch.save(optimizer.state_dict(), os.path.join(model_dir, 'optimizer_state_dict.pth'))


        val_loss = 0.0
        val_accu = 0.0

        # Iterate over the val_dataloader
        val_true = 0
        for idx, sample in enumerate(val_dataloader):
            curr_loss, val_num_true = validate(model, sample)
            val_true += val_num_true
            val_loss += curr_loss / len(val_dataloader)
        print("val_loss: ", val_loss)
        print("val_acc: ", val_true / len(val_dataset))

        accu = val_true / len(val_dataset)

        if max_accu < accu:
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_state_dict.pth'))
            best_epoch=epoch+1
        print("best epoch {}, max acc:{} ".format(best_epoch,max_accu))
