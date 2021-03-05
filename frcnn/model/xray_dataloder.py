import os
from PIL import Image
from xml.dom.minidom import parse
import torchvision
from torch.utils.data import Dataset
import numpy as np

class XrayDataset(Dataset):

    def __init__(self, data_path, phase):
        self.data_path = data_path
        self.phase = phase
        self.images_path = os.path.join(self.data_path, self.phase)
        self.image_names = sorted(os.listdir(self.images_path))
        self.class_to_idx = {"Gun": 1, "Knife": 2, "Wrench": 3, "Pliers": 4}

    def __len__(self):

        return len(self.image_names)

    def __getitem__(self, idx):
        full_image_path = os.path.join(self.images_path, self.image_names[idx])
        img = Image.open(full_image_path)

        boxes = []
        labels = []
        try:
            dom = parse("{}/Annotation/{}.xml".format(self.data_path, self.image_names[idx].split(".")[0]))
            data = dom.documentElement

            objects = data.getElementsByTagName('object')
            width = data.getElementsByTagName('width')[0].childNodes[0].nodeValue
            height = data.getElementsByTagName('height')[0].childNodes[0].nodeValue
            height, width = float(height), float(width)

            for obj in objects:
                name = obj.getElementsByTagName('name')[0].childNodes[0].nodeValue
                bndbox = obj.getElementsByTagName('bndbox')[0]
                xmin = bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue
                ymin = bndbox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue
                xmax = bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue
                ymax = bndbox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue
                labels.append(self.class_to_idx[name])
                boxes.append([xmin, ymin, xmax, ymax])
        except:
            height, width = 300.0, 300.0
            boxes = [[0, 0, 300, 300]]
            labels = [0]

        img = torchvision.transforms.functional.resize(img, (300, 300))
        H, W = height, width
        o_H, o_W = img.size[0], img.size[1]
        for idx, bbox in enumerate(boxes):
            boxes[idx] = self.resize_bbox(bbox, (H, W), (o_H, o_W))

        img = torchvision.transforms.functional.to_tensor(img)
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        img = torchvision.transforms.functional.normalize(img, mean, std)
        sample = dict()
        sample["img"] = img
        sample["size"] = [o_H, o_W]
        sample["boxes"] = np.array(boxes)
        sample["label"] = np.array(labels)
        return sample

      #  return img, boxes, labels

    def resize_bbox(self, bbox, in_size, out_size):
        y_scale = float(out_size[0]) / in_size[0]
        x_scale = float(out_size[1]) / in_size[1]
        xmin = x_scale * float(bbox[0])
        ymin = y_scale * float(bbox[1])
        xmax = x_scale * float(bbox[2])
        ymax = y_scale * float(bbox[3])
        return [xmin, ymin, xmax, ymax]
