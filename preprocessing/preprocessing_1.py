
"""
    collect the images, which satisfy the following condition:
        1. contain the category of dangerous item in the list "selected_items".
        2. the number of dangerous items equals to "num_pos"

"""

import os
import shutil
from xml.dom.minidom import parse

image_in_path = " " # raw dataset
anno_in_path = " "  # raw annotation
image_out_path = "data\images"
anno_out_path = "data\\Annotation"
for path in [image_out_path,anno_out_path]:
    if not os.path.exists(path):
        os.makedirs(path)
    shutil.rmtree(path)
    os.mkdir(path)


class_to_idx = {"Gun": 1, "Knife": 2, "Wrench": 3, "Pliers": 4, "'Scissors'": 5}
selected_items=["Gun","Knife","Wrench"]
num_pos=1

def get(image):
    dom = parse("{}\\{}.xml".format(anno_in_path, image.split(".")[0]))
    data = dom.documentElement
    objects = data.getElementsByTagName('object')
    width = data.getElementsByTagName('width')[0].childNodes[0].nodeValue
    height = data.getElementsByTagName('height')[0].childNodes[0].nodeValue
    height, width = float(height), float(width)
    boxes = []
    labels = []
    is_selected = True
    for obj in objects:
        name = obj.getElementsByTagName('name')[0].childNodes[0].nodeValue
        bndbox = obj.getElementsByTagName('bndbox')[0]
        xmin = bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue
        ymin = bndbox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue
        xmax = bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue
        ymax = bndbox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue
        labels.append(name)
        if name not in selected_items:
            is_selected = False
        boxes.append([xmin, ymin, xmax, ymax])
    return boxes, [width, height], is_selected


l = []
files = os.listdir(image_in_path)
for file_name in files:
    try:
        boxes, size, is_selected = get(file_name)
        w, h = size
        if len(boxes) == num_pos and w / h < 1.5 and h / w < 1.5 and is_selected:
            l.append(file_name)
    except:
        print("error file: {}".format(file_name))


for i, f in enumerate(l):
    shutil.copyfile(os.path.join(image_in_path, f),
                    "{}\\{}.jpg".format(image_out_path, str(i).zfill(4)))
    shutil.copyfile("{}\\{}.xml".format(anno_in_path, f.split(".")[0]),
                    "{}\\{}.xml".format(anno_out_path, str(i).zfill(4)))


# num_class = {"Gun": 535, "Knife": 283, "Wrench": 344, "Pliers": 1216, "'Scissors'": 601}
# gun, knife and wrench: 1162
