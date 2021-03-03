"""
    create the dataset for trainging VGG16

"""

import os
import shutil
from xml.dom.minidom import parse
from PIL import Image

img_path = "data\images"
ann_path="data\Annotation"
img_out_path = "data_vgg\images"
if not os.path.exists(img_out_path):
    os.makedirs(img_out_path)
shutil.rmtree(img_out_path)
os.mkdir(img_out_path)

files = os.listdir(img_path)
for file_name in files:
    dom = parse("{}/{}.xml".format(ann_path, file_name.split(".")[0]))
    data = dom.documentElement
    objects = data.getElementsByTagName('object')
    width = data.getElementsByTagName('width')[0].childNodes[0].nodeValue
    height = data.getElementsByTagName('height')[0].childNodes[0].nodeValue
    height, width = float(height), float(width)
    for obj in objects:
                name = obj.getElementsByTagName('name')[0].childNodes[0].nodeValue
                bndbox = obj.getElementsByTagName('bndbox')[0]
                xmin =float (bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue)
                ymin = float (bndbox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue)
                xmax = float (bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue)
                ymax = float (bndbox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue)

                img= Image.open(os.path.join(img_path, file_name))

                if ymax-ymin>xmax-xmin:
                    y2,y1=ymax,ymin
                    x1=max((xmax+xmin)/2-(ymax-ymin)/2,0)
                    x2=min((xmax+xmin)/2+(ymax-ymin)/2,width)
                    box = (x1, y1, x2,y2)
                    box_bg =(x1+(ymax-ymin), y1, x2+(ymax-ymin),y2)
                else:
                    x2, x1 = xmax, xmin
                    y1 = max((ymax + ymin)/ 2 - (xmax - xmin) /2,0)
                    y2 = min((ymax + ymin) / 2 + (xmax - xmin) /2,height)
                    box = (x1, y1, x2, y2)
                    box_bg =(x1, y1 + (xmax - xmin), x2, y2 + (xmax - xmin))

                region = img.crop(box)
                file_name="{}.{}.jpg".format(file_name.split(".")[0],name)
                region.save(os.path.join(img_out_path, file_name))

                if box_bg[0]>0 and box_bg[1]>0 and box_bg[2]<width and box_bg[3]<height :
                    region = img.crop(box_bg)
                    file_name = "{}.{}.jpg".format(file_name.split(".")[0], "background")
                    region.save(os.path.join(img_out_path, file_name))



files = os.listdir(img_out_path)

if not os.path.exists('data_vgg\\train'):
    os.makedirs('data_vgg\\train')
shutil.rmtree('data_vgg\\train')
os.mkdir('data_vgg\\train')

if not os.path.exists('data_vgg\\val'):
    os.makedirs('data_vgg\\val')
shutil.rmtree('data_vgg\\val')
os.mkdir('data_vgg\\val')


path="data_vgg\images"
files = os.listdir(path)
for i in range(1500):
    file = files[i]

    fp = os.path.join(path, file)
    if i % 10 < 7:
        shutil.copyfile(fp, "data_vgg\\train\{}".format(file))
    elif 7 <= i % 10 <10:
        shutil.copyfile(fp, "data_vgg\\val\{}".format(file))

