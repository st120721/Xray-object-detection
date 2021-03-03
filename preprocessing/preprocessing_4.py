import os
import shutil
from xml.dom.minidom import parse
from PIL import Image

img_size= 1100
stride=10
image_in_path = "D:\BaiduNetdiskDownload\\0\\0"
img_out_path = "data\\test_positiv"
if not os.path.exists(img_out_path):
    os.makedirs(img_out_path)
shutil.rmtree(img_out_path)
os.mkdir(img_out_path)

files = os.listdir(image_in_path)
for i, file_name in enumerate(files):
    if i%stride==0:
        shutil.copyfile(os.path.join(image_in_path, file_name),
                        "{}\\{}.jpg".format(img_out_path, "P"+str( i/stride).zfill(4)))
    if i/stride>img_size:
        break