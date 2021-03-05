
import torch
from torchvision import transforms
from model.tools import *
from PIL import Image, ImageDraw, ImageFont
from ssd.model.ssd_model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
distinct_colors = ['#d2f53c', '#000080', '#bd0000', '#fabebe', '#aa6e28']
label_map = {0: "Background", 1: "Gun", 2: "Knife", 3: "Wrench", 4: "Pliers"}

reverse_label_map = {v: k for k, v in label_map.items()}
label_color_map = {k: distinct_colors[i] for i, k in enumerate(reverse_label_map.keys())}

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def detect(model,image_path, min_score=0.2, max_overlap=0.5, top_k=10, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """
    original_image = Image.open(image_path)
    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    det_scores = det_scores[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("../calibril.ttf", 15)

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])

        # Text
        text_size = font.getsize(f'{det_labels[i].upper()}: CONF: {det_scores[i]}')
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=f'{det_labels[i].upper()}   CONF: {det_scores[i]}', fill='white',
                  font=font)
    del draw

    return annotated_image

# Change the path to detect another image
image_path = '../dataset/test/0769.jpg'

model_dir = "..//results"
model = SSD300(4,path_pretrained_state_dict=False)
state_dict = torch.load(os.path.join(model_dir, 'state_dict.pth' ))
model.load_state_dict(state_dict)

detect(model, image_path, min_score=0.2, max_overlap=0.5, top_k=10).show()
