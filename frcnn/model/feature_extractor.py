from torchvision.models import vgg16
from torch import nn


def get_feature_extractor():
    """
    Uses VGG16 as the feature extractor.
    Input size of VGG16 is (batch_size, 3, 224, 224).
    Output size is (batch_size, 512, 14, 14), which is the size of the feature map.

    Returns:
        fe_extractor: the backbone for Faster R-CNN
        classifier  : the classifier is used as an input for roi_head
    """
    model = vgg16(pretrained=True)
    features = list(model.features)[:30]

    # feature extractor
    fe_extractor = nn.Sequential(*features)

    # classifier
    classifier = model.classifier
    classifier = list(classifier)
    # delete last layer of classifier to adapt the output classes
    del classifier[6]
    classifier = nn.Sequential(*classifier)

    # # freeze 10 layers
    for layer in features[:10]:
         for p in layer.parameters():
            p.requires_grad = False

    return fe_extractor, classifier

