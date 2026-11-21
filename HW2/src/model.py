import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights

#construction of the 3 model for the classification

NUM_CLASSES = 37 #race of oxfordpet


def build_model(approach: str): #returns a model ready to be trained according to the approach
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights) #charge the MobileNetV3 model with pretrained weights

    #substitute the last layer of the classifier with a new one with NUM_CLASSES(37) 
    in_features = model.classifier[-1].in_features   #its 1024
    model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)

    #configure what is trainable according to the approach
    if approach == "full":
        pass #all are trainable for default

    elif approach == "head":
        #freeze features, train only the classifier
        for param in model.features.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

    elif approach == "feat":
        #freeze everything, just extract features 
        for param in model.parameters():
            param.requires_grad = False

    return model


def count_trainable_params(model) -> int: #count how many parsmeters are trainable
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for approach in ["full", "head", "feat"]:
        model = build_model(approach)
        n = count_trainable_params(model)
        print(f"{approach:6s}  trainable params: {n:,}")