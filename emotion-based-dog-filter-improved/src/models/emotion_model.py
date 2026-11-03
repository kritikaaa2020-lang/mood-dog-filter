
import torch
import torch.nn as nn
from torchvision import models

EMOTION_CLASSES = ["Happy", "Sad", "Surprise"]


def create_emotion_model(pretrained=False):

    model = models.resnet18(
        weights=(
            models.ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )
    )

    # Freeze all original layers first
    for param in model.parameters():
        param.requires_grad = False

    # Fine-tune the final block; the dataset is too small to tune more safely.
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace final layer with three emotion outputs
    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, 3)

    return model


if __name__ == "__main__":

    model = create_emotion_model()

    print("Fine-tuning model created!")
    print("Classes:", EMOTION_CLASSES)
    print("Output classes:", model.fc.out_features)
