import torch.nn as nn
import torchvision.models as models


def get_model(model_name, weights=False, num_classes=1000, **kwargs):
    w = "IMAGENET1K_V1" if weights else None
    model_fn = getattr(models, model_name)
    model = model_fn(weights=w, **kwargs)

    if num_classes != 1000:
        if hasattr(model, "fc"):  # ResNet
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif hasattr(model, "classifier"):  # VGG, DenseNet, MobileNet
            if isinstance(model.classifier, nn.Sequential):
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(in_features, num_classes)
            else:
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)
        elif hasattr(model, "heads"):  # ViT, Swin
            if isinstance(model.heads, nn.Linear):
                model.heads = nn.Linear(model.heads.in_features, num_classes)
            elif isinstance(model.heads, nn.Sequential):
                in_features = model.heads[-1].in_features
                model.heads[-1] = nn.Linear(in_features, num_classes)
        elif hasattr(model, "head"):  # EfficientNet_V2, ConvNeXt
            if isinstance(model.head, nn.Linear):
                model.head = nn.Linear(model.head.in_features, num_classes)
            elif isinstance(model.head, nn.Sequential):
                in_features = model.head[-1].in_features
                model.head[-1] = nn.Linear(in_features, num_classes)
    return model


if __name__ == "__main__":
    # Example usage
    model = get_model("convnext_tiny", weights=None, num_classes=10)
    print(model)
