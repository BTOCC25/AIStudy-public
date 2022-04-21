from torchvision import models
import bentoml

model = models.resnet18(pretrained=True)

bentoml.pytorch.save("imagenet_cls", model)
