import bentoml
import torch
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from bentoml.io import Image, JSON

imagenet_runner = bentoml.pytorch.load_runner("imagenet_cls:latest")
image_cls_service = bentoml.Service("imagenet_cls_service", runners=[imagenet_runner])

image_pre_pro = Compose(
    [
        Resize((224, 224)),
        ToTensor(),
        Normalize([0.5, 0.5, 0.5], [0.23, 0.224, 0.225])
    ]
)

@image_cls_service.api(input=Image(), output=JSON())
def classifier(input_img):
    input_tensor = image_pre_pro(image_img)
    logit = imagenet_runner.run(input_tensor)
    porb = torch.softmax(logit, dim=0)
    sort_porb = porb.sort(descending=True)
    top_k_dict = dict()
    for val, idx in zip(sort_porb[0][:5], sort_porb[1][:5]):
        top_k_dict[idx.item()] = val.item()
        
    return top_k_dict