import torch
from torchvision import transforms
import os
from PIL import Image
import numpy as np

class FeatureExtractor:

    def __init__(self):
        
        # self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        #self.model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
        # self.model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
        
        self.preprocess = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ])

    def extract(self, img):
        input_tensor = self.preprocess(img)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_batch)
            output = output
        return output

if __name__ == "__main__":
    
    img_path = os.path.join('static', 'img', 'a1.jpg')
    fe = FeatureExtractor()

    x = fe.extract(img=Image.open(img_path).convert('RGB'))[0]

    feature = x / np.linalg.norm(x)

    print(feature[:20])