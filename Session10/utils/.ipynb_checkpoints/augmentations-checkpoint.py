# A common class to return any albumentation transform based on the dict passed.
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensor

class getCompose_simple:
    def __init__(self, transform_dict, compose_prob=1.0):
        compose_list = []
        for transform, its_param in transform_dict.items():
            compose_list.append(getattr(A, transform)(**its_param))
    
        compose_list.append(ToTensor())
        self.compose_obj = A.Compose(compose_list, p=compose_prob)

    def __call__(self, img): 
        img = np.array(img)
        img = self.compose_obj(image=img)['image']
        # return Image.fromarray(img) to use within torchvision transform
        return img
