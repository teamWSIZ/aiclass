from random import randint

import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as TS


# ↑↑ TF.pad, etc...
# Pad
# ColorJitter
# Grayscale
# RandomAffine (also shear)
# RandomApply (random transformations, some selected)
#


# https://medium.com/secure-and-private-ai-writing-challenge/loading-image-using-pytorch-c2e2dcce6ef2

class SuperposeSign(object):
    """
    Superpose sign on an image.
    """
    sign: Image
    mask: Image

    def __init__(self):
        # operacje na PIL.Image
        sign = Image.open('sign.png').resize((64, 64))
        # self.sign = sign.rotate(0, resample=Image.BICUBIC, expand=True)
        self.sign = sign
        self.mask = sign.copy().convert('L')  # 'cień'; "single channel image"
        self.mask = ImageEnhance.Brightness(self.mask).enhance(4)  # maska jest typu ~0.4; (zbyt transparentna)
        # self.mask.show()

    def __call__(self, image):
        """
        Args:
            image (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            PIL: Transformed image.
        """
        left = randint(0,200)
        top = randint(0,200)
        image.paste(im=self.sign, box=(left, top), mask=self.mask)  # naklejenie maski na obrazek
        # box:  (left,upper) lub (left,upper,right,lower)
        # mask: - tylko wybrany region będzie update'owany; tam gdzie jest 0 nie będzie zmiany
        return image

    def __repr__(self):
        return self.__class__.__name__ + '()'


transform = TS.Compose(
    [TS.Resize((256, 256)),
     TS.RandomAffine(degrees=2, fillcolor=(0, 0, 0), translate=(0.05, 0)),
     TS.GaussianBlur(kernel_size=3),
     TS.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
     # SuperposeSign(),
     # TS.RandomAffine(degrees=30, fillcolor=(0, 0, 0), translate=(0.1, 0.1)),
     TS.ToTensor()])

dataset = datasets.ImageFolder('small', transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for i in range(2):
    for (images, classes) in dataloader:
        #classes -- narazie same zera, tensor([0, 0, 0, 0])
        print(images.size(), type(images), classes)  #torch.Size([4, 3, 256, 256]) <class 'torch.Tensor'>
        for i in images:
            TF.to_pil_image(i).show()
        # print(classes)  # [0, 0, 0]
        # TF.to_pil_image(x[0][0]).show()
