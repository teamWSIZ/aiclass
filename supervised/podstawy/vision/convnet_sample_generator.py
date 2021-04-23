from random import randint

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as TS


class SuperposeSign(object):
    """
    Klasa którą można użyć jako "Transformację" (jak: torchvision.transforms) w kompozycji transformacji
    obrazów.
    """
    sign: Image
    mask: Image

    def __init__(self, sign_filename='sign.png'):
        sign = Image.open(sign_filename).resize((64, 64))
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
        left = randint(0, 200)
        top = randint(0, 200)
        image.paste(im=self.sign, box=(left, top), mask=self.mask)  # naklejenie maski na obrazek
        # box:  (left,upper) lub (left,upper,right,lower)
        # mask: - tylko wybrany region będzie update'owany; tam gdzie jest 0 nie będzie zmiany
        return image

    def __repr__(self):
        return self.__class__.__name__ + '()'


transform_negative = TS.Compose(
    [TS.Resize((256, 256)),
     TS.RandomAffine(degrees=2, fillcolor=(0, 0, 0), translate=(0.05, 0)),
     TS.GaussianBlur(kernel_size=3),
     TS.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
     # SuperposeSign(),
     # TS.RandomAffine(degrees=30, fillcolor=(0, 0, 0), translate=(0.1, 0.1)),
     TS.ToTensor()])

transform_positive = TS.Compose(
    [TS.Resize((256, 256)),
     TS.RandomAffine(degrees=2, fillcolor=(0, 0, 0), translate=(0.05, 0)),
     TS.GaussianBlur(kernel_size=3),
     TS.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
     TS.ToTensor()])


def generate_transform(resolution=256, sign_filename=None):
    """
    Tworzy zestaw transformacji przygotowujących/randomizujących próbki.
    Jeśli `sign_filename` jest podane, to ten obrazek będzie nakładany na tło poprzednich.
    :return:
    """
    # podstawowe transformacje
    tts = [TS.Resize((resolution, resolution)),
           TS.RandomAffine(degrees=2, fillcolor=(0, 0, 0), translate=(0.05, 0)),
           TS.GaussianBlur(kernel_size=3),
           TS.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)]
    if sign_filename is not None:
        tts.append(SuperposeSign(sign_filename))
    tts.append(TS.ToTensor())

    return TS.Compose(tts)


def generate_sample(img_count, sign_filename=None):
    """
    :return: Tensor typu [80, 3, 128, 128], i.e. nr. obrazka, nr koloru, rząd, kolumna

    Uwaga: liczba obrazków zawsze jest wielokrotnością liczby obrazków w folderze z których są pobierane.
    """
    t = generate_transform(resolution=256, sign_filename=sign_filename)
    dataset = datasets.ImageFolder('small', transform=t)
    dataloader = DataLoader(dataset, batch_size=15, shuffle=True)  # adjust to number of pictures
    res = None
    while res is None or res.size()[0] < img_count:
        for (images, classes) in dataloader:
            for i in images:
                TF.to_pil_image(i).show()
            if res is None:
                res = images
            else:
                res = torch.cat((res, images), 0)
    if sign_filename is not None:
        print(f'generated for [{sign_filename:10}]:', res.size())
    else:
        print(f'generated backgrounds:', res.size())
    return res


generate_sample(3, 'sign.png')
