import PIL
import torchvision.transforms.functional as TF
from PIL import Image

from supervised.podstawy.utils import print_tensor

"""
Podstawy PIL (manilpulacja obrazkami w Python)
https://pillow.readthedocs.io/en/stable/reference/Image.html

"""

im = Image.open('raw/1/plasma13.png')
im = im.rotate(45, PIL.Image.CUBIC, expand=1, fillcolor=(0, 100, 100))
# im.rotate(45).show()
# white = (255,255,255)
# pil_image.rotate(angle, PIL.Image.NEAREST, expand = 1, fillcolor = white)

# image = Image.open('img/1/star32g.png')
# image = Image.open('star.png').crop((0, 0, 128, 128)).resize((128, 128)).rotate(5)
# image = Image.open('sign.png').resize((32, 32)).rotate(5)
# image.save('gg.png')

im = im.resize((32, 32))
# im.show()

x = TF.to_tensor(im)  # [3, 32, 32] -- channel, row, col; values: [0..1]; [4 ... jesli jest alpha
print(x.size()) #batch obrazków kolorowych 2d ... indeksy: [sample_number][channel][row][column]

# print_tensor(x[1])
# print(list_to_string(x[0].tolist()[16]))
print_tensor(x[2])
# im.show()

# showing tensors as images
iim = TF.to_pil_image(x)
iim.show()
