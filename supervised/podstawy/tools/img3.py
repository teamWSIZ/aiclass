from PIL import Image
from torchvision import datasets, transforms as TS

image = Image.open('_/star_128.png')

sign = Image.open('sth.png').resize((64, 64))
mask = sign.copy()
mask.convert('L')  # just a shadow -- good for masking

sign.show()
image.paste(sign, (20, 20), mask)
image.show()
