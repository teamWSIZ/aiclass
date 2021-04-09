from PIL import Image


def adjust(path, name, extension, size=256, center=False):
    filename = f'{path}{name}{extension}'
    modified_filename = f'{path}{name}_{size}{extension}'
    image = Image.open(filename)
    left_offset, top_offset = 0, 0
    if center:
        h, w = image.size
        left_offset = (w - size) / 2
        top_offset = (h - size) / 2

    image = image.crop((left_offset, top_offset, left_offset + size, top_offset + size))
    image.save(f'{modified_filename}')


for i in range(1, 2):
    adjust('raw/', f'plasma{i}', '.png', size=256, center=False)
