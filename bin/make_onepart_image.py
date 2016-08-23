"""
python3 bin/make_onepart_image.py \
    --num_part 1 4 9 16 25 49 \
    --part_size 1 3 \
    --path_image `find /mnt/project/comicolorization/dataset/reference_images_224x224/*` \
    --path_save /mnt/project/comicolorization/dataset/reference_images_onepart \

"""
import argparse
import itertools
import numpy
import os
from PIL import Image
import sys

ROOT_PATH = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(ROOT_PATH)
import comicolorization

parser = argparse.ArgumentParser()
parser.add_argument('--path_image', nargs='+', required=True)
parser.add_argument('--path_save', required=True)
parser.add_argument('--save_filename_format', type=str, default='{filename}_num{num_part}_size{part_size}.png')
parser.add_argument('--num_part', type=int, nargs='+', default=[3])
parser.add_argument('--part_size', type=int, nargs='+', default=[3])
args = parser.parse_args()

image_dataset = comicolorization.dataset.PILImageDataset(paths=args.path_image)

image_and_path_list = zip(image_dataset, args.path_image)
num_part_list = args.num_part
part_size_list = args.part_size

for (image, path), num_part, part_size in \
        itertools.product(
            zip(image_dataset, args.path_image),
            num_part_list,
            part_size_list,
        ):
    width, height = image.size

    array = numpy.asarray(image, dtype=numpy.uint8)[:, :, :3]
    array_output = numpy.zeros((width, height, 4), dtype=numpy.uint8)

    nx = ny = numpy.ceil(numpy.sqrt(num_part))
    xs = numpy.linspace(0, width - 1, nx + 2, dtype=numpy.int32)[1:-1]
    ys = numpy.linspace(0, height - 1, ny + 2, dtype=numpy.int32)[1:-1]
    points = numpy.asarray([[x, y] for x in xs for y in ys])

    ixs = iys = numpy.arange(part_size) - part_size // 2
    for x, y in points[:num_part]:
        for ix, iy in itertools.product(ixs, iys):
            array_output[x + ix, y + iy, :] = \
                numpy.append(array[x + ix, y + iy, :], numpy.array(255, dtype=numpy.uint8))

    image_output = Image.fromarray(array_output)

    filename = args.save_filename_format.format(
        filename=os.path.splitext(os.path.basename(path))[0],
        num_part=num_part,
        part_size=part_size,
    )
    image_output.save(os.path.join(args.path_save, filename))
