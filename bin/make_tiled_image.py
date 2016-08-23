"""
python3 make_tiled_image.py \
    --root /result/projectX/test_50000 \

"""

import argparse
import glob
import math
import os
import re
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--root', required=True, type=str)
parser.add_argument('--col', type=int)
parser.add_argument('--row', type=int)
args = parser.parse_args()

paths_all = glob.glob(args.root + '/*')
filenames_all = [os.path.splitext(os.path.basename(path))[0] for path in paths_all]


def get_filename(path):
    return os.path.splitext(os.path.basename(path))[0]


def get_number(path):
    filename = get_filename(path)
    pos_number = re.search(r'\d+$', filename)
    return int(filename[pos_number.start():pos_number.end()])


def save_tiled_image(paths_input, path_output, num_image=None, col=None, row=None):
    if col is None:
        col = math.ceil(math.sqrt(num_image))

    if row is None:
        row = math.ceil(num_image / col)

    command = \
        '''
        montage \
        -tile {col}x{row} \
        -geometry +0 \
        -border 10x10 \
        {paths_input} \
        {path_output}
        '''.format(
            col=col,
            row=row,
            paths_input=' '.join(paths_input),
            path_output=path_output,
        )
    subprocess.check_output(command, shell=True)


# collect tile base name
tile_basename_list = set()
for filename in filenames_all:
    if not re.match(r'.*\d+$', filename):
        continue

    pos_number = re.search(r'\d+$', filename)
    tile_basename = filename[:pos_number.start()]

    tile_basename_list.add(tile_basename)

# make tile image
for tile_basename in tile_basename_list:
    paths_image = [
        path
        for path, filename in zip(paths_all, filenames_all)
        if re.match(tile_basename + '\d+$', filename)
        ]

    paths_image.sort(key=lambda x: get_number(x))
    save_tiled_image(
        paths_input=paths_image,
        path_output=os.path.join(args.root, 'tiled_' + tile_basename + '.png'),
        num_image=len(paths_image),
        col=args.col,
        row=args.row,
    )
