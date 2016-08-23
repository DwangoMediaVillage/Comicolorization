"""
python -i bin/make_characteronly_dataset.py \
    --child_image_dir /path/to/character /
    --parent_image_dir /path/to/source /
    --output_image_dir /path/to/output /
"""
import chainer
from chainer.dataset import dataset_mixin
import argparse
import glob
import os
import sys
import six
import json
import numpy

parser = argparse.ArgumentParser()
parser.add_argument('--child_image_dir', type=str)
parser.add_argument('--parent_image_dir', type=str)
parser.add_argument('--output_image_dir', type=str)
args = parser.parse_args()


class MakeCharacteronlyDataset(dataset_mixin.DatasetMixin):
    @staticmethod
    def get_image_id(path):
        """
        >>> MakeCharacteronlyDataset.get_image_id('/path/to/image/12345.png')
        '12345'
        >>> MakeCharacteronlyDataset.get_image_id('/path/to/image/12345-67.png')
        '12345'
        """
        filename = os.path.splitext(os.path.basename(path))[0]
        image_id = filename.split('-')[0]
        return image_id

    def __init__(self, paths):
        if isinstance(paths, six.string_types):
            with open(paths) as paths_file:
                paths = [path.strip() for path in paths_file]
        self._paths = paths
        self.filtered_image_id_list = []
        for i in range(len(self._paths)):
            self.filtered_image_id_list.append(self.get_image_id(self._paths[i]))

    def __len__(self):
        return len(self.filtered_image_id_list)

    def get_example(self, i):
        filtered_imge_path = os.path.join(args.parent_image_dir, self.filtered_image_id_list[i])
        return filtered_imge_path


pathes = glob.glob("{}/*".format(args.child_image_dir))
dataset = MakeCharacteronlyDataset(pathes)
iterator = chainer.iterators.MultiprocessIterator(dataset, batch_size=100, repeat=False, shuffle=False)

for batch in iterator:
    for path in batch:
        basename = os.path.basename(path)
        if os.path.exists(os.path.join(args.parent_image_dir, basename)):
            os.symlink(os.path.join(args.parent_image_dir, basename), os.path.join(args.output_image_dir, basename))
