import argparse
import glob
import os
import sys

import chainer

ROOT_PATH = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(ROOT_PATH)
import comicolorization

parser = argparse.ArgumentParser()
parser.add_argument("dataset_dir")
parser.add_argument("output_dir")
parser.add_argument("--margin_ratio", type=float, default=0.3)
parser.add_argument("--size_image", type=int, default=128)
args = parser.parse_args()
print(args)

pathes = glob.glob("{}/*".format(args.dataset_dir))
dataset = comicolorization.dataset.FaceImageDataset(pathes, margin_ratio=args.margin_ratio,
                                                    classifier_path=os.path.join(ROOT_PATH, "animeface",
                                                                                 "lbpcascade_animeface.xml"),
                                                    output_resize=(args.size_image, args.size_image))
iterator = chainer.iterators.MultiprocessIterator(dataset, batch_size=100, repeat=False, shuffle=False)

for batch in iterator:
    for path, image in batch:
        if image is not None:
            basename = os.path.basename(path)
            image.save(os.path.join(args.output_dir, basename))
