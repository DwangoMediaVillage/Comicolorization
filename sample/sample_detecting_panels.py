import argparse
import json
import os
import sys

ROOT_PATH = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(ROOT_PATH)
import comicolorization

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='./sample/HinagikuKenzan_026.jpg',
                    help='the path of input page image.')
parser.add_argument('--mfe', default='./manga-frame-extraction/MangaFrameExtraction/MFE',
                    help='the path of manga-frame-extraction\'s binary file.')
parser.add_argument('--output', default='panel_rectangle.json',
                    help='the path of output panel rectangle information.')
args = parser.parse_args()

panel_rectangle = comicolorization.pipeline.detect_panel_rectangle(args.input, args.mfe)
json.dump(panel_rectangle, open(args.output, 'w'))
