import argparse
import json
import os
from PIL import Image
import sys

ROOT_PATH = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(ROOT_PATH)
import comicolorization
import comicolorization_sr

parser = argparse.ArgumentParser()
parser.add_argument('--input_image', default='./sample/HinagikuKenzan_026.jpg', help='path of input page image')
parser.add_argument('--reference_images', nargs='+', default=[
    './sample/TasogareTsushin-1.png',
    './sample/Belmondo-1.png',
    './sample/HinagikuKenzan-1.png',
    './sample/Belmondo-1.png',
    './sample/TasogareTsushin-1.png',
], help='paths of reference images')
parser.add_argument('--panel_rectangle', default='./sample/panel_rectangle.json',
                    help='path of json file written panel rectangle')
parser.add_argument('--comicolorizatoin_model_directory', default='./model/comicolorization/',
                    help='the trained model directory for the comicolorization task.')
parser.add_argument('--comicolorizatoin_model_iteration', type=int, default=550000,
                    help='the trained model iteration for the comicolorization task.')
parser.add_argument('--super_resolution_model_directory', default='./model/super_resolution/',
                    help='the trained model directory for the super resolution task.')
parser.add_argument('--super_resolution_model_iteration', type=int, default=80000,
                    help='the trained model iteration for the super resolution task.')
parser.add_argument('--gpu', type=int, default=-1,
                    help='gpu number (-1 means the cpu mode).')
parser.add_argument('--output', default='colorized_image.png',
                    help='the path of colorized image.')
args = parser.parse_args()

# prepare neural network model
drawer = comicolorization.drawer.Drawer(
    path_result_directory=args.comicolorizatoin_model_directory,
    gpu=args.gpu,
)
drawer.load_model(iteration=args.comicolorizatoin_model_iteration)

drawer_sr = comicolorization_sr.drawer.Drawer(
    path_result_directory=args.super_resolution_model_directory,
    gpu=args.gpu,
    colorization_class=comicolorization_sr.colorization_task.ComicolorizationTask,
)
drawer_sr.load_model(iteration=args.super_resolution_model_iteration)

# prepare datas
image = Image.open(args.input_image).convert('RGB')
rects = json.load(open(args.panel_rectangle))
reference_images = [Image.open(path).convert('RGB') for path in args.reference_images]
assert len(rects) == len(reference_images)

# prepare pipeline
pipeline = comicolorization.pipeline.PagePipeline(
    drawer=drawer,
    drawer_sr=drawer_sr,
    image=image,
    reference_images=reference_images,
    threshold_binary=190,
    threshold_line=130,
    panel_rects=rects,
)

# draw
drawn_image = pipeline.process()
drawn_image.save(args.output)
