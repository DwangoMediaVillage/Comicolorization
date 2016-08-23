"""
プロジェクトのディレクトリにimages_with_trained_modelディレクトリを作成して、
test_dataset_pathで指定した画像から白黒画像を作成し、学習済みモデルに入力する。

（本物線画画像を使って色塗りする例）
python3 -i bin/make_image_with_trained_model.py \
    [target projects] \
    --path_root /path/to/result \
    --gpu -1 \
    --num_image 49 \
    --test_dataset_path /path/to/image/directory \
    --loss_type Lab \
    --binarization_input \
    --prefix_save_filename binarization_input_real_linedraw_ \
    --save_grayimage_color_normalized \
    --reference_image_path \
        [path to reference image 1].png \
        [path to reference image 2].png \

全部の学習済みモデルで画像を作ると時間がかかるので、適宜target_iterationを指定すると便利。
"""

import glob
import itertools
import json
import more_itertools
import os
import sys
import typing

import numpy

ROOT_PATH = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(ROOT_PATH)
import comicolorization

parser = comicolorization.utility.config.get_predict_parser()
args = parser.parse_args()

paths_result_directory = args.paths_result_directory
if args.path_root is not None:
    paths_result_directory = [os.path.join(args.path_root, relpath) for relpath in paths_result_directory]


def make_image_arrays(dataset, indexes):
    batch = [dataset[i] for i in indexes]

    color_image_list = [one_data[0] for one_data in batch]
    gray_image_list = [one_data[1] for one_data in batch]
    rgb_image_list = [one_data[2] for one_data in batch]

    color_images_array = numpy.asarray(color_image_list)
    gray_images_array = numpy.asarray(gray_image_list)
    rgb_image_list = numpy.asarray(rgb_image_list)

    return color_images_array, gray_images_array, rgb_image_list


def save_images_from_array(
        color_images_array,
        mode,
        gray_images_array,
        path_directory,
        prefix_filename,
        color_normalize=False,
        index_base=0,
):
    images_rgb = comicolorization.utility.image.array_to_image(
        color_images_array=color_images_array,
        gray_images_array=gray_images_array,
        mode=mode,
        color_normalize=color_normalize,
    )
    comicolorization.utility.image.save_images(
        images=images_rgb,
        path_directory=path_directory,
        prefix_filename=prefix_filename,
        index_base=index_base,
    )


for path_result_directory in paths_result_directory:
    print(path_result_directory)

    drawer = comicolorization.drawer.Drawer(
        path_result_directory=path_result_directory,
        gpu=args.gpu,
    )

    args_default = comicolorization.utility.config.get_default_train_args()
    args_train = json.load(open(os.path.join(path_result_directory, 'argument.json')))  # type: typing.Dict

    if args.loss_type is not None:
        loss_type = args.loss_type
    else:
        loss_type = args_train['loss_type']

    batchsize = args_train['batchsize']

    # make dataset
    if args.test_dataset_path is None:
        paths = glob.glob("{}/*".format(args_train['dataset_path']))
        random_state = numpy.random.RandomState(args_train['random_seed_test'])
        paths = random_state.permutation(paths)
        num_dataset_test = args_train['num_dataset_test']
    else:
        paths = glob.glob("{}/*".format(args.test_dataset_path))
        num_dataset_test = len(paths)

    # 線画化処理が一つしかなかった時の、線画化モードでモデルで回す時
    if 'use_line_drawing_mode' in args_train and args_train['use_line_drawing_mode']:
        line_drawing_mode = 'adaptive_threshold'
    # 線画化処理が複数になった最新モデルで回す時
    else:
        line_drawing_mode = args_train['line_drawing_mode']

    if args.direct_input:
        augmentation = False
        line_drawing_mode = None
        size_image = None
    else:
        augmentation = args_train['augmentation']
        size_image = [args_train['size_image'], args_train['size_image']]

    datasets = comicolorization.utility.dataset.choose_dataset(
        paths=paths,
        num_dataset_test=num_dataset_test,
        loss_type=loss_type,
        augmentation=augmentation,
        size_image_augmentation=[args_train['size_image_augmentation'], args_train['size_image_augmentation']],
        size_image=size_image,
        use_ltbc_classification=False,
        path_tag_list=None,
        path_tag_list_each_image=None,
        line_drawing_mode=line_drawing_mode,
        max_pixel_drawing=None,
        max_size_pixel_drawing=None,
        use_binarization_dataset=args.use_binarization_dataset,
    )

    test_dataset = datasets['test']

    histogram_dataset = []
    if args.reference_image_path is not None:
        histogram_dataset = comicolorization.dataset.LabImageDataset(args.reference_image_path)

    target_iteration = args.target_iteration
    if target_iteration is None:
        save_result_iteration = args_train['save_result_iteration']
        target_iteration = itertools.count(start=save_result_iteration, step=save_result_iteration)

    path_save_images = os.path.join(path_result_directory, args.dirname_save_images)
    if not os.path.exists(path_save_images):
        os.mkdir(path_save_images)

    # save raw colored image
    path_test_raw_images = os.path.join(path_save_images, 'test_raw_images')
    if not os.path.exists(path_test_raw_images):
        os.mkdir(path_test_raw_images)

    for indexes in more_itertools.chunked(range(0, args.num_image), batchsize):
        color_images_array, gray_images_array, _ = make_image_arrays(test_dataset, indexes)

        save_images_from_array(
            color_images_array=color_images_array,
            gray_images_array=gray_images_array,
            mode=loss_type,
            path_directory=path_test_raw_images,
            prefix_filename=args.prefix_save_filename + 'test_raw_image_',
            index_base=indexes[0],
        )

    # save input gray image
    path_test_input_images = os.path.join(path_save_images, 'test_input_images')
    if not os.path.exists(path_test_input_images):
        os.mkdir(path_test_input_images)

    for indexes in more_itertools.chunked(range(0, args.num_image), batchsize):
        _, gray_images_array, _ = make_image_arrays(test_dataset, indexes)

        save_images_from_array(
            color_images_array=None,
            gray_images_array=gray_images_array,
            mode='gray',
            path_directory=path_test_input_images,
            prefix_filename=args.prefix_save_filename + 'test_input_image_',
            color_normalize=False,
            index_base=indexes[0],
        )

        if args.save_grayimage_color_normalized:
            save_images_from_array(
                color_images_array=None,
                gray_images_array=gray_images_array,
                mode='gray',
                path_directory=path_test_input_images,
                prefix_filename=args.prefix_save_filename + 'test_input_image_normalized_',
                color_normalize=True,
                index_base=indexes[0],
            )

    # save reference colored image
    if args.reference_image_path is not None:
        path_test_reference_images = os.path.join(path_save_images, 'test_reference_images')
        if not os.path.exists(path_test_reference_images):
            os.mkdir(path_test_reference_images)

        for indexes in more_itertools.chunked(range(0, len(histogram_dataset)), batchsize):
            _, _, rgb_images_array = make_image_arrays(histogram_dataset, indexes)

            save_images_from_array(
                gray_images_array=None,
                color_images_array=rgb_images_array,
                mode='RGB',
                path_directory=path_test_reference_images,
                prefix_filename=args.prefix_save_filename + 'test_reference_image_',
                index_base=indexes[0],
            )

    # save made image
    for index_iteration in target_iteration:
        if not drawer.load_model(iteration=index_iteration):
            continue

        path_directory = os.path.join(path_save_images, 'test_{}'.format(index_iteration))

        # loop with input [test image] x [histograms + 1]
        indexes_chunked = more_itertools.chunked(range(0, args.num_image), batchsize)
        indexes_histogram = [None] + list(range(len(histogram_dataset)))
        indexes_rebalance_hist_rates = [None] + list(range(args.num_rebalance_hist_rate_split))
        for indexes, i_histogram, i_rebalance \
                in itertools.product(indexes_chunked, indexes_histogram, indexes_rebalance_hist_rates):
            prefix_reference = ''

            histogram = None
            if i_histogram is not None:
                prefix_reference = 'ref{}_'.format(i_histogram)

                _, _, histogram_image_array = make_image_arrays(histogram_dataset, [i_histogram])
                histogram = comicolorization.utility.image.make_histogram(
                    histogram_image_array[0],
                    num_bins=args_train['num_bins_histogram'],
                    multidim=args_train['use_multidimensional_histogram'],
                    threshold_palette=args_train['threshold_histogram_palette'],
                )
                if i_rebalance is not None:
                    prefix_reference += 'reb{}_'.format(i_rebalance)
                    rebalance_hist_rate = numpy.linspace(0, 2, num=args.num_rebalance_hist_rate_split)[i_rebalance]
                    histogram = comicolorization.utility.image.rebalance_top_histogram(
                        histogram,
                        rate=rebalance_hist_rate,
                    )
                histogram = numpy.repeat(histogram[numpy.newaxis], len(indexes), axis=0)

            _, input_images_array, rgb_images_array = make_image_arrays(test_dataset, indexes)
            input_images_array = input_images_array * args.scale_input

            images = drawer.draw(
                input_images_array=input_images_array,
                rgb_images_array=rgb_images_array,
                histogram_array=histogram,
            )

            comicolorization.utility.image.save_images(
                images=images,
                path_directory=path_directory,
                prefix_filename=args.prefix_save_filename + prefix_reference,
                index_base=indexes[0],
            )
