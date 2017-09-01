import chainer
import json
import numpy
import os
from skimage.color import rgb2lab
from skimage.filters import threshold_otsu
import typing

import comicolorization
from comicolorization_sr.colorization_task import BaseColorizationTask
from comicolorization_sr.config import Config
from comicolorization_sr import dataset


class ComicolorizationTask(BaseColorizationTask):
    def __init__(self, config, load_model=True):
        # type: (Config, any) -> None
        super(ComicolorizationTask, self).__init__(config, load_model)

        path_result_directory = config.model_config.other['path_result_directory']
        args_default = comicolorization.utility.config.get_default_train_args()
        args_train = json.load(open(os.path.join(path_result_directory, 'argument.json')))  # type: typing.Dict

        self._path_result_directory = path_result_directory
        self._args_train = args_train

        self.model = None

        for k, v in args_default.items():
            args_train.setdefault(k, v)

        if load_model:
            self._load_model()

    def _load_model(self):
        # make model
        model, reinput = comicolorization.utility.model.make_ltbc(self._args_train)
        assert reinput is None or len(reinput) == 0
        self.model = model

        # load model
        iteration = self.config.model_config.other['iteration']
        path_model = os.path.join(self._path_result_directory, '{}.model'.format(iteration))
        chainer.serializers.load_npz(path_model, model)

        use_gpu = (self.config.train_config.gpu >= 0)
        if use_gpu:
            model.to_gpu()

    def get_input_process(self):
        return dataset.ChainProcess([
            LabOtsuThresholdImageProcess(),
            LabSeveralPixelDrawingImageProcess(
                max_point=self._args_train['max_pixel_drawing'],
                max_size=self._args_train['max_size_pixel_drawing'],
            ),
            InputAdapterProcess(),
        ])

    def get_concat_process(self):
        return dataset.ChainProcess([
            LabOtsuThresholdImageProcess(),
            ConcatAdapterProcess(),
        ])

    def _colorizer(self, input, test):
        model = self.model

        image_input, image_rgb = input['image_input'], input['image_rgb']

        if self._args_train['use_histogram_network']:
            outputs = model(image_input, x_rgb=image_rgb, test=test)
        else:
            outputs = model(image_input, test=test)

        if self._args_train['alpha_ltbc_classification'] is None:
            output_color = outputs
        else:
            output_color, _ = outputs

        output_color /= 100  # normalize
        return output_color

    def get_colorizer(self):
        return self._colorizer


class LabOtsuThresholdImageProcess(dataset.BaseDataProcess):
    def __call__(self, image, test):
        image_data = numpy.asarray(image, dtype=numpy.float32)[:, :, :3]
        rgb_image_data = image_data.transpose(2, 0, 1)
        lab_image_data = rgb2lab(image_data / 255).transpose(2, 0, 1).astype(numpy.float32)
        luminous_image_data = lab_image_data[0].astype(numpy.uint8)

        try:
            th = threshold_otsu(luminous_image_data)
        except:
            import traceback
            print(traceback.format_exc())
            th = 0

        linedrawing = (luminous_image_data > th).astype(numpy.float32)
        linedrawing = numpy.expand_dims(linedrawing, axis=0)

        return lab_image_data, linedrawing, rgb_image_data


class LabSeveralPixelDrawingImageProcess(dataset.BaseDataProcess):
    def __init__(self, max_point, max_size):
        # type: (int, int) -> None
        self.max_point = max_point
        self.max_size = max_size

    def __call__(self, data, test):
        lab_image_data, linedrawing, rgb_image_data = data
        color_linedrawing = numpy.pad(linedrawing, ((0, 2), (0, 0), (0, 0)), mode='constant', constant_values=0)

        width, height = linedrawing.shape[1:]

        if not test:
            num_point = numpy.random.randint(low=0, high=self.max_point)
            size_point = numpy.random.randint(low=0, high=self.max_size) + 1

            top_points = numpy.asarray([
                [x, y]
                for x in range(width - (size_point - 1))
                for y in range(height - (size_point - 1))
            ])
            top_points = numpy.random.permutation(top_points)[:num_point]

            expanded_points_list = [
                top_points + numpy.array([x, y])
                for x in range(size_point)
                for y in range(size_point)
            ]
            points = numpy.concatenate(expanded_points_list)
        else:
            nx = ny = 3
            xs = numpy.linspace(0, width - 1, nx + 2, dtype=numpy.int32)[1:-1]
            ys = numpy.linspace(0, width - 1, ny + 2, dtype=numpy.int32)[1:-1]
            points = numpy.asarray([[x, y] for x in xs for y in ys])

        for x, y in points:
            linedrawing_luminance = self.get_input_luminance_range()
            lab_luminance = self.get_output_range()[0]
            color_linedrawing[0, x, y] = comicolorization.utility.color.normalize(
                lab_image_data[0, x, y],
                in_min=lab_luminance[0], in_max=lab_luminance[1],
                out_min=linedrawing_luminance[0], out_max=linedrawing_luminance[1],
            )
            color_linedrawing[1:, x, y] = lab_image_data[1:, x, y]

        return lab_image_data, color_linedrawing, rgb_image_data

    def get_input_luminance_range(self):
        return (0, 1)

    def get_output_range(self):
        return tuple(
            (comicolorization.utility.color.lab_min_max[0][i], comicolorization.utility.color.lab_min_max[1][i])
            for i in range(3)
        )


class InputAdapterProcess(dataset.BaseDataProcess):
    def __call__(self, data, test):
        lab_image_data, linedrawing, rgb_image_data = data
        return {
            'image_input': linedrawing,
            'image_rgb': rgb_image_data,
        }


class ConcatAdapterProcess(dataset.BaseDataProcess):
    def __call__(self, data, test):
        lab_image_data, linedrawing, rgb_image_data = data
        return linedrawing
