import chainer
import json
import numpy
import os
import typing

import comicolorization


class Drawer(object):
    """
    結果保存ディレクトリを指定して初期化することで、設定を取得してモデルを読み込み、色塗り用のメソッドを提供するクラス。
    """

    def __init__(self, path_result_directory, gpu):
        args_default = comicolorization.utility.config.get_default_train_args()
        args_train = json.load(open(os.path.join(path_result_directory, 'argument.json')))  # type: typing.Dict

        self.path_result_directory = path_result_directory
        self.gpu = gpu
        self.args_train = args_train
        self.target_iteration = None

        for k, v in args_default.items():
            args_train.setdefault(k, v)

        if args_train['network_model'] == 'SimpleConvolution':
            self.model = comicolorization.models.SimpleConvolution(loss_type=args_train['loss_type'])
        elif args_train['network_model'] == 'LTBC':
            self.model, reinput_model = comicolorization.utility.model.make_ltbc(args_train)
            assert len(reinput_model) == 0

    def _get_path_model(self, iteration):
        return os.path.join(self.path_result_directory, '{}.model'.format(iteration))

    def exist_save_model(self, iteration):
        path_model = self._get_path_model(iteration)
        return os.path.exists(path_model)

    def load_model(self, iteration):
        if not self.exist_save_model(iteration):
            print("warning! iteration {iteration} model is not found.".format(iteration=iteration))
            return False

        path_model = self._get_path_model(iteration)

        print("making iteration{}'s images...".format(iteration))
        chainer.serializers.load_npz(path_model, self.model)
        if self.gpu >= 0:
            chainer.cuda.get_device(self.gpu).use()  # Make the GPU current
            self.model.to_gpu()

        self.target_iteration = iteration
        return True

    @property
    def can_input_color_image(self):
        """
        3チャンネルの入力（部分塗り）に対応しているか
        """
        return self.args_train['max_pixel_drawing'] is not None

    def draw(
            self,
            input_images_array,
            rgb_images_array=None,
            histogram_image_array=None,
            histogram_array=None,
    ):
        """
        :param input_images_array: 入力画像。チャンネル数が3ならそのままforward、1なら白黒画像とみなしてチャンネル数3にpadding
        """
        if self.can_input_color_image and input_images_array.shape[1] == 1:
            input_images_array = comicolorization.utility.image.padding_channel_1to3(input_images_array)

        images = comicolorization.utility.image.draw(
            self.model,
            input_images_array, rgb_images_array,
            histogram_image_array, histogram_array,
            self.gpu,
        )
        return images
