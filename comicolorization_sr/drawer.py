import chainer
import os

from comicolorization_sr.colorization_task import BaseColorizationTask
from comicolorization_sr.config import Config
from comicolorization_sr.forwarder import Forwarder
from comicolorization_sr.model import prepare_model
from comicolorization_sr import utility


class Drawer(object):
    def __init__(self, path_result_directory, gpu, colorization_class):
        config_path = Config.get_config_path(path_result_directory)
        config = Config(config_path)

        self.model = None
        self.forwarder = None  # type: Forwarder

        self.path_result_directory = path_result_directory
        self.dataset_config = config.dataset_config
        self.model_config = config.model_config
        self.gpu = gpu

        # colorization
        self.colorization = colorization_class(config)  # type: BaseColorizationTask

    def _get_path_model(self, iteration):
        return os.path.join(self.path_result_directory, '{}.model'.format(iteration))

    def exist_save_model(self, iteration):
        path_model = self._get_path_model(iteration)
        return os.path.exists(path_model)

    def load_model(self, iteration):
        if not self.exist_save_model(iteration):
            print("warning! iteration {iteration} model is not found.".format(iteration=iteration))
            return False

        self.model = prepare_model(self.model_config)
        path_model = self._get_path_model(iteration)

        print("load {} ...".format(path_model))
        chainer.serializers.load_npz(path_model, self.model)
        if self.gpu >= 0:
            chainer.cuda.get_device(self.gpu).use()
            self.model.to_gpu(self.gpu)

        colorizer = self.colorization.get_colorizer()
        self.forwarder = Forwarder(self.model_config, model=self.model, colorizer=colorizer)
        return True

    def draw_raw(
            self,
            input,
            concat,
    ):
        device = self.gpu
        input = utility.chainer.to_variable_recursive(input, device=device, volatile=True)
        concat = utility.chainer.to_variable_recursive(concat, device=device, volatile=True)

        output = self.forwarder.forward(input, concat, test=True)['image']
        output.to_cpu()

        return utility.image.lab_array_to_image(output.data, normalized=True)

    def draw_only_super_pixel(
            self,
            image,
            concat,
    ):
        device = self.gpu
        image = utility.chainer.to_variable_recursive(image, device=device, volatile=True)
        concat = utility.chainer.to_variable_recursive(concat, device=device, volatile=True)

        output = self.forwarder.forward_super_pixel(image, concat, test=True)['image']
        output.to_cpu()

        return utility.image.lab_array_to_image(output.data)
