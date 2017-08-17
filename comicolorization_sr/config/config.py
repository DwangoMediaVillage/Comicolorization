import json
import os
import shutil
import typing


class Config(object):
    @staticmethod
    def get_config_path(project_path):
        return os.path.join(project_path, 'config.json')

    def __init__(self, path_json):
        self.path_json = path_json
        self.config = json.load(open(path_json, encoding='utf-8'))

        self.dataset_config = DatasetConfig(self.config.get('dataset'))
        self.loss_config = LossConfig(self.config.get('loss'))
        self.model_config = ModelConfig(self.config.get('model'))
        self.train_config = TrainConfig(self.config.get('train'))
        self.project_config = ProjectConfig(self.config.get('project'))

        project_path = self.project_config.get_project_path()
        os.path.exists(project_path) or os.mkdir(project_path)

    def copy_config_json(self):
        project_path = self.project_config.get_project_path()
        config_path = self.get_config_path(project_path)
        shutil.copy(self.path_json, config_path)


class DatasetConfig(object):
    def __init__(self, config):
        self.seed_evaluation = config.get('seed_evaluation')
        self.num_test = config.get('num_test')
        self.images_glob = config.get('images_glob')
        self.scale_input = config.get('scale_input')  # type: float
        self.target_width = config.get('target_width')  # type: int


class LossConfig(object):
    def __init__(self, config):
        self.name = config.get('name')
        self.blend = config.get('blend')


class ModelConfig(object):
    def __init__(self, config):
        self.name = config.get('name')
        self.scale = config.get('scale')  # type: int
        self.colorizer_name = config.get('colorizer_name')
        self.other = config.get('other')  # type: typing.Dict


class TrainConfig(object):
    def __init__(self, config):
        self.batchsize = config.get('batchsize')
        self.gpu = config.get('gpu')
        self.log_iteration = config.get('log_iteration')
        self.save_iteration = config.get('save_iteration')
        self.optimizer = config.get('optimizer')  # type: typing.Dict


class ProjectConfig(object):
    def __init__(self, config):
        self.name = config.get('name')
        self.result_path = config.get('result_path')
        self.tags = config.get('tags')
        self.comment = config.get('comment')

    def get_project_path(self):
        return os.path.join(self.result_path, self.name)
