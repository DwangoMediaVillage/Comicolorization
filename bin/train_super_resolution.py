import argparse
import chainer
import os

from comicolorization_sr.colorization_task import ComicolorizationTask
from comicolorization_sr.config import Config
from comicolorization_sr import dataset
from comicolorization_sr.forwarder import Forwarder
from comicolorization_sr.loss import LossMaker
from comicolorization_sr.model import prepare_model
from comicolorization_sr.updater import Updater
from comicolorization_sr.trainer import create_optimizer, create_trainer
from comicolorization_sr import utility

parser = argparse.ArgumentParser()
parser.add_argument('config_json_path')
config_json_path = parser.parse_args().config_json_path

# load config
config = Config(config_json_path)
project_path = config.project_config.get_project_path()
if not os.path.exists(project_path):
    os.mkdir(project_path)

config.copy_config_json()
train_config = config.train_config

nb = train_config.batchsize
use_gpu = (train_config.gpu >= 0)

# setup trainer
if use_gpu:
    chainer.cuda.get_device(train_config.gpu).use()

# setup colorization
colorization = ComicolorizationTask(config)

# setup dataset
datasets = dataset.create(
    config.dataset_config,
    input_process=colorization.get_input_process(),
    concat_process=colorization.get_concat_process(),
)
IteratorClass = chainer.iterators.MultiprocessIterator
iterator_train = IteratorClass(datasets.train, nb, repeat=True, shuffle=True)
iterator_validation = IteratorClass(datasets.test, nb, repeat=False, shuffle=False)
iterator_train_validation = IteratorClass(datasets.train_varidation, nb, repeat=False, shuffle=False)

# setup model
model = prepare_model(config.model_config)
use_gpu and model.to_gpu()
models = {'main': model}

optimizer = create_optimizer(train_config, model, 'main')
optimizers = {'main': optimizer}

# setup forwarder
forwarder = Forwarder(config.model_config, colorizer=colorization.get_colorizer(), model=model)

# setup loss
loss_maker = LossMaker(config.loss_config, forwarder, model)

# setup updater
updater = Updater(
    optimizer=optimizers,
    iterator=iterator_train,
    loss_maker=loss_maker,
    device=train_config.gpu,
    converter=utility.chainer.converter_recursive,
)

# train
trainer = create_trainer(
    config=train_config,
    project_path=config.project_config.get_project_path(),
    updater=updater,
    model=models,
    eval_func=loss_maker.test,
    iterator_test=iterator_validation,
    iterator_train_varidation=iterator_train_validation,
    loss_names=loss_maker.get_loss_names(),
    converter=lambda *args: utility.chainer.change_volatile_variable_recursive(utility.chainer.converter_recursive(*args), volatile='on'),
)
trainer.run()
