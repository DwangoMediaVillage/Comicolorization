from abc import ABCMeta, abstractmethod
import typing
import six

from comicolorization_sr.config import Config
from comicolorization_sr.data_process import BaseDataProcess

@six.add_metaclass(ABCMeta)
class BaseColorizationTask(object):
    def __init__(self, config, load_model=True):
        # type: (Config, any) -> None
        self.config = config
        self.load_model = load_model

    @abstractmethod
    def get_input_process(self):
        # type: (any) -> BaseDataProcess
        pass

    @abstractmethod
    def get_concat_process(self):
        # type: (any) -> BaseDataProcess
        pass

    @abstractmethod
    def get_colorizer(self):
        # type: (any) -> typing.Callable[[typing.Any, bool], typing.Any]
        pass
