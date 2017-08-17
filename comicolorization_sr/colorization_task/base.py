from abc import ABCMeta, abstractmethod
import typing

from comicolorization_sr.config import Config
from comicolorization_sr.data_process import BaseDataProcess


class BaseColorizationTask(metaclass=ABCMeta):
    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def get_input_process(self) -> BaseDataProcess:
        pass

    @abstractmethod
    def get_concat_process(self) -> BaseDataProcess:
        pass

    @abstractmethod
    def get_colorizer(self) -> typing.Callable[[typing.Any, bool], typing.Any]:
        pass
