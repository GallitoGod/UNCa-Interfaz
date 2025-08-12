from .config_schema import OutputConfig
from typing import Callable

class Reactive_output_config(OutputConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._on_change_callback = None

    def set_on_change(self, callback: Callable):
        self._on_change_callback = callback

    @property
    def confidence_threshold(self):
        return self.__dict__['confidence_threshold']

    @confidence_threshold.setter
    def confidence_threshold(self, value):
        if self.__dict__['confidence_threshold'] != value:
            self.__dict__['confidence_threshold'] = value
            if self._on_change_callback:
                self._on_change_callback()
