from abc import ABC, abstractmethod
from typing import Any

class GeneralModel(ABC):

    @staticmethod
    @abstractmethod
    def loader(model_path: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def input_adapter(input_data: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def output_adapter(inference_data: Any) -> Any:
        pass

