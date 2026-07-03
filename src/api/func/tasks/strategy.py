# tasks/strategy.py — la abstraccion central del seam.
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class TaskStrategy:
    """
    Bundle inmutable que ata, por model_type, todo lo que depende del tipo de modelo.

    Inmutable a proposito: los pasos del pipeline siguen siendo closures stateless
    (no se guarda estado por-frame), preservando la propiedad de la reforma 8.

    Campos:
      - task:           "detection" | "classification" | "segmentation".
      - build_pipeline: (config, model_path, logger) -> runner
                        runner es (img, debug=False) -> (result, timings):
                          result  = resultado de dominio (deteccion: ndarray (N,6)).
                          timings = dict {pre_ms, inf_ms, post_ms} para el PerfMeter.
      - serialize:      (result) -> dato JSON-listo para el campo 'result' del envelope.
    """
    task: str
    build_pipeline: Callable
    serialize: Callable
