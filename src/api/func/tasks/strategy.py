# tasks/strategy.py — la abstraccion central del seam.
from dataclasses import dataclass
from typing import Callable, Optional


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
                          result  = resultado de dominio segun el tipo:
                                      deteccion      -> sv.Detections (supervision)
                                      clasificacion  -> ndarray (K,2) [class_id, score]
                          timings = dict {pre_ms, inf_ms, post_ms} para el PerfMeter.
      - serialize:      (result) -> dato JSON-listo para el campo 'result' del envelope.
      - output_kind:    "json"  -> la respuesta del WS es el envelope de texto (clasificacion
                                   y TODOS los errores).
                        "frame" -> la respuesta del WS es BINARIA: el frame ya compuesto
                                   por el backend (deteccion, y segmentacion cuando exista).
                        El WS despacha por output_kind y NO por task: agregar un tipo nuevo
                        sigue siendo "registrar una estrategia", sin que el handler crezca
                        un if por tipo.
      - render:         (result, img_bgr, draw_cfg) -> bytes (JPEG compuesto).
                        Obligatorio si output_kind == "frame"; None si es "json".
                        Recibe el img_bgr YA decodificado que el handler tiene en la mano,
                        asi que no hay un decode extra por frame.
    """
    task: str
    build_pipeline: Callable
    serialize: Callable
    output_kind: str = "json"
    render: Optional[Callable] = None
