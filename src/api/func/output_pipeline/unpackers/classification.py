# api/func/output_pipeline/unpackers/classification.py
"""
Unpackers de CLASIFICACION: tensor crudo del modelo -> vector de puntajes por clase.

CONTRATO DE FORMA (el analogo del (N,K) de deteccion):
    ndarray 1D float32 de largo C, donde C = cantidad de clases del modelo.
    Cada posicion es el puntaje de esa clase. Si el pack_format implica una
    activacion, aca ya viene aplicada. El postprocesador
    (build_classification_postprocessor) recibe SIEMPRE este vector.

QUIEN DECIDE LA ACTIVACION (la regla importante: evita aplastar los puntajes
dos veces, que es el error clasico de este pipeline):

    1) tensor_structure.output_format == "probabilities"
       El modelo YA emite probabilidades. NO se aplica ninguna activacion, sea
       cual sea el pack_format. Volver a pasar softmax sobre probabilidades no
       "normaliza": achata todo hacia 1/C y destruye la informacion.

    2) tensor_structure.output_format == "logits"
       Se aplica la activacion que nombra el pack_format:
           softmax_out -> softmax   (clases mutuamente excluyentes)
           sigmoid_out -> sigmoide  (multi-etiqueta: cada clase es independiente)
           logits_raw  -> ninguna   (se devuelven los logits crudos tal cual)

NOTA sobre output.apply_softmax / output.apply_sigmoid: son REDUNDANTES con
pack_format (dicen lo mismo con otro nombre) y NO se usan aca para decidir. La
coherencia entre ambos se chequea al armar el pipeline
(tasks/classification.py), que es donde hay logger para avisar. Quedan como
candidatos a poda del schema, igual que se hizo con el flag muerto 'quantized'.
"""
from __future__ import annotations

import numpy as np


def _to_score_vector(raw_output) -> np.ndarray:
    """
    Normaliza la salida cruda del backend a un vector 1D (C,) float32.

    Acepta lo que devuelven los distintos loaders:
      - ONNX: list de ndarrays -> [ (1, C) ]
      - TFLite: ndarray o tuple
      - Keras/TF: tensor
      - PyTorch: list
    Y las formas tipicas: (1, C), (C,), (1, 1, C).

    Falla ruidosamente si hay mas de un tensor: elegir "cual es el de clases"
    a ojo es exactamente el tipo de silencio que este proyecto evita.
    """
    arr = raw_output

    if isinstance(arr, (list, tuple)):
        if len(arr) == 0:
            raise ValueError("clasificacion: el modelo no devolvio ningun tensor.")
        if len(arr) > 1:
            raise ValueError(
                f"clasificacion: el modelo devolvio {len(arr)} tensores y no se "
                "puede saber cual es el de clases. Un clasificador debe exponer "
                "una sola salida (revisar la exportacion del modelo)."
            )
        arr = arr[0]

    arr = np.asarray(arr, dtype=np.float32)

    # (1, C) / (1, 1, C) -> (C,). squeeze saca TODAS las dimensiones de tamano 1.
    arr = np.squeeze(arr)

    if arr.ndim == 0:            # modelo de 1 sola clase: (1,) -> escalar
        arr = arr.reshape(1)

    if arr.ndim != 1:
        raise ValueError(
            f"clasificacion: se esperaba un vector de clases (C,) pero el tensor "
            f"desempaquetado tiene shape {arr.shape}. Si el modelo emite un batch "
            "de mas de 1 imagen, no es compatible con el stream (1 frame por vez)."
        )

    return arr


def _softmax(v: np.ndarray) -> np.ndarray:
    """Softmax estable (resta el maximo antes de exponenciar para no desbordar)."""
    m = v - v.max()
    np.exp(m, out=m)
    return m / (m.sum() + 1e-12)


def _sigmoid(v: np.ndarray) -> np.ndarray:
    """Sigmoide por clase (multi-etiqueta: las clases no compiten entre si)."""
    return 1.0 / (1.0 + np.exp(-v))


def _build_with_activation(output_cfg, activation):
    """
    Factory comun de los tres unpackers. 'activation' es la funcion a aplicar
    cuando el modelo emite logits, o None si el pack_format no activa nada.
    """
    ts = getattr(output_cfg, "tensor_structure", None)
    already_probabilities = (
        getattr(ts, "output_format", "logits") == "probabilities" if ts else False
    )

    def fn(raw_output, sh=None) -> np.ndarray:
        # sh (runtime) no se usa: la clasificacion no tiene geometria que escalar.
        # Se acepta el parametro para respetar la firma comun de los unpackers.
        scores = _to_score_vector(raw_output)

        if already_probabilities or activation is None:
            return scores

        return activation(scores).astype(np.float32, copy=False)

    return fn


def build_softmax_out(output_cfg):
    """Clases mutuamente excluyentes: softmax sobre los logits."""
    return _build_with_activation(output_cfg, _softmax)


def build_sigmoid_out(output_cfg):
    """Multi-etiqueta: sigmoide por clase sobre los logits."""
    return _build_with_activation(output_cfg, _sigmoid)


def build_logits_raw(output_cfg):
    """Sin activacion: el consumidor recibe los logits tal como salen del modelo."""
    return _build_with_activation(output_cfg, None)
