# api/func/reader_pipeline/output_shape.py
# Fase 4 tarea 3 (columna izquierda de la vista de mapeo): reporta la forma de los
# tensores de SALIDA del modelo SIN armar el pipeline completo. ONNX/TFLite/Keras
# exponen las shapes estaticamente; TorchScript NO (haria falta un forward dummy),
# asi que para .pt/.pth se devuelve available=False y el frontend cae al ingreso
# manual (estrategia hibrida elegida por el usuario).
from __future__ import annotations
import os


def introspect_output_shapes(model_path: str) -> dict:
    """Devuelve un dict JSON-serializable:
        {
          "backend": str,            # onnxruntime | tflite | keras | pytorch | desconocido
          "available": bool,         # True si se pudo leer la(s) shape(s)
          "shapes": [[...], ...] | None,  # una lista por tensor de salida
          "detail": str,             # 'ok' o el motivo de no estar disponible
        }
    Las dims dinamicas pueden venir como string (ONNX, ej: 'batch') o -1.
    Nunca lanza: ante cualquier fallo devuelve available=False con el detalle.
    """
    ext = os.path.splitext(model_path)[1].lower()
    try:
        if ext == ".onnx":
            return _onnx_shapes(model_path)
        if ext == ".tflite":
            return _tflite_shapes(model_path)
        if ext in (".h5", ".keras"):
            return _keras_shapes(model_path)
        if ext in (".pt", ".pth"):
            return {
                "backend": "pytorch",
                "available": False,
                "shapes": None,
                "detail": "TorchScript no expone shapes de salida sin ejecutar; ingresar a mano.",
            }
        return {
            "backend": "desconocido",
            "available": False,
            "shapes": None,
            "detail": f"extension '{ext}' no soportada para introspeccion.",
        }
    except Exception as e:  # introspeccion best-effort: nunca rompe el endpoint
        return {
            "backend": ext.lstrip(".") or "desconocido",
            "available": False,
            "shapes": None,
            "detail": f"no se pudo inspeccionar: {e}",
        }


def _onnx_shapes(model_path: str) -> dict:
    import onnxruntime as ort

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    shapes = [list(o.shape) for o in sess.get_outputs()]
    return {"backend": "onnxruntime", "available": True, "shapes": shapes, "detail": "ok"}


def _tflite_shapes(model_path: str) -> dict:
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    shapes = [list(map(int, od["shape"])) for od in interp.get_output_details()]
    return {"backend": "tflite", "available": True, "shapes": shapes, "detail": "ok"}


def _keras_shapes(model_path: str) -> dict:
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path)
    outs = model.outputs
    if not isinstance(outs, (list, tuple)):
        outs = [outs]
    # dims None (batch dinamico) -> -1 para que sea JSON-serializable
    shapes = [[(int(d) if d is not None else -1) for d in tuple(o.shape)] for o in outs]
    return {"backend": "keras", "available": True, "shapes": shapes, "detail": "ok"}
