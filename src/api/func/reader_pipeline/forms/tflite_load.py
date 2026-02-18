# tflite_load.py
from __future__ import annotations

from typing import Optional, Sequence, Any, Callable, List
import numpy as np
import tensorflow as tf


def _try_load_gpu_delegate(logger=None):
    """
    Intenta cargar el GPU delegate de TFLite (best-effort).
    En Windows muchas veces NO esta disponible en la instalacion estandar de TF.
    """
    candidates = [
        # Windows (nombres comunes; dependen del build/distribucion)
        "tensorflowlite_gpu_delegate.dll",
        "libtensorflowlite_gpu_delegate.dll",
        # Linux / WSL
        "libtensorflowlite_gpu_delegate.so",
        # macOS
        "libtensorflowlite_gpu_delegate.dylib",
    ]

    for lib in candidates:
        try:
            delegate = tf.lite.experimental.load_delegate(lib)
            if logger:
                logger.info(f"TFLite: GPU delegate cargado: {lib}")
            return delegate
        except Exception:
            continue

    if logger:
        logger.warning("TFLite: GPU delegate no disponible (fallback CPU).")
    return None


def tfliteLoader(model_path: str, runtime_cfg: Any = None, logger=None) -> Callable[[Any], Any]:
    """
    Crea un interpreter de TFLite con:
      - num_threads desde config (siempre que exista)
      - GPU delegate si device=='gpu' o si tflite.delegates incluye 'gpu' (best-effort)
      - Fallback a CPU si no se puede cargar el delegate

    Devuelve predict_fn(x) -> np.ndarray o tuple(np.ndarray, ...)
    """

    runtime_cfg = runtime_cfg or object()

    device = (getattr(runtime_cfg, "device", "cpu") or "cpu").lower()

    threads_cfg = getattr(runtime_cfg, "threads", None)
    num_threads = getattr(threads_cfg, "num_threads", None) if threads_cfg else None
    if not (isinstance(num_threads, int) and num_threads > 0):
        num_threads = None

    tflite_cfg = getattr(runtime_cfg, "tflite", None)
    delegates_req = list(getattr(tflite_cfg, "delegates", []) or []) if tflite_cfg else []

    want_gpu = (device == "gpu") or any(str(d).lower() == "gpu" for d in delegates_req)

    delegates = None
    gpu_enabled = False
    if want_gpu:
        d = _try_load_gpu_delegate(logger=logger)
        if d is not None:
            delegates = [d]
            gpu_enabled = True

    interpreter = tf.lite.Interpreter(
        model_path=str(model_path),
        experimental_delegates=delegates,
        num_threads=num_threads,
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if not input_details:
        raise RuntimeError("TFLite: el modelo no tiene inputs detectables (input_details vacio).")

    in0 = input_details[0]
    in_index = in0["index"]
    in_shape = tuple(in0["shape"])
    in_dtype = in0["dtype"]

    out_indices: List[int] = [od["index"] for od in output_details]
    single_output = (len(out_indices) == 1) # Cache indices de salida (hot path)

    expects_batch1 = (len(in_shape) >= 1 and in_shape[0] == 1)  # Heuristica

    if logger:
        logger.info(
            f"TFLite loaded | device_req={device} | gpu_delegate={'ON' if gpu_enabled else 'OFF'} "
            f"| num_threads={num_threads} | input_shape={in_shape} | input_dtype={in_dtype} "
            f"| outputs={len(out_indices)}"
        )
        logger.info("TFLite: XNNPACK detect: unknown/best-effort (depende del build de TF/TFLite).")

    def tflite_predict(x: Any):
        arr = np.asarray(x)

        if expects_batch1 and tuple(arr.shape) == in_shape[1:]:
            arr = arr[None, ...]

        if tuple(arr.shape) != in_shape:
            raise ValueError(f"TFLite: shape {arr.shape} != esperada {in_shape}")

        if arr.dtype != in_dtype:
            arr = arr.astype(in_dtype, copy=False)

        interpreter.set_tensor(in_index, arr)
        interpreter.invoke()

        if single_output:
            return interpreter.get_tensor(out_indices[0])

        # Multi-output: devuelve tuple de np.ndarray
        return tuple(interpreter.get_tensor(i) for i in out_indices)

    return tflite_predict

