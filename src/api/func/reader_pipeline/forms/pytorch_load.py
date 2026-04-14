# pytorch_load.py
from __future__ import annotations

from typing import Any, Callable
import numpy as np
import torch


def pytorchLoader(model_path: str, runtime_cfg: Any = None, logger=None) -> Callable[[Any], Any]:
    """
    Loader PyTorch (TorchScript):
      - Requiere modelos exportados con torch.jit.save() — autocontenidos, sin dependencia del codigo fuente.
      - GPU si device=='gpu' y CUDA disponible, fallback a CPU con warning.
      - Threads configurables para CPU.
      - Devuelve predict_fn(x) -> list[np.ndarray] (mismo contrato que ONNX loader).
    """
    runtime_cfg = runtime_cfg or object()

    device_req = (getattr(runtime_cfg, "device", "cpu") or "cpu").lower()

    cuda_available = torch.cuda.is_available()
    if device_req == "gpu":
        if cuda_available:
            device = torch.device("cuda")
        else:
            if logger:
                logger.warning(
                    "PyTorch: device='gpu' solicitado pero CUDA no esta disponible. "
                    "Fallback a CPU."
                )
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    threads_cfg = getattr(runtime_cfg, "threads", None)
    num_threads = getattr(threads_cfg, "num_threads", None) if threads_cfg else None
    if isinstance(num_threads, int) and num_threads > 0 and device.type == "cpu":
        torch.set_num_threads(num_threads)

    if logger:
        logger.info(f"PyTorch: version: {torch.__version__}")
        logger.info(f"PyTorch: CUDA disponible: {cuda_available}")
        if cuda_available:
            logger.info(f"PyTorch: GPU detectada: {torch.cuda.get_device_name(0)}")
        logger.info(f"PyTorch: device solicitado: {device_req.upper()}")
        logger.info(f"PyTorch: device efectivo: {device}")
        if device.type == "cpu":
            logger.info(f"PyTorch: num_threads={torch.get_num_threads()}")

    try:
        model = torch.jit.load(model_path, map_location=device)
    except RuntimeError as e:
        raise RuntimeError(
            f"PyTorch: no se pudo cargar '{model_path}' como TorchScript. "
            f"Asegurate de exportar el modelo con torch.jit.save(). "
            f"Error original: {e}"
        ) from e

    model.eval()

    if logger:
        logger.info(f"PyTorch: modelo cargado en {device} | modo eval activado")

    def predict_fn(x: Any):
        arr = np.asarray(x)
        tensor = torch.from_numpy(arr).to(device)

        with torch.no_grad():
            output = model(tensor)

        # Normalizar salida a list[np.ndarray] — mismo contrato que ONNX loader
        if isinstance(output, torch.Tensor):
            return [output.cpu().numpy()]
        if isinstance(output, (list, tuple)):
            return [t.cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
                    for t in output]
        return [np.asarray(output)]

    return predict_fn
