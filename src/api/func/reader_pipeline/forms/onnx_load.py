# onnx_load.py
from __future__ import annotations

import glob
import os

import onnxruntime as ort

# Bandera de una sola vez: preparar las DLL es idempotente pero no gratis, y
# onnxLoader se llama en cada carga de modelo.
_dlls_listas = False


def _preparar_dlls_nvidia(logger=None):
    """
    Deja a ORT en condiciones de usar la GPU con CUDA/cuDNN instalados POR PIP
    dentro del venv (paquetes nvidia-*-cu12 en site-packages), sin toolkit global.

    Son DOS cosas distintas y hacen falta las dos:

    1. os.environ["PATH"]: cudnn64_9.dll carga sus sublibs (cudnn_engines_*,
       cudnn_graph, cudnn_heuristic...) con LoadLibrary por NOMBRE SIMPLE, asi que
       solo las encuentra si su carpeta esta en el PATH del proceso.
       os.add_dll_directory() NO alcanza aca: probado, sigue sin encontrarlas.
    2. ort.preload_dlls(): ORT no carga las DLL de CUDA solo al importarse.

    Si falta cualquiera de las dos, ORT NO tira excepcion al crear la sesion: se
    cae a CPU en pleno run ("Falling back to CPUExecutionProvider"). Medido: el
    clasificador pasa de 9 ms a 63 ms. Por eso abajo se loguean los providers
    REALES de la sesion, que es lo unico que dice la verdad.
    """
    global _dlls_listas
    if _dlls_listas:
        return
    try:
        import nvidia  # paquete namespace de los wheels nvidia-*-cu12
        base = os.path.dirname(nvidia.__file__)
        bins = [d for d in glob.glob(os.path.join(base, "*", "bin")) if os.path.isdir(d)]
        if bins:
            os.environ["PATH"] = os.pathsep.join(bins) + os.pathsep + os.environ.get("PATH", "")
    except ImportError:
        # Sin wheels de nvidia: CUDA puede venir de un toolkit del sistema (o no
        # haber GPU). No es un error; se sigue y los providers lo diran.
        pass

    # getattr: preload_dlls no existe antes de ORT ~1.21; en runtimes viejos se saltea.
    _preload = getattr(ort, "preload_dlls", None)
    if callable(_preload):
        try:
            _preload()
        except Exception as e:
            if logger:
                logger.warning(f"ort.preload_dlls() fallo, se sigue sin precargar: {e}")

    _dlls_listas = True


def onnxLoader(model_path: str, runtime_cfg, logger=None):
    _preparar_dlls_nvidia(logger)

    so = ort.SessionOptions()

    try:
        threads = getattr(runtime_cfg, "threads", None)
        if threads is not None:
            intra = getattr(threads, "intra_op", None)
            inter = getattr(threads, "inter_op", None)

            if isinstance(intra, int) and intra > 0:
                so.intra_op_num_threads = intra
            if isinstance(inter, int) and inter > 0:
                so.inter_op_num_threads = inter

        available = ort.get_available_providers()

        onnx_cfg = getattr(runtime_cfg, "onnx", None)
        cfg_providers = getattr(onnx_cfg, "providers", None) if onnx_cfg else None

        device = (getattr(runtime_cfg, "device", "cpu") or "cpu").lower()

        if cfg_providers:
            desired = list(cfg_providers)
        else:
            if device == "gpu":
                desired = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                desired = ["CPUExecutionProvider"]

        providers = [p for p in desired if p in available]

        if not providers:
            providers = ["CPUExecutionProvider"]

        provider_options = None
        cfg_opts = getattr(onnx_cfg, "provider_options", None) if onnx_cfg else None
        if isinstance(cfg_opts, dict) and cfg_opts:
            provider_options = [cfg_opts.get(p, {}) for p in providers]

        if provider_options is not None:
            session = ort.InferenceSession(model_path, sess_options=so,
                                        providers=providers, provider_options=provider_options)
        else:
            session = ort.InferenceSession(model_path, sess_options=so, providers=providers)

        input_name = session.get_inputs()[0].name

        logger.info(f"ORT available providers: {available}")
        logger.info(f"ORT desired providers: {desired}")
        logger.info(f"ORT session providers: {session.get_providers()}")


    except Exception as e:
        if logger:
            logger.exception(e)
        raise

    def predict_fn(x):
        # devuelve numpy arrays - NO tolist
        return session.run(None, {input_name: x})

    return predict_fn
