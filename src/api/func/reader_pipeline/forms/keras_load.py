# keras_load.py
from __future__ import annotations

from typing import Any, Callable, Optional
import tensorflow as tf


def _configure_tf_runtime(logger=None, enable_memory_growth: bool = True):
    """
    Loggea GPUs visibles y (opcional) habilita memory growth.
    IMPORTANTE: esto debe ejecutarse ANTES de que TF inicialice el runtime GPU "de verdad".
    Si ya se uso TF antes, puede tirar RuntimeError al setear memory growth.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if logger:
        logger.info(f"TF: GPUs fisicas: {gpus}")

    if enable_memory_growth and gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                if logger:
                    logger.info(f"TF: crecimiento de memoria habilitado para {gpu.name}")
            except Exception as e:
                # Suele fallar si TF ya inicializo el runtime. Lo digo por experiencia :(
                if logger:
                    logger.warning(f"TF: no se pudo establecer el crecimiento de la memoria para {gpu.name}: {e}")

    if logger:
        lgpus = tf.config.list_logical_devices("GPU")
        logger.info(f"TF: GPUs logicas: {lgpus}")


def _log_device_placement(model: tf.keras.Model, logger=None):
    """
    Best-effort: intenta inferir en que device cae una operacion simple.
    No es 100% garantia, pero ayuda a diagnosticar.
    """
    if not logger:
        return

    try:
        # Crear un tensor dummy pequeño solo para ver placement
        x = tf.zeros((1, 1), dtype=tf.float32)
        y = model(x, training=False) if callable(model) else x

        # Intenta de leer device de un tensor resultante
        dev = getattr(y, "device", None)
        logger.info(f"TF: ejemplo de dispositivo del tensor: {dev}")
    except Exception as e:
        logger.info(f"TF: se omitio la verificacion de la ubicacion del dispositivo: {e}")


def kerasLoader(model_path: str, runtime_cfg: Any = None, logger=None) -> Callable[[Any], Any]:
    """
    Loader Keras/TF:
      - loggea GPUs visibles
      - (opcional) memory growth si hay GPU
      - devuelve predict_fn(x) que retorna np/TF tensors (NO list)

    runtime_cfg (si existe) puede traer algo como:
      runtime_cfg.tf.enable_memory_growth: bool     [ SPOILER NO EXISTE ]
    """
    runtime_cfg = runtime_cfg or object()

    # config opcional desde runtime_cfg.tf
    tf_cfg = getattr(runtime_cfg, "tf", None)   # Tampoco existe, capaz algun dia lo creo.
    enable_memory_growth = True
    if tf_cfg is not None:
        v = getattr(tf_cfg, "enable_memory_growth", None)
        if isinstance(v, bool):
            enable_memory_growth = v

    _configure_tf_runtime(logger=logger, enable_memory_growth=enable_memory_growth)

    model = tf.keras.models.load_model(model_path)

    @tf.function
    def _forward(x):
        return model(x, training=False)

    if logger:
        try:
            logger.info(f"TF: construido con CUDA: {tf.test.is_built_with_cuda()}")
        except Exception:
            pass
        _log_device_placement(model, logger=logger)

    def predict_fn(x):
        return _forward(x)
        '''
            Esto de aca parece cualquier cosa y que no va a andar, pero es un tremendo salto de optimisacion a comparacion
        con el caso anterior:
                predict_fn = lambda img: model.predict(img).tolist()
            La forma actual no genera overhead, no maneja loops, tampoco batchings ni callbacks internos ya que predict esta 
            pensado para funcionar con batches, no para un frame por llamada.
        '''
    

    return predict_fn
