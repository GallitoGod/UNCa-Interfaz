
import logging
import os

def setup_model_logger(model_name: str, log_dir: str = "logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{model_name}.log")

    logger = logging.getLogger(model_name)
    #logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

#   ETAPAS A LOGUEAR:

# | Zona del pipeline        | Nivel     | Que loguear                                                     |
# | ------------------------ | --------- | --------------------------------------------------------------- |
# | `ModelLoader`            | `INFO`    | Modelo cargado con éxito (nombre, formato, tipo)                |
# |                          | `ERROR`   | Fallo al cargar modelo (formato incompatible, ruta inexistente) |
# | `ConfigLoader` (JSON)    | `INFO`    | Config cargada correctamente                                    |
# |                          | `ERROR`   | Campo faltante o inválido (con Pydantic)                        |
# | `Preprocessor`           | `INFO`    | Transformación aplicada, tamaño de imagen final                 |
# |                          | `WARNING` | Transformación no compatible o parámetro ignorado               |
# | `InferenceEngine`        | `INFO`    | Inicio/fin de inferencia, tiempo                                |
# |                          | `ERROR`   | Fallo en inferencia (shape incorrecto, dtype, etc)              |
# | `Unpacker / Adapter`     | `INFO`    | Shape esperado vs recibido                                      |
# |                          | `ERROR`   | No se puede desempaquetar la salida                             |
# | `Postprocessor`          | `INFO`    | Salida final transformada                                       |
# |                          | `ERROR`   | Fallo en NMS, umbral inválido, etc                              |
