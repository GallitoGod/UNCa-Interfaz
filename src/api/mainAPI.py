from collections import deque
import asyncio
import base64
import datetime
import json
import re
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
import numpy as np
import cv2
from pathlib import Path
from api.func.model_controller import ModelController
from api.func.reader_pipeline.config_schema import (
    ModelConfig,
    build_config_template,
    anchor_defaults,
)

# Rutas absolutas relativas a este archivo (src/api/mainAPI.py → ../../)
_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = _ROOT / "models"
CONFIGS_DIR = _ROOT / "configs"

MODEL_EXTENSIONS = {".onnx", ".tflite", ".h5", ".keras", ".pt", ".pth"}
# Orden de preferencia cuando un basename tiene varios archivos (ej: yolo.onnx + yolo.tflite)
_EXTENSION_PREFERENCE = [".onnx", ".tflite", ".h5", ".keras", ".pt", ".pth"]

app = FastAPI(
    title="UNCaLens — Sistema de Vision por Computadora",
    description=(
        "API para carga de modelos de deteccion y ejecucion de inferencias sobre "
        "imagenes y video. El streaming va por WebSocket `/video_stream`: el cliente "
        "envia frames JPEG binarios y recibe JSON con las detecciones "
        "`[x1, y1, x2, y2, conf, cls]` en pixeles de la imagen original. "
        "El dibujo de cajas es responsabilidad del cliente."
    ),
    version="2",
)

# Electron carga desde file:// — sin esto todos los fetch() y WS fallan por CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

controller = ModelController()

# Ultimos 50 errores de inferencia (in-memory)
_inference_errors: deque = deque(maxlen=50)


class ModelPathRequest(BaseModel):
    model_path: str = Field(description="Ruta absoluta o relativa al archivo del modelo (.onnx, .tflite, .h5, .keras, .pt, .pth)")


class SelectModelRequest(BaseModel):
    model_name: str = Field(description="Nombre base del modelo, sin extension. Debe existir models/<nombre>.* y configs/<nombre>.json")


class ConfidenceUpdateRequest(BaseModel):
    value: float = Field(ge=0.0, le=1.0, description="Umbral de confianza en [0, 1]. Se aplica en vivo al stream.")


def _find_model_file(model_name: str) -> str:
    """Busca en models/ el archivo con el basename indicado (orden de preferencia fijo)."""
    matches = [p for p in MODELS_DIR.glob(f"{model_name}.*")
               if p.suffix.lower() in MODEL_EXTENSIONS]
    if not matches:
        raise FileNotFoundError(
            f"No se encontro archivo de modelo para '{model_name}' en {MODELS_DIR}/")
    matches.sort(key=lambda p: _EXTENSION_PREFERENCE.index(p.suffix.lower()))
    return str(matches[0])


def _load_and_validate(model_path: str) -> dict:
    """Carga + validacion cruzada JSON↔modelo. Mapea fallos a errores HTTP honestos."""
    try:
        controller.load_model(model_path)
        validation = controller.validate_pipeline()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except (ValidationError, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fallo inesperado al cargar el modelo: {e}")
    return validation


# ════════════════════════════════════════
# 1a Listar modelos disponibles
# ════════════════════════════════════════

@app.get("/get_models", summary="Listar modelos disponibles")
def get_models():
    """Modelos con config JSON valido Y archivo de pesos presente en models/.

    Los JSON de configs/ sin archivo de modelo (ej: plantillas) no se listan.
    """
    try:
        models = []
        for cfg in sorted(CONFIGS_DIR.glob("*.json")):
            has_weights = any(p.suffix.lower() in MODEL_EXTENSIONS
                              for p in MODELS_DIR.glob(f"{cfg.stem}.*"))
            if has_weights:
                models.append(cfg.stem)
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════
# 1a-bis Listar TODOS los pesos con su estado de config (vista Modelos)
# ════════════════════════════════════════

@app.get("/models", summary="Listar archivos de pesos con estado de config")
def list_models():
    """Escanea models/ y devuelve cada peso soportado con si tiene config JSON.

    A diferencia de /get_models (que lista solo los cargables = config+pesos, para el
    selector de inferencia), este lista TODOS los pesos para la vista de Modelos: por
    eso incluye los que todavia no tienen config (hasConfig=false). Reemplaza el viejo
    IPC 'models:list' (el frontend ya no toca disco — ver SDD: thin client sin disco).
    """
    try:
        models = []
        for p in sorted(MODELS_DIR.glob("*")):
            ext = p.suffix.lower()
            if ext not in MODEL_EXTENSIONS:
                continue
            base = p.stem
            has_config = (CONFIGS_DIR / f"{base}.json").exists()
            models.append({
                "file": p.name,
                "ext": ext.lstrip("."),
                "baseName": base,
                "hasConfig": has_config,
            })
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════
# 1b Seleccionar modelo por nombre
# ════════════════════════════════════════

@app.post("/select_model", summary="Cargar un modelo por nombre")
def select_model(data: SelectModelRequest):
    """Busca el archivo en models/, arma el pipeline y corre una validacion post-carga.

    Si el JSON no coincide con lo que el modelo realmente devuelve, responde 422
    con el detalle (antes respondia "ok" y el error aparecia recien en el stream).
    """
    try:
        model_path = _find_model_file(data.model_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    validation = _load_and_validate(model_path)
    return {
        "status": "ok",
        "message": f"Modelo cargado y validado: {data.model_name}",
        "validation": validation,
    }


# ════════════════════════════════════════
# 1c Cargar modelo por path directo
# ════════════════════════════════════════

@app.post("/model/load", summary="Cargar un modelo por ruta directa")
def load_model(data: ModelPathRequest):
    validation = _load_and_validate(data.model_path)
    return {
        "status": "ok",
        "message": f"Modelo cargado y validado: {data.model_path}",
        "validation": validation,
    }


# ════════════════════════════════════════
# 2 Actualizar umbral de confianza
# ════════════════════════════════════════

@app.post("/config/confidence", summary="Actualizar umbral de confianza en vivo")
def update_confidence(data: ConfidenceUpdateRequest):
    try:
        controller.update_confidence(data.value)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"status": "ok", "new_confidence": data.value}


# ════════════════════════════════════════
# 3 Descargar modelo
# ════════════════════════════════════════

@app.post("/model/unload", summary="Liberar el modelo cargado")
def unload_model():
    controller.unload_model()
    return {"status": "ok", "message": "Modelo descargado."}


# ════════════════════════════════════════
# 4 WebSocket streaming con inferencia
# ════════════════════════════════════════

def _decode_frame(message: dict):
    """Acepta frames JPEG binarios (protocolo actual) o base64 (compatibilidad)."""
    data = message.get("bytes")
    if data is None:
        text = message.get("text") or ""
        if "," in text:  # data URL: "data:image/jpeg;base64,...."
            text = text.split(",", 1)[1]
        try:
            data = base64.b64decode(text)
        except Exception:
            return None
    img_np = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(img_np, cv2.IMREAD_COLOR)


@app.websocket("/video_stream")
async def video_stream(websocket: WebSocket):
    """Protocolo: el cliente envia un frame JPEG (binario) y espera UN mensaje JSON:

        {"detections": [[x1, y1, x2, y2, conf, cls], ...], "error": null}

    Las coordenadas vienen en pixeles de la imagen original; el cliente dibuja.
    SIEMPRE se responde (aunque el frame sea invalido o falle la inferencia) para
    que el cliente nunca quede esperando un frame que no va a llegar.
    """
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            response = {"detections": [], "error": None}
            img_bgr = _decode_frame(message)

            if img_bgr is None:
                response["error"] = "frame_invalido"
            elif controller.predict_fn is None:
                response["error"] = "no_model"
            else:
                try:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    # En threadpool: la inferencia es bloqueante y no debe congelar
                    # el event loop (los endpoints REST siguen respondiendo).
                    loop = asyncio.get_event_loop()
                    detections = await loop.run_in_executor(
                        None, controller.inference, img_rgb)
                    response["detections"] = [
                        [round(float(v), 2) for v in det] for det in detections
                    ]
                except Exception as e:
                    _inference_errors.append({
                        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                        "error": str(e),
                    })
                    response["error"] = "inference_error"

            await websocket.send_json(response)

    except WebSocketDisconnect:
        pass


# ════════════════════════════════════════
# 5 Obtener logs de inferencia
# ════════════════════════════════════════

@app.get("/logs/inference", summary="Ultimos errores de inferencia")
def get_inference_logs():
    return {"logs": list(_inference_errors)}


# ════════════════════════════════════════
# 6 Metricas de rendimiento
# ════════════════════════════════════════

@app.get("/metrics", summary="Metricas de rendimiento del pipeline")
def get_metrics():
    stats = controller.perf.stats()
    if stats is None:
        return {"status": "no_data", "metrics": None}
    return {"status": "ok", "metrics": stats}


# ════════════════════════════════════════
# 7 Templates de config + escritura de config (single source of truth)
# ════════════════════════════════════════

_VALID_MODEL_TYPES = {"detection", "classification", "segmentation"}
# Nombre de config seguro: sin separadores ni "..". Mismo criterio que el IPC del front.
_SAFE_CONFIG_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


@app.get("/config/template/{model_type}", summary="Defaults de config por tipo de modelo")
def config_template(model_type: str):
    """Defaults generados desde el schema Pydantic (no duplicados en el frontend).

    Para detection incluye ademas los defaults de anchors (pack_format anchor_deltas).
    """
    if model_type not in _VALID_MODEL_TYPES:
        raise HTTPException(status_code=404, detail=f"model_type desconocido: {model_type}")
    return {
        "config": build_config_template(model_type),
        "anchor_defaults": anchor_defaults() if model_type == "detection" else None,
    }


@app.post("/configs/{name}", summary="Validar y guardar una config de modelo")
def write_config(name: str, body: dict = Body(...)):
    """Valida el body contra ModelConfig (estricto) y lo escribe en configs/<name>.json.

    422 si el nombre es inseguro o el body no cumple el schema.
    """
    if not _SAFE_CONFIG_NAME.match(name):
        raise HTTPException(status_code=422, detail=f"nombre de config inseguro: '{name}'")
    try:
        cfg = ModelConfig.model_validate(body)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=json.loads(e.json()))

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIGS_DIR / f"{name}.json"
    path.write_text(
        json.dumps(cfg.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {"ok": True, "path": str(path)}


@app.get("/configs/{name}", summary="Leer la config existente de un modelo")
def read_config(name: str):
    """Devuelve configs/<name>.json parseado, o config:null si no existe.

    Reemplaza el viejo IPC 'configs:read'. Contrato pensado para el wizard:
    - faltante  -> 200 {config: null}  (el wizard arranca con el template del backend).
    - corrupto  -> 500  (el frontend cae al template igual; regla SDD 4.1.4: no bloquea).
    - nombre inseguro -> 422.
    """
    if not _SAFE_CONFIG_NAME.match(name):
        raise HTTPException(status_code=422, detail=f"nombre de config inseguro: '{name}'")
    path = CONFIGS_DIR / f"{name}.json"
    if not path.exists():
        return {"config": None}
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise HTTPException(status_code=500, detail=f"config corrupta o ilegible: {e}")
    return {"config": config}


# ════════════════════════════════════════
# 8 Subida de pesos por multipart (reemplaza el IPC 'models:import')
# ════════════════════════════════════════

# Tamano de chunk para volcar el upload a disco sin cargarlo entero en RAM.
_UPLOAD_CHUNK = 1024 * 1024  # 1 MiB


@app.post("/models/upload", summary="Subir un archivo de pesos a models/")
async def upload_model(file: UploadFile = File(...)):
    """Recibe UN peso por multipart y lo escribe en models/<archivo>.

    Valida extension y nombre seguro ANTES de leer el stream (fail fast) y copia en
    chunks para soportar archivos grandes sin agotar memoria. Sobrescribe si ya existe
    (mismo comportamiento que el viejo copyFileSync del IPC). El frontend sube de a un
    archivo (un request por archivo) para poder reportar progreso y errores por archivo.
    """
    # Path(...).name descarta cualquier componente de ruta que venga en el filename.
    filename = Path(file.filename or "").name
    stem = Path(filename).stem
    ext = Path(filename).suffix.lower()

    if ext not in MODEL_EXTENSIONS:
        raise HTTPException(status_code=422, detail=f"extension no soportada: '{ext}'")
    if not _SAFE_CONFIG_NAME.match(stem):
        raise HTTPException(status_code=422, detail=f"nombre de archivo inseguro: '{stem}'")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / f"{stem}{ext}"
    try:
        with dest.open("wb") as out:
            while True:
                chunk = await file.read(_UPLOAD_CHUNK)
                if not chunk:
                    break
                out.write(chunk)
    except OSError as e:
        # Limpia el archivo parcial para no dejar un peso corrupto a medio escribir.
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"fallo al escribir el modelo: {e}")
    finally:
        await file.close()

    return {"ok": True, "file": dest.name}
