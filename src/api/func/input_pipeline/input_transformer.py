from api.func.reader_pipeline.config_schema import InputConfig, RuntimeConfig
from typing import Callable, Dict, Tuple, Union
import numpy as np
import cv2

# Tipo del metadata por-frame que produce el preprocesador y consume el
# postprocesador. Viaja como valor de retorno (no como estado compartido),
# asi cada frame lleva su propia info y dos streams no se pisan entre si.
FrameMeta = Dict[str, Union[int, float, bool]]


def _make_meta(orig_w: int, orig_h: int,
               scale: float = 1.0, pad_left: float = 0.0, pad_top: float = 0.0,
               letterbox_used: bool = False) -> FrameMeta:
    """Construye el dict de metadata de UN frame.

    Contrato (lo lee _undo_transform_xyxy_inplace en output_transformer):
      - orig_width/orig_height: tamano de la imagen original recibida.
      - scale/pad_left/pad_top: parametros del letterbox aplicado (si hubo).
      - letterbox_used: True si las cajas deben deshacerse con scale/pads;
        False si alcanza con re-escalar por (orig/input).
    """
    return {
        "orig_width": int(orig_w),
        "orig_height": int(orig_h),
        "scale": float(scale),
        "pad_left": float(pad_left),
        "pad_top": float(pad_top),
        "letterbox_used": bool(letterbox_used),
    }


def build_letterbox(pad_color, input_width: int, input_height: int):
    """
    Letterbox preprocessor builder.
    - Redimensiona preservando aspect ratio al tamano (input_width x input_height)
      y rellena el sobrante con pad_color.
    - Cachea scale/pads por tamano de frame: en un stream la resolucion no cambia,
      asi que el calculo se hace una sola vez por cambio de resolucion.
    - NO toca estado compartido: devuelve (imagen, meta) y el meta acompana al
      frame por el resto del pipeline.
    """
    iw = int(input_width)
    ih = int(input_height)

    # Cache local del closure: ultima resolucion vista y sus parametros derivados.
    last_wh = None
    cached = None  # (scale, nw, nh, pad_left, pad_right, pad_top, pad_bottom)

    def letterbox(img) -> Tuple[np.ndarray, FrameMeta]:
        nonlocal last_wh, cached

        h, w = img.shape[:2]

        # Recalcular scale/pads solo si cambio la resolucion del frame
        if last_wh != (w, h) or cached is None:
            scale = min(iw / w, ih / h)          # factor que hace entrar la imagen sin deformar
            nw, nh = int(w * scale), int(h * scale)

            # El padding reparte el sobrante en dos mitades (centrado)
            pad_w = iw - nw
            pad_h = ih - nh
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top

            cached = (scale, nw, nh, pad_left, pad_right, pad_top, pad_bottom)
            last_wh = (w, h)

        scale, nw, nh, pad_left, pad_right, pad_top, pad_bottom = cached

        # INTER_AREA para achicar (mejor calidad), INTER_LINEAR para agrandar
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR

        resized = cv2.resize(img, (nw, nh), interpolation=interp)

        out = cv2.copyMakeBorder(
            resized,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=pad_color
        )

        # El meta de ESTE frame: el postprocesador lo usa para deshacer la transformacion
        meta = _make_meta(w, h, scale=scale, pad_left=pad_left, pad_top=pad_top,
                          letterbox_used=True)
        return out, meta

    return letterbox


def build_preprocessor(config: InputConfig, runtime: RuntimeConfig) -> Callable[[np.ndarray], Tuple[np.ndarray, FrameMeta]]:
    """
    Arma el preprocesador completo: geometria (letterbox o resize directo) +
    escala/normalizacion fusionadas en una sola pasada.

    El callable devuelto toma la imagen original y devuelve (tensor, meta):
      - tensor: imagen lista para el input_adapter.
      - meta:   dict por-frame (ver _make_meta) que el postprocesador necesita
                para llevar las cajas de vuelta al espacio de la imagen original.

    runtime.runtimeShapes solo recibe constantes de carga (input_width/height,
    que los unpackers leen para escalar coordenadas normalizadas): se escribe
    UNA vez aca y no se vuelve a mutar por frame.
    """
    try:
        # Constantes de carga: tamano del tensor de entrada segun el JSON.
        # Los unpackers las leen via runtime; no cambian durante el stream.
        runtime.runtimeShapes.input_width  = int(config.width)
        runtime.runtimeShapes.input_height = int(config.height)

        use_letterbox = bool(config.letterbox and config.preserve_aspect_ratio)

        if use_letterbox:
            # Paso geometrico con letterbox: ya devuelve (img, meta)
            geometric_step = build_letterbox(config.auto_pad_color, config.width, config.height)
        else:
            # Resize directo (deforma si el aspect ratio difiere). El meta indica
            # letterbox_used=False para que el post re-escale por (orig/input).
            def geometric_step(img) -> Tuple[np.ndarray, FrameMeta]:
                h, w = img.shape[:2]
                resized = cv2.resize(img, (config.width, config.height))
                return resized, _make_meta(w, h, letterbox_used=False)

        # ---- Escala / normalizacion (fusionadas para una sola pasada numpy) ----
        _mean = np.array(config.mean, dtype=np.float32).reshape(1, 1, -1)
        _std  = np.array(config.std,  dtype=np.float32).reshape(1, 1, -1)

        if config.normalize and np.any(_std == 0):
            raise ValueError("std no puede contener ceros para la normalizacion")

        # Si mean=0 y std=1 la normalizacion es identidad: conviene saltearla
        trivial_normalize = (
            not config.normalize or
            (np.allclose(_mean, 0.0) and np.allclose(_std, 1.0))
        )

        if config.scale and not trivial_normalize:
            # Fusiona escala + normalizacion en una sola pasada:
            # (img/255 - mean) / std  ==  img * factor + offset
            _factor = np.float32(1.0 / 255.0) / _std
            _offset = -_mean / _std
            value_step = lambda img: img.astype(np.float32) * _factor + _offset
        elif config.scale:
            # Normalizacion trivial (mean=0, std=1): solo escala a [0,1]
            value_step = lambda img: img.astype(np.float32) * np.float32(1.0 / 255.0)
        elif not trivial_normalize:
            # Sin escala, normalize no trivial
            value_step = lambda img: (img.astype(np.float32) - _mean) / _std
        else:
            # Ni escala ni normalizacion: pasa tal cual (el adapter castea dtype)
            value_step = None

        def preprocess(img) -> Tuple[np.ndarray, FrameMeta]:
            # 1) geometria: resize/letterbox + metadata del frame
            img, meta = geometric_step(img)
            # 2) valores: escala/normalizacion (si aplica)
            if value_step is not None:
                img = value_step(img)
            return img, meta

        return preprocess
    except Exception as e:
        raise ValueError(e) from e
