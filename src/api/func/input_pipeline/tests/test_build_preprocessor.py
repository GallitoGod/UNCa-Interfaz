import numpy as np
import pytest
from api.func.input_pipeline.input_transformer import build_preprocessor
from api.func.reader_pipeline.config_schema import InputConfig, RuntimeConfig, RuntimeShapes

# El preprocesador devuelve (tensor, meta). runtimeShapes solo guarda constantes
# de carga (input_width/height); el estado por-frame viaja en el meta.


def _make_runtime():
    # RuntimeShapes ya no tiene orig_width/orig_height/metadata_letter:
    # esos datos son por-frame y viven en el meta que devuelve el preprocesador.
    return RuntimeConfig(runtimeShapes=RuntimeShapes(
        input_width=0,
        input_height=0,
        channels=3,
        out_coords_space="normalized_0_1",
    ))


def test_preprocessor_with_letterbox_and_scaling_and_normalization():
    cfg = InputConfig(
        width=640,
        height=360,
        channels=3,
        normalize=True,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        scale=True,
        letterbox=True,
        auto_pad_color=(0, 0, 0),
        preserve_aspect_ratio=True,
        color_order="RGB",
        input_str=None,
    )
    runtime = _make_runtime()
    preprocess_fn = build_preprocessor(cfg, runtime)

    img = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
    out, meta = preprocess_fn(img)

    # Tamano final debe coincidir con config
    assert out.shape == (360, 640, 3)

    # Debe estar normalizado: valor medio ~ 1.0 para una imagen blanca
    assert np.allclose(out, (1.0 - 0.5) / 0.5, atol=1e-6)

    # El meta del frame refleja el letterbox aplicado y el tamano original
    assert meta["letterbox_used"] is True
    assert meta["orig_width"] == 1920
    assert meta["orig_height"] == 1080

    # build_preprocessor dejo las constantes de carga en runtimeShapes
    assert runtime.runtimeShapes.input_width == 640
    assert runtime.runtimeShapes.input_height == 360


def test_preprocessor_without_letterbox():
    cfg = InputConfig(
        width=320,
        height=240,
        channels=3,
        normalize=True,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        scale=True,
        letterbox=False,
        auto_pad_color=(0, 0, 0),
        preserve_aspect_ratio=False,
        color_order="RGB",
        input_str=None,
    )
    runtime = _make_runtime()
    preprocess_fn = build_preprocessor(cfg, runtime)

    img = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
    out, meta = preprocess_fn(img)

    assert out.shape == (240, 320, 3)
    # Sin letterbox: el post debera re-escalar por (orig/input), no por scale/pads
    assert meta["letterbox_used"] is False
    assert meta["orig_width"] == 1920
    assert meta["orig_height"] == 1080


def test_preprocessor_invalid_config():
    # Provocar un error en normalizacion (std=0 genera division por cero).
    # Se usa un runtime real para que el fallo venga del std y no de otra cosa.
    cfg = InputConfig(
        width=320,
        height=240,
        channels=3,
        normalize=True,
        mean=[0, 0, 0],
        std=[0, 0, 0],
        scale=False,
        letterbox=False,
        auto_pad_color=(0, 0, 0),
        preserve_aspect_ratio=False,
        color_order="RGB",
        input_str=None,
    )
    runtime = _make_runtime()

    with pytest.raises(ValueError):
        build_preprocessor(cfg, runtime)


# ── El value_step in-place (2026-08-27) ─────────────────────────────────────
# La escala/normalizacion paso de encadenar operadores (`img.astype(f32) * factor
# + offset`) a convertir UNA vez y operar in-place sobre esa copia. Cada operador
# encadenado alocaba y recorria un tensor entero de mas: con 640x640x3 float32 son
# 4,9 MB por copia. Estos tests fijan las dos propiedades que lo hacen seguro.

def _cfg_valores(scale, normalize, mean, std):
    return InputConfig(
        width=8, height=8, channels=3,
        normalize=normalize, mean=mean, std=std, scale=scale,
        letterbox=False, preserve_aspect_ratio=False,
        auto_pad_color=[114, 114, 114], color_order="RGB",
    )


# Las cuatro ramas de value_step, con el resultado esperado calculado a mano.
CASOS = [
    # (scale, normalize, mean, std, formula de referencia)
    (True,  True,  [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], lambda a: (a / 255.0 - 0.5) / 0.25),
    (True,  True,  [0.0, 0.0, 0.0], [1.0, 1.0, 1.0],    lambda a: a / 255.0),
    (True,  False, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0],    lambda a: a / 255.0),
    (False, True,  [10.0, 20.0, 30.0], [2.0, 4.0, 8.0],
     lambda a: (a - np.array([10.0, 20.0, 30.0], dtype=np.float32)) / np.array([2.0, 4.0, 8.0], dtype=np.float32)),
    (False, False, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0],    lambda a: a),
]


@pytest.mark.parametrize("scale,normalize,mean,std,referencia", CASOS)
def test_value_step_da_lo_mismo_que_la_formula(scale, normalize, mean, std, referencia):
    """La aritmetica no cambio al pasar a in-place: solo se dejo de copiar."""
    pre = build_preprocessor(_cfg_valores(scale, normalize, mean, std), _make_runtime())
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)

    tensor, _ = pre(img)
    esperado = referencia(img.astype(np.float32))

    np.testing.assert_allclose(tensor.astype(np.float64), np.asarray(esperado, dtype=np.float64),
                               rtol=0, atol=1e-6)


@pytest.mark.parametrize("scale,normalize,mean,std,_ref", CASOS)
def test_value_step_no_muta_el_frame_recibido(scale, normalize, mean, std, _ref):
    """
    LA propiedad que hace seguro el in-place: astype() copia siempre (copy=True es su
    default), asi que las operaciones posteriores escriben sobre esa copia y nunca
    sobre el frame del llamador. Si alguien "optimizara" con copy=False, el
    preprocesador empezaria a corromper el frame que le dieron.

    Hoy el handler del WS le pasa el img_rgb que produjo cv2.cvtColor —una copia
    aparte del img_bgr sobre el que despues dibuja—, asi que el radio de dano seria
    chico. Pero la invariante es del PREPROCESADOR, no del llamador de turno: es lo
    que permite que cualquiera reutilice su frame despues de inferir.
    """
    pre = build_preprocessor(_cfg_valores(scale, normalize, mean, std), _make_runtime())
    img = np.full((8, 8, 3), 200, dtype=np.uint8)
    intacto = img.copy()

    pre(img)

    assert np.array_equal(img, intacto), "el preprocesador mutó el frame que recibió"
