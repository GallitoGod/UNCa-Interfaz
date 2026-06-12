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
