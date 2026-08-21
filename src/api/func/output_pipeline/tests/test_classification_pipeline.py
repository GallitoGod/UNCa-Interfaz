# Tests del pipeline de salida de CLASIFICACION: unpackers + postprocesador.
import numpy as np
import pytest

from api.func.reader_pipeline.config_schema import (
    ClassificationOutput,
    TensorClassification,
)
from api.func.output_pipeline.unpackers.registry import unpack_out
from api.func.output_pipeline import build_classification_postprocessor


def make_cfg(pack_format="softmax_out", output_format="logits", num_classes=3,
             top_k=0, confidence_threshold=0.0, **kw):
    """Arma un ClassificationOutput real (no un mock): valida el contrato de paso."""
    return ClassificationOutput(
        pack_format=pack_format,
        top_k=top_k,
        confidence_threshold=confidence_threshold,
        tensor_structure=TensorClassification(
            num_classes=num_classes, output_format=output_format, **kw),
    )


# ── Unpackers: quien decide la activacion ─────────────────────────────────────

def test_probabilities_no_reaplica_activacion():
    """
    LA regla del pipeline de clasificacion: si el modelo ya emite probabilidades,
    el pack_format NO debe volver a activarlas. Es el caso de saved_model_class,
    cuya firma se llama 'logits' pero devuelve sigmoides ya aplicadas.
    """
    cfg = make_cfg(pack_format="sigmoid_out", output_format="probabilities")
    fn = unpack_out(cfg)
    probs = np.array([0.6, 0.4, 0.01], dtype=np.float32)
    out = fn([probs[None, :]], None)
    np.testing.assert_allclose(out, probs, rtol=1e-6)


def test_softmax_out_sobre_logits_suma_uno():
    cfg = make_cfg(pack_format="softmax_out", output_format="logits")
    out = unpack_out(cfg)(np.array([[1.0, 2.0, 3.0]], dtype=np.float32), None)
    assert pytest.approx(float(out.sum()), abs=1e-5) == 1.0
    # softmax es monotona: conserva el orden de los logits
    assert int(np.argmax(out)) == 2


def test_softmax_es_estable_con_logits_grandes():
    # Sin restar el maximo, exp(1000) desborda a inf y sale nan.
    cfg = make_cfg(pack_format="softmax_out", output_format="logits")
    out = unpack_out(cfg)(np.array([[1000.0, 1001.0, 999.0]], dtype=np.float32), None)
    assert np.all(np.isfinite(out))
    assert pytest.approx(float(out.sum()), abs=1e-5) == 1.0


def test_sigmoid_out_sobre_logits():
    cfg = make_cfg(pack_format="sigmoid_out", output_format="logits")
    out = unpack_out(cfg)(np.array([[0.0, 2.0, -2.0]], dtype=np.float32), None)
    np.testing.assert_allclose(out[0], 0.5, atol=1e-6)
    # multi-etiqueta: las clases no compiten, la suma NO tiene por que dar 1
    assert float(out.sum()) > 1.0


def test_logits_raw_no_toca_nada():
    cfg = make_cfg(pack_format="logits_raw", output_format="logits")
    logits = np.array([-3.0, 0.0, 7.5], dtype=np.float32)
    out = unpack_out(cfg)(logits[None, :], None)
    np.testing.assert_allclose(out, logits)


# ── Unpackers: normalizacion de forma ─────────────────────────────────────────

@pytest.mark.parametrize("raw", [
    np.zeros((1, 3), dtype=np.float32),          # (1, C) tipico de ONNX
    np.zeros((3,), dtype=np.float32),            # ya plano
    np.zeros((1, 1, 3), dtype=np.float32),       # doble dimension de batch
    [np.zeros((1, 3), dtype=np.float32)],        # list de un tensor (ONNX Runtime)
    (np.zeros((1, 3), dtype=np.float32),),       # tuple de un tensor (TFLite)
])
def test_unpacker_normaliza_a_vector_1d(raw):
    out = unpack_out(make_cfg(pack_format="logits_raw"))(raw, None)
    assert out.shape == (3,)
    assert out.dtype == np.float32


def test_varios_tensores_falla_ruidoso():
    # Elegir "cual es el de clases" a ojo seria justo el silencio que el proyecto evita.
    fn = unpack_out(make_cfg(pack_format="logits_raw"))
    with pytest.raises(ValueError, match="2 tensores"):
        fn([np.zeros((1, 3)), np.zeros((1, 3))], None)


def test_batch_mayor_a_uno_falla():
    fn = unpack_out(make_cfg(pack_format="logits_raw"))
    with pytest.raises(ValueError, match="vector de clases"):
        fn(np.zeros((2, 3), dtype=np.float32), None)


# ── Postprocesador: umbral, top-k y orden ─────────────────────────────────────

def test_ordena_por_score_descendente():
    post = build_classification_postprocessor(make_cfg())
    out = post(np.array([0.1, 0.9, 0.5], dtype=np.float32))
    assert out.shape == (3, 2)
    assert [int(c) for c in out[:, 0]] == [1, 2, 0]


def test_top_k_corta_despues_de_ordenar():
    post = build_classification_postprocessor(make_cfg(top_k=2))
    out = post(np.array([0.1, 0.9, 0.5], dtype=np.float32))
    assert [int(c) for c in out[:, 0]] == [1, 2]


def test_umbral_conserva_los_class_id_originales():
    """
    El bug clasico de este paso: filtrar por umbral y despues reportar la POSICION
    dentro del vector filtrado en vez del id de clase real. Las clases 0 y 1 caen,
    asi que la sobreviviente debe seguir siendo la 2, no la 0.
    """
    post = build_classification_postprocessor(make_cfg(confidence_threshold=0.5))
    out = post(np.array([0.1, 0.2, 0.8], dtype=np.float32))
    assert out.shape == (1, 2)
    assert int(out[0, 0]) == 2
    assert pytest.approx(float(out[0, 1]), abs=1e-6) == 0.8


def test_nada_sobre_el_umbral_devuelve_vacio():
    post = build_classification_postprocessor(make_cfg(confidence_threshold=0.9))
    out = post(np.array([0.1, 0.2, 0.8], dtype=np.float32))
    assert out.shape == (0, 2)


def test_umbral_se_lee_en_cada_llamada():
    """El slider de confianza del cliente debe tener efecto en vivo, sin recargar."""
    cfg = make_cfg(confidence_threshold=0.0)
    post = build_classification_postprocessor(cfg)
    scores = np.array([0.1, 0.2, 0.8], dtype=np.float32)

    assert post(scores).shape[0] == 3
    cfg.confidence_threshold = 0.5      # lo que hace ModelController.update_confidence
    assert post(scores).shape[0] == 1


def test_vector_vacio_no_explota():
    post = build_classification_postprocessor(make_cfg())
    assert post(np.empty((0,), dtype=np.float32)).shape == (0, 2)


def test_rechaza_entrada_que_no_sea_vector():
    post = build_classification_postprocessor(make_cfg())
    with pytest.raises(ValueError, match=r"\(C,\)"):
        post(np.zeros((2, 3), dtype=np.float32))
