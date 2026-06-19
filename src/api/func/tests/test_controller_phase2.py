# tests/test_controller_phase2.py
# Cubre las dos reformas de comportamiento del controller en la Fase 2:
#   * Tarea 1: inference() corre en paralelo (el lock ya no la serializa).
#   * Tarea 3: el despachador por model_type reconoce classification/segmentation
#     pero levanta NotImplementedError honesto (la API lo mapea a 501).
import time
import threading
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from api.func.model_controller import ModelController


def _fake_meta():
    return {
        "orig_width": 2, "orig_height": 2,
        "scale": 1.0, "pad_left": 0.0, "pad_top": 0.0,
        "letterbox_used": False,
    }


def _wire_fake_pipeline(mc, predict_fn):
    """Cablea un pipeline falso directo en los atributos del controller (sin pasar
    por load_model). pack_format 'boxes_scores' evita el paso de adapter."""
    mc.preprocess_fn = lambda img: (img, _fake_meta())
    mc.input_adapter = lambda x: x
    mc.predict_fn = predict_fn
    mc.unpack_fn = lambda raw, rt: np.asarray(raw, dtype=np.float32)
    mc.output_adapter = lambda r: r
    mc.postprocess_fn = lambda rows, meta: [list(map(float, r)) for r in rows]

    cfg = MagicMock()
    cfg.output.pack_format = "boxes_scores"   # NO necesita adapter
    cfg.runtime.runtimeShapes.input_width = 2
    cfg.runtime.runtimeShapes.input_height = 2
    mc.config = cfg
    mc.logger = MagicMock()


def test_inference_corre_en_paralelo():
    """Si la inferencia siguiera serializada por el lock, N frames con un predict
    de 0.2s tardarian ~N*0.2s. Concurrentes tardan ~0.2s. Exigimos que NO sea
    serial: elapsed < (N-1)*sleep."""
    mc = ModelController()
    sleep_s = 0.2
    N = 4

    def slow_predict(x):
        time.sleep(sleep_s)  # time.sleep libera el GIL -> permite paralelismo real
        return np.array([[0.0, 0.0, 1.0, 1.0, 0.9, 0.0]], dtype=np.float32)

    _wire_fake_pipeline(mc, slow_predict)

    results = [None] * N

    def worker(i):
        results[i] = mc.inference(np.zeros((2, 2, 3), dtype=np.uint8))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0

    assert all(r is not None for r in results), "todas las inferencias deben completar"
    assert elapsed < (N - 1) * sleep_s, (
        f"la inferencia parece serializada: {elapsed:.3f}s para {N} frames "
        f"de {sleep_s}s (serial seria ~{N*sleep_s:.1f}s)")


def test_stats_consistentes_bajo_concurrencia():
    """El contador de frames y el PerfMeter se actualizan bajo _stats_lock: tras
    M inferencias concurrentes, el contador debe ser exactamente M (sin perdidas
    por carrera)."""
    mc = ModelController()

    def fast_predict(x):
        return np.array([[0.0, 0.0, 1.0, 1.0, 0.5, 0.0]], dtype=np.float32)

    _wire_fake_pipeline(mc, fast_predict)

    M = 50
    threads = [threading.Thread(
        target=lambda: mc.inference(np.zeros((2, 2, 3), dtype=np.uint8)))
        for _ in range(M)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert mc._frame_idx == M


def _snapshot_fue_logueado(logger):
    return any("SNAPSHOT" in str(c.args[0]) for c in logger.info.call_args_list)


def test_snapshot_metrics_vuelca_al_log_y_devuelve_stats():
    """Fase 4 tarea 4: snapshot_metrics() registra las metricas en el log y las
    devuelve, sin descargar el modelo."""
    mc = ModelController()
    _wire_fake_pipeline(mc, lambda x: np.array([[0, 0, 1, 1, 0.5, 0.0]], dtype=np.float32))

    for _ in range(5):
        mc.inference(np.zeros((2, 2, 3), dtype=np.uint8))

    stats = mc.snapshot_metrics()
    assert stats and stats["n"] == 5
    assert _snapshot_fue_logueado(mc.logger)
    # No descargo el modelo: sigue inferible.
    assert mc.predict_fn is not None


def test_unload_vuelca_snapshot_y_resetea_metricas():
    mc = ModelController()
    _wire_fake_pipeline(mc, lambda x: np.array([[0, 0, 1, 1, 0.5, 0.0]], dtype=np.float32))

    for _ in range(3):
        mc.inference(np.zeros((2, 2, 3), dtype=np.uint8))

    logger = mc.logger
    mc.unload_model()

    assert _snapshot_fue_logueado(logger)
    assert mc.perf.stats() is None    # metricas reseteadas al cerrar sesion
    assert mc.predict_fn is None      # pipeline liberado


def _fake_config(model_type):
    cfg = MagicMock()
    cfg.model_type = model_type
    return cfg


@pytest.mark.parametrize("model_type", ["classification", "segmentation"])
def test_despachador_rechaza_tipos_no_implementados(model_type):
    """El despachador reconoce el tipo (no es 'desconocido') pero su pipeline no
    existe -> NotImplementedError (la API responde 501 honesto). Y el controller
    queda descargado tras el fallo (carga atomica)."""
    with patch("api.func.model_controller.load_model_config",
               return_value=_fake_config(model_type)), \
         patch("api.func.model_controller.setup_model_logger",
               return_value=MagicMock()):
        mc = ModelController()
        with pytest.raises(NotImplementedError):
            mc.load_model("fake.onnx")
        assert mc.predict_fn is None and mc.config is None


def test_despachador_rechaza_tipo_desconocido():
    with patch("api.func.model_controller.load_model_config",
               return_value=_fake_config("teletransportacion")), \
         patch("api.func.model_controller.setup_model_logger",
               return_value=MagicMock()):
        mc = ModelController()
        with pytest.raises(NotImplementedError):
            mc.load_model("fake.onnx")
