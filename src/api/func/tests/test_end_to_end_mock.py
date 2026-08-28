import pytest
from unittest.mock import patch, MagicMock
from api.func.model_controller import ModelController
from api.func.reader_pipeline.config_schema import RuntimeConfig, RuntimeShapes

@pytest.fixture
def fake_config():
    cfg = MagicMock()
    cfg.model_path = "fake_model.onnx"
    cfg.model_type = "detection"

    cfg.input = MagicMock()

    cfg.output = MagicMock()
    cfg.output.pack_format = "raw"
    # tensor_structure con indices reales: inference() valida que los indices
    # declarados caben en el ancho del tensor desempaquetado.
    cfg.output.tensor_structure = MagicMock()
    cfg.output.tensor_structure.coordinates = {"x1": 0, "y1": 1, "x2": 2, "y2": 3}
    cfg.output.tensor_structure.confidence_index = 4
    cfg.output.tensor_structure.class_index = 5
    cfg.output.confidence_threshold = 0.5

    # runtimeShapes solo guarda constantes de carga; el estado por-frame
    # (orig_width/height, letterbox) ahora viaja en el meta del preprocesador.
    cfg.runtime = MagicMock()
    cfg.runtime.runtimeShapes = MagicMock()
    cfg.runtime.runtimeShapes.input_width = 320
    cfg.runtime.runtimeShapes.input_height = 320
    cfg.runtime.runtimeShapes.out_coords_space = "tensor_pixels"

    cfg.runtime.warmup = MagicMock()
    cfg.runtime.warmup.enabled = False
    cfg.runtime.warmup.runs = 0

    return cfg


def test_load_and_inference_integration_strict_runtime(fake_config):
    fake_logger = MagicMock()
    # Contrato nuevo del preprocesador: devuelve (tensor, meta). El meta es el
    # dict por-frame que el controller debe pasar tal cual al postprocesador.
    FAKE_META = {
        "orig_width": 320, "orig_height": 320,
        "scale": 1.0, "pad_left": 0.0, "pad_top": 0.0,
        "letterbox_used": False,
    }
    fake_pre = MagicMock(side_effect=lambda x: (f"pre:{x}", FAKE_META))
    fake_input_adapter = MagicMock(side_effect=lambda x: f"adapt_in:{x}")
    fake_predict = MagicMock(side_effect=lambda x: f"raw:{x}")

    import numpy as np
    import supervision as sv
    def _fake_unpack(raw, runtime=None):
        return np.array([[10.0, 10.0, 20.0, 20.0, 0.9, 0.0]], dtype=np.float32)
    fake_unpack = MagicMock(side_effect=_fake_unpack)

    fake_output_adapter = MagicMock(side_effect=lambda row: row)
    # El postprocesador ahora recibe (rows, meta)
    fake_post = MagicMock(side_effect=lambda rows, meta: rows)

    # El armado del pipeline vive ahora en tasks/detection.py: los símbolos del armado
    # se parchean alli. load_model_config y setup_model_logger siguen en el controller.
    with patch("api.func.model_controller.load_model_config", return_value=fake_config), \
         patch("api.func.tasks.detection.Model_loader.load", return_value=fake_predict), \
         patch("api.func.tasks.detection.build_preprocessor", return_value=fake_pre), \
         patch("api.func.tasks.detection.generate_input_adapter", return_value=fake_input_adapter), \
         patch("api.func.tasks.detection.unpack_out", autospec=True) as mock_unpack_factory, \
         patch("api.func.tasks.detection.generate_output_adapter", return_value=fake_output_adapter), \
         patch("api.func.tasks.detection.buildPostprocessor", return_value=fake_post), \
         patch("api.func.model_controller.setup_model_logger", return_value=fake_logger):

        mock_unpack_factory.return_value = fake_unpack

        mc = ModelController()
        mc.load_model("fake_model.onnx")

        mock_unpack_factory.assert_called_once_with(fake_config.output)

        result = mc.inference("image_data")

        fake_pre.assert_called_once_with("image_data")
        fake_input_adapter.assert_called_once_with("pre:image_data")
        fake_predict.assert_called_once_with("adapt_in:pre:image_data")

        fake_unpack.assert_called_once()
        args, kwargs = fake_unpack.call_args
        assert args[0] == "raw:adapt_in:pre:image_data"
        assert args[1] is fake_config.runtime

        # UNA sola llamada con el TENSOR ENTERO (N,K), no una por fila: el adapter se
        # vectorizo el 2026-08-27 porque con heads crudos (anchor_deltas entrega ~19k
        # anchors) el list comprehension dominaba el postproceso. Por eso lo esperado
        # es 2D: si alguien volviera a llamarlo fila a fila, este test lo delata.
        assert fake_output_adapter.call_count == 1
        np.testing.assert_array_almost_equal(
            fake_output_adapter.call_args[0][0],
            [[10.0, 10.0, 20.0, 20.0, 0.9, 0.0]]
        )
        # El controller debe pasar al post el MISMO meta que devolvio el pre
        assert fake_post.call_count == 1
        assert fake_post.call_args[0][1] is FAKE_META

        # inference() devuelve el tipo de dominio (sv.Detections), no el (N,6) pelado.
        assert isinstance(result, sv.Detections)
        assert len(result) == 1
        np.testing.assert_array_almost_equal(result.xyxy, [[10.0, 10.0, 20.0, 20.0]])
        np.testing.assert_array_almost_equal(result.confidence, [0.9])
        assert result.class_id.tolist() == [0]


def test_update_confidence_strict_runtime(fake_config):
    fake_logger = MagicMock()

    with patch("api.func.model_controller.load_model_config", return_value=fake_config), \
         patch("api.func.tasks.detection.Model_loader.load", return_value=lambda x: x), \
         patch("api.func.tasks.detection.build_preprocessor", return_value=lambda x: (x, {})), \
         patch("api.func.tasks.detection.generate_input_adapter", return_value=lambda x: x), \
         patch("api.func.tasks.detection.unpack_out", return_value=lambda raw, runtime=None: [[0, 0, 1, 1, 0.5, 0.0]]), \
         patch("api.func.tasks.detection.generate_output_adapter", return_value=lambda row: row), \
         patch("api.func.tasks.detection.buildPostprocessor", return_value=lambda rows, meta: rows), \
         patch("api.func.model_controller.setup_model_logger", return_value=fake_logger):

        mc = ModelController()
        mc.load_model("fake_model.onnx")

        mc.update_confidence(0.9)
        assert mc.config.output.confidence_threshold == 0.9