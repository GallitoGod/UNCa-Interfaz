import pytest
from unittest.mock import patch, MagicMock
from api.func.model_controller import ModelController

@pytest.fixture
def fake_config():
    cfg = MagicMock()
    cfg.model_path = "fake_model.onnx"
    cfg.input = MagicMock()
    cfg.output = MagicMock()
    cfg.output.model_dump.return_value = {"dummy": "data"}
    cfg.output.output_tensor.output_format = "raw"
    cfg.output.tensor_structure = "fake_structure"
    cfg.output.confidence_threshold = 0.5
    cfg.runtime = MagicMock()
    return cfg


def test_load_and_inference_integration(fake_config):
    fake_logger = MagicMock()
    fake_pre = MagicMock(side_effect=lambda x: f"pre:{x}")
    fake_input_adapter = MagicMock(side_effect=lambda x: f"adapt_in:{x}")
    fake_predict = MagicMock(side_effect=lambda x: f"raw:{x}")

    def _fake_unpack(*args, **kwargs):
        raw = args[0]
        return [f"unpacked:{raw}"]
    fake_unpack = MagicMock(side_effect=_fake_unpack)

    fake_output_adapter = MagicMock(side_effect=lambda rows: f"adapt_out:{rows[0]}")
    fake_post = MagicMock(side_effect=lambda rows: [f"post:{row}" for row in rows])

    with patch("api.func.model_controller.load_model_config", return_value=fake_config), \
         patch("api.func.model_controller.Reactive_output_config", side_effect=lambda **kw: fake_config.output), \
         patch("api.func.model_controller.Model_loader.load", return_value=fake_predict), \
         patch("api.func.model_controller.build_preprocessor", return_value=fake_pre), \
         patch("api.func.model_controller.generate_input_adapter", return_value=fake_input_adapter), \
         patch("api.func.model_controller.unpack_out", autospec=True) as mock_unpack_factory, \
         patch("api.func.model_controller.generate_output_adapter", return_value=fake_output_adapter), \
         patch("api.func.model_controller.buildPostprocessor", return_value=fake_post), \
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
        # Acepta ambas firmas: (raw) o (raw, runtime)
        assert (len(args) == 1) or (len(args) == 2 and args[1] is fake_config.runtime)

        fake_output_adapter.assert_called_once_with(["unpacked:raw:adapt_in:pre:image_data"])
        fake_post.assert_called_once_with(["adapt_out:unpacked:raw:adapt_in:pre:image_data"])
        assert result == ["post:adapt_out:unpacked:raw:adapt_in:pre:image_data"]

        fake_logger.info.assert_any_call("Modelo cargado correctamente")


def test_update_confidence(fake_config):
    fake_logger = MagicMock()

    with patch("api.func.model_controller.load_model_config", return_value=fake_config), \
         patch("api.func.model_controller.Reactive_output_config", side_effect=lambda **kw: fake_config.output), \
         patch("api.func.model_controller.Model_loader.load", return_value=lambda x: x), \
         patch("api.func.model_controller.build_preprocessor", return_value=lambda x: x), \
         patch("api.func.model_controller.generate_input_adapter", return_value=lambda x: x), \
         patch("api.func.model_controller.unpack_out", return_value=lambda *a, **k: a[0]), \
         patch("api.func.model_controller.generate_output_adapter", return_value=lambda x: x), \
         patch("api.func.model_controller.buildPostprocessor", return_value=lambda x: x), \
         patch("api.func.model_controller.setup_model_logger", return_value=fake_logger):

        mc = ModelController()
        mc.load_model("fake_model.onnx")

        mc.update_confidence(0.9)
        assert mc.config.output.confidence_threshold == 0.9
        fake_logger.info.assert_any_call("Confianza actualizada a 0.9.")


def test_unload_model(fake_config):
    mc = ModelController()
    mc.config = fake_config
    mc.predict_fn = lambda x: x

    mc.unload_model()

    assert mc.predict_fn is None
    assert mc.config is None
