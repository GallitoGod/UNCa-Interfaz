
from api.func.output_pipeline.output_transformer import buildPostprocessor
from api.func.reader_pipeline.config_schema import OutputConfig, TensorStructure
import numpy as np

def test_postprocessor_filters_confidence():
    preds = [
        [0, 0, 10, 10, 0.9, 1],
        [5, 5, 15, 15, 0.3, 0],
        [20, 20, 30, 30, 0.95, 2]
    ]

    ts = TensorStructure(
        box_format="xyxy",
        coordinates={"x1": 0, "y1": 1, "x2": 2, "y2": 3},
        confidence_index=4,
        class_index=5,
        num_classes=80
    )

    config = OutputConfig(
        confidence_threshold=0.5,
        tensor_structure=ts
    )

    post_fn = buildPostprocessor(config, transform_info=None)
    result = post_fn(preds)

    assert len(result) == 2
    assert all(p[4] >= 0.5 for p in result)
