
from api.func.general.transformers import buildPostprocessor
from api.func.general.config_schema import OutputConfig
import numpy as np

def test_postprocessor_filters_confidence():
    preds = [
        [0, 0, 10, 10, 0.9, 1],
        [5, 5, 15, 15, 0.3, 0],
        [20, 20, 30, 30, 0.95, 2]
    ]
    config = OutputConfig(confidence_threshold=0.5)
    post_fn = buildPostprocessor(config)
    result = post_fn(preds)
    assert len(result) == 2
    assert all(p[4] >= 0.5 for p in result)
