
from api.func.output_pipeline.output_unpacker import unpack_out
import numpy as np

def test_softmax_unpack():
    raw = [[0.1, 0.9, 0.0]]
    unpack = unpack_out("softmax")
    result = unpack(raw)
    assert result[0][5] == 1
    assert 0.89 < result[0][4] < 0.91
