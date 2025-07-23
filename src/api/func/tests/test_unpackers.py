
from api.func.general.unpackers import build_unpacker
import numpy as np

def test_softmax_unpack():
    raw = [[0.1, 0.9, 0.0]]
    unpack = build_unpacker("softmax")
    result = unpack(raw)
    assert result[0][5] == 1
    assert 0.89 < result[0][4] < 0.91
