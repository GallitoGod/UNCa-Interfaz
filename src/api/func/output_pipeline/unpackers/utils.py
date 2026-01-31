from typing import Any, List, Tuple
import numpy as np

NDArray = np.ndarray
ImageSize = Tuple[int, int]  # (W, H)


def to_2d(array_like: Any) -> NDArray:
    """
    Convierte a np.ndarray y normaliza a 2D:
      - (1, N, F) -> (N, F)
      - (N,) -> (N, 1)
      - (N, A, B, ...) -> (N, A*B*...)
    """
    arr = np.asarray(array_like)
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    if arr.ndim > 2:
        return arr.reshape(arr.shape[0], -1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr

def looks_normalized_xyxy(x1: NDArray, y1: NDArray, x2: NDArray, y2: NDArray) -> bool:
    """
    si los maximos estan cerca de 1, se asume [0..1], por ende, normalizados.
    """
    max_val = float(np.max([x1.max(initial=0.0), y1.max(initial=0.0),
                            x2.max(initial=0.0), y2.max(initial=0.0)]))
    return max_val <= 1.5

def looks_normalized_cxcywh(cx: NDArray, cy: NDArray, w: NDArray, h: NDArray) -> bool:
    max_val = float(np.max([cx.max(initial=0.0), cy.max(initial=0.0),
                            w.max(initial=0.0),  h.max(initial=0.0)]))
    return max_val <= 1.5

def scale_xyxy_inplace(x1: NDArray, y1: NDArray, x2: NDArray, y2: NDArray,
                        input_tensor_size: ImageSize) -> None:
    W, H = input_tensor_size
    x1 *= W; x2 *= W
    y1 *= H; y2 *= H

def scale_cxcywh_inplace(cx: NDArray, cy: NDArray, w: NDArray, h: NDArray,
                          input_tensor_size: ImageSize) -> None:
    W, H = input_tensor_size
    cx *= W; w *= W
    cy *= H; h *= H

def stack_as_float_lists(columns: List[NDArray]) -> List[List[float]]:
    """
    Apila columnas 1D → matriz 2D y la devuelve como List[List[float]].
    """
    mat = np.column_stack(columns).astype(float, copy=False)
    return mat.tolist()

def stack_as_float32_matrix(columns: List[NDArray]) -> NDArray:
    """
    Apila columnas 1D → matriz 2D float32 (N,F). No crea objetos Python.
    """
    return np.column_stack(columns).astype(np.float32, copy=False)


def decode_anchor_deltas_to_yxyx(
    deltas: NDArray, anchors: NDArray, variance: NDArray
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    deltas:  (N,4) -> [ty, tx, th, tw]
    anchors: (N,4) -> [ay, ax, ah, aw]   (centro y tamaño), NORMALIZADOS [0..1]
    variance: (4,) -> [sy, sx, sh, sw]
    Devuelve: (ymin, xmin, ymax, xmax) NORMALIZADOS [0..1]
    """
    ty, tx, th, tw = deltas.T
    ay, ax, ah, aw = anchors.T
    sy, sx, sh, sw = variance.astype(np.float32)

    yc = ty * sy * ah + ay
    xc = tx * sx * aw + ax
    h  = np.exp(th * sh) * ah
    w  = np.exp(tw * sw) * aw

    ymin = yc - 0.5 * h
    xmin = xc - 0.5 * w
    ymax = yc + 0.5 * h
    xmax = xc + 0.5 * w
    return ymin, xmin, ymax, xmax

'''
    Hay una deuda con el sistema en la normalizacion, tengo que arreglarlo en algun momento.
'''