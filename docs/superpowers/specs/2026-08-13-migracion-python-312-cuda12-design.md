# Paso 1: migración a Python 3.12 + CUDA 12 (venv-local)

- **Fecha**: 2026-08-13
- **Rama**: `refactor-frontend-react` (crear `migracion-python-312` a partir de ella)
- **Estado**: propuesto — pendiente de ejecutar

## 1. Contexto y objetivo

El proyecto corre sobre **Python 3.8.10**, que es el techo real de todo lo demás:
`onnxruntime-gpu` quedó pineado en 1.17.1 (CUDA 11.8) y `supervision` no se puede
instalar en ninguna versión posterior a 0.25.1 (de fines de 2024).

Este es el **paso 1 de tres** en el plan acordado:

1. **Migrar a Python 3.12 + CUDA 12** ← este documento
2. Meter `sv.Detections` como tipo de dominio interno detrás del seam de `tasks/`,
   sin tocar el transporte
3. Dibujar en el backend con los anotadores de supervision y mandar el frame
   compuesto por el WebSocket

**Objetivo del paso 1**: dejar el proyecto corriendo sobre 3.12 con GPU funcionando,
**sin cambiar ni una línea de lógica de negocio**, y con una regresión numérica que
demuestre que detección y clasificación siguen dando exactamente lo mismo.

Este paso **no** desbloquea clasificación ni segmentación por sí solo (clasificación ya
funciona en 3.8, implementada el 2026-08-13). Lo que desbloquea es supervision moderno,
salir del pin de ORT 1.17 y dejar de estar clavados a un intérprete que ya no recibe
parches de seguridad.

## 2. Alcance

### Entra

- Venv nuevo en Python 3.12 con todas las dependencias reinstaladas.
- CUDA 12 **instalado por pip dentro del venv** (no toolkit global).
- Una línea nueva en `src/api/func/reader_pipeline/forms/onnx_load.py`
  (`ort.preload_dlls()`).
- `requirements.txt` reescrito con pines reales + `requirements.lock.txt` generado.
- Verificación por regresión contra la línea de base de la sección 4.

### NO entra (explícitamente)

- **Ningún cambio de lógica**: ni pipelines, ni schema, ni endpoints, ni cliente.
  Si algo hay que arreglar para que corra, se anota y se hace en un commit aparte.
- **supervision**: se instala en el paso 2, no acá. Un paso, un riesgo.
- **Sacar TensorFlow** (ver §8.1): es una oportunidad real que aparece de este
  análisis, pero es un cambio con su propio riesgo y merece su propio paso.
- **Empaquetado / instalador** (pendiente #20).

## 3. Matriz de versiones (verificada contra PyPI el 2026-08-13)

El intérprete queda acotado por los dos extremos:

| paquete | piso | techo | elegido |
|---|---|---|---|
| onnxruntime-gpu | **3.11** (desde 1.25 no publica wheels para 3.10) | 3.14 | — |
| tensorflow 2.21 | 3.10 | **3.13** (no hay wheels 3.14) | — |
| torch 2.13 | 3.10 | 3.14 | — |
| supervision 0.30 | 3.10 | — | — |
| **Python** | 3.11 | 3.13 | **3.12** |

**Python 3.12** queda cómodo en el medio, con cobertura completa de wheels en los
cuatro. Ya está instalado en la máquina: `C:\Python312\python.exe` (verificado con
`py -0p`; también hay 3.11 y 3.13 si hiciera falta bajar o subir).

### Versión de CUDA

`onnxruntime-gpu` cambió de CUDA según la versión:

| onnxruntime-gpu | CUDA que declara |
|---|---|
| 1.24.2 – **1.26.0** | **12** |
| 1.27.0 – 1.28.0 | 13 |

**Se elige `onnxruntime-gpu==1.26.0`** (la última con CUDA 12). CUDA 13 es demasiado
nuevo para estrenarlo en el mismo movimiento en que se cambia de intérprete.

### CUDA por pip: cómo funciona y qué no cubre

`onnxruntime-gpu[cuda,cudnn]==1.26.0` arrastra cinco paquetes de NVIDIA, **todos con
wheel `win_amd64`** (verificado en PyPI):

| paquete | tamaño Windows |
|---|---|
| nvidia-cuda-runtime-cu12 | 4 MB |
| nvidia-cuda-nvrtc-cu12 | 76 MB |
| nvidia-curand-cu12 | 69 MB |
| nvidia-cufft-cu12 | 200 MB |
| nvidia-cudnn-cu12 | 737 MB |

Total **~1,1 GB dentro del venv**. Anotarlo para el pendiente #20 (instalador).

Verificado leyendo el código del wheel de ORT 1.26.0: tiene `_get_nvidia_dll_paths()`
que busca las DLL **dentro de site-packages** (`nvidia/cuda_runtime/bin/cudart64_12.dll`,
`nvidia/cudnn/bin/cudnn64_9.dll`, etc.). Está diseñado para este escenario.

**Lo que pip NO instala: el driver de NVIDIA.** Sigue siendo instalación a nivel
sistema. El driver actual ya corre CUDA 11.8, y por compatibilidad de versión menor un
driver que soporte 12.0 corre cualquier 12.x. **Chequear antes de empezar** con
`nvidia-smi` (la versión de CUDA que reporta arriba a la derecha debe ser ≥ 12.0).

## 4. Línea de base para la regresión (medida el 2026-08-13, en 3.8.10)

Estos son los números contra los que hay que comparar. Si después de migrar dan lo
mismo, la migración salió bien.

### Entorno actual

```
Python 3.8.10 · onnxruntime-gpu 1.17.1 (CUDA 11.8) · tensorflow 2.12.0
numpy 1.23.5 · torch 2.4.1 · opencv-python 4.11.0.86 · pydantic 2.10.6
fastapi 0.115.8 · uvicorn 0.33.0
```

### Salidas esperadas

| verificación | resultado esperado |
|---|---|
| `pytest` | **82 passed** |
| WS con `yolov7-tiny` sobre `horses.jpg` | `task=detection`, **5 cajas** |
| WS con `saved_model_class` sobre `horses.jpg` | `task=classification`, **3 clases: 663, 813, 51** |
| clasificador, umbral 0.05 | **5 clases: 663, 813, 51, 1092, 903** |
| clasificador directo (sin recompresión JPEG) | 663 → **0.6152**, 813 → **0.439** |
| `session.get_providers()` | incluye `CUDAExecutionProvider` |
| clasificador en GPU | **~14 ms** por frame |
| clasificador en CPU (ONNX) | ~52 ms |
| `npm run typecheck` / `npm run build` | limpios |

> Las cajas y clases deben coincidir **exactamente**. Diferencias de milésimas en los
> scores son aceptables (cambio de versión de ORT/cuDNN); un cambio en *qué* clases o
> *cuántas* cajas salen, no: eso es una regresión real.

## 5. Procedimiento

### 5.1 Preparación

1. Crear rama `migracion-python-312` desde `refactor-frontend-react`.
2. `nvidia-smi` → confirmar que el driver soporta CUDA ≥ 12.0. Si no, actualizar el
   driver **antes** de seguir.
3. **No borrar `.venv`.** El venv viejo es el plan de rollback (§7).

### 5.2 Venv nuevo

```bash
py -3.12 -m venv .venv312
.venv312\Scripts\python.exe -m pip install --upgrade pip
```

### 5.3 Instalación

```bash
# Core API
pip install "fastapi[standard]" uvicorn websockets python-multipart

# Inferencia — ORT con CUDA 12 traido por pip
pip install "onnxruntime-gpu[cuda,cudnn]==1.26.0"
pip install tensorflow torch

# Imagen
pip install opencv-python pillow numpy
```

> ⚠️ **NO instalar `onnxruntime-directml`.** El venv actual tiene
> `onnxruntime-gpu==1.17.1` **y** `onnxruntime-directml==1.19.2` a la vez. Dos paquetes
> de ORT en el mismo directorio es un conflicto conocido — el propio ORT emite un
> warning por eso. Que no se repita en el venv nuevo.

> `tf2onnx` y `onnx` **no van**: son herramienta de conversión puntual, no dependencia
> de runtime. Instalarlas a mano si hay que convertir otro modelo.

### 5.4 El único cambio de código

`src/api/func/reader_pipeline/forms/onnx_load.py`, al principio de `onnxLoader`:

```python
def onnxLoader(model_path: str, runtime_cfg, logger=None):
    ort.preload_dlls()      # carga CUDA/cuDNN desde site-packages (Windows)
    so = ort.SessionOptions()
```

**Esto es obligatorio.** Verificado en el código del wheel: `preload_dlls()` NO se llama
sola al importar. Sin ella, ORT no encuentra las DLL, no falla, y **se cae en silencio
al `CPUExecutionProvider`** — que es la peor forma de romperse, porque parece que anda.

Por eso el chequeo de providers de la §6 no es opcional.

### 5.5 Congelar

```bash
pip freeze > requirements.lock.txt
```

Y reescribir `requirements.txt` con los pines que quedaron (ver §8.2: hoy dice cosas
falsas, como que hay que usar un intérprete 3.8).

### 5.6 Electron

`src/backend-process.js` resuelve el intérprete así: `UNCA_PYTHON` > `.venv/Scripts/python.exe`
> `python` del PATH. Mientras el venv nuevo se llame `.venv312`, hay que arrancar con:

```bash
set UNCA_PYTHON=D:\Documentos\NewGen\UNCaLens\UNCa-Interfaz\.venv312\Scripts\python.exe
npm start
```

Recién cuando todo esté verde se renombra `.venv312` → `.venv` y la variable deja de
hacer falta.

## 6. Verificación (en orden; no seguir si una falla)

1. **CUDA carga**
   ```python
   import onnxruntime as ort
   ort.preload_dlls()
   print(ort.get_available_providers())   # debe incluir CUDAExecutionProvider
   ort.print_debug_info()                 # lista las DLL realmente cargadas
   ```
   `print_debug_info()` es la herramienta de diagnóstico si algo falla: dice qué DLL
   cargó, con qué CUDA se compiló ORT y cómo está el PATH.

2. **Tests**: `pytest` → 82 passed.

3. **Detección**: cargar `yolov7-tiny`, mandar `horses.jpg` por el WS → 5 cajas,
   y comprobar en el log que `ORT session providers` incluye CUDA.

4. **Clasificación**: cargar `saved_model_class` → clases 663, 813, 51. Bajar el umbral
   a 0.05 → 5 clases.

5. **TFLite**: cargar `efficientdet-lite0` y mandar un frame. Es el único camino que
   pasa por TensorFlow (`tf.lite.Interpreter`) y el que más chance tiene de romperse
   con el salto TF 2.12 → 2.21.

6. **Cliente**: `npm run typecheck` y `npm run build`.

7. **Electron**: `npm start` con `UNCA_PYTHON` apuntando al venv nuevo. Que la ventana
   abra, liste modelos y haga inferencia sobre un archivo.

## 7. Rollback

El venv viejo (`.venv`, Python 3.8) queda intacto durante todo el proceso. Si algo sale
mal: borrar `.venv312`, revertir la línea de `onnx_load.py`, y todo vuelve a estar como
hoy. **Nada de esto toca el sistema operativo** — CUDA vive adentro del venv — así que
el rollback es total y no deja restos.

Esa es justamente la ventaja grande de instalar CUDA por pip en vez de con el toolkit
global: se terminó el "ya instalé CUDA 12 y ahora se me rompió el otro proyecto".

## 8. Riesgos y hallazgos

### 8.1 TensorFlow: riesgo alto, y una oportunidad

El salto es **TF 2.12 → 2.21**, nueve versiones menores, e incluye **Keras 2 → Keras 3**
(TF 2.16+ trae Keras 3 por defecto). Es el mayor riesgo del paso.

Mitigantes encontrados al revisar el código:

- **No hay ningún modelo que use el backend `tensorflow`/Keras.** `models/` tiene
  `.tflite` ×2, `.onnx` ×2, `.pt`, `.pth` — ni un `.h5` ni un `.keras`. O sea que
  `keras_load.py` es código sin ejercitar: si Keras 3 lo rompe, no rompe nada vivo.
- Lo que **sí** importa es que `tflite_load.py` hace `tf.lite.Interpreter` (verificado):
  los dos EfficientDet dependen de TensorFlow. Por eso TF no se puede sacar sin más y
  por eso el techo de Python es 3.13.

> **Oportunidad para un paso futuro:** si `tf.lite.Interpreter` se reemplaza por
> `ai-edge-litert` (el paquete standalone de LiteRT que sucedió a `tflite-runtime`),
> **TensorFlow sale del proyecto entero**: se van ~600 MB de dependencia, desaparece
> todo el riesgo de Keras, y el techo de Python sube de 3.13 a lo que aguante ORT.
> No entra en este paso, pero conviene evaluarlo antes del paso 3.

### 8.2 numpy 1.23 → 2.x: riesgo bajo (verificado)

En 3.12 no se puede instalar numpy 1.23.5; va a entrar numpy 2.x. Escaneé el código
buscando las rupturas típicas:

- `np.float_`, `np.NaN`, `np.product`, `np.alltrue`, `np.in1d`, etc.: **cero hallazgos**.
- `copy=False`: los 10 usos son todos `ndarray.astype(..., copy=False)`, que **no
  cambió**. La ruptura de numpy 2 es en `np.array(x, copy=False)` (la función), que no
  se usa en ningún lado.

Conclusión: el código de pipelines debería pasar sin tocar nada. El riesgo real de
numpy 2 está en que TF/torch/ORT estén de acuerdo entre sí, no en el código propio.

### 8.3 Divergencias de CLAUDE.md — la "Fase 1" no existe

Verificado el 2026-08-13: **nada de lo que CLAUDE.md acredita a la "Fase 1 — entorno y
CI" está en el árbol.**

| CLAUDE.md afirma | realidad |
|---|---|
| `requirements.lock.txt` (freeze Windows) | **no existe** |
| `.gitattributes` con models en git-LFS | **no existe** |
| CI en `.github/workflows/ci.yml` | **no existe el directorio** |
| `pytest.ini` con `pythonpath=src` | **no existe** (por eso `pytest` necesita `PYTHONPATH=src`) |

Consecuencia inmediata y urgente: **`models/` NO está en git-LFS**. Los pesos se
commitean como blobs normales (`yolov7.pt`, 75 MB, ya está así). El
`saved_model_class.onnx` que se generó hoy pesa **171 MB y GitHub rechaza archivos de
más de 100 MB**: no se puede commitear como está. Hay que configurar LFS de verdad, o
dejar los pesos fuera del repo.

Este paso es el momento natural para cerrar de verdad lo que la Fase 1 dice haber hecho
(lock file, LFS, pytest.ini, CI con matriz 3.12), ya que es exactamente "entorno".
Se decide al ejecutar si entra acá o en un paso propio.

### 8.4 torch con CUDA en Windows

`torch` declara sus dependencias de nvidia solo para `platform_system == "Linux"`. En
Windows, la build con CUDA sigue viniendo del índice de PyTorch
(`--index-url https://download.pytorch.org/whl/cu124`), no de PyPI. El truco de
CUDA-por-pip **aplica solo a ONNX Runtime**. Como ORT es quien corre el detector y el
clasificador, alcanza; pero si alguna vez se quiere el `.pth` en GPU, hay que instalar
torch aparte.

## 9. Criterio de terminado

- Los 7 puntos de la §6 en verde.
- `.venv312` renombrado a `.venv`, `requirements.txt` y `requirements.lock.txt`
  actualizados y commiteados.
- CLAUDE.md actualizado: versión de Python, el `preload_dlls()` como parte del contrato
  de carga de ONNX, y las divergencias de §8.3 resueltas o anotadas como pendientes.
- El venv 3.8 se puede borrar sin miedo.
