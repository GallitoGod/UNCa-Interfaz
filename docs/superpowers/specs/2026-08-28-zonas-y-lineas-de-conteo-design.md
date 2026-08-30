# Zonas poligonales y líneas de conteo — diseño

**Fecha:** 2026-08-28
**Estado:** propuesta, sin código escrito
**Cierra:** pendiente #24 del Tier B (§4.4 y §4.5 de `docs/supervision-catalogo.md`)
**Depende de:** el Tier B ya hecho (2026-08-27) — `StreamSession`, `pipeline_generation`,
tracking con `trackers==2.6.0`

---

## 1. Qué se agrega

Dos cosas que se dibujan parecido y que **no son la misma cosa**:

- **Zona poligonal** (`sv.PolygonZone` + `PolygonZoneAnnotator`): un polígono sobre la
  escena que responde, frame a frame, cuántas detecciones caen adentro. **No necesita
  tracking**: es un test de punto en polígono, cuesta 0,03 ms.
- **Línea de conteo** (`sv.LineZone` + su annotator): un segmento que cuenta cuántos
  objetos lo cruzaron en cada sentido. Cuesta 0,05 ms pero **exige `tracker_id`**, así que
  hereda el Tier B entero.

Se implementan **en ese orden**, y no es arbitrario: la zona se puede montar y verificar
sola, sin tracking de por medio, y arrastra consigo la pieza cara que las dos comparten —el
editor de polígonos del cliente y su transformación de coordenadas—, que la línea después
reusa con dos vértices en vez de N.

**Fuera de alcance:** el mapa de calor (§4.6, +6,93 ms), el export de detecciones (§5.2) y
todo el Tier C.

---

## 2. Por qué esto entra, si el catálogo lo había descartado

`docs/supervision-catalogo.md` §4.5 juzgaba a las líneas como *"lo menos alineado de todo el
catálogo con lo que UNCaLens es: no dice nada sobre el modelo, dice algo sobre la escena"*,
con veredicto "sólo si aparece la necesidad de mostrar la app".

**Ese juicio estaba mal fundado y se corrigió el 2026-08-28.** Partía de asumir un único
objetivo —el banco de pruebas— cuando el sistema tiene **dos**: además de medir modelos,
UNCaLens busca **mostrar las capacidades de la IA de manera educativa**. Bajo ese segundo
objetivo, "habla de la escena y no del modelo" deja de ser una objeción: que la app cuente
autos cruzando una línea en vivo es precisamente la clase de cosa que vuelve legible, para
alguien de afuera, lo que el modelo está haciendo.

La lección no es sobre las líneas. Es que **el catálogo juzga contra los objetivos que tiene
escritos, y tenía uno de menos**. Toda evaluación futura pesa dos preguntas, no una:
(a) ¿mide o diagnostica al modelo?; (b) ¿hace visible lo que el modelo hace?
Una feature que sólo cumple (b) es legítima.

---

## 3. Las dos vidas: qué muere cuándo

Es la decisión de fondo, y la que el catálogo tenía mal. Su payload propuesto metía `zones`
adentro del bloque `session`, junto al tracking. Eso es un error: **`StreamSession.sync()`
resetea por `pipeline_generation`, o sea al cambiar de modelo**, y una zona **no debe morir
cuando cambia el modelo** — el polígono describe *la escena*, y comparar dos modelos sobre
la misma zona es exactamente el caso de uso del banco de pruebas.

| Estado | Se resetea en | Por qué |
|---|---|---|
| tracker, smoother, trazas, **contador de la línea** | `sync()` (cambio de modelo) **y** cierre de la conexión | Todo depende de `tracker_id`. Si las identidades se reinician y el contador no, **cuenta doble**. |
| **geometría** de zonas y líneas | **sólo** el cierre de la conexión | Describe la escena, no el modelo. |

O sea: la geometría y el conteo **no son el mismo estado y no cuelgan del mismo dueño**.
`StreamSession` se parte en dos cajones y `sync()` vacía uno solo. La línea queda repartida:
su **segmento** en el cajón de la escena, su **contador** en el del tracker.

**Decidido con el usuario (2026-08-28): la geometría muere con la fuente. No se persiste.**
No hay `localStorage` de zonas, y por lo tanto tampoco existe el problema de con qué clave
identificar una fuente (el nombre de archivo es frágil: `Prueba.mp4` y `Prueba_4x3.mp4` son
escenas distintas, y renombrar cualquiera lo rompe).

**El agujero conocido, dicho de frente:** la conexión también se cae en una **reconexión por
backoff**, y ahí se pierde la zona sin que el usuario haya cambiado de fuente. Con el backend
en localhost es raro, y se acepta. Es el único caso donde "muere con la fuente" miente.

**`stateful=false` NO apaga la geometría.** El camino one-shot de imágenes declara que no hay
**memoria temporal** que construir; una zona sobre una foto suelta es perfectamente legítima
(contar vehículos en una región de una imagen). Así que el cajón de la escena **no se gatea
por `_stateful`** — el de tracking sí, como hoy. Consecuencia: sobre una imagen fija las
zonas funcionan y las líneas quedan inertes, igual que el tracking.

---

## 4. Coordenadas: normalizadas `[0,1]`

El polígono se guarda y viaja como fracciones del ancho y el alto del frame, no en píxeles.

**El motivo NO es la portabilidad entre resoluciones.** Ese era el argumento cuando estaba
sobre la mesa persistir las zonas; sin persistencia, el tamaño del frame no cambia mientras
la zona vive y no hay nada que portar. El motivo real es otro y es mejor: **hace que el
cliente no necesite saber la resolución del frame.**

El canvas es un elemento reemplazado con `max-h-full max-w-full` y sin `width`/`height` en
CSS (`VisionWorkspace.tsx:45-48`), así que el navegador le conserva la relación de aspecto
del bitmap: `rect.width / canvas.width` y `rect.height / canvas.height` son **el mismo
factor**. Con almacenamiento normalizado, la conversión del click colapsa a:

```ts
const rect = canvas.getBoundingClientRect();
const nx = (ev.clientX - rect.left) / rect.width;   // 0..1
const ny = (ev.clientY - rect.top)  / rect.height;  // 0..1
```

`canvas.width` **no aparece**. Y del otro lado el backend ya calcula `h, w = img_bgr.shape[:2]`
en `render_detection` para el `auto_scale` (`tasks/detection.py:222`), así que multiplicar
sale gratis. `PolygonZone` quiere enteros: `np.round(pts * [w, h]).astype(int)` **una vez al
construir la zona**, no por frame.

**Defensivo:** aunque hoy el elemento conserva el aspecto, el cálculo debe derivar la caja de
contenido explícitamente en vez de asumirlo. Son seis líneas y vuelven al editor inmune a un
cambio futuro de CSS que fuerce `width` y `height` a la vez (ahí `object-contain` empezaría a
poner bandas *adentro* del elemento y la fórmula de arriba se rompería en silencio).

### 4.1 El aspecto: se mide, no se declara

No hay que preguntarle a nadie cuál es el aspecto de la fuente — ni a la cámara, ni al
contenedor del archivo, ni al usuario. El frame lo lleva encima, y de hecho está medido
**dos veces, independientemente**:

- **cliente**: `canvas.width/height`, que `present.ts:79-81` fija desde el bitmap recibido;
- **backend**: `img_bgr.shape[:2]`.

Y como el backend **no redimensiona** (anota sobre el frame que le llegó y lo re-encodea del
mismo tamaño, verificado en `tasks/detection.py:215-241`), esos dos números **tienen que
coincidir**. Eso es una invariante gratis para un test.

Se guarda **`w` y `h`, no el cociente**: son dos enteros, es estrictamente más información, y
hace que el día que falle el mensaje diga *"la zona se dibujó sobre 1440×1080 y están
llegando frames de 1920×1080"* en vez de *"1.333 ≠ 1.778"*.

**Es una aserción, no una función.** Sin persistencia el chequeo no puede dispararse en uso
normal: la fuente tiene la resolución que tiene y la zona muere con ella. Si se dispara, es un
bug. Por eso **loguea y no dibuja la zona**, y **no lleva interfaz** — construirle UI sería
inflar el panel por un caso que no ocurre. Tolerancia 1%: los aspectos reales están lejísimos
entre sí (4:3 = 1,333 · 3:2 = 1,500 · 16:10 = 1,600 · 16:9 = 1,778; el par más cercano se
lleva 4%) y el ruido de redondeo es despreciable (854×480 = 1,779 contra 1,778, o sea 0,08%).

---

## 5. Cómo llega la geometría al backend: un canal de control en el WS

**Esta es la decisión estructural del documento.**

Los ajustes del Tier A y B viajan por `POST /config/draw` a un **singleton de proceso**, y
está bien: son preferencias **del usuario**, iguales para toda la app. **Una zona no lo es.**
Es geometría atada a **una conexión** (muere con ella, §3), y un endpoint HTTP no sabe a qué
WebSocket le está hablando.

**Propuesta: el polígono viaja por el propio WebSocket, como mensaje de texto del cliente.**

Razones:

1. **Direcciona sola.** El mensaje llega por la conexión a la que pertenece la zona. No hay
   que inventar un registro de sesiones vivas ni difundir a todas.
2. **No hay carrera.** Con HTTP, el `POST` y los frames en vuelo compiten: no está definido si
   el frame N se compone con la zona vieja o la nueva. Por el mismo canal, el orden **es** el
   orden.
3. **No inventa un dueño nuevo.** La conexión ya es el dueño elegido y verificado en el
   Tier B; esto sólo le agrega una puerta de entrada.

**Alternativa considerada y rechazada:** `POST /config/zones` sobre un singleton, con el
cliente encargado de limpiarlo al cambiar de fuente. Se rechaza porque viola la regla 4 del
catálogo (*"el backend resetea, el cliente no recuerda"*): el reseteo dejaría de ser
estructural y pasaría a depender de que alguien se acuerde de llamar a algo — que es
exactamente el problema que `StreamSession` existe para no tener.

### 5.1 La trampa del canal de texto

`_decode_frame` (`mainAPI.py:344-356`) **ya usa el canal de texto** para frames en base64, por
compatibilidad. Un mensaje de control en texto cae en esa rama, no decodifica como imagen,
devuelve `None` y el cliente recibe `frame_invalido`. Hay que discriminar **antes**:

> intentar `json.loads` sobre `message["text"]`; si resulta un **dict con clave `"type"`**, es
> control. Un JPEG en base64 nunca parsea como objeto JSON, así que el discriminador no es
> ambiguo.

### 5.2 La invariante que no se toca

**El WS sigue respondiendo UN mensaje por mensaje recibido.** Un mensaje de control se
contesta con un **ack JSON** que devuelve el **estado efectivo** de la geometría (misma regla
que `/config/draw`: se devuelve lo que quedó, no lo que se pidió). Así el cliente puede
mostrar que la zona entró, y el "siempre responde" que evita el deadlock del stream se
mantiene sin excepciones.

```jsonc
// cliente -> backend, por el WS
{ "type": "geometry",
  "frame": { "w": 1440, "h": 1080 },        // sobre qué se dibujó (§4.1)
  "zones": [ { "id": "z1", "points": [[0.10,0.20],[0.80,0.20],[0.80,0.75],[0.10,0.75]] } ],
  "lines": [ { "id": "l1", "a": [0.10,0.50], "b": [0.90,0.50] } ] }

// backend -> cliente, ack (estado efectivo)
{ "type": "geometry_ack", "zones": 1, "lines": 1, "error": null }
```

Es **declarativo y completo**, no incremental: el mensaje trae *toda* la geometría vigente,
no un delta. Un delta obliga a las dos puntas a coincidir sobre un historial, y ese es
justamente el acoplamiento que este proyecto viene evitando en el resto de la superficie.

---

## 6. El editor en el cliente

Lo barato es el backend. **Lo caro es esto**, y el catálogo ya lo marcaba: *"eso es un editor,
no un toggle"*.

**Se dibuja con un SVG en `overlayRoot`, no con un segundo canvas.** Un
`<svg viewBox="0 0 1 1" preserveAspectRatio="none">` superpuesto renderiza las coordenadas
normalizadas **sin una sola línea de aritmética** — escala el navegador. Y los vértices
arrastrables salen como `<circle>` con sus propios eventos, así que **el hit-testing tampoco
lo escribimos nosotros**. Dos detalles obligatorios:

- `vector-effect="non-scaling-stroke"` en trazos y vértices, o el `viewBox` de 1×1 deforma el
  grosor hasta volverlo invisible;
- `overlayRoot` hoy es `pointer-events-none` (`VisionWorkspace.tsx:51`): se habilitan **sólo
  mientras se edita**, para no robarle el cursor al resto de la app.

### 6.1 La excepción deliberada al "el cliente no dibuja"

**Mientras se está dibujando, el polígono lo pinta el cliente.** No hay alternativa: no se
puede ir y volver al backend a la velocidad del arrastre. Recién al **confirmar** pasa a
dibujarlo el backend con `PolygonZoneAnnotator`.

Eso roza la regla del paso 3 (*"el cliente no contiene ni una línea que dibuje una caja"*).
La lectura es que **no la viola**: no está dibujando un **resultado del modelo**, está
dibujando un **control**. Pero queda escrito acá porque, sin esta nota, dentro de tres meses
se lee como una regresión.

**El riesgo concreto es el salto**: si las dos representaciones no coinciden exactamente, el
polígono "salta" al confirmarse. Lo evita que ambas consuman **las mismas coordenadas
normalizadas** — el SVG las renderiza directo y el backend las multiplica por `w,h`.

### 6.2 La sección `Zonas`

Ya estaba anticipada en el pendiente #25: el patrón de `shared/ui/Section` soporta una sección
que contiene un **modo**, no sólo interruptores, y su encabezado plegado mostraría `ZONAS · 1`.

Cuelga de la **misma condición** que `Render` y `Seguimiento` (`panelDeRenderAplica()`): con un
clasificador no hay geometría sobre la que contar nada.

**La regla 3 del catálogo se aplica igual que en `Seguimiento`:** la línea de conteo **exige
tracking**, así que se dibuja indentada con guía vertical bajo un maestro, pedirla prende el
seguimiento sola, y el backend sigue siendo la autoridad (`update_draw_config()` ya fuerza esa
coherencia para `smoothing` y `traces`; las líneas se suman a esa misma regla). La **zona
no** — no depende de nada y va suelta.

---

## 7. Qué se mide, y cómo no engañarse

Los buckets aislados son medición; comparar `avg_with_draw_ms` entre configs **no lo es** —
la varianza de inferencia entre corridas (11,4–13,7 ms medidos el mismo día con el mismo
modelo) es varias veces mayor que todo lo que se está midiendo acá. Y el `PerfMeter` tiene
ventana de 300 frames: hay que **recargar el modelo entre configuraciones** (hace
`perf.reset()`) o se promedia con la corrida anterior.

A medir:

- costo del test de pertenencia con **muchas** detecciones. El catálogo midió 0,03 ms, pero
  `best` sobre material aéreo entrega **~70 detecciones por frame**, no 6. Hay que confirmar
  que escala como se espera.
- costo del `PolygonZoneAnnotator` dentro de `draw_ms`.
- que prender una zona **no cambie ni una detección**: es una capa de lectura, no un filtro.

---

## 8. Riesgos

1. **El salto al confirmar** (§6.1). Mitigado por coordenadas compartidas; hay que verificarlo
   a ojo, no por test.
2. **Contar con `-1`.** `LineZone` agrupa por `tracker_id` y todos los tracks sin confirmar
   comparten ese valor: sin filtrar, el contador es ficción. Es la misma regla que ya fuerza
   `StreamSession._con_identidad()`, y hay que reusarla, no reescribirla.
3. **El polígono degenerado.** Menos de 3 vértices, auto-intersección, área cero. El backend
   valida y responde `error` en el ack en vez de romper el frame — el hot path prefiere
   dibujar algo antes que caerse (mismo criterio que un `box_style` desconocido).
4. **Las etiquetas ya tapan la imagen** (pendiente #27, abierto). Con 70 detecciones los
   carteles cubren la escena; sumarle el contador de una zona encima **empeora un problema que
   ya existe**. #27 debería ir antes, o al menos junto.

---

## 9. Plan de implementación

1. **El cajón de la escena en `StreamSession`**, vacío, con sus tests: geometría que
   `sync()` **no** toca, tracking que sí. Verificar el andamiaje **antes** de colgarle nada —
   es el consejo que ya salió bien en el Tier B.
2. **El canal de control del WS** (§5): discriminador, ack, invariante de "siempre responde".
   Sin geometría real todavía, sólo el transporte.
3. **Zona en el backend**: `PolygonZone` + annotator + el guard de aspecto.
4. **El editor SVG en el cliente** y la sección `Zonas`. Es el paso más largo.
5. **Línea de conteo**: reusa todo lo anterior con dos vértices, y suma el contador al cajón
   del tracker (no al de la escena).
6. **Actualizar `CLAUDE.md`** (§4, §5 y el pendiente #24) y el catálogo.

Los pasos 1–3 se verifican por HTTP+WS sin tocar el cliente; el 4 por CDP, como el panel
`Seguimiento`.
