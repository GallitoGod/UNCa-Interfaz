# render/session.py — la memoria POR CONEXION del stream.
#
# Por que existe (Tier B del catalogo de supervision, ver docs/supervision-catalogo.md):
# el tracking, el suavizado y las trazas son las primeras piezas del sistema que
# RECUERDAN el frame anterior. Todo el resto del hot path es sin estado por diseno
# (reforma 8: los pasos del pipeline son closures puros y lo que varia por frame
# viaja en el dict 'meta'). Esa memoria nueva necesita un dueno explicito.
#
# El dueno NO puede ser el ModelController: es un singleton de proceso, asi que dos
# clientes conectados a la vez compartirian el tracker y se mezclarian las
# identidades. Tampoco puede ser el pipeline que arma build_pipeline(): son closures
# stateless a proposito, y meterles estado por frame rompe la reforma 8.
#
# El dueno correcto es la CONEXION del WebSocket, y elegirlo resuelve tres de los
# cuatro casos de reseteo de forma ESTRUCTURAL, sin que nadie tenga que acordarse
# de llamar a nada:
#   - cambio de fuente        -> el cliente cierra el WS y abre otro  -> sesion nueva
#   - reconexion tras caida   -> WS nuevo                             -> sesion nueva
#   - imagen fija (one-shot)  -> WS efimero de un solo frame          -> muere sola
#
# El cuarto NO se resuelve solo, y es el importante: al CAMBIAR DE MODELO el cliente
# mantiene abierto el mismo WebSocket (el effect de useVisionSession depende de la
# fuente, no del modelo). Sin nada mas, el tracker seguiria arrastrando tracks
# nacidos con otro pipeline, donde los class_id significaban otra cosa. Para eso
# esta sync(): el controller lleva un contador de generacion que avanza en cada
# carga y descarga, la sesion recuerda bajo cual nacio, y cuando no coinciden se
# olvida de todo. Es el mismo truco que la 'version' de DrawConfig con el cache de
# annotators, y por la misma razon: se auto-repara, en vez de exigir que el endpoint
# REST le avise a mano a cada conexion viva.
#
# Por que este archivo vive en render/: todo lo que esta memoria guarda existe
# unicamente para cambiar el frame compuesto. El usuario lo prende y lo apaga desde
# el mismo panel Render que los colores, y sus ajustes viajan por el mismo
# POST /config/draw.
#
# Por que el paquete 'trackers' y no sv.ByteTrack: el de supervision esta deprecado
# desde la 0.28 y SE ELIMINA en la 0.31. El reemplazo oficial de Roboflow no arrastra
# ninguna dependencia nueva (verificado: todo lo que pide ya lo trajo supervision) y
# cambia el metodo de update_with_detections() a update().

import supervision as sv
from trackers import ByteTrackTracker

from .annotators import annotators_for


class StreamSession:
    """
    Estado que sobrevive de un frame al siguiente DENTRO de una conexion del stream.

    Se crea una por WebSocket aceptado y se descarta al cerrarse. No es thread-safe
    y no necesita serlo: el protocolo del stream es de UN frame en vuelo, asi que
    nunca hay dos llamadas a process() de la misma sesion al mismo tiempo.
    """

    def __init__(self, stateful: bool = True):
        """
        stateful=False para el camino one-shot de imagenes (una foto suelta no es una
        secuencia: no hay nada que rastrear entre frames que no existen). Se declara
        explicito desde el cliente en vez de deducirlo, porque una conexion que
        todavia no recibio su segundo frame es indistinguible de una que nunca lo va
        a recibir.
        """
        self._stateful = bool(stateful)
        # Generacion del pipeline bajo la que se construyo la memoria vigente.
        # None = todavia no se sincronizo con ninguna (sesion recien nacida).
        self._generation = None
        # Tracker vigente, o None si el tracking esta apagado o esta conexion no
        # recuerda nada. Se construye perezosamente: mientras nadie prenda el toggle,
        # esta clase no cuesta nada.
        self._tracker = None
        # Umbral de confianza con el que se construyo el tracker. Sus umbrales se
        # fijan al construirlo, asi que si el usuario mueve el slider hay que rehacerlo.
        self._tracker_conf = None
        # Suavizador vigente y la ventana con la que se armo (cambiarla lo rehace).
        self._smoother = None
        self._smoother_length = None
        # Annotator de trazas. NO puede vivir en el cache global de render/annotators.py
        # aunque se le parezca: tiene estado propio (su atributo 'trace' guarda el
        # recorrido de cada objeto), y ese cache es compartido por todas las conexiones.
        # Se rehace cuando cambia la config o la resolucion, igual que sus hermanos.
        self._trace = None
        self._trace_key = None

    @property
    def stateful(self) -> bool:
        """False si esta conexion es una foto suelta y no debe recordar nada."""
        return self._stateful

    @property
    def generation(self):
        """Generacion del pipeline con la que esta sincronizada, o None si ninguna."""
        return self._generation

    def sync(self, generation: int) -> bool:
        """
        Alinea la sesion con la generacion actual del pipeline.

        Devuelve True si hubo que olvidar la memoria acumulada (cambio de modelo),
        False si venia alineada. Se llama UNA vez por frame, antes de process():
        es una comparacion de enteros, mas barata que cualquier alternativa que
        exija avisarle a la conexion desde afuera.

        La primera sincronizacion de una sesion nueva NO cuenta como reseteo: no
        habia nada que olvidar.
        """
        if self._generation == generation:
            return False
        primera_vez = self._generation is None
        self._generation = generation
        if primera_vez:
            return False
        self.reset()
        return True

    @property
    def tracking_activo(self) -> bool:
        """True si esta conexion esta rastreando de verdad (para tests y logs)."""
        return self._tracker is not None

    @property
    def suavizado_activo(self) -> bool:
        """True si esta conexion esta promediando posiciones (para tests y logs)."""
        return self._smoother is not None

    @property
    def trazas_activas(self) -> bool:
        """True si esta conexion viene acumulando recorridos (para tests y logs)."""
        return self._trace is not None

    def reset(self) -> None:
        """
        Olvida todo lo recordado de frames anteriores, dejando la sesion como recien
        creada (salvo 'stateful', que es una propiedad de la conexion, no del estado).
        """
        # Se descarta el objeto entero en vez de llamar a su reset(): asi el proximo
        # frame lo reconstruye con el umbral que este vigente en ese momento, que es
        # justamente lo que puede haber cambiado.
        self._tracker = None
        self._tracker_conf = None
        self._smoother = None
        self._smoother_length = None
        self._trace = None
        self._trace_key = None

    def _tracker_para(self, conf_threshold: float):
        """
        El tracker vigente, construyendolo si hace falta.

        Los dos umbrales salen del umbral de confianza del USUARIO, y esto no es un
        detalle: los defaults del paquete (track_activation_threshold=0.7,
        high_conf_det_threshold=0.6) estan pensados para un pipeline que le entrega al
        tracker TODAS las detecciones, incluidas las de baja confianza, para que
        ByteTrack haga su asociacion en dos pasadas (primero las confiables, despues
        las dudosas para recuperar objetos tapados).

        Aca eso no pasa: nuestro postprocesador YA filtro por el umbral del usuario
        antes de que el tracker vea nada, asi que la banda de baja confianza que
        ByteTrack querria explotar llega vacia por construccion. Dejar los defaults
        tiene una consecuencia concreta y medida: con los tres modelos del repo
        (umbrales 0.5 / 0.3 / 0.25), una deteccion de 0.5 NUNCA recibe un tracker_id
        y el toggle queda prendido sin hacer absolutamente nada — exactamente el
        sintoma que el catalogo prohibe. El que bloquea es high_conf_det_threshold:
        bajar solo track_activation_threshold no alcanza (verificado).
        """
        if self._tracker is None or self._tracker_conf != conf_threshold:
            self._tracker = ByteTrackTracker(
                track_activation_threshold=conf_threshold,
                high_conf_det_threshold=conf_threshold,
            )
            self._tracker_conf = conf_threshold
        return self._tracker

    @staticmethod
    def _con_identidad(detections):
        """
        Solo las detecciones con un tracker_id confirmado (>= 0).

        Es la regla que comparten el suavizado y las trazas, y existe porque los
        tracks sin confirmar comparten TODOS el valor -1: cualquier cosa que agrupe
        por tracker_id los toma por un mismo objeto. En el suavizado eso funde varias
        cajas en una; en las trazas dibuja una estela que salta de un objeto a otro,
        que ademas es justo el sintoma que el usuario deberia leer como "el tracker
        esta confundiendo identidades". Un artefacto que imita al bug que la
        herramienta sirve para detectar es peor que no tener la herramienta.
        """
        if detections.tracker_id is None:
            return detections[[False] * len(detections)]
        return detections[detections.tracker_id >= 0]

    def anotar_trazas(self, scene, detections, cfg, resolution_wh):
        """
        Dibuja la estela de cada objeto rastreado sobre 'scene' y la devuelve.

        La llama render_detection como una capa mas. Vive aca y no en render/ porque
        el annotator es ESTADO de la conexion, no configuracion compartida: dos
        clientes mirando fuentes distintas no pueden compartir un buffer de recorridos.

        Devuelve la escena intacta si las trazas estan apagadas o si esta conexion no
        recuerda nada (foto suelta).
        """
        if not self._stateful or cfg is None or not getattr(cfg, "traces", False):
            if self._trace is not None:
                self._trace = None
                self._trace_key = None
            return scene

        largo = int(getattr(cfg, "traces_length", 30))
        clave = (cfg.version, resolution_wh, largo)
        if self._trace is None or self._trace_key != clave:
            # Rehacer pierde los recorridos acumulados. Es aceptable: se vuelven a
            # llenar en 'largo' frames (~1 s a 30 fps) y solo pasa cuando el usuario
            # toca un ajuste, no en el hot path.
            ann = annotators_for(cfg, resolution_wh)
            self._trace = sv.TraceAnnotator(
                color=sv.Color.from_hex(cfg.bbox_color),
                thickness=ann.thickness,
                trace_length=largo,
                # INDEX y no el default CLASS: con ColorLookup.CLASS supervision
                # ignora el color elegido y pinta una estela por clase. Misma trampa
                # que documenta render/annotators.py.
                color_lookup=sv.ColorLookup.INDEX,
            )
            self._trace_key = clave

        rastreadas = self._con_identidad(detections)
        if len(rastreadas) == 0:
            # Con cero detecciones el annotator no dibuja nada (verificado), asi que
            # llamarlo seria trabajo puro. Ademas evita tocar una escena que en el
            # camino "sin detecciones" todavia no fue copiada.
            return scene
        return self._trace.annotate(scene=scene, detections=rastreadas)

    def _suavizar(self, detections, length: int):
        """
        Promedia la posicion de cada objeto sobre los ultimos 'length' frames.

        Las detecciones se PARTEN en dos antes de suavizar, y eso no es una
        optimizacion: es correccion. sv.DetectionsSmoother agrupa por tracker_id, y
        todos los tracks sin confirmar comparten el valor -1, asi que para el
        suavizador son EL MISMO objeto. Verificado: dos detecciones separadas ambas
        con tracker_id=-1 entran, y sale UNA SOLA caja promediada entre las dos, en un
        punto de la imagen donde no hay nada. Entran 2, sale 1.

        Asi que por el suavizador pasan solo las que tienen identidad; las demas se
        devuelven intactas y se reunen despues. El orden cambia (primero las
        suavizadas), lo cual es inofensivo: el color es unico por eleccion del usuario
        y las etiquetas se calculan sobre el mismo objeto.
        """
        if detections.tracker_id is None:
            # Sin identidades no hay nada que agrupar. No deberia pasar (el singleton
            # garantiza que suavizado implica tracking), pero no vale la pena romper
            # un frame por eso.
            return detections

        if self._smoother is None or self._smoother_length != length:
            self._smoother = sv.DetectionsSmoother(length=length)
            self._smoother_length = length

        con_identidad = self._con_identidad(detections)
        sin_identidad = detections[detections.tracker_id < 0]

        suavizadas = self._smoother.update_with_detections(con_identidad)
        if len(sin_identidad) == 0:
            return suavizadas
        return sv.Detections.merge([suavizadas, sin_identidad])

    def process(self, detections, cfg, conf_threshold: float = 0.0):
        """
        Punto de insercion de todo lo que necesita memoria entre frames: recibe el
        sv.Detections que produjo el pipeline y devuelve el que se va a dibujar.

        Corre ENTRE controller.inference() y controller.render_result(), no adentro
        del pipeline (ver el encabezado del modulo).

        El tracker NUNCA descarta detecciones (verificado): entran N y salen N. Las
        que todavia no tienen identidad confirmada salen con tracker_id = -1, que es
        el valor que el resto del sistema tiene que saber leer. Eso importa porque
        significa que prender el tracking no puede hacer desaparecer una caja.
        """
        # Apagado, o conexion que no recuerda (foto suelta): se suelta lo que hubiera
        # quedado construido y se devuelve el resultado crudo, sin tocar.
        if not self._stateful or cfg is None or not getattr(cfg, "tracking", False):
            if self._tracker is not None:
                self.reset()
            return detections

        # Se llama SIEMPRE, incluso con cero detecciones: un frame vacio es
        # informacion para el tracker (los tracks vivos envejecen y expiran).
        # Saltearlo dejaria vivo para siempre a un objeto que ya se fue de escena.
        rastreadas = self._tracker_para(conf_threshold).update(detections)

        if not getattr(cfg, "smoothing", False):
            if self._smoother is not None:
                self._smoother = None
                self._smoother_length = None
            return rastreadas

        return self._suavizar(rastreadas, int(getattr(cfg, "smoothing_length", 5)))
