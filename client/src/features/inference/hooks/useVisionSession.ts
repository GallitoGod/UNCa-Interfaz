// useVisionSession.ts — orquestador de la sesion de inferencia en vivo.
// Reacciona a la fuente activa (streamStore): maneja el media (getUserMedia / src),
// arranca el transporte WS y, por cada respuesta, repinta el frame y delega en la
// estrategia del modelo activo (presentFrame). Es el dueno del ciclo de vida.

import { useEffect, useRef, type RefObject } from 'react';
import { useStreamStore } from '../store/streamStore';
import { useUiStore } from '@/app/store/uiStore';
import { useWorkspaceStore } from '@/features/vision-workspace/store/workspaceStore';
import { presentFrame, releaseOverlay } from '@/features/vision-workspace/services/present';
import {
  sendSingleFrame,
  startVideoStream,
  type StreamPayload,
  type VideoStreamHandle,
} from '../services/videoStream';

interface SessionRefs {
  videoRef: RefObject<HTMLVideoElement | null>;
  canvasRef: RefObject<HTMLCanvasElement | null>;
  overlayRef: RefObject<HTMLDivElement | null>;
}

export function useVisionSession({ videoRef, canvasRef, overlayRef }: SessionRefs) {
  const source = useStreamStore((s) => s.source);
  const activeView = useUiStore((s) => s.activeView);
  // Disparadores de re-inferencia para fuentes ESTATICAS (ver stillNonce en el store).
  const stillNonce = useStreamStore((s) => s.stillNonce);
  const activeType = useWorkspaceStore((s) => s.activeModel?.type ?? null);

  // Refs vivos a la sesion en curso, para que el effect de navegacion pueda
  // pausar/reanudar SIN re-ejecutar el effect principal (que reconstruiria todo).
  const handleRef = useRef<VideoStreamHandle | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  // Frame fijo de la fuente 'file-image' y el render vigente, guardados para poder
  // volver a inferir sin re-ejecutar el effect principal (que revoca el objectURL).
  const stillRef = useRef<HTMLCanvasElement | null>(null);
  const renderRef = useRef<((payload: StreamPayload, src: HTMLCanvasElement) => void) | null>(null);

  useEffect(() => {
    if (!videoRef.current || !canvasRef.current || !overlayRef.current) return;
    if (source.kind === 'none') return;

    // Re-vinculo con tipo no-nulo explicito: asi los closures anidados (start /
    // cleanup / onload) ven elementos no nulos sin perder el narrowing.
    const video: HTMLVideoElement = videoRef.current;
    const canvas: HTMLCanvasElement = canvasRef.current;
    const overlay: HTMLDivElement = overlayRef.current;

    let handle: VideoStreamHandle | null = null;
    let mediaStream: MediaStream | null = null;
    let cancelled = false;

    // Render de un frame: lee modelo activo + colores en el momento (live).
    const render = (payload: StreamPayload, src: HTMLCanvasElement) => {
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      const { activeModel, drawSettings } = useWorkspaceStore.getState();
      presentFrame({
        canvas,
        ctx,
        overlayRoot: overlay,
        source: src,
        payload,
        modelType: activeModel?.type ?? null,
        drawSettings,
      });
    };
    // Se expone para los re-envios de fuente estatica (effect de mas abajo).
    renderRef.current = render;

    const setStatus = useStreamStore.getState().setStatus;

    // Alinea la pausa con la vista activa actual (leida fresh). Se llama al crear el
    // handle para cubrir la carrera de navegar antes de que getUserMedia resuelva.
    function syncToView() {
      const onInference = useUiStore.getState().activeView === 'inference';
      mediaStreamRef.current?.getVideoTracks().forEach((t) => (t.enabled = onInference));
      if (onInference) handleRef.current?.resume();
      else handleRef.current?.pause();
    }

    async function start() {
      try {
        if (source.kind === 'camera') {
          mediaStream = await navigator.mediaDevices.getUserMedia({
            video: {
              deviceId: { exact: source.deviceId },
              width: { ideal: 1280 },
              height: { ideal: 720 },
              frameRate: { ideal: 30 },
            },
          });
          if (cancelled) {
            mediaStream.getTracks().forEach((t) => t.stop());
            return;
          }
          mediaStreamRef.current = mediaStream;
          video.srcObject = mediaStream;
          video.muted = true;
          await video.play();
          // mirror:true -> espejo solo para camara.
          handle = startVideoStream({ videoElement: video, mirror: true, onMessage: render, onStatus: setStatus });
          handleRef.current = handle;
          syncToView(); // si arrancamos fuera de Inferencia, nacer en pausa
        } else if (source.kind === 'file-video') {
          video.srcObject = null;
          video.src = source.url;
          video.muted = true;
          video.loop = true;
          await video.play();
          handle = startVideoStream({ videoElement: video, mirror: false, onMessage: render, onStatus: setStatus });
          handleRef.current = handle;
          syncToView();
        } else if (source.kind === 'file-image') {
          // One-shot: cargar la imagen, dibujarla a un canvas temporal y mandar 1 frame.
          const img = new Image();
          img.onload = () => {
            if (cancelled) return;
            const tmp = document.createElement('canvas');
            tmp.width = img.naturalWidth;
            tmp.height = img.naturalHeight;
            tmp.getContext('2d')?.drawImage(img, 0, 0);
            // Se cachea para poder re-inferir el mismo frame al cambiar de modelo o
            // de umbral, sin volver a cargar la imagen ni tocar el objectURL.
            stillRef.current = tmp;
            sendSingleFrame(tmp, render);
          };
          img.src = source.url;
        }
      } catch (err) {
        useStreamStore.getState().setError(err instanceof Error ? err.message : String(err));
      }
    }
    void start();

    // Cleanup: al cambiar de fuente o desmontar, cerrar todo y liberar recursos.
    // OJO: navegar a Modelos NO desmonta InferenceView (queda oculta), asi que este
    // cleanup NO corre al navegar — solo al cambiar de fuente o cerrar la app. Por eso
    // la pausa/reanudacion vive en el effect de abajo, no aca.
    return () => {
      cancelled = true;
      handle?.close();
      if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop());
      handleRef.current = null;
      mediaStreamRef.current = null;
      stillRef.current = null; // el frame fijo pertenece a ESTA fuente
      video.srcObject = null;
      video.removeAttribute('src');
      if (source.kind === 'file-video' || source.kind === 'file-image') {
        URL.revokeObjectURL(source.url);
      }
    };
  }, [source, videoRef, canvasRef, overlayRef]);

  // Navegacion (regla SDD 4.1.2): al salir de Inferencia se pausa la sesion (loop +
  // <video> + captura de camara) sin cerrar el WS ni soltar el permiso; al volver se
  // reanuda. Solo togglea la sesion existente; crearla/destruirla es del effect de arriba.
  useEffect(() => {
    const onInference = activeView === 'inference';
    mediaStreamRef.current?.getVideoTracks().forEach((t) => (t.enabled = onInference));
    if (onInference) handleRef.current?.resume();
    else handleRef.current?.pause();
  }, [activeView]);

  // Cambio de TIPO de modelo: soltar la capa de la estrategia anterior en el acto.
  // Con camara/video el proximo frame la limpiaria solo, pero con una imagen fija no
  // hay proximo frame y los labels quedaban pegados en pantalla.
  useEffect(() => {
    const canvas = canvasRef.current;
    const overlay = overlayRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !overlay || !ctx) return;
    releaseOverlay(overlay, {
      canvas,
      ctx,
      overlayRoot: overlay,
      frameWidth: canvas.width,
      frameHeight: canvas.height,
      settings: useWorkspaceStore.getState().drawSettings,
    });
  }, [activeType, canvasRef, overlayRef]);

  // Re-inferir el frame fijo cuando cambia un parametro (modelo, umbral). El envio
  // inicial lo hace el effect principal; aca solo se atienden los re-envios, por eso
  // se ignora el nonce 0.
  useEffect(() => {
    if (stillNonce === 0) return;
    if (source.kind !== 'file-image') return; // camara/video ya refrescan solos
    const still = stillRef.current;
    const render = renderRef.current;
    if (!still || !render) return;
    sendSingleFrame(still, render);
  }, [stillNonce, source.kind]);
}
