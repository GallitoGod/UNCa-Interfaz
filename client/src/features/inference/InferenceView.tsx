// InferenceView.tsx — vista de inferencia con la anatomia de 3 zonas del spec:
//   izquierda  -> fuente (camara/archivo) + lista de modelos
//   centro     -> feed (heroe) + barra de transporte
//   derecha    -> parametros + metricas + errores
// Es duena de los refs (video/canvas/overlay) y del orquestador (useVisionSession).

import { useRef, useState } from 'react';
import { VisionWorkspace } from '@/features/vision-workspace/components/VisionWorkspace';
import { useWorkspaceStore } from '@/features/vision-workspace/store/workspaceStore';
import { Tabs } from '@/shared/ui/Tabs';
import { Button } from '@/shared/ui/Button';
import { Badge } from '@/shared/ui/Badge';
import { SectionLabel } from '@/shared/ui/SectionLabel';
import { useStreamStore } from './store/streamStore';
import { useVisionSession } from './hooks/useVisionSession';
import { CameraSource } from './components/CameraSource';
import { FileSource } from './components/FileSource';
import { ModelSelector } from './components/ModelSelector';
import { ConfidenceSlider } from './components/ConfidenceSlider';
import { RenderSettings, panelDeRenderAplica } from './components/RenderSettings';
import { MetricsHUD } from './components/MetricsHUD';
import { MetricsPanel } from './components/MetricsPanel';
import { LogPanel } from './components/LogPanel';
import { Recorder } from './components/Recorder';
import { DrawSettingsModal } from './components/DrawSettingsModal';

type SourceTab = 'camera' | 'file';

export default function InferenceView() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);

  const [sourceTab, setSourceTab] = useState<SourceTab>('camera');
  const [settingsOpen, setSettingsOpen] = useState(false);

  const sourceKind = useStreamStore((s) => s.source.kind);
  const hasSource = sourceKind !== 'none';
  const isLive = sourceKind === 'camera';

  // El panel de render gobierna COMO se dibuja el resultado sobre el frame, cosa que
  // solo existe para los tipos que el backend compone. Con un clasificador cargado la
  // columna lo esconde en vez de dejar perillas que no mueven nada.
  const activeType = useWorkspaceStore((s) => s.activeModel?.type ?? null);
  const mostrarRender = panelDeRenderAplica(activeType);

  // Orquesta media + stream + render segun la fuente activa.
  useVisionSession({ videoRef, canvasRef, overlayRef });

  return (
    <div className="grid h-full grid-cols-[200px_1fr_230px] gap-3 bg-canvas p-3">
      {/* ── Zona izquierda: fuente + modelos ── */}
      <aside className="flex flex-col gap-5 overflow-y-auto rounded-[var(--radius-lg)] border border-border bg-surface p-4">
        <div className="flex flex-col gap-2.5">
          <SectionLabel>Fuente</SectionLabel>
          <Tabs
            aria-label="Fuente de video"
            tabs={[
              { key: 'camera', label: 'Camara' },
              { key: 'file', label: 'Archivo' },
            ]}
            value={sourceTab}
            onChange={setSourceTab}
          />
          {sourceTab === 'camera' ? <CameraSource /> : <FileSource />}
        </div>

        <div className="flex min-h-0 flex-col gap-2.5">
          <SectionLabel>Modelo</SectionLabel>
          <ModelSelector />
        </div>
      </aside>

      {/* ── Zona central: feed (heroe) + transporte ── */}
      <div className="flex min-w-0 flex-col gap-3">
        <VisionWorkspace
          videoRef={videoRef}
          canvasRef={canvasRef}
          overlayRef={overlayRef}
          hasSource={hasSource}
        >
          <MetricsHUD open />
          {isLive && (
            <div className="absolute right-3 top-3">
              <Badge variant="live">En vivo</Badge>
            </div>
          )}
        </VisionWorkspace>

        {/* Barra de transporte */}
        <div className="flex items-center gap-4 rounded-[var(--radius-lg)] border border-border bg-surface px-4 py-3">
          <Recorder canvasRef={canvasRef} />
        </div>
      </div>

      {/* ── Zona derecha: parametros + metricas + errores ── */}
      <aside className="flex flex-col gap-5 overflow-y-auto rounded-[var(--radius-lg)] border border-border bg-surface p-4">
        <div className="flex flex-col gap-3">
          <SectionLabel>Parametros</SectionLabel>
          <ConfidenceSlider />
        </div>

        {/* La seccion ENTERA se va, no solo los controles: un <SectionLabel>Render</
            SectionLabel> huerfano sobre un hueco se lee como un panel roto. */}
        {mostrarRender && (
          <div className="flex flex-col gap-2.5">
            <SectionLabel>Render</SectionLabel>
            <RenderSettings />
          </div>
        )}

        <div className="flex flex-col gap-2.5">
          <SectionLabel>Metricas</SectionLabel>
          <MetricsPanel />
        </div>

        <div className="flex min-h-0 flex-col gap-2.5">
          <SectionLabel>Errores</SectionLabel>
          <LogPanel open />
        </div>

        {/* Misma regla que la seccion Render: el modal solo tiene colores de caja y de
            etiqueta, o sea puro dibujo. Con un clasificador no hay nada que colorear. */}
        {mostrarRender && (
          <Button variant="outline" className="mt-auto" onClick={() => setSettingsOpen(true)}>
            colores de label
          </Button>
        )}
      </aside>

      {/* El && con mostrarRender no es redundante con esconder el boton: cierra el modal
          si el tipo de modelo cambiara con el abierto, en vez de dejarlo huerfano. */}
      <DrawSettingsModal open={settingsOpen && mostrarRender} onClose={() => setSettingsOpen(false)} />
    </div>
  );
}
