// InferenceView.tsx — vista de inferencia: video heroe (VisionWorkspace) + panel
// de controles. Es dueno de los refs (video/canvas/overlay) y del orquestador.

import { useRef, useState } from 'react';
import { VisionWorkspace } from '@/features/vision-workspace/components/VisionWorkspace';
import { Tabs } from '@/shared/ui/Tabs';
import { Button } from '@/shared/ui/Button';
import { useStreamStore } from './store/streamStore';
import { useVisionSession } from './hooks/useVisionSession';
import { CameraSource } from './components/CameraSource';
import { FileSource } from './components/FileSource';
import { ConfidenceSlider } from './components/ConfidenceSlider';
import { MetricsHUD } from './components/MetricsHUD';
import { LogPanel } from './components/LogPanel';
import { Recorder } from './components/Recorder';
import { DrawSettingsModal } from './components/DrawSettingsModal';

type SourceTab = 'camera' | 'file';

export default function InferenceView() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);

  const [sourceTab, setSourceTab] = useState<SourceTab>('camera');
  const [metricsOpen, setMetricsOpen] = useState(false);
  const [logsOpen, setLogsOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);

  const hasSource = useStreamStore((s) => s.source.kind !== 'none');

  // Orquesta media + stream + render segun la fuente activa.
  useVisionSession({ videoRef, canvasRef, overlayRef });

  return (
    <div className="grid h-full grid-cols-[1fr_20rem] gap-4 p-4">
      {/* Video heroe */}
      <VisionWorkspace
        videoRef={videoRef}
        canvasRef={canvasRef}
        overlayRef={overlayRef}
        hasSource={hasSource}
      >
        <MetricsHUD open={metricsOpen} />
      </VisionWorkspace>

      {/* Panel de controles */}
      <aside className="flex flex-col gap-4 overflow-y-auto rounded-[var(--radius-lg)] border border-border bg-surface p-4">
        <Tabs
          aria-label="Fuente de video"
          tabs={[
            { key: 'camera', label: 'Camara' },
            { key: 'file', label: 'Cargar' },
          ]}
          value={sourceTab}
          onChange={setSourceTab}
        />

        {sourceTab === 'camera' ? <CameraSource /> : <FileSource />}

        <div className="h-px bg-border" />

        <ConfidenceSlider />

        <div className="flex flex-col gap-2">
          <Button variant="outline" onClick={() => setMetricsOpen((v) => !v)}>
            {metricsOpen ? 'Ocultar metricas' : 'Metricas'}
          </Button>
          <Button variant="outline" onClick={() => setSettingsOpen(true)}>
            Configuracion avanzada
          </Button>
          <Button variant="outline" onClick={() => setLogsOpen((v) => !v)}>
            {logsOpen ? 'Ocultar errores' : 'Registro de errores'}
          </Button>
          <Recorder canvasRef={canvasRef} />
        </div>

        <LogPanel open={logsOpen} />
      </aside>

      <DrawSettingsModal open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </div>
  );
}
