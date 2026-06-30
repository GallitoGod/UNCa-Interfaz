// Step4Runtime.tsx — runtime simplificado: la vista tipica es Backend + Dispositivo.
// providers (ONNX) / delegates (TFLite) se derivan de device; warmup, threads y el
// override manual de providers viven en "Avanzado".

import type { ModelConfig } from '@/shared/api/types';
import { AdvancedSection, CheckField, FieldGroup, NumberField, SelectField } from './fields';
import { delegatesForDevice, providersForDevice } from '../../lib/wizardPresets';

interface Props {
  config: ModelConfig;
  setField: (path: string, value: unknown) => void;
}

export function Step4Runtime({ config, setField }: Props) {
  const rt = config.runtime;
  const providers = rt.onnx?.providers ?? [];
  const delegates = rt.tflite?.delegates ?? [];

  // Al cambiar el dispositivo, derivar providers/delegates del backend activo. El
  // override manual sigue disponible en "Avanzado" para casos raros.
  function onDevice(v: 'cpu' | 'gpu') {
    setField('runtime.device', v);
    if (rt.backend === 'onnxruntime') setField('runtime.onnx.providers', providersForDevice(v));
    if (rt.backend === 'tflite') setField('runtime.tflite.delegates', delegatesForDevice(v));
  }

  function toggleProvider(name: string, on: boolean) {
    const set = new Set(providers);
    if (on) set.add(name);
    else set.delete(name);
    // CUDA primero (orden de prioridad).
    const arr = [...set].sort((a) => (a.includes('CUDA') ? -1 : 1));
    setField('runtime.onnx.providers', arr);
  }

  function toggleDelegate(name: string, on: boolean) {
    const set = new Set(delegates);
    if (on) set.add(name);
    else set.delete(name);
    setField('runtime.tflite.delegates', [...set]);
  }

  return (
    <div className="space-y-6">
      <FieldGroup title="Backend e inferencia">
        <div className="grid grid-cols-2 gap-3">
          <SelectField label="Backend" value={rt.backend} options={['onnxruntime', 'tflite', 'tensorflow', 'pytorch'] as const} onChange={(v) => setField('runtime.backend', v)} />
          <SelectField label="Dispositivo" value={rt.device} options={['cpu', 'gpu'] as const} onChange={onDevice} />
        </div>
      </FieldGroup>

      <AdvancedSection>
        <NumberField label="Warmup runs" value={rt.warmup.runs} min={0} onChange={(v) => setField('runtime.warmup.runs', v)} />

        {rt.backend === 'onnxruntime' && (
          <>
            <div className="space-y-2">
              <CheckField label="CUDAExecutionProvider" checked={providers.includes('CUDAExecutionProvider')} onChange={(v) => toggleProvider('CUDAExecutionProvider', v)} />
              <CheckField label="CPUExecutionProvider" checked={providers.includes('CPUExecutionProvider')} onChange={(v) => toggleProvider('CPUExecutionProvider', v)} />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <NumberField label="Threads intra-op (vacio = auto)" value={rt.threads.intra_op} min={1} onChange={(v) => setField('runtime.threads.intra_op', v)} />
              <NumberField label="Threads inter-op (vacio = auto)" value={rt.threads.inter_op} min={1} onChange={(v) => setField('runtime.threads.inter_op', v)} />
            </div>
          </>
        )}

        {rt.backend === 'tflite' && (
          <>
            <CheckField label="GPU delegate" checked={delegates.includes('gpu')} onChange={(v) => toggleDelegate('gpu', v)} />
            <NumberField label="Threads (vacio = auto)" value={rt.threads.num_threads} min={1} onChange={(v) => setField('runtime.threads.num_threads', v)} />
          </>
        )}
      </AdvancedSection>
    </div>
  );
}
