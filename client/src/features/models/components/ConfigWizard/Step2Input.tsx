// Step2Input.tsx — dimensiones, color/layout/dtype, preprocesamiento y (deteccion)
// letterbox. Lee config.input y escribe por path con setField.

import type { ModelConfig } from '@/shared/api/types';
import { CheckField, FieldGroup, NumberField, SelectField } from './fields';

interface Props {
  config: ModelConfig;
  setField: (path: string, value: unknown) => void;
}

export function Step2Input({ config, setField }: Props) {
  const inp = config.input;
  const str = inp.input_str;
  const isDet = config.model_type === 'detection';

  return (
    <div className="space-y-6">
      <FieldGroup title="Dimensiones">
        <div className="grid grid-cols-3 gap-3">
          <NumberField label="Ancho" value={inp.width} min={1} onChange={(v) => setField('input.width', v)} />
          <NumberField label="Alto" value={inp.height} min={1} onChange={(v) => setField('input.height', v)} />
          <NumberField label="Canales" value={inp.channels} min={1} onChange={(v) => setField('input.channels', v)} />
        </div>
        <div className="grid grid-cols-3 gap-3">
          <SelectField label="Orden de color" value={inp.color_order} options={['RGB', 'BGR', 'GRAY'] as const} onChange={(v) => setField('input.color_order', v)} />
          <SelectField label="Layout" value={str?.layout ?? 'HWC'} options={['HWC', 'CHW', 'NHWC', 'NCHW'] as const} onChange={(v) => setField('input.input_str.layout', v)} />
          <SelectField label="Dtype" value={str?.dtype ?? 'float32'} options={['float32', 'uint8', 'int8'] as const} onChange={(v) => setField('input.input_str.dtype', v)} />
        </div>
      </FieldGroup>

      <FieldGroup title="Preprocesamiento">
        <div className="space-y-2">
          <CheckField label="Escalar a [0,1] (/255)" checked={inp.scale} onChange={(v) => setField('input.scale', v)} />
          <CheckField label="Normalizar con media y std" checked={inp.normalize} onChange={(v) => setField('input.normalize', v)} />
          <CheckField label="Modelo cuantizado" checked={str?.quantized ?? false} onChange={(v) => setField('input.input_str.quantized', v)} />
        </div>
        {inp.normalize && (
          <div className="space-y-2">
            <div className="grid grid-cols-3 gap-3">
              <NumberField label="mean R" value={inp.mean[0] ?? 0} step={0.01} onChange={(v) => setField('input.mean.0', v)} />
              <NumberField label="mean G" value={inp.mean[1] ?? 0} step={0.01} onChange={(v) => setField('input.mean.1', v)} />
              <NumberField label="mean B" value={inp.mean[2] ?? 0} step={0.01} onChange={(v) => setField('input.mean.2', v)} />
            </div>
            <div className="grid grid-cols-3 gap-3">
              <NumberField label="std R" value={inp.std[0] ?? 1} step={0.01} onChange={(v) => setField('input.std.0', v)} />
              <NumberField label="std G" value={inp.std[1] ?? 1} step={0.01} onChange={(v) => setField('input.std.1', v)} />
              <NumberField label="std B" value={inp.std[2] ?? 1} step={0.01} onChange={(v) => setField('input.std.2', v)} />
            </div>
          </div>
        )}
      </FieldGroup>

      {isDet && (
        <FieldGroup title="Letterbox">
          <div className="space-y-2">
            <CheckField label="Aplicar letterbox" checked={inp.letterbox} onChange={(v) => setField('input.letterbox', v)} />
            <CheckField label="Preservar aspect ratio" checked={inp.preserve_aspect_ratio ?? true} onChange={(v) => setField('input.preserve_aspect_ratio', v)} />
          </div>
          {inp.letterbox && (
            <div className="grid grid-cols-3 gap-3">
              <NumberField label="pad R" value={inp.auto_pad_color?.[0] ?? 114} min={0} onChange={(v) => setField('input.auto_pad_color.0', v)} />
              <NumberField label="pad G" value={inp.auto_pad_color?.[1] ?? 114} min={0} onChange={(v) => setField('input.auto_pad_color.1', v)} />
              <NumberField label="pad B" value={inp.auto_pad_color?.[2] ?? 114} min={0} onChange={(v) => setField('input.auto_pad_color.2', v)} />
            </div>
          )}
        </FieldGroup>
      )}
    </div>
  );
}
