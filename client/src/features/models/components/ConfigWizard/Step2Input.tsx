// Step2Input.tsx — dimensiones, color, normalizacion (preset) y (deteccion) ajuste
// geometrico. Simplificado: channels se deriva de color_order; la normalizacion se
// elige por preset (las cajas mean/std solo en 'Personalizado'); layout/dtype viven
// en "Avanzado"; letterbox + preserve_aspect_ratio se unifican en un selector.

import { useState } from 'react';
import type { ModelConfig } from '@/shared/api/types';
import {
  AdvancedSection,
  ColorField,
  FieldGroup,
  NumberField,
  SelectField,
} from './fields';
import {
  NORM_PRESET_LABELS,
  applyNormPreset,
  inferNormPreset,
  type NormPreset,
} from '../../lib/wizardPresets';

interface Props {
  config: ModelConfig;
  setField: (path: string, value: unknown) => void;
}

// Selector geometrico: dos modos en vez de dos checkboxes que interactuan.
type GeoMode = 'resize' | 'letterbox';
const GEO_LABELS: Record<GeoMode, string> = {
  resize: 'Resize directo (deforma)',
  letterbox: 'Letterbox (preserva + padding)',
};

export function Step2Input({ config, setField }: Props) {
  const inp = config.input;
  const str = inp.input_str;
  const isDet = config.model_type === 'detection';

  // El preset activo se infiere del config al montar; se guarda en estado local para
  // respetar la eleccion 'Personalizado' aunque sus valores coincidan con otro preset.
  const [normPreset, setNormPreset] = useState<NormPreset>(() => inferNormPreset(inp));

  // color_order es la unica fuente de verdad de los canales: GRAY=1, RGB/BGR=3.
  function onColorOrder(v: InputColorOrder) {
    setField('input.color_order', v);
    setField('input.channels', v === 'GRAY' ? 1 : 3);
  }

  function onNormPreset(p: NormPreset) {
    setNormPreset(p);
    const f = applyNormPreset(p, inp);
    setField('input.scale', f.scale);
    setField('input.normalize', f.normalize);
    setField('input.mean', f.mean);
    setField('input.std', f.std);
  }

  // Modo geometrico derivado: letterbox real = letterbox && preserve_aspect_ratio.
  const geoMode: GeoMode = inp.letterbox && (inp.preserve_aspect_ratio ?? true) ? 'letterbox' : 'resize';
  function onGeoMode(m: GeoMode) {
    if (m === 'letterbox') {
      setField('input.letterbox', true);
      setField('input.preserve_aspect_ratio', true);
    } else {
      setField('input.letterbox', false);
    }
  }

  return (
    <div className="space-y-6">
      <FieldGroup title="Dimensiones">
        <div className="grid grid-cols-3 gap-3">
          <NumberField label="Ancho" value={inp.width} min={1} onChange={(v) => setField('input.width', v)} />
          <NumberField label="Alto" value={inp.height} min={1} onChange={(v) => setField('input.height', v)} />
          <SelectField label="Orden de color" value={inp.color_order} options={['RGB', 'BGR', 'GRAY'] as const} onChange={onColorOrder} />
        </div>
      </FieldGroup>

      <FieldGroup title="Normalizacion">
        <SelectField
          label="Preset"
          value={normPreset}
          options={['none', 'scale01', 'imagenet', 'custom'] as const}
          labels={NORM_PRESET_LABELS}
          onChange={onNormPreset}
        />
        {normPreset === 'custom' && (
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
        <FieldGroup title="Ajuste geometrico">
          <SelectField
            label="Modo"
            value={geoMode}
            options={['resize', 'letterbox'] as const}
            labels={GEO_LABELS}
            onChange={onGeoMode}
          />
          {geoMode === 'letterbox' && (
            <ColorField label="Color de relleno (pad)" value={inp.auto_pad_color} onChange={(v) => setField('input.auto_pad_color', v)} />
          )}
        </FieldGroup>
      )}

      <AdvancedSection>
        <div className="grid grid-cols-2 gap-3">
          <SelectField label="Layout" value={str?.layout ?? 'HWC'} options={['HWC', 'CHW', 'NHWC', 'NCHW'] as const} onChange={(v) => setField('input.input_str.layout', v)} />
          <SelectField label="Dtype" value={str?.dtype ?? 'float32'} options={['float32', 'uint8', 'int8'] as const} onChange={(v) => setField('input.input_str.dtype', v)} />
        </div>
      </AdvancedSection>
    </div>
  );
}

type InputColorOrder = ModelConfig['input']['color_order'];
