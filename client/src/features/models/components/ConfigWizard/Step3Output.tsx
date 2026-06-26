// Step3Output.tsx — configuracion de salida, distinta por tipo de modelo.
// Detection es la completa (formato, anchors, estructura de tensor, NMS).
// Classification/Segmentation son mas cortas.

import type {
  AnchorConfig,
  ClassificationOutput,
  DetectionOutput,
  ModelConfig,
  RuntimeShapes,
  SegmentationOutput,
} from '@/shared/api/types';
import { CheckField, FieldGroup, NumberField, SelectField } from './fields';

interface Props {
  config: ModelConfig;
  setField: (path: string, value: unknown) => void;
  anchorDefaults: AnchorConfig | null;
}

const COORD_KEYS: Record<string, string[]> = {
  xyxy: ['x1', 'y1', 'x2', 'y2'],
  cxcywh: ['cx', 'cy', 'w', 'h'],
  yxyx: ['y1', 'x1', 'y2', 'x2'],
};

export function Step3Output({ config, setField, anchorDefaults }: Props) {
  if (config.model_type === 'detection') {
    return (
      <DetectionStep
        out={config.output}
        rs={config.runtime.runtimeShapes}
        setField={setField}
        anchorDefaults={anchorDefaults}
      />
    );
  }
  if (config.model_type === 'classification') {
    return <ClassificationStep out={config.output} setField={setField} />;
  }
  return <SegmentationStep out={config.output} setField={setField} />;
}

function DetectionStep({
  out,
  rs,
  setField,
  anchorDefaults,
}: {
  out: DetectionOutput;
  rs: RuntimeShapes | null;
  setField: Props['setField'];
  anchorDefaults: AnchorConfig | null;
}) {
  const ts = out.tensor_structure;
  const keys = COORD_KEYS[ts.box_format] ?? COORD_KEYS.xyxy;
  const ac = out.anchor_config;

  function onPackFormat(v: DetectionOutput['pack_format']) {
    setField('output.pack_format', v);
    if (v === 'anchor_deltas' && !out.anchor_config && anchorDefaults) {
      setField('output.anchor_config', anchorDefaults);
    }
  }

  function onBoxFormat(v: string) {
    setField('output.tensor_structure.box_format', v);
    // Reset de coordenadas a las llaves del nuevo formato.
    const fresh: Record<string, number> = {};
    (COORD_KEYS[v] ?? COORD_KEYS.xyxy).forEach((k) => (fresh[k] = 0));
    setField('output.tensor_structure.coordinates', fresh);
  }

  return (
    <div className="space-y-6">
      <FieldGroup title="Formato de salida">
        <div className="grid grid-cols-2 gap-3">
          <SelectField
            label="Empaquetado (pack_format)"
            value={out.pack_format}
            options={['raw', 'yolo_flat', 'boxes_scores', 'tflite_detpost', 'anchor_deltas'] as const}
            onChange={onPackFormat}
          />
          <SelectField
            label="Espacio de coordenadas"
            value={rs?.out_coords_space ?? 'tensor_pixels'}
            options={['tensor_pixels', 'normalized_0_1'] as const}
            onChange={(v) => setField('runtime.runtimeShapes.out_coords_space', v)}
          />
        </div>
      </FieldGroup>

      {out.pack_format === 'anchor_deltas' && ac && (
        <FieldGroup title="Anchors (salida cruda)">
          <div className="grid grid-cols-3 gap-3">
            <NumberField label="min_level" value={ac.min_level} onChange={(v) => setField('output.anchor_config.min_level', v)} />
            <NumberField label="max_level" value={ac.max_level} onChange={(v) => setField('output.anchor_config.max_level', v)} />
            <NumberField label="num_scales" value={ac.num_scales} onChange={(v) => setField('output.anchor_config.num_scales', v)} />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <NumberField label="anchor_scale" value={ac.anchor_scale} step={0.1} onChange={(v) => setField('output.anchor_config.anchor_scale', v)} />
            <SelectField label="scores_activation" value={ac.scores_activation} options={['none', 'sigmoid', 'softmax'] as const} onChange={(v) => setField('output.anchor_config.scores_activation', v)} />
          </div>
        </FieldGroup>
      )}

      <FieldGroup title="Estructura por deteccion">
        <div className="grid grid-cols-2 gap-3">
          <SelectField label="Formato de boxes" value={ts.box_format} options={['xyxy', 'cxcywh', 'yxyx'] as const} onChange={onBoxFormat} />
          <NumberField label="Numero de clases" value={ts.num_classes ?? 80} onChange={(v) => setField('output.tensor_structure.num_classes', v)} />
        </div>
        <div className="grid grid-cols-2 gap-3">
          <NumberField label="Indice de confianza" value={ts.confidence_index} onChange={(v) => setField('output.tensor_structure.confidence_index', v)} />
          <NumberField label="Indice de clase" value={ts.class_index} onChange={(v) => setField('output.tensor_structure.class_index', v)} />
        </div>
        <div className="grid grid-cols-4 gap-3">
          {keys.map((k) => (
            <NumberField
              key={k}
              label={k}
              value={ts.coordinates[k] ?? 0}
              onChange={(v) => setField(`output.tensor_structure.coordinates.${k}`, v)}
            />
          ))}
        </div>
      </FieldGroup>

      <FieldGroup title="Filtrado y NMS">
        <div className="grid grid-cols-2 gap-3">
          <NumberField label="Umbral de confianza" value={out.confidence_threshold} step={0.01} onChange={(v) => setField('output.confidence_threshold', v)} />
          <NumberField label="Top-K (0 = sin limite)" value={out.top_k} onChange={(v) => setField('output.top_k', v)} />
        </div>
        <div className="space-y-2">
          <CheckField label="Aplicar filtro de confianza" checked={out.apply_conf_filter} onChange={(v) => setField('output.apply_conf_filter', v)} />
          <CheckField label="Aplicar NMS" checked={out.apply_nms} onChange={(v) => setField('output.apply_nms', v)} />
          <CheckField label="NMS por clase" checked={out.nms_per_class} onChange={(v) => setField('output.nms_per_class', v)} />
        </div>
        {out.apply_nms && (
          <NumberField label="Umbral IoU para NMS" value={out.nms_threshold} step={0.01} onChange={(v) => setField('output.nms_threshold', v)} />
        )}
      </FieldGroup>
    </div>
  );
}

function ClassificationStep({
  out,
  setField,
}: {
  out: ClassificationOutput;
  setField: Props['setField'];
}) {
  return (
    <div className="space-y-6">
      <FieldGroup title="Formato de salida">
        <div className="grid grid-cols-2 gap-3">
          <SelectField label="Empaquetado" value={out.pack_format} options={['softmax_out', 'sigmoid_out', 'logits_raw'] as const} onChange={(v) => setField('output.pack_format', v)} />
          <SelectField label="El modelo emite" value={out.tensor_structure.output_format} options={['logits', 'probabilities'] as const} onChange={(v) => setField('output.tensor_structure.output_format', v)} />
        </div>
        <CheckField label="Multi-etiqueta (sigmoid por clase)" checked={out.tensor_structure.multi_label} onChange={(v) => setField('output.tensor_structure.multi_label', v)} />
      </FieldGroup>
      <FieldGroup title="Postprocesamiento">
        <div className="space-y-2">
          <CheckField label="Aplicar softmax" checked={out.apply_softmax} onChange={(v) => setField('output.apply_softmax', v)} />
          <CheckField label="Aplicar sigmoid (multi-label)" checked={out.apply_sigmoid} onChange={(v) => setField('output.apply_sigmoid', v)} />
        </div>
        <div className="grid grid-cols-3 gap-3">
          <NumberField label="Numero de clases" value={out.tensor_structure.num_classes} onChange={(v) => setField('output.tensor_structure.num_classes', v)} />
          <NumberField label="Top-K" value={out.top_k} onChange={(v) => setField('output.top_k', v)} />
          <NumberField label="Umbral de confianza" value={out.confidence_threshold} step={0.01} onChange={(v) => setField('output.confidence_threshold', v)} />
        </div>
      </FieldGroup>
    </div>
  );
}

function SegmentationStep({
  out,
  setField,
}: {
  out: SegmentationOutput;
  setField: Props['setField'];
}) {
  return (
    <div className="space-y-6">
      <FieldGroup title="Formato de salida">
        <div className="grid grid-cols-2 gap-3">
          <SelectField label="Empaquetado" value={out.pack_format} options={['argmax_map', 'softmax_map', 'binary_mask'] as const} onChange={(v) => setField('output.pack_format', v)} />
          <SelectField label="El modelo emite" value={out.tensor_structure.output_format} options={['argmax_map', 'softmax_map'] as const} onChange={(v) => setField('output.tensor_structure.output_format', v)} />
        </div>
      </FieldGroup>
      <FieldGroup title="Estructura del tensor">
        <div className="grid grid-cols-3 gap-3">
          <NumberField label="Numero de clases" value={out.tensor_structure.num_classes} onChange={(v) => setField('output.tensor_structure.num_classes', v)} />
          <NumberField label="Output stride" value={out.tensor_structure.output_stride} onChange={(v) => setField('output.tensor_structure.output_stride', v)} />
          <NumberField label="Umbral de confianza" value={out.confidence_threshold} step={0.01} onChange={(v) => setField('output.confidence_threshold', v)} />
        </div>
        <CheckField label="Redimensionar mascara al tamano original" checked={out.tensor_structure.resize_to_input} onChange={(v) => setField('output.tensor_structure.resize_to_input', v)} />
      </FieldGroup>
    </div>
  );
}
