// DrawSettingsModal.tsx — colores de dibujo (client-side, sin backend). Draft local
// que se commitea al workspaceStore solo al Guardar (como el modal viejo).

import { useEffect, useState } from 'react';
import { Modal } from '@/shared/ui/Modal';
import { Button } from '@/shared/ui/Button';
import { useWorkspaceStore } from '@/features/vision-workspace/store/workspaceStore';

interface Props {
  open: boolean;
  onClose: () => void;
}

export function DrawSettingsModal({ open, onClose }: Props) {
  const drawSettings = useWorkspaceStore((s) => s.drawSettings);
  const setDrawSettings = useWorkspaceStore((s) => s.setDrawSettings);

  const [bbox, setBbox] = useState(drawSettings.bboxColor);
  const [label, setLabel] = useState(drawSettings.labelColor);

  // Resincronizar el draft con el estado real cada vez que se abre.
  useEffect(() => {
    if (open) {
      setBbox(drawSettings.bboxColor);
      setLabel(drawSettings.labelColor);
    }
  }, [open, drawSettings.bboxColor, drawSettings.labelColor]);

  function save() {
    setDrawSettings({ bboxColor: bbox, labelColor: label });
    onClose();
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="Configuracion avanzada"
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>
            Cancelar
          </Button>
          <Button variant="primary" onClick={save}>
            Guardar
          </Button>
        </>
      }
    >
      <div className="space-y-4">
        <ColorRow label="Color de bounding box" value={bbox} onChange={setBbox} />
        <ColorRow label="Color de etiquetas" value={label} onChange={setLabel} />
      </div>
    </Modal>
  );
}

function ColorRow({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="flex items-center justify-between gap-4">
      <span className="text-sm text-fg">{label}</span>
      <div className="flex items-center gap-2">
        <span className="font-mono text-xs text-fg-muted">{value}</span>
        <input
          type="color"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="size-8 cursor-pointer rounded-[var(--radius-sm)] border border-border bg-transparent"
        />
      </div>
    </div>
  );
}
