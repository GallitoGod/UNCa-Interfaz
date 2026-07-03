// ConfigWizard.tsx — wizard de 4 pasos. Lee el estado del wizardStore; el guardado
// va por POST /configs (useSaveConfig, que aplica toBackendConfig). El cambio de tipo
// pide el template del nuevo tipo al backend.

import { useState } from 'react';
import type { AnchorConfig, ModelType } from '@/shared/api/types';
import { cn } from '@/shared/ui/cn';
import { Button } from '@/shared/ui/Button';
import { useWizardStore, type WizardStep } from '../../store/wizardStore';
import { useSaveConfig } from '../../hooks/useModelsList';
import { getConfigTemplate } from '../../api/configs';
import { fieldIssuesFrom, labelForPath, stepOfPath } from '../../lib/validationErrors';
import { Step1Type } from './Step1Type';
import { Step2Input } from './Step2Input';
import { Step3Output } from './Step3Output';
import { Step4Runtime } from './Step4Runtime';

const STEP_LABELS = ['Tipo', 'Input', 'Output', 'Runtime'];
// Progreso de la micro-barra por paso (spec stepper 1b: 12 -> 37 -> 62 -> 100).
const STEP_PROGRESS = [12, 37, 62, 100];

export function ConfigWizard({
  onClose,
  initialAnchorDefaults,
}: {
  onClose: () => void;
  initialAnchorDefaults: AnchorConfig | null;
}) {
  const config = useWizardStore((s) => s.config);
  const step = useWizardStore((s) => s.step);
  const baseName = useWizardStore((s) => s.baseName);
  const setStep = useWizardStore((s) => s.setStep);
  const setField = useWizardStore((s) => s.setField);
  const setModelType = useWizardStore((s) => s.setModelType);
  const fieldErrors = useWizardStore((s) => s.fieldErrors);
  const setFieldErrors = useWizardStore((s) => s.setFieldErrors);

  const [anchorDefaults, setAnchorDefaults] = useState(initialAnchorDefaults);
  const [msg, setMsg] = useState<{ text: string; ok: boolean } | null>(null);
  const save = useSaveConfig();

  if (!config) return null;

  async function handleSelectType(type: ModelType) {
    const cur = useWizardStore.getState().config;
    if (!cur || type === cur.model_type) return;
    try {
      const tpl = await getConfigTemplate(type);
      setModelType(type, tpl.config.output);
      setAnchorDefaults(tpl.anchor_defaults);
    } catch (e) {
      console.error('No se pudo traer el template del tipo:', e);
    }
  }

  async function handleSave() {
    const { config: cur, baseName: name } = useWizardStore.getState();
    if (!cur) return;
    setFieldErrors([]); // limpiar el resultado del intento anterior
    try {
      await save.mutateAsync({ name, config: cur });
      setMsg({ text: 'Config guardada correctamente', ok: true });
      setTimeout(onClose, 800);
    } catch (e) {
      // Capacidad critica (SDD 1.2): decodificar el 422, abortar el guardado y
      // redirigir al campo que causo el conflicto.
      const issues = fieldIssuesFrom(e);
      if (issues.length > 0) {
        setFieldErrors(issues);
        const firstStep = issues
          .map((i) => stepOfPath(i.path))
          .find((s): s is WizardStep => s !== null);
        if (firstStep) setStep(firstStep);
        setMsg({
          text: `El backend rechazo la config: revisá los campos marcados (${issues.length}).`,
          ok: false,
        });
      } else {
        setMsg({ text: e instanceof Error ? e.message : 'Error al guardar', ok: false });
      }
    }
  }

  function next() {
    if (step < 4) setStep((step + 1) as WizardStep);
    else void handleSave();
  }
  function prev() {
    if (step > 1) setStep((step - 1) as WizardStep);
  }

  return (
    <div className="space-y-5">
      {/* Stepper 1b (spec §04): header contextual + barra de instrumento segmentada
          + micro-barra de progreso. El actual va solido en cian (texto ink), los
          completados en cian-soft con ✓ y son clicables (retroceso); los pendientes
          quedan apagados y no navegables (se avanza solo con "Siguiente"). */}
      <div className="space-y-3 border-b border-border pb-4">
        <div className="flex items-baseline gap-2">
          <span className="text-xs text-fg-muted">Configurando</span>
          <span className="font-mono text-sm font-semibold text-fg">{baseName}</span>
        </div>
        <div className="flex gap-[5px]">
          {STEP_LABELS.map((label, i) => {
            const n = (i + 1) as WizardStep;
            const done = step > n;
            const cur = step === n;
            return (
              <button
                key={label}
                type="button"
                disabled={!done}
                onClick={() => setStep(n)}
                className={cn(
                  'flex h-[50px] flex-1 flex-col justify-center gap-0.5 rounded-[9px] border px-[15px] text-left transition-colors',
                  cur
                    ? 'border-accent bg-accent'
                    : done
                      ? 'border-accent-border bg-[rgba(52,214,255,0.07)] hover:bg-accent-soft'
                      : 'border-border bg-control disabled:pointer-events-none',
                )}
              >
                <span
                  className={cn(
                    'font-mono text-[11px] font-bold tracking-[0.5px]',
                    cur ? 'text-accent-fg' : done ? 'text-accent' : 'text-fg-subtle',
                  )}
                >
                  {String(n).padStart(2, '0')}
                  {done && ' ✓'}
                </span>
                <span className={cn('text-[13px] font-semibold', cur ? 'text-accent-fg' : 'text-fg-muted')}>
                  {label}
                </span>
              </button>
            );
          })}
        </div>
        <div className="h-[3px] overflow-hidden rounded-[2px] bg-border">
          <div
            className="h-full bg-accent transition-[width] duration-300"
            style={{ width: `${STEP_PROGRESS[step - 1]}%` }}
          />
        </div>
      </div>

      {/* Banner de campos rechazados por el backend (clickable -> salta al paso). */}
      {fieldErrors.length > 0 && (
        <div className="rounded-[var(--radius-md)] border border-danger/40 bg-danger/10 px-3 py-2 text-xs">
          <p className="mb-1 font-medium text-danger">Campos rechazados por el backend:</p>
          <ul className="space-y-0.5">
            {fieldErrors.map((e) => (
              <li key={e.path}>
                <button
                  type="button"
                  onClick={() => {
                    const s = stepOfPath(e.path);
                    if (s) setStep(s);
                  }}
                  className="font-mono text-danger underline-offset-2 hover:underline"
                >
                  {labelForPath(e.path)}
                </button>
                <span className="text-fg-muted"> — {e.msg}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Contenido del paso */}
      <div className="min-h-[16rem]">
        {step === 1 && <Step1Type config={config} onSelectType={handleSelectType} />}
        {step === 2 && <Step2Input config={config} setField={setField} />}
        {step === 3 && <Step3Output config={config} setField={setField} anchorDefaults={anchorDefaults} />}
        {step === 4 && <Step4Runtime config={config} setField={setField} />}
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between border-t border-border pt-4">
        <Button variant="outline" onClick={prev} disabled={step === 1}>
          Anterior
        </Button>
        <span className="font-mono text-[13px] font-semibold text-fg-muted">{step} / 4</span>
        <Button variant="primary" onClick={next} disabled={save.isPending}>
          {step === 4 ? (save.isPending ? 'Guardando...' : 'Guardar') : 'Siguiente'}
        </Button>
      </div>

      {msg && (
        <p className={cn('text-sm', msg.ok ? 'text-success' : 'text-danger')}>{msg.text}</p>
      )}
    </div>
  );
}
