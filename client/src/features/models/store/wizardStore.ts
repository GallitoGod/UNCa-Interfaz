// wizardStore.ts — estado del ConfigWizard (4 pasos). Reemplaza el _state de modulo
// del configBuilder viejo. La config se edita por path (setField), y al cambiar de
// tipo se reemplaza output (manteniendo input/runtime).

import { create } from 'zustand';
import type { ModelConfig, ModelType } from '@/shared/api/types';
import { setDeep } from '../lib/setDeep';
import type { FieldIssue } from '../lib/validationErrors';

export type WizardStep = 1 | 2 | 3 | 4;

interface WizardState {
  modelFile: string;
  baseName: string;
  step: WizardStep;
  config: ModelConfig | null;
  // Campos rechazados por el backend en el ultimo intento de guardado (422).
  fieldErrors: FieldIssue[];

  init: (modelFile: string, config: ModelConfig) => void;
  reset: () => void;
  setStep: (step: WizardStep) => void;
  setField: (path: string, value: unknown) => void;
  setFieldErrors: (issues: FieldIssue[]) => void;
  // Cambia el tipo: reemplaza model_type + output (default del nuevo tipo), conserva input/runtime.
  setModelType: (type: ModelType, defaultOutput: unknown) => void;
}

export const useWizardStore = create<WizardState>((set) => ({
  modelFile: '',
  baseName: '',
  step: 1,
  config: null,
  fieldErrors: [],

  init: (modelFile, config) =>
    set({
      modelFile,
      baseName: modelFile.replace(/\.[^.]+$/, ''),
      step: 1,
      config,
      fieldErrors: [],
    }),

  reset: () =>
    set({ modelFile: '', baseName: '', step: 1, config: null, fieldErrors: [] }),

  setStep: (step) => set({ step }),

  setField: (path, value) =>
    set((s) =>
      s.config
        ? {
            config: setDeep(s.config, path, value),
            // Al editar un campo, su error previo deja de aplicar.
            fieldErrors: s.fieldErrors.filter((e) => e.path !== path),
          }
        : s,
    ),

  setFieldErrors: (issues) => set({ fieldErrors: issues }),

  setModelType: (type, defaultOutput) =>
    set((s) => {
      if (!s.config) return s;
      // Cast: la union se rearma con el nuevo output; el wizard valida shape al guardar.
      const next = {
        ...s.config,
        model_type: type,
        output: defaultOutput,
      } as unknown as ModelConfig;
      return { config: next };
    }),
}));
