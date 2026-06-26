// validationErrors.ts — decodifica los errores de validacion (422) del backend para el
// wizard. El backend valida el config contra ModelConfig estricto y devuelve la lista
// Pydantic { loc, msg, type } en `detail`; aca la convertimos en issues por-campo y
// resolvemos en que paso del wizard cae cada uno. Es la "capacidad critica" del SDD 1.2:
// decodificar el fallo y poder redirigir al campo especifico que lo causo.

import { ApiError } from '@/shared/api/errors';

export interface FieldIssue {
  path: string; // dotted, alineado con los paths de setField (ej: "input.width")
  msg: string;
}

interface PydItem {
  loc?: (string | number)[];
  msg?: string;
}

// La raiz del loc determina el paso del wizard.
const STEP_OF_ROOT: Record<string, 1 | 2 | 3 | 4> = {
  model_type: 1,
  input: 2,
  output: 3,
  runtime: 4,
};

const STEP_LABEL: Record<string, string> = {
  model_type: 'Tipo',
  input: 'Input',
  output: 'Output',
  runtime: 'Runtime',
};

// Extrae los issues por-campo de un error del backend. Devuelve [] si no es un 422 con
// lista Pydantic; en ese caso el caller cae al mensaje generico (no rompe el flujo).
export function fieldIssuesFrom(err: unknown): FieldIssue[] {
  if (!(err instanceof ApiError) || err.status !== 422) return [];
  if (!Array.isArray(err.detail)) return [];
  const issues: FieldIssue[] = [];
  for (const it of err.detail as PydItem[]) {
    // 'body' aparece cuando FastAPI valida un body param; aca validamos el dict
    // directo, pero lo filtramos por las dudas.
    const loc = (it.loc ?? []).filter((s) => s !== 'body');
    if (loc.length === 0) continue;
    issues.push({ path: loc.join('.'), msg: it.msg ?? 'valor invalido' });
  }
  return issues;
}

// Paso del wizard donde vive un path (segun su raiz). null si no mapea.
export function stepOfPath(path: string): 1 | 2 | 3 | 4 | null {
  return STEP_OF_ROOT[path.split('.')[0]] ?? null;
}

// Etiqueta legible de un path: "Input → width", "Output → tensor_structure.confidence_index".
export function labelForPath(path: string): string {
  const parts = path.split('.');
  const head = STEP_LABEL[parts[0]] ?? parts[0];
  const rest = parts.slice(1).join('.');
  return rest ? `${head} → ${rest}` : head;
}
