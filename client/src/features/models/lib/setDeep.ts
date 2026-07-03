// setDeep.ts — asignacion inmutable por path (ej. "output.tensor_structure.num_classes").
// Reemplaza el setDeep mutable del configBuilder viejo. Crea arrays cuando el
// siguiente segmento del path es numerico (ej. "input.mean.0").

type Mutable = Record<string, unknown>;

export function setDeep<T extends object>(obj: T, path: string, value: unknown): T {
  const parts = path.split('.');
  const clone = structuredClone(obj) as Mutable;

  let cur: Mutable = clone;
  for (let i = 0; i < parts.length - 1; i++) {
    const part = parts[i];
    const next = parts[i + 1];
    if (cur[part] == null) {
      cur[part] = /^\d+$/.test(next) ? [] : {};
    }
    cur = cur[part] as Mutable;
  }
  cur[parts[parts.length - 1]] = value;
  return clone as T;
}
