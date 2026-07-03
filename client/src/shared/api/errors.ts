// errors.ts — error tipado del cliente HTTP + normalizacion de los errores de Axios.
//
// El backend FastAPI responde con HTTPException -> { "detail": <string | lista> }.
//   - 404/409/501/500: detail suele ser un string ("modelo no encontrado", ...).
//   - 422 (validacion Pydantic): detail es una lista de objetos
//     [{ loc, msg, type }, ...].
// Esta capa colapsa todo eso en un ApiError con un message legible y el contexto
// crudo preservado (status + detail) para debugging. Principios aplicados:
// preservar contexto, mensajes con significado, errores tipados, no tragar nada.

import { AxiosError } from 'axios';

export type ApiErrorKind = 'http' | 'network' | 'timeout' | 'canceled' | 'unknown';

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status?: number;
  readonly detail?: unknown; // payload crudo del backend, sin tocar
  readonly cause?: unknown;

  constructor(
    message: string,
    opts: { kind: ApiErrorKind; status?: number; detail?: unknown; cause?: unknown },
  ) {
    super(message);
    this.name = 'ApiError';
    this.kind = opts.kind;
    this.status = opts.status;
    this.detail = opts.detail;
    this.cause = opts.cause;
  }
}

// Forma de un item de error de validacion de FastAPI/Pydantic (422).
interface ValidationItem {
  loc?: (string | number)[];
  msg?: string;
  type?: string;
}

// Extrae un mensaje legible del 'detail' del backend (string, lista 422, u otro).
function detailToMessage(detail: unknown): string | null {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    const items = detail as ValidationItem[];
    const parts = items
      .map((it) => {
        const where = it.loc?.slice(1).join('.') ?? it.loc?.join('.');
        return where ? `${where}: ${it.msg ?? 'invalido'}` : (it.msg ?? null);
      })
      .filter((p): p is string => !!p);
    if (parts.length) return parts.join('; ');
  }
  return null;
}

// Convierte cualquier error de Axios en un ApiError. Nunca tira: siempre devuelve
// un ApiError, asi los consumidores (hooks de TanStack Query) tienen un tipo unico.
export function normalizeAxiosError(err: unknown): ApiError {
  if (err instanceof ApiError) return err;

  if (err instanceof AxiosError) {
    if (err.code === 'ERR_CANCELED') {
      return new ApiError('Peticion cancelada', { kind: 'canceled', cause: err });
    }
    if (err.code === 'ECONNABORTED') {
      return new ApiError('La peticion al backend supero el tiempo limite', {
        kind: 'timeout',
        cause: err,
      });
    }
    if (err.response) {
      const status = err.response.status;
      const detail = (err.response.data as { detail?: unknown })?.detail;
      const msg =
        detailToMessage(detail) ?? `El backend respondio ${status}`;
      return new ApiError(msg, { kind: 'http', status, detail, cause: err });
    }
    // request salio pero no hubo respuesta -> backend caido / sin red
    return new ApiError('No se pudo contactar al backend (ver que uvicorn este corriendo)', {
      kind: 'network',
      cause: err,
    });
  }

  return new ApiError('Error desconocido en la peticion', { kind: 'unknown', cause: err });
}
