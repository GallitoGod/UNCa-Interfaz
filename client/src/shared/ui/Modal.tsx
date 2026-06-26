// Modal.tsx — dialogo modal sobre el <dialog> NATIVO.
// Usar la plataforma (regla "native first"): <dialog>.showModal() ya trae focus
// trap, cierre con Escape y backdrop accesible, que a mano cuestan dias de ARIA.
// Aca solo se sincroniza el prop `open` con el elemento y se estiliza.

import { useEffect, useRef, type ReactNode } from 'react';
import { cn } from './cn';

interface ModalProps {
  open: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
  footer?: ReactNode;
}

export function Modal({ open, onClose, title, children, footer }: ModalProps) {
  const ref = useRef<HTMLDialogElement>(null);

  // Sincroniza el prop open con el metodo imperativo del <dialog>.
  useEffect(() => {
    const dlg = ref.current;
    if (!dlg) return;
    if (open && !dlg.open) dlg.showModal();
    else if (!open && dlg.open) dlg.close();
  }, [open]);

  if (!open) return null;

  return (
    <dialog
      ref={ref}
      // El evento 'close' nativo (Escape, dlg.close()) avisa al padre.
      onClose={onClose}
      // Click en el backdrop (target === dialog) cierra. El contenido va en un
      // <div> interno, asi clickear adentro no dispara este handler.
      onClick={(e) => {
        if (e.target === ref.current) onClose();
      }}
      className={cn(
        'm-auto w-[min(92vw,28rem)] rounded-[var(--radius-lg)] p-0',
        'bg-surface-raised text-fg border border-border',
        'backdrop:bg-black/50',
      )}
    >
      <div className="flex items-center justify-between border-b border-border px-5 py-3.5">
        <h3 className="text-sm font-semibold tracking-tight">{title}</h3>
        <button
          type="button"
          onClick={onClose}
          aria-label="Cerrar"
          className={cn(
            'grid size-7 place-items-center rounded-[var(--radius-sm)] text-fg-muted',
            'transition-colors hover:bg-control hover:text-fg',
          )}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>

      <div className="px-5 py-4">{children}</div>

      {footer && (
        <div className="flex justify-end gap-2 border-t border-border px-5 py-3.5">
          {footer}
        </div>
      )}
    </dialog>
  );
}
