// FileSource.tsx — carga de archivo (video o imagen). Crea un object URL y setea
// la fuente; el orquestador decide video (loop) vs imagen (one-shot). Sin espejo.

import { useRef } from 'react';
import { useStreamStore } from '../store/streamStore';
import { Button } from '@/shared/ui/Button';

export function FileSource() {
  const setFileVideo = useStreamStore((s) => s.setFileVideo);
  const setFileImage = useStreamStore((s) => s.setFileImage);
  const inputRef = useRef<HTMLInputElement>(null);

  function onFile(file: File) {
    const url = URL.createObjectURL(file);
    if (file.type.startsWith('video/')) setFileVideo(url);
    else if (file.type.startsWith('image/')) setFileImage(url);
    else {
      URL.revokeObjectURL(url);
      console.error('Tipo de archivo no soportado:', file.type);
    }
  }

  return (
    <div className="space-y-1.5">
      <label className="text-xs font-medium text-fg-muted">Archivo local</label>
      <input
        ref={inputRef}
        type="file"
        accept="video/*,image/*"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onFile(file);
          e.target.value = '';
        }}
      />
      <Button variant="outline" className="w-full" onClick={() => inputRef.current?.click()}>
        Cargar video o imagen
      </Button>
    </div>
  );
}
