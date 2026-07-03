// useCameras.ts — enumera las camaras disponibles. Pide permiso primero (si no,
// los labels vienen vacios). Es API del browser, no del backend: NO usa TanStack Query.

import { useCallback, useEffect, useState } from 'react';

export function useCameras() {
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);

  const refresh = useCallback(async () => {
    try {
      // Permiso primero para poblar labels; se sueltan los tracks enseguida.
      const tmp = await navigator.mediaDevices.getUserMedia({ video: true });
      tmp.getTracks().forEach((t) => t.stop());
      const devices = await navigator.mediaDevices.enumerateDevices();
      setCameras(devices.filter((d) => d.kind === 'videoinput'));
    } catch (err) {
      console.error('Error enumerando camaras:', err);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return { cameras, refresh };
}
