// ModelsView.tsx — vista de modelos: grilla + dropzone + ConfigWizard al seleccionar.
// Export default: lo exige el lazy() del router.

import { useState } from 'react';
import { ModelsGrid } from './components/ModelsGrid';
import { ModelDropzone } from './components/ModelDropzone';
import { ConfigWizardPanel } from './components/ConfigWizard/ConfigWizardPanel';

export default function ModelsView() {
  const [selectedFile, setSelectedFile] = useState<string | null>(null);

  return (
    <div className="flex h-full flex-col gap-4 overflow-y-auto p-4">
      <div className="grid grid-cols-[1fr_18rem] gap-4">
        <div className="rounded-[var(--radius-lg)] border border-border bg-surface p-4">
          <ModelsGrid selectedFile={selectedFile} onSelect={setSelectedFile} />
        </div>
        <ModelDropzone />
      </div>

      {selectedFile && (
        <div className="rounded-[var(--radius-lg)] border border-border bg-surface p-5">
          {/* key fuerza re-init del wizard al cambiar de modelo */}
          <ConfigWizardPanel
            key={selectedFile}
            file={selectedFile}
            onClose={() => setSelectedFile(null)}
          />
        </div>
      )}
    </div>
  );
}
