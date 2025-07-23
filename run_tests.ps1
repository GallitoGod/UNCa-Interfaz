# Activar el entorno virtual
& ".venv/Scripts/Activate.ps1"

# Setear PYTHONPATH relativo a src
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$srcPath = Join-Path $projectRoot "src"
$env:PYTHONPATH = $srcPath

# Ejecutar pytest
pytest "$srcPath/api/func/tests/"
