# Activar el entorno virtual
& ".venv/Scripts/Activate.ps1"

# Setear PYTHONPATH relativo a src
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$srcPath = Join-Path $projectRoot "src"
$env:PYTHONPATH = $srcPath

# Ejecutar pytest para pruebas en func
#pytest "$srcPath/api/func/tests"

# Ejecutar pytest para pruebas en output_pipeline
#pytest "$srcPath/api/func/output_pipeline/tests"

# Ejecutar pytest para pruebas en input_pipeline
#pytest "$srcPath/api/func/input_pipeline/tests"

# Ejecutar pytest para pruebas en reader_pipeline
pytest "$srcPath/api/func/reader_pipeline/tests"


