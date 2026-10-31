# Run once from PowerShell:  powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
# Keeps EVERYTHING off C: (venv, pip cache, torch/ultralytics caches, temp).
$root = "E:\drone_track"
New-Item -ItemType Directory -Force "$root\.cache\pip","$root\.cache\torch","$root\.cache\tmp","$root\data","$root\weights" | Out-Null

# Persist env vars for the current user so every new terminal uses E:
[Environment]::SetEnvironmentVariable("PIP_CACHE_DIR", "$root\.cache\pip", "User")
[Environment]::SetEnvironmentVariable("TORCH_HOME",    "$root\.cache\torch", "User")
[Environment]::SetEnvironmentVariable("TMP",           "$root\.cache\tmp", "User")
[Environment]::SetEnvironmentVariable("TEMP",          "$root\.cache\tmp", "User")
$env:PIP_CACHE_DIR="$root\.cache\pip"; $env:TORCH_HOME="$root\.cache\torch"
$env:TMP="$root\.cache\tmp"; $env:TEMP="$root\.cache\tmp"

# Venv on E:
python -m venv "$root\.venv"
& "$root\.venv\Scripts\python.exe" -m pip install --upgrade pip
# CPU-only torch is enough for tracking dev/eval (small download). Training happens on Colab.
& "$root\.venv\Scripts\pip.exe" install torch torchvision --index-url https://download.pytorch.org/whl/cpu
& "$root\.venv\Scripts\pip.exe" install ultralytics opencv-python numpy scipy motmetrics onnx onnxruntime evo
Write-Host "Done. Activate with: E:\drone_track\.venv\Scripts\Activate.ps1"
