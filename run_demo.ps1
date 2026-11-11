<#
 ColorSAR - one-command live demo

   .\run_demo.ps1          runs the whole demo (~5 min on CPU, ~1 min on GPU)
   .\run_demo.ps1 -Fast    skips the evaluation step (just the visuals, ~1 min)

 What it shows:
   1. SAR input -> Model 3 (direct SAR model), and the original DnCNN + Model 2 pipeline
   2. Urban SAR (hardest terrain) -> Model 3
   3. An optical satellite tile -> Model 1b (in-domain; the everyday photo
      sample_input.jpeg is out of domain, see README)
   4. A live evaluation of Model 3 and Model 1b on 32 held-out pairs each: PSNR/SSIM next
      to the no-color and mean-color baselines, a/b and brightness error

 All figures are written next to the script and opened at the end.
#>
param([switch]$Fast)

$ErrorActionPreference = "Stop"

function Step($n, $msg) {
    Write-Host ""
    Write-Host "[$n] $msg" -ForegroundColor Cyan
}

Step "1/4" "SAR (s1) -> Model 3, and the original DnCNN -> Model 2 pipeline"
$agri = Get-ChildItem "v_2\agri\s1\*.png" | Select-Object -First 1
python colorize.py --image $agri.FullName --out demo_1_sar.png
if ($LASTEXITCODE -ne 0) { throw "colorize.py failed on the SAR image" }
python colorize.py --image $agri.FullName --sar_model 2 --out demo_1_sar_model2.png
if ($LASTEXITCODE -ne 0) { throw "colorize.py failed on the SAR image (Model 2)" }

Step "2/4" "Urban SAR (hardest terrain) -> Model 3"
$urban = Get-ChildItem "v_2\urban\s1\*.png" | Select-Object -First 1
python colorize.py --image $urban.FullName --out demo_2_urban.png
if ($LASTEXITCODE -ne 0) { throw "colorize.py failed on the urban image" }

Step "3/4" "Optical satellite tile -> Model 1b"
$optical = Get-ChildItem "v_2\grassland\s2\*.png" | Select-Object -First 1
python colorize.py --image $optical.FullName --out demo_3_optical.png
if ($LASTEXITCODE -ne 0) { throw "colorize.py failed on the optical tile" }

if (-not $Fast) {
    Step "4/4" "Live evaluation on held-out pairs (Model 3, then Model 1b)"
    python evaluate.py --model 3 --n_images 32
    if ($LASTEXITCODE -ne 0) { throw "evaluate.py failed (Model 3)" }
    python evaluate.py --model 1 --checkpoint model1b_run/model_1.pth --n_images 32
    if ($LASTEXITCODE -ne 0) { throw "evaluate.py failed (Model 1b)" }
}

Step "done" "Opening the results..."
@(
    ".\demo_1_sar.png",
    ".\demo_1_sar_model2.png",
    ".\demo_2_urban.png",
    ".\demo_3_optical.png",
    ".\model3_run\sample_predictions.png",
    ".\model1b_run\sample_predictions.png"
) | ForEach-Object { if (Test-Path $_) { Invoke-Item $_ } }

Write-Host ""
Write-Host "Demo figures: demo_1_sar.png / demo_1_sar_model2.png / demo_2_urban.png / demo_3_optical.png" -ForegroundColor Green
Write-Host "Raw colorized images: demo_*_raw.png" -ForegroundColor Green
