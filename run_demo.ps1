<#
 ColorSAR - one-command live demo

   .\run_demo.ps1          runs the whole demo (~3-4 min on CPU, <1 min on GPU)
   .\run_demo.ps1 -Fast    skips the evaluation step (just the visuals, ~30 s)

 What it shows:
   1. SAR input  -> auto-detect -> DnCNN denoise -> Model 2 -> colorized + ground truth
   2. Urban SAR (hardest terrain) through the same path
   3. A regular photo -> Model 1
   4. A live evaluation: prints the encoder fingerprint and PSNR/SSIM to compare
      against the upstream ColorSAR numbers (29.9230 / 0.9400 and 14.4466 / 0.2314)

 All figures are written next to the script and opened at the end.
#>
param([switch]$Fast)

$ErrorActionPreference = "Stop"

function Step($n, $msg) {
    Write-Host ""
    Write-Host "[$n] $msg" -ForegroundColor Cyan
}

Step "1/4" "SAR (s1) -> DnCNN -> Model 2 colorization"
$agri = Get-ChildItem "v_2\agri\s1\*.png" | Select-Object -First 1
python colorize.py --image $agri.FullName --out demo_1_sar.png
if ($LASTEXITCODE -ne 0) { throw "colorize.py failed on the SAR image" }

Step "2/4" "Urban SAR (hardest terrain) -> Model 2 colorization"
$urban = Get-ChildItem "v_2\urban\s1\*.png" | Select-Object -First 1
python colorize.py --image $urban.FullName --out demo_2_urban.png
if ($LASTEXITCODE -ne 0) { throw "colorize.py failed on the urban image" }

Step "3/4" "Regular photo -> Model 1 colorization"
python colorize.py --image sample_input.jpeg --mode gray --out demo_3_photo.png
if ($LASTEXITCODE -ne 0) { throw "colorize.py failed on the photo" }

if (-not $Fast) {
    Step "4/4" "Live evaluation (fingerprint + PSNR/SSIM on 32 image pairs)"
    python evaluate.py --model 1 --checkpoint model1_run/model_1.pth --n_images 32
    if ($LASTEXITCODE -ne 0) { throw "evaluate.py failed" }
}

Step "done" "Opening the results..."
@(
    ".\demo_1_sar.png",
    ".\demo_2_urban.png",
    ".\demo_3_photo.png",
    ".\model1_run\sample_predictions.png",
    ".\model2_run\sample_predictions.png"
) | ForEach-Object { if (Test-Path $_) { Invoke-Item $_ } }

Write-Host ""
Write-Host "Demo figures: demo_1_sar.png / demo_2_urban.png / demo_3_photo.png" -ForegroundColor Green
Write-Host "Raw colorized images: demo_*_raw.png" -ForegroundColor Green
