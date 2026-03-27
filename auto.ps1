$ErrorActionPreference = "Stop"

Write-Host "Starting Stage 1..."
python train.py --stage 1 --env bottleneck --num-agents 2 --workers 4 --train-batch-size 4000 --stop-iters 200

Write-Host "Stage 1 complete."

Write-Host "Starting Stage 2..."
python train.py --stage 2 --env bottleneck --num-agents 5 --workers 6 --train-batch-size 8000 --stop-iters 300 --resume "C:\Code\MARL-Traffic-System\checkpoints"

Write-Host "Stage 2 complete."

Write-Host "Starting Stage 3..."
python train.py --stage 3 --env bottleneck --num-agents 6 --workers 6 --train-batch-size 10000 --stop-iters 400 --resume "C:\Code\MARL-Traffic-System\checkpoints"

Write-Host "Training pipeline complete."