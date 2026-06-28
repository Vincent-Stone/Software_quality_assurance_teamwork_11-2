# 训练三个不同lambda值的模型
$lambdaValues = @(0.1, 0.5, 1.0)
$outputDir = "train_results"

# 确保输出目录存在
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

foreach ($lambda in $lambdaValues) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "开始训练 lambda = $lambda" -ForegroundColor Cyan
    Write-Host "========================================`n" -ForegroundColor Cyan
    
    $logFile = Join-Path $outputDir "lambda_${lambda}_training.log"
    
    # 运行训练命令，使用 -u 禁用缓冲
    $cmd = "cd robustness_verification; python -u train.py --lambda-weight $lambda --bound-method CROWN-IBP --epochs 5"
    
    Write-Host "执行命令: $cmd" -ForegroundColor Yellow
    Write-Host "日志文件: $logFile`n" -ForegroundColor Yellow
    
    # 执行命令并捕获输出
    Invoke-Expression $cmd 2>&1 | Tee-Object -FilePath $logFile
    
    Write-Host "`n========================================" -ForegroundColor Green
    Write-Host "lambda = $lambda 训练完成!" -ForegroundColor Green
    Write-Host "========================================`n" -ForegroundColor Green
}

Write-Host "`n所有训练任务完成!" -ForegroundColor Green
Write-Host "日志文件保存在: $outputDir" -ForegroundColor Green
