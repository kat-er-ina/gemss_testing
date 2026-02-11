# PowerShell script for running all experimental tiers sequentially with logging

# Run all tiers (1-6)
#     .\run_tiers.ps1

# Run specific tiers
#     .\run_tiers.ps1 -tiers @("1", "2", "3")

# Skip tier 7 (unreliable and unbalanced responses) - tier under construction
#     .\run_tiers.ps1 -skipTier7

# Use custom parameters file
#     .\run_tiers.ps1 -parametersFile "custom_experiments.json"

# Quick test run of shortened tiers 1 and 2 only
#    .\run_tiers.ps1 -tiers @("1", "2") -parametersFile "experiments_parameters_short.json"

#-----------------------------------------------------------------------------------

param(
    [string]$parametersFile = "experiment_parameters.json",
    [string[]]$tiers = @("1", "2", "3", "4", "5", "6", "7"),
    [switch]$skipTier7
)

# --- Validate parameters file exists ---
if (-not (Test-Path $parametersFile)) {
    Write-Error "Parameters file '$parametersFile' not found. Please ensure the file exists in the current directory."
    exit 1
}

# --- Load experiment data to validate available tiers ---
Write-Host "Loading experiment parameters from: $parametersFile"
try {
    $experimentData = Get-Content $parametersFile -Raw | ConvertFrom-Json
}
catch {
    Write-Error "Failed to parse parameters file: $($_.Exception.Message)"
    exit 1
}

# Validate all requested tiers exist
$availableTiers = $experimentData.tiers.PSObject.Properties.Name | ForEach-Object { $_.Substring(4) }
foreach ($tier in $tiers) {
    if ($tier -notin $availableTiers) {
        Write-Error "Invalid tier '$tier'. Available tiers: $($availableTiers -join ', ')"
        exit 1
    }
}

# Remove tier 7 if skipTier7 is specified
if ($skipTier7) {
    $tiers = $tiers | Where-Object { $_ -ne "7" }
    Write-Host "Skipping Tier 7 (unreliable and unbalanced responses) as requested."
}

# --- Setup logging directory with better error handling ---
$currentDir = Get-Location
Write-Host "Current directory: $currentDir"

$resultsDir = Join-Path $currentDir "results"
$logsDir = Join-Path $resultsDir "logs"

Write-Host "Results directory: $resultsDir"
Write-Host "Logs directory: $logsDir"

# Create directories with error handling
try {
    if (-not (Test-Path $resultsDir)) {
        Write-Host "Creating results directory: $resultsDir"
        New-Item -ItemType Directory -Path $resultsDir -Force | Out-Null
    }
    
    if (-not (Test-Path $logsDir)) {
        Write-Host "Creating logs directory: $logsDir"
        New-Item -ItemType Directory -Path $logsDir -Force | Out-Null
    }
    
    # Verify directories were created
    if (-not (Test-Path $logsDir)) {
        throw "Failed to create logs directory: $logsDir"
    }
}
catch {
    Write-Error "Failed to create directory structure: $($_.Exception.Message)"
    exit 1
}

# --- Display run overview ---
Write-Host ""
Write-Host "================================================================================"
Write-Host "                    GEMSS EXPERIMENTAL TIERS - FULL RUN"
Write-Host "================================================================================"
Write-Host "Start time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "Tiers to run: $($tiers -join ', ')"
Write-Host "Parameters file: $parametersFile"
Write-Host "Logs directory: $logsDir"
Write-Host ""

# Calculate total experiments across all tiers
$totalExperiments = 0
$tierSummary = @()
foreach ($tier in $tiers) {
    $tierKey = "tier$tier"
    $tierData = $experimentData.tiers.$tierKey
    $tierExperimentCount = $tierData.combinations.Count
    $totalExperiments += $tierExperimentCount
    
    $tierSummary += [PSCustomObject]@{
        Tier = $tier
        Name = $tierData.name
        Experiments = $tierExperimentCount
        Type = if ($tierData.algorithm_parameters.BINARIZE) { "Binary Classification" } else { "Regression" }
    }
}

Write-Host "TIER SUMMARY:"
Write-Host "-------------"
$tierSummary | ForEach-Object {
    Write-Host "Tier $($_.Tier): $($_.Name)"
    Write-Host "  - Experiments: $($_.Experiments)"
    Write-Host "  - Type: $($_.Type)"
    Write-Host ""
}
Write-Host "TOTAL EXPERIMENTS ACROSS ALL TIERS: $totalExperiments"
Write-Host "================================================================================"
Write-Host ""

# --- Ask for confirmation ---
$confirmation = Read-Host "This will run $totalExperiments experiments across $($tiers.Count) tiers. Continue? (y/N)"
if ($confirmation -notmatch '^[Yy]') {
    Write-Host "Operation cancelled."
    exit 0
}

# --- Check if run_sweep_with_tiers.ps1 exists ---
$sweepScript = "run_sweep_with_tiers.ps1"
if (-not (Test-Path $sweepScript)) {
    Write-Error "Required script '$sweepScript' not found in current directory."
    exit 1
}

# --- Run each tier ---
$overallStartTime = Get-Date
$tierResults = @()

foreach ($tier in $tiers) {
    $tierStartTime = Get-Date
    $tierKey = "tier$tier"
    $tierData = $experimentData.tiers.$tierKey
    $logFile = Join-Path $logsDir "log_tier_$tier.txt"
    $errorFile = Join-Path $logsDir "log_tier_$tier.errors.txt"
    
    Write-Host ""
    Write-Host "================================================================================"
    Write-Host "STARTING TIER $tier : $($tierData.name)"
    Write-Host "================================================================================"
    Write-Host "Tier $tier - Start time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host "Tier $tier - Experiments: $($tierData.combinations.Count)"
    Write-Host "Tier $tier - Type: $(if ($tierData.algorithm_parameters.BINARIZE) { 'Binary Classification' } else { 'Regression' })"
    Write-Host "Tier $tier - Log file: $logFile"
    Write-Host "Tier $tier - Error file: $errorFile"
    Write-Host ""
    
    # Run the tier sweep and capture all output to log file
    try {
        Write-Host "Executing: powershell.exe -File $sweepScript -tier $tier -parametersFile $parametersFile"
        
        $tierProcess = Start-Process -FilePath "powershell.exe" `
            -ArgumentList "-ExecutionPolicy", "Bypass", "-File", $sweepScript, "-tier", $tier, "-parametersFile", $parametersFile `
            -RedirectStandardOutput $logFile `
            -RedirectStandardError $errorFile `
            -NoNewWindow `
            -Wait `
            -PassThru
        
        $tierEndTime = Get-Date
        $tierDuration = $tierEndTime - $tierStartTime
        
        if ($tierProcess.ExitCode -eq 0) {
            $status = "SUCCESS"
            Write-Host "Tier $tier completed successfully!" -ForegroundColor Green
        } else {
            $status = "FAILED"
            Write-Host "Tier $tier failed with exit code: $($tierProcess.ExitCode)" -ForegroundColor Red
            
            # Display error information if available
            if (Test-Path $errorFile) {
                $errorContent = Get-Content $errorFile -Raw
                if ($errorContent) {
                    Write-Host "Error details:" -ForegroundColor Red
                    Write-Host $errorContent -ForegroundColor Red
                }
            }
        }
    }
    catch {
        $tierEndTime = Get-Date
        $tierDuration = $tierEndTime - $tierStartTime
        $status = "ERROR"
        Write-Host "Tier $tier encountered an error: $($_.Exception.Message)" -ForegroundColor Red
    }
    
    # Store tier results
    $tierResults += [PSCustomObject]@{
        Tier = $tier
        Name = $tierData.name
        Status = $status
        Experiments = $tierExperimentCount
        Duration = $tierDuration
        StartTime = $tierStartTime
        EndTime = $tierEndTime
        LogFile = $logFile
        ErrorFile = $errorFile
    }
    
    Write-Host "Tier $tier - End time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host "Tier $tier - Duration: $($tierDuration.ToString('hh\:mm\:ss'))"
    Write-Host "Tier $tier - Status: $status"
}

# --- Overall completion summary ---
$overallEndTime = Get-Date
$overallDuration = $overallEndTime - $overallStartTime

Write-Host ""
Write-Host "================================================================================"
Write-Host "                    ALL TIERS COMPLETED"
Write-Host "================================================================================"
Write-Host "Overall start time: $($overallStartTime.ToString('yyyy-MM-dd HH:mm:ss'))"
Write-Host "Overall end time: $($overallEndTime.ToString('yyyy-MM-dd HH:mm:ss'))"
Write-Host "Total duration: $($overallDuration.ToString('hh\:mm\:ss'))"
Write-Host ""

Write-Host "TIER EXECUTION SUMMARY:"
Write-Host "-----------------------"
$tierResults | ForEach-Object {
    $statusColor = switch ($_.Status) {
        "SUCCESS" { "Green" }
        "FAILED" { "Red" }
        "ERROR" { "Magenta" }
        default { "White" }
    }
    
    Write-Host "Tier $($_.Tier): $($_.Name)" 
    Write-Host "  Status: " -NoNewline
    Write-Host $_.Status -ForegroundColor $statusColor
    Write-Host "  Duration: $($_.Duration.ToString('hh\:mm\:ss'))"
    Write-Host "  Experiments: $($_.Experiments)"
    Write-Host "  Log: $($_.LogFile)"
    if (Test-Path $_.ErrorFile) {
        Write-Host "  Errors: $($_.ErrorFile)" -ForegroundColor Yellow
    }
    Write-Host ""
}

# --- Generate summary statistics ---
$successfulTiers = $tierResults | Where-Object { $_.Status -eq "SUCCESS" }
$failedTiers = $tierResults | Where-Object { $_.Status -ne "SUCCESS" }
$totalSuccessfulExperiments = if ($successfulTiers) { ($successfulTiers | Measure-Object -Property Experiments -Sum).Sum } else { 0 }

Write-Host "EXECUTION STATISTICS:"
Write-Host "--------------------"
Write-Host "Total tiers attempted: $($tierResults.Count)"
Write-Host "Successful tiers: $($successfulTiers.Count)"
Write-Host "Failed tiers: $($failedTiers.Count)"
Write-Host "Total successful experiments: $totalSuccessfulExperiments"
Write-Host "Total planned experiments: $totalExperiments"
Write-Host ""

if ($failedTiers.Count -gt 0) {
    Write-Host "FAILED TIERS:" -ForegroundColor Red
    $failedTiers | ForEach-Object {
        Write-Host "  - Tier $($_.Tier): $($_.Name) ($($_.Status))" -ForegroundColor Red
    }
    Write-Host ""
    Write-Host "Check error files for details:" -ForegroundColor Yellow
    $failedTiers | ForEach-Object {
        Write-Host "  - $($_.ErrorFile)" -ForegroundColor Yellow
    }
    Write-Host ""
}

# --- Save execution summary to file with better error handling ---
try {
    # Ensure logsDir is valid before using it
    if (-not $logsDir -or -not (Test-Path $logsDir)) {
        throw "Logs directory is invalid or does not exist: $logsDir"
    }
    
    $summaryFile = Join-Path $logsDir "execution_summary_$(Get-Date -Format 'yyyyMMdd_HHmm').txt"
    
    $summaryContent = @"
GEMSS EXPERIMENTAL TIERS - EXECUTION SUMMARY
============================================
Execution Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
Parameters File: $parametersFile
User: $env:USERNAME
Computer: $env:COMPUTERNAME
Current Directory: $currentDir

OVERALL TIMING:
Start: $($overallStartTime.ToString('yyyy-MM-dd HH:mm:ss'))
End: $($overallEndTime.ToString('yyyy-MM-dd HH:mm:ss'))
Duration: $($overallDuration.ToString('hh\:mm\:ss'))

TIER RESULTS:
$($tierResults | ForEach-Object {
"Tier $($_.Tier): $($_.Name)
  Status: $($_.Status)
  Duration: $($_.Duration.ToString('hh\:mm\:ss'))
  Experiments: $($_.Experiments)
  Log File: $($_.LogFile)
  Error File: $($_.ErrorFile)
"
} | Out-String)

STATISTICS:
Total tiers: $($tierResults.Count)
Successful: $($successfulTiers.Count)
Failed: $($failedTiers.Count)
Total experiments: $totalSuccessfulExperiments / $totalExperiments
"@

    Set-Content -Path $summaryFile -Value $summaryContent
    Write-Host "Execution summary saved to: $summaryFile"
}
catch {
    Write-Warning "Failed to save execution summary: $($_.Exception.Message)"
    Write-Host "Logs directory was: $logsDir"
}

Write-Host ""
Write-Host "================================================================================"
if ($failedTiers.Count -eq 0) {
    Write-Host "ALL TIERS COMPLETED SUCCESSFULLY!" -ForegroundColor Green
} else {
    Write-Host "EXECUTION COMPLETED WITH SOME FAILURES" -ForegroundColor Yellow
}
Write-Host "Check individual log files in $logsDir for detailed output."
Write-Host "================================================================================"
