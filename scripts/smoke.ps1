$ErrorActionPreference = "Stop"

$baseUrl = $env:BASE_URL
if ([string]::IsNullOrWhiteSpace($baseUrl)) {
    $baseUrl = "http://localhost:5001"
}

$healthUrl = "$baseUrl/health"
$predictUrl = "$baseUrl/api/predict"

Write-Host "Smoke test against $baseUrl"

$healthResponse = $null
try {
    $healthResponse = Invoke-RestMethod -Uri $healthUrl -Method Get
} catch {
    Write-Error "Health check failed: $($_.Exception.Message)"
    exit 1
}

if ($healthResponse.status -ne "healthy") {
    Write-Error "Health check did not return status=healthy."
    $healthResponse | ConvertTo-Json -Depth 10 | Write-Host
    exit 1
}

if (-not $healthResponse.model_loaded) {
    Write-Error "Health check indicates model artifacts are not loaded."
    $healthResponse | ConvertTo-Json -Depth 10 | Write-Host
    exit 1
}

$payload = @{
    CreditScore     = 619
    Geography       = "France"
    Gender          = "Female"
    Age             = 42
    Tenure          = 2
    Balance         = 0
    NumOfProducts   = 1
    HasCrCard       = 1
    IsActiveMember  = 1
    EstimatedSalary = 101348.88
} | ConvertTo-Json

$predictResponse = $null
try {
    $predictResponse = Invoke-RestMethod -Uri $predictUrl -Method Post -ContentType "application/json" -Body $payload
} catch {
    Write-Error "Predict request failed: $($_.Exception.Message)"
    exit 1
}

if ($null -eq $predictResponse.p_churn) {
    Write-Error "Predict response missing p_churn."
    $predictResponse | ConvertTo-Json -Depth 10 | Write-Host
    exit 1
}

Write-Host "Smoke test passed."
