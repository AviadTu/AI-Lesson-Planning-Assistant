#Requires -Version 5.1
<#
    start.ps1 - One-command local startup for the AI Lesson Planning Assistant.

    Starts, in order:
        1. Ollama daemon  (embeddings)   - http://localhost:11434
        2. RAG service    (port 8001)    - services/rag
        3. Gateway service(port 8000)    - services/gateway  (also serves the frontend)
    Waits for each /health to succeed, then opens http://localhost:8000 in the
    default browser.

    Usage:
        .\start.ps1

    Notes:
        * Uses the existing .venv (never creates one, never installs packages).
        * Detects services already running and skips duplicate starts.
        * n8n is intentionally NOT started (not implemented yet).
#>

$ErrorActionPreference = 'Stop'

$Root       = $PSScriptRoot
$Python     = Join-Path $Root '.venv\Scripts\python.exe'
$OllamaUrl  = 'http://localhost:11434'
$RagUrl     = 'http://localhost:8001'
$GatewayUrl = 'http://localhost:8000'
$LogDir     = Join-Path $Root 'logs'

function Write-Status {
    param([string]$Component, [string]$Message, [string]$Color = 'Cyan')
    Write-Host ('[{0,-8}] {1}' -f $Component, $Message) -ForegroundColor $Color
}

function Stop-WithError {
    param([string]$Message)
    Write-Host ''
    Write-Host "ERROR: $Message" -ForegroundColor Red
    exit 1
}

function Test-Url {
    param([string]$Url)
    try {
        Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3 | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Wait-Health {
    param([string]$Url, [int]$TimeoutSec = 60)
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (Test-Url $Url) { return $true }
        Start-Sleep -Seconds 1
    }
    return $false
}

# ── Pre-flight ───────────────────────────────────────────────────────
if (-not (Test-Path $Python)) {
    Stop-WithError "Python virtual environment not found at '$Python'. Create '.venv' and install each service's requirements first. This script never creates a venv or installs packages."
}
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

Write-Host ''
Write-Host 'Starting AI Lesson Planning Assistant (local)...' -ForegroundColor White
Write-Host ''

# ── 1. Ollama daemon (embeddings) ────────────────────────────────────
if (Test-Url "$OllamaUrl/api/tags") {
    Write-Status 'Ollama' "already running at $OllamaUrl" 'Green'
} else {
    Write-Status 'Ollama' "not running - starting 'ollama serve'..." 'Yellow'
    try {
        Start-Process -FilePath 'ollama' -ArgumentList 'serve' -WindowStyle Hidden
    } catch {
        Stop-WithError "Could not launch 'ollama serve'. Is Ollama installed and on PATH? ($($_.Exception.Message))"
    }
    if (Wait-Health "$OllamaUrl/api/tags" 30) {
        Write-Status 'Ollama' "started and reachable at $OllamaUrl" 'Green'
    } else {
        Stop-WithError "Ollama did not become reachable at $OllamaUrl within 30s."
    }
}

# ── 2. RAG service (port 8001) ───────────────────────────────────────
if (Test-Url "$RagUrl/health") {
    Write-Status 'RAG' "already running at $RagUrl (skipping duplicate start)" 'Green'
} else {
    Write-Status 'RAG' 'starting from services/rag on port 8001...' 'Yellow'
    Start-Process -FilePath $Python -ArgumentList '-m', 'app.main' `
        -WorkingDirectory (Join-Path $Root 'services\rag') `
        -RedirectStandardOutput (Join-Path $LogDir 'rag.out.log') `
        -RedirectStandardError  (Join-Path $LogDir 'rag.err.log') `
        -NoNewWindow | Out-Null
    if (Wait-Health "$RagUrl/health" 60) {
        Write-Status 'RAG' "healthy at $RagUrl" 'Green'
    } else {
        Stop-WithError "RAG service did not become healthy at $RagUrl/health within 60s. See '$LogDir\rag.err.log'."
    }
}

# ── 3. Gateway service (port 8000, serves the frontend) ──────────────
if (Test-Url "$GatewayUrl/health") {
    Write-Status 'Gateway' "already running at $GatewayUrl (skipping duplicate start)" 'Green'
} else {
    Write-Status 'Gateway' 'starting from services/gateway on port 8000...' 'Yellow'
    Start-Process -FilePath $Python -ArgumentList '-m', 'app.main' `
        -WorkingDirectory (Join-Path $Root 'services\gateway') `
        -RedirectStandardOutput (Join-Path $LogDir 'gateway.out.log') `
        -RedirectStandardError  (Join-Path $LogDir 'gateway.err.log') `
        -NoNewWindow | Out-Null
    if (Wait-Health "$GatewayUrl/health" 60) {
        Write-Status 'Gateway' "healthy at $GatewayUrl" 'Green'
    } else {
        Stop-WithError "Gateway service did not become healthy at $GatewayUrl/health within 60s. See '$LogDir\gateway.err.log'."
    }
}

# ── 4. Open the app (only after both services are healthy) ───────────
Write-Host ''
Write-Status 'Browser' "opening $GatewayUrl" 'Green'
Start-Process $GatewayUrl

Write-Host ''
Write-Host 'All services healthy. The application is running.' -ForegroundColor Green
Write-Host "  Gateway (app): $GatewayUrl"
Write-Host "  RAG API:       $RagUrl"
Write-Host "  Ollama:        $OllamaUrl"
Write-Host "  Service logs:  $LogDir"
Write-Host ''
