# The whole fine-tuning experiment in one command (Windows, about 8-10 hours on an 8 GB laptop GPU):
# train the LoRA add-on, answer the 163 test questions three ways, then grade the answers with the LLM judge.
#     powershell -ExecutionPolicy Bypass -File finetune\run_all.ps1
# Every step is resumable: if the run is cut short, run the same command again and finished work is skipped
# (training restarts from zero only if it had not finished).
# Log: results\finetune_run.log

Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONWARNINGS = "ignore"
$env:PYTHONIOENCODING = "utf-8"  # Arabic and accents through the pipe to the log
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$py = ".venv\Scripts\python"
$log = "results\finetune_run.log"

function Step($name, [scriptblock]$block) {
    "`n=== $name  (start $(Get-Date -Format 'HH:mm'))" | Tee-Object -FilePath $log -Append
    & $block 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
    "=== $name  (end $(Get-Date -Format 'HH:mm'))" | Tee-Object -FilePath $log -Append
}

ollama stop qwen3:8b 2>$null

# 1. training (about 3.5-4 h); skipped if the add-on is already there
if (-not (Test-Path "finetune\out\qwen3-4b-demarches\adapter_config.json")) {
    Step "1/7 training" { & $py finetune\train_lora.py }
}

# 2. the fine-tuned 4B (Hugging Face backend, no Ollama)
$env:LLM_BACKEND = "hf"
$env:HF_TIMING = "1"  # prints the speed of every answer
$env:HF_ADAPTER = "finetune\out\qwen3-4b-demarches"
Step "2/7 fine-tuned 4B" { & $py scripts\run_answers.py --tag _ft4b --skip-judge }

# 3. the plain 4B, same setup, no add-on
Remove-Item Env:HF_ADAPTER
Step "3/7 base 4B" { & $py scripts\run_answers.py --tag _base4b --skip-judge }

# 4. v7 (Qwen3-8B with reasoning, Ollama) with the smaller context that fits in the GPU, to check it answers the same
Remove-Item Env:LLM_BACKEND
Remove-Item Env:HF_TIMING
$env:NUM_CTX_THINK = "8192"
if (-not (Get-Process ollama -ErrorAction SilentlyContinue)) {
    Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 15
}
Step "4/7 v7 at 8192" { & $py scripts\run_answers.py --tag _v7ctx8k --think --skip-judge }
Remove-Item Env:NUM_CTX_THINK

# 5-7. the LLM judge (Qwen3-8B with reasoning, Ollama) grades the three new sets of answers
foreach ($t in @("_ft4b", "_base4b", "_v7ctx8k")) {
    Step "judge $t" { & $py scripts\run_answers.py --tag $t --judge-only --judge-out "judgements2$t.jsonl" }
}
ollama stop qwen3:8b 2>$null

"`n=== all finished $(Get-Date -Format 'HH:mm')" | Tee-Object -FilePath $log -Append
