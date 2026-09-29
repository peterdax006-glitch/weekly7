# Wait for a Test-loop process (by PID) to exit, then relaunch the loop detached on the current code.
# Used when the loop stops itself at a stale-code check after a code change. Never kills anything.
param([int]$WaitPid, [string]$Log = "data\loop2_run4.log")
Set-Location "C:\Users\Peter\weekly7"
while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 30 }
$already = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'livesim_loop2.py 60' }
if ($already) { "loop already running: $($already.ProcessId)" | Out-File -Append data\relaunch_loop.log; exit }
$env:PYTHONIOENCODING = 'utf-8'
$p = Start-Process -FilePath ".venv\Scripts\python.exe" -ArgumentList "-u","scripts/livesim_loop2.py","60" -WorkingDirectory "C:\Users\Peter\weekly7" -RedirectStandardOutput $Log -RedirectStandardError ($Log -replace '\.log$','.err') -WindowStyle Hidden -PassThru
"$(Get-Date -Format s) relaunched loop PID $($p.Id) after $WaitPid exited -> $Log" | Out-File -Append data\relaunch_loop.log
