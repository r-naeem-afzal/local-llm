@echo off
REM ===========================================================================
REM  Start the local-llm dashboard: the model server, the API, and the UI.
REM
REM  This replaces an earlier version that launched a JavaScript dashboard
REM  reading .local-llm-dashboard.html — a file that was never written, because
REM  the frontend became a Next.js app instead. That script always failed.
REM
REM  Three processes, started in dependency order:
REM    1. lms server   - the model server on :1234
REM    2. uvicorn      - the Python API on :7878
REM    3. next start   - the dashboard on :3000
REM ===========================================================================

setlocal
set ROOT=%~dp0
set TOOLKIT=%ROOT%toolkit
set DASHBOARD=%ROOT%dashboard
set LMS=%USERPROFILE%\.lmstudio\bin\lms.exe

echo.
echo === 1/3  Local model server ===
REM Called twice on purpose. From cold the first invocation reports
REM "Timed out waiting for LM Studio daemon to start" even with Bionic already
REM running, and the second succeeds immediately. Treating the first failure as
REM fatal would stop the script when nothing is actually wrong.
"%LMS%" server start >nul 2>&1
"%LMS%" server start
if errorlevel 1 (
    echo   WARNING: could not start the model server.
    echo   The dashboard will still run and will show "model server down".
)

echo.
echo === 2/3  API on http://127.0.0.1:7878 ===
cd /d "%TOOLKIT%"
REM `start "title" /min` launches this in its own minimised window so the script
REM can continue. Without it the API would block here and the dashboard would
REM never start.
start "local-llm API" /min cmd /c "python -m uvicorn local_llm.api:app --port 7878"

REM The dashboard's first request fires as soon as the page loads, so give the
REM API a moment to bind its port. Otherwise the UI opens, fails its first
REM fetch, and shows "cannot reach the API" until the next refresh.
timeout /t 4 /nobreak >nul

echo.
echo === 3/3  Dashboard on http://localhost:3000 ===
cd /d "%DASHBOARD%"
if not exist "node_modules" (
    echo   node_modules missing - installing dependencies first...
    call npm install --no-audit --no-fund
)
if not exist ".next" (
    echo   No production build found - building...
    call npm run build
)
REM Invoked through node rather than the npx shim: `npx next start` launched via
REM `start` does not reliably attach on Windows, and fails silently when it does
REM not. Calling the binary directly is dependable.
start "local-llm dashboard" /min cmd /c "node node_modules\next\dist\bin\next start -p 3000"

timeout /t 5 /nobreak >nul
start "" http://localhost:3000

echo.
echo Running. Close the two minimised windows to stop the API and dashboard.
echo   API       http://127.0.0.1:7878
echo   Dashboard http://localhost:3000
echo.
endlocal
