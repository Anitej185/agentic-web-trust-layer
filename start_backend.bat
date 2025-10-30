@echo off
echo Starting AgentCert MVP Backend...
echo.
uvicorn backend:app --reload
pause

