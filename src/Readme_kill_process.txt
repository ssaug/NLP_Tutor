1. Find the Process ID (PID):
netstat -aon | findstr :8501

2. Kill the Process:
taskkill /PID 12345 /F

3. Kill All Python Processes
taskkill /IM python.exe /F

