@echo off
::This script exists to help you if you are getting illegal memory accesses when running your CUDA code
::Simply call the batch file in cmd specifying the path to your executable, and any trailing arguments required
::e.g. cudamemchk.bat "x64\Release\Particles.exe" CUDA

set compute-sanitizer="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\compute-sanitizer\compute-sanitizer.exe"
IF "%1"=="" (
    echo "Usage: %0 <executable> <run args>"
	pause
	goto :eof
)
%compute-sanitizer% --print-limit 1 %* 