python -OO -m nuitka --standalone --windows-console-mode=disable  --include-data-dir=includes=. --module-parameter=torch-disable-jit=no --run QEndoscopy.py
@REM --enable-plugins=upx
@REM ..\qe_venv\Scripts\pyinstaller QEndoscopy.py -D --hidden-import timm --hidden-import midas --hidden-import socket
@REM -w

