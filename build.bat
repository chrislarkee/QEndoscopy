@REM python -OO -m nuitka --standalone --windows-console-mode=disable  --include-data-dir=includes=. --module-parameter=torch-disable-jit=no --run QEndoscopy.py
@REM --enable-plugins=upx
..\venv\Scripts\pyinstaller QEndoscopy.py -D --hidden-import timm --hidden-import socket --hidden-import vispy.app.backends._wx --hidden-import vispy.glsl --noconfirm
@REM -w

