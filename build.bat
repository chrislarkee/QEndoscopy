python -OO -m nuitka --standalone --disable-console --include-data-dir=includes=. --nofollow-import-to=torch,midas --run QEndoscopy.py
@REM python -OO -m nuitka --standalone --include-package=timm --include-package=torch --include-plugin-directory=midas --include-package=midas --run QEndoscopy.py
@REM --disable-console 
@REM ..\qe_venv\Scripts\pyinstaller QEndoscopy.py -D --hidden-import timm --hidden-import midas --hidden-import socket
@REM -w

