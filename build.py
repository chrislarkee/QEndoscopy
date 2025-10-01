#!/usr/bin/env python3
"""
build.py - Scripted Nuitka build for the project.
"""

import subprocess
import shutil
import sys

def run_nuitka(entrypoint: str):
    cmd = [
        sys.executable, "-m", "nuitka",
        "--standalone",
        "--windows-console-mode=disable",
        "--include-package=vispy",
        "--include-package=vispy.glsl",
        "--include-module=vispy.glsl",
        "--include-module=vispy.app.backends._wx",
        "--include-module=wx._xml",
        "--include-data-dir=includes=.",
        "--module-parameter=torch-disable-jit=no",  
        "--output-dir=..\\build",
        entrypoint,
    ]
    
    print("Running Nuitka build:\n", " ".join(cmd))
    subprocess.run(cmd, check=True)
    
    #copy vispy.glsl manually? It segfaults without this    
    shutil.copytree("..\\venv\\Lib\\site-packages\\vispy\\glsl", "..\\build\\QEndoscopy.dist\\vispy\\glsl")


if __name__ == "__main__":
    run_nuitka("QEndoscopy.py")
