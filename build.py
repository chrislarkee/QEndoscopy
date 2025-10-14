#!/usr/bin/env python3
"""
build.py - Scripted Nuitka build for the project.
"""

import subprocess
import shutil
import sys
from datetime import date

def run_nuitka(entrypoint: str):
    #"--module-parameter=torch-disable-jit=no",  
    cmd = [
        sys.executable, "-m", "nuitka",
        "--standalone",
        "--windows-console-mode=disable",
        "--include-package=vispy",
        "--include-package=vispy.glsl",
        "--include-module=vispy.app.backends._wx",
        "--include-module=vispy.glsl.build_spatial_filters",        
        "--include-module=vispy.io._data",        
        "--include-module=wx._xml",
        "--include-data-dir=includes=.",
        "--nofollow-import-to=*.tests",        
        "--output-dir=..\\build",
        entrypoint,
    ]
    
    print("Running Nuitka build:\n", " ".join(cmd))
    subprocess.run(cmd, check=True)
    
    #copy vispy.glsl manually? It segfaults without this    
    shutil.copytree("..\\venv\\Lib\\site-packages\\vispy\\glsl", "..\\build\\QEndoscopy.dist\\vispy\\glsl")
    shutil.copytree("..\\venv\\Lib\\site-packages\\vispy\\io", "..\\build\\QEndoscopy.dist\\vispy\\io")
    
def run_7zip():    
    # Paths
    seven_zip = "C:\\Program Files\\7-Zip\\7z.exe"  # or just "7z" if in PATH
    
    timestamp = date.today().strftime("%Y%m%d")
    output_archive = f"..\\build\\QE_Build_{timestamp}.7z"
    source_folder = "..\\build\\QEndoscopy.dist"
    
    cmd = [seven_zip, 'a', '-mx7', '-r', '-y', output_archive, source_folder]
    
    subprocess.run(cmd, capture_output=False, check=True)


if __name__ == "__main__":
    run_nuitka("QEndoscopy.py")
    run_7zip()
    