# !/usr/bin/env python

import sys, os
import pathlib
import subprocess

# This script runs the build of this software.

def main():
    # What's the absolute location of this file?
    repo_dir = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))


    build_dir = repo_dir / pathlib.Path("build/")
    src_dir   = repo_dir / pathlib.Path("src/")

    # Create the build dir if needed:
    build_dir.mkdir(exist_ok=True)


    run_cmake(src_dir, build_dir)
    run_build(build_dir)

def run_build(_build_dir):

    # Run 'make' with -j in the build directory.


    command = ['make']
    command += ['-j', str(32)]

    proc = subprocess.Popen(
        command,
        cwd    = _build_dir,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT
    )

    while True:
        output = proc.stdout.readline()
        if proc.poll() is not None:
            break
        if output:
            print(output.strip().decode())
    # Get the return code
    rc = proc.poll()

    if rc != 0:
        raise Exception("BUILD stage failed")

def run_cmake(src_dir, _build_dir):

    # First, we check on some imports

    if 'LIB_TORCH' in os.environ:
        cmake_prefix = os.environ['LIB_TORCH']
    else:
        print("Failed to find LibTorch in env variables, looking for pytorch installation.")
        try:
            import torch
            cmake_prefix = torch.utils.cmake_prefix_path
        except:
            print("Failed to import torch from python")
            raise Exception("Could not find any installation of libtorch.")

    command = ['cmake']
    command += [f'-DCMAKE_PREFIX_PATH={cmake_prefix}']

    command += [str(src_dir)]

    proc = subprocess.Popen(
        command,
        cwd    = _build_dir,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT
    )

    while True:
        output = proc.stdout.readline()
        if proc.poll() is not None:
            break
        if output:
            print(output.strip().decode())
    # Get the return code
    rc = proc.poll()

    if rc != 0:
        raise Exception("CMAKE stage failed")

if __name__ == "__main__":
    main()
