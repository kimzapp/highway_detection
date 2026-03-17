"""
Build Script for Highway Detection Vision App
Sử dụng PyInstaller để build ứng dụng Windows
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


def get_project_root() -> Path:
    """Lấy thư mục gốc của project"""
    return Path(__file__).parent.parent


def clean_build_artifacts():
    """Xóa các file build cũ"""
    root = get_project_root()
    
    dirs_to_clean = ['build', 'dist']
    for dir_name in dirs_to_clean:
        dir_path = root / dir_name
        if dir_path.exists():
            print(f"Removing {dir_path}...")
            shutil.rmtree(dir_path)
    
    # Remove .pyc files
    for pycache in root.rglob('__pycache__'):
        shutil.rmtree(pycache)


def check_dependencies():
    """Kiểm tra các dependencies cần thiết"""
    print("Checking dependencies...")
    
    required = ['torch', 'ultralytics', 'PyQt5', 'cv2', 'numpy', 'PyInstaller']
    missing = []
    
    for pkg in required:
        try:
            if pkg == 'cv2':
                import cv2
            elif pkg == 'PyInstaller':
                import PyInstaller
            else:
                __import__(pkg)
            print(f"  [OK] {pkg}")
        except ImportError:
            print(f"  [MISSING] {pkg}")
            missing.append(pkg)
    
    if missing:
        print(f"\nError: Missing packages: {missing}")
        print("Install them with: pip install " + " ".join(missing))
        return False
    
    return True


def run_pyinstaller(clean: bool = False, debug: bool = False):
    """
    Chạy PyInstaller với spec file
    
    Args:
        clean: Xóa build artifacts trước khi build
        debug: Build với console để debug
    """
    root = get_project_root()
    spec_file = root / 'vision_app.spec'
    
    if not spec_file.exists():
        print(f"Error: Spec file not found: {spec_file}")
        return False
    
    if clean:
        clean_build_artifacts()
    
    # Build command
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--noconfirm',  # Replace output without asking
        '--clean',      # Clean cache before building
        str(spec_file)
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    print("This may take several minutes...\n")
    
    # Run PyInstaller
    try:
        result = subprocess.run(
            cmd,
            cwd=str(root),
            check=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Build failed with error code: {e.returncode}")
        return False


def verify_build():
    """Kiểm tra kết quả build"""
    root = get_project_root()
    exe_path = root / 'dist' / 'vision_app.exe'
    
    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print(f"\nBuild successful!")
        print(f"  Output: {exe_path}")
        print(f"  Size: {size_mb:.1f} MB")
        return True
    else:
        print("\nBuild failed - executable not found")
        return False


def run_release_pipeline(set_version: str = None, installer: bool = False):
    """Run release staging script after build."""
    root = get_project_root()
    release_script = root / 'scripts' / 'release.py'

    if not release_script.exists():
        print(f"Error: Release script not found: {release_script}")
        return False

    cmd = [sys.executable, str(release_script)]
    if set_version:
        cmd.extend(['--set-version', set_version])
    if installer:
        cmd.append('--installer')

    print(f"\nRunning release pipeline: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=str(root), check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Release pipeline failed with error code: {e.returncode}")
        return False


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build Highway Detection Vision App')
    parser.add_argument('--clean', action='store_true', 
                       help='Clean build artifacts before building')
    parser.add_argument('--debug', action='store_true',
                       help='Build with console window for debugging')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check dependencies, do not build')
    parser.add_argument('--release', action='store_true',
                       help='Stage release folder after successful build')
    parser.add_argument('--installer', action='store_true',
                       help='Stage release folder and build installer after successful build')
    parser.add_argument('--set-version', type=str,
                       help='Set semantic version before release, e.g. 1.2.3')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Highway Detection Vision App - Build Script")
    print("=" * 50 + "\n")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    if args.check_only:
        print("\nDependency check passed!")
        return
    
    # Run build
    if run_pyinstaller(clean=args.clean, debug=args.debug):
        if not verify_build():
            sys.exit(1)

        if args.release or args.installer or args.set_version:
            if not run_release_pipeline(set_version=args.set_version, installer=args.installer):
                sys.exit(1)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
