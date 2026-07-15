#!/usr/bin/env python3
"""
Build and package the darts Python distribution.

This script drives Meson for the native build and then stages the produced
extension module plus runtime data into the setuptools build tree.
"""

from __future__ import annotations

import codecs
import os
import shutil
import subprocess
import sysconfig
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as _build_py

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
except Exception:  # pragma: no cover - wheel is expected in build env
    _bdist_wheel = None


PY_DIR = Path(__file__).resolve().parent
ROOT_DIR = PY_DIR.parent
DEFAULT_MESON_BUILD_DIR = ROOT_DIR / "build" / "meson"


def long_description() -> str:
    readme = ROOT_DIR / "README.md"
    with codecs.open(readme, "r", "utf-8") as fp:
        return fp.read()


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    return Path(value).expanduser().resolve()


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _meson_binary() -> str:
    meson = shutil.which("meson")
    if not meson:
        raise RuntimeError("meson is required to build the native extension")
    return meson


def _build_options() -> list[str]:
    options = [
        "-Dbuild-python=true",
        "-Dbuild-tests=false",
        "-Dlink-static=true",
    ]
    onnx_dir = os.environ.get("DARTS_ONNXRUNTIME_DIR") or os.environ.get("ONNXRUNTIME_HOME")
    if onnx_dir:
        options.append(f"-Donnxruntime-dir={Path(onnx_dir).expanduser().resolve()}")
    options.append(f"-Dpython-include-dir={sysconfig.get_paths()['include']}")
    if os.environ.get("DARTS_DEBUG_MEMORY") in {"1", "true", "True", "yes"}:
        options.append("-Ddebug-memory=true")
    return options


def ensure_meson_build() -> Path:
    meson = _meson_binary()
    build_dir = Path(os.environ.get("DARTS_MESON_BUILD_DIR", DEFAULT_MESON_BUILD_DIR)).expanduser().resolve()
    build_dir.mkdir(parents=True, exist_ok=True)

    setup_cmd = [meson, "setup", "--reconfigure", str(build_dir), str(ROOT_DIR), *_build_options()]
    coredata = build_dir / "meson-private" / "coredata.dat"
    if not coredata.exists():
        setup_cmd = [meson, "setup", str(build_dir), str(ROOT_DIR), *_build_options()]
    _run(setup_cmd)
    _run([meson, "compile", "-C", str(build_dir)])
    install_dir = build_dir / "package-install"
    if install_dir.exists():
        shutil.rmtree(install_dir)
    _run([meson, "install", "-C", str(build_dir), "--destdir", str(install_dir)])
    return install_dir


def find_native_extension(build_dir: Path) -> Path:
    patterns = ("**/cdarts*.so", "**/cdarts*.pyd", "**/cdarts*.dylib")
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(p for p in build_dir.glob(pattern) if p.is_file())
    if not candidates:
        raise RuntimeError(f"unable to locate built cdarts extension under {build_dir}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def copy_tree(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copytree(src, dst, dirs_exist_ok=True)


def copy_runtime_data(dst: Path) -> None:
    """Stage only files required at runtime, never downloaded training corpora."""
    data_dir = ROOT_DIR / "data"
    for name in ("conf.json", "readme.md", "sources.json", "sources.lock.json"):
        shutil.copy2(data_dir / name, dst / name)
    for directory in ("codes", "demo", "kernel", "licenses", "models"):
        copy_tree(data_dir / directory, dst / directory)


def copy_runtime_libs_if_requested(package_dir: Path) -> None:
    runtime_dir = _env_path("DARTS_ONNXRUNTIME_DIR") or _env_path("ONNXRUNTIME_HOME")
    if runtime_dir:
        lib_dir = runtime_dir / "lib"
        if lib_dir.exists():
            for pattern in ("libonnxruntime*.so*", "libonnxruntime*.dylib", "onnxruntime*.dll"):
                for item in lib_dir.glob(pattern):
                    if item.is_file():
                        shutil.copy2(item, package_dir / item.name)

    deps_dir = _env_path("DARTS_DEPS_DIR")
    if deps_dir:
        for pattern in ("libprotobuf*.so*", "libjsoncpp*.so*", "libz.so*"):
            for item in (deps_dir / "sysroot" / "usr" / "lib").glob(f"**/{pattern}"):
                if item.is_file():
                    shutil.copy2(item, package_dir / item.name)


class build_py(_build_py):
    """Build python package data and stage the native extension."""

    def run(self):
        build_dir = ensure_meson_build()
        super().run()

        build_lib = Path(self.build_lib)
        package_dir = build_lib / "darts"
        package_dir.mkdir(parents=True, exist_ok=True)

        native_ext = find_native_extension(build_dir)
        shutil.copy2(native_ext, package_dir / native_ext.name)

        data_target = package_dir / "data"
        if data_target.exists():
            shutil.rmtree(data_target)
        data_target.mkdir(parents=True)
        copy_runtime_data(data_target)
        copy_runtime_libs_if_requested(package_dir)


if _bdist_wheel is not None:

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            super().finalize_options()
            self.root_is_pure = False

else:  # pragma: no cover
    bdist_wheel = None


setup_kwargs = dict(
    name="darts",
    author="Xu En",
    author_email="xuen@mokahr.com",
    description="darts python wrapper",
    long_description=long_description(),
    long_description_content_type="text/markdown",
    version="2.0.1",
    packages=find_packages(include=["darts", "darts.*"]),
    url="https://github.com/xuenhappy/darts",
    license="Apache",
    platforms=["Unix", "MacOS"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    cmdclass={
        "build_py": build_py,
        **({"bdist_wheel": bdist_wheel} if bdist_wheel is not None else {}),
    },
    package_data={
        "darts": [
            "*.so",
            "*.pyd",
            "*.dylib",
            "*.json",
            "*.txt",
            "*.mbs",
            "*.pbs",
            "*.bdf",
            "*.onnx",
            "*.map",
            "*.tmap",
            "*.proto",
        ]
    },
    include_package_data=True,
    python_requires=">=3.8",
)


setup(**setup_kwargs)
