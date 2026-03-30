from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import sys
import os


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(os.path.abspath(sourcedir))


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        ext_fullpath = os.fspath(self.get_ext_fullpath(ext.name))
        extdir = os.path.abspath(os.path.dirname(ext_fullpath))

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DAVX512=ON",
            "-DCUDA_ARCH=86",
            "-GNinja",
        ]

        build_args = ["-j12"]
        build_temp = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_temp, exist_ok=True)

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args],
            cwd=build_temp,
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", ".", *build_args],
            cwd=build_temp,
            check=True,
        )


setup(
    name="turboquant",
    version="0.1.0",
    author="ikharoz",
    description="TurboQuant: Extreme quantization for LLM inference",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    ext_modules=[CMakeExtension("turboquant._turboquant", sourcedir="../core")],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "numpy>=1.26.0",
        "transformers>=4.40.0",
        "safetensors>=0.4.0",
        "tqdm>=4.66.0",
        "rich>=13.7.0",
        "click>=8.1.0",
    ],
    extras_require={
        "dev": ["pytest", "pytest-benchmark", "mypy", "black", "ruff"],
        "server": ["fastapi", "uvicorn", "httpx", "pydantic"],
    },
)
