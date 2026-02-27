from setuptools import setup, Extension

class get_pybind_include(object):
    def __str__(self):
        import pybind11
        return pybind11.get_include()

ext_modules = [
    Extension(
        "fast_scanner",
        ["src/cpp/fast_scanner.cpp"],
        include_dirs=[get_pybind_include()],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"]
    ),
]

setup(
    ext_modules=ext_modules,
    setup_requires=["pybind11>=2.10.0"],
)
