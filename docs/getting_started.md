# Getting Started

## Prerequisites

- C++20 compiler (GCC 11+, Clang 14+, MSVC 2022+)
- CMake 3.24+
- Python 3.9+ (for bindings)
- CUDA Toolkit 11+ (optional, for GPU acceleration)

## Building from Source

### C++ Library

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build
```

### With CUDA

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DAOT_ENABLE_CUDA=ON
cmake --build build -j$(nproc)
```

### Python Package

```bash
pip install .
# or for development:
pip install -e ".[dev]"
```

### CMake Presets

```bash
cmake --preset default    # Debug, CPU-only
cmake --preset release    # Release, CPU-only
cmake --preset cuda       # Debug with CUDA
cmake --preset python     # Release with Python bindings
```

## Using in Your CMake Project

```cmake
find_package(angle_only REQUIRED)
target_link_libraries(my_app PRIVATE angle_only::angle_only)
```

Or as a subdirectory:

```cmake
add_subdirectory(angle_only)
target_link_libraries(my_app PRIVATE angle_only::angle_only)
```
