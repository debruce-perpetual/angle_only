cmake_minimum_required(VERSION 3.24)
project(angle_only
    VERSION 0.1.0
    LANGUAGES CXX
    DESCRIPTION "GPU-accelerated angle-only tracking library"
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(AOT_BUILD_TESTS "Build C++ tests" ON)
option(AOT_BUILD_PYTHON "Build Python bindings" OFF)
option(AOT_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(AOT_ENABLE_CUDA "Enable CUDA GPU acceleration" OFF)

# CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerWarnings)

# Dependencies
include(FetchContent)

# Eigen (header-only) — disable its tests/docs
set(BUILD_TESTING_SAVED ${BUILD_TESTING})
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(eigen)
set(BUILD_TESTING ${BUILD_TESTING_SAVED} CACHE BOOL "" FORCE)

# CUDA (optional)
if(AOT_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        find_package(CUDAToolkit REQUIRED)
        add_compile_definitions(AOT_HAS_CUDA=1)
        message(STATUS "CUDA enabled: ${CMAKE_CUDA_COMPILER_VERSION}")
    else()
        message(WARNING "CUDA compiler not found, building CPU-only")
        set(AOT_ENABLE_CUDA OFF)
    endif()
endif()

# Position-independent code (needed for shared library / Python module)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Main library
add_library(angle_only)
add_library(angle_only::angle_only ALIAS angle_only)

target_sources(angle_only PRIVATE
    src/core/state.cpp
    src/coords/transforms.cpp
    src/motion/constant_velocity_msc.cpp
    src/motion/cv.cpp
    src/motion/ca.cpp
    src/motion/ct.cpp
    src/measurement/msc_measurement.cpp
    src/measurement/spherical_measurement.cpp
    src/filters/msc_ekf.cpp
    src/filters/ekf.cpp
    src/filters/gmphd.cpp
    src/fusion/triangulate_los.cpp
    src/fusion/static_detection_fuser.cpp
    src/association/gating.cpp
    src/association/gnn.cpp
    src/association/jpda.cpp
)

if(AOT_ENABLE_CUDA)
    target_sources(angle_only PRIVATE
        src/gpu/dispatch.cpp
        src/gpu/device_buffer.cu
        src/gpu/batch_predict.cu
        src/gpu/batch_correct.cu
        src/gpu/batch_likelihood.cu
        src/gpu/batch_triangulate.cu
        src/gpu/batch_gating.cu
    )
    target_link_libraries(angle_only PRIVATE CUDA::cublas CUDA::cusolver)
else()
    target_sources(angle_only PRIVATE
        src/gpu/dispatch.cpp
        src/gpu/batch_operations_cpu.cpp
    )
endif()

target_include_directories(angle_only
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Mark Eigen as SYSTEM to suppress warnings from its headers
target_link_libraries(angle_only PUBLIC Eigen3::Eigen)
get_target_property(_eigen_inc Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(angle_only SYSTEM PUBLIC ${_eigen_inc})
aot_set_warnings(angle_only)

# Tests
if(AOT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()

# Python bindings
if(AOT_BUILD_PYTHON)
    add_subdirectory(python)
endif()

# Benchmarks
if(AOT_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Install
include(GNUInstallDirs)
install(TARGETS angle_only
    EXPORT angle_only-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/angle_only
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(EXPORT angle_only-targets
    NAMESPACE angle_only::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/angle_only
)
