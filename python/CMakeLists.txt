find_package(Python COMPONENTS Interpreter Development REQUIRED)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.12.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(_angle_only_cpp
    src/bindings.cpp
    src/bind_core.cpp
    src/bind_coords.cpp
    src/bind_motion.cpp
    src/bind_measurement.cpp
    src/bind_filters.cpp
    src/bind_fusion.cpp
    src/bind_association.cpp
    src/bind_gpu.cpp
)

target_link_libraries(_angle_only_cpp PRIVATE angle_only)
target_include_directories(_angle_only_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

install(TARGETS _angle_only_cpp DESTINATION angle_only)
