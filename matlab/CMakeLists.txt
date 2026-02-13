find_package(Matlab COMPONENTS MX_LIBRARY)

if(NOT Matlab_FOUND)
    message(STATUS "MATLAB not found — skipping MEX bindings")
    return()
endif()

message(STATUS "MATLAB found: ${Matlab_ROOT_DIR}")
message(STATUS "  MEX extension: ${Matlab_MEX_EXTENSION}")

matlab_add_mex(
    NAME angle_only_mex
    SRC src/angle_only_mex.cpp
    OUTPUT_NAME angle_only_mex
    LINK_TO angle_only
)

target_include_directories(angle_only_mex PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Install MEX binary and MATLAB files together
install(TARGETS angle_only_mex
    DESTINATION matlab
)
install(DIRECTORY +aot
    DESTINATION matlab
)
install(FILES setup.m
    DESTINATION matlab
)
install(DIRECTORY examples
    DESTINATION matlab
)
