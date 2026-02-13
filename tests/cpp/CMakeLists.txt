include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.2
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(Catch2)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

function(aot_add_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE angle_only Catch2::Catch2WithMain)
    catch_discover_tests(${name})
endfunction()

aot_add_test(test_msc_state test_msc_state.cpp)
aot_add_test(test_transforms test_transforms.cpp)
aot_add_test(test_motion_models test_motion_models.cpp)
aot_add_test(test_measurement_models test_measurement_models.cpp)
aot_add_test(test_msc_ekf test_msc_ekf.cpp)
aot_add_test(test_triangulate test_triangulate.cpp)
aot_add_test(test_association test_association.cpp)
aot_add_test(test_gmphd test_gmphd.cpp)
aot_add_test(test_batch_ops test_batch_ops.cpp)
