#include "angle_only/gpu/dispatch.hpp"

namespace aot::gpu {

bool cuda_available() {
#ifdef AOT_HAS_CUDA
    // Runtime check for CUDA device
    extern bool cuda_runtime_available();
    return cuda_runtime_available();
#else
    return false;
#endif
}

int device_count() {
#ifdef AOT_HAS_CUDA
    extern int cuda_device_count();
    return cuda_device_count();
#else
    return 0;
#endif
}

} // namespace aot::gpu
