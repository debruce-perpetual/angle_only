#pragma once

#include "mex.h"
#include <unordered_map>
#include <memory>
#include <atomic>
#include <string>

namespace aot::mex {

/// Thread-safe object store mapping uint64 handles to C++ objects.
/// Used to persist C++ objects across MEX calls from MATLAB.
template <typename T>
class ObjectStore {
public:
    static ObjectStore& instance() {
        static ObjectStore store;
        return store;
    }

    /// Create a new object and return its handle
    template <typename... Args>
    uint64_t create(Args&&... args) {
        uint64_t h = next_handle_++;
        objects_[h] = std::make_unique<T>(std::forward<Args>(args)...);
        return h;
    }

    /// Store an existing object and return its handle
    uint64_t store(std::unique_ptr<T> obj) {
        uint64_t h = next_handle_++;
        objects_[h] = std::move(obj);
        return h;
    }

    /// Get pointer to object (throws if not found)
    T* get(uint64_t handle) {
        auto it = objects_.find(handle);
        if (it == objects_.end()) {
            mexErrMsgIdAndTxt("aot:handle", "Invalid handle: %llu",
                              static_cast<unsigned long long>(handle));
            return nullptr;  // unreachable
        }
        return it->second.get();
    }

    /// Destroy object by handle
    void destroy(uint64_t handle) {
        auto it = objects_.find(handle);
        if (it != objects_.end()) {
            objects_.erase(it);
        }
    }

    /// Destroy all objects (called on mexAtExit)
    void clear() {
        objects_.clear();
    }

    /// Number of live objects
    size_t size() const {
        return objects_.size();
    }

private:
    ObjectStore() = default;
    ObjectStore(const ObjectStore&) = delete;
    ObjectStore& operator=(const ObjectStore&) = delete;

    std::unordered_map<uint64_t, std::unique_ptr<T>> objects_;
    std::atomic<uint64_t> next_handle_{1};
};

} // namespace aot::mex
