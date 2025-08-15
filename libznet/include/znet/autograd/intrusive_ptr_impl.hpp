#pragma once

#include <atomic>
#include <cstddef>  // for std::nullptr_t
#include <utility>  // for std::move

namespace znet {

// 🔧 Required base class for intrusive pointer counting
class IntrusiveBase {
public:
    void increment_refcount() {
        refcount_.fetch_add(1, std::memory_order_relaxed);
    }

    int decrement_refcount() {
        return refcount_.fetch_sub(1, std::memory_order_acq_rel) - 1;
    }

    virtual ~IntrusiveBase() = default;

private:
    std::atomic<int> refcount_{1};
};

template <typename T>
class IntrusivePtr {
public:
    using element_type = T;

    IntrusivePtr() noexcept : ptr_(nullptr) {}
    IntrusivePtr(std::nullptr_t) noexcept : ptr_(nullptr) {}
    explicit IntrusivePtr(T* ptr) : ptr_(ptr) { retain(); }

    IntrusivePtr(const IntrusivePtr& other) : ptr_(other.ptr_) { retain(); }
    IntrusivePtr(IntrusivePtr&& other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }

    
    // ✅ Upcast constructor
    // template <typename U, typename = std::enable_if_t<std::is_convertible<U*, T*>::value>>
    // IntrusivePtr(const IntrusivePtr<U>& other) noexcept : ptr_(other.get()) {
    //     retain();
    // }
    template <typename U, typename = std::enable_if_t<std::is_convertible<U*, T*>::value>>
    IntrusivePtr(const IntrusivePtr<U>& other) : ptr_(other.get()) {
        retain();
    }

    IntrusivePtr& operator=(const IntrusivePtr& other) {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            retain();
        }
        return *this;
    }

    IntrusivePtr& operator=(IntrusivePtr&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    IntrusivePtr& operator=(std::nullptr_t) noexcept {
        release();
        ptr_ = nullptr;
        return *this;
    }

    ~IntrusivePtr() { release(); }

    void reset() {
        release();
        ptr_ = nullptr;
    }

    void reset(T* new_ptr) {
        release();
        ptr_ = new_ptr;
        retain();
    }

    T* get() const { return ptr_; }
    T& operator*() const { return *ptr_; }
    T* operator->() const { return ptr_; }
    operator bool() const { return ptr_ != nullptr; }

private:
    void retain() {
        if (ptr_) ptr_->increment_refcount();
    }

    void release() {
        if (ptr_ && ptr_->decrement_refcount() == 0) {
            delete ptr_;
        }
    }

    T* ptr_;
};

// ✅ Factory method (like make_shared)
template <typename T, typename... Args>
IntrusivePtr<T> make_ptr(Args&&... args) {
    return IntrusivePtr<T>(new T(std::forward<Args>(args)...));
}

} // namespace znet
