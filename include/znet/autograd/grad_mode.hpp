// include/znet/autograd/grad_mode.hpp
#pragma once

namespace znet {

struct GradMode {
    // Is autograd building enabled on this thread?
    static bool is_enabled();
    // Force-enable/disable autograd building on this thread.
    static void set_enabled(bool enabled);

    // RAII: disable autograd within this scope
    struct NoGradGuard {
        bool prev_;
        NoGradGuard() : prev_(GradMode::is_enabled()) { GradMode::set_enabled(false); }
        ~NoGradGuard() { GradMode::set_enabled(prev_); }
    };

    // RAII: enable autograd within this scope
    struct EnableGradGuard {
        bool prev_;
        EnableGradGuard() : prev_(GradMode::is_enabled()) { GradMode::set_enabled(true); }
        ~EnableGradGuard() { GradMode::set_enabled(prev_); }
    };

    // RAII: set to specific value within this scope
    struct Scoped {
        bool prev_;
        Scoped(bool on) : prev_(GradMode::is_enabled()) { GradMode::set_enabled(on); }
        ~Scoped() { GradMode::set_enabled(prev_); }
    };
};

} // namespace znet
