// src/autograd/grad_mode.cpp
#include <znet/autograd/grad_mode.hpp>

namespace znet {

static thread_local bool g_grad_enabled = true;

bool GradMode::is_enabled() { return g_grad_enabled; }
void GradMode::set_enabled(bool enabled) { g_grad_enabled = enabled; }

} // namespace znet
