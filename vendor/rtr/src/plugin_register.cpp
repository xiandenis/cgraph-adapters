#if defined(_WIN32)
#define CGRAPH_PLUGIN_EXPORT __declspec(dllexport)
#else
#define CGRAPH_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

#include "cgraph/runtime.hpp"
#include "rtr/ops.hpp"

extern "C" CGRAPH_PLUGIN_EXPORT void cgraph_plugin_register(cgraph::Runtime& rt) {
  rtr::register_rtr_ops(rt.registry());
}
