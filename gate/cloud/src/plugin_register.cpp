#if defined(_WIN32)
#define CGRAPH_PLUGIN_EXPORT __declspec(dllexport)
#else
#define CGRAPH_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

#include "cgraph/runtime.hpp"
#include "cloud/ops.hpp"

extern "C" CGRAPH_PLUGIN_EXPORT void cgraph_plugin_register(cgraph::Runtime& rt) {
  cloud::register_plugin();
  cloud::register_ops(rt.registry(), rt.converts());
}
