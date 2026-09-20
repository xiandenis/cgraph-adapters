#include "check.hpp"

#include "cgraph/dynlib_plugin.hpp"
#include "cgraph/runtime.hpp"

int main(int argc, char** argv) {
  RTR_CHECK(argc >= 2);
  cgraph::Runtime rt;
  cgraph::load_dynlib_plugin(rt, argv[1], true);
  for (const char* id : {"rtr.rough_global_reg", "rtr.fine_registration",
                         "rtr.align_result.unpack", "rtr.point_cloud.load",
                         "rtr.registration_initializer", "rtr.registration_report"}) {
    const auto* op = rt.registry().get(id);
    RTR_CHECK(op != nullptr);
    RTR_CHECK(!op->usage().principle.empty());
  }
  RTR_CHECK(rt.registry().get("rtr.session.open") == nullptr);
  return 0;
}
