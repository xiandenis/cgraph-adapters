#include "check.hpp"

#include "cgraph/dynlib_plugin.hpp"
#include "cgraph/runtime.hpp"

int main(int argc, char** argv) {
  RTR_CHECK(argc >= 2);
  cgraph::Runtime rt;
  cgraph::load_dynlib_plugin(rt, argv[1], true);
  for (const char* id : {"rtr.rough_global_reg", "rtr.fine_registration",
                         "rtr.align_result.unpack", "rtr.point_cloud.load",
                         "rtr.registration_initializer", "rtr.registration_report",
                         "rtr.angle_downsample", "rtr.voxel_medoid_downsample",
                         "rtr.resolution_estimate", "rtr.align_result.save",
                         "rtr.align_result.load", "rtr.lidar_frame.save",
                         "rtr.lidar_frame.load", "rtr.global_matrix.save",
                         "rtr.global_matrix.load", "rtr.point_cloud.to_buffer",
                         "rtr.point_cloud.to_file"}) {
    const auto* op = rt.registry().get(id);
    RTR_CHECK(op != nullptr);
    RTR_CHECK(!op->usage().principle.empty());
  }
  RTR_CHECK(rt.registry().get("rtr.session.open") == nullptr);
  return 0;
}
