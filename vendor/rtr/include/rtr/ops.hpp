#pragma once

#include "cgraph/ops.hpp"
#include "rtr/json.hpp"

namespace rtr {

void register_unpack(cgraph::OpRegistry& registry);
void register_rough_global_reg(cgraph::OpRegistry& registry);
void register_fine_registration(cgraph::OpRegistry& registry);
void register_point_cloud_load(cgraph::OpRegistry& registry);
void register_registration_initializer(cgraph::OpRegistry& registry);
void register_registration_report(cgraph::OpRegistry& registry);
void register_angle_downsample(cgraph::OpRegistry& registry);
void register_voxel_medoid_downsample(cgraph::OpRegistry& registry);
void register_resolution_estimate(cgraph::OpRegistry& registry);
void register_align_result_io(cgraph::OpRegistry& registry);
void register_lidar_frame_io(cgraph::OpRegistry& registry);
void register_global_matrix_io(cgraph::OpRegistry& registry);
void register_point_cloud_materialize(cgraph::OpRegistry& registry);

inline void register_rtr_ops(cgraph::OpRegistry& registry) {
  register_rtr_types();
  register_unpack(registry);
  register_rough_global_reg(registry);
  register_fine_registration(registry);
  register_point_cloud_load(registry);
  register_registration_initializer(registry);
  register_registration_report(registry);
  register_angle_downsample(registry);
  register_voxel_medoid_downsample(registry);
  register_resolution_estimate(registry);
  register_align_result_io(registry);
  register_lidar_frame_io(registry);
  register_global_matrix_io(registry);
  register_point_cloud_materialize(registry);
}

}  // namespace rtr
