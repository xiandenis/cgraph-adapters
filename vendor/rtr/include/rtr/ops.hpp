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

inline void register_rtr_ops(cgraph::OpRegistry& registry) {
  register_rtr_types();
  register_unpack(registry);
  register_rough_global_reg(registry);
  register_fine_registration(registry);
  register_point_cloud_load(registry);
  register_registration_initializer(registry);
  register_registration_report(registry);
}

}  // namespace rtr
