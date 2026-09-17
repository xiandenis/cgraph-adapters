#pragma once

#include "cgraph/ops.hpp"

namespace rtr {

void register_unpack(cgraph::OpRegistry& registry);
void register_rough_global_reg(cgraph::OpRegistry& registry);
void register_fine_registration(cgraph::OpRegistry& registry);

inline void register_rtr_ops(cgraph::OpRegistry& registry) {
  register_unpack(registry);
  register_rough_global_reg(registry);
  register_fine_registration(registry);
}

}  // namespace rtr
