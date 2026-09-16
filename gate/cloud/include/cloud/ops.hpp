#pragma once

#include "cgraph/ops.hpp"

namespace cloud {

void register_plugin();
void register_ops(cgraph::OpRegistry& registry, cgraph::ConvertRegistry& converts);

}  // namespace cloud
