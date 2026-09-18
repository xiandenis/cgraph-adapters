#pragma once

#include "cgraph/type_id.hpp"

namespace fixture {

void register_fx_types();

inline cgraph::TypeId type_vector() {
  return cgraph::TypeId::parse("fx.type.vector");
}

inline cgraph::TypeId type_matrix() {
  return cgraph::TypeId::parse("fx.type.matrix");
}

inline cgraph::TypeId type_vector_list() {
  return cgraph::type_ids::list(type_vector());
}

inline cgraph::TypeId type_float_list() {
  return cgraph::type_ids::list(cgraph::type_ids::floating());
}

inline cgraph::TypeId type_matrix_list() {
  return cgraph::type_ids::list(type_matrix());
}

}  // namespace fixture
