#include "fixture/fx_types.hpp"

#include "cgraph/type_catalog.hpp"
#include "cgraph/type_pack_yaml.hpp"

namespace fixture {
namespace {

// Embedded so the plugin does not depend on source-tree paths or a shared
// operators/cgraph-types.yaml name (RTR already owns that sidecar).
constexpr char kFxTypesYaml[] = R"YAML(
pack_id: fx.types
namespace: fx
depends: [cgraph]
semantics: []
types:
  - type_id: fx.type.vector
    form: tensor
    tensor: {dtype: f64, shape: []}
    semantics: [cgraph.semantic.number]
    display: {family: tensor, label: vector}
  - type_id: fx.type.matrix
    form: record
    semantics: [cgraph.semantic.number]
    display: {family: record, label: matrix}
    fields:
      - {name: rows, type_id: cgraph.type.int, semantic_id: cgraph.semantic.count, required: true}
      - {name: cols, type_id: cgraph.type.int, semantic_id: cgraph.semantic.count, required: true}
      - {name: data, type_id: "cgraph.type.list[cgraph.type.float]", semantic_id: cgraph.semantic.number, required: true}
)YAML";

}  // namespace

void register_fx_types() {
  if (cgraph::TypeCatalog::global().contains_pack("fx.types")) {
    return;
  }
  cgraph::TypeCatalog::global().install(
      cgraph::load_type_pack_yaml_string(kFxTypesYaml));
}

}  // namespace fixture
