#include "check.hpp"
#include "fixture/ops.hpp"

#include "cgraph/validate.hpp"

#include <memory>
#include <string>

static cgraph::Node op_node(const cgraph::MemoryOperator& op, std::string id) {
  cgraph::Node node;
  node.id = std::move(id);
  node.kind = cgraph::NodeKind::Operator;
  node.op_id = std::string(op.op_id());
  node.inputs = op.signature().inputs;
  node.outputs = op.signature().outputs;
  return node;
}

int main() {
  cgraph::OpRegistry reg;
  fixture::register_fixture_ops(reg);

  const auto* cmat = reg.get("fx.const_matrix");
  const auto* csc = reg.get("fx.const_scalar");
  const auto* sin = reg.get("fx.sin");
  RTR_CHECK(cmat != nullptr);
  RTR_CHECK(csc != nullptr);
  RTR_CHECK(sin != nullptr);

  cgraph::Graph bad;
  bad.add_node(op_node(*cmat, "M"));
  bad.add_node(op_node(*sin, "S"));
  cgraph::Edge e;
  e.src_node = "M";
  e.src_port = "out";
  e.dst_node = "S";
  e.dst_port = "in";
  bad.add_edge(std::move(e));
  const cgraph::ValidationReport blocked = cgraph::validate(bad, reg);
  RTR_CHECK(!blocked.ok);
  bool saw_type = false;
  for (const auto& issue : blocked.errors) {
    if (issue.code == "type_mismatch") {
      saw_type = true;
    }
  }
  RTR_CHECK(saw_type);

  cgraph::Graph ok;
  ok.add_node(op_node(*csc, "x"));
  ok.add_node(op_node(*sin, "S"));
  cgraph::Edge good;
  good.src_node = "x";
  good.src_port = "out";
  good.dst_node = "S";
  good.dst_port = "in";
  ok.add_edge(std::move(good));
  const cgraph::ValidationReport ok_report = cgraph::validate(ok, reg);
  for (const auto& issue : ok_report.errors) {
    RTR_CHECK(issue.code != "type_mismatch");
    RTR_CHECK(issue.code != "semantic_mismatch");
  }
  return 0;
}
