#include "check.hpp"
#include "fixture/ops.hpp"
#include "rtr/ops.hpp"

#include "cgraph/error.hpp"
#include "cgraph/graph.hpp"
#include "cgraph/ops.hpp"
#include "cgraph/validate.hpp"

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
  rtr::register_rtr_ops(reg);

  const auto* box = reg.get("fx.box");
  const auto* unpack = reg.get("rtr.align_result.unpack");
  const auto* fine = reg.get("rtr.fine_registration");
  RTR_CHECK(box != nullptr);
  RTR_CHECK(unpack != nullptr);
  RTR_CHECK(fine != nullptr);

  cgraph::Graph blocked;
  blocked.add_node(op_node(*box, "box"));
  blocked.add_node(op_node(*unpack, "unpack"));
  cgraph::Edge bad;
  bad.src_node = "box";
  bad.src_port = "out";
  bad.dst_node = "unpack";
  bad.dst_port = "align";
  blocked.add_edge(std::move(bad));
  const cgraph::ValidationReport blocked_report = cgraph::validate(blocked, reg);
  RTR_CHECK(!blocked_report.ok);
  bool saw_type = false;
  for (const auto& issue : blocked_report.errors) {
    if (issue.code == "type_mismatch") {
      saw_type = true;
    }
  }
  RTR_CHECK(saw_type);

  cgraph::Graph ok_graph;
  ok_graph.add_node(op_node(*unpack, "unpack"));
  ok_graph.add_node(op_node(*fine, "fine"));
  cgraph::Edge good;
  good.src_node = "unpack";
  good.src_port = "matrix";
  good.dst_node = "fine";
  good.dst_port = "guess";
  ok_graph.add_edge(std::move(good));
  const cgraph::ValidationReport ok_report = cgraph::validate(ok_graph, reg);
  for (const auto& issue : ok_report.errors) {
    RTR_CHECK(issue.code != "type_mismatch");
    RTR_CHECK(issue.code != "semantic_mismatch");
    RTR_CHECK(issue.code != "kind_mismatch");
    RTR_CHECK(issue.code != "format_mismatch");
  }
  return 0;
}
