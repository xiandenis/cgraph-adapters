#include "cgraph/ops.hpp"

#include <memory>
#include <stdexcept>

namespace fixture {
namespace {

class CondGt0Op final : public cgraph::MemoryOperator {
 public:
  CondGt0Op() {
    op_id_ = "fx.cond_gt0";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, "json");
    signature_.outputs["pred"] =
        cgraph::make_port("pred", cgraph::PortKind::Value, "json");
    capability_.summary = "Predicate: pred = (in > 0)";
    capability_.tags = {"math", "control", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire scalar into in; pred is JSON bool for CONDITION.";
    usage_.tune = "No parameters.";
    usage_.inspect = "pred true iff in > 0.";
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const auto it = inputs.find("in");
    if (it == inputs.end() || !it->second.is_number()) {
      throw std::invalid_argument("fx.cond_gt0: missing numeric 'in'");
    }
    return {{"pred", it->second.get<double>() > 0.0}};
  }
};

class BranchMergeOp final : public cgraph::MemoryOperator {
 public:
  BranchMergeOp() {
    op_id_ = "fx.branch_merge";
    signature_.inputs["true"] =
        cgraph::make_port("true", cgraph::PortKind::Value, "json");
    signature_.inputs["false"] =
        cgraph::make_port("false", cgraph::PortKind::Value, "json");
    signature_.inputs["pred"] =
        cgraph::make_port("pred", cgraph::PortKind::Value, "json");
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, "json");
    capability_.summary = "Select true/false branch by pred";
    capability_.tags = {"math", "control", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire both branches + boolean pred; read out.";
    usage_.tune =
        "For not_scheduled false side, use IR CONDITION then/else graphs.";
    usage_.inspect = "out = pred ? true : false.";
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const auto pit = inputs.find("pred");
    const auto tit = inputs.find("true");
    const auto fit = inputs.find("false");
    if (pit == inputs.end() || !pit->second.is_boolean()) {
      throw std::invalid_argument("fx.branch_merge: pred must be boolean");
    }
    if (tit == inputs.end() || fit == inputs.end()) {
      throw std::invalid_argument("fx.branch_merge: missing true/false");
    }
    return {{"out", pit->second.get<bool>() ? tit->second : fit->second}};
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_cond_gt0() {
  return std::make_shared<CondGt0Op>();
}
std::shared_ptr<cgraph::MemoryOperator> make_branch_merge() {
  return std::make_shared<BranchMergeOp>();
}

}  // namespace fixture
