#include "cgraph/ops.hpp"
#include "fx_typed.hpp"

#include <memory>
#include <stdexcept>

namespace fixture {
namespace {

class CondGt0Op final : public cgraph::MemoryOperator {
 public:
  CondGt0Op() {
    op_id_ = "fx.cond_gt0";
    signature_.inputs["in"] = fx_typed::float_port("in");
    signature_.outputs["pred"] = fx_typed::bool_port("pred");
    capability_.summary = "Predicate: pred = (in > 0)";
    capability_.tags = {"math", "control", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire float into in; pred is bool for CONDITION / merge.";
    usage_.tune = "No parameters.";
    usage_.inspect = "pred true iff in > 0.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const double x = fx_typed::require_float_port(data_in, "in", "fx.cond_gt0");
    return {{"pred", fx_typed::make_bool_pred(x > 0.0)}};
  }
};

class BranchMergeOp final : public cgraph::MemoryOperator {
 public:
  BranchMergeOp() {
    op_id_ = "fx.branch_merge";
    signature_.inputs["true"] = fx_typed::float_port("true");
    signature_.inputs["false"] = fx_typed::float_port("false");
    signature_.inputs["pred"] = fx_typed::bool_port("pred");
    signature_.outputs["out"] = fx_typed::float_port("out");
    capability_.summary = "Select true/false float by pred";
    capability_.tags = {"math", "control", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire both float branches + bool pred; read out.";
    usage_.tune =
        "For not_scheduled false side, use IR CONDITION then/else graphs.";
    usage_.inspect = "out = pred ? true : false.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const bool pred = fx_typed::require_bool(
        fx_typed::require_obj(data_in, "pred", "fx.branch_merge"),
        "fx.branch_merge", "pred");
    const double t =
        fx_typed::require_float_port(data_in, "true", "fx.branch_merge");
    const double f =
        fx_typed::require_float_port(data_in, "false", "fx.branch_merge");
    return {{"out", fx_typed::make_number_float(pred ? t : f)}};
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
