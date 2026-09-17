#include "cgraph/ops.hpp"
#include "fx_data.hpp"

#include <memory>
#include <stdexcept>

namespace fixture {
namespace {

const nlohmann::json kTUnit = {
    {"R", {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}},
    {"t", {0, 0, 0}},
};

class CoarsePoseOp final : public cgraph::MemoryOperator {
 public:
  CoarsePoseOp() {
    op_id_ = "fx.coarse_pose";
    signature_.inputs["src"] =
        cgraph::make_port("src", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.inputs["tgt"] =
        cgraph::make_port("tgt", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.outputs["T"] =
        cgraph::make_port("T", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    capability_.summary = "P0 unit-T pose stub (needs both src and tgt)";
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire src and tgt; read T (identity).";
    usage_.tune = "No parameters. Deterministic unit T.";
    usage_.inspect = "T is identity rigid transform. Not a real registrar.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const auto inputs = fx::unwrap(data_in);
    if (inputs.find("src") == inputs.end() || inputs.find("tgt") == inputs.end()) {
      throw std::invalid_argument("fx.coarse_pose: missing src or tgt");
    }
    return fx::wrap(signature_, {{"T", kTUnit}});
  }
};

class CoarseRegOp final : public cgraph::MemoryOperator {
 public:
  CoarseRegOp() {
    op_id_ = "fx.coarse_reg";
    signature_.inputs["src"] =
        cgraph::make_port("src", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.inputs["tgt"] =
        cgraph::make_port("tgt", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.outputs["T"] =
        cgraph::make_port("T", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.outputs["inliers"] =
        cgraph::make_port("inliers", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    capability_.summary = "P0 black-box coarse registration (unit T)";
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Same Signature as algo.coarse_reg.v1 Model.";
    usage_.tune = "No parameters.";
    usage_.inspect = "T is identity; inliers is [].";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const auto inputs = fx::unwrap(data_in);
    if (inputs.find("src") == inputs.end() || inputs.find("tgt") == inputs.end()) {
      throw std::invalid_argument("fx.coarse_reg: missing src or tgt");
    }
    return fx::wrap(signature_, {{"T", kTUnit}, {"inliers", nlohmann::json::array()}});
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_coarse_pose() {
  return std::make_shared<CoarsePoseOp>();
}

std::shared_ptr<cgraph::MemoryOperator> make_coarse_reg() {
  return std::make_shared<CoarseRegOp>();
}

}  // namespace fixture
