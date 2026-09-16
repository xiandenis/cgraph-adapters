#include "cgraph/ops.hpp"

#include <memory>
#include <stdexcept>

namespace fixture {
namespace {

class BoxOp final : public cgraph::MemoryOperator {
 public:
  BoxOp() {
    op_id_ = "fx.box";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, "json");
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, "json");
    cgraph::ParamSpec tag;
    tag.name = "tag";
    tag.dtype = "string";
    tag.invalidate = true;
    tag.bindable = false;
    signature_.params["tag"] = std::move(tag);
    capability_.summary = "Black-box stamp (same result as fx.stamp)";
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire json into in; read out. Same Signature as fx.stamp.";
    usage_.tune = "params.tag (invalidate).";
    usage_.inspect = "out is {\"p\": in, \"t\": tag}.";
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs,
      const nlohmann::json& params, const cgraph::ExecContext&) const override {
    const auto it = inputs.find("in");
    if (it == inputs.end()) {
      throw std::invalid_argument("fx.box: missing input 'in'");
    }
    if (!params.is_object() || !params.contains("tag") || !params["tag"].is_string()) {
      throw std::invalid_argument("fx.box: missing or invalid param 'tag'");
    }
    nlohmann::json out = nlohmann::json::object();
    out["p"] = it->second;
    out["t"] = params["tag"];
    return {{"out", std::move(out)}};
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_box() {
  return std::make_shared<BoxOp>();
}

}  // namespace fixture
