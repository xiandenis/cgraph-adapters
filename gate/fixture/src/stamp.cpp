#include "cgraph/ops.hpp"
#include "fx_data.hpp"

#include <memory>
#include <stdexcept>

namespace fixture {
namespace {

class StampOp final : public cgraph::MemoryOperator {
 public:
  StampOp() {
    op_id_ = "fx.stamp";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    cgraph::ParamSpec tag;
    tag.name = "tag";
    tag.dtype = "string";
    tag.invalidate = true;
    tag.doc = "Identity tag wrapped into out.t";
    tag.bindable = false;
    signature_.params["tag"] = std::move(tag);
    capability_.summary = "Wrap a JSON value with a tag";
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire any json value into in; read out.";
    usage_.tune = "Set params.tag (string, invalidate). Changing tag changes node_key.";
    usage_.inspect = "out is {\"p\": in, \"t\": tag}.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in,
      const nlohmann::json& params, const cgraph::ExecContext&) const override {
    const auto inputs = fx::unwrap(data_in);
    const auto it = inputs.find("in");
    if (it == inputs.end()) {
      throw std::invalid_argument("fx.stamp: missing input 'in'");
    }
    if (!params.is_object() || !params.contains("tag") ||
        !params["tag"].is_string()) {
      throw std::invalid_argument("fx.stamp: missing or invalid param 'tag'");
    }
    nlohmann::json out = nlohmann::json::object();
    out["p"] = it->second;
    out["t"] = params["tag"];
    return fx::wrap(signature_, {{"out", std::move(out)}});
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_stamp() {
  return std::make_shared<StampOp>();
}

}  // namespace fixture
