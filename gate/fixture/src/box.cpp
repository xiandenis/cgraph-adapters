#include "cgraph/data_helpers.hpp"
#include "cgraph/ops.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fixture {
namespace {

class BoxOp final : public cgraph::MemoryOperator {
 public:
  BoxOp() {
    op_id_ = "fx.box";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, cgraph::type_ids::untyped(),
                          cgraph::SemanticSpec::of("document"));
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, cgraph::type_ids::untyped(),
                          cgraph::SemanticSpec::of("document"));
    cgraph::ParamSpec tag;
    tag.name = "tag";
    tag.dtype = "string";
    tag.invalidate = true;
    tag.bindable = false;
    signature_.params["tag"] = std::move(tag);
    capability_.summary = "Black-box stamp (untyped document)";
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire an untyped document into in; read the tagged document from out.";
    usage_.tune = "params.tag (invalidate).";
    usage_.inspect = "out is an untyped document wrapping in with tag.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& inputs,
      const nlohmann::json& params, const cgraph::ExecContext&) const override {
    const auto it = inputs.find("in");
    if (it == inputs.end()) {
      throw std::invalid_argument("fx.box: missing input 'in'");
    }
    if (!params.is_object() || !params.contains("tag") || !params["tag"].is_string()) {
      throw std::invalid_argument("fx.box: missing or invalid param 'tag'");
    }
    const std::string tag = params["tag"].get<std::string>();
    std::string inner;
    if (it->second.payload.kind() == cgraph::Payload::Kind::Untyped) {
      const auto& b = it->second.payload.as_untyped();
      inner.assign(b.begin(), b.end());
    } else if (it->second.payload.kind() == cgraph::Payload::Kind::String) {
      inner = it->second.payload.as_string();
    } else {
      inner = it->second.content_fingerprint;
    }
    const std::string packed = tag + "\n" + inner;
    return {{"out", cgraph::make_data_object(
                        cgraph::type_ids::untyped(), cgraph::SemanticSpec::of("document"),
                        cgraph::Payload::untyped(std::vector<std::uint8_t>(
                            packed.begin(), packed.end())))}};
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_box() {
  return std::make_shared<BoxOp>();
}

}  // namespace fixture
