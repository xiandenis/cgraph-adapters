#include "cgraph/ops.hpp"
#include "fx_data.hpp"

#include <memory>
#include <stdexcept>

namespace fixture {
namespace {

nlohmann::json first_entry(const nlohmann::json& value) {
  nlohmann::json head = nlohmann::json::object();
  if (!value.is_object() || value.empty()) {
    return head;
  }
  const auto it = value.begin();
  head[it.key()] = it.value();
  return head;
}

bool wants(const cgraph::ExecContext& ctx, const char* port) {
  return ctx.requested_outputs.empty() ||
         ctx.requested_outputs.find(port) != ctx.requested_outputs.end();
}

class DupOp final : public cgraph::MemoryOperator {
 public:
  DupOp() {
    op_id_ = "fx.dup";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("number"));
    signature_.outputs["full"] =
        cgraph::make_port("full", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("number"));
    signature_.outputs["head"] =
        cgraph::make_port("head", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("number"));
    capability_.summary = "Split a JSON object into full copy and first entry";
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Connect in; take full and/or head. Unconnected head may be skipped.";
    usage_.tune = "No parameters.";
    usage_.inspect = "full=in; head is the lexicographically first key-value pair.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in,
      const nlohmann::json&, const cgraph::ExecContext& ctx) const override {
    const auto inputs = fx::unwrap(data_in);
    const auto it = inputs.find("in");
    if (it == inputs.end()) {
      throw std::invalid_argument("fx.dup: missing input 'in'");
    }
    std::map<std::string, nlohmann::json> out;
    if (wants(ctx, "full")) {
      out.emplace("full", it->second);
    }
    if (wants(ctx, "head")) {
      out.emplace("head", first_entry(it->second));
    }
    return fx::wrap(signature_, out);
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_dup() {
  return std::make_shared<DupOp>();
}

}  // namespace fixture
