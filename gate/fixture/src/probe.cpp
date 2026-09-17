#include "cgraph/ops.hpp"
#include "fx_data.hpp"

#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace fixture {
namespace {

std::string read_probe_file(const nlohmann::json& params) {
  if (!params.is_object() || !params.contains("probe") || !params["probe"].is_string()) {
    return {};
  }
  const std::string path = params["probe"].get<std::string>();
  std::ifstream in(path);
  if (!in) {
    return {};
  }
  std::ostringstream buf;
  buf << in.rdbuf();
  return buf.str();
}

class ProbeOp final : public cgraph::MemoryOperator {
 public:
  ProbeOp() {
    op_id_ = "fx.probe";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("number"));
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("number"));
    capability_.summary = "Echo JSON; fingerprint reads params.probe file";
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire json into in; set params.probe to a token file.";
    usage_.tune = "File bytes enter fingerprint(), not params_digest.";
    usage_.inspect = "out is identical to in.";
  }

  std::optional<std::string> fingerprint(
      const std::map<std::string, cgraph::DataObject>&,
      const nlohmann::json& params) const override {
    const std::string token = read_probe_file(params);
    if (token.empty()) {
      return std::nullopt;
    }
    return token;
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in,
      const nlohmann::json&, const cgraph::ExecContext&) const override {
    const auto inputs = fx::unwrap(data_in);
    const auto it = inputs.find("in");
    if (it == inputs.end()) {
      throw std::invalid_argument("fx.probe: missing input 'in'");
    }
    return fx::wrap(signature_, {{"out", it->second}});
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_probe() {
  return std::make_shared<ProbeOp>();
}

}  // namespace fixture
