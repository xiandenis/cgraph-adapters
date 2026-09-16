#include "cgraph/ops.hpp"

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
        cgraph::make_port("in", cgraph::PortKind::Value, "json");
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, "json");
    capability_.summary = "Echo JSON; fingerprint reads params.probe file";
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire json into in; set params.probe to a token file.";
    usage_.tune = "File bytes enter fingerprint(), not params_digest.";
    usage_.inspect = "out is identical to in.";
  }

  std::optional<std::string> fingerprint(
      const std::map<std::string, nlohmann::json>&,
      const nlohmann::json& params) const override {
    const std::string token = read_probe_file(params);
    if (token.empty()) {
      return std::nullopt;
    }
    return token;
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs,
      const nlohmann::json&, const cgraph::ExecContext&) const override {
    const auto it = inputs.find("in");
    if (it == inputs.end()) {
      throw std::invalid_argument("fx.probe: missing input 'in'");
    }
    return {{"out", it->second}};
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_probe() {
  return std::make_shared<ProbeOp>();
}

}  // namespace fixture
