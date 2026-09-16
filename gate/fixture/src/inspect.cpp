#include "cgraph/ops.hpp"

#include <cstdint>
#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace fixture {
namespace {

std::string json_type_name(const nlohmann::json& v) {
  if (v.is_number()) {
    return "number";
  }
  if (v.is_string()) {
    return "string";
  }
  if (v.is_boolean()) {
    return "boolean";
  }
  if (v.is_null()) {
    return "null";
  }
  if (v.is_array()) {
    return "array";
  }
  if (v.is_object()) {
    return "object";
  }
  return "object";
}

nlohmann::json build_summary(const nlohmann::json& value, const std::string& label,
                             std::size_t max_chars) {
  const std::string dump = value.dump();
  const std::size_t char_len = dump.size();
  bool truncated = false;
  std::string preview = dump;
  if (max_chars < 1) {
    max_chars = 4096;
  }
  if (char_len > max_chars) {
    preview = dump.substr(0, max_chars);
    preview.append("…");
    truncated = true;
  }

  nlohmann::json meta = nlohmann::json::object();
  if (value.is_array()) {
    meta["length"] = value.size();
  } else if (value.is_object()) {
    nlohmann::json keys = nlohmann::json::array();
    std::size_t n = 0;
    for (auto it = value.begin(); it != value.end() && n < 16; ++it, ++n) {
      keys.push_back(it.key());
    }
    meta["keys"] = std::move(keys);
  }
  if (truncated || (value.is_array() && value.size() > 512) ||
      (value.is_object() && value.size() > 512)) {
    meta["approx_chars"] = char_len;
  }

  return nlohmann::json{
      {"op", "fx.inspect"},
      {"label", label},
      {"dtype", "json"},
      {"json_type", json_type_name(value)},
      {"preview_text", preview},
      {"truncated", truncated},
      {"char_len", char_len},
      {"meta", std::move(meta)},
  };
}

class InspectOp final : public cgraph::MemoryOperator {
 public:
  InspectOp() {
    op_id_ = "fx.inspect";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, "json");
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, "json");
    signature_.outputs["summary"] =
        cgraph::make_port("summary", cgraph::PortKind::Value, "json");

    cgraph::ParamSpec label;
    label.name = "label";
    label.dtype = "string";
    label.default_value = "inspect";
    label.doc = "Prefix for Console / 产物";
    label.bindable = false;
    signature_.params["label"] = std::move(label);

    cgraph::ParamSpec max_chars;
    max_chars.name = "max_chars";
    max_chars.dtype = "int";
    max_chars.default_value = 4096;
    max_chars.doc = "Max preview_text characters before truncate";
    signature_.params["max_chars"] = std::move(max_chars);

    cgraph::ParamSpec also_stdout;
    also_stdout.name = "also_stdout";
    also_stdout.dtype = "bool";
    also_stdout.default_value = false;
    also_stdout.doc = "Also write one line to process stdout";
    also_stdout.bindable = false;
    signature_.params["also_stdout"] = std::move(also_stdout);

    capability_.summary = "Inspect JSON input; passthrough + summary";
    capability_.tags = {"inspect", "debug", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire json into in; read out (passthrough) or summary.";
    usage_.tune = "params.label, max_chars, also_stdout.";
    usage_.inspect = "summary is structured preview; out equals in.";
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs,
      const nlohmann::json& params, const cgraph::ExecContext&) const override {
    const auto it = inputs.find("in");
    if (it == inputs.end()) {
      throw std::invalid_argument("fx.inspect: missing input 'in'");
    }

    std::string label = "inspect";
    std::size_t max_chars = 4096;
    bool also_stdout = false;
    if (params.is_object()) {
      if (params.contains("label") && params["label"].is_string()) {
        label = params["label"].get<std::string>();
      }
      if (params.contains("max_chars") && params["max_chars"].is_number_integer()) {
        const auto v = params["max_chars"].get<std::int64_t>();
        if (v >= 1) {
          max_chars = static_cast<std::size_t>(v);
        }
      }
      if (params.contains("also_stdout") && params["also_stdout"].is_boolean()) {
        also_stdout = params["also_stdout"].get<bool>();
      }
    }

    const nlohmann::json summary = build_summary(it->second, label, max_chars);
    if (also_stdout) {
      std::cout << "[" << label << "] "
                << summary["preview_text"].get<std::string>() << std::endl;
    }
    return {{"out", it->second}, {"summary", summary}};
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_inspect() {
  return std::make_shared<InspectOp>();
}

}  // namespace fixture
