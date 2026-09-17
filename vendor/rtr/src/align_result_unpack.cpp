#include "rtr/ops.hpp"

#include "cgraph/ops.hpp"

#include <memory>
#include <string>

namespace rtr {
namespace {

class AlignResultUnpackOp final : public cgraph::MemoryOperator {
 public:
  AlignResultUnpackOp() {
    op_id_ = "rtr.align_result.unpack";
    signature_.inputs["align"] =
        cgraph::make_port("align", cgraph::PortKind::Value, "json");
    signature_.outputs["matrix"] =
        cgraph::make_port("matrix", cgraph::PortKind::Value, "json");
    signature_.outputs["src_name"] =
        cgraph::make_port("src_name", cgraph::PortKind::Value, "string");
    signature_.outputs["tgt_name"] =
        cgraph::make_port("tgt_name", cgraph::PortKind::Value, "string");
    signature_.outputs["rms"] =
        cgraph::make_port("rms", cgraph::PortKind::Value, "float");
    signature_.outputs["final_rms"] =
        cgraph::make_port("final_rms", cgraph::PortKind::Value, "float");
    signature_.outputs["similarity"] =
        cgraph::make_port("similarity", cgraph::PortKind::Value, "float");
    signature_.outputs["weight"] =
        cgraph::make_port("weight", cgraph::PortKind::Value, "float");
    signature_.outputs["feature_num"] =
        cgraph::make_port("feature_num", cgraph::PortKind::Value, "int");
    signature_.outputs["state"] =
        cgraph::make_port("state", cgraph::PortKind::Value, "int");
    signature_.outputs["auto_reg"] =
        cgraph::make_port("auto_reg", cgraph::PortKind::Value, "int");
    signature_.outputs["information"] =
        cgraph::make_port("information", cgraph::PortKind::Value, "json");
    capability_.summary = "Split AlignResult JSON into field ports";
    cost_.cost_class = "cpu.tiny";
    effect_.effect = cgraph::EffectClass::Pure;
    effect_.cache = cgraph::CachePolicy::Memoizable;
    usage_.principle =
        "将 AlignResult JSON 按字段拆成独立 Value 口，数值不变。下游只连需要的口。";
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs, const nlohmann::json&,
      const cgraph::ExecContext& ctx) const override {
    const auto it = inputs.find("align");
    if (it == inputs.end() || !it->second.is_object()) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.align_result.unpack: align must be an object");
    }
    const nlohmann::json& align = it->second;
    auto want = [&](const char* port) {
      return ctx.requested_outputs.empty() ||
             ctx.requested_outputs.count(port) != 0;
    };
    auto require = [&](const char* key) -> const nlohmann::json& {
      if (!align.contains(key)) {
        throw cgraph::OperatorError(
            cgraph::ErrorCode::OpFailed,
            std::string("rtr.align_result.unpack: missing field ") + key);
      }
      return align.at(key);
    };
    std::map<std::string, nlohmann::json> out;
    if (want("matrix")) {
      out.emplace("matrix", require("matrix"));
    }
    if (want("src_name")) {
      out.emplace("src_name", require("src_name"));
    }
    if (want("tgt_name")) {
      out.emplace("tgt_name", require("tgt_name"));
    }
    if (want("rms")) {
      out.emplace("rms", require("rms"));
    }
    if (want("final_rms")) {
      out.emplace("final_rms", require("final_rms"));
    }
    if (want("similarity")) {
      out.emplace("similarity", require("similarity"));
    }
    if (want("weight")) {
      out.emplace("weight", require("weight"));
    }
    if (want("feature_num")) {
      out.emplace("feature_num", require("feature_num"));
    }
    if (want("state")) {
      out.emplace("state", require("state"));
    }
    if (want("auto_reg")) {
      out.emplace("auto_reg", require("auto_reg"));
    }
    if (want("information")) {
      out.emplace("information", require("information"));
    }
    return out;
  }
};

}  // namespace

void register_unpack(cgraph::OpRegistry& registry) {
  registry.add(std::make_shared<AlignResultUnpackOp>());
}

}  // namespace rtr
