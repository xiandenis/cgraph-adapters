#include "rtr/ops.hpp"

#include "cgraph/data_helpers.hpp"
#include "cgraph/ops.hpp"

#include <memory>
#include <string>
#include <vector>

namespace rtr {
namespace {

class AlignResultUnpackOp final : public cgraph::MemoryOperator {
 public:
  AlignResultUnpackOp() {
    op_id_ = "rtr.align_result.unpack";
    signature_.inputs["align"] = align_port("align");
    signature_.outputs["matrix"] = matrix_port("matrix");
    signature_.outputs["src_name"] = scalar_string_port("src_name");
    signature_.outputs["tgt_name"] = scalar_string_port("tgt_name");
    signature_.outputs["rms"] = scalar_float_port("rms");
    signature_.outputs["final_rms"] = scalar_float_port("final_rms");
    signature_.outputs["similarity"] = scalar_float_port("similarity");
    signature_.outputs["weight"] = scalar_float_port("weight");
    signature_.outputs["feature_num"] = scalar_int_port("feature_num");
    signature_.outputs["state"] = scalar_int_port("state");
    signature_.outputs["auto_reg"] = scalar_int_port("auto_reg");
    signature_.outputs["information"] = information_port("information");
    capability_.summary = "Split AlignResult into typed field ports";
    cost_.cost_class = "cpu.tiny";
    effect_.effect = cgraph::EffectClass::Pure;
    effect_.cache = cgraph::CachePolicy::Memoizable;
    usage_.principle =
        "将 AlignResult 记录按字段拆成独立 Value 口。matrix 为源→目标刚体变换。";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& inputs, const nlohmann::json&,
      const cgraph::ExecContext& ctx) const override {
    const auto it = inputs.find("align");
    if (it == inputs.end()) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.align_result.unpack: missing align");
    }
    const Ddx::AlignResult align = align_result_from_data(it->second);
    auto want = [&](const char* port) {
      return ctx.requested_outputs.empty() ||
             ctx.requested_outputs.count(port) != 0;
    };
    std::map<std::string, cgraph::DataObject> out;
    if (want("matrix")) {
      out.emplace("matrix",
                  cgraph::make_data_object(cgraph::type_ids::matrix_r4c4_f64(),
                                           rigid_transform_semantic(),
                                           matrix_to_payload(align.matrix_)));
    }
    if (want("src_name")) {
      out.emplace("src_name", cgraph::make_string(align.src_name_,
                                                  cgraph::SemanticSpec::of("src_name")));
    }
    if (want("tgt_name")) {
      out.emplace("tgt_name", cgraph::make_string(align.tgt_name_,
                                                  cgraph::SemanticSpec::of("tgt_name")));
    }
    if (want("rms")) {
      out.emplace("rms",
                  cgraph::make_float(align.rms_, cgraph::SemanticSpec::of("rms")));
    }
    if (want("final_rms")) {
      out.emplace("final_rms", cgraph::make_float(align.final_rms_,
                                                  cgraph::SemanticSpec::of("final_rms")));
    }
    if (want("similarity")) {
      out.emplace("similarity",
                  cgraph::make_float(align.similarity_,
                                     cgraph::SemanticSpec::of("similarity")));
    }
    if (want("weight")) {
      out.emplace("weight", cgraph::make_float(align.weight_,
                                               cgraph::SemanticSpec::of("weight")));
    }
    if (want("feature_num")) {
      out.emplace("feature_num",
                  cgraph::make_int(align.feature_num_,
                                   cgraph::SemanticSpec::of("feature_num")));
    }
    if (want("state")) {
      out.emplace("state",
                  cgraph::make_int(align.state_, cgraph::SemanticSpec::of("state")));
    }
    if (want("auto_reg")) {
      out.emplace("auto_reg", cgraph::make_int(align.auto_reg_,
                                               cgraph::SemanticSpec::of("auto_reg")));
    }
    if (want("information")) {
      const std::string info = matrix6d_to_json(align.information_).dump();
      out.emplace("information",
                  cgraph::make_data_object(
                      cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("information"),
                      cgraph::Payload::untyped(std::vector<std::uint8_t>(
                          info.begin(), info.end()))));
    }
    return out;
  }
};

}  // namespace

void register_unpack(cgraph::OpRegistry& registry) {
  registry.add(std::make_shared<AlignResultUnpackOp>());
}

}  // namespace rtr
