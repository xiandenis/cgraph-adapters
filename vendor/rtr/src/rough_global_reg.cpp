#include "rtr/json.hpp"
#include "rtr/ops.hpp"

#include <rough_global_reg/rough_global_reg.hpp>

#include "cgraph/ops.hpp"

#include <filesystem>
#include <memory>

namespace rtr {
namespace {

float param_float(const nlohmann::json& params, const char* key, float def) {
  if (params.is_object() && params.contains(key) && params[key].is_number()) {
    return params[key].get<float>();
  }
  return def;
}

int param_int(const nlohmann::json& params, const char* key, int def) {
  if (params.is_object() && params.contains(key) && params[key].is_number_integer()) {
    return params[key].get<int>();
  }
  return def;
}

bool param_bool(const nlohmann::json& params, const char* key, bool def) {
  if (params.is_object() && params.contains(key) && params[key].is_boolean()) {
    return params[key].get<bool>();
  }
  return def;
}

class RoughGlobalRegOp final : public cgraph::MemoryOperator {
 public:
  RoughGlobalRegOp() {
    op_id_ = "rtr.rough_global_reg";
    signature_.inputs["src"] =
        cgraph::make_port("src", cgraph::PortKind::Artifact, "file");
    signature_.inputs["tgt"] =
        cgraph::make_port("tgt", cgraph::PortKind::Artifact, "file");
    auto guess = cgraph::make_port("guess", cgraph::PortKind::Value, "json");
    guess.optional = true;
    signature_.inputs["guess"] = std::move(guess);
    signature_.outputs["align"] =
        cgraph::make_port("align", cgraph::PortKind::Value, "json");
    cgraph::ParamSpec res;
    res.name = "resolution";
    res.dtype = "float";
    res.default_value = 0.15;
    res.doc = "Voxel / feature scale in meters";
    signature_.params["resolution"] = res;
    cgraph::ParamSpec it;
    it.name = "iter_number";
    it.dtype = "int";
    it.default_value = -1;
    signature_.params["iter_number"] = it;
    cgraph::ParamSpec compass;
    compass.name = "use_compass";
    compass.dtype = "bool";
    compass.default_value = false;
    signature_.params["use_compass"] = compass;
    capability_.summary =
        "Coarse global registration (RTR RoughGlobalReg, file path)";
    cost_.cost_class = "cpu.heavy";
    effect_.effect = cgraph::EffectClass::External;
    effect_.cache = cgraph::CachePolicy::Volatile;
    usage_.principle =
        "在特征空间求源点云到目标点云的刚体变换（粗全局配准）。无初值时全局搜索；有 "
        "4×4 初值时在其邻域求解。resolution 控制体素与特征尺度。输出相对矩阵为源→目标。";
    usage_.notes = {"run()==false 视为失败，不返回单位阵",
                    "点云须为 RTR 可读文件（如 PCD）"};
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs, const nlohmann::json& params,
      const cgraph::ExecContext&) const override {
    if (!inputs.count("src") || !inputs.count("tgt")) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.rough_global_reg: missing src or tgt");
    }
    const auto src = artifact_file_path(inputs.at("src"));
    const auto tgt = artifact_file_path(inputs.at("tgt"));
    if (!std::filesystem::is_regular_file(src) ||
        !std::filesystem::is_regular_file(tgt)) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.rough_global_reg: src/tgt file missing");
    }
    const float resolution = param_float(params, "resolution", 0.15f);
    const int iter_number = param_int(params, "iter_number", -1);
    const bool use_compass = param_bool(params, "use_compass", false);
    Ddx::RoughGlobalReg algo;
    bool ok = false;
    if (inputs.count("guess")) {
      const Eigen::Matrix4d guess = matrix4d_from_json(inputs.at("guess"));
      ok = algo.run(src.string(), tgt.string(), resolution, guess, iter_number,
                    use_compass);
    } else {
      ok = algo.run(src.string(), tgt.string(), resolution, iter_number);
    }
    if (!ok) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.rough_global_reg: run() failed");
    }
    return {{"align", align_result_to_json(algo.getAlignResult())}};
  }
};

}  // namespace

void register_rough_global_reg(cgraph::OpRegistry& registry) {
  registry.add(std::make_shared<RoughGlobalRegOp>());
}

}  // namespace rtr
