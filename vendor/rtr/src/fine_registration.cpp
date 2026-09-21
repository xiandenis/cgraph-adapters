#include "rtr/cloud_file_util.hpp"
#include "rtr/ops.hpp"

#include <fine_registration/fine_registration.hpp>

#include "cgraph/ops.hpp"

#include <filesystem>
#include <memory>
#include <string>

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

class FineRegistrationOp final : public cgraph::MemoryOperator {
 public:
  FineRegistrationOp() {
    op_id_ = "rtr.fine_registration";
    signature_.inputs["src"] = cloud_port("src");
    signature_.inputs["tgt"] = cloud_port("tgt");
    signature_.inputs["guess"] = matrix_port("guess", true);
    signature_.outputs["align"] = align_port("align");
    auto add_f = [&](const char* name, double def, const char* doc) {
      cgraph::ParamSpec p;
      p.name = name;
      p.dtype = "float";
      p.default_value = def;
      p.doc = doc;
      signature_.params[name] = p;
    };
    add_f("max_correspondence_distance", 1.0, "SmallGICP max correspondence (m)");
    add_f("report_distance_thresh", 0.2, "Overlap distance thresh (m)");
    add_f("report_rms_inlier_thresh", 0.1, "Point-to-plane inlier thresh (m)");
    add_f("report_rms_downsample_voxel", 0.1, "RMS downsample voxel (m)");
    cgraph::ParamSpec sample;
    sample.name = "report_rms_sample_max";
    sample.dtype = "int";
    sample.default_value = 88888;
    signature_.params["report_rms_sample_max"] = sample;
    cgraph::ParamSpec mode;
    mode.name = "report_rms_direction_mode";
    mode.dtype = "string";
    mode.default_value = "forward_only";
    signature_.params["report_rms_direction_mode"] = mode;
    capability_.summary = "Fine registration (RTR FineRegistration, file path)";
    cost_.cost_class = "cpu.heavy";
    effect_.effect = cgraph::EffectClass::External;
    effect_.cache = cgraph::CachePolicy::Volatile;
    usage_.principle =
        "以给定 4×4 初值做精配准（当前库默认 SmallGICP 点到面迭代），并估计重叠与 "
        "RMS 报告。初值缺省为单位阵，因此可单独使用。输出相对矩阵为源→目标。";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& inputs, const nlohmann::json& params,
      const cgraph::ExecContext&) const override {
    if (!inputs.count("src") || !inputs.count("tgt")) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.fine_registration: missing src or tgt");
    }
    require_file_point_cloud(inputs.at("src"), "rtr.fine_registration");
    require_file_point_cloud(inputs.at("tgt"), "rtr.fine_registration");
    const auto src = artifact_file_path(inputs.at("src"));
    const auto tgt = artifact_file_path(inputs.at("tgt"));
    if (!std::filesystem::is_regular_file(src) ||
        !std::filesystem::is_regular_file(tgt)) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.fine_registration: src/tgt file missing");
    }
    Eigen::Matrix4d guess = Eigen::Matrix4d::Identity();
    if (inputs.count("guess")) {
      guess = matrix_from_payload(inputs.at("guess").payload);
    }
    Ddx::FineRegistration algo;
    algo.setMaxCorrespondenceDistance(
        param_float(params, "max_correspondence_distance", 1.0f));
    algo.setReportDistanceThresh(param_float(params, "report_distance_thresh", 0.2f));
    algo.setReportRmsInlierThresh(
        param_float(params, "report_rms_inlier_thresh", 0.1f));
    algo.setReportRmsDownsampleVoxel(
        param_float(params, "report_rms_downsample_voxel", 0.1f));
    algo.setReportRmsSampleMax(
        static_cast<std::size_t>(param_int(params, "report_rms_sample_max", 88888)));
    const std::string mode =
        (params.is_object() && params.contains("report_rms_direction_mode") &&
         params["report_rms_direction_mode"].is_string())
            ? params["report_rms_direction_mode"].get<std::string>()
            : std::string("forward_only");
    if (mode == "bidirectional_density") {
      algo.setReportRmsDirectionMode(
          Ddx::FineRegistration::ReportRmsDirectionMode::kBidirectionalDensityCompare);
    } else if (mode == "forward_only") {
      algo.setReportRmsDirectionMode(
          Ddx::FineRegistration::ReportRmsDirectionMode::kForwardOnly);
    } else {
      throw cgraph::OperatorError(
          cgraph::ErrorCode::OpFailed,
          "rtr.fine_registration: bad report_rms_direction_mode");
    }
    if (!algo.run(src.string(), tgt.string(), guess)) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.fine_registration: run() failed");
    }
    return {{"align", align_result_to_data(algo.getAlignResult())}};
  }
};

}  // namespace

void register_fine_registration(cgraph::OpRegistry& registry) {
  registry.add(std::make_shared<FineRegistrationOp>());
}

}  // namespace rtr
