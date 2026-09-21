#include "rtr/cloud_file_util.hpp"
#include "rtr/ops.hpp"

#include <point_cloud_io/point_cloud_io.hpp>
#include <registration_report/registration_report.hpp>

#include "cgraph/ops.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

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

class RegistrationReportOp final : public cgraph::MemoryOperator {
 public:
  RegistrationReportOp() {
    op_id_ = "rtr.registration_report";
    signature_.inputs["src"] = cloud_port("src");
    signature_.inputs["tgt"] = cloud_port("tgt");
    signature_.inputs["matrix"] = matrix_port("matrix");
    signature_.outputs["report"] = report_port("report");
    auto add_f = [&](const char* name, double def, const char* doc) {
      cgraph::ParamSpec p;
      p.name = name;
      p.dtype = "float";
      p.default_value = def;
      p.doc = doc;
      signature_.params[name] = p;
    };
    add_f("voxel_size", 0.5, "Overlap voxel size (m)");
    add_f("distance_thresh", 0.2, "Overlap distance thresh (m)");
    add_f("rms_downsample_voxel", 0.1, "RMS downsample voxel (m)");
    add_f("rms_inlier_thresh", 0.1, "Point-to-plane inlier thresh (m)");
    cgraph::ParamSpec knn;
    knn.name = "plane_knn";
    knn.dtype = "int";
    knn.default_value = 5;
    signature_.params["plane_knn"] = knn;
    cgraph::ParamSpec sample;
    sample.name = "rms_sample_max";
    sample.dtype = "int";
    sample.default_value = 88888;
    signature_.params["rms_sample_max"] = sample;
    cgraph::ParamSpec mode;
    mode.name = "rms_direction_mode";
    mode.dtype = "string";
    mode.default_value = "forward_only";
    signature_.params["rms_direction_mode"] = mode;
    capability_.summary =
        "Overlap + point-to-plane RMS report (RTR RegistrationReportEstimator)";
    cost_.cost_class = "cpu.heavy";
    effect_.effect = cgraph::EffectClass::External;
    effect_.cache = cgraph::CachePolicy::Volatile;
    usage_.principle =
        "在给定源→目标相对位姿下估计重叠率与点到平面 RMS。输入为文件 Artifact 点云与 "
        "4×4 矩阵；输出 typed registration_report，不写工程态。";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& inputs, const nlohmann::json& params,
      const cgraph::ExecContext&) const override {
    if (!inputs.count("src") || !inputs.count("tgt") || !inputs.count("matrix")) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.registration_report: missing src/tgt/matrix");
    }
    require_file_point_cloud(inputs.at("src"), "rtr.registration_report");
    require_file_point_cloud(inputs.at("tgt"), "rtr.registration_report");
    const auto src = artifact_file_path(inputs.at("src"));
    const auto tgt = artifact_file_path(inputs.at("tgt"));
    if (!std::filesystem::is_regular_file(src) ||
        !std::filesystem::is_regular_file(tgt)) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.registration_report: src/tgt file missing");
    }
    const Eigen::Matrix4d matrix = matrix_from_payload(inputs.at("matrix").payload);

    pcl::PointCloud<pcl::PointXYZ> source;
    pcl::PointCloud<pcl::PointXYZ> target;
    if (Ddx::point_cloud_io::load(src.string(), source) != 0 || source.empty()) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.registration_report: failed to load src");
    }
    if (Ddx::point_cloud_io::load(tgt.string(), target) != 0 || target.empty()) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.registration_report: failed to load tgt");
    }

    Ddx::RegistrationReportEstimator est(
        param_float(params, "voxel_size", 0.5f),
        param_float(params, "distance_thresh", 0.2f),
        param_float(params, "rms_downsample_voxel", 0.1f),
        param_int(params, "plane_knn", 5),
        param_float(params, "rms_inlier_thresh", 0.1f));
    est.setRmsSampleMax(
        static_cast<std::size_t>(param_int(params, "rms_sample_max", 88888)));
    const std::string mode =
        (params.is_object() && params.contains("rms_direction_mode") &&
         params["rms_direction_mode"].is_string())
            ? params["rms_direction_mode"].get<std::string>()
            : std::string("forward_only");
    if (mode == "bidirectional_density") {
      est.setRmsDirectionMode(
          Ddx::RegistrationReportEstimator::RmsDirectionMode::kBidirectionalDensityCompare);
    } else if (mode == "forward_only") {
      est.setRmsDirectionMode(
          Ddx::RegistrationReportEstimator::RmsDirectionMode::kForwardOnly);
    } else {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.registration_report: bad rms_direction_mode");
    }

    Ddx::RegistrationReportEstimator::Result result;
    if (!est.estimate(source, target, matrix, result)) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.registration_report: estimate() failed");
    }
    return {{"report", registration_report_to_data(result.overlap_ratio, result.rms,
                                                   result.overlap_count,
                                                   result.rms_sample_count,
                                                   result.rms_query_from_target)}};
  }
};

}  // namespace

void register_registration_report(cgraph::OpRegistry& registry) {
  registry.add(std::make_shared<RegistrationReportOp>());
}

}  // namespace rtr
