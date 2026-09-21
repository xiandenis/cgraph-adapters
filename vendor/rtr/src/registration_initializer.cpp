#include "rtr/cloud_file_util.hpp"
#include "rtr/ops.hpp"

#include <registration_initializer/registration_initializer.hpp>

#include "cgraph/ops.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace rtr {
namespace {

bool param_bool(const nlohmann::json& params, const char* key, bool def) {
  if (params.is_object() && params.contains(key) && params[key].is_boolean()) {
    return params[key].get<bool>();
  }
  return def;
}

int param_int(const nlohmann::json& params, const char* key, int def) {
  if (params.is_object() && params.contains(key) && params[key].is_number_integer()) {
    return params[key].get<int>();
  }
  return def;
}

class RegistrationInitializerOp final : public cgraph::MemoryOperator {
 public:
  RegistrationInitializerOp() {
    op_id_ = "rtr.registration_initializer";
    signature_.inputs["cloud"] = cloud_port("cloud");
    signature_.outputs["cloud"] = cloud_port("cloud");
    signature_.outputs["voxel_size"] = scalar_float_port("voxel_size");
    signature_.outputs["point_number"] = scalar_int_port("point_number");
    cgraph::ParamSpec use_root;
    use_root.name = "use_root";
    use_root.dtype = "bool";
    use_root.default_value = true;
    use_root.doc = "Angle-depth Root filter; writes Root_p.pcd when true";
    signature_.params["use_root"] = use_root;
    cgraph::ParamSpec res;
    res.name = "resolution";
    res.dtype = "int";
    res.default_value = 10;
    res.doc = "Sub-voxel resolution hint for initalState";
    signature_.params["resolution"] = res;
    capability_.summary =
        "Station preprocess (RTR RegistrationInitializer); emits file Artifact";
    cost_.cost_class = "cpu.heavy";
    effect_.effect = cgraph::EffectClass::External;
    effect_.cache = cgraph::CachePolicy::Volatile;
    usage_.principle =
        "对输入点云文件做测站初始化（角度深度 Root 筛选、体素/法向旁路写出）。"
        "中间文件与输出点云都落在图工作区，不把 LidarFrame 抬进图。";
    usage_.notes = {"中间 PCD 在 ExecContext.workdir", "不链接 RealTimeManager"};
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& inputs, const nlohmann::json& params,
      const cgraph::ExecContext& ctx) const override {
    if (!inputs.count("cloud")) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.registration_initializer: missing cloud");
    }
    require_file_point_cloud(inputs.at("cloud"), "rtr.registration_initializer");
    const auto src = artifact_file_path(inputs.at("cloud"));
    if (!std::filesystem::is_regular_file(src)) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.registration_initializer: cloud file missing");
    }
    if (ctx.workdir.empty()) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.registration_initializer: workdir required");
    }

    const bool use_root = param_bool(params, "use_root", true);
    const int resolution = param_int(params, "resolution", 10);
    std::filesystem::create_directories(ctx.workdir);

    Ddx::LidarFrame frame;
    frame.name_ = src.stem().string();
    frame.lasFn_ = src.string();
    frame.info_folder_ = ctx.workdir.string();

    Ddx::RegistrationInitializer init;
    init.setUseRoot(use_root);
    if (!init.initalState(frame, resolution, true)) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.registration_initializer: initalState failed");
    }
    if (!init.initializd(frame, true)) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.registration_initializer: initializd failed");
    }
    if (!init.check(frame)) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.registration_initializer: check failed");
    }

    const std::filesystem::path produced = frame.root_.cloud_path_.empty()
                                               ? src
                                               : std::filesystem::path(frame.root_.cloud_path_);
    if (!std::filesystem::is_regular_file(produced)) {
      throw cgraph::OperatorError(
          cgraph::ErrorCode::OpFailed,
          "rtr.registration_initializer: output cloud path missing");
    }
    const auto canonical = workspace_pcd_path(ctx, "cloud");
    std::error_code ec;
    const bool same =
        std::filesystem::exists(canonical) && std::filesystem::equivalent(produced, canonical, ec);
    if (!same) {
      std::filesystem::create_directories(canonical.parent_path());
      std::filesystem::copy_file(produced, canonical,
                                 std::filesystem::copy_options::overwrite_existing);
    }

    return {
        {"cloud", cloud_artifact_from_path(canonical)},
        {"voxel_size",
         cgraph::make_data_object(cgraph::type_ids::floating(),
                                  cgraph::SemanticSpec::of("rtr.semantic.voxel_size"),
                                  cgraph::Payload::floating(frame.voxel_size_))},
        {"point_number",
         cgraph::make_data_object(
             cgraph::type_ids::integer(),
             cgraph::SemanticSpec::of("rtr.semantic.point_count"),
             cgraph::Payload::integer(static_cast<std::int64_t>(frame.point_number_)))},
    };
  }
};

}  // namespace

void register_registration_initializer(cgraph::OpRegistry& registry) {
  registry.add(std::make_shared<RegistrationInitializerOp>());
}

}  // namespace rtr
