#include "rtr/cloud_file_util.hpp"
#include "rtr/ops.hpp"

#include <registration_initializer/angle_downsample.hpp>

#include <cstdint>
#include <memory>

namespace rtr {
namespace {

class AngleDownsampleOp final : public cgraph::MemoryOperator {
 public:
  AngleDownsampleOp() {
    op_id_ = "rtr.angle_downsample";
    signature_.inputs["cloud"] = cloud_port("cloud");
    signature_.outputs["cloud"] = cloud_port("cloud");
    signature_.outputs["point_number"] = scalar_int_port("point_number");
    cgraph::ParamSpec w;
    w.name = "width";
    w.dtype = "int";
    w.default_value = 5000;
    w.doc = "Panorama width (angular bins)";
    signature_.params["width"] = w;
    cgraph::ParamSpec h;
    h.name = "height";
    h.dtype = "int";
    h.default_value = 2500;
    h.doc = "Panorama height (angular bins)";
    signature_.params["height"] = h;
    cgraph::ParamSpec work;
    work.name = "work_dir";
    work.dtype = "string";
    work.default_value = "";
    work.doc = "Directory for output PCD; default beside input";
    signature_.params["work_dir"] = work;
    capability_.summary = "Angle-depth downsample (RTR AngleDownsample); file Artifact I/O";
    cost_.cost_class = "cpu.medium";
    effect_.effect = cgraph::EffectClass::External;
    effect_.cache = cgraph::CachePolicy::Volatile;
    usage_.principle =
        "按测站球面角度栅格做深度挑选下采样（角度深度下采样），保留每角格最近点。"
        "输入/输出均为点云文件 Artifact，结果写旁路 PCD。";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& inputs, const nlohmann::json& params,
      const cgraph::ExecContext&) const override {
    if (!inputs.count("cloud")) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.angle_downsample: missing cloud");
    }
    const auto src_path = artifact_file_path(inputs.at("cloud"));
    auto cloud = load_xyz_cloud(inputs.at("cloud"), "rtr.angle_downsample");
    Ddx::pcl_type::PointCloudPtr out;
    const int width = param_int(params, "width", 5000);
    const int height = param_int(params, "height", 2500);
    if (!Ddx::AngleDownsample::downsampling(cloud, out, width, height) || !out ||
        out->empty()) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.angle_downsample: downsampling failed");
    }
    const auto work = resolve_work_dir(params, src_path, ".cgraph_rtr_angle_");
    const auto out_path = work / (src_path.stem().string() + "_angle_ds.pcd");
    return {{"cloud", save_xyz_cloud_artifact(*out, out_path, "rtr.angle_downsample")},
            {"point_number", point_number_object(static_cast<std::int64_t>(out->size()))}};
  }
};

}  // namespace

void register_angle_downsample(cgraph::OpRegistry& registry) {
  registry.add(std::make_shared<AngleDownsampleOp>());
}

}  // namespace rtr
