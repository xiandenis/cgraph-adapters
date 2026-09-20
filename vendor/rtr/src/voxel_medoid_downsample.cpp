#include "rtr/cloud_file_util.hpp"
#include "rtr/ops.hpp"

#include <octree/octree_downsample.hpp>

#include <cstdint>
#include <memory>

namespace rtr {
namespace {

class VoxelMedoidDownsampleOp final : public cgraph::MemoryOperator {
 public:
  VoxelMedoidDownsampleOp() {
    op_id_ = "rtr.voxel_medoid_downsample";
    signature_.inputs["cloud"] = cloud_port("cloud");
    signature_.outputs["cloud"] = cloud_port("cloud");
    signature_.outputs["point_number"] = scalar_int_port("point_number");
    cgraph::ParamSpec vs;
    vs.name = "voxel_size";
    vs.dtype = "float";
    vs.default_value = 0.05;
    vs.doc = "Voxel edge length in meters";
    signature_.params["voxel_size"] = vs;
    cgraph::ParamSpec mt;
    mt.name = "multi_thread";
    mt.dtype = "bool";
    mt.default_value = true;
    signature_.params["multi_thread"] = mt;
    cgraph::ParamSpec work;
    work.name = "work_dir";
    work.dtype = "string";
    work.default_value = "";
    signature_.params["work_dir"] = work;
    capability_.summary =
        "Hash-voxel true Medoid downsample (RTR octree::downsampleMedoid)";
    cost_.cost_class = "cpu.medium";
    effect_.effect = cgraph::EffectClass::External;
    effect_.cache = cgraph::CachePolicy::Volatile;
    usage_.principle =
        "原点对齐体素格上取真 Medoid（距格心最近的原始点）做下采样。"
        "输入/输出为点云文件 Artifact；结果写旁路 PCD。";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& inputs, const nlohmann::json& params,
      const cgraph::ExecContext&) const override {
    if (!inputs.count("cloud")) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.voxel_medoid_downsample: missing cloud");
    }
    const auto src_path = artifact_file_path(inputs.at("cloud"));
    auto cloud = load_xyz_cloud(inputs.at("cloud"), "rtr.voxel_medoid_downsample");
    const float voxel_size = param_float(params, "voxel_size", 0.05f);
    const bool multi_thread = param_bool(params, "multi_thread", true);
    auto out = Ddx::octree::downsampleMedoid<pcl::PointXYZ>(cloud, voxel_size, multi_thread);
    if (!out || out->empty()) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.voxel_medoid_downsample: downsample failed");
    }
    const auto work = resolve_work_dir(params, src_path, ".cgraph_rtr_medoid_");
    const auto out_path = work / (src_path.stem().string() + "_medoid_ds.pcd");
    return {{"cloud", save_xyz_cloud_artifact(*out, out_path, "rtr.voxel_medoid_downsample")},
            {"point_number",
             point_number_object(static_cast<std::int64_t>(out->size()))}};
  }
};

}  // namespace

void register_voxel_medoid_downsample(cgraph::OpRegistry& registry) {
  registry.add(std::make_shared<VoxelMedoidDownsampleOp>());
}

}  // namespace rtr
