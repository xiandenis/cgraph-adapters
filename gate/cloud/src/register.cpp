#include "cloud/ops.hpp"

#include "cgraph/artifact.hpp"
#include "cgraph/plugin_dtype.hpp"
#include "cloud/cloud_value.hpp"

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

namespace cloud {

std::shared_ptr<cgraph::MemoryOperator> make_cloud_load();
std::shared_ptr<cgraph::MemoryOperator> make_cloud_dump();
std::shared_ptr<cgraph::MemoryOperator> make_cloud_voxel();
std::shared_ptr<cgraph::MemoryOperator> make_cloud_normal_est();
std::shared_ptr<cgraph::MemoryOperator> make_cloud_fpfh();
std::shared_ptr<cgraph::MemoryOperator> make_cloud_fpfh_proc();
std::shared_ptr<cgraph::MemoryOperator> make_cloud_match();
std::shared_ptr<cgraph::MemoryOperator> make_cloud_ransac();
std::shared_ptr<cgraph::MemoryOperator> make_cloud_icp_step();
std::shared_ptr<cgraph::MemoryOperator> make_cloud_passthrough();
std::shared_ptr<cgraph::MemoryOperator> make_cloud_stat_outlier();
std::shared_ptr<cgraph::MemoryOperator> make_cloud_uniform_sampling();
std::shared_ptr<cgraph::MemoryOperator> make_cloud_euclid_cluster();
std::shared_ptr<cgraph::MemoryOperator> make_cloud_shot();

namespace {

nlohmann::json convert_file_format(const nlohmann::json& handle,
                                   const cgraph::ExecContext& ctx,
                                   const std::string& src_format,
                                   const std::string& dst_format) {
  if (ctx.workdir.empty()) {
    throw std::invalid_argument("cloud convert: workdir required");
  }
  const cgraph::Artifact src = cgraph::artifact_from_json(handle);
  nlohmann::json cloud;
  if (src_format == "ply") {
    cloud = load_ply_ascii(src.path);
  } else if (src_format == "pcd") {
    cloud = load_pcd_ascii(src.path);
  } else {
    throw std::invalid_argument("cloud convert: bad src_format");
  }
  const std::filesystem::path out = ctx.workdir / "outputs" / "out";
  std::filesystem::create_directories(out.parent_path());
  if (dst_format == "ply") {
    dump_ply_ascii(cloud, out);
  } else if (dst_format == "pcd") {
    dump_pcd_ascii(cloud, out);
  } else {
    throw std::invalid_argument("cloud convert: bad dst_format");
  }
  return cgraph::artifact_to_json(cgraph::make_file_artifact(out));
}

}  // namespace

void register_plugin() {
  cgraph::register_plugin_dtype("cloud");
  cgraph::register_plugin_dtype("rigid_transform");
  cgraph::register_plugin_dtype("correspondence_set");
}

void register_ops(cgraph::OpRegistry& registry, cgraph::ConvertRegistry& converts) {
  registry.add(make_cloud_load());
  registry.add(make_cloud_dump());
  registry.add(make_cloud_voxel());
  registry.add(make_cloud_normal_est());
  registry.add(make_cloud_fpfh());
  registry.add(make_cloud_fpfh_proc());
  registry.add(make_cloud_match());
  registry.add(make_cloud_ransac());
  registry.add(make_cloud_icp_step());
  registry.add(make_cloud_passthrough());
  registry.add(make_cloud_stat_outlier());
  registry.add(make_cloud_uniform_sampling());
  registry.add(make_cloud_euclid_cluster());
  registry.add(make_cloud_shot());

  static const char* k_p1_ops[] = {
      "cloud.voxel",       "cloud.normal_est",    "cloud.fpfh",
      "cloud.match",       "cloud.ransac",        "cloud.icp_step",
      "cloud.stat_outlier", "cloud.passthrough", "cloud.euclid_cluster",
  };
  for (const char* id : k_p1_ops) {
    if (cgraph::MemoryOperator* op = registry.get_mut(id)) {
      op->load_cost_description("cloud");
    }
  }

  cgraph::ConvertSpec ply_to_pcd;
  ply_to_pcd.src_dtype = "file";
  ply_to_pcd.dst_dtype = "file";
  ply_to_pcd.src_format = "ply";
  ply_to_pcd.dst_format = "pcd";
  converts.add(ply_to_pcd, [](const nlohmann::json& value, const cgraph::ExecContext& ctx) {
    return convert_file_format(value, ctx, "ply", "pcd");
  });

  cgraph::ConvertSpec pcd_to_ply;
  pcd_to_ply.src_dtype = "file";
  pcd_to_ply.dst_dtype = "file";
  pcd_to_ply.src_format = "pcd";
  pcd_to_ply.dst_format = "ply";
  converts.add(pcd_to_ply, [](const nlohmann::json& value, const cgraph::ExecContext& ctx) {
    return convert_file_format(value, ctx, "pcd", "ply");
  });
}

}  // namespace cloud
