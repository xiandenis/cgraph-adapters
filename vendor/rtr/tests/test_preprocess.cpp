#include "check.hpp"
#include "pcd_io.hpp"
#include "rtr/ops.hpp"

#include "cgraph/ops.hpp"

#include <cmath>
#include <filesystem>
#include <vector>

static cgraph::DataObject path_obj(const std::filesystem::path& p) {
  return cgraph::make_data_object(cgraph::type_ids::string(),
                                  cgraph::SemanticSpec::of("cgraph.semantic.text"),
                                  cgraph::Payload::string(p.string()));
}

static std::vector<Eigen::Vector3f> make_sphere(int n) {
  std::vector<Eigen::Vector3f> pts;
  pts.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    const float theta = 6.2831853f * static_cast<float>(i % 100) / 100.f;
    const float phi = 3.1415926f * static_cast<float>(i % 50) / 50.f * 0.98f + 0.01f;
    const float r = 10.f + static_cast<float>(i % 7) * 0.01f;
    pts.emplace_back(r * std::sin(phi) * std::cos(theta),
                     r * std::sin(phi) * std::sin(theta), r * std::cos(phi));
  }
  return pts;
}

int main() {
  cgraph::OpRegistry reg;
  rtr::register_rtr_ops(reg);
  for (const char* id : {"rtr.angle_downsample", "rtr.voxel_medoid_downsample",
                         "rtr.resolution_estimate"}) {
    const auto* op = reg.get(id);
    RTR_CHECK(op != nullptr);
    RTR_CHECK(!op->usage().principle.empty());
  }

  const auto tmp = std::filesystem::temp_directory_path() / "cgraph_rtr_preprocess";
  const auto cloud_p = tmp / "sphere.pcd";
  write_ascii_xyz_pcd(cloud_p, make_sphere(4000));

  cgraph::ExecContext ctx;
  const auto* load = reg.get("rtr.point_cloud.load");
  const auto loaded =
      load->execute({{"path", path_obj(cloud_p)}}, nlohmann::json::object(), ctx);
  const std::int64_t n0 = 4000;

  const auto* angle = reg.get("rtr.angle_downsample");
  const auto angle_out = angle->execute(
      {{"cloud", loaded.at("cloud")}},
      nlohmann::json{{"width", 256},
                     {"height", 128},
                     {"work_dir", (tmp / "angle").string()}},
      ctx);
  RTR_CHECK(angle_out.at("point_number").payload.as_int() > 0);
  RTR_CHECK(angle_out.at("point_number").payload.as_int() < n0);

  const auto* medoid = reg.get("rtr.voxel_medoid_downsample");
  const auto medoid_out = medoid->execute(
      {{"cloud", loaded.at("cloud")}},
      nlohmann::json{{"voxel_size", 0.5},
                     {"multi_thread", false},
                     {"work_dir", (tmp / "medoid").string()}},
      ctx);
  RTR_CHECK(medoid_out.at("point_number").payload.as_int() > 0);
  RTR_CHECK(medoid_out.at("point_number").payload.as_int() < n0);

  const auto* res = reg.get("rtr.resolution_estimate");
  const auto res_out =
      res->execute({{"cloud", loaded.at("cloud")}}, nlohmann::json::object(), ctx);
  RTR_CHECK(res_out.at("voxel_size").payload.as_float() > 0.0);

  bool threw = false;
  try {
    medoid->execute({{"cloud", loaded.at("cloud")}},
                    nlohmann::json{{"voxel_size", -1.0}}, ctx);
  } catch (const std::exception&) {
    threw = true;
  }
  RTR_CHECK(threw);
  return 0;
}
