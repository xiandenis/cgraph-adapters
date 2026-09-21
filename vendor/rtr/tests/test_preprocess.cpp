#include "check.hpp"
#include "pcd_io.hpp"
#include "rtr/cloud_file_util.hpp"
#include "rtr/ops.hpp"

#include "cgraph/edge_check.hpp"
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

  cgraph::ExecContext load_ctx;
  const auto* load = reg.get("rtr.point_cloud.load");
  const auto loaded =
      load->execute({{"path", path_obj(cloud_p)}}, nlohmann::json::object(), load_ctx);
  const std::int64_t n0 = 4000;

  const auto* angle = reg.get("rtr.angle_downsample");
  RTR_CHECK(angle->signature().params.count("work_dir") == 0);
  RTR_CHECK(angle->signature().inputs.at("cloud").accepts_realisation ==
            std::vector<std::string>{"file"});
  cgraph::ExecContext angle_ctx;
  angle_ctx.workdir = tmp / "ws_angle";
  const auto angle_out = angle->execute(
      {{"cloud", loaded.at("cloud")}},
      nlohmann::json{{"width", 256}, {"height", 128}}, angle_ctx);
  RTR_CHECK(angle_out.at("point_number").payload.as_int() > 0);
  RTR_CHECK(angle_out.at("point_number").payload.as_int() < n0);
  const auto angle_path = rtr::artifact_file_path(angle_out.at("cloud"));
  RTR_CHECK(std::filesystem::weakly_canonical(angle_path) ==
            std::filesystem::weakly_canonical(angle_ctx.workdir / "outputs" / "cloud.pcd"));
  RTR_CHECK(std::filesystem::is_regular_file(angle_path));
  RTR_CHECK(!std::filesystem::exists(cloud_p.parent_path() /
                                     (".cgraph_rtr_angle_" + cloud_p.stem().string())));

  const auto* medoid = reg.get("rtr.voxel_medoid_downsample");
  cgraph::ExecContext medoid_ctx;
  medoid_ctx.workdir = tmp / "ws_medoid";
  const auto medoid_out = medoid->execute(
      {{"cloud", loaded.at("cloud")}},
      nlohmann::json{{"voxel_size", 0.5}, {"multi_thread", false}}, medoid_ctx);
  RTR_CHECK(medoid_out.at("point_number").payload.as_int() > 0);
  RTR_CHECK(medoid_out.at("point_number").payload.as_int() < n0);
  const auto medoid_path = rtr::artifact_file_path(medoid_out.at("cloud"));
  RTR_CHECK(std::filesystem::weakly_canonical(medoid_path) ==
            std::filesystem::weakly_canonical(medoid_ctx.workdir / "outputs" / "cloud.pcd"));

  cgraph::PortSpec buffer_src = rtr::cloud_buffer_port("cloud");
  RTR_CHECK(cgraph::check_port_edge(buffer_src, angle->signature().inputs.at("cloud")) ==
            cgraph::EdgeCompat::RealisationMismatch);
  RTR_CHECK(cgraph::check_port_edge(rtr::cloud_port("cloud"),
                                    medoid->signature().inputs.at("cloud")) ==
            cgraph::EdgeCompat::Ok);

  const auto* res = reg.get("rtr.resolution_estimate");
  RTR_CHECK(res->signature().inputs.count("cloud") == 1);
  const auto res_out =
      res->execute({{"cloud", loaded.at("cloud")}}, nlohmann::json::object(), load_ctx);
  RTR_CHECK(res_out.at("voxel_size").payload.as_float() > 0.0);

  const auto* save = reg.get("rtr.point_cloud.save");
  RTR_CHECK(save != nullptr);
  const auto user_pcd = tmp / "user_export" / "kept.pcd";
  const auto saved = save->execute(
      {{"cloud", medoid_out.at("cloud")}, {"path", path_obj(user_pcd)}},
      nlohmann::json{{"format", "pcd"}}, load_ctx);
  RTR_CHECK(std::filesystem::is_regular_file(user_pcd));
  RTR_CHECK(std::filesystem::is_regular_file(medoid_path));
  RTR_CHECK(std::filesystem::weakly_canonical(rtr::artifact_file_path(saved.at("cloud"))) ==
            std::filesystem::weakly_canonical(user_pcd));

  bool bad_ext = false;
  try {
    save->execute({{"cloud", medoid_out.at("cloud")},
                   {"path", path_obj(tmp / "user_export" / "bad.pcd")}},
                  nlohmann::json{{"format", "las"}}, load_ctx);
  } catch (const std::exception&) {
    bad_ext = true;
  }
  RTR_CHECK(bad_ext);

  bool bad_fmt = false;
  try {
    save->execute({{"cloud", medoid_out.at("cloud")},
                   {"path", path_obj(tmp / "user_export" / "bad.obj")}},
                  nlohmann::json{{"format", "obj"}}, load_ctx);
  } catch (const std::exception&) {
    bad_fmt = true;
  }
  RTR_CHECK(bad_fmt);

  const auto* convert = reg.get("rtr.point_cloud.convert");
  RTR_CHECK(convert != nullptr);
  cgraph::ExecContext convert_ctx;
  convert_ctx.workdir = tmp / "ws_convert";
  const auto converted = convert->execute({{"cloud", medoid_out.at("cloud")}},
                                          nlohmann::json{{"format", "ply"}}, convert_ctx);
  const auto ply_path = rtr::artifact_file_path(converted.at("cloud"));
  RTR_CHECK(std::filesystem::weakly_canonical(ply_path) ==
            std::filesystem::weakly_canonical(convert_ctx.workdir / "outputs" / "cloud.ply"));
  RTR_CHECK(std::filesystem::is_regular_file(ply_path));

  const auto user_las = tmp / "user_export" / "kept.las";
  const auto saved_las = save->execute(
      {{"cloud", medoid_out.at("cloud")}, {"path", path_obj(user_las)}},
      nlohmann::json::object(), load_ctx);
  RTR_CHECK(std::filesystem::is_regular_file(user_las));
  RTR_CHECK(std::filesystem::weakly_canonical(rtr::artifact_file_path(saved_las.at("cloud"))) ==
            std::filesystem::weakly_canonical(user_las));

  bool empty_workdir = false;
  try {
    cgraph::ExecContext bare;
    medoid->execute({{"cloud", loaded.at("cloud")}},
                    nlohmann::json{{"voxel_size", 0.5}}, bare);
  } catch (const std::exception&) {
    empty_workdir = true;
  }
  RTR_CHECK(empty_workdir);

  bool threw = false;
  try {
    medoid->execute({{"cloud", loaded.at("cloud")}},
                    nlohmann::json{{"voxel_size", -1.0}}, medoid_ctx);
  } catch (const std::exception&) {
    threw = true;
  }
  RTR_CHECK(threw);
  return 0;
}
