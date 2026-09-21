#include "check.hpp"
#include "pcd_io.hpp"
#include "rtr/ops.hpp"

#include "cgraph/artifact.hpp"
#include "cgraph/data_helpers.hpp"
#include "cgraph/ops.hpp"

#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

static cgraph::DataObject file_obj(const std::filesystem::path& p) {
  return rtr::cloud_artifact_from_path(p);
}

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

  for (const char* id : {"rtr.point_cloud.load", "rtr.registration_initializer",
                         "rtr.registration_report", "rtr.rough_global_reg",
                         "rtr.fine_registration", "rtr.align_result.unpack"}) {
    const auto* op = reg.get(id);
    RTR_CHECK(op != nullptr);
    RTR_CHECK(!op->usage().principle.empty());
  }
  RTR_CHECK(reg.get("rtr.session.open") == nullptr);

  const auto tmp = std::filesystem::temp_directory_path() / "cgraph_rtr_mainchain";
  const auto src_p = tmp / "src.pcd";
  const auto tgt_p = tmp / "tgt.pcd";
  // Asymmetric station (same family as rough_fine gate). Dense enough for initializer.
  auto src_pts = make_asymmetric_station(0.04f);
  auto tgt_pts = src_pts;
  for (auto& p : tgt_pts) {
    p.x() += 0.05f;
  }
  write_ascii_xyz_pcd(src_p, src_pts);
  write_ascii_xyz_pcd(tgt_p, tgt_pts);

  // Separate denser cloud for initializer (needs more points / scene scale).
  const auto init_p = tmp / "init_src.pcd";
  write_ascii_xyz_pcd(init_p, make_sphere(4000));

  cgraph::ExecContext ctx;
  const auto* load = reg.get("rtr.point_cloud.load");
  bool threw = false;
  try {
    load->execute({{"path", path_obj("Z:/missing/cloud.pcd")}}, nlohmann::json::object(),
                  ctx);
  } catch (const std::exception&) {
    threw = true;
  }
  RTR_CHECK(threw);

  const auto loaded =
      load->execute({{"path", path_obj(src_p)}}, nlohmann::json::object(), ctx);
  RTR_CHECK(loaded.count("cloud"));
  const auto cloud_uri = rtr::artifact_file_path(loaded.at("cloud"));
  RTR_CHECK(std::filesystem::is_regular_file(cloud_uri));

  const auto loaded_init =
      load->execute({{"path", path_obj(init_p)}}, nlohmann::json::object(), ctx);
  const auto* init = reg.get("rtr.registration_initializer");
  cgraph::ExecContext init_ctx;
  init_ctx.workdir = tmp / "ws_init";
  const auto init_out = init->execute(
      {{"cloud", loaded_init.at("cloud")}},
      nlohmann::json{{"use_root", false}, {"resolution", 10}}, init_ctx);
  RTR_CHECK(init_out.count("cloud"));
  RTR_CHECK(init_out.at("voxel_size").payload.as_float() > 0.0);
  RTR_CHECK(init_out.at("point_number").payload.as_int() >= 10);
  RTR_CHECK(std::filesystem::weakly_canonical(
                rtr::artifact_file_path(init_out.at("cloud"))) ==
            std::filesystem::weakly_canonical(init_ctx.workdir / "outputs" / "cloud.pcd"));

  const auto* rough = reg.get("rtr.rough_global_reg");
  const auto rough_out = rough->execute(
      {{"src", loaded.at("cloud")}, {"tgt", file_obj(tgt_p)}},
      nlohmann::json{{"resolution", 0.08}}, ctx);
  const Ddx::AlignResult rough_align = rtr::align_result_from_data(rough_out.at("align"));
  RTR_CHECK(rough_align.matrix_(3, 3) == 1.0);

  const auto* unpack = reg.get("rtr.align_result.unpack");
  const auto unpacked =
      unpack->execute({{"align", rough_out.at("align")}}, nlohmann::json::object(), ctx);
  RTR_CHECK(unpacked.count("matrix"));

  const auto* fine = reg.get("rtr.fine_registration");
  const auto fine_out = fine->execute(
      {{"src", loaded.at("cloud")},
       {"tgt", file_obj(tgt_p)},
       {"guess", unpacked.at("matrix")}},
      nlohmann::json::object(), ctx);
  const Ddx::AlignResult fine_align = rtr::align_result_from_data(fine_out.at("align"));
  RTR_CHECK(fine_align.matrix_(3, 3) == 1.0);

  const auto* report = reg.get("rtr.registration_report");
  const auto report_out = report->execute(
      {{"src", loaded.at("cloud")},
       {"tgt", file_obj(tgt_p)},
       {"matrix", unpacked.at("matrix")}},
      nlohmann::json{{"voxel_size", 0.3}, {"distance_thresh", 0.5}}, ctx);
  RTR_CHECK(report_out.count("report"));
  const auto& fields = report_out.at("report").payload.as_record();
  RTR_CHECK(fields.count("overlap_ratio"));
  RTR_CHECK(fields.at("overlap_ratio").as_float() >= 0.0);
  return 0;
}
