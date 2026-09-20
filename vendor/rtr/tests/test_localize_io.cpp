#include "check.hpp"
#include "pcd_io.hpp"
#include "rtr/ops.hpp"

#include "cgraph/edge_check.hpp"
#include "cgraph/ops.hpp"
#include "cgraph/realisation.hpp"

#include <cmath>
#include <filesystem>
#include <map>
#include <string>

static cgraph::DataObject path_obj(const std::string& p) {
  return cgraph::make_data_object(cgraph::type_ids::string(),
                                  cgraph::SemanticSpec::of("cgraph.semantic.text"),
                                  cgraph::Payload::string(p));
}

int main() {
  cgraph::OpRegistry reg;
  rtr::register_rtr_ops(reg);
  for (const char* id : {"rtr.align_result.save", "rtr.align_result.load",
                         "rtr.lidar_frame.save", "rtr.lidar_frame.load",
                         "rtr.global_matrix.save", "rtr.global_matrix.load",
                         "rtr.point_cloud.to_buffer", "rtr.point_cloud.to_file"}) {
    RTR_CHECK(reg.get(id) != nullptr);
    RTR_CHECK(!reg.get(id)->usage().principle.empty());
  }

  const auto tmp = std::filesystem::temp_directory_path() / "cgraph_rtr_localize";
  std::filesystem::create_directories(tmp);
  cgraph::ExecContext ctx;

  Ddx::AlignResult align;
  align.src_name_ = "s";
  align.tgt_name_ = "t";
  align.matrix_ = Eigen::Matrix4d::Identity();
  align.matrix_(0, 3) = 1.5;
  align.rms_ = 0.01;
  align.final_rms_ = 0.02;
  align.similarity_ = 0.9;
  align.weight_ = 0.01;
  align.feature_num_ = 12;
  align.state_ = 1;
  align.auto_reg_ = 0;
  align.information_ = Eigen::Matrix<double, 6, 6>::Identity();

  const auto align_path = (tmp / "align.json").string();
  const auto* asave = reg.get("rtr.align_result.save");
  asave->execute({{"align", rtr::align_result_to_data(align)},
                  {"path", path_obj(align_path)}},
                 nlohmann::json::object(), ctx);
  const auto* aload = reg.get("rtr.align_result.load");
  const auto loaded =
      aload->execute({{"path", path_obj(align_path)}}, nlohmann::json::object(), ctx);
  const Ddx::AlignResult round = rtr::align_result_from_data(loaded.at("align"));
  RTR_CHECK(round.src_name_ == "s");
  RTR_CHECK(std::abs(round.matrix_(0, 3) - 1.5) < 1e-9);

  const auto cloud_p = tmp / "cloud.pcd";
  write_ascii_xyz_pcd(cloud_p, make_asymmetric_station(0.2f));

  Ddx::LidarFrame frame;
  frame.name_ = "station_a";
  frame.lasFn_ = cloud_p.string();
  frame.info_folder_ = tmp.string();
  frame.voxel_size_ = 0.05;
  frame.point_number_ = 42;
  frame.global_matrix_ = Eigen::Matrix4d::Identity();
  frame.root_.name_ = "root";
  frame.root_.cloud_path_ = cloud_p.string();
  frame.root_.point_number_ = 42;
  Ddx::SubVoxel sub;
  sub.name_ = "v0";
  sub.voxel_size_ = 0.05;
  sub.point_number_ = 10;
  frame.subvoxel_vec_.push_back(sub);

  const auto frame_path = (tmp / "frame.json").string();
  const auto* fsave = reg.get("rtr.lidar_frame.save");
  fsave->execute({{"frame", rtr::lidar_frame_to_data(frame)},
                  {"path", path_obj(frame_path)}},
                 nlohmann::json::object(), ctx);
  const auto* fload = reg.get("rtr.lidar_frame.load");
  const auto frame_out =
      fload->execute({{"path", path_obj(frame_path)}}, nlohmann::json::object(), ctx);
  const Ddx::LidarFrame frame_round =
      rtr::lidar_frame_from_data(frame_out.at("frame"));
  RTR_CHECK(frame_round.name_ == "station_a");
  RTR_CHECK(frame_round.lasFn_ == cloud_p.string());
  RTR_CHECK(frame_round.subvoxel_vec_.size() == 1);
  RTR_CHECK(frame_round.subvoxel_vec_[0].name_ == "v0");

  const auto* pcload = reg.get("rtr.point_cloud.load");
  const auto cloud_file =
      pcload->execute({{"path", path_obj(cloud_p.string())}}, nlohmann::json::object(),
                      ctx);
  RTR_CHECK(cloud_file.at("cloud").realisation &&
            *cloud_file.at("cloud").realisation == cgraph::Realisation::File);

  const auto* to_buf = reg.get("rtr.point_cloud.to_buffer");
  const auto buf_out = to_buf->execute({{"cloud", cloud_file.at("cloud")}},
                                       nlohmann::json::object(), ctx);
  RTR_CHECK(buf_out.at("cloud").realisation &&
            *buf_out.at("cloud").realisation == cgraph::Realisation::Buffer);

  const auto out_pcd = (tmp / "from_buffer.pcd").string();
  const auto* to_file = reg.get("rtr.point_cloud.to_file");
  const auto file_out =
      to_file->execute({{"cloud", buf_out.at("cloud")}, {"path", path_obj(out_pcd)}},
                       nlohmann::json::object(), ctx);
  RTR_CHECK(file_out.at("cloud").realisation &&
            *file_out.at("cloud").realisation == cgraph::Realisation::File);
  RTR_CHECK(std::filesystem::is_regular_file(out_pcd));

  cgraph::PortSpec file_out_port = rtr::cloud_port("cloud");
  cgraph::PortSpec buffer_only = rtr::cloud_buffer_port("cloud");
  // Kind differs (Artifact vs Value) → KindMismatch before realisation.
  RTR_CHECK(cgraph::check_port_edge(file_out_port, buffer_only) ==
            cgraph::EdgeCompat::KindMismatch);

  std::map<std::string, Eigen::Matrix4d> table;
  Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
  m(1, 3) = 2.0;
  table.emplace("station_a", m);
  table.emplace("station_b", Eigen::Matrix4d::Identity());
  const auto gpath = (tmp / "global_matrix.json").string();
  const auto* gsave = reg.get("rtr.global_matrix.save");
  gsave->execute({{"table", rtr::global_matrix_table_to_data(table)},
                  {"path", path_obj(gpath)}},
                 nlohmann::json::object(), ctx);
  const auto* gload = reg.get("rtr.global_matrix.load");
  const auto gout =
      gload->execute({{"path", path_obj(gpath)}}, nlohmann::json::object(), ctx);
  const auto round_table = rtr::global_matrix_table_from_data(gout.at("table"));
  RTR_CHECK(round_table.count("station_a") == 1);
  RTR_CHECK(std::abs(round_table.at("station_a")(1, 3) - 2.0) < 1e-9);
  return 0;
}
