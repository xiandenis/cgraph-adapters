#include "rtr/json.hpp"

#include "cgraph/artifact.hpp"
#include "cgraph/data_helpers.hpp"
#include "cgraph/realisation.hpp"
#include "cgraph/type_catalog.hpp"
#include "cgraph/type_codec.hpp"
#include "cgraph/type_pack_yaml.hpp"

#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace rtr {
namespace {

class PointXyzF32Codec final : public cgraph::TypeCodec {
 public:
  std::string codec_id() const override { return "rtr.codec.point_xyz_f32"; }
  std::vector<std::uint8_t> canonicalize(
      const std::vector<std::uint8_t>& bytes) const override {
    return bytes;
  }
  void validate(const std::vector<std::uint8_t>& bytes) const override {
    if (bytes.size() < sizeof(std::uint64_t)) {
      throw std::invalid_argument("rtr.codec.point_xyz_f32: too small");
    }
    std::uint64_t count = 0;
    std::memcpy(&count, bytes.data(), sizeof(count));
    const std::size_t need =
        sizeof(count) + static_cast<std::size_t>(count) * 3 * sizeof(float);
    if (bytes.size() != need) {
      throw std::invalid_argument("rtr.codec.point_xyz_f32: size mismatch");
    }
  }
};

nlohmann::json matrix_to_nested(const Eigen::MatrixXd& m) {
  nlohmann::json rows = nlohmann::json::array();
  for (int r = 0; r < m.rows(); ++r) {
    nlohmann::json row = nlohmann::json::array();
    for (int c = 0; c < m.cols(); ++c) {
      row.push_back(m(r, c));
    }
    rows.push_back(std::move(row));
  }
  return rows;
}

Eigen::MatrixXd nested_to_matrix(const nlohmann::json& j, int rows, int cols,
                                 const char* what) {
  if (!j.is_array() || static_cast<int>(j.size()) != rows) {
    throw_json(std::string(what) + ": expected nested array");
  }
  Eigen::MatrixXd m(rows, cols);
  for (int r = 0; r < rows; ++r) {
    if (!j[r].is_array() || static_cast<int>(j[r].size()) != cols) {
      throw_json(std::string(what) + ": bad row");
    }
    for (int c = 0; c < cols; ++c) {
      if (!j[r][c].is_number()) {
        throw_json(std::string(what) + ": non-numeric");
      }
      m(r, c) = j[r][c].get<double>();
    }
  }
  return m;
}

}  // namespace

nlohmann::json matrix4d_to_json(const Eigen::Matrix4d& m) {
  return matrix_to_nested(m);
}

Eigen::Matrix4d matrix4d_from_json(const nlohmann::json& j) {
  if (j.is_object()) {
    throw_json("matrix4d: object form (e.g. {R,t}) is not accepted");
  }
  return nested_to_matrix(j, 4, 4, "matrix4d");
}

nlohmann::json matrix6d_to_json(const Eigen::Matrix<double, 6, 6>& m) {
  return matrix_to_nested(m);
}

Eigen::Matrix<double, 6, 6> matrix6d_from_json(const nlohmann::json& j) {
  return nested_to_matrix(j, 6, 6, "matrix6d");
}

nlohmann::json align_result_to_json(const Ddx::AlignResult& a) {
  return nlohmann::json{
      {"src_name", a.src_name_},
      {"tgt_name", a.tgt_name_},
      {"matrix", matrix4d_to_json(a.matrix_)},
      {"rms", a.rms_},
      {"final_rms", a.final_rms_},
      {"similarity", a.similarity_},
      {"weight", a.weight_},
      {"feature_num", a.feature_num_},
      {"state", a.state_},
      {"auto_reg", a.auto_reg_},
      {"information", matrix6d_to_json(a.information_)},
  };
}

Ddx::AlignResult align_result_from_json(const nlohmann::json& j) {
  if (!j.is_object()) {
    throw_json("align_result: expected object");
  }
  Ddx::AlignResult a;
  a.src_name_ = j.value("src_name", std::string());
  a.tgt_name_ = j.value("tgt_name", std::string());
  if (!j.contains("matrix")) {
    throw_json("align_result: missing matrix");
  }
  a.matrix_ = matrix4d_from_json(j.at("matrix"));
  a.rms_ = j.value("rms", 0.0);
  a.final_rms_ = j.value("final_rms", 0.0);
  a.similarity_ = j.value("similarity", 0.0);
  a.weight_ = j.value("weight", 0.0);
  a.feature_num_ = j.value("feature_num", 0);
  a.state_ = j.value("state", 0);
  a.auto_reg_ = j.value("auto_reg", 0);
  if (j.contains("information")) {
    a.information_ = matrix6d_from_json(j.at("information"));
  } else {
    a.information_ = Eigen::Matrix<double, 6, 6>::Identity();
  }
  return a;
}

std::filesystem::path artifact_file_path(const nlohmann::json& handle) {
  const cgraph::Artifact a = cgraph::artifact_from_json(handle);
  if (a.is_dir) {
    throw_json("artifact_file_path: expected file, got dir");
  }
  return a.location;
}

std::filesystem::path artifact_file_path(const cgraph::DataObject& obj) {
  if (obj.payload.kind() != cgraph::Payload::Kind::DataRef) {
    throw_json("artifact_file_path: expected DataRef");
  }
  const cgraph::DataRef& ref = obj.payload.as_data_ref();
  if (ref.is_dir) {
    throw_json("artifact_file_path: expected file, got dir");
  }
  return ref.uri;
}

cgraph::SemanticSpec rigid_transform_semantic() {
  cgraph::SemanticSpec s =
      cgraph::SemanticSpec::of("cgraph.semantic.rigid_transform");
  s.frame_from = "source";
  s.frame_to = "target";
  return s;
}

void register_rtr_types() {
  if (!cgraph::CodecRegistry::global().contains("rtr.codec.point_xyz_f32")) {
    cgraph::CodecRegistry::global().add(std::make_shared<PointXyzF32Codec>());
  }
  cgraph::TypeCatalog::global().install(cgraph::load_type_pack_yaml(
      std::filesystem::path(__FILE__).parent_path().parent_path() /
      "catalog" / "cgraph-types.yaml"));
}

cgraph::PortSpec cloud_port(std::string name) {
  cgraph::PortSpec p = cgraph::make_port(
      std::move(name), cgraph::PortKind::Artifact,
      cgraph::TypeId::parse("rtr.type.point_cloud"),
      cgraph::SemanticSpec::of("rtr.semantic.point_cloud"));
  p.produces_realisation = {"file"};
  p.accepts_realisation = {"file"};
  return p;
}

cgraph::PortSpec cloud_buffer_port(std::string name) {
  cgraph::PortSpec p = cgraph::make_port(
      std::move(name), cgraph::PortKind::Value,
      cgraph::TypeId::parse("rtr.type.point_cloud"),
      cgraph::SemanticSpec::of("rtr.semantic.point_cloud"));
  p.produces_realisation = {"buffer"};
  p.accepts_realisation = {"buffer"};
  return p;
}

cgraph::PortSpec lidar_frame_port(std::string name) {
  return cgraph::make_port(
      std::move(name), cgraph::PortKind::Value,
      cgraph::TypeId::parse("rtr.type.lidar_frame"),
      cgraph::SemanticSpec::of("rtr.semantic.lidar_frame"));
}

cgraph::PortSpec matrix_port(std::string name, bool optional) {
  cgraph::PortSpec p =
      cgraph::make_port(std::move(name), cgraph::PortKind::Value,
                        cgraph::type_ids::matrix_r4c4_f64(),
                        rigid_transform_semantic());
  p.optional = optional;
  return p;
}

cgraph::PortSpec align_port(std::string name) {
  return cgraph::make_port(std::move(name), cgraph::PortKind::Value,
                           cgraph::TypeId::parse("rtr.type.align_result"),
                           cgraph::SemanticSpec::of("rtr.semantic.align_result"));
}

cgraph::PortSpec report_port(std::string name) {
  return cgraph::make_port(
      std::move(name), cgraph::PortKind::Value,
      cgraph::TypeId::parse("rtr.type.registration_report"),
      cgraph::SemanticSpec::of("rtr.semantic.registration_report"));
}

cgraph::PortSpec path_port(std::string name) {
  return cgraph::make_port(std::move(name), cgraph::PortKind::Value,
                           cgraph::type_ids::string(),
                           cgraph::SemanticSpec::of("cgraph.semantic.text"));
}

cgraph::PortSpec align_file_port(std::string name) {
  return cgraph::make_port(
      std::move(name), cgraph::PortKind::Artifact,
      cgraph::TypeId::parse("rtr.type.align_result_file"),
      cgraph::SemanticSpec::of("rtr.semantic.align_result_file"));
}

cgraph::PortSpec lidar_frame_file_port(std::string name) {
  return cgraph::make_port(
      std::move(name), cgraph::PortKind::Artifact,
      cgraph::TypeId::parse("rtr.type.lidar_frame_file"),
      cgraph::SemanticSpec::of("rtr.semantic.lidar_frame_file"));
}

cgraph::PortSpec global_matrix_file_port(std::string name) {
  return cgraph::make_port(
      std::move(name), cgraph::PortKind::Artifact,
      cgraph::TypeId::parse("rtr.type.global_matrix_file"),
      cgraph::SemanticSpec::of("rtr.semantic.global_matrix_file"));
}

cgraph::PortSpec global_matrix_table_port(std::string name) {
  return cgraph::make_port(
      std::move(name), cgraph::PortKind::Value,
      cgraph::type_ids::list(cgraph::TypeId::parse("rtr.type.station_global_pose")),
      cgraph::SemanticSpec::of("rtr.semantic.global_matrix_table"));
}

cgraph::PortSpec scalar_float_port(std::string name) {
  static const std::unordered_map<std::string, const char*> kFloatSem{
      {"rms", "rtr.semantic.rms"},
      {"final_rms", "rtr.semantic.final_rms"},
      {"similarity", "rtr.semantic.similarity"},
      {"weight", "rtr.semantic.weight"},
      {"overlap_ratio", "rtr.semantic.overlap_ratio"},
      {"voxel_size", "rtr.semantic.voxel_size"},
  };
  const auto it = kFloatSem.find(name);
  if (it == kFloatSem.end()) {
    throw std::invalid_argument("scalar_float_port: unknown name: " + name);
  }
  return cgraph::make_port(std::move(name), cgraph::PortKind::Value,
                           cgraph::type_ids::floating(),
                           cgraph::SemanticSpec::of(it->second));
}

cgraph::PortSpec scalar_int_port(std::string name) {
  static const std::unordered_map<std::string, const char*> kIntSem{
      {"feature_num", "rtr.semantic.feature_num"},
      {"state", "rtr.semantic.state"},
      {"auto_reg", "rtr.semantic.auto_reg"},
      {"point_number", "rtr.semantic.point_count"},
      {"overlap_count", "rtr.semantic.overlap_count"},
      {"rms_sample_count", "rtr.semantic.rms_sample_count"},
  };
  const auto it = kIntSem.find(name);
  if (it == kIntSem.end()) {
    throw std::invalid_argument("scalar_int_port: unknown name: " + name);
  }
  return cgraph::make_port(std::move(name), cgraph::PortKind::Value,
                           cgraph::type_ids::integer(),
                           cgraph::SemanticSpec::of(it->second));
}

cgraph::PortSpec scalar_string_port(std::string name) {
  return cgraph::make_port(std::move(name), cgraph::PortKind::Value,
                           cgraph::type_ids::string(),
                           cgraph::SemanticSpec::of("cgraph.semantic.text"));
}

cgraph::PortSpec information_port(std::string name) {
  return cgraph::make_port(
      std::move(name), cgraph::PortKind::Value,
      cgraph::TypeId::parse("rtr.type.information_matrix"),
      cgraph::SemanticSpec::of("rtr.semantic.information"));
}

cgraph::Payload matrix_to_payload(const Eigen::Matrix4d& m) {
  cgraph::Matrix4x4 a{};
  int i = 0;
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      a[static_cast<std::size_t>(i++)] = m(r, c);
    }
  }
  return cgraph::Payload::matrix4x4(a);
}

Eigen::Matrix4d matrix_from_payload(const cgraph::Payload& payload) {
  const cgraph::Matrix4x4& a = payload.as_matrix4x4();
  Eigen::Matrix4d m;
  int i = 0;
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      m(r, c) = a[static_cast<std::size_t>(i++)];
    }
  }
  return m;
}

cgraph::DataObject align_result_to_data(const Ddx::AlignResult& a) {
  std::map<std::string, cgraph::Payload> fields;
  fields.emplace("matrix", matrix_to_payload(a.matrix_));
  fields.emplace("src_name", cgraph::Payload::string(a.src_name_));
  fields.emplace("tgt_name", cgraph::Payload::string(a.tgt_name_));
  fields.emplace("rms", cgraph::Payload::floating(a.rms_));
  fields.emplace("final_rms", cgraph::Payload::floating(a.final_rms_));
  fields.emplace("similarity", cgraph::Payload::floating(a.similarity_));
  fields.emplace("weight", cgraph::Payload::floating(a.weight_));
  fields.emplace("feature_num", cgraph::Payload::integer(a.feature_num_));
  fields.emplace("state", cgraph::Payload::integer(a.state_));
  fields.emplace("auto_reg", cgraph::Payload::integer(a.auto_reg_));
  {
    std::vector<cgraph::Payload> items;
    items.reserve(36);
    for (int r = 0; r < 6; ++r) {
      for (int c = 0; c < 6; ++c) {
        items.push_back(cgraph::Payload::floating(a.information_(r, c)));
      }
    }
    fields.emplace("information", cgraph::Payload::list(std::move(items)));
  }
  return cgraph::make_data_object(
      cgraph::TypeId::parse("rtr.type.align_result"),
      cgraph::SemanticSpec::of("rtr.semantic.align_result"),
      cgraph::Payload::record(std::move(fields)));
}

Ddx::AlignResult align_result_from_data(const cgraph::DataObject& obj) {
  if (obj.payload.kind() != cgraph::Payload::Kind::Record) {
    throw_json("align_result: expected record payload");
  }
  const auto& fields = obj.payload.as_record();
  auto require = [&](const char* key) -> const cgraph::Payload& {
    const auto it = fields.find(key);
    if (it == fields.end()) {
      throw_json(std::string("align_result: missing field ") + key);
    }
    return it->second;
  };
  Ddx::AlignResult a;
  a.matrix_ = matrix_from_payload(require("matrix"));
  a.src_name_ = require("src_name").as_string();
  a.tgt_name_ = require("tgt_name").as_string();
  a.rms_ = require("rms").as_float();
  a.final_rms_ = fields.count("final_rms") ? fields.at("final_rms").as_float() : 0;
  a.similarity_ =
      fields.count("similarity") ? fields.at("similarity").as_float() : 0;
  a.weight_ = fields.count("weight") ? fields.at("weight").as_float() : 0;
  a.feature_num_ =
      fields.count("feature_num") ? fields.at("feature_num").as_int() : 0;
  a.state_ = fields.count("state") ? fields.at("state").as_int() : 0;
  a.auto_reg_ = fields.count("auto_reg") ? fields.at("auto_reg").as_int() : 0;
  if (fields.count("information")) {
    const auto& info = fields.at("information");
    if (info.kind() != cgraph::Payload::Kind::List ||
        info.as_list().size() != 36) {
      throw_json("align_result: information must be 36 floats");
    }
    int i = 0;
    for (int r = 0; r < 6; ++r) {
      for (int c = 0; c < 6; ++c) {
        a.information_(r, c) =
            info.as_list()[static_cast<std::size_t>(i++)].as_float();
      }
    }
  } else {
    a.information_ = Eigen::Matrix<double, 6, 6>::Identity();
  }
  return a;
}

cgraph::DataObject cloud_artifact_from_path(const std::filesystem::path& path) {
  return file_artifact_from_path(path, "rtr.type.point_cloud",
                                 "rtr.semantic.point_cloud");
}

cgraph::DataObject file_artifact_from_path(const std::filesystem::path& path,
                                           const char* type_id,
                                           const char* semantic_id) {
  const auto abs = std::filesystem::weakly_canonical(path);
  const cgraph::Artifact art = cgraph::make_file_artifact(abs);
  cgraph::DataRef ref;
  ref.uri = art.location.string();
  ref.content_hash = art.content_hash;
  ref.is_dir = art.is_dir;
  return cgraph::make_artifact_object(cgraph::TypeId::parse(type_id),
                                      cgraph::SemanticSpec::of(semantic_id),
                                      std::move(ref));
}

cgraph::DataObject registration_report_to_data(float overlap_ratio, double rms,
                                               std::size_t overlap_count,
                                               std::size_t rms_sample_count,
                                               bool rms_query_from_target) {
  std::map<std::string, cgraph::Payload> fields;
  fields.emplace("overlap_ratio",
                 cgraph::Payload::floating(static_cast<double>(overlap_ratio)));
  fields.emplace("rms", cgraph::Payload::floating(rms));
  fields.emplace("overlap_count",
                 cgraph::Payload::integer(static_cast<std::int64_t>(overlap_count)));
  fields.emplace(
      "rms_sample_count",
      cgraph::Payload::integer(static_cast<std::int64_t>(rms_sample_count)));
  fields.emplace("rms_query_from_target",
                 cgraph::Payload::boolean(rms_query_from_target));
  return cgraph::make_data_object(
      cgraph::TypeId::parse("rtr.type.registration_report"),
      cgraph::SemanticSpec::of("rtr.semantic.registration_report"),
      cgraph::Payload::record(std::move(fields)));
}

cgraph::DataObject global_matrix_table_to_data(
    const std::map<std::string, Eigen::Matrix4d>& table) {
  std::vector<cgraph::Payload> entries;
  entries.reserve(table.size());
  for (const auto& [name, mat] : table) {
    std::map<std::string, cgraph::Payload> fields;
    fields.emplace("name", cgraph::Payload::string(name));
    fields.emplace("matrix", matrix_to_payload(mat));
    entries.push_back(cgraph::Payload::record(std::move(fields)));
  }
  return cgraph::make_data_object(
      cgraph::type_ids::list(cgraph::TypeId::parse("rtr.type.station_global_pose")),
      cgraph::SemanticSpec::of("rtr.semantic.global_matrix_table"),
      cgraph::Payload::list(std::move(entries)));
}

std::map<std::string, Eigen::Matrix4d> global_matrix_table_from_data(
    const cgraph::DataObject& obj) {
  if (obj.payload.kind() != cgraph::Payload::Kind::List) {
    throw_json("global_matrix_table: expected list payload");
  }
  std::map<std::string, Eigen::Matrix4d> out;
  for (const auto& item : obj.payload.as_list()) {
    if (item.kind() != cgraph::Payload::Kind::Record) {
      throw_json("global_matrix_table: entry must be record");
    }
    const auto& fields = item.as_record();
    if (!fields.count("name") || !fields.count("matrix")) {
      throw_json("global_matrix_table: entry missing name/matrix");
    }
    out.emplace(fields.at("name").as_string(),
                matrix_from_payload(fields.at("matrix")));
  }
  return out;
}

cgraph::Payload voxel_root_to_payload(const Ddx::VoxelRoot& root) {
  std::map<std::string, cgraph::Payload> fields;
  fields.emplace("name", cgraph::Payload::string(root.name_));
  fields.emplace("cloud_path", cgraph::Payload::string(root.cloud_path_));
  fields.emplace("normal_path", cgraph::Payload::string(root.normal_path_));
  fields.emplace("state", cgraph::Payload::integer(root.state_));
  fields.emplace("point_number", cgraph::Payload::integer(root.point_number_));
  fields.emplace("is_raw_input", cgraph::Payload::boolean(root.is_raw_input_));
  return cgraph::Payload::record(std::move(fields));
}

Ddx::VoxelRoot voxel_root_from_payload(const cgraph::Payload& payload) {
  const auto& fields = payload.as_record();
  Ddx::VoxelRoot root;
  root.name_ = fields.at("name").as_string();
  root.cloud_path_ = fields.at("cloud_path").as_string();
  root.normal_path_ = fields.at("normal_path").as_string();
  root.state_ = static_cast<Ddx::StateType>(fields.at("state").as_int());
  root.point_number_ = static_cast<int>(fields.at("point_number").as_int());
  root.is_raw_input_ = fields.at("is_raw_input").as_bool();
  return root;
}

cgraph::Payload sub_voxel_to_payload(const Ddx::SubVoxel& v) {
  std::map<std::string, cgraph::Payload> fields;
  fields.emplace("state", cgraph::Payload::integer(v.state_));
  fields.emplace("name", cgraph::Payload::string(v.name_));
  fields.emplace("point_number", cgraph::Payload::integer(v.point_number_));
  fields.emplace("voxel_size", cgraph::Payload::floating(v.voxel_size_));
  return cgraph::Payload::record(std::move(fields));
}

Ddx::SubVoxel sub_voxel_from_payload(const cgraph::Payload& payload) {
  const auto& fields = payload.as_record();
  Ddx::SubVoxel v;
  v.state_ = static_cast<Ddx::StateType>(fields.at("state").as_int());
  v.name_ = fields.at("name").as_string();
  v.point_number_ = static_cast<int>(fields.at("point_number").as_int());
  v.voxel_size_ = fields.at("voxel_size").as_float();
  return v;
}

cgraph::DataObject lidar_frame_to_data(const Ddx::LidarFrame& frame) {
  std::map<std::string, cgraph::Payload> fields;
  fields.emplace("name", cgraph::Payload::string(frame.name_));
  fields.emplace("stamp", cgraph::Payload::floating(frame.stamp_));
  fields.emplace("las_fn", cgraph::Payload::string(frame.lasFn_));
  fields.emplace("normal_fn", cgraph::Payload::string(frame.normalFn_));
  fields.emplace("json", cgraph::Payload::string(frame.json_));
  fields.emplace("info_folder", cgraph::Payload::string(frame.info_folder_));
  fields.emplace("sensor", cgraph::Payload::integer(frame.sensor_));
  fields.emplace("altimeter", cgraph::Payload::floating(frame.altimeter_));
  {
    std::vector<cgraph::Payload> gps;
    gps.push_back(cgraph::Payload::floating(frame.gps_.x()));
    gps.push_back(cgraph::Payload::floating(frame.gps_.y()));
    gps.push_back(cgraph::Payload::floating(frame.gps_.z()));
    fields.emplace("gps", cgraph::Payload::list(std::move(gps)));
  }
  fields.emplace("angle_resolution",
                 cgraph::Payload::integer(frame.angle_resolution_));
  fields.emplace("global_matrix", matrix_to_payload(frame.global_matrix_));
  fields.emplace("state", cgraph::Payload::integer(frame.state_));
  fields.emplace("point_number", cgraph::Payload::integer(frame.point_number_));
  fields.emplace("scene_scale", cgraph::Payload::floating(frame.scene_scale_));
  fields.emplace("voxel_size", cgraph::Payload::floating(frame.voxel_size_));
  fields.emplace("voxel_scene_class",
                 cgraph::Payload::integer(
                     static_cast<std::int64_t>(frame.voxel_scene_class_)));
  fields.emplace("root", voxel_root_to_payload(frame.root_));
  {
    std::vector<cgraph::Payload> subvoxels;
    subvoxels.reserve(frame.subvoxel_vec_.size());
    for (const Ddx::SubVoxel& v : frame.subvoxel_vec_) {
      subvoxels.push_back(sub_voxel_to_payload(v));
    }
    fields.emplace("subvoxel_vec", cgraph::Payload::list(std::move(subvoxels)));
  }
  return cgraph::make_data_object(
      cgraph::TypeId::parse("rtr.type.lidar_frame"),
      cgraph::SemanticSpec::of("rtr.semantic.lidar_frame"),
      cgraph::Payload::record(std::move(fields)));
}

Ddx::LidarFrame lidar_frame_from_data(const cgraph::DataObject& obj) {
  if (obj.payload.kind() != cgraph::Payload::Kind::Record) {
    throw_json("lidar_frame: expected record payload");
  }
  const auto& fields = obj.payload.as_record();
  auto require = [&](const char* key) -> const cgraph::Payload& {
    const auto it = fields.find(key);
    if (it == fields.end()) {
      throw_json(std::string("lidar_frame: missing field ") + key);
    }
    return it->second;
  };
  Ddx::LidarFrame frame;
  frame.name_ = require("name").as_string();
  frame.stamp_ = require("stamp").as_float();
  frame.lasFn_ = require("las_fn").as_string();
  frame.normalFn_ = require("normal_fn").as_string();
  frame.json_ = require("json").as_string();
  frame.info_folder_ = require("info_folder").as_string();
  frame.sensor_ = static_cast<Ddx::SensorInfo>(require("sensor").as_int());
  frame.altimeter_ = require("altimeter").as_float();
  {
    const auto& gps = require("gps").as_list();
    if (gps.size() != 3) {
      throw_json("lidar_frame: gps must be 3 floats");
    }
    frame.gps_ = Eigen::Vector3d(gps[0].as_float(), gps[1].as_float(),
                                 gps[2].as_float());
  }
  frame.angle_resolution_ = static_cast<int>(require("angle_resolution").as_int());
  frame.global_matrix_ = matrix_from_payload(require("global_matrix"));
  frame.state_ = static_cast<Ddx::StateType>(require("state").as_int());
  frame.point_number_ = static_cast<int>(require("point_number").as_int());
  frame.scene_scale_ = require("scene_scale").as_float();
  frame.voxel_size_ = require("voxel_size").as_float();
  frame.voxel_scene_class_ = static_cast<Ddx::VoxelSceneClass>(
      require("voxel_scene_class").as_int());
  frame.root_ = voxel_root_from_payload(require("root"));
  frame.subvoxel_vec_.clear();
  for (const auto& item : require("subvoxel_vec").as_list()) {
    frame.subvoxel_vec_.push_back(sub_voxel_from_payload(item));
  }
  return frame;
}

}  // namespace rtr
