#pragma once

#include "rtr/json.hpp"

#include <point_cloud_io/point_cloud_io.hpp>

#include "cgraph/ops.hpp"
#include "cgraph/realisation.hpp"
#include "cgraph/type_codec.hpp"

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace rtr {

inline constexpr const char* kPointXyzF32Codec = "rtr.codec.point_xyz_f32";

inline void require_file_point_cloud(const cgraph::DataObject& cloud_obj,
                                      const char* op_id) {
  if (cloud_obj.realisation &&
      *cloud_obj.realisation == cgraph::Realisation::Buffer) {
    throw cgraph::OperatorError(
        cgraph::ErrorCode::OpFailed,
        std::string(op_id) + ": point cloud must be file realisation");
  }
}

/// `{workdir}/outputs/{port}.pcd`. Throws if workdir is empty.
inline std::filesystem::path workspace_pcd_path(const cgraph::ExecContext& ctx,
                                                std::string_view port) {
  auto path = ctx.output_path(port);
  path += ".pcd";
  return path;
}

enum class CloudFileFormat { Las, Pcd, Ply };

inline std::string lowercase_ascii(std::string s) {
  for (char& c : s) {
    if (c >= 'A' && c <= 'Z') {
      c = static_cast<char>(c - 'A' + 'a');
    }
  }
  return s;
}

inline const char* cloud_format_ext(CloudFileFormat fmt) {
  switch (fmt) {
    case CloudFileFormat::Las:
      return ".las";
    case CloudFileFormat::Pcd:
      return ".pcd";
    case CloudFileFormat::Ply:
      return ".ply";
  }
  return ".las";
}

inline CloudFileFormat parse_cloud_format(const nlohmann::json& params, const char* op_id,
                                         const char* key = "format",
                                         const char* default_fmt = "las") {
  std::string raw = default_fmt;
  if (params.is_object() && params.contains(key)) {
    if (!params[key].is_string()) {
      throw cgraph::OperatorError(
          cgraph::ErrorCode::OpFailed,
          std::string(op_id) + ": format must be a string (las|pcd|ply)");
    }
    raw = params[key].get<std::string>();
  }
  const std::string v = lowercase_ascii(std::move(raw));
  if (v == "las") {
    return CloudFileFormat::Las;
  }
  if (v == "pcd") {
    return CloudFileFormat::Pcd;
  }
  if (v == "ply") {
    return CloudFileFormat::Ply;
  }
  throw cgraph::OperatorError(
      cgraph::ErrorCode::OpFailed,
      std::string(op_id) + ": unsupported format '" + v +
          "'; allowed: las, pcd, ply");
}

inline void require_path_matches_format(const std::filesystem::path& path,
                                        CloudFileFormat fmt, const char* op_id) {
  const std::string ext = lowercase_ascii(path.extension().string());
  const std::string need = cloud_format_ext(fmt);
  if (ext != need) {
    throw cgraph::OperatorError(
        cgraph::ErrorCode::OpFailed,
        std::string(op_id) + ": path extension '" + ext + "' does not match format '" +
            need.substr(1) + "' (path must end with " + need + ")");
  }
}

inline std::filesystem::path workspace_cloud_path(const cgraph::ExecContext& ctx,
                                                  std::string_view port,
                                                  CloudFileFormat fmt) {
  auto path = ctx.output_path(port);
  path += cloud_format_ext(fmt);
  return path;
}

inline cgraph::ParamSpec cloud_format_param() {
  cgraph::ParamSpec p;
  p.name = "format";
  p.dtype = "string";
  p.default_value = "las";
  p.doc = "Output format: las | pcd | ply";
  p.bindable = false;
  return p;
}

inline cgraph::DataObject save_xyz_cloud_format(
    const pcl::PointCloud<pcl::PointXYZ>& cloud, const std::filesystem::path& out_path,
    const char* op_id) {
  if (cloud.empty()) {
    throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                std::string(op_id) + ": refusing empty cloud");
  }
  std::filesystem::create_directories(out_path.parent_path());
  if (Ddx::point_cloud_io::save(out_path.string(), cloud) != 0) {
    throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                std::string(op_id) + ": failed to write " +
                                    out_path.string());
  }
  return cloud_artifact_from_path(out_path);
}

inline pcl::PointCloud<pcl::PointXYZ>::Ptr load_xyz_cloud(
    const cgraph::DataObject& cloud_obj, const char* op_id) {
  if (cloud_obj.realisation &&
      *cloud_obj.realisation == cgraph::Realisation::Buffer) {
    if (cloud_obj.payload.kind() != cgraph::Payload::Kind::Opaque ||
        cloud_obj.payload.as_codec_id() != kPointXyzF32Codec) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  std::string(op_id) + ": bad buffer codec");
    }
    const auto& bytes = cloud_obj.payload.as_opaque();
    if (bytes.size() < sizeof(std::uint64_t)) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  std::string(op_id) + ": buffer too small");
    }
    std::uint64_t count = 0;
    std::memcpy(&count, bytes.data(), sizeof(count));
    const std::size_t need =
        sizeof(count) + static_cast<std::size_t>(count) * 3 * sizeof(float);
    if (bytes.size() != need) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  std::string(op_id) + ": buffer size mismatch");
    }
    auto cloud =
        pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->resize(static_cast<std::uint32_t>(count));
    const float* xyz = reinterpret_cast<const float*>(bytes.data() + sizeof(count));
    for (std::uint64_t i = 0; i < count; ++i) {
      (*cloud)[static_cast<std::uint32_t>(i)].x = xyz[i * 3 + 0];
      (*cloud)[static_cast<std::uint32_t>(i)].y = xyz[i * 3 + 1];
      (*cloud)[static_cast<std::uint32_t>(i)].z = xyz[i * 3 + 2];
    }
    return cloud;
  }
  const auto path = artifact_file_path(cloud_obj);
  if (!std::filesystem::is_regular_file(path)) {
    throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                std::string(op_id) + ": cloud file missing");
  }
  auto cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  if (Ddx::point_cloud_io::load(path.string(), *cloud) != 0 || cloud->empty()) {
    throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                std::string(op_id) + ": failed to load cloud");
  }
  return cloud;
}

inline cgraph::DataObject point_cloud_buffer_from_xyz(
    const pcl::PointCloud<pcl::PointXYZ>& cloud) {
  const std::uint64_t count = cloud.size();
  std::vector<std::uint8_t> bytes(sizeof(count) +
                                  static_cast<std::size_t>(count) * 3 * sizeof(float));
  std::memcpy(bytes.data(), &count, sizeof(count));
  float* xyz = reinterpret_cast<float*>(bytes.data() + sizeof(count));
  for (std::uint64_t i = 0; i < count; ++i) {
    xyz[i * 3 + 0] = cloud[static_cast<std::uint32_t>(i)].x;
    xyz[i * 3 + 1] = cloud[static_cast<std::uint32_t>(i)].y;
    xyz[i * 3 + 2] = cloud[static_cast<std::uint32_t>(i)].z;
  }
  return cgraph::make_data_object(
      cgraph::TypeId::parse("rtr.type.point_cloud"),
      cgraph::SemanticSpec::of("rtr.semantic.point_cloud"),
      cgraph::Payload::opaque(kPointXyzF32Codec, std::move(bytes)),
      cgraph::Realisation::Buffer);
}

/// §9.3: when both PointCloud edge and LidarFrame are present, the edge is
/// the geometric truth source — never open frame.las_fn behind the edge's back.
enum class CloudTruthSource { Edge, FramePath };

inline CloudTruthSource resolve_cloud_truth_source(
    const std::map<std::string, cgraph::DataObject>& inputs,
    const char* cloud_key = "cloud", const char* frame_key = "frame") {
  const bool has_cloud = inputs.count(cloud_key) != 0;
  const bool has_frame = frame_key != nullptr && inputs.count(frame_key) != 0;
  if (has_cloud) {
    return CloudTruthSource::Edge;
  }
  if (has_frame) {
    return CloudTruthSource::FramePath;
  }
  throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                              "resolve_cloud_truth_source: missing cloud/frame");
}

inline pcl::PointCloud<pcl::PointXYZ>::Ptr load_xyz_cloud_prefer_edge(
    const std::map<std::string, cgraph::DataObject>& inputs, const char* op_id,
    const char* cloud_key = "cloud", const char* frame_key = "frame") {
  const CloudTruthSource src =
      resolve_cloud_truth_source(inputs, cloud_key, frame_key);
  if (src == CloudTruthSource::Edge) {
    return load_xyz_cloud(inputs.at(cloud_key), op_id);
  }
  const Ddx::LidarFrame frame = lidar_frame_from_data(inputs.at(frame_key));
  if (frame.lasFn_.empty()) {
    throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                std::string(op_id) + ": frame.las_fn empty");
  }
  cgraph::DataObject path_cloud = cloud_artifact_from_path(frame.lasFn_);
  return load_xyz_cloud(path_cloud, op_id);
}

inline cgraph::DataObject save_xyz_cloud_artifact(
    const pcl::PointCloud<pcl::PointXYZ>& cloud, const std::filesystem::path& out_path,
    const char* op_id) {
  std::filesystem::create_directories(out_path.parent_path());
  if (pcl::io::savePCDFileBinary(out_path.string(), cloud) != 0) {
    throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                std::string(op_id) + ": failed to write PCD");
  }
  return cloud_artifact_from_path(out_path);
}

inline cgraph::DataObject point_number_object(std::int64_t n) {
  return cgraph::make_data_object(cgraph::type_ids::integer(),
                                  cgraph::SemanticSpec::of("rtr.semantic.point_count"),
                                  cgraph::Payload::integer(n));
}

inline float param_float(const nlohmann::json& params, const char* key, float def) {
  if (params.is_object() && params.contains(key) && params[key].is_number()) {
    return params[key].get<float>();
  }
  return def;
}

inline int param_int(const nlohmann::json& params, const char* key, int def) {
  if (params.is_object() && params.contains(key) && params[key].is_number_integer()) {
    return params[key].get<int>();
  }
  return def;
}

inline bool param_bool(const nlohmann::json& params, const char* key, bool def) {
  if (params.is_object() && params.contains(key) && params[key].is_boolean()) {
    return params[key].get<bool>();
  }
  return def;
}

}  // namespace rtr
