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
#include <vector>

namespace rtr {

inline constexpr const char* kPointXyzF32Codec = "rtr.codec.point_xyz_f32";

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

inline std::filesystem::path resolve_work_dir(const nlohmann::json& params,
                                              const std::filesystem::path& src,
                                              const char* default_suffix) {
  std::string work;
  if (params.is_object() && params.contains("work_dir") && params["work_dir"].is_string()) {
    work = params["work_dir"].get<std::string>();
  }
  if (work.empty()) {
    work = (src.parent_path() / (std::string(default_suffix) + src.stem().string())).string();
  }
  std::filesystem::create_directories(work);
  return work;
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
