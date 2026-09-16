#pragma once

#include <cstddef>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace cloud {

nlohmann::json make_xyz_cloud(const std::vector<float>& xyz_flat);
void require_cloud(const nlohmann::json& value);
std::size_t cloud_n(const nlohmann::json& value);

nlohmann::json load_ply_ascii(const std::filesystem::path& path);
nlohmann::json load_pcd_ascii(const std::filesystem::path& path);
void dump_ply_ascii(const nlohmann::json& cloud, const std::filesystem::path& path);
void dump_pcd_ascii(const nlohmann::json& cloud, const std::filesystem::path& path);

}  // namespace cloud
