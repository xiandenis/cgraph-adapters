#include "cloud/cloud_value.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cloud {
namespace {

std::vector<float> read_xyz_from_cloud(const nlohmann::json& cloud) {
  require_cloud(cloud);
  std::vector<float> xyz;
  xyz.reserve(cloud["xyz"].size());
  for (const auto& el : cloud["xyz"]) {
    xyz.push_back(el.get<float>());
  }
  return xyz;
}

nlohmann::json cloud_from_xyz_rows(const std::vector<float>& xyz) {
  return make_xyz_cloud(xyz);
}

std::string trim(std::string s) {
  while (!s.empty() && (s.back() == '\r' || s.back() == ' ' || s.back() == '\t')) {
    s.pop_back();
  }
  std::size_t i = 0;
  while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) {
    ++i;
  }
  return s.substr(i);
}

}  // namespace

void dump_ply_ascii(const nlohmann::json& cloud, const std::filesystem::path& path) {
  const auto xyz = read_xyz_from_cloud(cloud);
  const std::size_t n = xyz.size() / 3;
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("cloud: cannot write " + path.string());
  }
  out << "ply\n"
      << "format ascii 1.0\n"
      << "element vertex " << n << "\n"
      << "property float x\n"
      << "property float y\n"
      << "property float z\n"
      << "end_header\n";
  out.setf(std::ios::fixed, std::ios::floatfield);
  out.precision(9);
  for (std::size_t i = 0; i < n; ++i) {
    out << xyz[3 * i] << ' ' << xyz[3 * i + 1] << ' ' << xyz[3 * i + 2] << '\n';
  }
}

nlohmann::json load_ply_ascii(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cloud: cannot read " + path.string());
  }
  std::string line;
  if (!std::getline(in, line) || trim(line) != "ply") {
    throw std::invalid_argument("cloud: not a ply file");
  }
  std::size_t n = 0;
  bool header_done = false;
  while (std::getline(in, line)) {
    line = trim(line);
    if (line.rfind("element vertex ", 0) == 0) {
      n = static_cast<std::size_t>(std::stoull(line.substr(15)));
    } else if (line == "end_header") {
      header_done = true;
      break;
    }
  }
  if (!header_done) {
    throw std::invalid_argument("cloud: ply missing end_header");
  }
  std::vector<float> xyz;
  xyz.reserve(n * 3);
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::getline(in, line)) {
      throw std::invalid_argument("cloud: ply truncated");
    }
    std::istringstream ss(line);
    float x = 0, y = 0, z = 0;
    if (!(ss >> x >> y >> z)) {
      throw std::invalid_argument("cloud: ply bad vertex row");
    }
    xyz.push_back(x);
    xyz.push_back(y);
    xyz.push_back(z);
  }
  return cloud_from_xyz_rows(xyz);
}

void dump_pcd_ascii(const nlohmann::json& cloud, const std::filesystem::path& path) {
  const auto xyz = read_xyz_from_cloud(cloud);
  const std::size_t n = xyz.size() / 3;
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("cloud: cannot write " + path.string());
  }
  out << "# .PCD v0.7\n"
      << "VERSION 0.7\n"
      << "FIELDS x y z\n"
      << "SIZE 4 4 4\n"
      << "TYPE F F F\n"
      << "COUNT 1 1 1\n"
      << "WIDTH " << n << "\n"
      << "HEIGHT 1\n"
      << "VIEWPOINT 0 0 0 1 0 0 0\n"
      << "POINTS " << n << "\n"
      << "DATA ascii\n";
  out.setf(std::ios::fixed, std::ios::floatfield);
  out.precision(9);
  for (std::size_t i = 0; i < n; ++i) {
    out << xyz[3 * i] << ' ' << xyz[3 * i + 1] << ' ' << xyz[3 * i + 2] << '\n';
  }
}

nlohmann::json load_pcd_ascii(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cloud: cannot read " + path.string());
  }
  std::string line;
  std::size_t n = 0;
  bool data = false;
  while (std::getline(in, line)) {
    line = trim(line);
    if (line.empty() || line[0] == '#') {
      continue;
    }
    if (line.rfind("POINTS ", 0) == 0) {
      n = static_cast<std::size_t>(std::stoull(line.substr(7)));
    } else if (line.rfind("DATA ", 0) == 0) {
      if (line.substr(5) != "ascii") {
        throw std::invalid_argument("cloud: only ascii pcd supported");
      }
      data = true;
      break;
    }
  }
  if (!data) {
    throw std::invalid_argument("cloud: pcd missing DATA ascii");
  }
  std::vector<float> xyz;
  xyz.reserve(n * 3);
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::getline(in, line)) {
      throw std::invalid_argument("cloud: pcd truncated");
    }
    std::istringstream ss(line);
    float x = 0, y = 0, z = 0;
    if (!(ss >> x >> y >> z)) {
      throw std::invalid_argument("cloud: pcd bad point row");
    }
    xyz.push_back(x);
    xyz.push_back(y);
    xyz.push_back(z);
  }
  return cloud_from_xyz_rows(xyz);
}

}  // namespace cloud
