#pragma once

#include <Eigen/Core>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

inline void write_ascii_xyz_pcd(const std::filesystem::path& path,
                                const std::vector<Eigen::Vector3f>& pts) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("write_ascii_xyz_pcd: open failed");
  }
  out << "# .PCD v0.7 - Point Cloud Data file format\n"
      << "VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
      << "WIDTH " << pts.size() << "\nHEIGHT 1\n"
      << "VIEWPOINT 0 0 0 1 0 0 0\nPOINTS " << pts.size() << "\nDATA ascii\n";
  for (const auto& p : pts) {
    out << p.x() << " " << p.y() << " " << p.z() << "\n";
  }
}

inline std::vector<Eigen::Vector3f> make_asymmetric_station(float step) {
  std::vector<Eigen::Vector3f> pts;
  // Floor 2 m x 1 m (asymmetric in XY).
  for (float x = 0; x <= 2.0f; x += step) {
    for (float y = 0; y <= 1.0f; y += step) {
      pts.emplace_back(x, y, 0.f);
    }
  }
  // Tall unique wall at x=0 (1.6 m), not a closed cube.
  for (float y = 0; y <= 1.0f; y += step) {
    for (float z = 0; z <= 1.6f; z += step) {
      pts.emplace_back(0.f, y, z);
    }
  }
  // Offset pillar so the pose is unique.
  for (float z = 0; z <= 0.8f; z += step) {
    pts.emplace_back(1.7f, 0.2f, z);
    pts.emplace_back(1.7f, 0.25f, z);
    pts.emplace_back(1.75f, 0.2f, z);
  }
  return pts;
}
