#include "cloud/knn.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace cloud {

std::vector<std::vector<int>> knn_indices(const std::vector<float>& xyz, int k) {
  const std::size_t n = xyz.size() / 3;
  std::vector<std::vector<int>> out(n);
  if (n == 0 || k <= 0) {
    return out;
  }
  const int kk = std::min(k, static_cast<int>(n > 1 ? n - 1 : 1));
  for (std::size_t i = 0; i < n; ++i) {
    std::vector<std::pair<float, int>> dist;
    dist.reserve(n);
    const float xi = xyz[3 * i];
    const float yi = xyz[3 * i + 1];
    const float zi = xyz[3 * i + 2];
    for (std::size_t j = 0; j < n; ++j) {
      if (j == i) {
        continue;
      }
      const float dx = xyz[3 * j] - xi;
      const float dy = xyz[3 * j + 1] - yi;
      const float dz = xyz[3 * j + 2] - zi;
      dist.emplace_back(dx * dx + dy * dy + dz * dz, static_cast<int>(j));
    }
    const int take = std::min(kk, static_cast<int>(dist.size()));
    std::partial_sort(dist.begin(), dist.begin() + take, dist.end());
    out[i].reserve(static_cast<std::size_t>(take));
    for (int t = 0; t < take; ++t) {
      out[i].push_back(dist[static_cast<std::size_t>(t)].second);
    }
  }
  return out;
}

std::vector<std::vector<int>> radius_indices(const std::vector<float>& xyz, float radius) {
  const std::size_t n = xyz.size() / 3;
  std::vector<std::vector<int>> out(n);
  if (n == 0 || radius <= 0.0f) {
    return out;
  }
  const float r2 = radius * radius;
  for (std::size_t i = 0; i < n; ++i) {
    const float xi = xyz[3 * i];
    const float yi = xyz[3 * i + 1];
    const float zi = xyz[3 * i + 2];
    for (std::size_t j = 0; j < n; ++j) {
      const float dx = xyz[3 * j] - xi;
      const float dy = xyz[3 * j + 1] - yi;
      const float dz = xyz[3 * j + 2] - zi;
      if (dx * dx + dy * dy + dz * dz <= r2) {
        out[i].push_back(static_cast<int>(j));
      }
    }
  }
  return out;
}

}  // namespace cloud
