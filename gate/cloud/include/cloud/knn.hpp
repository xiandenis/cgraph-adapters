#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace cloud {

/** Brute-force kNN on flat xyz. Returns for each point the k neighbor indices (excluding self when possible). */
std::vector<std::vector<int>> knn_indices(const std::vector<float>& xyz, int k);

/** Radius neighbors (inclusive of self). */
std::vector<std::vector<int>> radius_indices(const std::vector<float>& xyz, float radius);

}  // namespace cloud
