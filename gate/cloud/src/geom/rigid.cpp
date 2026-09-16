#include "cloud/rigid.hpp"

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>

#include <stdexcept>

namespace cloud {
namespace {

nlohmann::json mat3_to_json(const Eigen::Matrix3d& m) {
  nlohmann::json R = nlohmann::json::array();
  for (int r = 0; r < 3; ++r) {
    nlohmann::json row = nlohmann::json::array();
    for (int c = 0; c < 3; ++c) {
      row.push_back(m(r, c));
    }
    R.push_back(std::move(row));
  }
  return R;
}

}  // namespace

nlohmann::json make_identity_rigid() {
  return make_rigid({1, 0, 0, 0, 1, 0, 0, 0, 1}, {0, 0, 0});
}

nlohmann::json make_rigid(const std::vector<double>& R9, const std::vector<double>& t3) {
  if (R9.size() != 9 || t3.size() != 3) {
    throw std::invalid_argument("make_rigid: R9 size 9 and t3 size 3 required");
  }
  nlohmann::json out;
  out["R"] = nlohmann::json::array(
      {nlohmann::json::array({R9[0], R9[1], R9[2]}),
       nlohmann::json::array({R9[3], R9[4], R9[5]}),
       nlohmann::json::array({R9[6], R9[7], R9[8]})});
  out["t"] = nlohmann::json::array({t3[0], t3[1], t3[2]});
  return out;
}

void require_rigid(const nlohmann::json& value) {
  if (!value.is_object() || !value.contains("R") || !value.contains("t")) {
    throw std::invalid_argument("rigid_transform: need object with R,t");
  }
  if (!value["R"].is_array() || value["R"].size() != 3) {
    throw std::invalid_argument("rigid_transform: R must be 3x3");
  }
  for (const auto& row : value["R"]) {
    if (!row.is_array() || row.size() != 3) {
      throw std::invalid_argument("rigid_transform: R must be 3x3");
    }
  }
  if (!value["t"].is_array() || value["t"].size() != 3) {
    throw std::invalid_argument("rigid_transform: t must be length 3");
  }
}

nlohmann::json make_correspondences(const std::vector<std::pair<int, int>>& pairs) {
  nlohmann::json out;
  out["pairs"] = nlohmann::json::array();
  for (const auto& p : pairs) {
    out["pairs"].push_back(nlohmann::json::array({p.first, p.second}));
  }
  return out;
}

void require_correspondences(const nlohmann::json& value) {
  if (!value.is_object() || !value.contains("pairs") || !value["pairs"].is_array()) {
    throw std::invalid_argument("correspondence_set: need object with pairs[]");
  }
}

nlohmann::json kabsch_rigid(const std::vector<float>& src_xyz,
                            const std::vector<float>& tgt_xyz) {
  if (src_xyz.size() != tgt_xyz.size() || src_xyz.size() % 3 != 0) {
    throw std::invalid_argument("kabsch_rigid: src/tgt flat xyz length mismatch");
  }
  const Eigen::Index n = static_cast<Eigen::Index>(src_xyz.size() / 3);
  if (n < 3) {
    throw std::invalid_argument("kabsch_rigid: need at least 3 points");
  }

  Eigen::MatrixXd P(3, n);
  Eigen::MatrixXd Q(3, n);
  for (Eigen::Index i = 0; i < n; ++i) {
    P(0, i) = src_xyz[static_cast<std::size_t>(3 * i)];
    P(1, i) = src_xyz[static_cast<std::size_t>(3 * i + 1)];
    P(2, i) = src_xyz[static_cast<std::size_t>(3 * i + 2)];
    Q(0, i) = tgt_xyz[static_cast<std::size_t>(3 * i)];
    Q(1, i) = tgt_xyz[static_cast<std::size_t>(3 * i + 1)];
    Q(2, i) = tgt_xyz[static_cast<std::size_t>(3 * i + 2)];
  }

  const Eigen::Vector3d mu_p = P.rowwise().mean();
  const Eigen::Vector3d mu_q = Q.rowwise().mean();
  Eigen::MatrixXd X = P.colwise() - mu_p;
  Eigen::MatrixXd Y = Q.colwise() - mu_q;
  const Eigen::Matrix3d H = X * Y.transpose();
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d R = svd.matrixV() * svd.matrixU().transpose();
  if (R.determinant() < 0.0) {
    Eigen::Matrix3d V = svd.matrixV();
    V.col(2) *= -1.0;
    R = V * svd.matrixU().transpose();
  }
  const Eigen::Vector3d t = mu_q - R * mu_p;

  nlohmann::json out;
  out["R"] = mat3_to_json(R);
  out["t"] = nlohmann::json::array({t(0), t(1), t(2)});
  return out;
}

}  // namespace cloud
