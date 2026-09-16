#include "cloud/fpfh_core.hpp"

#include "cgraph/ops.hpp"
#include "cloud/cloud_value.hpp"
#include "cloud/knn.hpp"
#include "cloud/usage_helpers.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace cloud {
namespace {

constexpr int kHistBins = 11;
constexpr int kFeatDim = 33;

std::vector<float> json_floats(const nlohmann::json& arr) {
  std::vector<float> out;
  out.reserve(arr.size());
  for (const auto& v : arr) {
    out.push_back(v.get<float>());
  }
  return out;
}

void pair_features(const Eigen::Vector3d& p, const Eigen::Vector3d& n_p,
                   const Eigen::Vector3d& q, const Eigen::Vector3d& n_q, float* f123) {
  Eigen::Vector3d d = q - p;
  const double dn = d.norm();
  if (dn < 1e-12) {
    f123[0] = f123[1] = f123[2] = 0.5f;
    return;
  }
  d /= dn;
  Eigen::Vector3d u = n_p;
  Eigen::Vector3d v = d.cross(u);
  const double vn = v.norm();
  if (vn < 1e-12) {
    f123[0] = f123[1] = f123[2] = 0.5f;
    return;
  }
  v /= vn;
  Eigen::Vector3d w = u.cross(v);
  const double f1 = v.dot(n_q);
  const double f2 = u.dot(d);
  const double f3 = std::atan2(w.dot(n_q), u.dot(n_q));
  f123[0] = static_cast<float>((f1 + 1.0) * 0.5);
  f123[1] = static_cast<float>((f2 + 1.0) * 0.5);
  f123[2] = static_cast<float>((f3 + 3.141592653589793) / (2.0 * 3.141592653589793));
}

void accumulate_spfh(const std::vector<float>& xyz, const std::vector<float>& normals,
                     int i, const std::vector<int>& neigh, float* hist33) {
  std::fill(hist33, hist33 + kFeatDim, 0.0f);
  if (neigh.size() <= 1) {
    return;
  }
  Eigen::Vector3d p(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]);
  Eigen::Vector3d n_p(normals[3 * i], normals[3 * i + 1], normals[3 * i + 2]);
  int count = 0;
  for (int j : neigh) {
    if (j == i) {
      continue;
    }
    Eigen::Vector3d q(xyz[3 * j], xyz[3 * j + 1], xyz[3 * j + 2]);
    Eigen::Vector3d n_q(normals[3 * j], normals[3 * j + 1], normals[3 * j + 2]);
    float f[3];
    pair_features(p, n_p, q, n_q, f);
    for (int c = 0; c < 3; ++c) {
      int bin = static_cast<int>(std::floor(f[c] * kHistBins));
      if (bin < 0) {
        bin = 0;
      }
      if (bin >= kHistBins) {
        bin = kHistBins - 1;
      }
      hist33[c * kHistBins + bin] += 1.0f;
    }
    ++count;
  }
  if (count > 0) {
    const float inv = 1.0f / static_cast<float>(count);
    for (int b = 0; b < kFeatDim; ++b) {
      hist33[b] *= inv;
    }
  }
}

void fill_fpfh_signature(cgraph::Signature& signature) {
  signature.inputs["in"] = cgraph::make_port("in", cgraph::PortKind::Value, "cloud");
  signature.inputs["in"].formats = {"native"};
  signature.inputs["in"].schema = {{"require", nlohmann::json::array({"xyz", "normal"})}};
  signature.outputs["out"] = cgraph::make_port("out", cgraph::PortKind::Value, "cloud");
  signature.outputs["out"].formats = {"native"};
  signature.outputs["out"].schema = {
      {"fields", nlohmann::json::array({"xyz", "normal", "feat"})}, {"feat", 33}};
  cgraph::ParamSpec radius_f;
  radius_f.name = "radius_f";
  radius_f.dtype = "float";
  radius_f.default_value = 0.25;
  radius_f.invalidate = true;
  signature.params["radius_f"] = std::move(radius_f);
}

}  // namespace

nlohmann::json compute_fpfh(const nlohmann::json& cloud_with_normals, double radius_f) {
  require_cloud(cloud_with_normals);
  if (!cloud_with_normals.contains("normal")) {
    throw std::invalid_argument("compute_fpfh: missing normal");
  }
  auto xyz = json_floats(cloud_with_normals["xyz"]);
  auto normals = json_floats(cloud_with_normals["normal"]);
  const std::size_t n = xyz.size() / 3;
  auto neigh = radius_indices(xyz, static_cast<float>(radius_f));
  std::vector<std::vector<float>> spfh(n, std::vector<float>(kFeatDim, 0.0f));
  for (std::size_t i = 0; i < n; ++i) {
    accumulate_spfh(xyz, normals, static_cast<int>(i), neigh[i], spfh[i].data());
  }
  std::vector<float> feat(n * kFeatDim, 0.0f);
  for (std::size_t i = 0; i < n; ++i) {
    float* out = feat.data() + i * kFeatDim;
    for (int b = 0; b < kFeatDim; ++b) {
      out[b] = spfh[i][static_cast<std::size_t>(b)];
    }
    float wsum = 1.0f;
    Eigen::Vector3d p(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]);
    for (int j : neigh[i]) {
      if (j == static_cast<int>(i)) {
        continue;
      }
      Eigen::Vector3d q(xyz[3 * j], xyz[3 * j + 1], xyz[3 * j + 2]);
      const float dist = static_cast<float>((q - p).norm());
      const float w = 1.0f / std::max(dist, 1e-6f);
      for (int b = 0; b < kFeatDim; ++b) {
        out[b] += w * spfh[static_cast<std::size_t>(j)][static_cast<std::size_t>(b)];
      }
      wsum += w;
    }
    const float inv = 1.0f / wsum;
    for (int b = 0; b < kFeatDim; ++b) {
      out[b] *= inv;
    }
  }
  nlohmann::json out = cloud_with_normals;
  out["feat"] = feat;
  out["feat_dim"] = kFeatDim;
  out["fields"] = nlohmann::json::array({"xyz", "normal", "feat"});
  return out;
}

namespace {

class FpfhOp final : public cgraph::MemoryOperator {
 public:
  FpfhOp() {
    op_id_ = "cloud.fpfh";
    fill_fpfh_signature(signature_);
    capability_.summary = "FPFH feat[33] gate Memory (Eigen)";
    capability_.tags = {"feature", "fpfh"};
    cost_.cost_class = "cpu.n_k";
    usage_.connect = "Require cloud{xyz,normal}; out has feat_dim=33.";
    usage_.tune = "params.radius_f neighborhood radius.";
    usage_.inspect = "feat length = 33 * n.";
    usage_.for_whom = "要做粗配准特征，听说过 FPFH 但没读 PCL 手册";
    usage_.connect_bullets = {
        "输入云必须已有 normal（没有就先插 NormalEst）",
        "输出带 feat[33]，接到 Match，不要接到 Pose",
    };
    usage_.set_bullets = {
        "radius_f 必须大于法向半径；常用 radius_f ≈ 5 × radius_n",
    };
    usage_.look_bullets = {
        "口上出现 feat[33]；看下游对应线是否成片",
    };
    usage_.typically_after.push_back(cgraph::UsagePlaceHint{
        "normal", "FPFH 需要法向", "cloud.normal_est"});
    usage_.instead_use.push_back(cgraph::UsageInstead{
        "algo.coarse_reg.fpfh_ransac.v1", "需要刚体 T 时用 CoarseReg/Pose，不要直连 FPFH 出口"});
    usage_.instead_use.push_back(
        cgraph::UsageInstead{"cloud.ransac", "已有对应时估 T，用 RANSAC/Pose 而不是 FPFH"});
    usage_.first_values.push_back(
        cgraph::UsageFirstValue{"radius_f", 0.25, "约 5×radius_n；单位与点云相同"});
    cloud::usage::set_tune(usage_, {
        {"对应线乱、内外点一半", "radius_f", "略增 radius_f 或检查法向"},
        {"极慢", "radius_f", "上游加大 leaf，不要无限增大 radius_f"},
        {"缺 normal 字段", "—", "先接 cloud.normal_est"},
    });
    usage_.min_graph = cloud::usage::make_min_graph_node(
        "normal_est", "cloud.normal_est", {{"k", 16}},
        {{"in", cloud::usage::cloud_in_xyz()}},
        {{"out", cloud::usage::cloud_out_xyz_normal()}});
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs, const nlohmann::json& params,
      const cgraph::ExecContext&) const override {
    const auto it = inputs.find("in");
    if (it == inputs.end()) {
      throw std::invalid_argument("cloud.fpfh: missing in");
    }
    double radius_f = 0.25;
    if (params.is_object() && params.contains("radius_f") &&
        params["radius_f"].is_number()) {
      radius_f = params["radius_f"].get<double>();
    }
    return {{"out", compute_fpfh(it->second, radius_f)}};
  }
};

class FpfhProcOp final : public cgraph::ProcessOperator {
 public:
  FpfhProcOp() {
    op_id_ = "cloud.fpfh_proc";
    fill_fpfh_signature(signature_);
    // Same algorithm fingerprint family as Memory for gate isomorphism demos.
    impl_fingerprint_ = "fpfh-gate-v1";
    capability_.summary = "FPFH feat[33] gate fake Process (same compute_fpfh)";
    cost_.cost_class = "cpu.n_k";
    usage_.connect = "Same ports as cloud.fpfh; Process kind for Memory↔Process swap.";
    usage_.tune = "params.radius_f; no Open3D.";
    usage_.inspect = "Output digest matches cloud.fpfh for same inputs.";
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs, const nlohmann::json& params,
      const cgraph::ExecContext& ctx) const override {
    if (ctx.sandbox == nullptr) {
      throw std::invalid_argument("cloud.fpfh_proc: sandbox required");
    }
    const auto it = inputs.find("in");
    if (it == inputs.end()) {
      throw std::invalid_argument("cloud.fpfh_proc: missing in");
    }
    double radius_f = 0.25;
    if (params.is_object() && params.contains("radius_f") &&
        params["radius_f"].is_number()) {
      radius_f = params["radius_f"].get<double>();
    }
    return {{"out", compute_fpfh(it->second, radius_f)}};
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_cloud_fpfh() {
  auto op = std::make_shared<FpfhOp>();
  op->set_impl_fingerprint("fpfh-gate-v1");
  return op;
}

std::shared_ptr<cgraph::MemoryOperator> make_cloud_fpfh_proc() {
  return std::make_shared<FpfhProcOp>();
}

}  // namespace cloud
