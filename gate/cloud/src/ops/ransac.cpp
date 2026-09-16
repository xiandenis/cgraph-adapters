#include "cgraph/ops.hpp"
#include "cloud/cloud_value.hpp"
#include "cloud/rigid.hpp"
#include "cloud/usage_helpers.hpp"

#include <Eigen/Core>

#include <cmath>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace cloud {
namespace {

std::vector<float> xyz_of(const nlohmann::json& cloud) {
  require_cloud(cloud);
  std::vector<float> xyz;
  xyz.reserve(cloud["xyz"].size());
  for (const auto& v : cloud["xyz"]) {
    xyz.push_back(v.get<float>());
  }
  return xyz;
}

Eigen::Matrix3d R_of(const nlohmann::json& T) {
  Eigen::Matrix3d R;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      R(r, c) = T["R"][r][c].get<double>();
    }
  }
  return R;
}

Eigen::Vector3d t_of(const nlohmann::json& T) {
  return Eigen::Vector3d(T["t"][0].get<double>(), T["t"][1].get<double>(),
                         T["t"][2].get<double>());
}

class RansacOp final : public cgraph::MemoryOperator {
 public:
  RansacOp() {
    op_id_ = "cloud.ransac";
    signature_.inputs["src"] =
        cgraph::make_port("src", cgraph::PortKind::Value, "cloud");
    signature_.inputs["src"].formats = {"native"};
    signature_.inputs["tgt"] =
        cgraph::make_port("tgt", cgraph::PortKind::Value, "cloud");
    signature_.inputs["tgt"].formats = {"native"};
    signature_.inputs["corr"] =
        cgraph::make_port("corr", cgraph::PortKind::Value, "correspondence_set");
    signature_.inputs["corr"].formats = {"native"};
    signature_.outputs["T"] =
        cgraph::make_port("T", cgraph::PortKind::Value, "rigid_transform");
    signature_.outputs["T"].formats = {"native"};
    signature_.outputs["inliers"] =
        cgraph::make_port("inliers", cgraph::PortKind::Value, "correspondence_set");
    signature_.outputs["inliers"].formats = {"native"};
    cgraph::ParamSpec n_iter;
    n_iter.name = "n_iter";
    n_iter.dtype = "int";
    n_iter.default_value = 128;
    n_iter.invalidate = true;
    signature_.params["n_iter"] = std::move(n_iter);
    cgraph::ParamSpec thresh;
    thresh.name = "thresh";
    thresh.dtype = "float";
    thresh.default_value = 0.05;
    thresh.invalidate = true;
    signature_.params["thresh"] = std::move(thresh);
    cgraph::ParamSpec seed;
    seed.name = "seed";
    seed.dtype = "int";
    seed.default_value = nullptr;
    seed.invalidate = true;
    signature_.params["seed"] = std::move(seed);
    cgraph::ParamSpec vol;
    vol.name = "volatile";
    vol.dtype = "bool";
    vol.default_value = false;
    vol.invalidate = false;  // does not alone change recipe when false default
    vol.bindable = false;
    signature_.params["volatile"] = std::move(vol);
    capability_.summary = "RANSAC Kabsch pose (gate; requires seed or volatile)";
    cost_.cost_class = "cpu.n_iter";
    usage_.connect = "src/tgt xyz + corr → T + inliers.";
    usage_.tune = "Must set params.seed or params.volatile=true.";
    usage_.inspect = "inliers subset of corr under thresh.";
    usage_.for_whom = "有对应线，要从特征对应估计刚体 T";
    usage_.min_graph = cloud::usage::make_min_graph_node(
        "ransac", "cloud.ransac", {{"seed", 1}, {"n_iter", 128}, {"thresh", 0.05}},
        {{"src", cgraph::make_port("src", cgraph::PortKind::Value, "cloud")},
         {"tgt", cgraph::make_port("tgt", cgraph::PortKind::Value, "cloud")},
         {"corr", cgraph::make_port("corr", cgraph::PortKind::Value, "correspondence_set")}},
        {{"T", cgraph::make_port("T", cgraph::PortKind::Value, "rigid_transform")},
         {"inliers", cgraph::make_port("inliers", cgraph::PortKind::Value, "correspondence_set")}});
    effect_.effect = cgraph::EffectClass::Stochastic;
    effect_.cache = cgraph::CachePolicy::Memoizable;
    effect_.seed_param = "seed";
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs, const nlohmann::json& params,
      const cgraph::ExecContext&) const override {
    const auto sit = inputs.find("src");
    const auto tit = inputs.find("tgt");
    const auto cit = inputs.find("corr");
    if (sit == inputs.end() || tit == inputs.end() || cit == inputs.end()) {
      throw std::invalid_argument("cloud.ransac: missing inputs");
    }
    require_correspondences(cit->second);
    auto src = xyz_of(sit->second);
    auto tgt = xyz_of(tit->second);
    std::vector<std::pair<int, int>> pairs;
    for (const auto& p : cit->second["pairs"]) {
      pairs.emplace_back(p[0].get<int>(), p[1].get<int>());
    }
    if (pairs.size() < 3) {
      return {{"T", make_identity_rigid()}, {"inliers", make_correspondences({})}};
    }

    int n_iter = 128;
    double thresh = 0.05;
    std::uint32_t seed = 0;
    bool have_seed = false;
    if (params.is_object()) {
      if (params.contains("n_iter") && params["n_iter"].is_number_integer()) {
        n_iter = params["n_iter"].get<int>();
      }
      if (params.contains("thresh") && params["thresh"].is_number()) {
        thresh = params["thresh"].get<double>();
      }
      if (params.contains("seed") && params["seed"].is_number_integer()) {
        seed = static_cast<std::uint32_t>(params["seed"].get<std::int64_t>());
        have_seed = true;
      }
    }
    if (!have_seed) {
      // validate should reject; execute still needs deterministic fallback
      seed = 1;
    }
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> uni(0, static_cast<int>(pairs.size()) - 1);
    const double thresh2 = thresh * thresh;

    nlohmann::json best_T = make_identity_rigid();
    std::vector<std::pair<int, int>> best_inliers;
    for (int it = 0; it < n_iter; ++it) {
      int i0 = uni(rng), i1 = uni(rng), i2 = uni(rng);
      if (i0 == i1 || i0 == i2 || i1 == i2) {
        continue;
      }
      std::vector<float> Ps, Qt;
      for (int idx : {i0, i1, i2}) {
        const auto& pr = pairs[static_cast<std::size_t>(idx)];
        const int si = pr.first;
        const int tj = pr.second;
        Ps.push_back(src[3 * si]);
        Ps.push_back(src[3 * si + 1]);
        Ps.push_back(src[3 * si + 2]);
        Qt.push_back(tgt[3 * tj]);
        Qt.push_back(tgt[3 * tj + 1]);
        Qt.push_back(tgt[3 * tj + 2]);
      }
      nlohmann::json T;
      try {
        T = kabsch_rigid(Ps, Qt);
      } catch (...) {
        continue;
      }
      const Eigen::Matrix3d R = R_of(T);
      const Eigen::Vector3d t = t_of(T);
      std::vector<std::pair<int, int>> inliers;
      for (const auto& pr : pairs) {
        Eigen::Vector3d p(src[3 * pr.first], src[3 * pr.first + 1],
                          src[3 * pr.first + 2]);
        Eigen::Vector3d q(tgt[3 * pr.second], tgt[3 * pr.second + 1],
                          tgt[3 * pr.second + 2]);
        Eigen::Vector3d tp = R * p + t;
        if ((tp - q).squaredNorm() <= thresh2) {
          inliers.push_back(pr);
        }
      }
      if (inliers.size() > best_inliers.size()) {
        best_inliers = inliers;
        best_T = T;
      }
    }
    // Final expand + refine
    if (best_inliers.size() >= 3) {
      const Eigen::Matrix3d R = R_of(best_T);
      const Eigen::Vector3d t = t_of(best_T);
      const double expand2 = (thresh * 2.0) * (thresh * 2.0);
      std::vector<std::pair<int, int>> expanded;
      for (const auto& pr : pairs) {
        Eigen::Vector3d p(src[3 * pr.first], src[3 * pr.first + 1],
                          src[3 * pr.first + 2]);
        Eigen::Vector3d q(tgt[3 * pr.second], tgt[3 * pr.second + 1],
                          tgt[3 * pr.second + 2]);
        if ((R * p + t - q).squaredNorm() <= expand2) {
          expanded.push_back(pr);
        }
      }
      if (expanded.size() >= 3) {
        std::vector<float> Ps2, Qt2;
        for (const auto& pr : expanded) {
          Ps2.push_back(src[3 * pr.first]);
          Ps2.push_back(src[3 * pr.first + 1]);
          Ps2.push_back(src[3 * pr.first + 2]);
          Qt2.push_back(tgt[3 * pr.second]);
          Qt2.push_back(tgt[3 * pr.second + 1]);
          Qt2.push_back(tgt[3 * pr.second + 2]);
        }
        try {
          best_T = kabsch_rigid(Ps2, Qt2);
          best_inliers = expanded;
        } catch (...) {
        }
      }
    }
    return {{"T", best_T}, {"inliers", make_correspondences(best_inliers)}};
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_cloud_ransac() {
  return std::make_shared<RansacOp>();
}

}  // namespace cloud
