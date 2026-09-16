#include "cgraph/ops.hpp"
#include "cloud/cloud_value.hpp"
#include "cloud/rigid.hpp"
#include "cloud/usage_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace cloud {
namespace {

std::vector<float> feat_row(const nlohmann::json& cloud, std::size_t i, int dim) {
  std::vector<float> row(static_cast<std::size_t>(dim));
  const auto& feat = cloud["feat"];
  for (int d = 0; d < dim; ++d) {
    row[static_cast<std::size_t>(d)] =
        feat[i * static_cast<std::size_t>(dim) + static_cast<std::size_t>(d)].get<float>();
  }
  return row;
}

float l2(const std::vector<float>& a, const std::vector<float>& b) {
  float s = 0.0f;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const float d = a[i] - b[i];
    s += d * d;
  }
  return s;
}

class MatchOp final : public cgraph::MemoryOperator {
 public:
  MatchOp() {
    op_id_ = "cloud.match";
    signature_.inputs["src"] =
        cgraph::make_port("src", cgraph::PortKind::Value, "cloud");
    signature_.inputs["src"].formats = {"native"};
    signature_.inputs["src"].schema = {{"feat", 33}};
    signature_.inputs["tgt"] =
        cgraph::make_port("tgt", cgraph::PortKind::Value, "cloud");
    signature_.inputs["tgt"].formats = {"native"};
    signature_.inputs["tgt"].schema = {{"feat", 33}};
    signature_.outputs["corr"] =
        cgraph::make_port("corr", cgraph::PortKind::Value, "correspondence_set");
    signature_.outputs["corr"].formats = {"native"};
    cgraph::ParamSpec k;
    k.name = "k";
    k.dtype = "int";
    k.default_value = 1;
    k.invalidate = true;
    signature_.params["k"] = std::move(k);
    cgraph::ParamSpec ratio;
    ratio.name = "ratio";
    ratio.dtype = "float";
    ratio.default_value = 0.9;
    ratio.invalidate = true;
    signature_.params["ratio"] = std::move(ratio);
    capability_.summary = "Feature mutual NN match (gate)";
    cost_.cost_class = "cpu.n_m";
    usage_.connect = "Wire two cloud{feat[33]} → correspondence_set.";
    usage_.tune = "params.k (kept 1); params.ratio Lowe-style second-NN.";
    usage_.inspect = "corr.pairs list of [i,j].";
    usage_.for_whom = "已有两侧 FPFH 特征，要做描述子对应";
    usage_.min_graph = cloud::usage::make_min_graph_node(
        "match", "cloud.match", {{"k", 1}, {"ratio", 0.9}},
        {{"src", cgraph::make_port("src", cgraph::PortKind::Value, "cloud")},
         {"tgt", cgraph::make_port("tgt", cgraph::PortKind::Value, "cloud")}},
        {{"corr", cgraph::make_port("corr", cgraph::PortKind::Value, "correspondence_set")}});
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs, const nlohmann::json& params,
      const cgraph::ExecContext&) const override {
    const auto sit = inputs.find("src");
    const auto tit = inputs.find("tgt");
    if (sit == inputs.end() || tit == inputs.end()) {
      throw std::invalid_argument("cloud.match: missing src/tgt");
    }
    require_cloud(sit->second);
    require_cloud(tit->second);
    if (!sit->second.contains("feat") || !tit->second.contains("feat")) {
      throw std::invalid_argument("cloud.match: missing feat");
    }
    const int dim = sit->second["feat_dim"].get<int>();
    const std::size_t ns = cloud_n(sit->second);
    const std::size_t nt = cloud_n(tit->second);
    double ratio = 0.9;
    if (params.is_object() && params.contains("ratio") && params["ratio"].is_number()) {
      ratio = params["ratio"].get<double>();
    }

    std::vector<int> src_to_tgt(ns, -1);
    std::vector<float> src_best(ns, 0.0f);
    for (std::size_t i = 0; i < ns; ++i) {
      auto fi = feat_row(sit->second, i, dim);
      float best = std::numeric_limits<float>::infinity();
      float second = std::numeric_limits<float>::infinity();
      int best_j = -1;
      for (std::size_t j = 0; j < nt; ++j) {
        auto fj = feat_row(tit->second, j, dim);
        const float d = l2(fi, fj);
        if (d < best) {
          second = best;
          best = d;
          best_j = static_cast<int>(j);
        } else if (d < second) {
          second = d;
        }
      }
      if (best_j >= 0) {
        const bool pass_ratio =
            ratio >= 1.0 || second == std::numeric_limits<float>::infinity() ||
            best <= ratio * ratio * second;
        if (pass_ratio) {
          src_to_tgt[i] = best_j;
          src_best[i] = best;
        }
      }
    }

    std::vector<int> tgt_to_src(nt, -1);
    for (std::size_t j = 0; j < nt; ++j) {
      auto fj = feat_row(tit->second, j, dim);
      float best = std::numeric_limits<float>::infinity();
      int best_i = -1;
      for (std::size_t i = 0; i < ns; ++i) {
        auto fi = feat_row(sit->second, i, dim);
        const float d = l2(fi, fj);
        if (d < best) {
          best = d;
          best_i = static_cast<int>(i);
        }
      }
      tgt_to_src[j] = best_i;
    }

    std::vector<std::pair<int, int>> pairs;
    for (std::size_t i = 0; i < ns; ++i) {
      const int j = src_to_tgt[i];
      if (j < 0) {
        continue;
      }
      // Prefer mutual; fall back to one-way if mutual empty later.
      if (tgt_to_src[static_cast<std::size_t>(j)] == static_cast<int>(i)) {
        pairs.emplace_back(static_cast<int>(i), j);
      }
    }
    if (pairs.size() < 3) {
      pairs.clear();
      for (std::size_t i = 0; i < ns; ++i) {
        const int j = src_to_tgt[i];
        if (j >= 0) {
          pairs.emplace_back(static_cast<int>(i), j);
        }
      }
    }
    return {{"corr", make_correspondences(pairs)}};
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_cloud_match() {
  return std::make_shared<MatchOp>();
}

}  // namespace cloud
