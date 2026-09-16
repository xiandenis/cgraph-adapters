#include "cgraph/ops.hpp"
#include "cloud/cloud_value.hpp"
#include "cloud/knn.hpp"
#include "cloud/usage_helpers.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cloud {
namespace {

class StatOutlierOp final : public cgraph::MemoryOperator {
 public:
  StatOutlierOp() {
    op_id_ = "cloud.stat_outlier";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, "cloud");
    signature_.inputs["in"].formats = {"native"};
    signature_.inputs["in"].schema = {{"fields", nlohmann::json::array({"xyz"})}};
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, "cloud");
    signature_.outputs["out"].formats = {"native"};
    signature_.outputs["out"].schema = {{"fields", nlohmann::json::array({"xyz"})}};
    cgraph::ParamSpec mean_k;
    mean_k.name = "mean_k";
    mean_k.dtype = "int";
    mean_k.default_value = 8;
    mean_k.invalidate = true;
    signature_.params["mean_k"] = std::move(mean_k);
    cgraph::ParamSpec std_mul;
    std_mul.name = "std_mul";
    std_mul.dtype = "float";
    std_mul.default_value = 1.0;
    std_mul.invalidate = true;
    signature_.params["std_mul"] = std::move(std_mul);
    cgraph::ParamSpec min_points;
    min_points.name = "min_points";
    min_points.dtype = "int";
    min_points.default_value = 0;
    min_points.invalidate = true;
    signature_.params["min_points"] = std::move(min_points);
    capability_.summary = "Statistical outlier removal + min_points gate";
    cost_.cost_class = "cpu.n2";
    usage_.connect = "Wire cloud{xyz} into in; read filtered cloud from out.";
    usage_.tune =
        "mean_k neighbors; drop if mean dist > mu+std_mul*sigma; "
        "if remaining n < min_points emit empty cloud.";
    usage_.inspect = "out.n <= in.n; empty when below min_points.";
    usage_.for_whom = "飞点太多，想按邻域距离统计剔除外点";
    usage_.min_graph = cloud::usage::make_min_graph_node(
        "sor", "cloud.stat_outlier", {{"mean_k", 8}, {"std_mul", 1.0}},
        {{"in", cloud::usage::cloud_in_xyz()}},
        {{"out", cloud::usage::cloud_out_xyz()}});
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs, const nlohmann::json& params,
      const cgraph::ExecContext&) const override {
    const auto it = inputs.find("in");
    if (it == inputs.end()) {
      throw std::invalid_argument("cloud.stat_outlier: missing input 'in'");
    }
    require_cloud(it->second);
    int mean_k = 8;
    double std_mul = 1.0;
    int min_points = 0;
    if (params.is_object()) {
      if (params.contains("mean_k") && params["mean_k"].is_number_integer()) {
        mean_k = params["mean_k"].get<int>();
      }
      if (params.contains("std_mul") && params["std_mul"].is_number()) {
        std_mul = params["std_mul"].get<double>();
      }
      if (params.contains("min_points") && params["min_points"].is_number_integer()) {
        min_points = params["min_points"].get<int>();
      }
    }
    if (mean_k < 1) {
      throw std::invalid_argument("cloud.stat_outlier: mean_k >= 1");
    }
    const auto n = cloud_n(it->second);
    const auto& xyz_json = it->second["xyz"];
    std::vector<float> xyz;
    xyz.reserve(n * 3);
    for (std::size_t i = 0; i < n * 3; ++i) {
      xyz.push_back(xyz_json[i].get<float>());
    }
    if (n == 0) {
      return {{"out", make_xyz_cloud({})}};
    }

    const auto knn = knn_indices(xyz, mean_k);
    std::vector<double> mean_dist(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      if (knn[i].empty()) {
        mean_dist[i] = 0.0;
        continue;
      }
      double sum = 0.0;
      for (int j : knn[i]) {
        const double dx = xyz[3 * i] - xyz[3 * static_cast<std::size_t>(j)];
        const double dy = xyz[3 * i + 1] - xyz[3 * static_cast<std::size_t>(j) + 1];
        const double dz = xyz[3 * i + 2] - xyz[3 * static_cast<std::size_t>(j) + 2];
        sum += std::sqrt(dx * dx + dy * dy + dz * dz);
      }
      mean_dist[i] = sum / static_cast<double>(knn[i].size());
    }
    double mu = 0.0;
    for (double d : mean_dist) {
      mu += d;
    }
    mu /= static_cast<double>(n);
    double var = 0.0;
    for (double d : mean_dist) {
      const double t = d - mu;
      var += t * t;
    }
    var /= static_cast<double>(n);
    const double sigma = std::sqrt(var);
    const double thresh = mu + std_mul * sigma;

    std::vector<float> out_xyz;
    out_xyz.reserve(n * 3);
    for (std::size_t i = 0; i < n; ++i) {
      if (mean_dist[i] <= thresh) {
        out_xyz.push_back(xyz[3 * i]);
        out_xyz.push_back(xyz[3 * i + 1]);
        out_xyz.push_back(xyz[3 * i + 2]);
      }
    }
    if (static_cast<int>(out_xyz.size() / 3) < min_points) {
      return {{"out", make_xyz_cloud({})}};
    }
    return {{"out", make_xyz_cloud(out_xyz)}};
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_cloud_stat_outlier() {
  return std::make_shared<StatOutlierOp>();
}

}  // namespace cloud
