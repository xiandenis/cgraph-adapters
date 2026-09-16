#include "cgraph/ops.hpp"
#include "cloud/cloud_value.hpp"
#include "cloud/usage_helpers.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cloud {
namespace {

class EuclidClusterOp final : public cgraph::MemoryOperator {
 public:
  EuclidClusterOp() {
    op_id_ = "cloud.euclid_cluster";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, "cloud");
    signature_.inputs["in"].formats = {"native"};
    signature_.inputs["in"].schema = {{"fields", nlohmann::json::array({"xyz"})}};
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, "list[cloud]");
    signature_.outputs["out"].formats = {"native"};
    cgraph::ParamSpec tol;
    tol.name = "tolerance";
    tol.dtype = "float";
    tol.default_value = 0.05;
    tol.invalidate = true;
    signature_.params["tolerance"] = std::move(tol);
    cgraph::ParamSpec min_size;
    min_size.name = "min_size";
    min_size.dtype = "int";
    min_size.default_value = 1;
    min_size.invalidate = true;
    signature_.params["min_size"] = std::move(min_size);
    cgraph::ParamSpec max_size;
    max_size.name = "max_size";
    max_size.dtype = "int";
    max_size.default_value = 1000000;
    max_size.invalidate = true;
    signature_.params["max_size"] = std::move(max_size);
    capability_.summary = "Euclidean clustering → list[cloud] (gate)";
    cost_.cost_class = "cpu.n2";
    usage_.connect = "Wire cloud{xyz} into in; read list[cloud] clusters from out.";
    usage_.tune =
        "tolerance = link distance; drop components outside [min_size, max_size]. "
        "Case-12 min_points belongs on MAP body filter, not here.";
    usage_.inspect = "out is JSON array of cloud objects; order by first seed index.";
    usage_.for_whom = "要把点云按欧氏距离分成多个簇";
    usage_.min_graph = cloud::usage::make_min_graph_node(
        "cluster", "cloud.euclid_cluster", {{"tolerance", 0.05}, {"min_size", 1}},
        {{"in", cloud::usage::cloud_in_xyz()}},
        {{"out", cgraph::make_port("out", cgraph::PortKind::Value, "list[cloud]")}});
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs, const nlohmann::json& params,
      const cgraph::ExecContext&) const override {
    const auto it = inputs.find("in");
    if (it == inputs.end()) {
      throw std::invalid_argument("cloud.euclid_cluster: missing input 'in'");
    }
    require_cloud(it->second);
    double tolerance = 0.05;
    int min_size = 1;
    int max_size = 1000000;
    if (params.is_object()) {
      if (params.contains("tolerance") && params["tolerance"].is_number()) {
        tolerance = params["tolerance"].get<double>();
      }
      if (params.contains("min_size") && params["min_size"].is_number_integer()) {
        min_size = params["min_size"].get<int>();
      }
      if (params.contains("max_size") && params["max_size"].is_number_integer()) {
        max_size = params["max_size"].get<int>();
      }
    }
    if (tolerance <= 0.0 || min_size < 1 || max_size < min_size) {
      throw std::invalid_argument("cloud.euclid_cluster: bad params");
    }
    const double t2 = tolerance * tolerance;
    const auto n = cloud_n(it->second);
    const auto& xyz_json = it->second["xyz"];
    std::vector<float> xyz;
    xyz.reserve(n * 3);
    for (std::size_t i = 0; i < n * 3; ++i) {
      xyz.push_back(xyz_json[i].get<float>());
    }

    std::vector<char> visited(n, 0);
    nlohmann::json clusters = nlohmann::json::array();
    for (std::size_t seed = 0; seed < n; ++seed) {
      if (visited[seed]) {
        continue;
      }
      std::vector<std::size_t> comp;
      std::queue<std::size_t> q;
      q.push(seed);
      visited[seed] = 1;
      while (!q.empty()) {
        const std::size_t i = q.front();
        q.pop();
        comp.push_back(i);
        for (std::size_t j = 0; j < n; ++j) {
          if (visited[j]) {
            continue;
          }
          const double dx = xyz[3 * i] - xyz[3 * j];
          const double dy = xyz[3 * i + 1] - xyz[3 * j + 1];
          const double dz = xyz[3 * i + 2] - xyz[3 * j + 2];
          if (dx * dx + dy * dy + dz * dz <= t2) {
            visited[j] = 1;
            q.push(j);
          }
        }
      }
      const int sz = static_cast<int>(comp.size());
      if (sz < min_size || sz > max_size) {
        continue;
      }
      std::vector<float> out_xyz;
      out_xyz.reserve(static_cast<std::size_t>(sz) * 3);
      for (std::size_t idx : comp) {
        out_xyz.push_back(xyz[3 * idx]);
        out_xyz.push_back(xyz[3 * idx + 1]);
        out_xyz.push_back(xyz[3 * idx + 2]);
      }
      clusters.push_back(make_xyz_cloud(out_xyz));
    }
    return {{"out", std::move(clusters)}};
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_cloud_euclid_cluster() {
  return std::make_shared<EuclidClusterOp>();
}

}  // namespace cloud
