#include "cgraph/ops.hpp"
#include "cloud/cloud_value.hpp"
#include "cloud/knn.hpp"
#include "cloud/usage_helpers.hpp"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace cloud {
namespace {

std::vector<float> json_xyz(const nlohmann::json& cloud) {
  require_cloud(cloud);
  std::vector<float> xyz;
  xyz.reserve(cloud["xyz"].size());
  for (const auto& v : cloud["xyz"]) {
    xyz.push_back(v.get<float>());
  }
  return xyz;
}

class NormalEstOp final : public cgraph::MemoryOperator {
 public:
  NormalEstOp() {
    op_id_ = "cloud.normal_est";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, "cloud");
    signature_.inputs["in"].formats = {"native"};
    signature_.inputs["in"].schema = {{"fields", nlohmann::json::array({"xyz"})}};
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, "cloud");
    signature_.outputs["out"].formats = {"native"};
    signature_.outputs["out"].schema = {
        {"fields", nlohmann::json::array({"xyz", "normal"})}};
    cgraph::ParamSpec k;
    k.name = "k";
    k.dtype = "int";
    k.default_value = 16;
    k.invalidate = true;
    signature_.params["k"] = std::move(k);
    cgraph::ParamSpec radius_n;
    radius_n.name = "radius_n";
    radius_n.dtype = "float";
    radius_n.default_value = 0.0;
    radius_n.invalidate = true;
    signature_.params["radius_n"] = std::move(radius_n);
    capability_.summary = "PCA normals (gate; Eigen)";
    cost_.cost_class = "cpu.n_log_n";
    usage_.connect = "Wire cloud{xyz}; read cloud{xyz,normal}.";
    usage_.tune = "params.k neighbors; radius_n>0 uses radius instead of k.";
    usage_.inspect = "out.fields includes normal; length 3n.";
    usage_.for_whom = "下一步要 FPFH / 点到面 ICP，云上还没有 normal";
    usage_.connect_bullets = {
        "输入只要 xyz；输出 cloud 带 normal，可直连 FPFH",
    };
    usage_.set_bullets = {
        "radius_n 先取 2～3 个 leaf，或 k=8～16",
    };
    usage_.look_bullets = {
        "法向探针应大致垂直表面，不应随机朝向",
    };
    usage_.first_values.push_back(
        cgraph::UsageFirstValue{"k", 16, "邻域点数；也可用 radius_n>0"});
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
      throw std::invalid_argument("cloud.normal_est: missing in");
    }
    auto xyz = json_xyz(it->second);
    const std::size_t n = xyz.size() / 3;
    int k = 16;
    double radius_n = 0.0;
    if (params.is_object()) {
      if (params.contains("k") && params["k"].is_number_integer()) {
        k = params["k"].get<int>();
      }
      if (params.contains("radius_n") && params["radius_n"].is_number()) {
        radius_n = params["radius_n"].get<double>();
      }
    }
    std::vector<std::vector<int>> neigh;
    if (radius_n > 0.0) {
      neigh = radius_indices(xyz, static_cast<float>(radius_n));
    } else {
      neigh = knn_indices(xyz, std::max(3, k));
    }
    std::vector<float> normals(n * 3, 0.0f);
    for (std::size_t i = 0; i < n; ++i) {
      auto idxs = neigh[i];
      if (idxs.empty()) {
        normals[3 * i + 2] = 1.0f;
        continue;
      }
      if (std::find(idxs.begin(), idxs.end(), static_cast<int>(i)) == idxs.end()) {
        idxs.push_back(static_cast<int>(i));
      }
      Eigen::Vector3d mu = Eigen::Vector3d::Zero();
      for (int j : idxs) {
        mu += Eigen::Vector3d(xyz[3 * j], xyz[3 * j + 1], xyz[3 * j + 2]);
      }
      mu /= static_cast<double>(idxs.size());
      Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
      for (int j : idxs) {
        Eigen::Vector3d d =
            Eigen::Vector3d(xyz[3 * j], xyz[3 * j + 1], xyz[3 * j + 2]) - mu;
        cov += d * d.transpose();
      }
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
      Eigen::Vector3d nrm = es.eigenvectors().col(0);
      if (nrm.z() < 0.0) {
        nrm = -nrm;
      }
      normals[3 * i] = static_cast<float>(nrm.x());
      normals[3 * i + 1] = static_cast<float>(nrm.y());
      normals[3 * i + 2] = static_cast<float>(nrm.z());
    }
    nlohmann::json out = it->second;
    out["normal"] = normals;
    nlohmann::json fields = nlohmann::json::array({"xyz", "normal"});
    out["fields"] = fields;
    return {{"out", out}};
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_cloud_normal_est() {
  return std::make_shared<NormalEstOp>();
}

}  // namespace cloud
