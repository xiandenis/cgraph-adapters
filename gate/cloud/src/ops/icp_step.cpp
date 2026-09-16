#include "cgraph/ops.hpp"
#include "cloud/cloud_value.hpp"
#include "cloud/rigid.hpp"
#include "cloud/usage_helpers.hpp"

#include <Eigen/Core>

#include <cmath>
#include <limits>
#include <memory>
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
  return {T["t"][0].get<double>(), T["t"][1].get<double>(), T["t"][2].get<double>()};
}

class IcpStepOp final : public cgraph::MemoryOperator {
 public:
  IcpStepOp() {
    op_id_ = "cloud.icp_step";
    signature_.inputs["src"] =
        cgraph::make_port("src", cgraph::PortKind::Value, "cloud");
    signature_.inputs["src"].formats = {"native"};
    signature_.inputs["tgt"] =
        cgraph::make_port("tgt", cgraph::PortKind::Value, "cloud");
    signature_.inputs["tgt"].formats = {"native"};
    signature_.inputs["T"] =
        cgraph::make_port("T", cgraph::PortKind::Value, "rigid_transform");
    signature_.inputs["T"].formats = {"native"};
    signature_.outputs["T"] =
        cgraph::make_port("T", cgraph::PortKind::Value, "rigid_transform");
    signature_.outputs["T"].formats = {"native"};
    signature_.outputs["rmse"] =
        cgraph::make_port("rmse", cgraph::PortKind::Value, "json");
    signature_.outputs["rmse"].formats = {"native"};
    cgraph::ParamSpec max_corr;
    max_corr.name = "maxCorrDist";
    max_corr.dtype = "float";
    max_corr.default_value = 0.1;
    max_corr.invalidate = true;
    signature_.params["maxCorrDist"] = std::move(max_corr);
    capability_.summary = "One ICP iteration (NN + Kabsch); gate";
    cost_.cost_class = "cpu.n_m";
    usage_.connect = "src/tgt cloud + T_in → T_out + rmse; use inside LOOP.";
    usage_.tune = "params.maxCorrDist correspondence cutoff.";
    usage_.inspect = "rmse is mean inlier distance after apply T_out.";
    usage_.for_whom = "已有粗 T，要在 LOOP 里做点云 ICP 迭代";
    cloud::usage::set_tune(usage_, {
        {"rmse 几乎不降", "maxCorrDist", "略增大 maxCorrDist"},
        {"T 把云甩飞", "maxCorrDist", "减小 maxCorrDist"},
        {"对应点过少", "maxCorrDist", "检查上游配准或加大 leaf"},
    });
    usage_.min_graph = cloud::usage::make_min_graph_node(
        "icp", "cloud.icp_step", {{"maxCorrDist", 0.1}},
        {{"src", cgraph::make_port("src", cgraph::PortKind::Value, "cloud")},
         {"tgt", cgraph::make_port("tgt", cgraph::PortKind::Value, "cloud")},
         {"T", cgraph::make_port("T", cgraph::PortKind::Value, "rigid_transform")}},
        {{"T", cgraph::make_port("T", cgraph::PortKind::Value, "rigid_transform")},
         {"rmse", cgraph::make_port("rmse", cgraph::PortKind::Value, "json")}});
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs, const nlohmann::json& params,
      const cgraph::ExecContext&) const override {
    const auto sit = inputs.find("src");
    const auto tit = inputs.find("tgt");
    const auto Tit = inputs.find("T");
    if (sit == inputs.end() || tit == inputs.end() || Tit == inputs.end()) {
      throw std::invalid_argument("cloud.icp_step: missing inputs");
    }
    require_rigid(Tit->second);
    auto src = xyz_of(sit->second);
    auto tgt = xyz_of(tit->second);
    const std::size_t ns = src.size() / 3;
    const std::size_t nt = tgt.size() / 3;
    double max_corr = 0.1;
    if (params.is_object() && params.contains("maxCorrDist") &&
        params["maxCorrDist"].is_number()) {
      max_corr = params["maxCorrDist"].get<double>();
    }
    const double max2 = max_corr * max_corr;
    const Eigen::Matrix3d R = R_of(Tit->second);
    const Eigen::Vector3d t = t_of(Tit->second);

    std::vector<float> Ps, Qt;
    double sse = 0.0;
    int n_corr = 0;
    for (std::size_t i = 0; i < ns; ++i) {
      Eigen::Vector3d p(src[3 * i], src[3 * i + 1], src[3 * i + 2]);
      Eigen::Vector3d tp = R * p + t;
      float best = std::numeric_limits<float>::infinity();
      std::size_t best_j = 0;
      for (std::size_t j = 0; j < nt; ++j) {
        const float dx = static_cast<float>(tp.x() - tgt[3 * j]);
        const float dy = static_cast<float>(tp.y() - tgt[3 * j + 1]);
        const float dz = static_cast<float>(tp.z() - tgt[3 * j + 2]);
        const float d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < best) {
          best = d2;
          best_j = j;
        }
      }
      if (best <= max2) {
        Ps.push_back(src[3 * i]);
        Ps.push_back(src[3 * i + 1]);
        Ps.push_back(src[3 * i + 2]);
        Qt.push_back(tgt[3 * best_j]);
        Qt.push_back(tgt[3 * best_j + 1]);
        Qt.push_back(tgt[3 * best_j + 2]);
        sse += static_cast<double>(best);
        ++n_corr;
      }
    }
    if (n_corr < 3) {
      return {{"T", Tit->second}, {"rmse", nlohmann::json(max_corr)}};
    }
    nlohmann::json T_abs = kabsch_rigid(Ps, Qt);
    // Kabsch maps src→tgt directly; use as new absolute T (not delta compose).
    // Recompute rmse under T_abs.
    const Eigen::Matrix3d Ra = R_of(T_abs);
    const Eigen::Vector3d ta = t_of(T_abs);
    double sse2 = 0.0;
    int n2 = 0;
    for (std::size_t i = 0; i < ns; ++i) {
      Eigen::Vector3d p(src[3 * i], src[3 * i + 1], src[3 * i + 2]);
      Eigen::Vector3d tp = Ra * p + ta;
      float best = std::numeric_limits<float>::infinity();
      for (std::size_t j = 0; j < nt; ++j) {
        const float dx = static_cast<float>(tp.x() - tgt[3 * j]);
        const float dy = static_cast<float>(tp.y() - tgt[3 * j + 1]);
        const float dz = static_cast<float>(tp.z() - tgt[3 * j + 2]);
        best = std::min(best, dx * dx + dy * dy + dz * dz);
      }
      if (best <= max2) {
        sse2 += static_cast<double>(best);
        ++n2;
      }
    }
    const double rmse =
        n2 > 0 ? std::sqrt(sse2 / static_cast<double>(n2)) : max_corr;
    return {{"T", T_abs}, {"rmse", nlohmann::json(rmse)}};
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_cloud_icp_step() {
  return std::make_shared<IcpStepOp>();
}

}  // namespace cloud
