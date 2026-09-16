#include "cgraph/ops.hpp"
#include "math_nd_util.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

namespace fixture {
namespace {

class SvdOp final : public cgraph::MemoryOperator {
 public:
  SvdOp() {
    op_id_ = "fx.svd";
    signature_.inputs["A"] =
        cgraph::make_port("A", cgraph::PortKind::Value, "json");
    auto in_alias = cgraph::make_port("in", cgraph::PortKind::Value, "json");
    in_alias.optional = true;
    signature_.inputs["in"] = std::move(in_alias);
    signature_.outputs["U"] =
        cgraph::make_port("U", cgraph::PortKind::Value, "json");
    signature_.outputs["S"] =
        cgraph::make_port("S", cgraph::PortKind::Value, "json");
    signature_.outputs["Vt"] =
        cgraph::make_port("Vt", cgraph::PortKind::Value, "json");
    capability_.summary = "Thin SVD: A = U * diag(S) * Vt (all outs always)";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire matrix A (or in); read U, S, Vt.";
    usage_.tune = "No parameters; full thin SVD always computed.";
    usage_.inspect = "S is 1D singular values descending; Vt is V^T.";
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const nlohmann::json& a_json =
        math_nd::require_input_alias(inputs, "A", "in", "fx.svd");
    const Eigen::MatrixXd A = math_nd::parse_matrix(a_json, "fx.svd", "A");
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::MatrixXd U = svd.matrixU();
    const Eigen::VectorXd S = svd.singularValues();
    const Eigen::MatrixXd Vt = svd.matrixV().transpose();
    return {{"U", math_nd::make_nd(U)},
            {"S", math_nd::make_vec_nd(S)},
            {"Vt", math_nd::make_nd(Vt)}};
  }
};

class SliceTopKOp final : public cgraph::MemoryOperator {
 public:
  SliceTopKOp() {
    op_id_ = "fx.slice_topk";
    signature_.inputs["mat"] =
        cgraph::make_port("mat", cgraph::PortKind::Value, "json");
    signature_.inputs["k"] =
        cgraph::make_port("k", cgraph::PortKind::Value, "json");
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, "json");
    capability_.summary = "Take first k rows of a matrix (e.g. Vt top-k)";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire mat and k; read out (k x n).";
    usage_.tune = "k must be in [1, rows].";
    usage_.inspect = "math_07: SliceTopK on Vt keeps leading principal rows.";
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const Eigen::MatrixXd mat =
        math_nd::parse_matrix(math_nd::require_input(inputs, "mat", "fx.slice_topk"),
                              "fx.slice_topk", "mat");
    const int k = math_nd::require_positive_int(
        math_nd::require_input(inputs, "k", "fx.slice_topk"), "fx.slice_topk",
        "k");
    if (k > mat.rows()) {
      throw std::invalid_argument("fx.slice_topk: k exceeds matrix rows");
    }
    return {{"out", math_nd::make_nd(mat.topRows(k))}};
  }
};

class VarianceRatioOp final : public cgraph::MemoryOperator {
 public:
  VarianceRatioOp() {
    op_id_ = "fx.variance_ratio";
    signature_.inputs["s"] =
        cgraph::make_port("s", cgraph::PortKind::Value, "json");
    auto S_alias = cgraph::make_port("S", cgraph::PortKind::Value, "json");
    S_alias.optional = true;
    signature_.inputs["S"] = std::move(S_alias);
    signature_.inputs["k"] =
        cgraph::make_port("k", cgraph::PortKind::Value, "json");
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, "json");
    capability_.summary = "Explained variance ratio from singular values";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire singular values s (or S) and k; read out scalar.";
    usage_.tune = "out = sum(s_i^2, i<k) / sum(s_j^2).";
    usage_.inspect = "Uses squared singular values as variance proxy.";
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const nlohmann::json& s_json =
        math_nd::require_input_alias(inputs, "s", "S", "fx.variance_ratio");
    const Eigen::VectorXd s =
        math_nd::parse_vector(s_json, "fx.variance_ratio", "s");
    const int k = math_nd::require_positive_int(
        math_nd::require_input(inputs, "k", "fx.variance_ratio"),
        "fx.variance_ratio", "k");
    if (k > s.size()) {
      throw std::invalid_argument("fx.variance_ratio: k exceeds |S|");
    }
    double num = 0.0;
    double den = 0.0;
    for (Eigen::Index i = 0; i < s.size(); ++i) {
      const double v = s(i) * s(i);
      den += v;
      if (i < k) {
        num += v;
      }
    }
    if (!(den > 0.0) || !std::isfinite(den)) {
      throw std::invalid_argument("fx.variance_ratio: zero or non-finite |S|");
    }
    return {{"out", num / den}};
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_svd() {
  return std::make_shared<SvdOp>();
}
std::shared_ptr<cgraph::MemoryOperator> make_slice_topk() {
  return std::make_shared<SliceTopKOp>();
}
std::shared_ptr<cgraph::MemoryOperator> make_variance_ratio() {
  return std::make_shared<VarianceRatioOp>();
}

}  // namespace fixture
