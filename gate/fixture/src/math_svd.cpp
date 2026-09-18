#include "cgraph/ops.hpp"
#include "fx_typed.hpp"

#include <Eigen/SVD>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

namespace fixture {
namespace {

const cgraph::DataObject& require_matrix_alias(
    const std::map<std::string, cgraph::DataObject>& data_in, const char* primary,
    const char* alias, const char* op) {
  if (data_in.count(primary) != 0) {
    return data_in.at(primary);
  }
  if (data_in.count(alias) != 0) {
    return data_in.at(alias);
  }
  throw std::invalid_argument(std::string(op) + ": missing '" + primary +
                              "' (or alias '" + alias + "')");
}

class SvdOp final : public cgraph::MemoryOperator {
 public:
  SvdOp() {
    op_id_ = "fx.svd";
    signature_.inputs["A"] = fx_typed::matrix_port("A");
    auto in_alias = fx_typed::matrix_port("in");
    in_alias.optional = true;
    signature_.inputs["in"] = std::move(in_alias);
    signature_.outputs["U"] = fx_typed::matrix_port("U");
    signature_.outputs["S"] = fx_typed::vector_port("S");
    signature_.outputs["Vt"] = fx_typed::matrix_port("Vt");
    capability_.summary = "Thin SVD: A = U * diag(S) * Vt (all outs always)";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire matrix A (or in); read U, S, Vt.";
    usage_.tune = "No parameters; full thin SVD always computed.";
    usage_.inspect = "S is 1D singular values descending; Vt is V^T.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const Eigen::MatrixXd A = fx_typed::require_matrix(
        require_matrix_alias(data_in, "A", "in", "fx.svd"), "fx.svd", "A");
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    return {{"U", fx_typed::make_matrix(svd.matrixU())},
            {"S", fx_typed::make_vector(svd.singularValues())},
            {"Vt", fx_typed::make_matrix(svd.matrixV().transpose())}};
  }
};

class SliceTopKOp final : public cgraph::MemoryOperator {
 public:
  SliceTopKOp() {
    op_id_ = "fx.slice_topk";
    signature_.inputs["mat"] = fx_typed::matrix_port("mat");
    signature_.inputs["k"] = fx_typed::int_port("k");
    signature_.outputs["out"] = fx_typed::matrix_port("out");
    capability_.summary = "Take first k rows of a matrix (e.g. Vt top-k)";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire mat and k; read out (k x n).";
    usage_.tune = "k must be in [1, rows].";
    usage_.inspect = "math_07: SliceTopK on Vt keeps leading principal rows.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const Eigen::MatrixXd mat = fx_typed::require_matrix(
        fx_typed::require_obj(data_in, "mat", "fx.slice_topk"), "fx.slice_topk",
        "mat");
    const int k = static_cast<int>(fx_typed::require_int(
        fx_typed::require_obj(data_in, "k", "fx.slice_topk"), "fx.slice_topk",
        "k"));
    if (k <= 0 || k > mat.rows()) {
      throw std::invalid_argument("fx.slice_topk: k out of range");
    }
    return {{"out", fx_typed::make_matrix(mat.topRows(k))}};
  }
};

class VarianceRatioOp final : public cgraph::MemoryOperator {
 public:
  VarianceRatioOp() {
    op_id_ = "fx.variance_ratio";
    signature_.inputs["s"] = fx_typed::vector_port("s");
    auto S_alias = fx_typed::vector_port("S");
    S_alias.optional = true;
    signature_.inputs["S"] = std::move(S_alias);
    signature_.inputs["k"] = fx_typed::int_port("k");
    signature_.outputs["out"] = fx_typed::float_port("out");
    capability_.summary = "Explained variance ratio from singular values";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire singular values s (or S) and k; read out scalar.";
    usage_.tune = "out = sum(s_i^2, i<k) / sum(s_j^2).";
    usage_.inspect = "Uses squared singular values as variance proxy.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const Eigen::VectorXd s = fx_typed::require_vector(
        require_matrix_alias(data_in, "s", "S", "fx.variance_ratio"),
        "fx.variance_ratio", "s");
    const int k = static_cast<int>(fx_typed::require_int(
        fx_typed::require_obj(data_in, "k", "fx.variance_ratio"),
        "fx.variance_ratio", "k"));
    if (k <= 0 || k > s.size()) {
      throw std::invalid_argument("fx.variance_ratio: k out of range");
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
    return {{"out", fx_typed::make_number_float(num / den)}};
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
