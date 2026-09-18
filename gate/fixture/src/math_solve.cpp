#include "cgraph/ops.hpp"
#include "fx_typed.hpp"

#include <Eigen/LU>
#include <Eigen/QR>
#include <Eigen/SVD>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

namespace fixture {
namespace {

const cgraph::DataObject& require_alias(
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

double residual_norm(const Eigen::MatrixXd& A, const Eigen::VectorXd& x,
                     const Eigen::VectorXd& b) {
  return (A * x - b).norm();
}

double param_number(const nlohmann::json& params, const char* key,
                    double fallback) {
  if (!params.is_object() || !params.contains(key) || !params[key].is_number()) {
    return fallback;
  }
  return params[key].get<double>();
}

int param_int(const nlohmann::json& params, const char* key, int fallback) {
  if (!params.is_object() || !params.contains(key)) {
    return fallback;
  }
  if (params[key].is_number_integer()) {
    return params[key].get<int>();
  }
  if (params[key].is_number()) {
    return static_cast<int>(params[key].get<double>());
  }
  return fallback;
}

class CondEstimateOp final : public cgraph::MemoryOperator {
 public:
  CondEstimateOp() {
    op_id_ = "fx.cond_estimate";
    signature_.inputs["A"] = fx_typed::matrix_port("A");
    auto in_alias = fx_typed::matrix_port("in");
    in_alias.optional = true;
    signature_.inputs["in"] = std::move(in_alias);
    signature_.outputs["CondNum"] = fx_typed::float_port("CondNum");
    signature_.outputs["IsWell"] = fx_typed::bool_port("IsWell");
    cgraph::ParamSpec thr;
    thr.name = "threshold";
    thr.dtype = "number";
    thr.default_value = 10000.0;
    thr.doc = "IsWell = CondNum < threshold";
    signature_.params["threshold"] = std::move(thr);
    capability_.summary = "2-norm condition estimate via SVD singular values";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire A (or in); read CondNum and IsWell.";
    usage_.tune = "params.threshold (default 1e4).";
    usage_.inspect = "cond = sigma_max / sigma_min.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in,
      const nlohmann::json& params, const cgraph::ExecContext&) const override {
    const Eigen::MatrixXd A = fx_typed::require_matrix(
        require_alias(data_in, "A", "in", "fx.cond_estimate"), "fx.cond_estimate",
        "A");
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A);
    const Eigen::VectorXd s = svd.singularValues();
    if (s.size() == 0) {
      throw std::invalid_argument("fx.cond_estimate: empty matrix");
    }
    const double smax = s(0);
    const double smin = s(s.size() - 1);
    if (!(smin > 0.0) || !std::isfinite(smin) || !std::isfinite(smax)) {
      throw std::invalid_argument("fx.cond_estimate: singular or non-finite");
    }
    const double cond = smax / smin;
    const double thr = param_number(params, "threshold", 10000.0);
    return {{"CondNum", fx_typed::make_number_float(cond)},
            {"IsWell", fx_typed::make_bool_pred(cond < thr)}};
  }
};

class ConditionRouterOp final : public cgraph::MemoryOperator {
 public:
  ConditionRouterOp() {
    op_id_ = "fx.condition_router";
    signature_.inputs["in"] = fx_typed::bool_port("in");
    auto pred = fx_typed::bool_port("pred");
    pred.optional = true;
    signature_.inputs["pred"] = std::move(pred);
    signature_.outputs["out"] = fx_typed::bool_port("out");
    capability_.summary = "Normalize pred/flag to boolean branch token";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire in (or pred); read boolean out.";
    usage_.tune = "No parameters.";
    usage_.inspect = "Pass-through boolean for BranchMerge / schedule.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const bool v = fx_typed::require_bool(
        require_alias(data_in, "in", "pred", "fx.condition_router"),
        "fx.condition_router", "in");
    return {{"out", fx_typed::make_bool_pred(v)}};
  }
};

class LuSolveOp final : public cgraph::MemoryOperator {
 public:
  LuSolveOp() {
    op_id_ = "fx.lu_solve";
    signature_.inputs["A"] = fx_typed::matrix_port("A");
    signature_.inputs["b"] = fx_typed::vector_port("b");
    signature_.outputs["x"] = fx_typed::vector_port("x");
    signature_.outputs["res"] = fx_typed::float_port("res");
    capability_.summary = "Solve Ax=b via partial-pivoted LU";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire A and b; read x and residual norm res.";
    usage_.tune = "No parameters.";
    usage_.inspect = "res = ||Ax-b||_2.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const Eigen::MatrixXd A = fx_typed::require_matrix(
        fx_typed::require_obj(data_in, "A", "fx.lu_solve"), "fx.lu_solve", "A");
    const Eigen::VectorXd b = fx_typed::require_vector(
        fx_typed::require_obj(data_in, "b", "fx.lu_solve"), "fx.lu_solve", "b");
    if (A.rows() != A.cols() || A.rows() != b.size()) {
      throw std::invalid_argument("fx.lu_solve: A must be square matching b");
    }
    Eigen::PartialPivLU<Eigen::MatrixXd> lu(A);
    const Eigen::VectorXd x = lu.solve(b);
    return {{"x", fx_typed::make_vector(x)},
            {"res", fx_typed::make_number_float(residual_norm(A, x, b))}};
  }
};

class QrSolveOp final : public cgraph::MemoryOperator {
 public:
  QrSolveOp() {
    op_id_ = "fx.qr_solve";
    signature_.inputs["A"] = fx_typed::matrix_port("A");
    signature_.inputs["b"] = fx_typed::vector_port("b");
    signature_.outputs["x"] = fx_typed::vector_port("x");
    signature_.outputs["res"] = fx_typed::float_port("res");
    capability_.summary = "Solve Ax=b via Householder QR";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire A and b; read x and residual norm res.";
    usage_.tune = "Same signature as fx.lu_solve for replace_op.";
    usage_.inspect = "res = ||Ax-b||_2.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const Eigen::MatrixXd A = fx_typed::require_matrix(
        fx_typed::require_obj(data_in, "A", "fx.qr_solve"), "fx.qr_solve", "A");
    const Eigen::VectorXd b = fx_typed::require_vector(
        fx_typed::require_obj(data_in, "b", "fx.qr_solve"), "fx.qr_solve", "b");
    if (A.rows() != b.size()) {
      throw std::invalid_argument("fx.qr_solve: A rows must match b");
    }
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
    const Eigen::VectorXd x = qr.solve(b);
    return {{"x", fx_typed::make_vector(x)},
            {"res", fx_typed::make_number_float(residual_norm(A, x, b))}};
  }
};

class IterativeSolveOp final : public cgraph::MemoryOperator {
 public:
  IterativeSolveOp() {
    op_id_ = "fx.iterative_solve";
    signature_.inputs["A"] = fx_typed::matrix_port("A");
    signature_.inputs["b"] = fx_typed::vector_port("b");
    signature_.outputs["x"] = fx_typed::vector_port("x");
    signature_.outputs["res"] = fx_typed::float_port("res");
    cgraph::ParamSpec tol;
    tol.name = "tol";
    tol.dtype = "number";
    tol.default_value = 1.0e-8;
    signature_.params["tol"] = std::move(tol);
    cgraph::ParamSpec mi;
    mi.name = "max_iters";
    mi.dtype = "int";
    mi.default_value = 1000;
    signature_.params["max_iters"] = std::move(mi);
    capability_.summary = "Jacobi iterative solve for Ax=b";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire A and b; read x and residual norm res.";
    usage_.tune = "params.tol, params.max_iters.";
    usage_.inspect = "Requires nonzero diagonal; starts from zeros.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in,
      const nlohmann::json& params, const cgraph::ExecContext&) const override {
    const Eigen::MatrixXd A = fx_typed::require_matrix(
        fx_typed::require_obj(data_in, "A", "fx.iterative_solve"),
        "fx.iterative_solve", "A");
    const Eigen::VectorXd b = fx_typed::require_vector(
        fx_typed::require_obj(data_in, "b", "fx.iterative_solve"),
        "fx.iterative_solve", "b");
    if (A.rows() != A.cols() || A.rows() != b.size()) {
      throw std::invalid_argument(
          "fx.iterative_solve: A must be square matching b");
    }
    const double tol = param_number(params, "tol", 1.0e-8);
    const int max_iters = param_int(params, "max_iters", 1000);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(b.size());
    Eigen::VectorXd x_new = x;
    for (int it = 0; it < max_iters; ++it) {
      for (Eigen::Index i = 0; i < A.rows(); ++i) {
        const double diag = A(i, i);
        if (!(std::abs(diag) > 0.0)) {
          throw std::invalid_argument(
              "fx.iterative_solve: zero diagonal at Jacobi");
        }
        double sigma = b(i);
        for (Eigen::Index j = 0; j < A.cols(); ++j) {
          if (j == i) {
            continue;
          }
          sigma -= A(i, j) * x(j);
        }
        x_new(i) = sigma / diag;
      }
      x = x_new;
      if (residual_norm(A, x, b) < tol) {
        break;
      }
    }
    return {{"x", fx_typed::make_vector(x)},
            {"res", fx_typed::make_number_float(residual_norm(A, x, b))}};
  }
};

class BranchMergeVecOp final : public cgraph::MemoryOperator {
 public:
  BranchMergeVecOp() {
    op_id_ = "fx.branch_merge_vec";
    signature_.inputs["pred"] = fx_typed::bool_port("pred");
    signature_.inputs["true"] = fx_typed::vector_port("true");
    signature_.inputs["false"] = fx_typed::vector_port("false");
    signature_.outputs["out"] = fx_typed::vector_port("out");
    capability_.summary = "Select vector from true/false by pred";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire pred, true, false; read out.";
    usage_.tune = "No parameters.";
    usage_.inspect = "Functional select; schedule not_scheduled is separate.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const bool pred = fx_typed::require_bool(
        fx_typed::require_obj(data_in, "pred", "fx.branch_merge_vec"),
        "fx.branch_merge_vec", "pred");
    const auto& chosen = fx_typed::require_obj(
        data_in, pred ? "true" : "false", "fx.branch_merge_vec");
    return {{"out", fx_typed::make_vector(fx_typed::require_vector(
                        chosen, "fx.branch_merge_vec", pred ? "true" : "false"))}};
  }
};

class BranchMergeScalarOp final : public cgraph::MemoryOperator {
 public:
  BranchMergeScalarOp() {
    op_id_ = "fx.branch_merge_scalar";
    signature_.inputs["pred"] = fx_typed::bool_port("pred");
    signature_.inputs["true"] = fx_typed::float_port("true");
    signature_.inputs["false"] = fx_typed::float_port("false");
    signature_.outputs["out"] = fx_typed::float_port("out");
    capability_.summary = "Select scalar from true/false by pred";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire pred, true, false; read out.";
    usage_.tune = "No parameters.";
    usage_.inspect = "Functional select; schedule not_scheduled is separate.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const bool pred = fx_typed::require_bool(
        fx_typed::require_obj(data_in, "pred", "fx.branch_merge_scalar"),
        "fx.branch_merge_scalar", "pred");
    const double v = fx_typed::require_float_port(
        data_in, pred ? "true" : "false", "fx.branch_merge_scalar");
    return {{"out", fx_typed::make_number_float(v)}};
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_cond_estimate() {
  return std::make_shared<CondEstimateOp>();
}
std::shared_ptr<cgraph::MemoryOperator> make_condition_router() {
  return std::make_shared<ConditionRouterOp>();
}
std::shared_ptr<cgraph::MemoryOperator> make_lu_solve() {
  return std::make_shared<LuSolveOp>();
}
std::shared_ptr<cgraph::MemoryOperator> make_qr_solve() {
  return std::make_shared<QrSolveOp>();
}
std::shared_ptr<cgraph::MemoryOperator> make_iterative_solve() {
  return std::make_shared<IterativeSolveOp>();
}
std::shared_ptr<cgraph::MemoryOperator> make_branch_merge_vec() {
  return std::make_shared<BranchMergeVecOp>();
}
std::shared_ptr<cgraph::MemoryOperator> make_branch_merge_scalar() {
  return std::make_shared<BranchMergeScalarOp>();
}

}  // namespace fixture
