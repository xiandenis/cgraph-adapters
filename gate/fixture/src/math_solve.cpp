#include "cgraph/ops.hpp"
#include "fx_data.hpp"
#include "math_nd_util.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

namespace fixture {
namespace {

double residual_norm(const Eigen::MatrixXd& A, const Eigen::VectorXd& x,
                     const Eigen::VectorXd& b) {
  return (A * x - b).norm();
}

class CondEstimateOp final : public cgraph::MemoryOperator {
 public:
  CondEstimateOp() {
    op_id_ = "fx.cond_estimate";
    signature_.inputs["A"] =
        cgraph::make_port("A", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    auto in_alias = cgraph::make_port("in", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    in_alias.optional = true;
    signature_.inputs["in"] = std::move(in_alias);
    signature_.outputs["CondNum"] =
        cgraph::make_port("CondNum", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.outputs["IsWell"] =
        cgraph::make_port("IsWell", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
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
    const auto inputs = fx::unwrap(data_in);
    const nlohmann::json& a_json =
        math_nd::require_input_alias(inputs, "A", "in", "fx.cond_estimate");
    const Eigen::MatrixXd A =
        math_nd::parse_matrix(a_json, "fx.cond_estimate", "A");
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
    const double thr = math_nd::param_number(params, "threshold", 10000.0);
    const bool well = cond < thr;
    return fx::wrap(signature_, {{"CondNum", cond}, {"IsWell", well}});
  }
};

class ConditionRouterOp final : public cgraph::MemoryOperator {
 public:
  ConditionRouterOp() {
    op_id_ = "fx.condition_router";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    auto pred = cgraph::make_port("pred", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    pred.optional = true;
    signature_.inputs["pred"] = std::move(pred);
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
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
    const auto inputs = fx::unwrap(data_in);
    const nlohmann::json& v =
        math_nd::require_input_alias(inputs, "in", "pred", "fx.condition_router");
    return fx::wrap(signature_, {{"out", math_nd::as_bool_pred(v, "fx.condition_router")}});
  }
};

class LuSolveOp final : public cgraph::MemoryOperator {
 public:
  LuSolveOp() {
    op_id_ = "fx.lu_solve";
    signature_.inputs["A"] =
        cgraph::make_port("A", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.inputs["b"] =
        cgraph::make_port("b", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.outputs["x"] =
        cgraph::make_port("x", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.outputs["res"] =
        cgraph::make_port("res", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
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
    const auto inputs = fx::unwrap(data_in);
    const Eigen::MatrixXd A = math_nd::parse_matrix(
        math_nd::require_input(inputs, "A", "fx.lu_solve"), "fx.lu_solve", "A");
    const Eigen::VectorXd b = math_nd::parse_vector(
        math_nd::require_input(inputs, "b", "fx.lu_solve"), "fx.lu_solve", "b");
    if (A.rows() != A.cols() || A.rows() != b.size()) {
      throw std::invalid_argument("fx.lu_solve: A must be square matching b");
    }
    Eigen::PartialPivLU<Eigen::MatrixXd> lu(A);
    const Eigen::VectorXd x = lu.solve(b);
    return fx::wrap(signature_, {{"x", math_nd::make_vec_nd(x)},
            {"res", residual_norm(A, x, b)}});
  }
};

class QrSolveOp final : public cgraph::MemoryOperator {
 public:
  QrSolveOp() {
    op_id_ = "fx.qr_solve";
    signature_.inputs["A"] =
        cgraph::make_port("A", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.inputs["b"] =
        cgraph::make_port("b", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.outputs["x"] =
        cgraph::make_port("x", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.outputs["res"] =
        cgraph::make_port("res", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
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
    const auto inputs = fx::unwrap(data_in);
    const Eigen::MatrixXd A = math_nd::parse_matrix(
        math_nd::require_input(inputs, "A", "fx.qr_solve"), "fx.qr_solve", "A");
    const Eigen::VectorXd b = math_nd::parse_vector(
        math_nd::require_input(inputs, "b", "fx.qr_solve"), "fx.qr_solve", "b");
    if (A.rows() != b.size()) {
      throw std::invalid_argument("fx.qr_solve: A rows must match b");
    }
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
    const Eigen::VectorXd x = qr.solve(b);
    return fx::wrap(signature_, {{"x", math_nd::make_vec_nd(x)},
            {"res", residual_norm(A, x, b)}});
  }
};

class IterativeSolveOp final : public cgraph::MemoryOperator {
 public:
  IterativeSolveOp() {
    op_id_ = "fx.iterative_solve";
    signature_.inputs["A"] =
        cgraph::make_port("A", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.inputs["b"] =
        cgraph::make_port("b", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.outputs["x"] =
        cgraph::make_port("x", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.outputs["res"] =
        cgraph::make_port("res", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
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
    const auto inputs = fx::unwrap(data_in);
    const Eigen::MatrixXd A = math_nd::parse_matrix(
        math_nd::require_input(inputs, "A", "fx.iterative_solve"),
        "fx.iterative_solve", "A");
    const Eigen::VectorXd b = math_nd::parse_vector(
        math_nd::require_input(inputs, "b", "fx.iterative_solve"),
        "fx.iterative_solve", "b");
    if (A.rows() != A.cols() || A.rows() != b.size()) {
      throw std::invalid_argument(
          "fx.iterative_solve: A must be square matching b");
    }
    const double tol = math_nd::param_number(params, "tol", 1.0e-8);
    const int max_iters = math_nd::param_int(params, "max_iters", 1000);
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
    return fx::wrap(signature_, {{"x", math_nd::make_vec_nd(x)},
            {"res", residual_norm(A, x, b)}});
  }
};

class BranchMergeVecOp final : public cgraph::MemoryOperator {
 public:
  BranchMergeVecOp() {
    op_id_ = "fx.branch_merge_vec";
    signature_.inputs["pred"] =
        cgraph::make_port("pred", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.inputs["true"] =
        cgraph::make_port("true", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.inputs["false"] =
        cgraph::make_port("false", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
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
    const auto inputs = fx::unwrap(data_in);
    const bool pred = math_nd::as_bool_pred(
        math_nd::require_input(inputs, "pred", "fx.branch_merge_vec"),
        "fx.branch_merge_vec");
    const std::string port = pred ? "true" : "false";
    return fx::wrap(signature_, {{"out", math_nd::require_input(inputs, port, "fx.branch_merge_vec")}});
  }
};

class BranchMergeScalarOp final : public cgraph::MemoryOperator {
 public:
  BranchMergeScalarOp() {
    op_id_ = "fx.branch_merge_scalar";
    signature_.inputs["pred"] =
        cgraph::make_port("pred", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.inputs["true"] =
        cgraph::make_port("true", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.inputs["false"] =
        cgraph::make_port("false", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
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
    const auto inputs = fx::unwrap(data_in);
    const bool pred = math_nd::as_bool_pred(
        math_nd::require_input(inputs, "pred", "fx.branch_merge_scalar"),
        "fx.branch_merge_scalar");
    const std::string port = pred ? "true" : "false";
    return fx::wrap(signature_, {
        {"out", math_nd::require_input(inputs, port, "fx.branch_merge_scalar")}});
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
