#include "cgraph/ops.hpp"
#include "fx_typed.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

namespace fixture {
namespace {

void require_quadratic_l2(const nlohmann::json& params, const char* op) {
  std::string mode;
  if (params.is_object()) {
    if (params.contains("mode") && params["mode"].is_string()) {
      mode = params["mode"].get<std::string>();
    } else if (params.contains("f") && params["f"].is_string()) {
      mode = params["f"].get<std::string>();
    }
  }
  if (mode.empty()) {
    throw std::invalid_argument(std::string(op) +
                                ": params.mode (or f) required");
  }
  if (mode != "quadratic_l2") {
    throw std::invalid_argument(std::string(op) +
                                ": only mode/f=quadratic_l2 supported");
  }
}

double loss_quadratic_l2(const Eigen::VectorXd& x, const Eigen::MatrixXd* A,
                         const Eigen::VectorXd* b) {
  if (A != nullptr && b != nullptr) {
    const Eigen::VectorXd r = (*A) * x - (*b);
    return 0.5 * r.squaredNorm();
  }
  return 0.5 * x.squaredNorm();
}

Eigen::VectorXd grad_quadratic_l2(const Eigen::VectorXd& x,
                                  const Eigen::MatrixXd* A,
                                  const Eigen::VectorXd* b) {
  if (A != nullptr && b != nullptr) {
    return A->transpose() * ((*A) * x - (*b));
  }
  return x;
}

bool load_optional_Ab(const std::map<std::string, cgraph::DataObject>& data_in,
                      const char* op, Eigen::MatrixXd* Astore,
                      Eigen::VectorXd* bstore, const Eigen::MatrixXd** Ap,
                      const Eigen::VectorXd** bp) {
  const bool has_A = data_in.count("A") != 0;
  const bool has_b = data_in.count("b") != 0;
  if (!has_A && !has_b) {
    return false;
  }
  if (!has_A || !has_b) {
    throw std::invalid_argument(std::string(op) + ": A and b must both be set");
  }
  *Astore = fx_typed::require_matrix(data_in.at("A"), op, "A");
  *bstore = fx_typed::require_vector(data_in.at("b"), op, "b");
  *Ap = Astore;
  *bp = bstore;
  return true;
}

class GradientOp final : public cgraph::MemoryOperator {
 public:
  GradientOp() {
    op_id_ = "fx.gradient";
    signature_.inputs["x"] = fx_typed::vector_port("x");
    auto A = fx_typed::matrix_port("A");
    A.optional = true;
    signature_.inputs["A"] = std::move(A);
    auto b = fx_typed::vector_port("b");
    b.optional = true;
    signature_.inputs["b"] = std::move(b);
    signature_.outputs["out"] = fx_typed::vector_port("out");
    cgraph::ParamSpec mode;
    mode.name = "mode";
    mode.dtype = "string";
    mode.doc = "quadratic_l2 (alias params.f)";
    signature_.params["mode"] = mode;
    cgraph::ParamSpec f = mode;
    f.name = "f";
    signature_.params["f"] = std::move(f);
    capability_.summary =
        "Explicit gradient for quadratic_l2 (not auto-diff)";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire x; optional A,b for 0.5||Ax-b||^2; read out.";
    usage_.tune = "params.mode or params.f = quadratic_l2.";
    usage_.inspect = "Without A/b: grad = x for 0.5||x||^2.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in,
      const nlohmann::json& params, const cgraph::ExecContext&) const override {
    require_quadratic_l2(params, "fx.gradient");
    const Eigen::VectorXd x = fx_typed::require_vector(
        fx_typed::require_obj(data_in, "x", "fx.gradient"), "fx.gradient", "x");
    Eigen::MatrixXd Astore;
    Eigen::VectorXd bstore;
    const Eigen::MatrixXd* Ap = nullptr;
    const Eigen::VectorXd* bp = nullptr;
    load_optional_Ab(data_in, "fx.gradient", &Astore, &bstore, &Ap, &bp);
    return {{"out", fx_typed::make_vector(grad_quadratic_l2(x, Ap, bp))}};
  }
};

class LineSearchOp final : public cgraph::MemoryOperator {
 public:
  LineSearchOp() {
    op_id_ = "fx.line_search";
    signature_.inputs["x"] = fx_typed::vector_port("x");
    signature_.inputs["g"] = fx_typed::vector_port("g");
    signature_.inputs["lr"] = fx_typed::float_port("lr");
    auto A = fx_typed::matrix_port("A");
    A.optional = true;
    signature_.inputs["A"] = std::move(A);
    auto b = fx_typed::vector_port("b");
    b.optional = true;
    signature_.inputs["b"] = std::move(b);
    signature_.outputs["out"] = fx_typed::float_port("out");
    cgraph::ParamSpec mode;
    mode.name = "mode";
    mode.dtype = "string";
    signature_.params["mode"] = mode;
    cgraph::ParamSpec f = mode;
    f.name = "f";
    signature_.params["f"] = std::move(f);
    cgraph::ParamSpec bt;
    bt.name = "backtrack";
    bt.dtype = "bool";
    bt.default_value = true;
    signature_.params["backtrack"] = std::move(bt);
    capability_.summary = "Fixed or backtracking step from lr_base";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire x, g, lr; read alpha out.";
    usage_.tune = "params.backtrack (default true); mode/f=quadratic_l2.";
    usage_.inspect = "Backtrack: alpha *= 0.5 while loss increases.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in,
      const nlohmann::json& params, const cgraph::ExecContext&) const override {
    require_quadratic_l2(params, "fx.line_search");
    const Eigen::VectorXd x = fx_typed::require_vector(
        fx_typed::require_obj(data_in, "x", "fx.line_search"), "fx.line_search",
        "x");
    const Eigen::VectorXd g = fx_typed::require_vector(
        fx_typed::require_obj(data_in, "g", "fx.line_search"), "fx.line_search",
        "g");
    double alpha = fx_typed::require_float_port(data_in, "lr", "fx.line_search");
    if (!(alpha > 0.0) || !std::isfinite(alpha)) {
      throw std::invalid_argument("fx.line_search: lr must be positive finite");
    }
    Eigen::MatrixXd Astore;
    Eigen::VectorXd bstore;
    const Eigen::MatrixXd* Ap = nullptr;
    const Eigen::VectorXd* bp = nullptr;
    load_optional_Ab(data_in, "fx.line_search", &Astore, &bstore, &Ap, &bp);
    bool backtrack = true;
    if (params.is_object() && params.contains("backtrack") &&
        params["backtrack"].is_boolean()) {
      backtrack = params["backtrack"].get<bool>();
    }
    if (!backtrack) {
      return {{"out", fx_typed::make_number_float(alpha)}};
    }
    const double f0 = loss_quadratic_l2(x, Ap, bp);
    for (int i = 0; i < 20; ++i) {
      const Eigen::VectorXd x_try = x - alpha * g;
      const double f1 = loss_quadratic_l2(x_try, Ap, bp);
      if (f1 <= f0 || !std::isfinite(f1)) {
        break;
      }
      alpha *= 0.5;
    }
    return {{"out", fx_typed::make_number_float(alpha)}};
  }
};

class UpdateOp final : public cgraph::MemoryOperator {
 public:
  UpdateOp() {
    op_id_ = "fx.update";
    signature_.inputs["x"] = fx_typed::vector_port("x");
    signature_.inputs["g"] = fx_typed::vector_port("g");
    signature_.inputs["alpha"] = fx_typed::float_port("alpha");
    auto lr = fx_typed::float_port("lr");
    lr.optional = true;
    signature_.inputs["lr"] = std::move(lr);
    signature_.outputs["out"] = fx_typed::vector_port("out");
    capability_.summary = "GD step: x_new = x - alpha * g";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire x, g, alpha (or lr); read out.";
    usage_.tune = "No parameters.";
    usage_.inspect = "alpha preferred; lr accepted as alias.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const Eigen::VectorXd x = fx_typed::require_vector(
        fx_typed::require_obj(data_in, "x", "fx.update"), "fx.update", "x");
    const Eigen::VectorXd g = fx_typed::require_vector(
        fx_typed::require_obj(data_in, "g", "fx.update"), "fx.update", "g");
    double alpha = 0.0;
    if (data_in.count("alpha") != 0) {
      alpha = fx_typed::require_float(data_in.at("alpha"), "fx.update", "alpha");
    } else if (data_in.count("lr") != 0) {
      alpha = fx_typed::require_float(data_in.at("lr"), "fx.update", "lr");
    } else {
      throw std::invalid_argument("fx.update: missing alpha (or lr)");
    }
    if (x.size() != g.size()) {
      throw std::invalid_argument("fx.update: x and g size mismatch");
    }
    return {{"out", fx_typed::make_vector(x - alpha * g)}};
  }
};

class LossCalcOp final : public cgraph::MemoryOperator {
 public:
  LossCalcOp() {
    op_id_ = "fx.loss_calc";
    signature_.inputs["x"] = fx_typed::vector_port("x");
    auto A = fx_typed::matrix_port("A");
    A.optional = true;
    signature_.inputs["A"] = std::move(A);
    auto b = fx_typed::vector_port("b");
    b.optional = true;
    signature_.inputs["b"] = std::move(b);
    signature_.outputs["out"] = fx_typed::float_port("out");
    cgraph::ParamSpec mode;
    mode.name = "mode";
    mode.dtype = "string";
    signature_.params["mode"] = mode;
    cgraph::ParamSpec f = mode;
    f.name = "f";
    signature_.params["f"] = std::move(f);
    capability_.summary = "Scalar loss for quadratic_l2 (not AD)";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire x; optional A,b; read out.";
    usage_.tune = "params.mode or params.f = quadratic_l2.";
    usage_.inspect = "Without A/b: 0.5||x||^2.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in,
      const nlohmann::json& params, const cgraph::ExecContext&) const override {
    require_quadratic_l2(params, "fx.loss_calc");
    const Eigen::VectorXd x = fx_typed::require_vector(
        fx_typed::require_obj(data_in, "x", "fx.loss_calc"), "fx.loss_calc",
        "x");
    Eigen::MatrixXd Astore;
    Eigen::VectorXd bstore;
    const Eigen::MatrixXd* Ap = nullptr;
    const Eigen::VectorXd* bp = nullptr;
    load_optional_Ab(data_in, "fx.loss_calc", &Astore, &bstore, &Ap, &bp);
    return {{"out", fx_typed::make_number_float(loss_quadratic_l2(x, Ap, bp))}};
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_gradient() {
  return std::make_shared<GradientOp>();
}
std::shared_ptr<cgraph::MemoryOperator> make_line_search() {
  return std::make_shared<LineSearchOp>();
}
std::shared_ptr<cgraph::MemoryOperator> make_update() {
  return std::make_shared<UpdateOp>();
}
std::shared_ptr<cgraph::MemoryOperator> make_loss_calc() {
  return std::make_shared<LossCalcOp>();
}

}  // namespace fixture
