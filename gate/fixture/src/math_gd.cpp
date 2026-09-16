#include "cgraph/ops.hpp"
#include "math_nd_util.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

namespace fixture {
namespace {

void require_quadratic_l2(const nlohmann::json& params, const char* op) {
  const std::string mode = math_nd::mode_or_f(params);
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

class GradientOp final : public cgraph::MemoryOperator {
 public:
  GradientOp() {
    op_id_ = "fx.gradient";
    signature_.inputs["x"] =
        cgraph::make_port("x", cgraph::PortKind::Value, "json");
    auto A = cgraph::make_port("A", cgraph::PortKind::Value, "json");
    A.optional = true;
    signature_.inputs["A"] = std::move(A);
    auto b = cgraph::make_port("b", cgraph::PortKind::Value, "json");
    b.optional = true;
    signature_.inputs["b"] = std::move(b);
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, "json");
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

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs,
      const nlohmann::json& params, const cgraph::ExecContext&) const override {
    require_quadratic_l2(params, "fx.gradient");
    const Eigen::VectorXd x = math_nd::parse_vector(
        math_nd::require_input(inputs, "x", "fx.gradient"), "fx.gradient", "x");
    const Eigen::MatrixXd* Ap = nullptr;
    const Eigen::VectorXd* bp = nullptr;
    Eigen::MatrixXd Astore;
    Eigen::VectorXd bstore;
    const auto ait = inputs.find("A");
    const auto bit = inputs.find("b");
    if (ait != inputs.end() || bit != inputs.end()) {
      if (ait == inputs.end() || bit == inputs.end()) {
        throw std::invalid_argument("fx.gradient: A and b must both be set");
      }
      Astore = math_nd::parse_matrix(ait->second, "fx.gradient", "A");
      bstore = math_nd::parse_vector(bit->second, "fx.gradient", "b");
      Ap = &Astore;
      bp = &bstore;
    }
    return {{"out", math_nd::make_vec_nd(grad_quadratic_l2(x, Ap, bp))}};
  }
};

class LineSearchOp final : public cgraph::MemoryOperator {
 public:
  LineSearchOp() {
    op_id_ = "fx.line_search";
    signature_.inputs["x"] =
        cgraph::make_port("x", cgraph::PortKind::Value, "json");
    signature_.inputs["g"] =
        cgraph::make_port("g", cgraph::PortKind::Value, "json");
    signature_.inputs["lr"] =
        cgraph::make_port("lr", cgraph::PortKind::Value, "json");
    auto A = cgraph::make_port("A", cgraph::PortKind::Value, "json");
    A.optional = true;
    signature_.inputs["A"] = std::move(A);
    auto b = cgraph::make_port("b", cgraph::PortKind::Value, "json");
    b.optional = true;
    signature_.inputs["b"] = std::move(b);
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, "json");
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

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs,
      const nlohmann::json& params, const cgraph::ExecContext&) const override {
    require_quadratic_l2(params, "fx.line_search");
    const Eigen::VectorXd x = math_nd::parse_vector(
        math_nd::require_input(inputs, "x", "fx.line_search"), "fx.line_search",
        "x");
    const Eigen::VectorXd g = math_nd::parse_vector(
        math_nd::require_input(inputs, "g", "fx.line_search"), "fx.line_search",
        "g");
    double alpha = math_nd::require_number(
        math_nd::require_input(inputs, "lr", "fx.line_search"), "fx.line_search",
        "lr");
    if (!(alpha > 0.0) || !std::isfinite(alpha)) {
      throw std::invalid_argument("fx.line_search: lr must be positive finite");
    }
    const Eigen::MatrixXd* Ap = nullptr;
    const Eigen::VectorXd* bp = nullptr;
    Eigen::MatrixXd Astore;
    Eigen::VectorXd bstore;
    const auto ait = inputs.find("A");
    const auto bit = inputs.find("b");
    if (ait != inputs.end() && bit != inputs.end()) {
      Astore = math_nd::parse_matrix(ait->second, "fx.line_search", "A");
      bstore = math_nd::parse_vector(bit->second, "fx.line_search", "b");
      Ap = &Astore;
      bp = &bstore;
    }
    bool backtrack = true;
    if (params.is_object() && params.contains("backtrack") &&
        params["backtrack"].is_boolean()) {
      backtrack = params["backtrack"].get<bool>();
    }
    if (!backtrack) {
      return {{"out", alpha}};
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
    return {{"out", alpha}};
  }
};

class UpdateOp final : public cgraph::MemoryOperator {
 public:
  UpdateOp() {
    op_id_ = "fx.update";
    signature_.inputs["x"] =
        cgraph::make_port("x", cgraph::PortKind::Value, "json");
    signature_.inputs["g"] =
        cgraph::make_port("g", cgraph::PortKind::Value, "json");
    signature_.inputs["alpha"] =
        cgraph::make_port("alpha", cgraph::PortKind::Value, "json");
    auto lr = cgraph::make_port("lr", cgraph::PortKind::Value, "json");
    lr.optional = true;
    signature_.inputs["lr"] = std::move(lr);
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, "json");
    capability_.summary = "GD step: x_new = x - alpha * g";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire x, g, alpha (or lr); read out.";
    usage_.tune = "No parameters.";
    usage_.inspect = "alpha preferred; lr accepted as alias.";
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const Eigen::VectorXd x = math_nd::parse_vector(
        math_nd::require_input(inputs, "x", "fx.update"), "fx.update", "x");
    const Eigen::VectorXd g = math_nd::parse_vector(
        math_nd::require_input(inputs, "g", "fx.update"), "fx.update", "g");
    const nlohmann::json& a_json =
        math_nd::require_input_alias(inputs, "alpha", "lr", "fx.update");
    const double alpha = math_nd::require_number(a_json, "fx.update", "alpha");
    if (x.size() != g.size()) {
      throw std::invalid_argument("fx.update: x and g size mismatch");
    }
    return {{"out", math_nd::make_vec_nd(x - alpha * g)}};
  }
};

class LossCalcOp final : public cgraph::MemoryOperator {
 public:
  LossCalcOp() {
    op_id_ = "fx.loss_calc";
    signature_.inputs["x"] =
        cgraph::make_port("x", cgraph::PortKind::Value, "json");
    auto A = cgraph::make_port("A", cgraph::PortKind::Value, "json");
    A.optional = true;
    signature_.inputs["A"] = std::move(A);
    auto b = cgraph::make_port("b", cgraph::PortKind::Value, "json");
    b.optional = true;
    signature_.inputs["b"] = std::move(b);
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, "json");
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

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs,
      const nlohmann::json& params, const cgraph::ExecContext&) const override {
    require_quadratic_l2(params, "fx.loss_calc");
    const Eigen::VectorXd x = math_nd::parse_vector(
        math_nd::require_input(inputs, "x", "fx.loss_calc"), "fx.loss_calc",
        "x");
    const Eigen::MatrixXd* Ap = nullptr;
    const Eigen::VectorXd* bp = nullptr;
    Eigen::MatrixXd Astore;
    Eigen::VectorXd bstore;
    const auto ait = inputs.find("A");
    const auto bit = inputs.find("b");
    if (ait != inputs.end() && bit != inputs.end()) {
      Astore = math_nd::parse_matrix(ait->second, "fx.loss_calc", "A");
      bstore = math_nd::parse_vector(bit->second, "fx.loss_calc", "b");
      Ap = &Astore;
      bp = &bstore;
    }
    return {{"out", loss_quadratic_l2(x, Ap, bp)}};
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
