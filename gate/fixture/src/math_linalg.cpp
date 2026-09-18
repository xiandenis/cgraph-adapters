#include "cgraph/ops.hpp"
#include "fx_typed.hpp"

#include <Eigen/Core>
#include <Eigen/LU>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace fixture {
namespace {

[[noreturn]] void fail(const char* op, const std::string& msg) {
  throw std::invalid_argument(std::string(op) + ": " + msg);
}

const nlohmann::json& require_param_value(const nlohmann::json& params,
                                          const char* op) {
  if (!params.is_object() || !params.contains("value")) {
    fail(op, "params.value required");
  }
  return params["value"];
}

std::vector<double> parse_number_list(const nlohmann::json& arr, const char* op,
                                      const char* what) {
  if (!arr.is_array() || arr.empty()) {
    fail(op, std::string(what) + " must be a non-empty array of numbers");
  }
  std::vector<double> out;
  out.reserve(arr.size());
  for (const auto& x : arr) {
    if (!x.is_number()) {
      fail(op, std::string(what) + " elements must be numbers");
    }
    out.push_back(x.get<double>());
  }
  return out;
}

Eigen::VectorXd parse_vector_param(const nlohmann::json& j, const char* op) {
  if (j.is_object() && j.contains("shape") && j.contains("data")) {
    const auto data = parse_number_list(j.at("data"), op, "vector.data");
    Eigen::VectorXd v(static_cast<Eigen::Index>(data.size()));
    for (Eigen::Index i = 0; i < v.size(); ++i) {
      v(i) = data[static_cast<std::size_t>(i)];
    }
    return v;
  }
  if (!j.is_array()) {
    fail(op, "vector must be a 1-D number list");
  }
  if (!j.empty() && j[0].is_array()) {
    fail(op, "vector must be a 1-D number list, not a matrix");
  }
  const auto data = parse_number_list(j, op, "vector");
  Eigen::VectorXd v(static_cast<Eigen::Index>(data.size()));
  for (Eigen::Index i = 0; i < v.size(); ++i) {
    v(i) = data[static_cast<std::size_t>(i)];
  }
  return v;
}

Eigen::MatrixXd parse_matrix_param(const nlohmann::json& j, const char* op) {
  if (j.is_object() && j.contains("shape") && j.contains("data")) {
    const auto& shape = j.at("shape");
    if (!shape.is_array() || shape.size() != 2 || !shape[0].is_number_integer() ||
        !shape[1].is_number_integer()) {
      fail(op, "matrix.shape must be [rows, cols]");
    }
    const int rows = shape[0].get<int>();
    const int cols = shape[1].get<int>();
    if (rows <= 0 || cols <= 0) {
      fail(op, "matrix shape dims must be positive");
    }
    const auto data = parse_number_list(j.at("data"), op, "matrix.data");
    if (static_cast<int>(data.size()) != rows * cols) {
      fail(op, "matrix data length mismatch with shape");
    }
    Eigen::MatrixXd m(rows, cols);
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        m(r, c) = data[static_cast<std::size_t>(r * cols + c)];
      }
    }
    return m;
  }
  if (!j.is_array() || j.empty()) {
    fail(op, "matrix must be nested row-major lists");
  }
  if (!j[0].is_array()) {
    fail(op, "matrix must be 2-D nested lists");
  }
  const int rows = static_cast<int>(j.size());
  const int cols = static_cast<int>(j[0].size());
  Eigen::MatrixXd m(rows, cols);
  for (int r = 0; r < rows; ++r) {
    if (!j[r].is_array() || static_cast<int>(j[r].size()) != cols) {
      fail(op, "matrix rows must be equal length");
    }
    for (int c = 0; c < cols; ++c) {
      if (!j[r][c].is_number()) {
        fail(op, "matrix elements must be numbers");
      }
      m(r, c) = j[r][c].get<double>();
    }
  }
  return m;
}

class ConstMatrixOp final : public cgraph::MemoryOperator {
 public:
  ConstMatrixOp() {
    op_id_ = "fx.const_matrix";
    signature_.outputs["out"] = fx_typed::matrix_port("out");
    cgraph::ParamSpec value;
    value.name = "value";
    value.dtype = "json";
    value.bindable = false;
    value.doc = "Row-major nested list [[...],...] or {shape:[r,c], data:[...]}";
    signature_.params["value"] = std::move(value);
    capability_.summary = "Emit fx.type.matrix";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "No inputs. Read typed matrix out.";
    usage_.tune = "params.value nested lists or {shape,data}.";
    usage_.inspect = "Payload is record {rows,cols,data}.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>&, const nlohmann::json& params,
      const cgraph::ExecContext&) const override {
    const auto& value = require_param_value(params, "fx.const_matrix");
    return {{"out", fx_typed::make_matrix(
                        parse_matrix_param(value, "fx.const_matrix"))}};
  }
};

class ConstVectorOp final : public cgraph::MemoryOperator {
 public:
  ConstVectorOp() {
    op_id_ = "fx.const_vector";
    signature_.outputs["out"] = fx_typed::vector_port("out");
    cgraph::ParamSpec value;
    value.name = "value";
    value.dtype = "json";
    value.bindable = false;
    value.doc = "1-D number list [...]";
    signature_.params["value"] = std::move(value);
    capability_.summary = "Emit fx.type.vector";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "No inputs. Read typed vector out.";
    usage_.tune = "params.value = [x, y, ...].";
    usage_.inspect = "Payload is list[float].";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>&, const nlohmann::json& params,
      const cgraph::ExecContext&) const override {
    const auto& value = require_param_value(params, "fx.const_vector");
    return {{"out", fx_typed::make_vector(
                        parse_vector_param(value, "fx.const_vector"))}};
  }
};

class ConstVectorArrayOp final : public cgraph::MemoryOperator {
 public:
  ConstVectorArrayOp() {
    op_id_ = "fx.const_vector_array";
    signature_.outputs["out"] = fx_typed::vector_list_port("out");
    cgraph::ParamSpec value;
    value.name = "value";
    value.dtype = "json";
    value.bindable = false;
    value.doc = "List of vectors: [[...], [...], ...]";
    signature_.params["value"] = std::move(value);
    capability_.summary = "Emit list[fx.type.vector]";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "No inputs. Read vector list out.";
    usage_.tune = "params.value = list of equal-length vectors.";
    usage_.inspect = "Each element is fx.type.vector payload.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>&, const nlohmann::json& params,
      const cgraph::ExecContext&) const override {
    const auto& value = require_param_value(params, "fx.const_vector_array");
    if (!value.is_array() || value.empty()) {
      fail("fx.const_vector_array", "params.value must be a non-empty list of vectors");
    }
    std::vector<Eigen::VectorXd> vectors;
    Eigen::Index dim = -1;
    for (const auto& item : value) {
      const Eigen::VectorXd v = parse_vector_param(item, "fx.const_vector_array");
      if (dim < 0) {
        dim = v.size();
      } else if (v.size() != dim) {
        fail("fx.const_vector_array", "all vectors must have the same length");
      }
      vectors.push_back(v);
    }
    return {{"out", fx_typed::make_vector_list(vectors)}};
  }
};

class MatVecOp final : public cgraph::MemoryOperator {
 public:
  MatVecOp() {
    op_id_ = "fx.matvec";
    signature_.inputs["A"] = fx_typed::matrix_port("A");
    signature_.inputs["v"] = fx_typed::vector_port("v");
    signature_.outputs["out"] = fx_typed::vector_port("out");
    capability_.summary = "Matrix-vector product: out = A * v";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire matrix A and vector v; read out vector.";
    usage_.tune = "No parameters.";
    usage_.inspect = "Requires A.cols() == v.size().";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const Eigen::MatrixXd A = fx_typed::require_matrix(
        fx_typed::require_obj(data_in, "A", "fx.matvec"), "fx.matvec", "A");
    const Eigen::VectorXd v = fx_typed::require_vector(
        fx_typed::require_obj(data_in, "v", "fx.matvec"), "fx.matvec", "v");
    if (A.cols() != v.size()) {
      fail("fx.matvec", "A.cols must equal v.length");
    }
    return {{"out", fx_typed::make_vector(A * v)}};
  }
};

class VAddOp final : public cgraph::MemoryOperator {
 public:
  VAddOp() {
    op_id_ = "fx.vadd";
    signature_.inputs["a"] = fx_typed::vector_port("a");
    signature_.inputs["b"] = fx_typed::vector_port("b");
    signature_.outputs["out"] = fx_typed::vector_port("out");
    capability_.summary = "Vector add: out = a + b";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire vectors a and b; read out.";
    usage_.tune = "No parameters.";
    usage_.inspect = "Lengths must match.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const Eigen::VectorXd a = fx_typed::require_vector(
        fx_typed::require_obj(data_in, "a", "fx.vadd"), "fx.vadd", "a");
    const Eigen::VectorXd b = fx_typed::require_vector(
        fx_typed::require_obj(data_in, "b", "fx.vadd"), "fx.vadd", "b");
    if (a.size() != b.size()) {
      fail("fx.vadd", "vector lengths must match");
    }
    return {{"out", fx_typed::make_vector(a + b)}};
  }
};

class VSubOp final : public cgraph::MemoryOperator {
 public:
  VSubOp() {
    op_id_ = "fx.vsub";
    signature_.inputs["a"] = fx_typed::vector_port("a");
    signature_.inputs["b"] = fx_typed::vector_port("b");
    signature_.outputs["out"] = fx_typed::vector_port("out");
    capability_.summary = "Vector subtract: out = a - b";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire vectors a and b; read out.";
    usage_.tune = "No parameters.";
    usage_.inspect = "Lengths must match.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const Eigen::VectorXd a = fx_typed::require_vector(
        fx_typed::require_obj(data_in, "a", "fx.vsub"), "fx.vsub", "a");
    const Eigen::VectorXd b = fx_typed::require_vector(
        fx_typed::require_obj(data_in, "b", "fx.vsub"), "fx.vsub", "b");
    if (a.size() != b.size()) {
      fail("fx.vsub", "vector lengths must match");
    }
    return {{"out", fx_typed::make_vector(a - b)}};
  }
};

class L2SqOp final : public cgraph::MemoryOperator {
 public:
  L2SqOp() {
    op_id_ = "fx.l2sq";
    signature_.inputs["in"] = fx_typed::vector_port("in");
    signature_.outputs["out"] = fx_typed::float_port("out");
    capability_.summary = "Squared L2 norm of a vector";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire vector in; read scalar out.";
    usage_.tune = "No parameters.";
    usage_.inspect = "out = sum_i in[i]^2.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const Eigen::VectorXd v = fx_typed::require_vector(
        fx_typed::require_obj(data_in, "in", "fx.l2sq"), "fx.l2sq", "in");
    return {{"out", fx_typed::make_number_float(v.squaredNorm())}};
  }
};

class SumReduceOp final : public cgraph::MemoryOperator {
 public:
  SumReduceOp() {
    op_id_ = "fx.sum_reduce";
    signature_.inputs["in"] = fx_typed::float_list_port("in");
    signature_.outputs["out"] = fx_typed::float_port("out");
    capability_.summary = "Sum a list of floats";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire list[float] into in; read out.";
    usage_.tune = "No parameters.";
    usage_.inspect = "out = sum of elements.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const auto values = fx_typed::require_float_list(
        fx_typed::require_obj(data_in, "in", "fx.sum_reduce"), "fx.sum_reduce",
        "in");
    double s = 0.0;
    for (double x : values) {
      s += x;
    }
    return {{"out", fx_typed::make_number_float(s)}};
  }
};

class QuadraticFormOp final : public cgraph::MemoryOperator {
 public:
  QuadraticFormOp() {
    op_id_ = "fx.quadratic_form";
    signature_.inputs["A"] = fx_typed::matrix_port("A");
    signature_.inputs["p"] = fx_typed::vector_port("p");
    signature_.outputs["out"] = fx_typed::float_port("out");
    capability_.summary = "Quadratic form: out = p^T A p";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire matrix A and vector p; read scalar out.";
    usage_.tune = "No parameters.";
    usage_.inspect = "Requires square A with A.rows() == p.size().";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const Eigen::MatrixXd A = fx_typed::require_matrix(
        fx_typed::require_obj(data_in, "A", "fx.quadratic_form"),
        "fx.quadratic_form", "A");
    const Eigen::VectorXd p = fx_typed::require_vector(
        fx_typed::require_obj(data_in, "p", "fx.quadratic_form"),
        "fx.quadratic_form", "p");
    if (A.rows() != A.cols() || A.rows() != p.size()) {
      fail("fx.quadratic_form", "A must be square with size == p.length");
    }
    return {{"out", fx_typed::make_number_float(p.dot(A * p))}};
  }
};

class DetOp final : public cgraph::MemoryOperator {
 public:
  DetOp() {
    op_id_ = "fx.det";
    signature_.inputs["in"] = fx_typed::matrix_port("in");
    signature_.outputs["out"] = fx_typed::float_port("out");
    capability_.summary = "Determinant of a square matrix";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire square matrix in; read scalar out.";
    usage_.tune = "No parameters. Uses Eigen LU.";
    usage_.inspect = "Requires square matrix.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const Eigen::MatrixXd A = fx_typed::require_matrix(
        fx_typed::require_obj(data_in, "in", "fx.det"), "fx.det", "in");
    if (A.rows() != A.cols() || A.rows() < 1) {
      fail("fx.det", "matrix must be square");
    }
    return {{"out", fx_typed::make_number_float(A.determinant())}};
  }
};

class TraceOp final : public cgraph::MemoryOperator {
 public:
  TraceOp() {
    op_id_ = "fx.trace";
    signature_.inputs["in"] = fx_typed::matrix_port("in");
    signature_.outputs["out"] = fx_typed::float_port("out");
    capability_.summary = "Trace of a square matrix";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire square matrix in; read scalar out.";
    usage_.tune = "No parameters.";
    usage_.inspect = "out = sum of diagonal.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const Eigen::MatrixXd A = fx_typed::require_matrix(
        fx_typed::require_obj(data_in, "in", "fx.trace"), "fx.trace", "in");
    if (A.rows() != A.cols() || A.rows() < 1) {
      fail("fx.trace", "matrix must be square");
    }
    return {{"out", fx_typed::make_number_float(A.trace())}};
  }
};

class TransposeOp final : public cgraph::MemoryOperator {
 public:
  TransposeOp() {
    op_id_ = "fx.transpose";
    signature_.inputs["in"] = fx_typed::matrix_port("in");
    signature_.outputs["out"] = fx_typed::matrix_port("out");
    capability_.summary = "Matrix transpose";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire matrix in; read transposed matrix out.";
    usage_.tune = "No parameters.";
    usage_.inspect = "out = in^T.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const Eigen::MatrixXd A = fx_typed::require_matrix(
        fx_typed::require_obj(data_in, "in", "fx.transpose"), "fx.transpose",
        "in");
    return {{"out", fx_typed::make_matrix(A.transpose())}};
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_const_matrix() {
  return std::make_shared<ConstMatrixOp>();
}
std::shared_ptr<cgraph::MemoryOperator> make_const_vector() {
  return std::make_shared<ConstVectorOp>();
}
std::shared_ptr<cgraph::MemoryOperator> make_const_vector_array() {
  return std::make_shared<ConstVectorArrayOp>();
}
std::shared_ptr<cgraph::MemoryOperator> make_matvec() {
  return std::make_shared<MatVecOp>();
}
std::shared_ptr<cgraph::MemoryOperator> make_vadd() {
  return std::make_shared<VAddOp>();
}
std::shared_ptr<cgraph::MemoryOperator> make_vsub() {
  return std::make_shared<VSubOp>();
}
std::shared_ptr<cgraph::MemoryOperator> make_l2sq() {
  return std::make_shared<L2SqOp>();
}
std::shared_ptr<cgraph::MemoryOperator> make_sum_reduce() {
  return std::make_shared<SumReduceOp>();
}
std::shared_ptr<cgraph::MemoryOperator> make_quadratic_form() {
  return std::make_shared<QuadraticFormOp>();
}
std::shared_ptr<cgraph::MemoryOperator> make_det() {
  return std::make_shared<DetOp>();
}
std::shared_ptr<cgraph::MemoryOperator> make_trace() {
  return std::make_shared<TraceOp>();
}
std::shared_ptr<cgraph::MemoryOperator> make_transpose() {
  return std::make_shared<TransposeOp>();
}

}  // namespace fixture
