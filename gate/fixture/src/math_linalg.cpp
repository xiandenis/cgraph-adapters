#include "cgraph/ops.hpp"
#include "fx_data.hpp"

#include <Eigen/Core>
#include <Eigen/LU>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fixture {
namespace {

// JSON convention (OP-C): nested row-major lists.
//   vector:        [x, y, ...]
//   matrix:        [[r0c0, r0c1, ...], ...]
//   vector_array:  [[...], [...], ...]
// const_matrix also accepts { "shape": [rows, cols], "data": [...] } (row-major flat)
// and normalizes to nested lists on output.

[[noreturn]] void fail(const char* op, const std::string& msg) {
  throw std::invalid_argument(std::string(op) + ": " + msg);
}

const nlohmann::json& require_param_value(const nlohmann::json& params, const char* op) {
  if (!params.is_object() || !params.contains("value")) {
    fail(op, "params.value required");
  }
  return params["value"];
}

std::vector<double> parse_number_list(const nlohmann::json& arr, const char* op,
                                      const char* what) {
  if (!arr.is_array() || arr.empty()) {
    fail(op, std::string(what) + " must be a non-empty JSON array of numbers");
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

Eigen::VectorXd parse_vector(const nlohmann::json& j, const char* op) {
  if (j.is_object() && j.contains("shape") && j.contains("data")) {
    const auto& shape = j.at("shape");
    const auto data = parse_number_list(j.at("data"), op, "vector.data");
    if (!shape.is_array() || shape.empty()) {
      fail(op, "vector.shape must be a non-empty array");
    }
    std::size_t n = 1;
    for (const auto& d : shape) {
      if (!d.is_number_integer() || d.get<int>() <= 0) {
        fail(op, "vector.shape dims must be positive integers");
      }
      n *= static_cast<std::size_t>(d.get<int>());
    }
    if (data.size() != n) {
      fail(op, "vector data length mismatch with shape");
    }
    Eigen::VectorXd v(static_cast<Eigen::Index>(data.size()));
    for (Eigen::Index i = 0; i < v.size(); ++i) {
      v(i) = data[static_cast<std::size_t>(i)];
    }
    return v;
  }
  if (!j.is_array()) {
    fail(op, "vector must be a nested number list (or {shape,data})");
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

Eigen::MatrixXd parse_matrix(const nlohmann::json& j, const char* op) {
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
    Eigen::MatrixXd M(rows, cols);
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        M(r, c) = data[static_cast<std::size_t>(r * cols + c)];
      }
    }
    return M;
  }
  if (!j.is_array() || j.empty() || !j[0].is_array()) {
    fail(op, "matrix must be a row-major nested list (or {shape,data})");
  }
  const int rows = static_cast<int>(j.size());
  const int cols = static_cast<int>(j[0].size());
  if (cols <= 0) {
    fail(op, "matrix rows must be non-empty");
  }
  Eigen::MatrixXd M(rows, cols);
  for (int r = 0; r < rows; ++r) {
    if (!j[r].is_array() || static_cast<int>(j[r].size()) != cols) {
      fail(op, "matrix must be rectangular");
    }
    for (int c = 0; c < cols; ++c) {
      if (!j[r][c].is_number()) {
        fail(op, "matrix elements must be numbers");
      }
      M(r, c) = j[r][c].get<double>();
    }
  }
  return M;
}

nlohmann::json vector_to_json(const Eigen::VectorXd& v) {
  nlohmann::json out = nlohmann::json::array();
  for (Eigen::Index i = 0; i < v.size(); ++i) {
    out.push_back(v(i));
  }
  return out;
}

nlohmann::json matrix_to_json(const Eigen::MatrixXd& M) {
  nlohmann::json out = nlohmann::json::array();
  for (Eigen::Index r = 0; r < M.rows(); ++r) {
    nlohmann::json row = nlohmann::json::array();
    for (Eigen::Index c = 0; c < M.cols(); ++c) {
      row.push_back(M(r, c));
    }
    out.push_back(std::move(row));
  }
  return out;
}

const nlohmann::json& require_input(const std::map<std::string, nlohmann::json>& inputs,
                                    const std::string& name, const char* op) {
  const auto it = inputs.find(name);
  if (it == inputs.end()) {
    fail(op, "missing input '" + name + "'");
  }
  return it->second;
}

class ConstMatrixOp final : public cgraph::MemoryOperator {
 public:
  ConstMatrixOp() {
    op_id_ = "fx.const_matrix";
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    cgraph::ParamSpec value;
    value.name = "value";
    value.dtype = "json";
    value.doc = "Row-major nested list [[...],...] or {shape:[r,c], data:[...]}";
    signature_.params["value"] = std::move(value);
    capability_.summary = "Emit a matrix as nested row-major JSON lists";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "No inputs. Read out (nested row-major matrix).";
    usage_.tune =
        "params.value = [[...],...] or {shape:[rows,cols], data:[row-major flat]}.";
    usage_.inspect = "Output is always nested lists (normalized).";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>&, const nlohmann::json& params,
      const cgraph::ExecContext&) const override {
    const auto& value = require_param_value(params, "fx.const_matrix");
    return fx::wrap(signature_, {{"out", matrix_to_json(parse_matrix(value, "fx.const_matrix"))}});
  }
};

class ConstVectorOp final : public cgraph::MemoryOperator {
 public:
  ConstVectorOp() {
    op_id_ = "fx.const_vector";
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    cgraph::ParamSpec value;
    value.name = "value";
    value.dtype = "json";
    value.doc = "1-D number list [...]";
    signature_.params["value"] = std::move(value);
    capability_.summary = "Emit a vector as a JSON number list";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "No inputs. Read out ([x,y,...]).";
    usage_.tune = "params.value = [x, y, ...].";
    usage_.inspect = "JSON convention: nested lists (1-D for vectors).";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>&, const nlohmann::json& params,
      const cgraph::ExecContext&) const override {
    const auto& value = require_param_value(params, "fx.const_vector");
    return fx::wrap(signature_, {{"out", vector_to_json(parse_vector(value, "fx.const_vector"))}});
  }
};

class ConstVectorArrayOp final : public cgraph::MemoryOperator {
 public:
  ConstVectorArrayOp() {
    op_id_ = "fx.const_vector_array";
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    cgraph::ParamSpec value;
    value.name = "value";
    value.dtype = "json";
    value.doc = "List of vectors: [[...], [...], ...]";
    signature_.params["value"] = std::move(value);
    capability_.summary = "Emit a list of vectors as nested JSON lists";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "No inputs. Read out ([[...],...]).";
    usage_.tune = "params.value = list of equal-length vectors.";
    usage_.inspect = "Each element validated as a 1-D vector.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>&, const nlohmann::json& params,
      const cgraph::ExecContext&) const override {
    const auto& value = require_param_value(params, "fx.const_vector_array");
    if (!value.is_array() || value.empty()) {
      fail("fx.const_vector_array", "params.value must be a non-empty list of vectors");
    }
    nlohmann::json out = nlohmann::json::array();
    Eigen::Index dim = -1;
    for (const auto& item : value) {
      const Eigen::VectorXd v = parse_vector(item, "fx.const_vector_array");
      if (dim < 0) {
        dim = v.size();
      } else if (v.size() != dim) {
        fail("fx.const_vector_array", "all vectors must have the same length");
      }
      out.push_back(vector_to_json(v));
    }
    return fx::wrap(signature_, {{"out", std::move(out)}});
  }
};

class MatVecOp final : public cgraph::MemoryOperator {
 public:
  MatVecOp() {
    op_id_ = "fx.matvec";
    signature_.inputs["A"] =
        cgraph::make_port("A", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.inputs["v"] =
        cgraph::make_port("v", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    capability_.summary = "Matrix-vector product: out = A * v";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire matrix A and vector v; read out vector.";
    usage_.tune = "No parameters. Nested row-major lists.";
    usage_.inspect = "Requires A.cols() == v.size().";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const auto inputs = fx::unwrap(data_in);
    const Eigen::MatrixXd A = parse_matrix(require_input(inputs, "A", "fx.matvec"), "fx.matvec");
    const Eigen::VectorXd v = parse_vector(require_input(inputs, "v", "fx.matvec"), "fx.matvec");
    if (A.cols() != v.size()) {
      fail("fx.matvec", "A.cols must equal v.length");
    }
    return fx::wrap(signature_, {{"out", vector_to_json(A * v)}});
  }
};

class VAddOp final : public cgraph::MemoryOperator {
 public:
  VAddOp() {
    op_id_ = "fx.vadd";
    signature_.inputs["a"] =
        cgraph::make_port("a", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.inputs["b"] =
        cgraph::make_port("b", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
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
    const auto inputs = fx::unwrap(data_in);
    const Eigen::VectorXd a = parse_vector(require_input(inputs, "a", "fx.vadd"), "fx.vadd");
    const Eigen::VectorXd b = parse_vector(require_input(inputs, "b", "fx.vadd"), "fx.vadd");
    if (a.size() != b.size()) {
      fail("fx.vadd", "vector lengths must match");
    }
    return fx::wrap(signature_, {{"out", vector_to_json(a + b)}});
  }
};

class VSubOp final : public cgraph::MemoryOperator {
 public:
  VSubOp() {
    op_id_ = "fx.vsub";
    signature_.inputs["a"] =
        cgraph::make_port("a", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.inputs["b"] =
        cgraph::make_port("b", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
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
    const auto inputs = fx::unwrap(data_in);
    const Eigen::VectorXd a = parse_vector(require_input(inputs, "a", "fx.vsub"), "fx.vsub");
    const Eigen::VectorXd b = parse_vector(require_input(inputs, "b", "fx.vsub"), "fx.vsub");
    if (a.size() != b.size()) {
      fail("fx.vsub", "vector lengths must match");
    }
    return fx::wrap(signature_, {{"out", vector_to_json(a - b)}});
  }
};

class L2SqOp final : public cgraph::MemoryOperator {
 public:
  L2SqOp() {
    op_id_ = "fx.l2sq";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
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
    const auto inputs = fx::unwrap(data_in);
    const Eigen::VectorXd v = parse_vector(require_input(inputs, "in", "fx.l2sq"), "fx.l2sq");
    return fx::wrap(signature_, {{"out", v.squaredNorm()}});
  }
};

class SumReduceOp final : public cgraph::MemoryOperator {
 public:
  SumReduceOp() {
    op_id_ = "fx.sum_reduce";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    capability_.summary = "Sum a list of scalars";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire a JSON array of numbers into in; read out.";
    usage_.tune = "No parameters.";
    usage_.inspect = "Also accepts a single number (identity).";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const auto inputs = fx::unwrap(data_in);
    const auto& in = require_input(inputs, "in", "fx.sum_reduce");
    if (in.is_number()) {
      return fx::wrap(signature_, {{"out", in.get<double>()}});
    }
    if (!in.is_array()) {
      fail("fx.sum_reduce", "in must be a number or array of numbers");
    }
    double s = 0.0;
    for (const auto& x : in) {
      if (!x.is_number()) {
        fail("fx.sum_reduce", "array elements must be numbers");
      }
      s += x.get<double>();
    }
    return fx::wrap(signature_, {{"out", s}});
  }
};

class QuadraticFormOp final : public cgraph::MemoryOperator {
 public:
  QuadraticFormOp() {
    op_id_ = "fx.quadratic_form";
    signature_.inputs["A"] =
        cgraph::make_port("A", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.inputs["p"] =
        cgraph::make_port("p", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
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
    const auto inputs = fx::unwrap(data_in);
    const Eigen::MatrixXd A =
        parse_matrix(require_input(inputs, "A", "fx.quadratic_form"), "fx.quadratic_form");
    const Eigen::VectorXd p =
        parse_vector(require_input(inputs, "p", "fx.quadratic_form"), "fx.quadratic_form");
    if (A.rows() != A.cols() || A.rows() != p.size()) {
      fail("fx.quadratic_form", "A must be square with size == p.length");
    }
    return fx::wrap(signature_, {{"out", p.dot(A * p)}});
  }
};

class DetOp final : public cgraph::MemoryOperator {
 public:
  DetOp() {
    op_id_ = "fx.det";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    capability_.summary = "Determinant of a square matrix";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire square matrix in; read scalar out.";
    usage_.tune = "No parameters. Uses Eigen LU.";
    usage_.inspect = "Requires square matrix (2x2+).";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const auto inputs = fx::unwrap(data_in);
    const Eigen::MatrixXd A = parse_matrix(require_input(inputs, "in", "fx.det"), "fx.det");
    if (A.rows() != A.cols() || A.rows() < 1) {
      fail("fx.det", "matrix must be square");
    }
    return fx::wrap(signature_, {{"out", A.determinant()}});
  }
};

class TraceOp final : public cgraph::MemoryOperator {
 public:
  TraceOp() {
    op_id_ = "fx.trace";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
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
    const auto inputs = fx::unwrap(data_in);
    const Eigen::MatrixXd A = parse_matrix(require_input(inputs, "in", "fx.trace"), "fx.trace");
    if (A.rows() != A.cols() || A.rows() < 1) {
      fail("fx.trace", "matrix must be square");
    }
    return fx::wrap(signature_, {{"out", A.trace()}});
  }
};

class TransposeOp final : public cgraph::MemoryOperator {
 public:
  TransposeOp() {
    op_id_ = "fx.transpose";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    capability_.summary = "Matrix transpose";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire matrix in; read transposed nested-list out.";
    usage_.tune = "No parameters.";
    usage_.inspect = "out[i][j] = in[j][i].";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const auto inputs = fx::unwrap(data_in);
    const Eigen::MatrixXd A =
        parse_matrix(require_input(inputs, "in", "fx.transpose"), "fx.transpose");
    return fx::wrap(signature_, {{"out", matrix_to_json(A.transpose())}});
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
