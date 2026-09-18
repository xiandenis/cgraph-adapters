#pragma once

#include "fixture/fx_types.hpp"

#include "cgraph/data_helpers.hpp"
#include "cgraph/data_object.hpp"
#include "cgraph/ops.hpp"

#include <Eigen/Core>

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fixture {
namespace fx_typed {

inline cgraph::SemanticSpec number_sem() {
  return cgraph::SemanticSpec::of("cgraph.semantic.number");
}

inline cgraph::SemanticSpec predicate_sem() {
  return cgraph::SemanticSpec::of("cgraph.semantic.predicate");
}

inline cgraph::SemanticSpec count_sem() {
  return cgraph::SemanticSpec::of("cgraph.semantic.count");
}

inline const cgraph::DataObject& require_obj(
    const std::map<std::string, cgraph::DataObject>& inputs,
    const std::string& name, std::string_view op) {
  return cgraph::require_data(inputs, name, op);
}

inline double require_float(const cgraph::DataObject& obj, std::string_view op,
                            std::string_view port) {
  if (obj.type_id != cgraph::type_ids::floating() &&
      obj.payload.kind() != cgraph::Payload::Kind::Float) {
    // Accept int promotion only when payload is Int typed as int.
    if (obj.payload.kind() == cgraph::Payload::Kind::Int) {
      return static_cast<double>(obj.payload.as_int());
    }
    throw std::invalid_argument(std::string(op) + ": '" + std::string(port) +
                                "' must be float");
  }
  if (obj.payload.kind() != cgraph::Payload::Kind::Float) {
    throw std::invalid_argument(std::string(op) + ": '" + std::string(port) +
                                "' payload must be Float");
  }
  return obj.payload.as_float();
}

inline double require_float_port(
    const std::map<std::string, cgraph::DataObject>& inputs,
    const std::string& name, std::string_view op) {
  return require_float(require_obj(inputs, name, op), op, name);
}

inline bool require_bool(const cgraph::DataObject& obj, std::string_view op,
                         std::string_view port) {
  if (obj.payload.kind() != cgraph::Payload::Kind::Bool) {
    throw std::invalid_argument(std::string(op) + ": '" + std::string(port) +
                                "' must be bool");
  }
  return obj.payload.as_bool();
}

inline std::int64_t require_int(const cgraph::DataObject& obj,
                                std::string_view op, std::string_view port) {
  if (obj.payload.kind() == cgraph::Payload::Kind::Int) {
    return obj.payload.as_int();
  }
  if (obj.payload.kind() == cgraph::Payload::Kind::Float) {
    return static_cast<std::int64_t>(obj.payload.as_float());
  }
  throw std::invalid_argument(std::string(op) + ": '" + std::string(port) +
                              "' must be int");
}

inline Eigen::VectorXd require_vector(const cgraph::DataObject& obj,
                                      std::string_view op,
                                      std::string_view port) {
  if (obj.type_id != type_vector()) {
    throw std::invalid_argument(std::string(op) + ": '" + std::string(port) +
                                "' must be fx.type.vector");
  }
  if (obj.payload.kind() != cgraph::Payload::Kind::List) {
    throw std::invalid_argument(std::string(op) + ": '" + std::string(port) +
                                "' vector payload must be list");
  }
  const auto& items = obj.payload.as_list();
  if (items.empty()) {
    throw std::invalid_argument(std::string(op) + ": '" + std::string(port) +
                                "' empty vector");
  }
  Eigen::VectorXd v(static_cast<Eigen::Index>(items.size()));
  for (Eigen::Index i = 0; i < v.size(); ++i) {
    if (items[static_cast<std::size_t>(i)].kind() !=
        cgraph::Payload::Kind::Float) {
      throw std::invalid_argument(std::string(op) + ": '" + std::string(port) +
                                  "' vector elements must be float");
    }
    v(i) = items[static_cast<std::size_t>(i)].as_float();
  }
  return v;
}

inline Eigen::MatrixXd require_matrix(const cgraph::DataObject& obj,
                                      std::string_view op,
                                      std::string_view port) {
  if (obj.type_id != type_matrix()) {
    throw std::invalid_argument(std::string(op) + ": '" + std::string(port) +
                                "' must be fx.type.matrix");
  }
  if (obj.payload.kind() != cgraph::Payload::Kind::Record) {
    throw std::invalid_argument(std::string(op) + ": '" + std::string(port) +
                                "' matrix payload must be record");
  }
  const auto& fields = obj.payload.as_record();
  const auto rit = fields.find("rows");
  const auto cit = fields.find("cols");
  const auto dit = fields.find("data");
  if (rit == fields.end() || cit == fields.end() || dit == fields.end()) {
    throw std::invalid_argument(std::string(op) + ": '" + std::string(port) +
                                "' matrix missing rows/cols/data");
  }
  const int rows_i = rit->second.kind() == cgraph::Payload::Kind::Int
                         ? static_cast<int>(rit->second.as_int())
                         : static_cast<int>(rit->second.as_float());
  const int cols_i = cit->second.kind() == cgraph::Payload::Kind::Int
                         ? static_cast<int>(cit->second.as_int())
                         : static_cast<int>(cit->second.as_float());
  if (rows_i <= 0 || cols_i <= 0) {
    throw std::invalid_argument(std::string(op) + ": '" + std::string(port) +
                                "' matrix shape must be positive");
  }
  if (dit->second.kind() != cgraph::Payload::Kind::List) {
    throw std::invalid_argument(std::string(op) + ": '" + std::string(port) +
                                "' matrix data must be list");
  }
  const auto& data = dit->second.as_list();
  if (static_cast<int>(data.size()) != rows_i * cols_i) {
    throw std::invalid_argument(std::string(op) + ": '" + std::string(port) +
                                "' matrix data length mismatch");
  }
  Eigen::MatrixXd m(rows_i, cols_i);
  for (int r = 0; r < rows_i; ++r) {
    for (int c = 0; c < cols_i; ++c) {
      const auto& cell = data[static_cast<std::size_t>(r * cols_i + c)];
      if (cell.kind() != cgraph::Payload::Kind::Float) {
        throw std::invalid_argument(std::string(op) + ": '" +
                                    std::string(port) +
                                    "' matrix data must be floats");
      }
      m(r, c) = cell.as_float();
    }
  }
  return m;
}

inline std::vector<double> require_float_list(const cgraph::DataObject& obj,
                                              std::string_view op,
                                              std::string_view port) {
  if (obj.payload.kind() != cgraph::Payload::Kind::List) {
    throw std::invalid_argument(std::string(op) + ": '" + std::string(port) +
                                "' must be list[float]");
  }
  std::vector<double> out;
  out.reserve(obj.payload.as_list().size());
  for (const auto& item : obj.payload.as_list()) {
    if (item.kind() != cgraph::Payload::Kind::Float) {
      throw std::invalid_argument(std::string(op) + ": '" + std::string(port) +
                                  "' list elements must be float");
    }
    out.push_back(item.as_float());
  }
  return out;
}

inline cgraph::DataObject make_number_float(double value) {
  return cgraph::make_float(value, number_sem());
}

inline cgraph::DataObject make_bool_pred(bool value) {
  return cgraph::make_bool(value, predicate_sem());
}

inline cgraph::DataObject make_count_int(std::int64_t value) {
  return cgraph::make_int(value, count_sem());
}

inline cgraph::DataObject make_vector(const Eigen::VectorXd& v) {
  std::vector<cgraph::Payload> items;
  items.reserve(static_cast<std::size_t>(v.size()));
  for (Eigen::Index i = 0; i < v.size(); ++i) {
    items.push_back(cgraph::Payload::floating(v(i)));
  }
  return cgraph::make_data_object(type_vector(), number_sem(),
                                  cgraph::Payload::list(std::move(items)));
}

inline cgraph::DataObject make_matrix(const Eigen::MatrixXd& m) {
  const int rows = static_cast<int>(m.rows());
  const int cols = static_cast<int>(m.cols());
  std::vector<cgraph::Payload> data;
  data.reserve(static_cast<std::size_t>(rows * cols));
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      data.push_back(cgraph::Payload::floating(m(r, c)));
    }
  }
  std::map<std::string, cgraph::Payload> fields;
  fields.emplace("rows", cgraph::Payload::integer(rows));
  fields.emplace("cols", cgraph::Payload::integer(cols));
  fields.emplace("data", cgraph::Payload::list(std::move(data)));
  return cgraph::make_data_object(type_matrix(), number_sem(),
                                  cgraph::Payload::record(std::move(fields)));
}

inline cgraph::DataObject make_float_list(const std::vector<double>& values) {
  std::vector<cgraph::Payload> items;
  items.reserve(values.size());
  for (double x : values) {
    items.push_back(cgraph::Payload::floating(x));
  }
  return cgraph::make_data_object(type_float_list(), number_sem(),
                                  cgraph::Payload::list(std::move(items)));
}

inline cgraph::DataObject make_vector_list(
    const std::vector<Eigen::VectorXd>& vectors) {
  std::vector<cgraph::Payload> items;
  items.reserve(vectors.size());
  for (const auto& v : vectors) {
    items.push_back(make_vector(v).payload);
  }
  return cgraph::make_data_object(type_vector_list(), number_sem(),
                                  cgraph::Payload::list(std::move(items)));
}

inline cgraph::PortSpec float_port(std::string name) {
  return cgraph::make_port(std::move(name), cgraph::PortKind::Value,
                           cgraph::type_ids::floating(), number_sem());
}

inline cgraph::PortSpec vector_port(std::string name) {
  return cgraph::make_port(std::move(name), cgraph::PortKind::Value,
                           type_vector(), number_sem());
}

inline cgraph::PortSpec matrix_port(std::string name) {
  return cgraph::make_port(std::move(name), cgraph::PortKind::Value,
                           type_matrix(), number_sem());
}

inline cgraph::PortSpec bool_port(std::string name) {
  return cgraph::make_port(std::move(name), cgraph::PortKind::Value,
                           cgraph::type_ids::boolean(), predicate_sem());
}

inline cgraph::PortSpec int_port(std::string name) {
  return cgraph::make_port(std::move(name), cgraph::PortKind::Value,
                           cgraph::type_ids::integer(), count_sem());
}

inline cgraph::PortSpec float_list_port(std::string name) {
  return cgraph::make_port(std::move(name), cgraph::PortKind::Value,
                           type_float_list(), number_sem());
}

inline cgraph::PortSpec vector_list_port(std::string name) {
  return cgraph::make_port(std::move(name), cgraph::PortKind::Value,
                           type_vector_list(), number_sem());
}

inline cgraph::PortSpec matrix_list_port(std::string name) {
  return cgraph::make_port(std::move(name), cgraph::PortKind::Value,
                           type_matrix_list(), number_sem());
}

inline std::vector<Eigen::MatrixXd> require_matrix_list(
    const cgraph::DataObject& obj, std::string_view op,
    std::string_view port) {
  if (obj.type_id != type_matrix_list()) {
    throw std::invalid_argument(std::string(op) + ": '" + std::string(port) +
                                "' must be list[fx.type.matrix]");
  }
  if (obj.payload.kind() != cgraph::Payload::Kind::List) {
    throw std::invalid_argument(std::string(op) + ": '" + std::string(port) +
                                "' matrix list payload must be list");
  }
  std::vector<Eigen::MatrixXd> out;
  out.reserve(obj.payload.as_list().size());
  for (const auto& item : obj.payload.as_list()) {
    cgraph::DataObject tmp =
        cgraph::make_data_object(type_matrix(), number_sem(), item);
    out.push_back(require_matrix(tmp, op, port));
  }
  return out;
}

inline cgraph::DataObject make_matrix_list(
    const std::vector<Eigen::MatrixXd>& mats) {
  std::vector<cgraph::Payload> items;
  items.reserve(mats.size());
  for (const auto& m : mats) {
    items.push_back(make_matrix(m).payload);
  }
  return cgraph::make_data_object(type_matrix_list(), number_sem(),
                                  cgraph::Payload::list(std::move(items)));
}

inline cgraph::PortSpec directory_artifact_port(std::string name) {
  return cgraph::make_port(std::move(name), cgraph::PortKind::Artifact,
                           cgraph::type_ids::untyped(),
                           cgraph::SemanticSpec::of("cgraph.semantic.directory"));
}

inline cgraph::PortSpec string_list_port(std::string name) {
  return cgraph::make_port(std::move(name), cgraph::PortKind::Value,
                           type_string_list(),
                           cgraph::SemanticSpec::of("cgraph.semantic.text"));
}

inline std::vector<std::string> require_string_list(const cgraph::DataObject& obj,
                                                    std::string_view op,
                                                    std::string_view port) {
  if (obj.payload.kind() != cgraph::Payload::Kind::List) {
    throw std::invalid_argument(std::string(op) + ": '" + std::string(port) +
                                "' must be list[string]");
  }
  std::vector<std::string> out;
  out.reserve(obj.payload.as_list().size());
  for (const auto& item : obj.payload.as_list()) {
    if (item.kind() != cgraph::Payload::Kind::String) {
      throw std::invalid_argument(std::string(op) + ": '" + std::string(port) +
                                  "' list elements must be string");
    }
    out.push_back(item.as_string());
  }
  return out;
}

inline cgraph::DataObject make_string_list(const std::vector<std::string>& values) {
  std::vector<cgraph::Payload> items;
  items.reserve(values.size());
  for (const auto& s : values) {
    items.push_back(cgraph::Payload::string(s));
  }
  return cgraph::make_data_object(
      type_string_list(), cgraph::SemanticSpec::of("cgraph.semantic.text"),
      cgraph::Payload::list(std::move(items)));
}

inline cgraph::DataObject make_directory_artifact(const std::string& path,
                                                  const std::string& content_hash) {
  cgraph::DataRef ref;
  ref.uri = path;
  ref.content_hash = content_hash;
  ref.is_dir = true;
  return cgraph::make_artifact_object(
      cgraph::type_ids::untyped(),
      cgraph::SemanticSpec::of("cgraph.semantic.directory"), std::move(ref));
}

inline bool is_directory_artifact(const cgraph::DataObject& obj) {
  return obj.payload.kind() == cgraph::Payload::Kind::DataRef &&
         obj.payload.as_data_ref().is_dir;
}

inline std::string require_directory_uri(const cgraph::DataObject& obj,
                                         std::string_view op,
                                         std::string_view port) {
  if (!is_directory_artifact(obj)) {
    throw std::invalid_argument(std::string(op) + ": '" + std::string(port) +
                                "' must be directory Artifact");
  }
  return obj.payload.as_data_ref().uri;
}

}  // namespace fx_typed
}  // namespace fixture
