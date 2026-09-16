#pragma once

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/QR>
#include <Eigen/SVD>

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace fixture {
namespace math_nd {

// Prefer OP-C nested-list convention on output; still accept {shape,data}.
inline nlohmann::json make_nd(const Eigen::MatrixXd& m) {
  nlohmann::json out = nlohmann::json::array();
  for (Eigen::Index r = 0; r < m.rows(); ++r) {
    nlohmann::json row = nlohmann::json::array();
    for (Eigen::Index c = 0; c < m.cols(); ++c) {
      row.push_back(m(r, c));
    }
    out.push_back(std::move(row));
  }
  return out;
}

inline nlohmann::json make_vec_nd(const Eigen::VectorXd& v) {
  nlohmann::json out = nlohmann::json::array();
  for (Eigen::Index i = 0; i < v.size(); ++i) {
    out.push_back(v(i));
  }
  return out;
}

inline Eigen::MatrixXd parse_matrix(const nlohmann::json& j, const char* op,
                                    const char* port) {
  if (j.is_object() && j.contains("data") && j.contains("shape")) {
    const auto& shape = j.at("shape");
    if (!shape.is_array() || shape.size() < 1 || shape.size() > 2) {
      throw std::invalid_argument(std::string(op) + ": '" + port +
                                  "' shape must be 1D or 2D");
    }
    const int rows = shape[0].get<int>();
    const int cols = shape.size() == 1 ? 1 : shape[1].get<int>();
    if (rows <= 0 || cols <= 0) {
      throw std::invalid_argument(std::string(op) + ": '" + port +
                                  "' empty shape");
    }
    const auto& data = j.at("data");
    if (!data.is_array() ||
        static_cast<int>(data.size()) != rows * cols) {
      throw std::invalid_argument(std::string(op) + ": '" + port +
                                  "' data length mismatch");
    }
    Eigen::MatrixXd m(rows, cols);
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        m(r, c) = data[static_cast<std::size_t>(r * cols + c)].get<double>();
      }
    }
    return m;
  }
  if (j.is_array()) {
    if (j.empty()) {
      throw std::invalid_argument(std::string(op) + ": '" + port +
                                  "' empty matrix");
    }
    if (j[0].is_number()) {
      Eigen::MatrixXd m(static_cast<Eigen::Index>(j.size()), 1);
      for (std::size_t i = 0; i < j.size(); ++i) {
        m(static_cast<Eigen::Index>(i), 0) = j[i].get<double>();
      }
      return m;
    }
    if (j[0].is_array()) {
      const int rows = static_cast<int>(j.size());
      const int cols = static_cast<int>(j[0].size());
      Eigen::MatrixXd m(rows, cols);
      for (int r = 0; r < rows; ++r) {
        if (!j[static_cast<std::size_t>(r)].is_array() ||
            static_cast<int>(j[static_cast<std::size_t>(r)].size()) != cols) {
          throw std::invalid_argument(std::string(op) + ": '" + port +
                                      "' ragged nested array");
        }
        for (int c = 0; c < cols; ++c) {
          m(r, c) = j[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)]
                        .get<double>();
        }
      }
      return m;
    }
  }
  throw std::invalid_argument(std::string(op) + ": '" + port +
                              "' must be nested array or ndarray JSON");
}

inline Eigen::VectorXd parse_vector(const nlohmann::json& j, const char* op,
                                    const char* port) {
  Eigen::MatrixXd m = parse_matrix(j, op, port);
  if (m.cols() == 1) {
    return m.col(0);
  }
  if (m.rows() == 1) {
    return m.row(0).transpose();
  }
  throw std::invalid_argument(std::string(op) + ": '" + port +
                              "' must be a vector");
}

inline const nlohmann::json& require_input(
    const std::map<std::string, nlohmann::json>& inputs,
    const std::string& name, const char* op) {
  const auto it = inputs.find(name);
  if (it == inputs.end()) {
    throw std::invalid_argument(std::string(op) + ": missing input '" + name +
                                "'");
  }
  return it->second;
}

inline const nlohmann::json& require_input_alias(
    const std::map<std::string, nlohmann::json>& inputs,
    const std::string& primary, const std::string& alias, const char* op) {
  const auto it = inputs.find(primary);
  if (it != inputs.end()) {
    return it->second;
  }
  const auto jt = inputs.find(alias);
  if (jt != inputs.end()) {
    return jt->second;
  }
  throw std::invalid_argument(std::string(op) + ": missing input '" + primary +
                              "' (or '" + alias + "')");
}

inline double require_number(const nlohmann::json& j, const char* op,
                             const char* port) {
  if (!j.is_number()) {
    throw std::invalid_argument(std::string(op) + ": '" + port +
                                "' must be a JSON number");
  }
  return j.get<double>();
}

inline int require_positive_int(const nlohmann::json& j, const char* op,
                                const char* port) {
  const double v = require_number(j, op, port);
  const int k = static_cast<int>(v);
  if (k <= 0 || static_cast<double>(k) != v) {
    throw std::invalid_argument(std::string(op) + ": '" + port +
                                "' must be a positive integer");
  }
  return k;
}

inline bool as_bool_pred(const nlohmann::json& j, const char* op) {
  if (j.is_boolean()) {
    return j.get<bool>();
  }
  if (j.is_number()) {
    return j.get<double>() != 0.0;
  }
  if (j.is_string()) {
    const std::string s = j.get<std::string>();
    return s == "true" || s == "True" || s == "1" || s == "well";
  }
  throw std::invalid_argument(std::string(op) +
                              ": pred must be bool/number/token string");
}

inline std::string mode_or_f(const nlohmann::json& params) {
  if (!params.is_object()) {
    return {};
  }
  if (params.contains("mode") && params["mode"].is_string()) {
    return params["mode"].get<std::string>();
  }
  if (params.contains("f") && params["f"].is_string()) {
    return params["f"].get<std::string>();
  }
  return {};
}

inline double param_number(const nlohmann::json& params, const char* key,
                           double fallback) {
  if (params.is_object() && params.contains(key) && params[key].is_number()) {
    return params[key].get<double>();
  }
  return fallback;
}

inline int param_int(const nlohmann::json& params, const char* key,
                     int fallback) {
  if (params.is_object() && params.contains(key) &&
      params[key].is_number_integer()) {
    return params[key].get<int>();
  }
  if (params.is_object() && params.contains(key) && params[key].is_number()) {
    return static_cast<int>(params[key].get<double>());
  }
  return fallback;
}

}  // namespace math_nd
}  // namespace fixture
