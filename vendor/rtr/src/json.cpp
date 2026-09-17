#include "rtr/json.hpp"

namespace rtr {
namespace {

nlohmann::json matrix_to_nested(const Eigen::MatrixXd& m) {
  nlohmann::json rows = nlohmann::json::array();
  for (int r = 0; r < m.rows(); ++r) {
    nlohmann::json row = nlohmann::json::array();
    for (int c = 0; c < m.cols(); ++c) {
      row.push_back(m(r, c));
    }
    rows.push_back(std::move(row));
  }
  return rows;
}

Eigen::MatrixXd nested_to_matrix(const nlohmann::json& j, int rows, int cols,
                                 const char* what) {
  if (!j.is_array() || static_cast<int>(j.size()) != rows) {
    throw_json(std::string(what) + ": expected nested array");
  }
  Eigen::MatrixXd m(rows, cols);
  for (int r = 0; r < rows; ++r) {
    if (!j[r].is_array() || static_cast<int>(j[r].size()) != cols) {
      throw_json(std::string(what) + ": bad row");
    }
    for (int c = 0; c < cols; ++c) {
      if (!j[r][c].is_number()) {
        throw_json(std::string(what) + ": non-numeric");
      }
      m(r, c) = j[r][c].get<double>();
    }
  }
  return m;
}

}  // namespace

nlohmann::json matrix4d_to_json(const Eigen::Matrix4d& m) {
  return matrix_to_nested(m);
}

Eigen::Matrix4d matrix4d_from_json(const nlohmann::json& j) {
  if (j.is_object()) {
    throw_json("matrix4d: object form (e.g. {R,t}) is not accepted");
  }
  return nested_to_matrix(j, 4, 4, "matrix4d");
}

nlohmann::json matrix6d_to_json(const Eigen::Matrix<double, 6, 6>& m) {
  return matrix_to_nested(m);
}

Eigen::Matrix<double, 6, 6> matrix6d_from_json(const nlohmann::json& j) {
  return nested_to_matrix(j, 6, 6, "matrix6d");
}

nlohmann::json align_result_to_json(const Ddx::AlignResult& a) {
  return nlohmann::json{
      {"src_name", a.src_name_},
      {"tgt_name", a.tgt_name_},
      {"matrix", matrix4d_to_json(a.matrix_)},
      {"rms", a.rms_},
      {"final_rms", a.final_rms_},
      {"similarity", a.similarity_},
      {"weight", a.weight_},
      {"feature_num", a.feature_num_},
      {"state", a.state_},
      {"auto_reg", a.auto_reg_},
      {"information", matrix6d_to_json(a.information_)},
  };
}

Ddx::AlignResult align_result_from_json(const nlohmann::json& j) {
  if (!j.is_object()) {
    throw_json("align_result: expected object");
  }
  Ddx::AlignResult a;
  a.src_name_ = j.value("src_name", std::string());
  a.tgt_name_ = j.value("tgt_name", std::string());
  if (!j.contains("matrix")) {
    throw_json("align_result: missing matrix");
  }
  a.matrix_ = matrix4d_from_json(j.at("matrix"));
  a.rms_ = j.value("rms", 0.0);
  a.final_rms_ = j.value("final_rms", 0.0);
  a.similarity_ = j.value("similarity", 0.0);
  a.weight_ = j.value("weight", 0.0);
  a.feature_num_ = j.value("feature_num", 0);
  a.state_ = j.value("state", 0);
  a.auto_reg_ = j.value("auto_reg", 0);
  if (j.contains("information")) {
    a.information_ = matrix6d_from_json(j.at("information"));
  } else {
    a.information_ = Eigen::Matrix<double, 6, 6>::Identity();
  }
  return a;
}

std::filesystem::path artifact_file_path(const nlohmann::json& handle) {
  const cgraph::Artifact a = cgraph::artifact_from_json(handle);
  if (a.is_dir) {
    throw_json("artifact_file_path: expected file, got dir");
  }
  return a.path;
}

}  // namespace rtr
