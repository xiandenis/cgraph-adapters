#include "rtr/json.hpp"

#include "cgraph/type_registry.hpp"

#include <map>
#include <vector>

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
  return a.location;
}

std::filesystem::path artifact_file_path(const cgraph::DataObject& obj) {
  if (obj.payload.kind() != cgraph::Payload::Kind::DataRef) {
    throw_json("artifact_file_path: expected DataRef");
  }
  const cgraph::DataRef& ref = obj.payload.as_data_ref();
  if (ref.is_dir) {
    throw_json("artifact_file_path: expected file, got dir");
  }
  return ref.uri;
}

cgraph::SemanticSpec rigid_transform_semantic() {
  cgraph::SemanticSpec s = cgraph::SemanticSpec::of("rigid_transform");
  s.frame_from = "source";
  s.frame_to = "target";
  return s;
}

void register_rtr_types() {
  cgraph::TypeRegistry::builtins().add(
      cgraph::TypeId::parse("rtr.type.point_cloud"));
}

cgraph::PortSpec cloud_port(std::string name) {
  return cgraph::make_port(std::move(name), cgraph::PortKind::Artifact,
                           cgraph::TypeId::parse("rtr.type.point_cloud"),
                           cgraph::SemanticSpec::of("point_cloud"));
}

cgraph::PortSpec matrix_port(std::string name, bool optional) {
  cgraph::PortSpec p =
      cgraph::make_port(std::move(name), cgraph::PortKind::Value,
                        cgraph::type_ids::matrix_r4c4_f64(),
                        rigid_transform_semantic());
  p.optional = optional;
  return p;
}

cgraph::PortSpec align_port(std::string name) {
  return cgraph::make_port(std::move(name), cgraph::PortKind::Value,
                           cgraph::type_ids::record_align_result(),
                           cgraph::SemanticSpec::of("align_result"));
}

cgraph::PortSpec scalar_float_port(std::string name) {
  return cgraph::make_port(name, cgraph::PortKind::Value, cgraph::type_ids::floating(),
                           cgraph::SemanticSpec::of(name));
}

cgraph::PortSpec scalar_int_port(std::string name) {
  return cgraph::make_port(name, cgraph::PortKind::Value, cgraph::type_ids::integer(),
                           cgraph::SemanticSpec::of(name));
}

cgraph::PortSpec scalar_string_port(std::string name) {
  return cgraph::make_port(name, cgraph::PortKind::Value, cgraph::type_ids::string(),
                           cgraph::SemanticSpec::of(name));
}

cgraph::PortSpec information_port(std::string name) {
  return cgraph::make_port(std::move(name), cgraph::PortKind::Value,
                           cgraph::type_ids::tensor(),
                           cgraph::SemanticSpec::of("information"));
}

cgraph::Payload matrix_to_payload(const Eigen::Matrix4d& m) {
  cgraph::Matrix4x4 a{};
  int i = 0;
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      a[static_cast<std::size_t>(i++)] = m(r, c);
    }
  }
  return cgraph::Payload::matrix4x4(a);
}

Eigen::Matrix4d matrix_from_payload(const cgraph::Payload& payload) {
  const cgraph::Matrix4x4& a = payload.as_matrix4x4();
  Eigen::Matrix4d m;
  int i = 0;
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      m(r, c) = a[static_cast<std::size_t>(i++)];
    }
  }
  return m;
}

cgraph::DataObject align_result_to_data(const Ddx::AlignResult& a) {
  std::map<std::string, cgraph::Payload> fields;
  fields.emplace("matrix", matrix_to_payload(a.matrix_));
  fields.emplace("src_name", cgraph::Payload::string(a.src_name_));
  fields.emplace("tgt_name", cgraph::Payload::string(a.tgt_name_));
  fields.emplace("rms", cgraph::Payload::floating(a.rms_));
  fields.emplace("final_rms", cgraph::Payload::floating(a.final_rms_));
  fields.emplace("similarity", cgraph::Payload::floating(a.similarity_));
  fields.emplace("weight", cgraph::Payload::floating(a.weight_));
  fields.emplace("feature_num", cgraph::Payload::integer(a.feature_num_));
  fields.emplace("state", cgraph::Payload::integer(a.state_));
  fields.emplace("auto_reg", cgraph::Payload::integer(a.auto_reg_));
  const std::string info = matrix6d_to_json(a.information_).dump();
  fields.emplace("information",
                 cgraph::Payload::untyped(std::vector<std::uint8_t>(info.begin(),
                                                                    info.end())));
  return cgraph::make_data_object(cgraph::type_ids::record_align_result(),
                                  cgraph::SemanticSpec::of("align_result"),
                                  cgraph::Payload::record(std::move(fields)));
}

Ddx::AlignResult align_result_from_data(const cgraph::DataObject& obj) {
  if (obj.payload.kind() != cgraph::Payload::Kind::Record) {
    throw_json("align_result: expected record payload");
  }
  const auto& fields = obj.payload.as_record();
  auto require = [&](const char* key) -> const cgraph::Payload& {
    const auto it = fields.find(key);
    if (it == fields.end()) {
      throw_json(std::string("align_result: missing field ") + key);
    }
    return it->second;
  };
  Ddx::AlignResult a;
  a.matrix_ = matrix_from_payload(require("matrix"));
  a.src_name_ = require("src_name").as_string();
  a.tgt_name_ = require("tgt_name").as_string();
  a.rms_ = require("rms").as_float();
  a.final_rms_ = fields.count("final_rms") ? fields.at("final_rms").as_float() : 0;
  a.similarity_ =
      fields.count("similarity") ? fields.at("similarity").as_float() : 0;
  a.weight_ = fields.count("weight") ? fields.at("weight").as_float() : 0;
  a.feature_num_ =
      fields.count("feature_num") ? fields.at("feature_num").as_int() : 0;
  a.state_ = fields.count("state") ? fields.at("state").as_int() : 0;
  a.auto_reg_ = fields.count("auto_reg") ? fields.at("auto_reg").as_int() : 0;
  if (fields.count("information")) {
    const auto& bytes = fields.at("information").as_untyped();
    const nlohmann::json j =
        nlohmann::json::parse(std::string(bytes.begin(), bytes.end()));
    a.information_ = matrix6d_from_json(j);
  } else {
    a.information_ = Eigen::Matrix<double, 6, 6>::Identity();
  }
  return a;
}

}  // namespace rtr
