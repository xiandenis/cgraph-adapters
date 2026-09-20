#pragma once

#include <cgraph/artifact.hpp>
#include <cgraph/data_object.hpp>
#include <cgraph/port.hpp>
#include <registration_type/registration_type.hpp>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <stdexcept>
#include <string>

namespace rtr {

inline void throw_json(const std::string& msg) {
  throw std::invalid_argument(msg);
}

void register_rtr_types();

cgraph::SemanticSpec rigid_transform_semantic();
cgraph::PortSpec cloud_port(std::string name);
cgraph::PortSpec matrix_port(std::string name, bool optional = false);
cgraph::PortSpec align_port(std::string name);
cgraph::PortSpec report_port(std::string name);
cgraph::PortSpec path_port(std::string name);
cgraph::PortSpec scalar_float_port(std::string name);
cgraph::PortSpec scalar_int_port(std::string name);
cgraph::PortSpec scalar_string_port(std::string name);
cgraph::PortSpec information_port(std::string name);

cgraph::DataObject cloud_artifact_from_path(const std::filesystem::path& path);
cgraph::DataObject registration_report_to_data(float overlap_ratio, double rms,
                                               std::size_t overlap_count,
                                               std::size_t rms_sample_count,
                                               bool rms_query_from_target);

nlohmann::json matrix4d_to_json(const Eigen::Matrix4d& m);
Eigen::Matrix4d matrix4d_from_json(const nlohmann::json& j);
nlohmann::json matrix6d_to_json(const Eigen::Matrix<double, 6, 6>& m);
Eigen::Matrix<double, 6, 6> matrix6d_from_json(const nlohmann::json& j);
nlohmann::json align_result_to_json(const Ddx::AlignResult& a);
Ddx::AlignResult align_result_from_json(const nlohmann::json& j);

cgraph::Payload matrix_to_payload(const Eigen::Matrix4d& m);
Eigen::Matrix4d matrix_from_payload(const cgraph::Payload& payload);
cgraph::DataObject align_result_to_data(const Ddx::AlignResult& a);
Ddx::AlignResult align_result_from_data(const cgraph::DataObject& obj);
std::filesystem::path artifact_file_path(const nlohmann::json& handle);
std::filesystem::path artifact_file_path(const cgraph::DataObject& obj);

}  // namespace rtr
