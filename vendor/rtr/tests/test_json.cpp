#include "check.hpp"
#include "rtr/json.hpp"

#include <stdexcept>

int main() {
  Eigen::Matrix4d I = Eigen::Matrix4d::Identity();
  I(0, 3) = 1.5;
  const nlohmann::json j = rtr::matrix4d_to_json(I);
  RTR_CHECK(j.is_array());
  RTR_CHECK(j.size() == 4);
  RTR_CHECK(j[0].size() == 4);
  RTR_CHECK(j[0][3].get<double>() == 1.5);
  const Eigen::Matrix4d back = rtr::matrix4d_from_json(j);
  RTR_CHECK((back - I).norm() < 1e-12);

  bool threw = false;
  try {
    rtr::matrix4d_from_json(nlohmann::json{
        {"R", {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}}, {"t", {0, 0, 0}}});
  } catch (const std::exception&) {
    threw = true;
  }
  RTR_CHECK(threw);

  threw = false;
  try {
    rtr::matrix4d_from_json(nlohmann::json{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
  } catch (const std::exception&) {
    threw = true;
  }
  RTR_CHECK(threw);

  Ddx::AlignResult a;
  a.src_name_ = "s";
  a.tgt_name_ = "t";
  a.matrix_ = I;
  a.rms_ = 0.02;
  a.final_rms_ = 0.01;
  a.similarity_ = 0.8;
  a.weight_ = 0.1;
  a.feature_num_ = 12;
  a.state_ = 3;
  a.auto_reg_ = 1;
  a.information_ = Eigen::Matrix<double, 6, 6>::Identity();
  const nlohmann::json aj = rtr::align_result_to_json(a);
  RTR_CHECK(!aj.contains("matrix_"));
  RTR_CHECK(aj["src_name"] == "s");
  RTR_CHECK(aj["rms"].get<double>() == 0.02);
  const Ddx::AlignResult b = rtr::align_result_from_json(aj);
  RTR_CHECK(b.src_name_ == "s");
  RTR_CHECK((b.matrix_ - I).norm() < 1e-12);
  return 0;
}
