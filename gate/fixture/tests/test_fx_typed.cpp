#include "check.hpp"
#include "fixture/fx_types.hpp"
#include "fx_typed.hpp"

#include <Eigen/Core>
#include <cmath>

int main() {
  fixture::register_fx_types();

  const auto v = fixture::fx_typed::make_vector(Eigen::Vector3d(1.0, 2.0, 3.0));
  RTR_CHECK(v.type_id == fixture::type_vector());
  const Eigen::VectorXd got_v =
      fixture::fx_typed::require_vector(v, "test", "v");
  RTR_CHECK(got_v.size() == 3);
  RTR_CHECK(std::abs(got_v(0) - 1.0) < 1e-12);

  Eigen::Matrix2d m = Eigen::Matrix2d::Identity();
  m(1, 1) = 2.0;
  const auto mat = fixture::fx_typed::make_matrix(m);
  RTR_CHECK(mat.type_id == fixture::type_matrix());
  const Eigen::MatrixXd got_m =
      fixture::fx_typed::require_matrix(mat, "test", "m");
  RTR_CHECK(got_m.rows() == 2 && got_m.cols() == 2);
  RTR_CHECK(std::abs(got_m(1, 1) - 2.0) < 1e-12);

  const auto f = fixture::fx_typed::make_number_float(1.5);
  RTR_CHECK(f.type_id == cgraph::type_ids::floating());
  RTR_CHECK(std::abs(fixture::fx_typed::require_float(f, "t", "f") - 1.5) <
            1e-12);

  return 0;
}
