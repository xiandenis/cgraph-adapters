#include "check.hpp"
#include "rtr/json.hpp"
#include "rtr/ops.hpp"

#include "cgraph/data_helpers.hpp"
#include "cgraph/ops.hpp"

int main() {
  cgraph::OpRegistry reg;
  rtr::register_rtr_ops(reg);
  const cgraph::MemoryOperator* op = reg.get("rtr.align_result.unpack");
  RTR_CHECK(op != nullptr);
  RTR_CHECK(!op->usage().principle.empty());

  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T(0, 3) = 2.0;
  Ddx::AlignResult a;
  a.src_name_ = "a";
  a.tgt_name_ = "b";
  a.matrix_ = T;
  a.rms_ = 0.05;
  a.information_ = Eigen::Matrix<double, 6, 6>::Identity();
  const cgraph::DataObject sample = rtr::align_result_to_data(a);

  cgraph::ExecContext ctx;
  ctx.requested_outputs = {"matrix", "rms"};
  const auto out = op->execute({{"align", sample}}, nlohmann::json::object(), ctx);
  RTR_CHECK(out.count("matrix") == 1);
  RTR_CHECK(out.count("rms") == 1);
  RTR_CHECK(out.count("src_name") == 0);
  RTR_CHECK(out.at("rms").payload.as_float() == 0.05);
  RTR_CHECK(out.at("matrix").payload.as_matrix4x4()[3] == 2.0);

  ctx.requested_outputs = {"matrix"};
  bool threw = false;
  try {
    op->execute({{"align", cgraph::make_float(1.0, cgraph::SemanticSpec::of("rtr.semantic.rms"))}},
                nlohmann::json::object(), ctx);
  } catch (const cgraph::OperatorError&) {
    threw = true;
  } catch (const std::exception&) {
    threw = true;
  }
  RTR_CHECK(threw);
  return 0;
}
