#include "check.hpp"
#include "fixture/ops.hpp"
#include "fx_typed.hpp"

#include "cgraph/ops.hpp"

#include <Eigen/Core>
#include <Eigen/LU>
#include <cmath>
#include <vector>

int main() {
  cgraph::OpRegistry reg;
  fixture::register_fixture_ops(reg);

  // --- Case 4: h = p^T A p + alpha*det(A) - sin(tr(A)) ---
  {
    Eigen::Matrix2d A;
    A << 2.0, 1.0, 0.0, 3.0;
    Eigen::Vector2d p(1.0, 1.0);
    const double alpha = 0.5;
    const double expected =
        p.dot(A * p) + alpha * A.determinant() - std::sin(A.trace());

    const auto* qf = reg.get("fx.quadratic_form");
    const auto* det = reg.get("fx.det");
    const auto* tr = reg.get("fx.trace");
    const auto* sin = reg.get("fx.sin");
    const auto* add = reg.get("fx.add");
    const auto* sub = reg.get("fx.sub");
    RTR_CHECK(qf && det && tr && sin && add && sub);

    const auto Aobj = fixture::fx_typed::make_matrix(A);
    const auto pobj = fixture::fx_typed::make_vector(p);
    const double M1 =
        qf->execute({{"A", Aobj}, {"p", pobj}}, nlohmann::json::object(),
                    cgraph::ExecContext{})
            .at("out")
            .payload.as_float();
    const double M2 = det->execute({{"in", Aobj}}, nlohmann::json::object(),
                                   cgraph::ExecContext{})
                          .at("out")
                          .payload.as_float();
    const double M2a =
        add->execute({{"a", fixture::fx_typed::make_number_float(alpha)},
                      {"b", fixture::fx_typed::make_number_float(0.0)}},
                     nlohmann::json::object(), cgraph::ExecContext{})
            .at("sum")
            .payload.as_float();
    (void)M2a;
    const double M2scaled = alpha * M2;
    const double M3 = tr->execute({{"in", Aobj}}, nlohmann::json::object(),
                                  cgraph::ExecContext{})
                          .at("out")
                          .payload.as_float();
    const double M4 =
        sin->execute({{"in", fixture::fx_typed::make_number_float(M3)}},
                     nlohmann::json::object(), cgraph::ExecContext{})
            .at("out")
            .payload.as_float();
    const double sum1 =
        add->execute({{"a", fixture::fx_typed::make_number_float(M1)},
                      {"b", fixture::fx_typed::make_number_float(M2scaled)}},
                     nlohmann::json::object(), cgraph::ExecContext{})
            .at("sum")
            .payload.as_float();
    const double h =
        sub->execute({{"a", fixture::fx_typed::make_number_float(sum1)},
                      {"b", fixture::fx_typed::make_number_float(M4)}},
                     nlohmann::json::object(), cgraph::ExecContext{})
            .at("diff")
            .payload.as_float();
    RTR_CHECK(std::abs(h - expected) < 1e-9);
  }

  // --- Case 5: phi(a,b) ---
  {
    const double a = 2.0;
    const double b = 1.0;
    const double expected =
        std::tanh(std::pow(std::pow(a, 4.0) + std::pow(b, 2.0), 1.0 / 3.0)) *
        std::log2(1.0 + a / (b * b + 1.0));

    const auto* pow = reg.get("fx.pow");
    const auto* add = reg.get("fx.add");
    const auto* div = reg.get("fx.div");
    const auto* tanh = reg.get("fx.tanh");
    const auto* log2 = reg.get("fx.log2");
    RTR_CHECK(pow && add && div && tanh && log2);

    const double A4 =
        pow->execute({{"base", fixture::fx_typed::make_number_float(a)},
                      {"exp", fixture::fx_typed::make_number_float(4.0)}},
                     nlohmann::json::object(), cgraph::ExecContext{})
            .at("out")
            .payload.as_float();
    const double B2 =
        pow->execute({{"base", fixture::fx_typed::make_number_float(b)},
                      {"exp", fixture::fx_typed::make_number_float(2.0)}},
                     nlohmann::json::object(), cgraph::ExecContext{})
            .at("out")
            .payload.as_float();
    const double S1 =
        add->execute({{"a", fixture::fx_typed::make_number_float(A4)},
                      {"b", fixture::fx_typed::make_number_float(B2)}},
                     nlohmann::json::object(), cgraph::ExecContext{})
            .at("sum")
            .payload.as_float();
    const double root =
        pow->execute({{"base", fixture::fx_typed::make_number_float(S1)},
                      {"exp", fixture::fx_typed::make_number_float(1.0 / 3.0)}},
                     nlohmann::json::object(), cgraph::ExecContext{})
            .at("out")
            .payload.as_float();
    const double T1 =
        tanh->execute({{"in", fixture::fx_typed::make_number_float(root)}},
                      nlohmann::json::object(), cgraph::ExecContext{})
            .at("out")
            .payload.as_float();
    const double B2p1 =
        add->execute({{"a", fixture::fx_typed::make_number_float(B2)},
                      {"b", fixture::fx_typed::make_number_float(1.0)}},
                     nlohmann::json::object(), cgraph::ExecContext{})
            .at("sum")
            .payload.as_float();
    const double Ab =
        div->execute({{"a", fixture::fx_typed::make_number_float(a)},
                      {"b", fixture::fx_typed::make_number_float(B2p1)}},
                     nlohmann::json::object(), cgraph::ExecContext{})
            .at("quot")
            .payload.as_float();
    const double LogIn =
        add->execute({{"a", fixture::fx_typed::make_number_float(1.0)},
                      {"b", fixture::fx_typed::make_number_float(Ab)}},
                     nlohmann::json::object(), cgraph::ExecContext{})
            .at("sum")
            .payload.as_float();
    const double T2 =
        log2->execute({{"in", fixture::fx_typed::make_number_float(LogIn)}},
                      nlohmann::json::object(), cgraph::ExecContext{})
            .at("out")
            .payload.as_float();
    const double phi = T1 * T2;
    RTR_CHECK(std::abs(phi - expected) < 1e-9);
  }

  // --- Case 2: ICP residual sum N=2 ---
  {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t(0.1, 0.0, 0.0);
    Eigen::Vector3d a0(1.0, 0.0, 0.0);
    Eigen::Vector3d a1(0.0, 1.0, 0.0);
    Eigen::Vector3d b0(0.9, 0.0, 0.0);
    Eigen::Vector3d b1(0.0, 0.9, 0.0);
    const double expected =
        (a0 - (R * b0 + t)).squaredNorm() + (a1 - (R * b1 + t)).squaredNorm();

    const auto* matvec = reg.get("fx.matvec");
    const auto* vadd = reg.get("fx.vadd");
    const auto* vsub = reg.get("fx.vsub");
    const auto* l2sq = reg.get("fx.l2sq");
    const auto* sumr = reg.get("fx.sum_reduce");
    RTR_CHECK(matvec && vadd && vsub && l2sq && sumr);

    const auto Robj = fixture::fx_typed::make_matrix(R);
    const auto tobj = fixture::fx_typed::make_vector(t);
    auto one = [&](const Eigen::Vector3d& ai, const Eigen::Vector3d& bi) {
      const auto Bb =
          matvec
              ->execute({{"A", Robj},
                         {"v", fixture::fx_typed::make_vector(bi)}},
                        nlohmann::json::object(), cgraph::ExecContext{})
              .at("out");
      const auto Ti =
          vadd->execute({{"a", Bb}, {"b", tobj}}, nlohmann::json::object(),
                        cgraph::ExecContext{})
              .at("out");
      const auto ri =
          vsub->execute({{"a", fixture::fx_typed::make_vector(ai)}, {"b", Ti}},
                        nlohmann::json::object(), cgraph::ExecContext{})
              .at("out");
      return l2sq
          ->execute({{"in", ri}}, nlohmann::json::object(),
                    cgraph::ExecContext{})
          .at("out")
          .payload.as_float();
    };
    const double s0 = one(a0, b0);
    const double s1 = one(a1, b1);
    const double S =
        sumr
            ->execute({{"in", fixture::fx_typed::make_float_list({s0, s1})}},
                      nlohmann::json::object(), cgraph::ExecContext{})
            .at("out")
            .payload.as_float();
    RTR_CHECK(std::abs(S - expected) < 1e-9);
  }

  // --- Case 3: branch merge ---
  {
    const auto* cond = reg.get("fx.cond_gt0");
    const auto* merge = reg.get("fx.branch_merge");
    RTR_CHECK(cond && merge);

    const auto pred_pos =
        cond->execute({{"in", fixture::fx_typed::make_number_float(1.0)}},
                      nlohmann::json::object(), cgraph::ExecContext{})
            .at("pred");
    RTR_CHECK(pred_pos.payload.as_bool() == true);
    const double g1 =
        merge
            ->execute({{"true", fixture::fx_typed::make_number_float(10.0)},
                       {"false", fixture::fx_typed::make_number_float(-10.0)},
                       {"pred", pred_pos}},
                      nlohmann::json::object(), cgraph::ExecContext{})
            .at("out")
            .payload.as_float();
    RTR_CHECK(std::abs(g1 - 10.0) < 1e-12);

    const auto pred_neg =
        cond->execute({{"in", fixture::fx_typed::make_number_float(-1.0)}},
                      nlohmann::json::object(), cgraph::ExecContext{})
            .at("pred");
    RTR_CHECK(pred_neg.payload.as_bool() == false);
    const double g2 =
        merge
            ->execute({{"true", fixture::fx_typed::make_number_float(10.0)},
                       {"false", fixture::fx_typed::make_number_float(-10.0)},
                       {"pred", pred_neg}},
                      nlohmann::json::object(), cgraph::ExecContext{})
            .at("out")
            .payload.as_float();
    RTR_CHECK(std::abs(g2 + 10.0) < 1e-12);
  }

  return 0;
}
