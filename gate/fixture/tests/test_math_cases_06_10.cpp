#include "check.hpp"
#include "fixture/ops.hpp"
#include "fx_typed.hpp"

#include "cgraph/ops.hpp"
#include "cgraph/validate.hpp"

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static cgraph::Node op_node(const cgraph::MemoryOperator& op, std::string id) {
  cgraph::Node node;
  node.id = std::move(id);
  node.kind = cgraph::NodeKind::Operator;
  node.op_id = std::string(op.op_id());
  node.inputs = op.signature().inputs;
  node.outputs = op.signature().outputs;
  return node;
}

int main() {
  cgraph::OpRegistry reg;
  fixture::register_fixture_ops(reg);
  const cgraph::ExecContext ctx;

  // --- Case 7 slice: SVD + variance_ratio ---
  {
    Eigen::MatrixXd A(2, 2);
    A << 3.0, 0.0, 0.0, 1.0;
    Eigen::JacobiSVD<Eigen::MatrixXd> ref(
        A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const auto* svd = reg.get("fx.svd");
    const auto* vr = reg.get("fx.variance_ratio");
    RTR_CHECK(svd && vr);
    const auto outs =
        svd->execute({{"A", fixture::fx_typed::make_matrix(A)}},
                     nlohmann::json::object(), ctx);
    const Eigen::VectorXd S =
        fixture::fx_typed::require_vector(outs.at("S"), "t", "S");
    RTR_CHECK(S.size() == 2);
    RTR_CHECK(std::abs(S(0) - ref.singularValues()(0)) < 1e-9);
    RTR_CHECK(std::abs(S(1) - ref.singularValues()(1)) < 1e-9);
    const double ratio =
        vr->execute({{"s", outs.at("S")},
                     {"k", fixture::fx_typed::make_count_int(1)}},
                    nlohmann::json::object(), ctx)
            .at("out")
            .payload.as_float();
    const double s0 = ref.singularValues()(0);
    const double s1 = ref.singularValues()(1);
    const double expected = (s0 * s0) / (s0 * s0 + s1 * s1);
    RTR_CHECK(std::abs(ratio - expected) < 1e-9);
  }

  // --- Case 8 slice: cond + LU + branch_merge ---
  {
    Eigen::MatrixXd A(2, 2);
    A << 4.0, 1.0, 1.0, 3.0;
    Eigen::VectorXd b(2);
    b << 1.0, 2.0;
    const Eigen::VectorXd x_ref = A.lu().solve(b);
    const auto* cond = reg.get("fx.cond_estimate");
    const auto* lu = reg.get("fx.lu_solve");
    const auto* merge = reg.get("fx.branch_merge_vec");
    RTR_CHECK(cond && lu && merge);
    const auto c =
        cond->execute({{"A", fixture::fx_typed::make_matrix(A)}},
                      nlohmann::json::object(), ctx);
    RTR_CHECK(c.at("IsWell").payload.as_bool());
    const auto sol =
        lu->execute({{"A", fixture::fx_typed::make_matrix(A)},
                     {"b", fixture::fx_typed::make_vector(b)}},
                    nlohmann::json::object(), ctx);
    const Eigen::VectorXd x =
        fixture::fx_typed::require_vector(sol.at("x"), "t", "x");
    RTR_CHECK((x - x_ref).norm() < 1e-9);
    Eigen::VectorXd junk(2);
    junk << 9.0, 9.0;
    const auto merged =
        merge->execute({{"pred", c.at("IsWell")},
                        {"true", sol.at("x")},
                        {"false", fixture::fx_typed::make_vector(junk)}},
                       nlohmann::json::object(), ctx);
    const Eigen::VectorXd xm =
        fixture::fx_typed::require_vector(merged.at("out"), "t", "out");
    RTR_CHECK((xm - x_ref).norm() < 1e-9);
  }

  // --- Case 6: pad + split/merge/map ---
  {
    Eigen::MatrixXd M(2, 2);
    M << 1.0, 2.0, 3.0, 4.0;
    const auto* pad = reg.get("fx.pad");
    const auto* split = reg.get("fx.split_grid");
    const auto* mapc = reg.get("fx.map_chunk");
    const auto* merge = reg.get("fx.merge_grid");
    RTR_CHECK(pad && split && mapc && merge);
    const auto padded =
        pad->execute({{"in", fixture::fx_typed::make_matrix(M)}},
                     nlohmann::json{{"pad_h", 0}, {"pad_w", 0}}, ctx);
    const Eigen::MatrixXd Mp = fixture::fx_typed::require_matrix(
        padded.at("out"), "t", "out");
    RTR_CHECK((Mp - M).norm() < 1e-12);
    const auto tiles =
        split->execute({{"in", fixture::fx_typed::make_matrix(M)}},
                       nlohmann::json{{"tile_h", 1}, {"tile_w", 1}}, ctx);
    Eigen::MatrixXd ker(1, 1);
    ker << 2.0;
    const auto mapped =
        mapc->execute({{"items", tiles.at("out")},
                       {"kernel", fixture::fx_typed::make_matrix(ker)}},
                      nlohmann::json{{"body", "scale"}}, ctx);
    const auto back =
        merge->execute({{"chunks", mapped.at("out")},
                        {"grid_shape", tiles.at("grid_shape")}},
                       nlohmann::json::object(), ctx);
    const Eigen::MatrixXd out = fixture::fx_typed::require_matrix(
        back.at("out"), "t", "out");
    RTR_CHECK((out - (2.0 * M)).norm() < 1e-9);
  }

  // --- Case 9 slice: one GD step on ||x||^2/2 ---
  {
    Eigen::VectorXd x0(2);
    x0 << 2.0, -1.0;
    const auto* grad = reg.get("fx.gradient");
    const auto* ls = reg.get("fx.line_search");
    const auto* upd = reg.get("fx.update");
    const auto* loss = reg.get("fx.loss_calc");
    RTR_CHECK(grad && ls && upd && loss);
    const nlohmann::json mode = nlohmann::json{{"mode", "quadratic_l2"}};
    const auto g =
        grad->execute({{"x", fixture::fx_typed::make_vector(x0)}}, mode, ctx);
    const Eigen::VectorXd gvec =
        fixture::fx_typed::require_vector(g.at("out"), "t", "out");
    RTR_CHECK((gvec - x0).norm() < 1e-9);
    const auto alpha =
        ls->execute({{"x", fixture::fx_typed::make_vector(x0)},
                     {"g", g.at("out")},
                     {"lr", fixture::fx_typed::make_number_float(1.0)}},
                    mode, ctx)
            .at("out")
            .payload.as_float();
    RTR_CHECK(alpha > 0.0);
    const auto x1 =
        upd->execute({{"x", fixture::fx_typed::make_vector(x0)},
                      {"g", g.at("out")},
                      {"alpha", fixture::fx_typed::make_number_float(alpha)}},
                     nlohmann::json::object(), ctx);
    const double L =
        loss->execute({{"x", x1.at("out")}}, mode, ctx)
            .at("out")
            .payload.as_float();
    RTR_CHECK(L < 0.5 * x0.squaredNorm() + 1e-9);
  }

  // --- Case 10 slice: kalman + list_reduce_mean ---
  {
    const auto* kalman = reg.get("fx.kalman_iter");
    const auto* mean = reg.get("fx.list_reduce_mean");
    RTR_CHECK(kalman && mean);
    const std::vector<double> zs = {1.0, 1.1, 0.9, 1.05};
    const auto kr =
        kalman->execute({{"in", fixture::fx_typed::make_float_list(zs)},
                         {"Q", fixture::fx_typed::make_number_float(1.0)},
                         {"R", fixture::fx_typed::make_number_float(0.5)}},
                        nlohmann::json::object(), ctx);
    RTR_CHECK(std::isfinite(kr.at("final_state").payload.as_float()));
    const auto innovs = fixture::fx_typed::require_float_list(
        kr.at("innovations"), "t", "innovations");
    RTR_CHECK(innovs.size() == zs.size());
    const double m =
        mean->execute({{"in", fixture::fx_typed::make_float_list(
                                  {0.1, 0.2, 0.3})}},
                      nlohmann::json::object(), ctx)
            .at("out")
            .payload.as_float();
    RTR_CHECK(std::abs(m - 0.2) < 1e-12);
  }

  // --- Artifact dir smoke (temp csv folder) ---
  {
    const auto* input_dir = reg.get("fx.input_artifact_dir");
    const auto* map_files = reg.get("fx.map_chunk_files");
    RTR_CHECK(input_dir && map_files);
    const fs::path tmp =
        fs::temp_directory_path() / "cgraph_fx_artifact_wave2";
    fs::create_directories(tmp);
    {
      std::ofstream a(tmp / "a.csv");
      a << "z\n1.0\n1.2\n0.8\n";
      std::ofstream b(tmp / "b.csv");
      b << "z\n2.0\n2.1\n1.9\n";
    }
    const auto dir_obj =
        input_dir
            ->execute({}, nlohmann::json{{"path", tmp.string()}}, ctx)
            .at("out");
    RTR_CHECK(fixture::fx_typed::is_directory_artifact(dir_obj));
    const auto mapped =
        map_files->execute(
            {{"items", dir_obj},
             {"Q", fixture::fx_typed::make_number_float(1.0)},
             {"R", fixture::fx_typed::make_number_float(0.5)}},
            nlohmann::json::object(), ctx);
    const auto innovs = fixture::fx_typed::require_float_list(
        mapped.at("InnovList"), "t", "InnovList");
    RTR_CHECK(innovs.size() == 2);
    fs::remove_all(tmp);
  }

  // --- Wire: matrix cannot feed kalman float_list ---
  {
    const auto* cmat = reg.get("fx.const_matrix");
    const auto* kalman = reg.get("fx.kalman_iter");
    RTR_CHECK(cmat && kalman);
    cgraph::Graph bad;
    bad.add_node(op_node(*cmat, "M"));
    bad.add_node(op_node(*kalman, "K"));
    cgraph::Edge e;
    e.src_node = "M";
    e.src_port = "out";
    e.dst_node = "K";
    e.dst_port = "in";
    bad.add_edge(std::move(e));
    const cgraph::ValidationReport blocked = cgraph::validate(bad, reg);
    RTR_CHECK(!blocked.ok);
    bool saw = false;
    for (const auto& issue : blocked.errors) {
      if (issue.code == "type_mismatch") {
        saw = true;
      }
    }
    RTR_CHECK(saw);
  }

  return 0;
}
