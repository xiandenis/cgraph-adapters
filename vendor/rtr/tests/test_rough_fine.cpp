#include "check.hpp"
#include "pcd_io.hpp"
#include "rtr/ops.hpp"

#include "cgraph/artifact.hpp"
#include "cgraph/ops.hpp"

#include <filesystem>
#include <string>

static nlohmann::json file_art(const std::filesystem::path& p) {
  return cgraph::artifact_to_json(cgraph::make_file_artifact(p));
}

int main() {
  cgraph::OpRegistry reg;
  rtr::register_rtr_ops(reg);
  const auto* rough = reg.get("rtr.rough_global_reg");
  RTR_CHECK(rough != nullptr);
  RTR_CHECK(!rough->usage().principle.empty());
  RTR_CHECK(std::string(rough->cost().cost_class) == "cpu.heavy");

  const auto missing = nlohmann::json{
      {"path", "Z:/definitely/missing/cloud.pcd"},
      {"digest", "00"},
      {"dtype", "file"},
      {"size", 0},
      {"is_dir", false},
  };
  cgraph::ExecContext ctx;
  bool threw = false;
  try {
    rough->execute({{"src", missing}, {"tgt", missing}}, nlohmann::json::object(),
                   ctx);
  } catch (const cgraph::OperatorError& ex) {
    threw = (ex.code == cgraph::ErrorCode::OpFailed);
  } catch (const std::exception&) {
    threw = true;
  }
  RTR_CHECK(threw);

  const auto tmp = std::filesystem::temp_directory_path() / "cgraph_rtr_test";
  const auto src_p = tmp / "src.pcd";
  const auto tgt_p = tmp / "tgt.pcd";
  auto src_pts = make_asymmetric_station(0.06f);
  auto tgt_pts = src_pts;
  for (auto& p : tgt_pts) {
    p.x() += 0.05f;
  }
  write_ascii_xyz_pcd(src_p, src_pts);
  write_ascii_xyz_pcd(tgt_p, tgt_pts);
  const auto out = rough->execute({{"src", file_art(src_p)}, {"tgt", file_art(tgt_p)}},
                                  nlohmann::json{{"resolution", 0.08}}, ctx);
  RTR_CHECK(out.at("align").contains("matrix"));
  RTR_CHECK(out.at("align")["matrix"].size() == 4);
  RTR_CHECK(out.at("align")["matrix"][0].size() == 4);

  const auto* fine = reg.get("rtr.fine_registration");
  RTR_CHECK(fine != nullptr);
  RTR_CHECK(!fine->usage().principle.empty());
  threw = false;
  try {
    fine->execute({{"src", missing}, {"tgt", missing}}, nlohmann::json::object(),
                  ctx);
  } catch (const std::exception&) {
    threw = true;
  }
  RTR_CHECK(threw);

  const auto fine_out = fine->execute(
      {{"src", file_art(src_p)}, {"tgt", file_art(tgt_p)}}, nlohmann::json::object(),
      ctx);
  RTR_CHECK(fine_out.at("align")["matrix"].size() == 4);
  return 0;
}
