#include "check.hpp"
#include "pcd_io.hpp"
#include "rtr/ops.hpp"

#include "cgraph/artifact.hpp"
#include "cgraph/data_helpers.hpp"
#include "cgraph/ops.hpp"

#include <filesystem>
#include <string>

static cgraph::DataObject file_obj(const std::filesystem::path& p) {
  const cgraph::Artifact art = cgraph::make_file_artifact(p);
  cgraph::DataRef ref;
  ref.uri = art.location.string();
  ref.content_hash = art.content_hash;
  ref.is_dir = art.is_dir;
  return cgraph::make_artifact_object(cgraph::TypeId::parse("rtr.type.point_cloud"),
                                      cgraph::SemanticSpec::of("rtr.semantic.point_cloud"),
                                      std::move(ref));
}

static cgraph::DataObject missing_obj() {
  cgraph::DataRef ref;
  ref.uri = "Z:/definitely/missing/cloud.pcd";
  ref.content_hash = std::string(64, '0');
  return cgraph::make_artifact_object(cgraph::TypeId::parse("rtr.type.point_cloud"),
                                      cgraph::SemanticSpec::of("rtr.semantic.point_cloud"),
                                      std::move(ref));
}

int main() {
  cgraph::OpRegistry reg;
  rtr::register_rtr_ops(reg);
  const auto* rough = reg.get("rtr.rough_global_reg");
  RTR_CHECK(rough != nullptr);
  RTR_CHECK(!rough->usage().principle.empty());
  RTR_CHECK(std::string(rough->cost().cost_class) == "cpu.heavy");

  cgraph::ExecContext ctx;
  bool threw = false;
  try {
    rough->execute({{"src", missing_obj()}, {"tgt", missing_obj()}},
                   nlohmann::json::object(), ctx);
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
  const auto out = rough->execute({{"src", file_obj(src_p)}, {"tgt", file_obj(tgt_p)}},
                                  nlohmann::json{{"resolution", 0.08}}, ctx);
  const Ddx::AlignResult align = rtr::align_result_from_data(out.at("align"));
  RTR_CHECK(align.matrix_(0, 0) != 0.0 || align.matrix_(1, 1) != 0.0);

  const auto* fine = reg.get("rtr.fine_registration");
  RTR_CHECK(fine != nullptr);
  RTR_CHECK(!fine->usage().principle.empty());
  threw = false;
  try {
    fine->execute({{"src", missing_obj()}, {"tgt", missing_obj()}},
                  nlohmann::json::object(), ctx);
  } catch (const std::exception&) {
    threw = true;
  }
  RTR_CHECK(threw);

  const auto fine_out = fine->execute(
      {{"src", file_obj(src_p)}, {"tgt", file_obj(tgt_p)}}, nlohmann::json::object(),
      ctx);
  const Ddx::AlignResult fine_align = rtr::align_result_from_data(fine_out.at("align"));
  RTR_CHECK(fine_align.matrix_(3, 3) == 1.0);
  return 0;
}
