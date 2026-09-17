#include "cgraph/artifact.hpp"
#include "cgraph/ops.hpp"
#include "fx_data.hpp"

#include <fstream>
#include <memory>
#include <stdexcept>

namespace fixture {
namespace {

class DumpJsonOp final : public cgraph::MemoryOperator {
 public:
  DumpJsonOp() {
    op_id_ = "fx.dump_json";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Artifact, cgraph::type_ids::untyped(),
                          cgraph::SemanticSpec::of("cgraph.semantic.file"));
    capability_.summary = "P0 stand-in for Dump: json value to a file artifact";
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire json into in; read file out. Not stdlib Dump.";
    usage_.tune = "None.";
    usage_.inspect = "Writes canon-equivalent JSON text to workdir/outputs/out.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in,
      const nlohmann::json&, const cgraph::ExecContext& ctx) const override {
    const auto inputs = fx::unwrap(data_in);
    const auto it = inputs.find("in");
    if (it == inputs.end()) {
      throw std::invalid_argument("fx.dump_json: missing input 'in'");
    }
    if (ctx.workdir.empty()) {
      throw std::invalid_argument("fx.dump_json: workdir required");
    }
    const std::filesystem::path out = ctx.output_path("out");
    std::filesystem::create_directories(out.parent_path());
    std::ofstream file(out, std::ios::binary | std::ios::trunc);
    if (!file) {
      throw std::runtime_error("fx.dump_json: cannot write " + out.string());
    }
    const std::string text = it->second.dump();
    file.write(text.data(), static_cast<std::streamsize>(text.size()));
    file.close();
    return fx::wrap(signature_, {{"out", cgraph::artifact_to_json(cgraph::make_file_artifact(
                        out, cgraph::DType::parse("file"), ctx.artifact_digest_mode))}});
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_dump_json() {
  return std::make_shared<DumpJsonOp>();
}

}  // namespace fixture
