#include "cgraph/effect.hpp"
#include "cgraph/ops.hpp"
#include "cgraph/process_runner.hpp"

#include <memory>
#include <yaml-cpp/yaml.h>

namespace fixture {
namespace {

class SubprocChunkOp final : public cgraph::ProcessOperator {
 public:
  SubprocChunkOp() {
    op_id_ = "fx.subproc_chunk";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Artifact, cgraph::type_ids::untyped(),
                          cgraph::SemanticSpec::of("cgraph.semantic.file"));
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Artifact, cgraph::type_ids::untyped(),
                          cgraph::SemanticSpec::of("cgraph.semantic.file"));
    capability_.summary = "V2 gate: ProcessOperator.chunk (D12)";
    cost_.cost_class = "process.unbounded";
    usage_.connect = "Wire artifact into in; read aggregated out.";
    usage_.tune = "params.range_total; optional chunk_block_extra / nonce.";
    usage_.inspect = "Spawns per-block python script; cwd=durable chunk slot.";

    cgraph::apply_effect_yaml(effect_, YAML::Load(R"(
effect:
  class: external
  cache: fingerprinted
)"));
    effect_.seed_param.clear();

    cgraph::ChunkSpec cs;
    cs.enabled = true;
    cs.block_size = 1;
    cs.range_flag = "--rangeStart {rangeStart} --rangeSize {rangeSize}";
    cs.allow_partial = false;
    set_chunk(std::move(cs));

    const std::string py = cgraph::resolve_python_executable();
    const std::string script =
        cgraph::resolve_test_script("v2_chunk_proc.py").string();
    set_argv_template({py, script, "{nodeCacheFolder}"});
    set_shell(false);
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_subproc_chunk() {
  return std::make_shared<SubprocChunkOp>();
}

}  // namespace fixture
