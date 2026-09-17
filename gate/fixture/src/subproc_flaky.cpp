#include "cgraph/effect.hpp"
#include "cgraph/ops.hpp"
#include "cgraph/process_runner.hpp"

#include <memory>
#include <yaml-cpp/yaml.h>

namespace fixture {
namespace {

class SubprocFlakyOp final : public cgraph::ProcessOperator {
 public:
  SubprocFlakyOp() {
    op_id_ = "fx.subproc_flaky";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Artifact, cgraph::type_ids::untyped(),
                          cgraph::SemanticSpec::of("cgraph.semantic.file"));
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Artifact, cgraph::type_ids::untyped(),
                          cgraph::SemanticSpec::of("cgraph.semantic.file"));

    cgraph::ParamSpec fail_before;
    fail_before.name = "fail_before";
    fail_before.dtype = "int";
    fail_before.default_value = 2;
    fail_before.invalidate = true;
    fail_before.doc = "Fail the first N spawn attempts, then succeed (gate fixture).";
    signature_.params["fail_before"] = std::move(fail_before);

    capability_.summary = "test-only";
    cost_.cost_class = "process.unbounded";
    usage_.connect = "V4-A Process retry gate: wire artifact in → out.";
    usage_.tune = "params.fail_before (int); params.retry (schedule, not identity).";
    usage_.inspect = "Spawns v4_flaky_proc.py; cwd=slot; spawn.count persists.";

    cgraph::apply_effect_yaml(effect_, YAML::Load(R"(
effect:
  class: external
  cache: fingerprinted
)"));
    effect_.seed_param.clear();

    const std::string py = cgraph::resolve_python_executable();
    const std::string script =
        cgraph::resolve_test_script("v4_flaky_proc.py").string();
    set_argv_template({py, script, "{nodeCacheFolder}", "{params.fail_before}"});
    set_shell(false);
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_subproc_flaky() {
  return std::make_shared<SubprocFlakyOp>();
}

}  // namespace fixture
