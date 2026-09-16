#include "cgraph/effect.hpp"
#include "cgraph/ops.hpp"
#include "cgraph/process_runner.hpp"

#include <memory>
#include <yaml-cpp/yaml.h>

namespace fixture {
namespace {

class SubprocEchoOp final : public cgraph::ProcessOperator {
 public:
  explicit SubprocEchoOp(std::string op_id, std::string script_name) {
    op_id_ = std::move(op_id);
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Artifact, "file");
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Artifact, "file");
    capability_.summary = "V2 gate: real subprocess echo via tests/scripts";
    cost_.cost_class = "process.unbounded";
    usage_.connect = "Wire artifact file into in; read file out.";
    usage_.tune = "None (kernel acceptance fixture).";
    usage_.inspect = "Spawns python script with cwd=slot; stdout→log.txt.";

    cgraph::apply_effect_yaml(effect_, YAML::Load(R"(
effect:
  class: external
  cache: fingerprinted
)"));
    effect_.seed_param.clear();

    const std::string py = cgraph::resolve_python_executable();
    const std::string script = cgraph::resolve_test_script(script_name).string();
    set_argv_template({py, script, "{nodeCacheFolder}"});
    set_shell(false);
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_subproc_echo() {
  return std::make_shared<SubprocEchoOp>("fx.subproc_echo", "v2_echo_proc.py");
}

std::shared_ptr<cgraph::MemoryOperator> make_subproc_fail() {
  return std::make_shared<SubprocEchoOp>("fx.subproc_fail", "v2_fail_proc.py");
}

}  // namespace fixture
