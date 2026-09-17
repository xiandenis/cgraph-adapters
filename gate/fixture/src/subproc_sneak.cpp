#include "cgraph/effect.hpp"
#include "cgraph/ops.hpp"
#include "fx_data.hpp"
#include "cgraph/process_runner.hpp"
#include "cgraph/runtime.hpp"
#include "cgraph/sandbox.hpp"

#include <memory>
#include <string>
#include <yaml-cpp/yaml.h>

namespace fixture {
namespace {

class SubprocSneakOp final : public cgraph::ProcessOperator {
 public:
  SubprocSneakOp() {
    op_id_ = "fx.subproc_sneak";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Artifact, "file");
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Artifact, "file");
    capability_.summary = "test-only";
    cost_.cost_class = "process.unbounded";
    usage_.connect = "V4-A OS sandbox gate: optional undeclared read/write.";
    usage_.tune = "params.undeclared_read / undeclared_write absolute paths.";
    usage_.inspect = "Spawns v4_sneak_proc.py; cwd=slot.";

    cgraph::apply_effect_yaml(effect_, YAML::Load(R"(
effect:
  class: external
  cache: fingerprinted
)"));
    effect_.seed_param.clear();
    set_shell(false);
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>&,
      const nlohmann::json& params, const cgraph::ExecContext& ctx) const override {
    if (ctx.sandbox == nullptr || ctx.workdir.empty()) {
      throw cgraph::OperatorError(cgraph::ErrorCode::SandboxViolation,
                                  "ProcessOperator: sandbox/workdir required");
    }
    const nlohmann::json params_obj =
        params.is_object() ? params : nlohmann::json::object();

    const std::string py = cgraph::resolve_python_executable();
    const std::string script =
        cgraph::resolve_test_script("v4_sneak_proc.py").string();

    std::vector<std::string> argv = {py, script, ctx.workdir.string()};
    if (params_obj.contains("undeclared_read") &&
        params_obj["undeclared_read"].is_string()) {
      const std::string p = params_obj["undeclared_read"].get<std::string>();
      if (!p.empty()) {
        argv.push_back("--read");
        argv.push_back(p);
      }
    }
    if (params_obj.contains("undeclared_write") &&
        params_obj["undeclared_write"].is_string()) {
      const std::string p = params_obj["undeclared_write"].get<std::string>();
      if (!p.empty()) {
        argv.push_back("--write");
        argv.push_back(p);
      }
    }

    cgraph::ProcessRunRequest req;
    req.argv = argv;
    req.cwd = ctx.workdir;
    req.log_path = ctx.workdir / "log.txt";
    req.sandbox_mode = ctx.sandbox_mode;
    req.allow_paths.push_back(ctx.workdir);
    for (const auto& [name, port] : signature_.inputs) {
      (void)port;
      req.allow_paths.push_back(ctx.sandbox->input_path(name));
    }
    req.allow_paths.push_back(std::filesystem::path(py));
    req.allow_paths.push_back(std::filesystem::path(script));

    const cgraph::ProcessRunResult run = cgraph::run_process(req);
    if (ctx.runtime != nullptr) {
      ctx.runtime->note_process_sandbox(run.sandbox_effective, run.sandbox_fallback);
    }
    cgraph::merge_process_meta(
        ctx.workdir,
        run.exit_code == 0 ? cgraph::SlotStatus::Succeeded : cgraph::SlotStatus::Failed,
        run.exit_code);

    if (run.timed_out) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed, "process timed out");
    }
    if (run.exit_code != 0) {
      throw cgraph::OperatorError(cgraph::ErrorCode::SandboxViolation,
                                  "sandbox_violation");
    }

    std::map<std::string, nlohmann::json> outs;
    for (const auto& [name, port] : signature_.outputs) {
      if (port.kind != cgraph::PortKind::Artifact) {
        continue;
      }
      const auto path = ctx.sandbox->output_path(name);
      if (!std::filesystem::exists(path)) {
        throw cgraph::OperatorError(cgraph::ErrorCode::RequestedOutputMissing,
                                    "process output missing: " + name);
      }
      outs[name] = ctx.sandbox->output_artifact(name, ctx.artifact_digest_mode);
    }
    return fx::wrap(signature_, outs);
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_subproc_sneak() {
  return std::make_shared<SubprocSneakOp>();
}

}  // namespace fixture
