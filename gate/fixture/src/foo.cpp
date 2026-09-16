#include "cgraph/ops.hpp"

#include <memory>
#include <stdexcept>

namespace fixture {
namespace {

class FooWrapOp final : public cgraph::MemoryOperator {
 public:
  FooWrapOp() {
    op_id_ = "fx.foo_wrap";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, "json");
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, "foo");
    capability_.summary = "Wrap json as plugin dtype foo";
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire json into in; read foo from out. Do not connect to cloud.";
    usage_.tune = "No parameters.";
    usage_.inspect = "out equals in; digest is canon(in).";
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const auto it = inputs.find("in");
    if (it == inputs.end()) {
      throw std::invalid_argument("fx.foo_wrap: missing input 'in'");
    }
    return {{"out", it->second}};
  }
};

class FooUnwrapOp final : public cgraph::MemoryOperator {
 public:
  FooUnwrapOp() {
    op_id_ = "fx.foo_unwrap";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, "foo");
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, "json");
    capability_.summary = "Unwrap plugin dtype foo to json";
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire foo into in; read json from out.";
    usage_.tune = "No parameters.";
    usage_.inspect = "out equals in.";
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const auto it = inputs.find("in");
    if (it == inputs.end()) {
      throw std::invalid_argument("fx.foo_unwrap: missing input 'in'");
    }
    return {{"out", it->second}};
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_foo_wrap() {
  return std::make_shared<FooWrapOp>();
}

std::shared_ptr<cgraph::MemoryOperator> make_foo_unwrap() {
  return std::make_shared<FooUnwrapOp>();
}

}  // namespace fixture
