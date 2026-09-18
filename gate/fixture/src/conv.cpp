#include "cgraph/ops.hpp"
#include "fx_data.hpp"
#include "fx_typed.hpp"

#include <memory>
#include <stdexcept>
#include <string>

namespace fixture {
namespace {

nlohmann::json require_nd(const std::map<std::string, nlohmann::json>& inputs,
                          const std::string& port, const char* op) {
  const auto it = inputs.find(port);
  if (it == inputs.end() || !it->second.is_object() || !it->second.contains("data") ||
      !it->second.contains("shape")) {
    throw std::invalid_argument(std::string(op) + ": missing ndarray '" + port + "'");
  }
  return it->second;
}

double kernel_scale(const nlohmann::json& kernel) {
  const auto& data = kernel.at("data");
  if (!data.is_array() || data.empty() || !data[0].is_number()) {
    throw std::invalid_argument("kernel data must be a non-empty number array");
  }
  return data[0].get<double>();
}

nlohmann::json scaled_copy(const nlohmann::json& image, double k) {
  nlohmann::json out = image;
  nlohmann::json data = nlohmann::json::array();
  for (const auto& x : image.at("data")) {
    data.push_back(x.get<double>() * k);
  }
  out["data"] = std::move(data);
  return out;
}

class Filter2dOp final : public cgraph::MemoryOperator {
 public:
  Filter2dOp() {
    op_id_ = "fx.filter2d";
    signature_.inputs["image"] =
        cgraph::make_port("image", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.inputs["kernel"] =
        cgraph::make_port("kernel", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    capability_.summary = "P0 fused 1x1 conv (Filter2D)";
    capability_.tags = {"conv", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "image and kernel ndarray JSON → out.";
    usage_.tune = "P0: 1x1 kernel, pad=0, stride=1.";
    usage_.inspect = "out.data[i] = image.data[i] * kernel.data[0].";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const auto inputs = fx::unwrap(data_in);
    const nlohmann::json image = require_nd(inputs, "image", "fx.filter2d");
    const nlohmann::json kernel = require_nd(inputs, "kernel", "fx.filter2d");
    return fx::wrap(signature_, {{"out", scaled_copy(image, kernel_scale(kernel))}});
  }
};

class PadOp final : public cgraph::MemoryOperator {
 public:
  PadOp() {
    op_id_ = "fx.pad";
    signature_.inputs["in"] = fx_typed::matrix_port("in");
    signature_.outputs["out"] = fx_typed::matrix_port("out");
    cgraph::ParamSpec ph;
    ph.name = "pad_h";
    ph.dtype = "int";
    ph.default_value = 0;
    signature_.params["pad_h"] = ph;
    cgraph::ParamSpec pw = ph;
    pw.name = "pad_w";
    signature_.params["pad_w"] = pw;
    capability_.summary = "P0 pad; pad=0 is identity";
    capability_.tags = {"conv", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire matrix in; read matrix out.";
    usage_.tune = "pad_h/pad_w. Zero is identity.";
    usage_.inspect = "P0 only implements pad=0 copy.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in,
      const nlohmann::json& params, const cgraph::ExecContext&) const override {
    const Eigen::MatrixXd image = fx_typed::require_matrix(
        fx_typed::require_obj(data_in, "in", "fx.pad"), "fx.pad", "in");
    int pad_h = 0;
    int pad_w = 0;
    if (params.is_object()) {
      if (params.contains("pad_h") && params["pad_h"].is_number_integer()) {
        pad_h = params["pad_h"].get<int>();
      }
      if (params.contains("pad_w") && params["pad_w"].is_number_integer()) {
        pad_w = params["pad_w"].get<int>();
      }
    }
    if (pad_h != 0 || pad_w != 0) {
      throw std::invalid_argument("fx.pad: P0 only supports pad_h=pad_w=0");
    }
    return {{"out", fx_typed::make_matrix(image)}};
  }
};

class Im2ColOp final : public cgraph::MemoryOperator {
 public:
  Im2ColOp() {
    op_id_ = "fx.im2col";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    capability_.summary = "P0 im2col for 1x1 kernel";
    capability_.tags = {"conv", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "padded image → column matrix.";
    usage_.tune = "P0: flatten data to shape [N,1].";
    usage_.inspect = "Does not depend on kernel.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const auto inputs = fx::unwrap(data_in);
    const nlohmann::json image = require_nd(inputs, "in", "fx.im2col");
    nlohmann::json out = image;
    const auto n = image.at("data").size();
    out["shape"] = nlohmann::json::array({static_cast<int>(n), 1});
    return fx::wrap(signature_, {{"out", std::move(out)}});
  }
};

class FlattenKernelOp final : public cgraph::MemoryOperator {
 public:
  FlattenKernelOp() {
    op_id_ = "fx.flatten_kernel";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    capability_.summary = "Flatten kernel ndarray";
    capability_.tags = {"conv", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "kernel → column.";
    usage_.tune = "No parameters.";
    usage_.inspect = "shape [K,1].";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const auto inputs = fx::unwrap(data_in);
    const nlohmann::json kernel = require_nd(inputs, "in", "fx.flatten_kernel");
    nlohmann::json out = kernel;
    const auto n = kernel.at("data").size();
    out["shape"] = nlohmann::json::array({static_cast<int>(n), 1});
    return fx::wrap(signature_, {{"out", std::move(out)}});
  }
};

class GemmOp final : public cgraph::MemoryOperator {
 public:
  GemmOp() {
    op_id_ = "fx.gemm";
    signature_.inputs["a"] =
        cgraph::make_port("a", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.inputs["b"] =
        cgraph::make_port("b", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    capability_.summary = "P0 GEMM: column * scalar kernel";
    capability_.tags = {"conv", "math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "a is im2col, b is flattened kernel.";
    usage_.tune = "No parameters.";
    usage_.inspect = "out.data[i] = a.data[i] * b.data[0].";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const auto inputs = fx::unwrap(data_in);
    const nlohmann::json a = require_nd(inputs, "a", "fx.gemm");
    const nlohmann::json b = require_nd(inputs, "b", "fx.gemm");
    nlohmann::json out = a;
    out["data"] = scaled_copy(a, kernel_scale(b)).at("data");
    return fx::wrap(signature_, {{"out", std::move(out)}});
  }
};

class ReshapeOp final : public cgraph::MemoryOperator {
 public:
  ReshapeOp() {
    op_id_ = "fx.reshape";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    cgraph::ParamSpec shape;
    shape.name = "shape";
    shape.dtype = "json";
    shape.bindable = false;
    signature_.params["shape"] = shape;
    capability_.summary = "Reshape ndarray JSON";
    capability_.tags = {"conv", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "column → image shape.";
    usage_.tune = "params.shape = [H, W].";
    usage_.inspect = "data bytes unchanged.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in,
      const nlohmann::json& params, const cgraph::ExecContext&) const override {
    const auto inputs = fx::unwrap(data_in);
    nlohmann::json image = require_nd(inputs, "in", "fx.reshape");
    if (!params.is_object() || !params.contains("shape") || !params["shape"].is_array()) {
      throw std::invalid_argument("fx.reshape: params.shape required");
    }
    image["shape"] = params["shape"];
    return fx::wrap(signature_, {{"out", std::move(image)}});
  }
};

class MulOp final : public cgraph::MemoryOperator {
 public:
  MulOp() {
    op_id_ = "fx.mul";
    signature_.inputs["a"] = fixture::fx_typed::float_port("a");
    signature_.inputs["b"] = fixture::fx_typed::float_port("b");
    signature_.outputs["out"] = fixture::fx_typed::float_port("out");
    capability_.summary = "Multiply two floats";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire floats a,b; read out.";
    usage_.tune = "No parameters. For ndarray scale use fx.gemm / filter2d.";
    usage_.inspect = "out = a * b.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const double a =
        fixture::fx_typed::require_float_port(data_in, "a", "fx.mul");
    const double b =
        fixture::fx_typed::require_float_port(data_in, "b", "fx.mul");
    return {{"out", fixture::fx_typed::make_number_float(a * b)}};
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_filter2d() {
  return std::make_shared<Filter2dOp>();
}
std::shared_ptr<cgraph::MemoryOperator> make_pad() { return std::make_shared<PadOp>(); }
std::shared_ptr<cgraph::MemoryOperator> make_im2col() {
  return std::make_shared<Im2ColOp>();
}
std::shared_ptr<cgraph::MemoryOperator> make_flatten_kernel() {
  return std::make_shared<FlattenKernelOp>();
}
std::shared_ptr<cgraph::MemoryOperator> make_gemm() {
  return std::make_shared<GemmOp>();
}
std::shared_ptr<cgraph::MemoryOperator> make_reshape() {
  return std::make_shared<ReshapeOp>();
}
std::shared_ptr<cgraph::MemoryOperator> make_mul() { return std::make_shared<MulOp>(); }

}  // namespace fixture
