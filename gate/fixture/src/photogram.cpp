#include "cgraph/ops.hpp"
#include "fx_data.hpp"

#include <memory>

namespace fixture {
namespace {

class PhotogramSfm final : public cgraph::MemoryOperator {
 public:
  PhotogramSfm() {
    op_id_ = "photogram.sfm";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("number"));
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("number"));
    capability_.summary = "Photogrammetry SfM stub (kernel fixture, not AliceVision)";
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire views json into in; read sparse reconstruction json.";
    usage_.tune = "No parameters on the stub.";
    usage_.inspect = "out wraps in under key sfm.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const auto inputs = fx::unwrap(data_in);
    nlohmann::json views = nullptr;
    if (const auto it = inputs.find("in"); it != inputs.end()) {
      views = it->second;
    }
    return fx::wrap(signature_, {{"out", nlohmann::json{{"sfm", std::move(views)}}}});
  }
};

class PhotogramMesh final : public cgraph::MemoryOperator {
 public:
  PhotogramMesh() {
    op_id_ = "photogram.mesh";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("number"));
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("number"));
    capability_.summary = "Photogrammetry meshing stub (kernel fixture)";
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire SfM json into in; read mesh json.";
    usage_.tune = "No parameters on the stub.";
    usage_.inspect = "out wraps in under key mesh.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const auto inputs = fx::unwrap(data_in);
    nlohmann::json sfm = nullptr;
    if (const auto it = inputs.find("in"); it != inputs.end()) {
      sfm = it->second;
    }
    return fx::wrap(signature_, {{"out", nlohmann::json{{"mesh", std::move(sfm)}}}});
  }
};

class PhotogramTexturing final : public cgraph::MemoryOperator {
 public:
  PhotogramTexturing() {
    op_id_ = "photogram.texturing";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("number"));
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("number"));
    cgraph::ParamSpec down;
    down.name = "textureDownscale";
    down.dtype = "int";
    down.default_value = 2;
    down.doc = "Texture downscale factor; changing it must not rerun SfM/mesh.";
    signature_.params["textureDownscale"] = std::move(down);
    capability_.summary = "Photogrammetry texturing stub (kernel fixture)";
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire mesh json into in; read textured mesh json.";
    usage_.tune = "textureDownscale is an identity param (invalidate=true).";
    usage_.inspect = "out includes mesh and textureDownscale.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in,
      const nlohmann::json& params, const cgraph::ExecContext&) const override {
    const auto inputs = fx::unwrap(data_in);
    nlohmann::json mesh = nullptr;
    if (const auto it = inputs.find("in"); it != inputs.end()) {
      mesh = it->second;
    }
    int downscale = 2;
    if (params.is_object() && params.contains("textureDownscale") &&
        params["textureDownscale"].is_number_integer()) {
      downscale = params["textureDownscale"].get<int>();
    }
    return fx::wrap(signature_, {{"out", nlohmann::json{{"mesh", std::move(mesh)},
                                   {"textureDownscale", downscale}}}});
  }
};

}  // namespace

void register_photogram_ops(cgraph::OpRegistry& registry) {
  registry.add(std::make_shared<PhotogramSfm>());
  registry.add(std::make_shared<PhotogramMesh>());
  registry.add(std::make_shared<PhotogramTexturing>());
}

}  // namespace fixture
