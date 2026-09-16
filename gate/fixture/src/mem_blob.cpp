#include "cgraph/artifact.hpp"
#include "cgraph/ops.hpp"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>

namespace fixture {
namespace {

int require_nbytes(const nlohmann::json& params) {
  if (!params.is_object() || !params.contains("nbytes") ||
      !params["nbytes"].is_number_integer()) {
    throw std::invalid_argument("fx.mem_blob_write: params.nbytes must be int");
  }
  const int n = params["nbytes"].get<int>();
  if (n < 0) {
    throw std::invalid_argument("fx.mem_blob_write: nbytes must be >= 0");
  }
  return n;
}

std::string require_tag(const nlohmann::json& params, std::string_view op) {
  if (!params.is_object() || !params.contains("tag") || !params["tag"].is_string()) {
    throw std::invalid_argument(std::string(op) + ": params.tag must be string");
  }
  return params["tag"].get<std::string>();
}

std::filesystem::path require_path(const nlohmann::json& params, std::string_view op) {
  if (!params.is_object() || !params.contains("path") || !params["path"].is_string()) {
    throw std::invalid_argument(std::string(op) + ": params.path must be string");
  }
  return params["path"].get<std::string>();
}

std::string tagged_payload(int nbytes, const std::string& tag) {
  std::string payload = tag;
  if (static_cast<int>(payload.size()) > nbytes) {
    payload.resize(static_cast<std::size_t>(nbytes));
  } else {
    payload.resize(static_cast<std::size_t>(nbytes), '\0');
  }
  return payload;
}

nlohmann::json write_artifact_file(const std::filesystem::path& out,
                                   std::string_view bytes,
                                   const cgraph::ExecContext& ctx) {
  std::filesystem::create_directories(out.parent_path());
  std::ofstream file(out, std::ios::binary | std::ios::trunc);
  if (!file) {
    throw std::runtime_error("cannot write " + out.string());
  }
  file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  file.close();
  return cgraph::artifact_to_json(cgraph::make_file_artifact(
      out, cgraph::DType::parse("file"), ctx.artifact_digest_mode));
}

class MemBlobWriteOp final : public cgraph::MemoryOperator {
 public:
  MemBlobWriteOp() {
    op_id_ = "fx.mem_blob_write";
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Artifact, "file");
    cgraph::ParamSpec nbytes;
    nbytes.name = "nbytes";
    nbytes.dtype = "int";
    nbytes.invalidate = true;
    signature_.params["nbytes"] = std::move(nbytes);
    cgraph::ParamSpec tag;
    tag.name = "tag";
    tag.dtype = "string";
    tag.invalidate = true;
    tag.bindable = false;
    signature_.params["tag"] = std::move(tag);
    capability_.summary = "Write a tagged byte blob to an Artifact slot path";
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "No inputs; read Artifact out.";
    usage_.tune = "params.nbytes and params.tag (both invalidate).";
    usage_.inspect = "File length is nbytes; prefix is tag.";
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>&, const nlohmann::json& params,
      const cgraph::ExecContext& ctx) const override {
    const int nbytes = require_nbytes(params);
    const std::string tag = require_tag(params, "fx.mem_blob_write");
    return {{"out", write_artifact_file(ctx.output_path("out"),
                                       tagged_payload(nbytes, tag), ctx)}};
  }
};

class MemBlobXformOp final : public cgraph::MemoryOperator {
 public:
  MemBlobXformOp() {
    op_id_ = "fx.mem_blob_xform";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Artifact, "file");
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Artifact, "file");
    cgraph::ParamSpec tag;
    tag.name = "tag";
    tag.dtype = "string";
    tag.invalidate = true;
    tag.bindable = false;
    signature_.params["tag"] = std::move(tag);
    capability_.summary = "Append a tag to an Artifact blob";
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire Artifact in; read Artifact out.";
    usage_.tune = "params.tag (invalidate).";
    usage_.inspect = "out bytes = in bytes + tag.";
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>&, const nlohmann::json& params,
      const cgraph::ExecContext& ctx) const override {
    const std::string tag = require_tag(params, "fx.mem_blob_xform");
    const cgraph::Artifact src = ctx.input_artifact("in");
    std::ifstream in(src.path, std::ios::binary);
    if (!in) {
      throw std::runtime_error("fx.mem_blob_xform: cannot read " + src.path.string());
    }
    std::string payload((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());
    payload += tag;
    return {{"out", write_artifact_file(ctx.output_path("out"), payload, ctx)}};
  }
};

class MemBlobSneakOp final : public cgraph::MemoryOperator {
 public:
  MemBlobSneakOp() {
    op_id_ = "fx.mem_blob_sneak";
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Artifact, "file");
    cgraph::ParamSpec path;
    path.name = "path";
    path.dtype = "string";
    path.invalidate = true;
    path.bindable = false;
    signature_.params["path"] = std::move(path);
    cgraph::ParamSpec nbytes;
    nbytes.name = "nbytes";
    nbytes.dtype = "int";
    nbytes.invalidate = true;
    signature_.params["nbytes"] = std::move(nbytes);
    cgraph::ParamSpec tag;
    tag.name = "tag";
    tag.dtype = "string";
    tag.invalidate = true;
    tag.bindable = false;
    signature_.params["tag"] = std::move(tag);
    capability_.summary = "Test-only: write an Artifact outside the slot";
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Test-only. Do not put on the Studio palette.";
    usage_.tune = "params.path is an off-slot absolute path.";
    usage_.inspect = "Returns a handle to path; kernel must reject it.";
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>&, const nlohmann::json& params,
      const cgraph::ExecContext& ctx) const override {
    const auto out = require_path(params, "fx.mem_blob_sneak");
    const int nbytes = require_nbytes(params);
    const std::string tag = require_tag(params, "fx.mem_blob_sneak");
    return {{"out", write_artifact_file(out, tagged_payload(nbytes, tag), ctx)}};
  }
};

class MemValueSmuggleOp final : public cgraph::MemoryOperator {
 public:
  MemValueSmuggleOp() {
    op_id_ = "fx.mem_value_smuggle";
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, "json");
    cgraph::ParamSpec path;
    path.name = "path";
    path.dtype = "string";
    path.invalidate = true;
    path.bindable = false;
    signature_.params["path"] = std::move(path);
    cgraph::ParamSpec nbytes;
    nbytes.name = "nbytes";
    nbytes.dtype = "int";
    nbytes.invalidate = true;
    signature_.params["nbytes"] = std::move(nbytes);
    cgraph::ParamSpec tag;
    tag.name = "tag";
    tag.dtype = "string";
    tag.invalidate = true;
    tag.bindable = false;
    signature_.params["tag"] = std::move(tag);
    capability_.summary = "Test-only: smuggle an Artifact handle on a Value port";
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Test-only. Do not put on the Studio palette.";
    usage_.tune = "params.path is an off-slot file written as a handle.";
    usage_.inspect = "Value out is an artifact handle JSON; kernel must reject it.";
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>&, const nlohmann::json& params,
      const cgraph::ExecContext& ctx) const override {
    const auto out = require_path(params, "fx.mem_value_smuggle");
    const int nbytes = require_nbytes(params);
    const std::string tag = require_tag(params, "fx.mem_value_smuggle");
    return {{"out", write_artifact_file(out, tagged_payload(nbytes, tag), ctx)}};
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_mem_blob_write() {
  return std::make_shared<MemBlobWriteOp>();
}

std::shared_ptr<cgraph::MemoryOperator> make_mem_blob_xform() {
  return std::make_shared<MemBlobXformOp>();
}

std::shared_ptr<cgraph::MemoryOperator> make_mem_blob_sneak() {
  return std::make_shared<MemBlobSneakOp>();
}

std::shared_ptr<cgraph::MemoryOperator> make_mem_value_smuggle() {
  return std::make_shared<MemValueSmuggleOp>();
}

}  // namespace fixture
