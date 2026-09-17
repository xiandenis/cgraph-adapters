#include "cgraph/ops.hpp"
#include "fx_data.hpp"
#include "math_nd_util.hpp"

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fixture {
namespace {

int param_tile(const nlohmann::json& params, const char* primary,
               const char* alt, int fallback) {
  int v = math_nd::param_int(params, primary, -1);
  if (v > 0) {
    return v;
  }
  v = math_nd::param_int(params, alt, -1);
  if (v > 0) {
    return v;
  }
  // block_size applies to both axes when tile_* missing
  v = math_nd::param_int(params, "block_size", -1);
  if (v > 0) {
    return v;
  }
  return fallback;
}

bool param_bool(const nlohmann::json& params, const char* key, bool fallback) {
  if (!params.is_object() || !params.contains(key)) {
    return fallback;
  }
  const auto& j = params[key];
  if (j.is_boolean()) {
    return j.get<bool>();
  }
  if (j.is_number()) {
    return j.get<double>() != 0.0;
  }
  return fallback;
}

std::string body_name(const nlohmann::json& params) {
  if (!params.is_object() || !params.contains("body") ||
      !params["body"].is_string()) {
    return {};
  }
  return params["body"].get<std::string>();
}

double kernel_scale_scalar(const nlohmann::json& kernel, const char* op) {
  if (kernel.is_number()) {
    return kernel.get<double>();
  }
  const Eigen::MatrixXd K = math_nd::parse_matrix(kernel, op, "kernel");
  if (K.size() == 0) {
    throw std::invalid_argument(std::string(op) + ": empty kernel");
  }
  return K(0, 0);
}

nlohmann::json scale_patch(const nlohmann::json& patch, double k,
                           const char* op) {
  const Eigen::MatrixXd m = math_nd::parse_matrix(patch, op, "patch");
  return math_nd::make_nd(m * k);
}

// P0 Im2Col+GEMM stand-in: scale patch by kernel(0,0) (same as fx.gemm 1x1 path).
nlohmann::json body_im2col_matmul(const nlohmann::json& patch,
                                  const nlohmann::json* kernel,
                                  const char* op) {
  if (kernel == nullptr) {
    throw std::invalid_argument(std::string(op) +
                                ": Im2ColMatMul requires kernel input");
  }
  const double k = kernel_scale_scalar(*kernel, op);
  return scale_patch(patch, k, op);
}

nlohmann::json body_scale(const nlohmann::json& patch,
                          const nlohmann::json* kernel, const char* op) {
  double k = 1.0;
  if (kernel != nullptr) {
    k = kernel_scale_scalar(*kernel, op);
  }
  return scale_patch(patch, k, op);
}

nlohmann::json apply_body(const std::string& body, const nlohmann::json& item,
                          const nlohmann::json* kernel, const char* op) {
  if (body == "Im2ColMatMul" || body == "im2col_gemm") {
    return body_im2col_matmul(item, kernel, op);
  }
  if (body == "scale" || body == "Scale" || body.empty()) {
    return body_scale(item, kernel, op);
  }
  throw std::invalid_argument(std::string(op) + ": unknown body '" + body +
                              "'");
}

class SplitGridOp final : public cgraph::MemoryOperator {
 public:
  SplitGridOp() {
    op_id_ = "fx.split_grid";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.outputs["grid_shape"] =
        cgraph::make_port("grid_shape", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    cgraph::ParamSpec th;
    th.name = "tile_h";
    th.dtype = "int";
    th.default_value = 1;
    th.doc = "tile height (alias block_size)";
    signature_.params["tile_h"] = th;
    cgraph::ParamSpec tw = th;
    tw.name = "tile_w";
    tw.doc = "tile width (alias block_size)";
    signature_.params["tile_w"] = std::move(tw);
    cgraph::ParamSpec bs = th;
    bs.name = "block_size";
    bs.doc = "square tile when tile_h/tile_w omitted";
    signature_.params["block_size"] = std::move(bs);
    capability_.summary = "Split matrix into row-major list of tiles";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire matrix in; read patches out + grid_shape.";
    usage_.tune = "params.tile_h/tile_w or block_size.";
    usage_.inspect = "Non-divisible dims throw (no partial tiles).";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in,
      const nlohmann::json& params, const cgraph::ExecContext&) const override {
    const auto inputs = fx::unwrap(data_in);
    const Eigen::MatrixXd M = math_nd::parse_matrix(
        math_nd::require_input(inputs, "in", "fx.split_grid"), "fx.split_grid",
        "in");
    const int tile_h = param_tile(params, "tile_h", "tile_height", 1);
    const int tile_w = param_tile(params, "tile_w", "tile_width", tile_h);
    if (tile_h <= 0 || tile_w <= 0) {
      throw std::invalid_argument("fx.split_grid: tile size must be positive");
    }
    if (M.rows() % tile_h != 0 || M.cols() % tile_w != 0) {
      throw std::invalid_argument(
          "fx.split_grid: matrix dims must be divisible by tile size");
    }
    const int gh = static_cast<int>(M.rows() / tile_h);
    const int gw = static_cast<int>(M.cols() / tile_w);
    nlohmann::json patches = nlohmann::json::array();
    for (int bi = 0; bi < gh; ++bi) {
      for (int bj = 0; bj < gw; ++bj) {
        Eigen::MatrixXd tile = M.block(bi * tile_h, bj * tile_w, tile_h, tile_w);
        patches.push_back(math_nd::make_nd(tile));
      }
    }
    return fx::wrap(signature_, {{"out", std::move(patches)},
            {"grid_shape", nlohmann::json::array({gh, gw})}});
  }
};

class MergeGridOp final : public cgraph::MemoryOperator {
 public:
  MergeGridOp() {
    op_id_ = "fx.merge_grid";
    signature_.inputs["chunks"] =
        cgraph::make_port("chunks", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    auto in_alias = cgraph::make_port("in", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    in_alias.optional = true;
    signature_.inputs["in"] = std::move(in_alias);
    auto gs = cgraph::make_port("grid_shape", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    gs.optional = true;
    signature_.inputs["grid_shape"] = std::move(gs);
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    cgraph::ParamSpec rows;
    rows.name = "rows";
    rows.dtype = "int";
    signature_.params["rows"] = rows;
    cgraph::ParamSpec cols = rows;
    cols.name = "cols";
    signature_.params["cols"] = std::move(cols);
    capability_.summary = "Merge tile list back into a matrix";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire chunks (+ optional grid_shape); read out.";
    usage_.tune = "params.rows/cols or input grid_shape [gh,gw].";
    usage_.inspect = "Row-major tiles; null tiles become NaN blocks if present.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in,
      const nlohmann::json& params, const cgraph::ExecContext&) const override {
    const auto inputs = fx::unwrap(data_in);
    const nlohmann::json& chunks_j = math_nd::require_input_alias(
        inputs, "chunks", "in", "fx.merge_grid");
    if (!chunks_j.is_array() || chunks_j.empty()) {
      throw std::invalid_argument("fx.merge_grid: chunks must be non-empty array");
    }

    int gh = math_nd::param_int(params, "rows", -1);
    int gw = math_nd::param_int(params, "cols", -1);
    const auto gsit = inputs.find("grid_shape");
    if (gsit != inputs.end() && gsit->second.is_array() &&
        gsit->second.size() >= 2) {
      gh = gsit->second[0].get<int>();
      gw = gsit->second[1].get<int>();
    }
    if (gh <= 0 || gw <= 0) {
      // Infer square-ish grid from length
      const int n = static_cast<int>(chunks_j.size());
      const int side = static_cast<int>(std::lround(std::sqrt(static_cast<double>(n))));
      if (side * side != n) {
        throw std::invalid_argument(
            "fx.merge_grid: provide grid_shape or rows/cols");
      }
      gh = side;
      gw = side;
    }
    if (static_cast<int>(chunks_j.size()) != gh * gw) {
      throw std::invalid_argument("fx.merge_grid: chunks length != gh*gw");
    }

    // Probe first non-null tile for tile size
    int tile_h = -1;
    int tile_w = -1;
    for (const auto& t : chunks_j) {
      if (t.is_null() || (t.is_object() && t.contains("error"))) {
        continue;
      }
      const Eigen::MatrixXd sample =
          math_nd::parse_matrix(t, "fx.merge_grid", "chunk");
      tile_h = static_cast<int>(sample.rows());
      tile_w = static_cast<int>(sample.cols());
      break;
    }
    if (tile_h <= 0 || tile_w <= 0) {
      throw std::invalid_argument("fx.merge_grid: no valid tiles to infer size");
    }

    Eigen::MatrixXd out(gh * tile_h, gw * tile_w);
    out.setConstant(std::numeric_limits<double>::quiet_NaN());
    for (int bi = 0; bi < gh; ++bi) {
      for (int bj = 0; bj < gw; ++bj) {
        const auto& t = chunks_j[static_cast<std::size_t>(bi * gw + bj)];
        if (t.is_null() || (t.is_object() && t.contains("error"))) {
          continue;
        }
        const Eigen::MatrixXd tile =
            math_nd::parse_matrix(t, "fx.merge_grid", "chunk");
        if (tile.rows() != tile_h || tile.cols() != tile_w) {
          throw std::invalid_argument("fx.merge_grid: ragged tile sizes");
        }
        out.block(bi * tile_h, bj * tile_w, tile_h, tile_w) = tile;
      }
    }
    return fx::wrap(signature_, {{"out", math_nd::make_nd(out)}});
  }
};

class MapChunkOp final : public cgraph::MemoryOperator {
 public:
  MapChunkOp() {
    op_id_ = "fx.map_chunk";
    signature_.inputs["items"] =
        cgraph::make_port("items", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    auto chunks = cgraph::make_port("chunks", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    chunks.optional = true;
    signature_.inputs["chunks"] = std::move(chunks);
    auto kernel = cgraph::make_port("kernel", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    kernel.optional = true;
    signature_.inputs["kernel"] = std::move(kernel);
    auto shared = cgraph::make_port("shared", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    shared.optional = true;
    signature_.inputs["shared"] = std::move(shared);
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    cgraph::ParamSpec body;
    body.name = "body";
    body.dtype = "string";
    body.default_value = "scale";
    body.doc =
        "Im2ColMatMul|im2col_gemm (1x1 scale by kernel[0,0]) or scale";
    signature_.params["body"] = std::move(body);
    cgraph::ParamSpec ap;
    ap.name = "allow_partial";
    ap.dtype = "bool";
    ap.default_value = false;
    signature_.params["allow_partial"] = std::move(ap);
    capability_.summary =
        "Map built-in body over list items (P0 serial fixture Map)";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire items/chunks + optional kernel; read out list.";
    usage_.tune =
        "body=Im2ColMatMul|scale; allow_partial skips failed items as "
        "{error}.";
    usage_.inspect =
        "Im2ColMatMul P0: out_i = patch_i * kernel(0,0) (not full im2col).";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in,
      const nlohmann::json& params, const cgraph::ExecContext&) const override {
    const auto inputs = fx::unwrap(data_in);
    const nlohmann::json& items = math_nd::require_input_alias(
        inputs, "items", "chunks", "fx.map_chunk");
    if (!items.is_array()) {
      throw std::invalid_argument("fx.map_chunk: items must be a JSON array");
    }
    const nlohmann::json* kernel = nullptr;
    const auto kit = inputs.find("kernel");
    if (kit != inputs.end()) {
      kernel = &kit->second;
    } else {
      const auto sit = inputs.find("shared");
      if (sit != inputs.end()) {
        kernel = &sit->second;
      }
    }
    const std::string body = body_name(params);
    const bool allow_partial = param_bool(params, "allow_partial", false);

    nlohmann::json out = nlohmann::json::array();
    for (std::size_t i = 0; i < items.size(); ++i) {
      try {
        out.push_back(apply_body(body, items[i], kernel, "fx.map_chunk"));
      } catch (const std::exception& ex) {
        if (!allow_partial) {
          throw;
        }
        out.push_back(nlohmann::json{{"error", ex.what()}, {"index", i}});
      }
    }
    return fx::wrap(signature_, {{"out", std::move(out)}});
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_split_grid() {
  return std::make_shared<SplitGridOp>();
}
std::shared_ptr<cgraph::MemoryOperator> make_merge_grid() {
  return std::make_shared<MergeGridOp>();
}
std::shared_ptr<cgraph::MemoryOperator> make_map_chunk() {
  return std::make_shared<MapChunkOp>();
}

}  // namespace fixture
