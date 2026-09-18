#include "cgraph/ops.hpp"
#include "fx_typed.hpp"

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fixture {
namespace {

int param_int(const nlohmann::json& params, const char* key, int fallback) {
  if (!params.is_object() || !params.contains(key)) {
    return fallback;
  }
  if (params[key].is_number_integer()) {
    return params[key].get<int>();
  }
  if (params[key].is_number()) {
    return static_cast<int>(params[key].get<double>());
  }
  return fallback;
}

int param_tile(const nlohmann::json& params, const char* primary,
               const char* alt, int fallback) {
  int v = param_int(params, primary, -1);
  if (v > 0) {
    return v;
  }
  v = param_int(params, alt, -1);
  if (v > 0) {
    return v;
  }
  v = param_int(params, "block_size", -1);
  if (v > 0) {
    return v;
  }
  return fallback;
}

bool param_bool(const nlohmann::json& params, const char* key, bool fallback) {
  if (!params.is_object() || !params.contains(key)) {
    return fallback;
  }
  if (params[key].is_boolean()) {
    return params[key].get<bool>();
  }
  if (params[key].is_number()) {
    return params[key].get<double>() != 0.0;
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

Eigen::MatrixXd scale_patch(const Eigen::MatrixXd& m, double k) {
  return m * k;
}

Eigen::MatrixXd apply_body(const std::string& body, const Eigen::MatrixXd& item,
                           const Eigen::MatrixXd* kernel, const char* op) {
  double k = 1.0;
  if (kernel != nullptr) {
    if (kernel->size() == 0) {
      throw std::invalid_argument(std::string(op) + ": empty kernel");
    }
    k = (*kernel)(0, 0);
  }
  if (body == "Im2ColMatMul" || body == "im2col_gemm") {
    if (kernel == nullptr) {
      throw std::invalid_argument(std::string(op) +
                                  ": Im2ColMatMul requires kernel");
    }
    return scale_patch(item, k);
  }
  if (body == "scale" || body == "Scale" || body.empty()) {
    return scale_patch(item, k);
  }
  throw std::invalid_argument(std::string(op) + ": unknown body '" + body +
                              "'");
}

const cgraph::DataObject& require_alias(
    const std::map<std::string, cgraph::DataObject>& data_in, const char* primary,
    const char* alias, const char* op) {
  if (data_in.count(primary) != 0) {
    return data_in.at(primary);
  }
  if (data_in.count(alias) != 0) {
    return data_in.at(alias);
  }
  throw std::invalid_argument(std::string(op) + ": missing '" + primary +
                              "' (or '" + alias + "')");
}

class SplitGridOp final : public cgraph::MemoryOperator {
 public:
  SplitGridOp() {
    op_id_ = "fx.split_grid";
    signature_.inputs["in"] = fx_typed::matrix_port("in");
    signature_.outputs["out"] = fx_typed::matrix_list_port("out");
    signature_.outputs["grid_shape"] = fx_typed::vector_port("grid_shape");
    cgraph::ParamSpec th;
    th.name = "tile_h";
    th.dtype = "int";
    th.default_value = 1;
    signature_.params["tile_h"] = th;
    cgraph::ParamSpec tw = th;
    tw.name = "tile_w";
    signature_.params["tile_w"] = std::move(tw);
    cgraph::ParamSpec bs = th;
    bs.name = "block_size";
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
    const Eigen::MatrixXd M = fx_typed::require_matrix(
        fx_typed::require_obj(data_in, "in", "fx.split_grid"), "fx.split_grid",
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
    std::vector<Eigen::MatrixXd> patches;
    patches.reserve(static_cast<std::size_t>(gh * gw));
    for (int bi = 0; bi < gh; ++bi) {
      for (int bj = 0; bj < gw; ++bj) {
        patches.push_back(
            M.block(bi * tile_h, bj * tile_w, tile_h, tile_w));
      }
    }
    Eigen::VectorXd gs(2);
    gs << static_cast<double>(gh), static_cast<double>(gw);
    return {{"out", fx_typed::make_matrix_list(patches)},
            {"grid_shape", fx_typed::make_vector(gs)}};
  }
};

class MergeGridOp final : public cgraph::MemoryOperator {
 public:
  MergeGridOp() {
    op_id_ = "fx.merge_grid";
    signature_.inputs["chunks"] = fx_typed::matrix_list_port("chunks");
    auto in_alias = fx_typed::matrix_list_port("in");
    in_alias.optional = true;
    signature_.inputs["in"] = std::move(in_alias);
    auto gs = fx_typed::vector_port("grid_shape");
    gs.optional = true;
    signature_.inputs["grid_shape"] = std::move(gs);
    signature_.outputs["out"] = fx_typed::matrix_port("out");
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
    usage_.inspect = "Row-major tiles.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in,
      const nlohmann::json& params, const cgraph::ExecContext&) const override {
    const auto chunks = fx_typed::require_matrix_list(
        require_alias(data_in, "chunks", "in", "fx.merge_grid"), "fx.merge_grid",
        "chunks");
    if (chunks.empty()) {
      throw std::invalid_argument("fx.merge_grid: chunks must be non-empty");
    }
    int gh = param_int(params, "rows", -1);
    int gw = param_int(params, "cols", -1);
    if (data_in.count("grid_shape") != 0) {
      const Eigen::VectorXd gs = fx_typed::require_vector(
          data_in.at("grid_shape"), "fx.merge_grid", "grid_shape");
      if (gs.size() >= 2) {
        gh = static_cast<int>(gs(0));
        gw = static_cast<int>(gs(1));
      }
    }
    if (gh <= 0 || gw <= 0) {
      const int n = static_cast<int>(chunks.size());
      const int side =
          static_cast<int>(std::lround(std::sqrt(static_cast<double>(n))));
      if (side * side != n) {
        throw std::invalid_argument(
            "fx.merge_grid: provide grid_shape or rows/cols");
      }
      gh = side;
      gw = side;
    }
    if (static_cast<int>(chunks.size()) != gh * gw) {
      throw std::invalid_argument("fx.merge_grid: chunks length != gh*gw");
    }
    const int tile_h = static_cast<int>(chunks[0].rows());
    const int tile_w = static_cast<int>(chunks[0].cols());
    Eigen::MatrixXd out(gh * tile_h, gw * tile_w);
    out.setConstant(std::numeric_limits<double>::quiet_NaN());
    for (int bi = 0; bi < gh; ++bi) {
      for (int bj = 0; bj < gw; ++bj) {
        const auto& tile = chunks[static_cast<std::size_t>(bi * gw + bj)];
        if (tile.rows() != tile_h || tile.cols() != tile_w) {
          throw std::invalid_argument("fx.merge_grid: ragged tile sizes");
        }
        out.block(bi * tile_h, bj * tile_w, tile_h, tile_w) = tile;
      }
    }
    return {{"out", fx_typed::make_matrix(out)}};
  }
};

class MapChunkOp final : public cgraph::MemoryOperator {
 public:
  MapChunkOp() {
    op_id_ = "fx.map_chunk";
    signature_.inputs["items"] = fx_typed::matrix_list_port("items");
    auto chunks = fx_typed::matrix_list_port("chunks");
    chunks.optional = true;
    signature_.inputs["chunks"] = std::move(chunks);
    auto kernel = fx_typed::matrix_port("kernel");
    kernel.optional = true;
    signature_.inputs["kernel"] = std::move(kernel);
    auto shared = fx_typed::matrix_port("shared");
    shared.optional = true;
    signature_.inputs["shared"] = std::move(shared);
    signature_.outputs["out"] = fx_typed::matrix_list_port("out");
    cgraph::ParamSpec body;
    body.name = "body";
    body.dtype = "string";
    body.default_value = "scale";
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
    usage_.tune = "body=Im2ColMatMul|scale; allow_partial rethrows if false.";
    usage_.inspect = "Im2ColMatMul P0: out_i = patch_i * kernel(0,0).";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in,
      const nlohmann::json& params, const cgraph::ExecContext&) const override {
    const auto items = fx_typed::require_matrix_list(
        require_alias(data_in, "items", "chunks", "fx.map_chunk"), "fx.map_chunk",
        "items");
    const Eigen::MatrixXd* kernel = nullptr;
    Eigen::MatrixXd kstore;
    if (data_in.count("kernel") != 0) {
      kstore = fx_typed::require_matrix(data_in.at("kernel"), "fx.map_chunk",
                                        "kernel");
      kernel = &kstore;
    } else if (data_in.count("shared") != 0) {
      kstore = fx_typed::require_matrix(data_in.at("shared"), "fx.map_chunk",
                                        "shared");
      kernel = &kstore;
    }
    const std::string body = body_name(params);
    const bool allow_partial = param_bool(params, "allow_partial", false);
    std::vector<Eigen::MatrixXd> out;
    out.reserve(items.size());
    for (std::size_t i = 0; i < items.size(); ++i) {
      try {
        out.push_back(apply_body(body, items[i], kernel, "fx.map_chunk"));
      } catch (const std::exception&) {
        if (!allow_partial) {
          throw;
        }
        // Skip failed tile under allow_partial (typed list cannot hold error
        // objects); omit entry by pushing NaN 1x1 marker.
        Eigen::MatrixXd nan(1, 1);
        nan(0, 0) = std::numeric_limits<double>::quiet_NaN();
        out.push_back(nan);
      }
    }
    return {{"out", fx_typed::make_matrix_list(out)}};
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
