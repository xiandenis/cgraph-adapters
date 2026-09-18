#include "cgraph/ops.hpp"
#include "fx_typed.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fixture {
namespace {

namespace fs = std::filesystem;

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

std::string require_path_param(const nlohmann::json& params, const char* op) {
  if (!params.is_object() || !params.contains("path") ||
      !params["path"].is_string()) {
    throw std::invalid_argument(std::string(op) + ": params.path required");
  }
  return params["path"].get<std::string>();
}

std::vector<double> read_csv_numbers(const fs::path& path, const char* op) {
  std::ifstream in(path);
  if (!in) {
    throw std::invalid_argument(std::string(op) + ": cannot open " +
                                path.string());
  }
  std::vector<double> xs;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    std::stringstream ss(line);
    std::string tok;
    std::string last;
    while (std::getline(ss, tok, ',')) {
      std::stringstream ts(tok);
      std::string w;
      while (ts >> w) {
        last = w;
      }
    }
    if (last.empty() || last == "z" || last == "meas" || last == "value") {
      continue;
    }
    try {
      xs.push_back(std::stod(last));
    } catch (...) {
      throw std::invalid_argument(std::string(op) + ": bad CSV number in " +
                                  path.string());
    }
  }
  if (xs.empty()) {
    throw std::invalid_argument(std::string(op) + ": empty CSV " +
                                path.string());
  }
  return xs;
}

struct KalmanResult {
  double final_state = 0.0;
  double final_cov = 1.0;
  std::vector<double> innovations;
  double mean_innov = 0.0;
};

KalmanResult kalman_1d(const std::vector<double>& zs, double Q, double R) {
  if (!(Q >= 0.0) || !(R > 0.0)) {
    throw std::invalid_argument("kalman: Q>=0 and R>0 required");
  }
  KalmanResult r;
  double x = 0.0;
  double P = 1.0;
  r.innovations.reserve(zs.size());
  double innov_sum = 0.0;
  for (double z : zs) {
    const double x_pred = x;
    const double P_pred = P + Q;
    const double S = P_pred + R;
    const double K = P_pred / S;
    const double innov = z - x_pred;
    x = x_pred + K * innov;
    P = (1.0 - K) * P_pred;
    r.innovations.push_back(innov);
    innov_sum += innov;
  }
  r.final_state = x;
  r.final_cov = P;
  r.mean_innov =
      zs.empty() ? 0.0 : innov_sum / static_cast<double>(zs.size());
  return r;
}

std::vector<fs::path> list_csv_files(const fs::path& dir) {
  std::vector<fs::path> files;
  if (!fs::is_directory(dir)) {
    throw std::invalid_argument("not a directory: " + dir.string());
  }
  for (const auto& ent : fs::directory_iterator(dir)) {
    if (!ent.is_regular_file()) {
      continue;
    }
    const auto ext = ent.path().extension().string();
    if (ext == ".csv" || ext == ".CSV") {
      files.push_back(ent.path());
    }
  }
  std::sort(files.begin(), files.end());
  return files;
}

std::vector<fs::path> resolve_file_list(const cgraph::DataObject& items,
                                        const char* op) {
  if (fx_typed::is_directory_artifact(items)) {
    return list_csv_files(fx_typed::require_directory_uri(items, op, "items"));
  }
  if (items.type_id == type_string_list() ||
      items.payload.kind() == cgraph::Payload::Kind::List) {
    const auto paths = fx_typed::require_string_list(items, op, "items");
    std::vector<fs::path> out;
    out.reserve(paths.size());
    for (const auto& p : paths) {
      out.emplace_back(p);
    }
    return out;
  }
  throw std::invalid_argument(std::string(op) +
                              ": items must be directory Artifact or "
                              "list[string] paths");
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

class InputArtifactDirOp final : public cgraph::MemoryOperator {
 public:
  InputArtifactDirOp() {
    op_id_ = "fx.input_artifact_dir";
    signature_.outputs["out"] = fx_typed::directory_artifact_port("out");
    cgraph::ParamSpec path;
    path.name = "path";
    path.dtype = "string";
    path.invalidate = true;
    path.bindable = false;
    signature_.params["path"] = std::move(path);
    capability_.summary =
        "Emit a directory Artifact for a fixture CSV folder";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "params.path → out directory Artifact.";
    usage_.tune = "Point at a tiny fixture dir (2–3 csv).";
    usage_.inspect = "DataRef uri + content_hash from make_dir_artifact.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>&,
      const nlohmann::json& params, const cgraph::ExecContext&) const override {
    const fs::path dir = require_path_param(params, "fx.input_artifact_dir");
    if (!fs::is_directory(dir)) {
      throw std::invalid_argument("fx.input_artifact_dir: not a directory: " +
                                  dir.string());
    }
    // Path fingerprint is enough for fixture wiring; avoid full recursive
    // content digest (can be heavy / fragile on Windows temp dirs).
    const std::string hash = "fixture-dir:" + dir.lexically_normal().string();
    return {
        {"out", fx_typed::make_directory_artifact(dir.string(), hash)}};
  }
};

class KalmanIterOp final : public cgraph::MemoryOperator {
 public:
  KalmanIterOp() {
    op_id_ = "fx.kalman_iter";
    signature_.inputs["in"] = fx_typed::float_list_port("in");
    auto Q = fx_typed::float_port("Q");
    Q.optional = true;
    signature_.inputs["Q"] = std::move(Q);
    auto R = fx_typed::float_port("R");
    R.optional = true;
    signature_.inputs["R"] = std::move(R);
    signature_.outputs["out"] = fx_typed::float_port("out");
    signature_.outputs["final_state"] = fx_typed::float_port("final_state");
    signature_.outputs["final_cov"] = fx_typed::float_port("final_cov");
    signature_.outputs["mean_innov"] = fx_typed::float_port("mean_innov");
    signature_.outputs["innovations"] = fx_typed::float_list_port("innovations");
    cgraph::ParamSpec q;
    q.name = "Q";
    q.dtype = "float";
    q.default_value = 1.0;
    signature_.params["Q"] = q;
    cgraph::ParamSpec r = q;
    r.name = "R";
    r.default_value = 0.5;
    signature_.params["R"] = std::move(r);
    capability_.summary = "Simplified 1D Kalman filter over a float series";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire float_list series + optional Q/R floats.";
    usage_.tune = "F=H=1; scalar Q/R; x0=0, P0=1.";
    usage_.inspect = "out aliases mean_innov; innovations is full series.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in,
      const nlohmann::json& params, const cgraph::ExecContext&) const override {
    const auto zs = fx_typed::require_float_list(
        fx_typed::require_obj(data_in, "in", "fx.kalman_iter"), "fx.kalman_iter",
        "in");
    double Q = 1.0;
    double R = 0.5;
    if (params.is_object()) {
      if (params.contains("Q") && params["Q"].is_number()) {
        Q = params["Q"].get<double>();
      }
      if (params.contains("R") && params["R"].is_number()) {
        R = params["R"].get<double>();
      }
    }
    if (data_in.count("Q") != 0) {
      Q = fx_typed::require_float(data_in.at("Q"), "fx.kalman_iter", "Q");
    }
    if (data_in.count("R") != 0) {
      R = fx_typed::require_float(data_in.at("R"), "fx.kalman_iter", "R");
    }
    const KalmanResult kr = kalman_1d(zs, Q, R);
    return {{"out", fx_typed::make_number_float(kr.mean_innov)},
            {"final_state", fx_typed::make_number_float(kr.final_state)},
            {"final_cov", fx_typed::make_number_float(kr.final_cov)},
            {"mean_innov", fx_typed::make_number_float(kr.mean_innov)},
            {"innovations", fx_typed::make_float_list(kr.innovations)}};
  }
};

class MapChunkFilesOp final : public cgraph::MemoryOperator {
 public:
  MapChunkFilesOp() {
    op_id_ = "fx.map_chunk_files";
    signature_.inputs["items"] = fx_typed::directory_artifact_port("items");
    auto chunks = fx_typed::directory_artifact_port("chunks");
    chunks.optional = true;
    signature_.inputs["chunks"] = std::move(chunks);
    auto files = fx_typed::string_list_port("files");
    files.optional = true;
    signature_.inputs["files"] = std::move(files);
    auto Q = fx_typed::float_port("Q");
    Q.optional = true;
    signature_.inputs["Q"] = std::move(Q);
    auto R = fx_typed::float_port("R");
    R.optional = true;
    signature_.inputs["R"] = std::move(R);
    signature_.outputs["out"] = fx_typed::float_list_port("out");
    signature_.outputs["StatesDir"] = fx_typed::float_list_port("StatesDir");
    signature_.outputs["CovDir"] = fx_typed::float_list_port("CovDir");
    signature_.outputs["InnovList"] = fx_typed::float_list_port("InnovList");
    cgraph::ParamSpec body;
    body.name = "body";
    body.dtype = "string";
    body.default_value = "KalmanIter";
    signature_.params["body"] = std::move(body);
    cgraph::ParamSpec ap;
    ap.name = "allow_partial";
    ap.dtype = "bool";
    ap.default_value = false;
    signature_.params["allow_partial"] = std::move(ap);
    capability_.summary =
        "Map KalmanIter over CSV files from a directory Artifact";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect =
        "Wire items Artifact (or files list[string]) + Q/R; read float lists.";
    usage_.tune = "body=KalmanIter; allow_partial skips bad files as NaN.";
    usage_.inspect =
        "StatesDir/CovDir/InnovList are Value float_list stand-ins (P0).";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in,
      const nlohmann::json& params, const cgraph::ExecContext&) const override {
    const std::string body =
        (params.is_object() && params.contains("body") &&
         params["body"].is_string())
            ? params["body"].get<std::string>()
            : "KalmanIter";
    if (body != "KalmanIter" && body != "kalman_iter") {
      throw std::invalid_argument(
          "fx.map_chunk_files: only body=KalmanIter supported");
    }
    std::vector<fs::path> files;
    if (data_in.count("files") != 0) {
      files = resolve_file_list(data_in.at("files"), "fx.map_chunk_files");
    } else {
      files = resolve_file_list(
          require_alias(data_in, "items", "chunks", "fx.map_chunk_files"),
          "fx.map_chunk_files");
    }
    double Q = 1.0;
    double R = 0.5;
    if (data_in.count("Q") != 0) {
      Q = fx_typed::require_float(data_in.at("Q"), "fx.map_chunk_files", "Q");
    }
    if (data_in.count("R") != 0) {
      R = fx_typed::require_float(data_in.at("R"), "fx.map_chunk_files", "R");
    }
    const bool allow_partial = param_bool(params, "allow_partial", false);

    std::vector<double> states;
    std::vector<double> covs;
    std::vector<double> innovs;
    states.reserve(files.size());
    covs.reserve(files.size());
    innovs.reserve(files.size());
    for (std::size_t i = 0; i < files.size(); ++i) {
      try {
        const auto zs = read_csv_numbers(files[i], "fx.map_chunk_files");
        const KalmanResult kr = kalman_1d(zs, Q, R);
        states.push_back(kr.final_state);
        covs.push_back(kr.final_cov);
        innovs.push_back(kr.mean_innov);
      } catch (const std::exception&) {
        if (!allow_partial) {
          throw;
        }
        const double nan = std::numeric_limits<double>::quiet_NaN();
        states.push_back(nan);
        covs.push_back(nan);
        innovs.push_back(nan);
      }
    }
    return {{"out", fx_typed::make_float_list(innovs)},
            {"StatesDir", fx_typed::make_float_list(states)},
            {"CovDir", fx_typed::make_float_list(covs)},
            {"InnovList", fx_typed::make_float_list(innovs)}};
  }
};

class MergeArtifactDirsOp final : public cgraph::MemoryOperator {
 public:
  MergeArtifactDirsOp() {
    op_id_ = "fx.merge_artifact_dirs";
    signature_.inputs["in"] = fx_typed::float_list_port("in");
    auto items = fx_typed::float_list_port("items");
    items.optional = true;
    signature_.inputs["items"] = std::move(items);
    signature_.outputs["out"] = fx_typed::float_list_port("out");
    capability_.summary =
        "Merge float_list Artifact-dir stand-ins (concat, skip NaN optional)";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire float_list in; read flat out.";
    usage_.tune = "P0: concatenates list values.";
    usage_.inspect = "Value-plane merge for fixture tests (not file copy).";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in,
      const nlohmann::json&, const cgraph::ExecContext&) const override {
    const auto vals = fx_typed::require_float_list(
        require_alias(data_in, "in", "items", "fx.merge_artifact_dirs"),
        "fx.merge_artifact_dirs", "in");
    return {{"out", fx_typed::make_float_list(vals)}};
  }
};

class ListReduceMeanOp final : public cgraph::MemoryOperator {
 public:
  ListReduceMeanOp() {
    op_id_ = "fx.list_reduce_mean";
    signature_.inputs["in"] = fx_typed::float_list_port("in");
    signature_.outputs["out"] = fx_typed::float_port("out");
    capability_.summary = "Mean of a float_list (skips NaN)";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire float_list in; read scalar out.";
    usage_.tune = "None.";
    usage_.inspect = "Empty/all-NaN → error.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in,
      const nlohmann::json&, const cgraph::ExecContext&) const override {
    const auto vals = fx_typed::require_float_list(
        fx_typed::require_obj(data_in, "in", "fx.list_reduce_mean"),
        "fx.list_reduce_mean", "in");
    double sum = 0.0;
    int n = 0;
    for (double v : vals) {
      if (std::isnan(v)) {
        continue;
      }
      sum += v;
      ++n;
    }
    if (n == 0) {
      throw std::invalid_argument("fx.list_reduce_mean: empty list");
    }
    return {{"out", fx_typed::make_number_float(sum / static_cast<double>(n))}};
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_input_artifact_dir() {
  return std::make_shared<InputArtifactDirOp>();
}
std::shared_ptr<cgraph::MemoryOperator> make_kalman_iter() {
  return std::make_shared<KalmanIterOp>();
}
std::shared_ptr<cgraph::MemoryOperator> make_map_chunk_files() {
  return std::make_shared<MapChunkFilesOp>();
}
std::shared_ptr<cgraph::MemoryOperator> make_merge_artifact_dirs() {
  return std::make_shared<MergeArtifactDirsOp>();
}
std::shared_ptr<cgraph::MemoryOperator> make_list_reduce_mean() {
  return std::make_shared<ListReduceMeanOp>();
}

}  // namespace fixture
