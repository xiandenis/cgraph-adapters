#include "cgraph/artifact.hpp"
#include "cgraph/ops.hpp"
#include "math_nd_util.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
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

std::vector<double> parse_numeric_series(const nlohmann::json& j,
                                         const char* op) {
  if (j.is_array()) {
    std::vector<double> xs;
    xs.reserve(j.size());
    for (const auto& v : j) {
      if (!v.is_number()) {
        throw std::invalid_argument(std::string(op) +
                                    ": series entries must be numbers");
      }
      xs.push_back(v.get<double>());
    }
    return xs;
  }
  if (j.is_object() && j.contains("data") && j["data"].is_array()) {
    return parse_numeric_series(j["data"], op);
  }
  throw std::invalid_argument(std::string(op) +
                              ": series must be a number array");
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
    // Accept "z", "t,z", or whitespace-separated numbers; take last token as z.
    std::stringstream ss(line);
    std::string tok;
    std::string last;
    while (std::getline(ss, tok, ',')) {
      // also split spaces inside token
      std::stringstream ts(tok);
      std::string w;
      while (ts >> w) {
        last = w;
      }
    }
    if (last.empty() || last == "z" || last == "meas" || last == "value") {
      continue;  // header
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
    const double x_pred = x;  // F=1
    const double P_pred = P + Q;
    const double S = P_pred + R;
    const double K = P_pred / S;
    const double innov = z - x_pred;  // H=1
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

nlohmann::json kalman_result_json(const KalmanResult& r) {
  return nlohmann::json{{"final_state", r.final_state},
                        {"final_cov", r.final_cov},
                        {"innovations", r.innovations},
                        {"mean_innov", r.mean_innov}};
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

nlohmann::json files_to_json(const std::vector<fs::path>& files) {
  nlohmann::json arr = nlohmann::json::array();
  for (const auto& p : files) {
    arr.push_back(p.string());
  }
  return arr;
}

// Resolve items input: array of paths, dir manifest object, or dir artifact.
std::vector<fs::path> resolve_file_list(const nlohmann::json& items,
                                        const char* op) {
  std::vector<fs::path> out;
  if (cgraph::is_artifact_handle(items)) {
    const cgraph::Artifact art = cgraph::artifact_from_json(items);
    if (!art.is_dir) {
      throw std::invalid_argument(std::string(op) +
                                  ": artifact must be a directory");
    }
    return list_csv_files(art.path);
  }
  if (items.is_object()) {
    if (items.contains("files") && items["files"].is_array()) {
      for (const auto& f : items["files"]) {
        if (!f.is_string()) {
          throw std::invalid_argument(std::string(op) +
                                      ": files entries must be strings");
        }
        out.emplace_back(f.get<std::string>());
      }
      return out;
    }
    if (items.contains("path") && items["path"].is_string()) {
      return list_csv_files(items["path"].get<std::string>());
    }
  }
  if (items.is_array()) {
    for (const auto& f : items) {
      if (!f.is_string()) {
        throw std::invalid_argument(std::string(op) +
                                    ": file list entries must be strings");
      }
      out.emplace_back(f.get<std::string>());
    }
    return out;
  }
  if (items.is_string()) {
    return list_csv_files(items.get<std::string>());
  }
  throw std::invalid_argument(std::string(op) +
                              ": cannot resolve file list from items");
}

class InputArtifactDirOp final : public cgraph::MemoryOperator {
 public:
  InputArtifactDirOp() {
    op_id_ = "fx.input_artifact_dir";
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, "json");
    cgraph::ParamSpec path;
    path.name = "path";
    path.dtype = "string";
    path.invalidate = true;
    path.bindable = false;
    signature_.params["path"] = std::move(path);
    capability_.summary =
        "List CSV files under a fixture directory as a Value manifest";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "params.path → out={path,files:[...]}.";
    usage_.tune = "Point at a tiny fixture dir (2–3 csv).";
    usage_.inspect =
        "P0 returns JSON path list (not Artifact) for Runtime.execute tests.";
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>&,
      const nlohmann::json& params, const cgraph::ExecContext&) const override {
    const fs::path dir = require_path_param(params, "fx.input_artifact_dir");
    if (!fs::is_directory(dir)) {
      throw std::invalid_argument("fx.input_artifact_dir: not a directory: " +
                                  dir.string());
    }
    const auto files = list_csv_files(dir);
    nlohmann::json manifest = {{"path", dir.string()},
                               {"files", files_to_json(files)}};
    return {{"out", std::move(manifest)}};
  }
};

class KalmanIterOp final : public cgraph::MemoryOperator {
 public:
  KalmanIterOp() {
    op_id_ = "fx.kalman_iter";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, "json");
    auto Q = cgraph::make_port("Q", cgraph::PortKind::Value, "json");
    Q.optional = true;
    signature_.inputs["Q"] = std::move(Q);
    auto R = cgraph::make_port("R", cgraph::PortKind::Value, "json");
    R.optional = true;
    signature_.inputs["R"] = std::move(R);
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, "json");
    signature_.outputs["final_state"] =
        cgraph::make_port("final_state", cgraph::PortKind::Value, "json");
    signature_.outputs["final_cov"] =
        cgraph::make_port("final_cov", cgraph::PortKind::Value, "json");
    signature_.outputs["mean_innov"] =
        cgraph::make_port("mean_innov", cgraph::PortKind::Value, "json");
    cgraph::ParamSpec q;
    q.name = "Q";
    q.dtype = "float";
    q.default_value = 1.0;
    signature_.params["Q"] = q;
    cgraph::ParamSpec r = q;
    r.name = "R";
    r.default_value = 0.5;
    signature_.params["R"] = std::move(r);
    capability_.summary = "Simplified 1D Kalman filter over a numeric series";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire series (array or csv path string) + optional Q/R.";
    usage_.tune = "F=H=1; scalar Q/R; x0=0, P0=1.";
    usage_.inspect = "out has final_state, final_cov, innovations, mean_innov.";
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs,
      const nlohmann::json& params, const cgraph::ExecContext&) const override {
    const nlohmann::json& in =
        math_nd::require_input(inputs, "in", "fx.kalman_iter");
    std::vector<double> zs;
    if (in.is_string()) {
      zs = read_csv_numbers(in.get<std::string>(), "fx.kalman_iter");
    } else {
      zs = parse_numeric_series(in, "fx.kalman_iter");
    }
    double Q = math_nd::param_number(params, "Q", 1.0);
    double R = math_nd::param_number(params, "R", 0.5);
    const auto qit = inputs.find("Q");
    if (qit != inputs.end()) {
      Q = math_nd::require_number(qit->second, "fx.kalman_iter", "Q");
    }
    const auto rit = inputs.find("R");
    if (rit != inputs.end()) {
      R = math_nd::require_number(rit->second, "fx.kalman_iter", "R");
    }
    const KalmanResult kr = kalman_1d(zs, Q, R);
    return {{"out", kalman_result_json(kr)},
            {"final_state", kr.final_state},
            {"final_cov", kr.final_cov},
            {"mean_innov", kr.mean_innov}};
  }
};

class MapChunkFilesOp final : public cgraph::MemoryOperator {
 public:
  MapChunkFilesOp() {
    op_id_ = "fx.map_chunk_files";
    signature_.inputs["items"] =
        cgraph::make_port("items", cgraph::PortKind::Value, "json");
    auto chunks = cgraph::make_port("chunks", cgraph::PortKind::Value, "json");
    chunks.optional = true;
    signature_.inputs["chunks"] = std::move(chunks);
    auto Q = cgraph::make_port("Q", cgraph::PortKind::Value, "json");
    Q.optional = true;
    signature_.inputs["Q"] = std::move(Q);
    auto R = cgraph::make_port("R", cgraph::PortKind::Value, "json");
    R.optional = true;
    signature_.inputs["R"] = std::move(R);
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, "json");
    signature_.outputs["StatesDir"] =
        cgraph::make_port("StatesDir", cgraph::PortKind::Value, "json");
    signature_.outputs["CovDir"] =
        cgraph::make_port("CovDir", cgraph::PortKind::Value, "json");
    signature_.outputs["InnovList"] =
        cgraph::make_port("InnovList", cgraph::PortKind::Value, "json");
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
        "Map KalmanIter over CSV files listed by InputArtifactDir";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire items/chunks (+ Q,R); read StatesDir/CovDir/InnovList.";
    usage_.tune = "body=KalmanIter (default); allow_partial for bad files.";
    usage_.inspect = "Value lists stand in for artifact dirs in P0 tests.";
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs,
      const nlohmann::json& params, const cgraph::ExecContext&) const override {
    const nlohmann::json& items = math_nd::require_input_alias(
        inputs, "items", "chunks", "fx.map_chunk_files");
    const std::string body =
        (params.is_object() && params.contains("body") &&
         params["body"].is_string())
            ? params["body"].get<std::string>()
            : "KalmanIter";
    if (body != "KalmanIter" && body != "kalman_iter") {
      throw std::invalid_argument(
          "fx.map_chunk_files: only body=KalmanIter supported");
    }
    double Q = 1.0;
    double R = 0.5;
    const auto qit = inputs.find("Q");
    if (qit != inputs.end()) {
      Q = math_nd::require_number(qit->second, "fx.map_chunk_files", "Q");
    }
    const auto rit = inputs.find("R");
    if (rit != inputs.end()) {
      R = math_nd::require_number(rit->second, "fx.map_chunk_files", "R");
    }
    const bool allow_partial = param_bool(params, "allow_partial", false);
    const auto files = resolve_file_list(items, "fx.map_chunk_files");

    nlohmann::json states = nlohmann::json::array();
    nlohmann::json covs = nlohmann::json::array();
    nlohmann::json innovs = nlohmann::json::array();
    nlohmann::json results = nlohmann::json::array();
    for (std::size_t i = 0; i < files.size(); ++i) {
      try {
        const auto zs = read_csv_numbers(files[i], "fx.map_chunk_files");
        const KalmanResult kr = kalman_1d(zs, Q, R);
        nlohmann::json one = kalman_result_json(kr);
        one["file"] = files[i].string();
        results.push_back(one);
        states.push_back(kr.final_state);
        covs.push_back(kr.final_cov);
        innovs.push_back(kr.mean_innov);
      } catch (const std::exception& ex) {
        if (!allow_partial) {
          throw;
        }
        results.push_back(
            nlohmann::json{{"error", ex.what()}, {"index", i}});
        states.push_back(nullptr);
        covs.push_back(nullptr);
        innovs.push_back(nullptr);
      }
    }
    return {{"out", std::move(results)},
            {"StatesDir", std::move(states)},
            {"CovDir", std::move(covs)},
            {"InnovList", std::move(innovs)}};
  }
};

class MergeArtifactDirsOp final : public cgraph::MemoryOperator {
 public:
  MergeArtifactDirsOp() {
    op_id_ = "fx.merge_artifact_dirs";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, "json");
    auto items = cgraph::make_port("items", cgraph::PortKind::Value, "json");
    items.optional = true;
    signature_.inputs["items"] = std::move(items);
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, "json");
    capability_.summary =
        "Merge file-list / value-list artifact dir stand-ins into one list";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire in (list or nested lists); read flat out.";
    usage_.tune = "P0: concatenates arrays; skips nulls.";
    usage_.inspect = "Does not copy files; Value-plane merge for tests.";
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs,
      const nlohmann::json&, const cgraph::ExecContext&) const override {
    const nlohmann::json& in =
        math_nd::require_input_alias(inputs, "in", "items",
                                     "fx.merge_artifact_dirs");
    nlohmann::json out = nlohmann::json::array();
    auto append = [&](const nlohmann::json& v) {
      if (v.is_null()) {
        return;
      }
      if (v.is_array()) {
        for (const auto& x : v) {
          if (!x.is_null()) {
            out.push_back(x);
          }
        }
        return;
      }
      if (v.is_object() && v.contains("files") && v["files"].is_array()) {
        for (const auto& x : v["files"]) {
          out.push_back(x);
        }
        return;
      }
      out.push_back(v);
    };
    append(in);
    return {{"out", std::move(out)}};
  }
};

class ListReduceMeanOp final : public cgraph::MemoryOperator {
 public:
  ListReduceMeanOp() {
    op_id_ = "fx.list_reduce_mean";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, "json");
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, "json");
    capability_.summary = "Mean of a numeric JSON list (skips nulls)";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire numeric list in; read scalar out.";
    usage_.tune = "None.";
    usage_.inspect = "Empty/all-null → error.";
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs,
      const nlohmann::json&, const cgraph::ExecContext&) const override {
    const nlohmann::json& in =
        math_nd::require_input(inputs, "in", "fx.list_reduce_mean");
    if (!in.is_array()) {
      throw std::invalid_argument(
          "fx.list_reduce_mean: input must be a JSON array");
    }
    double sum = 0.0;
    int n = 0;
    for (const auto& v : in) {
      if (v.is_null()) {
        continue;
      }
      if (!v.is_number()) {
        throw std::invalid_argument(
            "fx.list_reduce_mean: non-numeric list entry");
      }
      sum += v.get<double>();
      ++n;
    }
    if (n == 0) {
      throw std::invalid_argument("fx.list_reduce_mean: empty list");
    }
    return {{"out", sum / static_cast<double>(n)}};
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
