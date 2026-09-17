#pragma once

#include "cgraph/artifact.hpp"
#include "cgraph/data_helpers.hpp"

#include <map>
#include <string>

namespace fx {

inline nlohmann::json as_json(const cgraph::DataObject& obj) {
  using cgraph::Payload;
  switch (obj.payload.kind()) {
    case Payload::Kind::Bool:
    case Payload::Kind::Int:
    case Payload::Kind::Float:
    case Payload::Kind::String:
      return cgraph::payload_to_param_json(obj.payload);
    case Payload::Kind::Untyped: {
      const auto& b = obj.payload.as_untyped();
      if (b.empty()) {
        return nullptr;
      }
      const std::string text(b.begin(), b.end());
      auto parsed = nlohmann::json::parse(text, nullptr, false);
      if (parsed.is_discarded()) {
        return nlohmann::json(text);
      }
      return parsed;
    }
    case Payload::Kind::DataRef: {
      const auto& ref = obj.payload.as_data_ref();
      return nlohmann::json{{"path", ref.uri},
                            {"digest", ref.content_hash},
                            {"dtype", ref.is_dir ? "dir" : "file"},
                            {"is_dir", ref.is_dir}};
    }
    case Payload::Kind::Matrix4x4: {
      nlohmann::json rows = nlohmann::json::array();
      const auto& m = obj.payload.as_matrix4x4();
      for (int r = 0; r < 4; ++r) {
        nlohmann::json row = nlohmann::json::array();
        for (int c = 0; c < 4; ++c) {
          row.push_back(m[static_cast<std::size_t>(r * 4 + c)]);
        }
        rows.push_back(std::move(row));
      }
      return rows;
    }
    default:
      return nlohmann::json::object();
  }
}

inline std::map<std::string, nlohmann::json> unwrap(
    const std::map<std::string, cgraph::DataObject>& inputs) {
  std::map<std::string, nlohmann::json> values;
  for (const auto& [k, v] : inputs) {
    values[k] = as_json(v);
  }
  return values;
}

inline cgraph::DataObject from_json(const nlohmann::json& value,
                                    const cgraph::PortSpec& spec) {
  if (spec.kind == cgraph::PortKind::Artifact && value.is_object()) {
    const cgraph::Artifact art = cgraph::artifact_from_json(value);
    cgraph::DataRef ref;
    ref.uri = art.location.string();
    ref.content_hash = art.content_hash;
    ref.is_dir = art.is_dir;
    return cgraph::make_artifact_object(spec.type.spec().empty()
                                            ? cgraph::type_ids::untyped()
                                            : spec.type,
                                        spec.semantic.meaning.empty()
                                            ? cgraph::SemanticSpec::of("file")
                                            : spec.semantic,
                                        std::move(ref));
  }
  cgraph::TypeId type =
      spec.type.spec().empty() ? cgraph::type_ids::untyped() : spec.type;
  cgraph::SemanticSpec sem = spec.semantic;
  if (sem.meaning.empty() || sem.meaning == "none") {
    sem = spec.kind == cgraph::PortKind::Artifact
              ? cgraph::SemanticSpec::of("file")
              : cgraph::SemanticSpec::of("document");
  }
  return cgraph::data_from_json_literal(std::move(type), std::move(sem), value);
}

struct JsonView {
  std::map<std::string, nlohmann::json> values;
  explicit JsonView(const std::map<std::string, cgraph::DataObject>& inputs) {
    for (const auto& [k, v] : inputs) {
      values[k] = as_json(v);
    }
  }
  std::size_t count(const std::string& k) const { return values.count(k); }
  auto find(const std::string& k) const { return values.find(k); }
  auto end() const { return values.end(); }
  const nlohmann::json& at(const std::string& k) const { return values.at(k); }
  nlohmann::json& operator[](const std::string& k) { return values[k]; }
};

inline std::map<std::string, cgraph::DataObject> wrap(
    const cgraph::Signature& signature,
    const std::map<std::string, nlohmann::json>& outputs) {
  std::map<std::string, cgraph::DataObject> out;
  for (const auto& [name, value] : outputs) {
    const auto it = signature.outputs.find(name);
    if (it == signature.outputs.end()) {
      out.emplace(name, cgraph::data_from_json_literal(
                            cgraph::type_ids::untyped(),
                            cgraph::SemanticSpec::of("document"), value));
    } else {
      out.emplace(name, from_json(value, it->second));
    }
  }
  return out;
}

}  // namespace fx
