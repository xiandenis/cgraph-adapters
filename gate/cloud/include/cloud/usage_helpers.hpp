#pragma once

#include "cgraph/graph.hpp"
#include "cgraph/node.hpp"
#include "cgraph/ops.hpp"
#include "cgraph/port.hpp"

#include <memory>
#include <string>
#include <utility>

namespace cloud::usage {

inline cgraph::PortSpec cloud_in_xyz() {
  auto p = cgraph::make_port("in", cgraph::PortKind::Value, "cloud");
  p.schema = {{"fields", nlohmann::json::array({"xyz"})}};
  return p;
}

inline cgraph::PortSpec cloud_out_xyz() {
  auto p = cgraph::make_port("out", cgraph::PortKind::Value, "cloud");
  p.schema = {{"fields", nlohmann::json::array({"xyz"})}};
  return p;
}

inline cgraph::PortSpec cloud_out_xyz_normal() {
  auto p = cgraph::make_port("out", cgraph::PortKind::Value, "cloud");
  p.schema = {{"fields", nlohmann::json::array({"xyz", "normal"})}};
  return p;
}

inline void set_tune(cgraph::Usage& usage, std::initializer_list<cgraph::UsageTuneEntry> entries) {
  usage.tune_entries.assign(entries);
}

inline std::shared_ptr<cgraph::Graph> make_min_graph_node(
    std::string node_id, std::string op_id, nlohmann::json params,
    std::map<std::string, cgraph::PortSpec> inputs,
    std::map<std::string, cgraph::PortSpec> outputs) {
  auto g = std::make_shared<cgraph::Graph>();
  cgraph::Node n;
  n.id = std::move(node_id);
  n.op_id = std::move(op_id);
  n.params = std::move(params);
  n.inputs = std::move(inputs);
  n.outputs = std::move(outputs);
  g->add_node(std::move(n));
  return g;
}

}  // namespace cloud::usage
