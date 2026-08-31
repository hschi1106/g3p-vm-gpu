#include <cassert>
#include <cstddef>
#include <set>
#include <stdexcept>
#include <string>

#include "g3pvm/core/builtin.hpp"
#include "g3pvm/evolution/node_descriptor.hpp"

int main() {
  using namespace g3pvm::evo;

  const auto& descriptors = all_node_descriptors();
  assert(descriptors.size() == k_node_kind_count);

  std::set<std::string> source_names;
  std::set<std::string> serialized_names;
  std::size_t builtin_count = 0;
  std::size_t dependency_count = 0;
  for (std::size_t i = 0; i < descriptors.size(); ++i) {
    const NodeDescriptor& descriptor = descriptors[i];
    assert(static_cast<std::size_t>(descriptor.kind) == i);
    assert(is_known_node_kind(static_cast<int>(i)));
    assert(&node_descriptor(descriptor.kind) == &descriptor);
    assert(!descriptor.source_name.empty());
    assert(!descriptor.serialized_name.empty());
    assert(source_names.insert(std::string(descriptor.source_name)).second);
    assert(serialized_names.insert(std::string(descriptor.serialized_name)).second);

    if (descriptor.is_builtin()) {
      ++builtin_count;
      assert(descriptor.category == NodeCategory::Expression);
      assert(descriptor.builtin_arity == descriptor.prefix_arity);
      g3pvm::BuiltinId id = g3pvm::BuiltinId::Abs;
      assert(g3pvm::builtin_id_from_int(descriptor.builtin_id, id));
      assert(std::string(g3pvm::builtin_name(id)).size() > 0U);
    } else {
      assert(descriptor.builtin_arity == -1);
    }

    if (descriptor.category == NodeCategory::DependencyMarker) {
      ++dependency_count;
      assert(descriptor.prefix_arity == 0);
      assert(descriptor.dependency_arity >= 1);
      assert(descriptor.dependency_family != DependencyFamily::None);
    }
  }

  assert(builtin_count == 27U);
  assert(dependency_count == 12U);
  assert(node_descriptor(NodeKind::PROGRAM).prefix_arity == 1);
  assert(node_descriptor(NodeKind::LINEAR_REC).metadata == NodeMetadataKind::LinearRecBinders);
  assert(node_descriptor(NodeKind::ASGP_DP2D).metadata == NodeMetadataKind::AsgpDp2dSpec);
  assert(node_descriptor(NodeKind::DP1_BACKWARD3).dependency_arity == 3);
  assert(node_descriptor(NodeKind::DP2_DIAGONAL_FORWARD).dependency_arity == 1);
  assert(node_descriptor(NodeKind::CALL_INDEX).builtin_id == static_cast<int>(g3pvm::BuiltinId::Index));

  assert(!is_known_node_kind(-1));
  assert(!is_known_node_kind(static_cast<int>(NodeKind::COUNT)));
  bool rejected = false;
  try {
    (void)node_descriptor(NodeKind::COUNT);
  } catch (const std::out_of_range&) {
    rejected = true;
  }
  assert(rejected);

  return 0;
}
