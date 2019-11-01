#include "tag.h"
#include <unordered_map>

std::unordered_map<std::string, size_t> &getTagMap() {
  static std::unordered_map<std::string, size_t> tags_;
  return tags_;
}

size_t getTag(std::string tag_name) {
  if (getTagMap().find(tag_name) == getTagMap().end()) {
    getTagMap()[tag_name] = getTagMap().size();
  }
  return getTagMap().at(tag_name);
}

std::string getTagName(size_t tag) {
  for (const auto &pair : getTagMap()) {
    if (pair.second == tag) {
      return pair.first;
    }
  }
  return "";
}
