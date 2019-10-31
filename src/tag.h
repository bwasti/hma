#pragma once

#include <cstddef>
#include <string>

// Rough string interning solution for device/engine tags
size_t getTag(std::string tag_name);
