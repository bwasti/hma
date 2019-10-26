#pragma once

#include <string>
#include <stdexcept>
#include <sstream>

class hma_exception : public std::runtime_error {
	std::string msg;
	public:
	hma_exception(const std::string &arg, const char *file, int line) : std::runtime_error(arg) {
		std::ostringstream o;
		o << file << ":" << line << ": " << arg;
		msg = o.str();
	}
	~hma_exception() throw() {}
	const char *what() const throw() {
		return msg.c_str();
	}
};

#define THROW_LINE(arg) throw hma_exception(arg, __FILE__, __LINE__);
#define HMA_ENFORCE(cond) \
	if (!(cond)) { \
		THROW_LINE(#cond " failed."); \
	}
