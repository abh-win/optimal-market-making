#pragma once
// Shared CLI helpers for the tools.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

namespace mm::cli {

inline std::map<std::string, std::string> args(int argc, char** argv) {
  std::map<std::string, std::string> m;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("--", 0) != 0) continue;
    a = a.substr(2);
    if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) m[a] = argv[++i];
    else m[a] = "1";
  }
  return m;
}

inline std::string get(const std::map<std::string, std::string>& m, const std::string& k, const std::string& d) {
  auto it = m.find(k);
  return it == m.end() ? d : it->second;
}

// Minimal reader for the flat JSON meta file written by prepare.py.
inline double jsonNumber(const std::string& path, const std::string& key) {
  std::ifstream in(path);
  std::stringstream ss;
  ss << in.rdbuf();
  std::string s = ss.str();
  auto p = s.find("\"" + key + "\"");
  if (p == std::string::npos) return 0;
  p = s.find(':', p);
  return std::strtod(s.c_str() + p + 1, nullptr);
}

}  // namespace mm::cli
