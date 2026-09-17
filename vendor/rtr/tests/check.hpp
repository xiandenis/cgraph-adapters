#pragma once
#include <iostream>
#include <string>
#define RTR_CHECK(cond)                                                          \
  do {                                                                           \
    if (!(cond)) {                                                               \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " " << #cond        \
                << "\n";                                                         \
      return 1;                                                                  \
    }                                                                            \
  } while (0)
