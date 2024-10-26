add_library(ws-thread-pool task_manager.c)
target_include_directories(ws-thread-pool PUBLIC ./)

if (ENABLE_TESTS)
  add_subdirectory(tests)
endif()
