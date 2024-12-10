CPMAddPackage("gh:chriskohlhoff/asio#asio-1-18-1@1.18.1")
if (asio_ADDED)
  # asio has no CMake support, so we create our own target
  add_library(asio INTERFACE)
  add_library(asio::asio ALIAS asio)
  target_include_directories(asio SYSTEM INTERFACE ${asio_SOURCE_DIR}/asio/include)
  target_compile_definitions(asio INTERFACE ASIO_STANDALONE)
endif()

CPMAddPackage(
  NAME Crow
  GIT_REPOSITORY https://github.com/CrowCpp/Crow.git
  GIT_TAG v1.2.0
  OPTIONS
    "CROW_INSTALL OFF"
)

add_library(nr_ue_rest_api MODULE nr_ue_rest_api.cpp)
target_include_directories(nr_ue_rest_api PUBLIC ./)
target_link_libraries(nr_ue_rest_api PUBLIC Crow::Crow nr_ue_phy_rest_api_module shlib_loader nlohmann_json::nlohmann_json)
set_target_properties(nr_ue_rest_api PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
