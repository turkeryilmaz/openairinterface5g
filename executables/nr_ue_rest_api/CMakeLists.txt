add_library(nr_ue_rest_api nr_ue_rest_api.cpp)
target_include_directories(nr_ue_rest_api PUBLIC ./)
target_link_libraries(nr_ue_rest_api PUBLIC Crow::Crow nr_ue_phy_rest_api_module)
