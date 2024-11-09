CPMAddPackage("gh:google/marl#690889fbb816a409b4fcd4bdf60e32a4deebeb08")

add_library(marl_if marl_if.cpp)
target_link_libraries(marl_if PRIVATE marl thread-pool)
target_include_directories(marl_if PUBLIC ./)
