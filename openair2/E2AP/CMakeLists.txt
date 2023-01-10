
set(E2AP_LIBRARY "STATIC" CACHE STRING "Static or dynamic library")
set_property(CACHE E2AP_LIBRARY PROPERTY STRINGS "STATIC" "DYNAMIC")
message(STATUS "Selected LIBRARY TYPE: ${E2AP_LIBRARY}")

if(E2AP_LIBRARY STREQUAL "STATIC")
  add_compile_options("-W;-Wall;-Wextra;-g;-Wno-unused-result;")
elseif(E2AP_LIBRARY STREQUAL "DYNAMIC")
  # -fPIC flag
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  add_compile_options("-W;-Wall;-Wextra;-g")
else()
  message(FATAL_ERROR "Unknown building type. Either choose a static or a dynamic library")
endif()


set(E2AP_ENCODING "E2AP_ASN" CACHE STRING "The E2AP encoding to use")
set_property(CACHE E2AP_ENCODING PROPERTY STRINGS "E2AP_ASN" "E2AP_FLATBUFFERS")
message(STATUS "Selected E2AP_ENCODING: ${E2AP_ENCODING}")

#######
## Service Models 
#######

# KPM service Model encoding definitions
set(SM_ENCODING_KPM "E2SM_KPM_ASN" CACHE STRING "The KPM SM encoding to use")
set_property(CACHE SM_ENCODING_KPM PROPERTY STRINGS "E2SM_KPM_PLAIN" "E2SM_KPM_ASN" "E2SM_KPM_FLATBUFFERS")
message(STATUS "Selected SM_ENCODING_KPM: ${SM_ENCODING_KPM}")

add_subdirectory(agent)
add_subdirectory(lib)
add_subdirectory(sm)
add_subdirectory(test)
