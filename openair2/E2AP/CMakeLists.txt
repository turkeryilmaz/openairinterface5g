
set(BUILDING_LIBRARY "STATIC" CACHE STRING "Static or dynamic library")
set_property(CACHE BUILDING_LIBRARY PROPERTY STRINGS "STATIC" "DYNAMIC")
message(STATUS "Selected LIBRARY TYPE: ${BUILDING_LIBRARY}")


set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(BUILDING_LIBRARY STREQUAL "STATIC")
  add_compile_options("-W;-Wall;-Wextra;-g;-Wno-unused-result;")
elseif(BUILDING_LIBRARY STREQUAL "DYNAMIC")
  # -fPIC flag
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  add_compile_options("-W;-Wall;-Wextra;-g")
else()
  message(FATAL_ERROR "Unknown building type. Either choose a static or a dynamic library")
endif()


set(E2AP_ENCODING "ASN" CACHE STRING "The E2AP encoding to use")
set_property(CACHE E2AP_ENCODING PROPERTY STRINGS "ASN" "FLATBUFFERS")
message(STATUS "Selected E2AP_ENCODING: ${E2AP_ENCODING}")


#######
## Service Models 
#######

add_definitions(-DSERVICE_MODEL_DIR_PATH="${SM_DIR_PATH}/")

# KPM service Model encoding definitions
set(SM_ENCODING_KPM "ASN" CACHE STRING "The KPM SM encoding to use")
set_property(CACHE SM_ENCODING_KPM PROPERTY STRINGS "PLAIN" "ASN" "FLATBUFFERS")
message(STATUS "Selected KPM SM_ENCODING: ${SM_ENCODING_KPM}")

# RC service Model encoding definitions
set(SM_ENCODING_RC "ASN" CACHE STRING "The RC SM encoding to use")
set_property(CACHE SM_ENCODING_RC PROPERTY STRINGS "PLAIN" "ASN" "FLATBUFFERS")
message(STATUS "Selected RC SM_ENCODING: ${SM_ENCODING_RC}")

# MAC Service Model
set(SM_ENCODING_MAC "PLAIN" CACHE STRING "The MAC SM encoding to use")
set_property(CACHE SM_ENCODING_MAC PROPERTY STRINGS "PLAIN" "ASN" "FLATBUFFERS")
message(STATUS "Selected MAC SM_ENCODING: ${SM_ENCODING_MAC}")

# RLC Service Model
set(SM_ENCODING_RLC "PLAIN" CACHE STRING "The RLC SM encoding to use")
set_property(CACHE SM_ENCODING_RLC PROPERTY STRINGS "PLAIN" "ASN" "FLATBUFFERS")
message(STATUS "Selected RLC SM_ENCODING: ${SM_ENCODING_RLC}")

# PDCP Service Model
set(SM_ENCODING_PDCP "PLAIN" CACHE STRING "The PDCP SM encoding to use")
set_property(CACHE SM_ENCODING_PDCP PROPERTY STRINGS "PLAIN" "ASN" "FLATBUFFERS")
message(STATUS "Selected PDCP SM_ENCODING: ${SM_ENCODING_PDCP}")

# SLICE Service Model
set(SM_ENCODING_SLICE "PLAIN" CACHE STRING "The SLICE SM encoding to use")
set_property(CACHE SM_ENCODING_SLICE PROPERTY STRINGS "PLAIN" "ASN" "FLATBUFFERS")
message(STATUS "Selected SLICE SM_ENCODING: ${SM_ENCODING_SLICE}")

# GTP Service Model
set(SM_ENCODING_GTP "PLAIN" CACHE STRING "The GTP SM encoding to use")
set_property(CACHE SM_ENCODING_GTP PROPERTY STRINGS "PLAIN")
message(STATUS "Selected GTP SM_ENCODING: ${SM_ENCODING_GTP}")

########
### Flatbuffer 
########
set(FlatCC_INCLUDE_DIR "" CACHE STRING "The Flatbuffers include directory")
set(FlatCC_LIB_DIR "" CACHE STRING "The Flatbuffers lib directory")

if(E2AP_ENCODING STREQUAL "FLATBUFFERS")
  find_library(FlatCC
    NAMES flatccrt_d
    HINTS ${FlatCC_LIB_DIR} 
    #"~/workspace/flatcc/lib/"
    )
endif()

###
# set install options
###
set(NEAR_RIC_INSTALL "FALSE" CACHE STRING "NEAR real time RIC install selection")
set_property(CACHE NEAR_RIC_INSTALL PROPERTY STRINGS "TRUE" "FALSE")
set(EMU_AGENT_INSTALL "FALSE" CACHE STRING "emulator of E2 agent")
set_property(CACHE EMU_AGENT_INSTALL PROPERTY STRINGS "TRUE" "FALSE")
set(XAPP_C_INSTALL "FALSE" CACHE STRING "XAPP in C install selection")
set_property(CACHE XAPP_C_INSTALL PROPERTY STRINGS "TRUE" "FALSE")
set(UNIT_TEST "TRUE" CACHE STRING "build test cases")
set_property(CACHE UNIT_TEST PROPERTY STRINGS "TRUE" "FALSE")



include_directories(flexric/src)
add_subdirectory(flexric/src/agent)
add_subdirectory(flexric/src/lib)
add_subdirectory(flexric/src/sm)
add_subdirectory(flexric/src/util)
add_subdirectory(RAN_FUNCTION)


