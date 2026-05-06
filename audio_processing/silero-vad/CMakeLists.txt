cmake_minimum_required(VERSION 3.10)

set (PROJECT_NAME silero-vad)
set (SRC_FILES ${PROJECT_NAME}.cpp ../../util/wave_reader.cpp)

set (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

project(${PROJECT_NAME} CXX)

include_directories(${INCLUDE_PATH})
link_directories(${LIBRARY_PATH})

add_executable(${PROJECT_NAME} ${SRC_FILES})

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_11)
if(UNIX)
	target_link_libraries(${PROJECT_NAME} ailia "-pthread") # for ailia SDK 1.4.0
else()
	target_link_libraries(${PROJECT_NAME} ailia)
endif()

set (CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})
install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION .)
