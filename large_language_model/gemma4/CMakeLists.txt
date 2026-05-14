cmake_minimum_required(VERSION 3.10)

project(gemma4 CXX)

set (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

include_directories(${INCLUDE_PATH})
link_directories(${LIBRARY_PATH})

add_executable(gemma4 gemma4.cpp)
target_compile_features(gemma4 PRIVATE cxx_std_11)
target_link_libraries(gemma4 ailia_llm)

add_executable(gemma4-multimodal gemma4-multimodal.cpp)
target_compile_features(gemma4-multimodal PRIVATE cxx_std_11)
target_link_libraries(gemma4-multimodal ailia_llm)

set (CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})
install(TARGETS gemma4 gemma4-multimodal RUNTIME DESTINATION .)
