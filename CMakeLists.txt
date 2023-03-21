cmake_minimum_required(VERSION 3.0)

set(PROJECT vulpes)
set(lib OpenGL32)

project(${PROJECT})

# GLFW
add_subdirectory(lib/glfw)
set(GLFW_INCLUDE lib/glfw/include)
list(APPEND lib glfw)

# GLAD
add_library(glad lib/glad/src/glad.c)
set(GLAD_INCLUDE lib/glad/include)
target_include_directories(glad PRIVATE ${GLAD_INCLUDE})
list(APPEND lib glad)

# CGLM
set(CGLM_INCLUDE lib/cglm/include)

# STB
set(STB_INCLUDE lib/stb)

# program
file(GLOB SRC src/*.c)
add_executable(${PROJECT} ${SRC})
target_link_libraries(${PROJECT} ${lib})
target_include_directories(
	${PROJECT} PRIVATE
	${CGLM_INCLUDE}
	${GLAD_INCLUDE}
	${GLFW_INCLUDE}
	${STB_INCLUDE})