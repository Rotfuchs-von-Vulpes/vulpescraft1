cmake_minimum_required(VERSION 3.0)

project(vulpes)

add_subdirectory(libs/glad)
add_subdirectory(libs/cimgui/backend_test/example_glfw_opengl3)

add_executable(vulpes src/main.c)
target_link_libraries(vulpes cimgui glad glfw cimgui)
target_include_directories(
	vulpes PRIVATE
	libs/cimgui
	libs/cimgui/generator/output
	libs/glad/include
	libs/glfw/include
	libs/cglm/include
	libs/stb)