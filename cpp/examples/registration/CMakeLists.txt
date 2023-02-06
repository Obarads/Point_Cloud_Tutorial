# CMakeLists.txt
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(ICP)
add_executable(main main.cpp)

# Include self libs
include_directories(include)

# Find and include PCL
find_package(PCL 1.3 REQUIRED COMPONENTS common io)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
target_link_libraries(main ${PCL_LIBRARIES})

# target_include_directories(main
#     PRIVATE ${PCL_INCLUDE_DIRS}
#     )
# target_link_libraries(main ${PCL_LIBRARIES})
# target_compile_definitions(main
#     PRIVATE ${PCL_DEFINITIONS}
#     )
