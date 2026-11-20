# Please run this script in the root directory of the project.

find . -name "*.cpp" -o -name "*.h" -o -name "*.cu" -o -name "*.cuh" | xargs clang-format -i -style=file