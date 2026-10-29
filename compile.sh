#!/usr/bin/env bash
# 在仓库 build/ 中配置并编译；工作目录不影响源码或产物位置。
set -euo pipefail

project_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -eq 1 && ( $1 == --help || $1 == -h ) ]]; then
  printf '用法：%s\n默认编译到仓库 build/，产物为 build/apps/opfusion。\n' "$0"
  printf '可选环境变量：OPFUSION_BUILD_DIR（构建目录）、CMAKE_BUILD_PARALLEL_LEVEL（并行数，默认 4）。\n'
  exit 0
fi
if [[ $# -ne 0 ]]; then
  printf '错误：compile.sh 不接受位置参数，使用 --help 查看用法。\n' >&2
  exit 2
fi
if ! command -v cmake >/dev/null 2>&1; then
  printf '错误：未找到 cmake，请安装 CMake 3.20 或更高版本及 C++17 编译器。\n' >&2
  exit 1
fi

build_dir="${OPFUSION_BUILD_DIR:-$project_root/build}"
if [[ $build_dir != /* ]]; then
  build_dir="$project_root/$build_dir"
fi
jobs="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
if [[ ! $jobs =~ ^[1-9][0-9]*$ ]]; then
  printf '错误：CMAKE_BUILD_PARALLEL_LEVEL 必须为正整数。\n' >&2
  exit 2
fi

# 已有目录沿用缓存生成器；新目录优先 Ninja，否则使用 CMake 默认生成器。
generator=()
if [[ ! -f $build_dir/CMakeCache.txt ]] && command -v ninja >/dev/null 2>&1; then
  generator=(-G Ninja)
fi
printf '准备锁版原生依赖（首次需要网络，后续复用构建缓存）...\n' >&2
cmake -S "$project_root" -B "$build_dir" "${generator[@]}" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DopfusionEnableOperatorPlugins=ON
cmake --build "$build_dir" --config Release --parallel "$jobs"

binary="$build_dir/apps/opfusion"
if [[ ! -x $binary && -x $build_dir/apps/Release/opfusion ]]; then
  binary="$build_dir/apps/Release/opfusion"
fi
printf '\n编译完成：%s\n' "$binary"
printf '运行模型：%s/run.sh 模型.onnx、文件.json 或目录\n' "$project_root"
printf '运行测试：ctest --test-dir "%s" -C Release --output-on-failure\n' "$build_dir"
