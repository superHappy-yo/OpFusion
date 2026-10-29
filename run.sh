#!/usr/bin/env bash
# 逐文件调用现有 inspect；JSONL 写 stdout，进度、错误和汇总写 stderr。
set -euo pipefail

usage() {
  printf '用法：%s [--context scenario.json] 模型.onnx|文件.json [更多模型.onnx|文件.json ...]\n' "$0"
  printf '      %s 目录 [更多文件或目录 ...]\n' "$0"
  printf '目录递归扫描所有 .onnx 和 .json；也支持由 shell 展开的 *.json。\n'
  printf '每个成功文件输出一行 JSON；当前只进行模型导入与校验。\n'
}

if [[ $# -eq 1 && ( $1 == --help || $1 == -h ) ]]; then
  usage
  exit 0
fi
if [[ $# -eq 0 ]]; then
  usage >&2
  exit 2
fi

context=""
inputs=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --context)
      if [[ $# -lt 2 || -z ${2:-} || -n $context ]]; then
        printf '错误：--context 需要且只允许指定一个文件。\n' >&2
        exit 2
      fi
      context=$2
      shift 2
      ;;
    --) shift; inputs+=("$@"); break ;;
    *) inputs+=("$1"); shift ;;
  esac
done
if [[ ${#inputs[@]} -eq 0 ]]; then usage >&2; exit 2; fi
context_args=()
if [[ -n $context ]]; then
  if [[ ! -f $context ]]; then printf '错误：context 文件不存在：%q\n' "$context" >&2; exit 2; fi
  if [[ $context != /* ]]; then context="$PWD/$context"; fi
  context_args=(--context "$context")
fi

project_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${OPFUSION_BUILD_DIR:-$project_root/build}"
if [[ $build_dir != /* ]]; then
  build_dir="$project_root/$build_dir"
fi
binary="$build_dir/apps/opfusion"
# 同时兼容 Ninja Multi-Config 等生成器的 Release 子目录。
if [[ ! -x $binary && -x $build_dir/apps/Release/opfusion ]]; then
  binary="$build_dir/apps/Release/opfusion"
fi
if [[ ! -x $binary ]]; then
  printf '错误：未找到可执行程序 %s，请先执行 %s/compile.sh。\n' "$binary" "$project_root" >&2
  exit 1
fi

files=()
failed=0
listing=""
cleanup() {
  if [[ -n $listing ]]; then
    rm -f -- "$listing"
  fi
}
trap cleanup EXIT

for input in "${inputs[@]}"; do
  if [[ -z $input ]]; then
    printf '错误：输入路径不能为空。\n' >&2
    failed=$((failed + 1))
    continue
  fi
  # 保留调用者的相对路径语义，绝对化后也避免以 '-' 开头的文件成为 CLI 选项。
  if [[ $input != /* ]]; then
    input="$PWD/$input"
  fi
  if [[ -d $input ]]; then
    if [[ -z $listing ]]; then
      listing="$(mktemp)"
    fi
    # NUL 分隔支持空格与换行；pipefail 让目录读取或排序失败不会被当作空目录成功。
    if ! find -H "$input" -type f \( -iname '*.json' -o -iname '*.onnx' \) -print0 | LC_ALL=C sort -z > "$listing"; then
      printf '错误：无法完整扫描目录 %q。\n' "$input" >&2
      failed=$((failed + 1))
      continue
    fi
    if [[ ! -s $listing ]]; then
      printf '错误：目录 %q 中没有模型文件。\n' "$input" >&2
      failed=$((failed + 1))
      continue
    fi
    before=${#files[@]}
    while IFS= read -r -d '' file; do
      if [[ -n $context && $file -ef $context ]]; then continue; fi
      files+=("$file")
    done < "$listing"
    if [[ ${#files[@]} -eq $before ]]; then
      printf '错误：目录 %q 中没有可运行的模型文件。\n' "$input" >&2
      failed=$((failed + 1))
    fi
  elif [[ -f $input && ( ${input,,} == *.json || ${input,,} == *.onnx ) ]]; then
    if [[ -n $context && $input -ef $context ]]; then continue; fi
    files+=("$input")
  else
    printf '错误：输入 %q 不是模型文件或目录（请检查路径及通配符是否匹配）。\n' "$input" >&2
    failed=$((failed + 1))
  fi
done

success=0
total=${#files[@]}
if [[ $total -eq 0 && $failed -eq 0 ]]; then
  printf '错误：排除 context 后没有可运行的模型。\n' >&2
  exit 1
fi
index=0
for file in "${files[@]}"; do
  index=$((index + 1))
  printf '[%d/%d] %q\n' "$index" "$total" "$file" >&2
  if "$binary" inspect "$file" --json "${context_args[@]}"; then
    success=$((success + 1))
  else
    status=$?
    if [[ $status -eq 130 || $status -eq 143 ]]; then
      exit "$status"
    fi
    printf '失败：%q（退出码 %d），继续处理其余文件。\n' "$file" "$status" >&2
    failed=$((failed + 1))
  fi
done
printf '完成：成功 %d 个模型，失败 %d 项（含无效输入或目录扫描错误）。\n' "$success" "$failed" >&2
if [[ $failed -gt 0 ]]; then
  exit 1
fi
exit 0
