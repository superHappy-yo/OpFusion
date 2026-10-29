/// @file
/// 静态插件表与 Registry 装配的应用内接口；不向领域模块暴露发现机制。
#pragma once

#include "opfusion/registry/Registry.h"

#include <vector>

namespace opfusion::app {

/// 构建生成的静态工厂描述；字符串与函数地址借用已链接程序的静态存储。
/// 非空记录由构建生成器完整填充；本记录不拥有动态库或设备资源。
struct LinkedPlugin {
  const char* id = nullptr;
  const char* manifest = nullptr;
  const char* contentDigest = nullptr;
  const char* factorySymbol = nullptr;
  /// 换行分隔的包内文件清单，用于没有磁盘包路径时校验相对引用。
  const char* packageFiles = nullptr;
  /// 显式向调用方 Registry 登记贡献；查询工厂表本身不调用该入口。
  void (*factory)(Registry&) = nullptr;
};

/// 返回当前程序链接的工厂表，不扫描动态共享库或执行注册。
std::vector<LinkedPlugin> linkedPlugins();
/// 装配内置语义、来源绑定与可选插件，再返回冻结的独立 Registry。
/// pluginRoot 仅在调用期间借用；为空时使用编译清单，否则核对指定根目录的包。
/// 空目录不移除内置贡献；未链接或不兼容的包明确失败，不探测设备或选择回退。
/// 校验失败抛出 Error；文件系统错误也可能向调用方传播，不返回部分 Registry。
Registry loadRegistry(const char* pluginRoot = nullptr);

}  // namespace opfusion::app
