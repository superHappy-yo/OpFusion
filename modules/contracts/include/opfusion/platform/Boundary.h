#pragma once

// 一次部署的驻留、交付、动态限额和计时口径，不改写平台的物理库存。
#include "opfusion/platform/Platform.h"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace opfusion {

/// 已有输入副本的声明；readyUs 使用计划计时轴，数据引用不等于已读取内容。
struct Residency {
  std::string id;
  TensorId tensor = 0;
  std::string representation;
  std::string dataRef;
  std::optional<double> readyUs;
};

/// 必须交付的完整逻辑输出及物理表示；不默认以最后一个 kernel 完成为交付。
struct OutputTarget {
  std::string id;
  TensorId tensor = 0;
  std::string representation;
};

/// 计时起点语义；初始化仍未建模，不能将其省略为零。
enum class TimingStart { Unknown, InputsAvailable, FirstRegionStart, BeforeInitialization };
/// 结束口径区分区域输出就绪和要求的输出交付。
enum class TimingFinish { Unknown, RegionOutputsReady, OutputsDelivered };

/// 部署时只声明计时口径；具体事件由以后生成的 Plan 绑定。
struct TimingPolicy {
  TimingStart start = TimingStart::Unknown;
  TimingFinish finish = TimingFinish::Unknown;
  bool includesInputTransfers = false, includesOutputTransfers = false,
       includesInitialization = false;
  Validation validate() const;
};

/// 计时范围的语义与具体事件引用；事件必须来自同一份 Plan。
/// InputsAvailable 等于全部声明 SourceReady 的最晚时刻；无来源时为计划原点零。
/// OutputsDelivered 等于全部要求 SinkComplete 的最晚时刻；无输出时等于起点。
/// 这两个零成本边界事件不能自行加入等待，以免截掉排队或虚增交付时延。
struct TimingScope {
  std::string id;
  TimingStart start = TimingStart::Unknown;
  TimingFinish finish = TimingFinish::Unknown;
  std::string startEvent;
  std::string finishEvent;
  bool includesInputTransfers = false;
  bool includesOutputTransfers = false;
  bool includesInitialization = false;
};

/// 一次部署的显式边界；未知动态容量不继承物理容量，更不能当作零占用。
struct Boundary {
  std::string id;
  std::string platformId;
  std::vector<Residency> inputs;
  std::vector<OutputTarget> outputs;
  std::map<PoolId, std::optional<Bytes>> availableBytesByPool;
  /// 单位沿用规范 Resource；不能声明大于物理容量的动态上限。
  std::map<ResourceId, std::optional<double>> availableResources;
  std::optional<TimingScope> timing;
  /// 只借用平台核对引用和上限；事件覆盖由 Plan 校验，不读取或移动输入。
  Validation validate(const Platform &platform) const;
};

} // namespace opfusion
