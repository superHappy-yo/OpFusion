# codegen — 固定接口的设备内结构优化

状态：规格目录，尚无实现。修改前阅读本目录 [AGENTS.md](AGENTS.md)。

## 唯一责任

在固定区域、设备、接口和预算下搜索内部计算/存储/内核结构，通过注入的后端和测量端口形成候选。

| 项目 | 契约 |
|---|---|
| 输入 | TuningRequest、参考程序、后端/测量端口 |
| 输出 | TuningResult、新实现、验证和完整成本记录或失败原因 |
| 允许依赖 | 仅依赖 contracts 的公开类型和端口；其他能力由 apps 注入。禁止导入邻居模块的具体实现。 |
| 不负责 | 禁止更改区域成员、输出集合、边界布局、事件政策或全局映射；不直接写候选快照和正在使用的计划。 |

## 未来内部组织

按真实任务增加 `src/`、`tests/` 和 `docs/`；接口变更建议放 `proposals/`。目录是所有权边界，不要求独立发布包或独立服务。

对 R123 优化 tile、循环或内核组织；若要改变边界，返回新接口提案而不是覆写原 iid。

## 验收重点

同 iid 替换仍核对数值、资源和时延；所有编译/测量成本计入预算；不将仅输出 Y 的程序替换要求 T1/Y 的接口。

当前没有测试/启动命令。第一次实现时记录真实环境、命令与结果，不能把这里的验收目标写成已通过事实。

## 生成接口与探索入口

TuningRequest 的 oracle 必需、seed_candidate 可选；apps 可以对尚无已选实现的保留区域发起生成。ProgramSpec 与 BackendCompileRequest 见 contracts/program-inputs.md。

目标相关规则由 extensions 提供。首版实现贡献为完整匹配区域生成方案，或分解为已有算子；不假设任意单算子 lowering 能拼接融合。TVM 张量程序路线优先 PoC，Triton 作为 GPU 选项；选择依据与限制见 docs/architecture/technology.md。
