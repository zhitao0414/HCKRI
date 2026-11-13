# 数据准备和使用边界

数据集不随代码分发。下载者须自行确认原数据的获取条件、许可与引用，发布者也须确认本地数据转换规则的使用权。

## TON-IoT

将网络数据CSV放置为`data/raw/TON_IoT/train_test_network.csv`。配置包含既有镜像链接和已记录SHA256，仅作输入追溯，不表示该镜像为官方发布源，也不保证链接始终可用。推荐`configs/ton_iot.json`只读这个确切文件，最多分层抽取20000条记录。

未知候选类型为`mitm`、`ransomware`。必须检查实际文件中是否包含这些类型，以及输出的未知测试窗口数；类型名称出现在配置中不代表实际样本数非零。仅网络CSV的实验只有网络来源，不能当作网络、主机、遥测融合实验。

## Edge-IIoTset

将CSV放置为`data/raw/Edge_IIoTset/ML-EdgeIIoT-dataset.csv`。推荐`configs/edge_iiot.json`使用完整输入。类型及阈值按配置读取，确认实际拼写、大小写归一化与未知留出计数。

可通过`prepare_edge_multisource.py`划分网络、传输、应用协议视图。这些视图由同一个表派生，使用事件副本分组检查，不等同于独立采集数据。

## Power System Attack Dataset（PSAD）

把既有多分类ARFF文件放在`data/raw/PowerSystemAttack/multiclass/`，执行：

```bash
python scripts/prepare_power_system_attack_dataset.py
```

脚本按字段前缀导出4个CSV至`processed_multisource/`：`control_panel.csv`、`power_pmu.csv`、`protective_relay.csv`、`snort_alert.csv`。原始ARFF中需包含`marker`，以及脚本预期的PMU/控制/继电器/Snort字段。先核对原始格式与字段，不要将其他版本直接套入。

`configs/psad.json`保留修正后的`split_group_column=time`，排除`marker`、`time`、`dataset_part`输入，并按非缺失观测识别数值字段。`time`是全局原始行编号，同一行的4个来源副本必须进入同一划分。

marker到类型的转换、再到四级风险标签的映射都由研究脚本/配置定义，不是官方风险等级。31-34等未命名场景由脚本保留为`scenario_*`，不在映射中的类型由核心加载器默认映射为1；应核对是否符合研究目的，并事先补全配置，而非根据测试表现修改标签。

## 共通注意事项

- 所有真实配置禁用自动下载和合成回退；缺文件直接报错。
- 分割在窗口化之前进行，编码统计只拟合训练数据。事件随机分组不等于按完整会话或时间段隔离；如研究要求会话隔离，应提供可靠会话标识并重新验证。
- 四级风险标签来自显式类型映射；模型概率产生的预测分数不能用来生成测试参考标签。
- 输出`split_audit.json`记录原始选中记录、窗口计数、未知留出计数和已启用的组间重叠检查。
- 多视图排序/按行对齐是研究假设，不代表物理时间同步已经验证。
