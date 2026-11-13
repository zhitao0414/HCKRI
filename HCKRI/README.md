# HCKRI

多层次关联建模与知识增强的安全数据风险识别研究代码。

本代码实现多尺度残差CNN、Bi-LSTM、来源门控、Transformer、多阶段表征融合、原型风险库、原型对比损失及高置信伪标签训练。输入为CSV事件表，输出包含正常类和四个风险等级，并单独保存未知攻击判定标记。

## 安装

建议创建独立Python环境。在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
```

CUDA版PyTorch应按机器驱动和GPU架构选择对应构建，不要把系统CUDA工具链版本与PyTorch运行时版本混为一谈。`requirements.txt`只声明依赖，不伪装成跨平台锁文件；已记录的5090环境版本见[复现说明](docs/REPRODUCIBILITY.md)。


