# 复现说明

## 默认配置

推荐配置沿用保存的三数据集5090运行记录。序列12、步长3、统一维度64、来源嵌入16、卷积核3/5/7、单层双向LSTM每方向64、Transformer 2层4头、Dropout 0.20；AdamW、学习率0.001、权重衰减0.0001、batch64、8轮。验证风险类宏平均F1最高检查点保留；无提前停止。

TON/Edge/PSAD种子为9500/9400/9300。统一入口默认确定性计算、关闭cuDNN benchmark，开启cuDNN确定性模式，设置`CUBLAS_WORKSPACE_CONFIG=:4096:8`，固定Python、NumPy、PyTorch种子。确定性开关不能保证跨GPU与依赖版本完全逐位相同。

5090记录的版本为Python 3.14.7、PyTorch 2.14.0+cu130、CUDA运行时13.0、NumPy 2.4.6、pandas 3.0.5、scikit-learn 1.9.0、SciPy 1.18.1、Matplotlib 3.11.1。这是原运行记录，不是所有机器可安装的版本承诺。发布时CPU快速自检使用的实际版本另见`VERIFICATION.json`；新运行会自动保存本机版本。

## 指标

- `precision/recall/f1/fpr`：风险类别1-4的宏平均；正常类是类别0。
- `uarc`：留出未知攻击窗口被判为任意风险（预测大于0）的比例。旧`udr`是这个量的兼容字段。
- `unknown_flag_recall/precision/f1`：基于独立保存的未知标记计算。
- `known_flag_rate`：已知测试窗口被错误标为未知的比例。
- `accuracy`：所有5类的准确率。
- `per_class_counts.csv`和混淆矩阵包括正常类。
