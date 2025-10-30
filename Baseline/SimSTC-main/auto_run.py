import os
import itertools
import subprocess

# 定义参数的取值范围
gpu_list = [0]  # 可用的 GPU 列表（根据你的 GPU 设置修改）
datasets = ['dy']  # 数据集类型
hidden_sizes = [128]  # 隐藏层大小
lrs = [1e-3]  # 学习率
dropouts = [0.5]  # dropout 比率
max_epochs = [10]  # 最大训练 epoch 数
warmup_epochs = [5]  # warmup epoch 数



# 定义运行命令的函数
def run_with_params(gpu, dataset, hidden_size, lr, dropout, max_epoch, warmup_epoch):
    cmd = [
        "python", "train.py",
        "--gpu", str(gpu),
        "--dataset", dataset,
        "--hidden_size", str(hidden_size),
        "--lr", str(lr),
        "--drop_out", str(dropout),
        "--max_epoch", str(max_epoch),
        "--warmup_epochs", str(warmup_epoch),

    ]
    # 打印命令以便检查
    print("正在运行命令:", " ".join(cmd))

    # 使用 subprocess 运行命令
    # 修改版
    import sys
    subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)


# 遍历所有参数组合
for gpu, dataset, hidden_size, lr, dropout, max_epoch, warmup_epoch in itertools.product(
        gpu_list, datasets, hidden_sizes, lrs, dropouts, max_epochs, warmup_epochs):
    run_with_params(gpu, dataset, hidden_size, lr, dropout, max_epoch, warmup_epoch)
