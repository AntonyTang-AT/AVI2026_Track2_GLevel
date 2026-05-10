本目录为从宿主机赛事数据目录拷贝的官方 CSV 镜像，便于在无 /data/Super-Lu 的机器上对齐标签与划分。
不含音频/视频特征（.npy），特征体积较大，请在本机设置环境变量 SUPERLU_DATASET 指向完整 dataset 根目录。

镜像来源（示例）：
  /data/Super-Lu/dataset/train_data.csv
  /data/Super-Lu/dataset/val_data.csv
  /data/Super-Lu/dataset/test_data_basic_information.csv
  /data/Super-Lu/dataset/submission.csv（若存在）

同步命令：
  bash tools/sync_superlu_official_csv.sh
