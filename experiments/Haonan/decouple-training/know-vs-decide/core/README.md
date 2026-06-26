# core — 最小可复现脚本

复现主结论：在冲突语境消歧上，从冻结模型中间层读出词义、晚一步作答，把 yes/no 准确率从原生前向的 0.527 升到 0.952；承诺温度旋钮（temperature）改变 calibration 而不动判别；纯排序训练的判别 0.914 接近 cross-entropy 的 0.949。

## 文件

- `common.py` —— 冻结 chat 模型的加载与 yes/no 答案打分（eval-only，默认 Qwen3-4B）。
- `extract_features.py` —— 对每条题、每个义跑一次冻结前向，按层池化语境位置的隐状态，存成特征缓存。
- `know_decide.py` —— 共享核心：读出头（per-candidate 共享权重线性打分）、排序损失与 cross-entropy、晚承诺（带 temperature）、AUC / ECE 指标。
- `run_p1_p3.py` —— 跑判别旋钮可分（P1）与晚承诺胜原生（P3），把数字写到 `outputs/p1_p3.json`。
- `data/corpus.json` —— carrier：bank/ATM 式语境消歧，436 条，含 split（train 210 / val 62 / gen_test 146 / test_hand 18）。主指标在 gen_test（n=146，按词留出）；test_hand 是 18 条人工题，仅作 face validity。
- `data/polysemy_pairs.json` —— 16 个一词多义基础词及其两义。
- `tests/test_know_decide.py` —— sanity 测试（设计恒等式、AUC/ECE 边界、temperature 不改 argmax）。

## 运行（两步）

特征缓存 `data/feats_qwen3_4b.pt`（约 700 MB）属于缓存产物，不随 PR 提交，需本地生成一次：

```
# 1) 需要一次 GPU：生成特征缓存 data/feats_qwen3_4b.pt
python extract_features.py

# 2) 纯 CPU 分析：写 outputs/p1_p3.json
python run_p1_p3.py

# sanity
python -m pytest tests -q
```

模型权重默认从本地 Hugging Face 缓存离线加载（见 `common.py` 顶部的路径与 `--model`）。
