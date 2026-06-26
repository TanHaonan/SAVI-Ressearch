# Haonan — a decoupled-training front-end for global decoding on LLMs

把 Viterbi / beam 这类全局解码套到大模型上，通常并不比逐步 greedy 更强：标准训练（one-hot cross-entropy）把"该有多确信"（calibration）和"承诺哪个答案"（commit）绑在一起，逼模型把概率压到单个 token 上，于是每一步只剩一个候选，全局解码没有可比较的备选。本目录用一个**解耦训练目标**把这两件事分开——让原生前向的输出在证据充分时尖、在证据不足时铺开（calibrated 多峰）。四个主实验依次给出：(a) 正确答案本就在前向里、只是被过早承诺盖住；(b) 这种"该尖则尖、该平则平"的形状能被训进模型本身、且胜过 one-hot CE；(c) 塌掉多峰的是训练目标而非模型容量；(d) 多峰输出接上全局解码确实能用、但瓶颈在约束的来源。两个支撑实验给出动机与边界：在真实大模型那种过尖的输出上 Viterbi 与逐位置 marginal 几乎不分高下（§2），以及 calibration 这个形状目前只进答案位置的 logit、不进自由生成（§6 边界）。取向：这套 front-end 让全局解码在大模型上良定义，而约束的可靠来源正是 SAVI 的 verifier 能补上的部分。

**通用设置（对应 (a)–(d) 四个主实验）**：frozen Qwen3-4B + 一个小 LoRA（rank 16），只训 LoRA；在答案位置对候选取 softmax 作为逐步分布；均在 held-out 上评测；对照含 coupled（= one-hot CE）与 shuffle（打乱标签）。§2 的"过尖度"测量在 **Qwen2.5-3B** 上（直接读出 emission，无 LoRA）；§6 的边界在另一组消歧题上。各目录的 VERDICT.md 标明各自设置。

> 解耦损失（issue §3）是一个**统一形式**；各实验只实例化其中的子集，实际损失以各目录 VERDICT.md 为准：(c)/(d) 用 KL(已知 target ‖ softmax) + fluency anchor（无显式 margin 项）；(b) 用 bounded-margin hinge + (p−0.5)² calibration + fluency anchor；所有 coupled 对照 = one-hot CE。

## issue 论断 → 对应目录 / 文件

| issue 位置 | 论断（plain） | 关键数字 | 目录 |
|---|---|---|---|
| §4(a) | 信念与承诺可分：从中间层读出词义、晚一步作答，恢复被过早承诺盖住的正确答案 | 0.527 → 0.952（gap +0.42, CI[0.34,0.51]） | `know-vs-decide/` |
| §4(b) | calibrated 的 yes/no 形状能训进原生前向，且同等判别下胜 one-hot CE | 确定题 0.97 与歧义题 \|p−0.5\| 0.04 同时达到；配对 CI[0.16,0.23]，3/3 seed 胜 | `know-decide-construct/` |
| §4(c) | 塌掉多峰的是训练目标不是容量：解耦保持均匀，one-hot 精确塌成单点 | 到真值 TV ≈ 0.02（最多 5 选项）对 one-hot 的 1−1/j（0.50–0.80） | `controllable-posterior/` |
| §4(d) | 多峰输出接全局解码可用，但瓶颈是约束来源 | 真约束 0.00 → 0.88 / 0.92；模型自判约束 AUC ≈ 0.67、合成后 0.04–0.08 | `latent-consistency-decode/` |
| §2（动机） | 真实大模型输出太尖（Qwen2.5-3B 上 top-1 ≈ 0.92），Viterbi ≈ 逐位置 marginal | 真实 emission 准确率差贴近 0；合成随机势 ≤ 0.05；仅对抗 fan 特例才分开 | `marginal-vs-path/` |
| §6（边界） | 判别会迁移到自由生成，calibration 的形状停在答案位置 logit、不进自由生成 | 确定题 confident-wrong 0.29→0.01；歧义题 confident-commit 0.31 ≈ base 0.34 | `native-abstain/` |

## 目录结构与复现

- 四个主目录（`know-vs-decide` / `know-decide-construct` / `controllable-posterior` / `latent-consistency-decode`）各含 `VERDICT.md`（结论 + 设置 + 结果 + 成立条件）、`PREREG.md`（跑前的 hypothesis + KILL）、`core/`（最小可复现脚本）、`results.json`（结果数值源文件）。
- `marginal-vs-path/` 与 `native-abstain/` 是支撑性证据，含 `VERDICT.md` + `PREREG.md` + `results.json`，不含复现脚本。
- 每个 `core/` 自包含：`latent-consistency-decode/core/` 把跨模块依赖 vendored 到 `core/_deps/`（无需任何外部目录）。复现按各目录 `core/README.md` 的两步说明：先用一次 GPU 生成特征缓存 / 训练 adapter（缓存与 adapter 体积大，不随本目录提交，需本地生成），再用 CPU 跑分析脚本写出结果；`results.json` 为对照参考。模型权重默认从本地 Hugging Face 缓存离线加载（`cache_dir` 读环境变量，未设则用 transformers 默认缓存）。
