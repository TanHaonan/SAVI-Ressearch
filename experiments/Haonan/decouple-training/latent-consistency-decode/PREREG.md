# PREREG — latent-consistency-decode（跑前预注册）

## Hypothesis

在一类无先验偏好的多变量一致性题上（slot 间有真实 same / different 约束，并植入一处局部-全局冲突），global consistency decoding 胜过 per-slot greedy，当且仅当 (a) 节点 emission 保留了正确 option 的概率（calibrated 读出，未塌缩）且 (b) 约束 edge 既有信息量又与生成模型 decorrelated；同时单条最优路径形式（viterbi / max-product）相对逐位置 marginal（sum-product）至多多约 0.05 accuracy——增益来自"用上约束"（globality），不是"取单条路径"。

## Setup

- frozen Qwen3-4B + 小 LoRA（rank 16），只训 LoRA；emission = 答案位置候选 option 的 softmax 分布。
- Carrier：T 个 slot（T∈{3,4}），每 slot k 个 option（k∈{3,4}，内容中性名词）；slot 间 same / different 约束；每题植一处局部-全局冲突，使 per-slot greedy 全局不一致；joint MAP（gold）精确可算。共 960 题，227 题留出。
- emission 三种：`decoupled`（calibrated soft target）/ `coupled`（one-hot CE）/ `base`（未训练读出）。
- edge 四种：`oracle`（真约束，上限）/ `self`（Qwen3-4B 判自己的逐对 yes/no query）/ `cheap`（更小且 decorrelated 的模型做同样 query）/ `shuffle`（跨题打乱的随机 hard 约束）。
- decoder 三种：`greedy` / `marginal`（hard 约束下精确后验 marginal 的 argmax）/ `viterbi`（hard 约束下精确 MAP）。
- metric：joint exact-match；逐 slot accuracy；item-bootstrap CI。每 epoch 在 oracle+marginal 上评一次，外加一次基于 checkpoint 的最终评测（重载 adapter → 建 edge → 解码）。

## Pre-registered prediction & KILL

**P1 — 节点子空间的必要性（emission 一侧：保留正确 option 的概率是否必要）。**
- PASS if：oracle edge 下，`decoupled` 的 marginal joint exact-match 高于 `coupled`（配对差的 CI 不含 0）——因为 coupled 把逐 slot 分布塌到局部偏好的 option，删掉了正确 option，连完美约束也救不回。
- KILL-A if：`coupled` + oracle ≈ `decoupled` + oracle —— 该 carrier 上读出本就没塌掉正确 option，calibration 在此不带来增益（记为条件受限的 null）。
- KILL-B if：连 `decoupled` + oracle ≈ `greedy` —— 正确 option 根本无法从前向读出（不是该 carrier 上的使用失败），方案不适用，瓶颈在节点一侧。

**P2 — 约束的 make-or-break（edge 一侧：有信息量 ∧ decorrelated；即"global 是否真胜 greedy"）。**
- PASS if：`decoupled` emission 下，`cheap` edge 的 marginal 胜 `greedy`（CI 不含 0），且 `self` ≈ `greedy`、`shuffle` ≤ `greedy`、`oracle` 为上限 —— 一个小的 decorrelated 模型能供给可用约束，增益来自 globality + decorrelation，不是 scale。
- KILL-A if：`oracle` 胜 `greedy` 但 `cheap` ≈ `greedy` —— 把自然语言约束抽成兼容矩阵（约束抽取）才是瓶颈，不是解码器；cheap 模型供不出有信息量的 edge。
- KILL-B if：连 `oracle` ≈ `greedy` —— carrier 没有可利用的真实跨 slot 耦合（冲突缺失或被 unary 平凡解决）→ 需修 carrier（提高约束密度 / 加强冲突）。

**P3 — viterbi vs marginal（解码器一侧）。**
- PASS（预期）if：`viterbi` − `marginal` joint exact-match ≤ 约 0.05（CI 上界小）—— 单条最优路径相对逐位置 marginal 没有独立的 accuracy 价值（除非对抗性 fan potential）。此项为报告项，不是 headline；改进的主张在 P2（global vs greedy），由 marginal 版即可交付。
- KILL of "path is special" if：`viterbi` ≫ `marginal`（> 0.05，CI 不含 0）—— 该 carrier 上存在真实 fan 结构，与先验相悖，会重新打开单路径价值的问题。

**P4 — cheap edge 是否真的 decorrelated。**
- PASS if：`cheap`-builder 的 edge 错误与生成模型（Qwen3-4B）的 edge 错误 φ < 0.5；`self`-builder φ ≥ 0.6（同源）。
- KILL if：`cheap` φ ≥ 0.6 —— 所谓"decorrelated"的模型在此并不独立，其净增益（若有）不来自独立性。
