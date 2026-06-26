# PREREG — controllable-posterior

## Hypothesis

在一个 prior-free、正确 posterior 精确已知的载体上，frozen LM 加一个小 LoRA 是否能让原生前向的 option 分布匹配"在 j 个存活选项上均匀"的真值——并把"能否产生这个铺开的分布形状"（oracle，直接点名存活选项）与"能否从线索推断该不确定"（self，只给消元线索）分开，从而判断是表示能力还是别的因素限制了多峰分布。

## Setup

- Model：frozen Qwen3-4B + LoRA(rank 16)，只训 LoRA；答案位置 option 字母 logits 的 softmax 为模型分布。
- Carrier：prior-free 消元谜题。k 个内容无关 option，消元线索排除 (k − j) 个，剩 j 个存活；uniform prior + 硬消元下真值精确等于"在 j 个存活选项上 uniform"。k ∈ {2, 3, 4, 5}；每题两种 prompt 视图（oracle 点名存活选项 / self 只给线索），目标相同。约 1120 题，留出集约 259 题。
- 条件网格：mode ∈ {oracle, self} × regime ∈ {decoupled, coupled, shuffle} × k ∈ {2, 3, 4, 5}，每个 run 混合训所有 j ∈ {1..k}，3 个 seed；外加 k=4 的 cross-talk 控制（确定题与歧义题分开单训，与混训对比）。
- Objective：decoupled = KL(已知真值 ‖ softmax(option logits)) + bounded-margin rank 项 + fluency KL anchor；coupled = 对单个存活选项做 one-hot cross-entropy；shuffle = 同一 k 组内打乱标签。
- Metric：每题到精确真值的 total variation，按 (k, j) 分格在留出集上汇报，item-bootstrap CI。次级指标：survivor mass（→1）、survivor 内部非均匀度（→0）、确定题 argmax 准确率。

## Pre-registered prediction & KILL

容量（headline）：
- PASS if：oracle + decoupled 对每个 k ∈ {2, 3, 4, 5}、每个 j 都达到 TV ≤ 0.10（CI 上界 ≤ 0.12），且 survivor 内部非均匀度 ≤ 0.10——产生铺在 j 个选项上的分布在该构造能力之内。
- KILL if：oracle 的 TV 随 j 单调上升、并在 j ≥ 3 时仍 > 0.20（即便已点名存活选项）——属真实的表示/容量上限。

推断代价：
- PASS if：self 的 TV 在匹配的 (k, j) 上高于 oracle 且随 k 增长，同时 self 的 survivor 内部非均匀度升高而 oracle 不升——限制定位在"推断该对哪些选项不确定"而非"产生分布形状"。
- KILL if：self 处处 ≈ oracle——载体太易（推断步骤近乎平凡），需要加间接线索并重跑。

objective 可靠性：
- PASS if：decoupled 在歧义题上的 survivor 内部非均匀度低于 coupled（coupled 把歧义题塌掉），同时确定题准确率不输，且 shuffle 失败（TV ≈ 随机水平）。
- KILL if：coupled 在歧义题上与 decoupled 打平——soft-target objective 在此没有带来收益。

cross-talk（确定题是否污染歧义题）：
- PASS if：混训的确定题准确率与歧义题 TV 与分开单训相比，差距分别在 +0.03 / +0.05 以内——确定项不实质性破坏歧义分布。
- KILL if：混训的歧义题 TV 比歧义题单训差 ≥ +0.10——确定题的 margin/CE 渗入并塌掉了歧义分布。
