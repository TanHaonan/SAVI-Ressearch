# VERDICT — controllable-posterior

## 1. 一句话结论

在一个 prior-free、正确 posterior 精确已知的载体上，frozen Qwen3-4B 加一个小 LoRA（rank 16）可以让模型原生前向的 option 分布与"在 j 个存活选项上均匀"的真值几乎重合（到真值的 total variation ≈ 0.02，且不随存活数 j 上升）；塌掉这一多峰分布的是 one-hot cross-entropy 这一 objective，而不是模型的表示能力——同一构造下用 one-hot cross-entropy 训练会把分布精确塌成单个选项。

## 2. 设置

- Model：frozen Qwen3-4B，外接一个 LoRA（rank 16），只训 LoRA，backbone 不动。读出位置取答案位置上各 option 字母的 logits，softmax 作为模型的逐步分布。
- Carrier：prior-free 的消元谜题。每题有 k 个内容无关的 option（随机名词映射到字母 A、B、……），一组消元线索排除掉 (k − j) 个，剩下 j 个存活选项。在 uniform prior 加硬消元的假设下，Bayes-optimal posterior 精确等于"在这 j 个存活选项上 uniform"。k ∈ {2, 3, 4, 5}；共 1120 题，其中 259 题为留出集。每题有两种 prompt 视图（目标相同）：oracle（直接在 prompt 里点名存活选项）与 self（只给消元线索，存活选项需自行推断）。
- Objective：decoupled = KL(已知真值 ‖ softmax(option logits)) + 一个在通用文本上的 fluency KL anchor（锚住原模型）。本载体的 target 精确已知，因此只用这个 KL shape 项 + fluency anchor，不含显式 margin / rank 项。
- Metric：每题取模型 option 分布到精确真值的 total variation（TV，越小越接近真值），按 (k, j) 分格在留出集上汇报，item-level bootstrap CI。
- 控制组：coupled = 对单个存活选项做 one-hot cross-entropy（标准 one-hot 训练）；shuffle = 在同一 k 组内打乱训练标签（信号被破坏的对照）。3 个 seed。

## 3. 结果

数值见 `results.json`；以下为 3 个 seed 的均值。

oracle + decoupled，到真值的 mean TV，按存活数 j：

| j=1 | j=2 | j=3 | j=4 | j=5 |
|---|---|---|---|---|
| 0.00 | 0.017 | 0.021 | 0.017 | 0.016 |

各 j 均 ≤ 0.10，且随 j 平（j=5 的"在 5 个选项上均匀"与 j=2 一样接近真值），没有随分布要铺得更开而上升。

objective 对照（oracle、ambiguous 即 j ≥ 2）：

| regime | survivor 内部非均匀度 | TV |
|---|---|---|
| decoupled（soft target） | 0.018 | ≈ 0.02，随 j 平 |
| coupled（one-hot cross-entropy） | 0.63 | 0.50 / 0.67 / 0.75 / 0.80（j=2/3/4/5） |
| shuffle（控制） | — | mean TV 0.37 |

coupled 的 TV 精确等于 1 − 1/j（0.50、0.667、0.75、0.80），这是把全部概率压到单个选项的特征——one-hot cross-entropy 把已校准的 posterior 精确塌成单点；decoupled 的 soft target 才能把分布稳在真值附近（TV 0.02 对 0.50–0.80）。shuffle 把信号打乱后 mean TV 升到 0.37，说明 decoupled 的结果不是来自记忆或泄漏。

self（存活选项需从线索推断）相对 oracle 的额外 TV，按 j：+0.021 / +0.016 / +0.019 / +0.010（j=2/3/4/5），约 0.01–0.02、不随 k 增长；self 在各格仍 TV ≤ 0.05。

## 4. 成立条件与边界

- 该结论成立的条件：option 内容无关、对称（无 frequency prior）；正确 posterior 由消元线索精确决定（真值已知、可直接量 TV）；每个 (k, j) 格有上百题；训练用 decoupled 的 soft-target objective。在这些条件下，"装下并保持一个铺在最多 5 个选项上的校准分布"在这一构造的能力之内；失败发生在 objective（one-hot cross-entropy 会塌），而非表示能力。
- 该结论不覆盖的范围：本载体的消元线索是显式、直接的，"推断哪些选项该保留"这一步在此被刻意做得简单（self 与 oracle 仅差约 0.01–0.02），因此这里没有压到"推断哪条题该不确定"这条轴；换成间接或软线索会对这条轴施压。
- 该结论限定在答案位置的读出层：上面量的是答案位置 option logits 的分布形状，未验证这一校准会迁移到自由生成。
- 关于 metric：以"到已知真值的 TV"来度量，能干净地把"在合法集合上铺开"与"塌成单点"分开，并因此暴露出 coupled 精确塌到 1 − 1/j；这是单纯的峰数代理指标做不到的。
