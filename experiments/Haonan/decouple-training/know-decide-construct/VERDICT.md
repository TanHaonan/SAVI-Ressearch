# VERDICT — calibrated yes/no shape can be trained into the native forward, and beats one-hot cross-entropy

## ① 一句话结论

用一个解耦的训练目标（margin discrimination + calibration + KL fluency leash）训一个小 LoRA 到收敛，frozen Qwen3-4B 的原生前向可以同时做到两件事：在证据唯一确定的题上自信答对（determinate accuracy 0.97），在两义都成立的题上保持校准的不确定（ambiguous |p−0.5| 0.04，entropy 接近最大）；在同等 discrimination（同等答对能力）下，这个解耦目标的校准优于标准 one-hot cross-entropy（配对 bootstrap CI 排除 0，3/3 seed 胜）；而且这个改变进入了原生前向本身，不是停在一个外接读出层上。

## ② 设置

- **Model**：frozen Qwen3-4B + LoRA（rank 16，作用于 q/k/v/o/gate/up/down projection），只训 LoRA。
- **Carrier（词义消歧 yes/no 题，两类）**：
  - *Determinate（证据唯一确定）*：scene 已给出某个词的语境义，后接一个针对某义的 yes/no 问题，gold 确定。共 436 条（train 210 / val 62 / gen_test 146 / 另有 18 条手检）；训练用 train、评测用 gen_test。
  - *Ambiguous（两义都成立）*：中性 scene（不透露该词取哪个义）+ 针对义 a 的 yes/no 问题，target p(yes)=0.5（target 用外部枚举的两个词义 50/50 给出）。共 16 词 × 4 个中性 scene = 64 条；按词留出，6/16 词（约 1/3）放进 held-out。
- **Metric**：在答案位置对 yes / no 两组 token 取 logits，`p(yes)=sigmoid(logit_yes − logit_no)`，读的是模型原生输出，不加任何外接解码。Determinate 报 accuracy；ambiguous 报 |p(yes)−0.5|（越小越接近校准的 50/50）与 entropy；通用文本上报 perplexity（相对 base 的比值）。
- **Loss（decoupled）**：见配套 issue §3。三项为 (i) margin hinge，只压正确答案排在前面（保 discrimination，不无条件推尖）；(ii) calibration 项 `(p(yes)−0.5)²`，把 ambiguous 题塑成应有的 50/50 形状；(iii) KL-to-base fluency leash，在一小批通用句子上锚住原模型。`λ_shape` 控制 calibration 项权重。
- **控制组**：
  - *coupled*：determinate 题用 hard one-hot cross-entropy（会无条件推尖），ambiguous 题无信号。
  - *shuffle*：解耦目标但打乱 determinate 的 gold 标签（用来验证"答对"是真信号而非记忆/伪迹）。
  - 各控制组与各 `λ_shape` 均跑 3 个 seed。

## ③ 结果

训 8 个 epoch 至收敛。前沿（decoupled，按 3 seed 取均值；数值见 `results.json`）：

| λ_shape | determinate accuracy | ambiguous \|p−0.5\| | ambiguous entropy | perplexity（×base） |
|---|---|---|---|---|
| 1 | 0.963 | 0.213 | 0.558 | 1.01 |
| 3 | 0.961 | 0.163 | 0.617 | 1.00 |
| 10 | 0.934 | 0.096 | 0.664 | 1.01 |
| 20 | 0.954 | 0.062 | 0.682 | 1.02 |
| **40** | **0.970** | **0.038** | **0.688** | 1.01 |
| 80 | 0.813 | 0.038 | 0.689 | 0.99 |
| coupled（one-hot CE，无 shape 项） | 0.947 | 0.408 | 0.237 | 1.00 |
| shuffle（控制） | 0.516 | — | — | — |

关键数字（与配套 issue 一致；数值见 `results.json`）：

- **两件事同时达到**：在 λ_shape=40，determinate accuracy 0.97 与 ambiguous |p−0.5| 0.04 同时成立，ambiguous entropy 接近最大（0.688，最大约 0.693）。
- **同等 discrimination 下解耦胜 coupled**：取均值 determinate accuracy 首次 ≥ coupled（0.947）的最小 λ_shape（λ_shape=1，accuracy 0.963 ≥ 0.947），在 held-out ambiguous 题上做配对 bootstrap，每 seed 校准优势 +0.16 到 +0.23，pooled CI [0.16, 0.23] 排除 0，3/3 seed 胜。此外 λ_shape=40 在两个轴上都优于 coupled（accuracy 0.970 > 0.947 且 |p−0.5| 0.038 ≪ 0.408）。
- **shuffle 控制塌到约 chance**：determinate accuracy 0.516（3 seed 均值），说明 determinate 题的恢复是真信号，不是记忆。
- **fluency 维持**：本实验（8 epoch）实测相对 base 的 perplexity ratio 约 0.96–1.04（贴近 1.0，语言能力基本维持）。[^smoke]

[^smoke]: 早先配套 issue 引用的 ×0.93–0.98 来自 2-epoch smoke run；本处与 `results.json` 一律以 8-epoch 实测（0.96–1.04）为准。

**改变进入了原生前向，不是停在外接读出层。** 在同一个回答用的前向上，额外训一个 layer-swept 线性 readout 去预测 gold yes/no，比较 readout accuracy 与 native accuracy（同一批 gen_test 题，数值见 `results.json`）：

| model | native accuracy | readout accuracy | readout − native |
|---|---|---|---|
| frozen base | 0.534 | 0.856 | +0.322 |
| decoupled λ_shape=20（3 seed） | 0.94–0.96 | 0.93–0.97 | −0.014 / +0.000 / +0.014 |

在 frozen base 上，外接 readout 比原生前向多恢复 +0.322——也就是说答案本就可从这个前向里读出，但原生输出没用上它。在加了 adapter 的模型上，外接 readout 再加 ≈0（λ_shape=20 最干净，每 seed ≤ +0.014）。即 adapter 把 discrimination 放进了原生前向，而不是在 readout 层掩盖一个仍然外置的能力。

## ④ 成立条件与边界

- **在以下条件下成立**：训到收敛（8 epoch）、用解耦目标、且 calibration 项权重足够（`λ_shape` 约 20–40）。此时一个小 LoRA 能把"该自信则自信、该 50/50 则 50/50"的 yes/no 形状放进原生前向；在同等 discrimination 下校准优于 one-hot CE（CI 排除 0）；discrimination 确实进了前向（外接 readout 加 ≈0，对比 frozen base 的 +0.32）；fluency 基本维持（8-epoch 实测 perplexity ratio 约 0.96–1.04）。
- **边界**：
  - **calibration 项权重不能太小也不能太大**。默认低权重（λ_shape=1）下 discrimination 压力盖过形状（accuracy 0.963 但 ambiguous |p−0.5| 0.213，比 coupled 还差）；λ_shape=80 又过度正则，determinate accuracy 掉到 0.813。两件事同时成立的工作点是 λ_shape=40。
  - **"进入前向"指的是答案位置的 logit**。外接 readout 加 ≈0 在 λ_shape=20 干净，在 λ_shape=10 随 seed 波动（3 seed 中 2 个仍部分外置）。能否迁移到自由生成（而非 yes/no 探针位置）是另一个问题。
  - **carrier 是 yes/no 二元**，是多峰输出的二元影子（即校准）；k≥3 的真·多峰不在本实验内。
  - **ambiguous 题的 target 50/50 来自外部枚举的两个词义**，不是模型自己产生的；开放生成里"应有的确定度"从何而来仍未解决。
  - LoRA 只证明*某些*权重改动可以承载这个形状，不指向"attention 内某个特定机制"。
- **会推翻的情形**：若在同等 discrimination 下解耦不再胜 coupled（CI 含 0 或偏向 coupled），或在加了 adapter 的模型上外接 readout 又能多恢复 ≥ +0.10——本实验中两者都未出现。
