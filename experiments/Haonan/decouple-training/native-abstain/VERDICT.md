# 训练得到的 calibration 形状是否进入自由生成

## 结论
discrimination（把先前知道却没用上的正确答案用出来）会迁移到自由文本生成；但 calibration 的形状（在证据不足时铺平概率）停在答案位置的 logit 上，不进入自由生成。

## 设置
- carrier：frozen Qwen3-4B + 一个小 LoRA（rank 16），用一个 decoupled training objective 训练（把 calibration 与 commit 解开）。比较四个模型：base（无 LoRA）、coupled（即 one-hot cross-entropy）、decoupled（两档强度，记为 lam=10 与 lam=40）。
- 任务：对确定题（determinate，上下文已唯一确定词义）和真歧义题（ambiguous，两义都成立）让模型自由作答，不加“只回答 yes/no”的约束，greedy 解码。
- metric：用一个不同的 cached 模型（Llama-3.1-8B-Instruct）把每个回答的立场分为 commit-yes / commit-no / hedge / abstain。确定题看 selective accuracy（在它给出明确答案时的正确率）与 confident-wrong rate；歧义题看 confident-commit rate（在两义都成立时仍强行选一边，这是要避免的行为）。另测通用文本 perplexity 作为 fluency 控制。
- 控制组：base 与 coupled 作为对照；自由生成与同一批模型在答案位置 logit 上读出的 calibration 形成对比。

## 结果
确定题上 discrimination 迁移到生成：confident-wrong rate 从 base 的 0.29 降到 0.01，selective accuracy 从 base 的 0.36 升到 0.92–0.97。歧义题上 calibration 不迁移：答案位置 logit 上最平的模型（decoupled lam=40，歧义题 |p−0.5|=0.04）在自由生成里 confident-commit rate 为 0.31，与 base 的 0.34 基本相同（两者 bootstrap 置信区间重叠），并不低于 coupled。各模型通用文本 perplexity 在 20.3–20.6（base 20.5），fluency 未塌。数值见 `results.json`。

| model | det selective acc | det confident-wrong | amb confident-commit | perplexity |
|---|---|---|---|---|
| base | 0.36 | 0.29 | 0.34 | 20.5 |
| coupled | 0.92 | 0.04 | 0.19 | 20.1 |
| decoupled lam=40 | 0.97 | 0.01 | 0.31 | 20.3 |

## 成立条件与边界
- discrimination 迁移到生成的条件：在答案位置 logit 上训练 discrimination（含 coupled），greedy 解码自由作答时正确答案会被用出来。
- calibration 不迁移的条件：calibration 形状只训练在答案位置的 yes/no logit 上；在该条件下，一个 p≈0.5 的 logit 不会让 greedy 生成出带 hedge 的文本，仍会选出一句明确表态。因此“形状进入原生前向”这一说法应限定到“进入答案位置的 logit，不进入自由生成”。
- 可能改变结论的条件：若把 shape objective 训练在生成出的 token 上（而非只在答案位置 logit 上），并在 determinate accuracy 持平时把歧义题 confident-commit 压到低于 coupled，则边界会移动。
- 判别模型的局限：Llama-3.1-8B 对 hedge 判得偏宽，hedge 的绝对比例依赖判别模型阈值；稳健的信号是相对比较（decoupled lam=40 ≈ base，且不低于 coupled），它在不同阈值下都指向 calibration 不迁移。
