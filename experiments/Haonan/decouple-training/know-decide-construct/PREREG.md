# PREREG — 解耦目标能否把 calibrated yes/no 形状训进原生前向

> 跑前预登记。判据写在跑之前，结果见 `VERDICT.md` 与 `results.json`。

## Hypothesis

用一个解耦的训练目标（margin discrimination + calibration + KL fluency leash）训一个小 LoRA，frozen Qwen3-4B 的原生前向能同时做到：在证据唯一确定的题上自信答对，在两义都成立的题上保持校准的 50/50；且这个解耦目标在同等 discrimination 下校准优于标准 one-hot cross-entropy；且这个改变进入原生前向本身，不是停在一个外接读出层上。

## Setup

- **Model**：frozen Qwen3-4B + LoRA（rank 16），只训 LoRA。
- **Carrier**：
  - *Determinate*：scene 唯一确定词义的 yes/no 题，gold 确定（gen_test 146 条）。
  - *Ambiguous*：中性 scene + 针对义 a 的 yes/no 题，target p(yes)=0.5（target 由外部枚举的两个词义 50/50 给出）；16 词 × 4 scene = 64 条，按词留出约 1/3。
- **Metric**：原生 `p(yes)=sigmoid(logit_yes − logit_no)`，不加外接解码。Determinate 报 accuracy；ambiguous 报 |p(yes)−0.5| 与 entropy；通用文本报 perplexity（×base）。
- **Loss（decoupled）**：margin hinge（只保排序）+ `(p(yes)−0.5)²` calibration + KL-to-base fluency leash；`λ_shape` 控制 calibration 项权重。
- **控制组**：coupled（determinate 用 one-hot CE、ambiguous 无信号）、shuffle（打乱 determinate gold）。
- **Sweep**：decoupled `λ_shape ∈ {1, 3, 10, 20, 40, 80}` × 3 seed；coupled、shuffle 各 3 seed。训 8 epoch。
- **是否进入原生前向（in-the-forward 检验）**：在加了 adapter 的同一前向上，额外训一个 layer-swept 线性 readout 去预测 gold yes/no（在 val 选层），比较 readout accuracy 与 native accuracy（同一批 gen_test）。frozen base 作对照（native ≈0.53，readout ≈0.86）。

## Pre-registered prediction & KILL

**预测**

- 同等 discrimination 下，decoupled 的 ambiguous 校准优于 coupled，配对 bootstrap CI 排除 0，且 ≥2/3 seed 胜（中等信心：早期小样本的优势较薄）。
- 进前向：在加了 adapter 的模型上，外接 readout 比 native 多恢复 ≤ +0.03（中等信心：也可能部分外置，约 +0.1）。
- shuffle 控制：determinate accuracy ≈ chance（bootstrap CI 含 0.5）。

**判据（跑前写定）**

- *同等 discrimination 胜出*：取均值 determinate accuracy 首次 ≥ coupled 的最小 `λ_shape`（即 decoupled 至少和 coupled 一样自信）。在该 `λ_shape`、对 held-out ambiguous 题做配对 bootstrap，Δ = |p−0.5|_coupled − |p−0.5|_decoupled。**PASS if** CI(Δ) 在 ≥2/3 seed 上排除 0 且偏向 decoupled。
- *进前向*：在加了 adapter 的模型上，readout accuracy − native accuracy ≤ +0.03 且 CI 含 0 → 视为进入前向。
- *shuffle*：determinate accuracy 的 bootstrap CI 含 chance（0.5）。

- **PASS if**：decoupled 在同等 discrimination 下校准胜 coupled（CI 排除 0，≥2/3 seed），且在加了 adapter 的模型上外接 readout 多恢复 ≤ +0.03，且 shuffle 塌到约 chance。
- **KILL if**：
  - 在同等 discrimination 下 decoupled 不胜 coupled（CI 含 0 或偏向 coupled）——说明解耦相对 one-hot CE 没买到校准，前沿优势降级为"`λ_shape` 只是在用 determinate accuracy 换 calibration、CE 在同一条曲线上"；或
  - 在加了 adapter 的模型上外接 readout 多恢复 ≥ +0.10——说明形状没进前向，adapter 只是在探针处掩盖了一个仍然外置的能力。
