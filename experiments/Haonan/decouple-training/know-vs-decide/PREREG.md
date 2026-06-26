# PREREG — KNOW（信念）与 DECIDE（承诺）能否分别被读出与控制（跑前登记）

把当前 output 分布焊在一起的两件活拆开：**KNOW** = 对各候选义的信念（排序 / 校准 / 在该多峰时多峰），**DECIDE** = 承诺到其中一个（应是单独、靠后、可改的一步）。下面三条预测在跑之前写定，全部在冻结 Qwen3-4B 上做（不训练 backbone，只在中间层隐状态上训一个很小的线性打分头），carrier 为 bank/ATM 式语境消歧（平衡 matched-pair 切片 n=146，按词留出约 1/3）。

## Hypothesis

正确答案在模型回答那一遍前向里本就可从中间层读出，原生 greedy 之所以错是承诺太早；并且"承诺多确信"（calibration）的旋钮与"判别对不对"（discrimination）的旋钮可以分别控制、判别不必依赖把概率推尖。

## 设置（简短）

- KNOW = 从每个义的语境读出隐状态，用一个 per-candidate 共享权重的线性头给两个义各打一个分。
- DECIDE = yes iff KNOW 选的义 == 问题问的义，在读出之后晚一步、带一个可调 temperature T 做。
- 对照：训练 KNOW 时用纯排序损失（margin hinge，不压幅值）对比 cross-entropy；承诺时扫 temperature T。
- 模型原生前向（greedy 承诺）的作答正确率作为下界。

## Pre-registered prediction & KILL

**P1 —— 两轴可分（判别与承诺是两个独立旋钮）。**
- PASS if 纯排序损失训出的判别准确率落在 cross-entropy 的 ±0.03 内（区分大体不需尖度压力），且扫 temperature T 时作答 accuracy/AUC 基本不动（波动 < 0.01）而 ECE 有明显跨度（≥ 0.05）。
- KILL if 纯排序损失训出的判别显著低于 cross-entropy —— 则区分确实要靠尖度压力、训练上不可分。
- 诚实注：accuracy/AUC 对单调 temperature 不变是构造保证（解耦设计的必然），非平凡内容是排序追平 CE 判别 + ECE 跨度存在。

**P3 —— 晚承诺胜过原生 greedy。**
- PASS if 读出义 + 晚承诺的作答准确率远高于原生前向的 0.527（逼近读出义本身的准确率约 0.9），且配对 bootstrap 95% CI 下界排除 0。
- KILL if 晚承诺 ≈ 原生前向 —— 则这不是"答案已在前向只是没被用上"，KNOW 没有多出可兑现的东西。
