# VERDICT — state-emission(校准能否传到 Φ 合并后的状态分布)

## 1. 一句话结论

换训练目标(one-hot CE → decoupled)之后,模型在**自由文本生成**里、经一个真正按意义合并的 Φ 之后,得到的状态分布**确实比 one-hot 明显不退化**——在每个歧义档(j=2…5)上,decoupled 采样覆盖的幸存意义更多、有效状态数更高、到真值的距离更小;**但"非退化"成立的同时"校准"只部分到位**:读出层近乎完美(到真值 ≈ 0.02),自由生成层却差 0.3–0.5,而且 decoupled 把不确定性大量转成**弃答**(结构化承诺格式里约 1/3 的样本不承诺)。所以从读出层到生成层的传递是**真实但有损的**:解码端能拿到多样、可区分的候选(这是发射要的),但它们的相对权重还不可信。

## 2. 设置

- Model:frozen Qwen3-4B + 小 LoRA(rank 16,oracle 模式,seed 0),四组只差训练目标:base(无 adapter)、coupled(one-hot CE)、decoupled(soft-target)、shuffle(打乱标签的对照)。在本目录新训于本任务的 `items.json`(与上一阶段的消元任务同构,prompt_stated/target/split 一致;额外带两种作答视图)。
- Carrier(构造的消元任务):消元谜题,k 个选项消到 j 个幸存,真值 = 在 j 个幸存上均匀(精确已知 → 校准可量)。同一批题两种生成视图:**结构化承诺(COMMIT 行)**(自由推理 + 末行 `COMMIT: <东西>`,Φ 确定性、低噪)与 **完全自由文本(独立模型判读)**(完全自由作答、无槽,Φ = 去相关的外部 Llama-3.1-8B 模型判读,权威)。
- 测量:每题从 canonical state 采 N=16 条(T=1.0),Φ 映成状态;量 coverage(覆盖幸存)、calibration-TV(到均匀真值)、k_eff、abstain,**并排**读答案位置读出分布的 readout-TV,gap = calibration-TV − readout-TV。留出集,item-level bootstrap CI。全量 73 分钟。

## 3. 结果

### 结构化承诺(确定性 COMMIT-slot Φ)——decoupled vs coupled,按 j

| 指标(j=2/3/4/5) | decoupled | coupled |
|---|---|---|
| coverage(覆盖幸存) | **0.85 / 0.72 / 0.68 / 0.56** | 0.66 / 0.54 / 0.41 / 0.38 |
| k_eff(不同状态数) | **1.80 / 2.27 / 2.70 / 2.80** | 1.32 / 1.63 / 1.70 / 1.90 |
| calibration-TV(越小越准) | **0.31 / 0.46 / 0.49 / 0.52** | 0.41 / 0.53 / 0.67 / 0.68 |
| readout-TV(读出到真值) | ≈ 0.02(各 j) | 0.50 / 0.67 / 0.75 / 0.80(= 1−1/j,精确塌) |
| gap = gen−readout | **+0.30 → +0.50** | 小(读出本就塌) |
| abstain(不承诺率) | **0.28 → 0.39** | 0.00 |

- **decoupled 的自由生成在每个 j≥2 上都比 one-hot 铺得更开、更准**:覆盖、k_eff、calibration-TV 三项全胜。one-hot 的读出精确 = 1−1/j(全压一个选项),生成也跟着塌。→ **"塌的是目标不是容量"从读出延伸到了真实采样,而且在 Φ 要按意义解析 COMMIT 的情况下仍成立。**
- **但两个具名的失效**:(a) **校准从读出到生成大幅衰减**——decoupled 读出到真值 ≈0.02,生成却差 0.3–0.5(gap 随 j 增大)。方向/排序对,但**比例不可信**。(b) **decoupled 把不确定性转成弃答**(结构化承诺格式约 1/3 样本不承诺);这或许是"诚实的不确定",但不是解码端想要的可比较候选。
- **覆盖单独分不开 decoupled 与 shuffle**:shuffle 的 coverage(j2=0.85, j3=0.81)与 decoupled 相当,但 shuffle 读出不跟真值(确定题 readout-TV=0.65,错的)。**干净的区分量是"读出跟不跟真值",不是覆盖。**

### 完全自由文本(外部 Llama 独立模型判读,权威)——只判到 j≤2

| j=2 | coverage | calibration-TV | k_eff |
|---|---|---|---|
| decoupled | **1.00** | **0.204** | **2.20** |
| coupled | 0.82 | 0.350 | 1.70 |
| base / shuffle | 0.65 / 0.77 | 0.44 / 0.34 | 1.30 / 1.75 |

- **decoupled 的优势进入真实自由文本(j=2):覆盖 1.00 对 one-hot 0.82,校准 0.20 对 0.35。** 排序在"按意义合并"下保住了。
- Φ-soundness 边界:判读假合并率 decoupled 0.113、coupled 0.036、shuffle 0.148、base 0.15——decoupled 在预登记的 0.15 阈值下(但不富裕);**独立模型判读只判到 j≤2**,j≥3 的自由文本仍是空白。

## 4. 成立条件与边界(对应预登记分支)

- **排除了"结构化承诺下就塌"这个强负结果**:decoupled 在自由推理里明确胜 one-hot。也**不是"完美校准"**:读出→生成 gap 0.02→0.3–0.5 是量化出来的失效。
- 因此落在**中间档:传递真实但有损**。gap 中等(非塌、也非准),指向**轻量的生成级校准杠杆**(覆盖/熵正则或序列级 shape),**不需要**最重的 Φ-pushforward RL。
- 本结论限定在:真值由构造精确已知的消元载体;Φ 在结构化承诺下是确定性 COMMIT 解析、在完全自由文本下是单一外部模型判读(假合并 ~0.11);N=16、T=1.0、单温度。**未覆盖**:完全自由文本的 j≥3(判读子集没够到)、温度/样本数稳健性(见 `../sampling-robustness/`)、纯改写对下的别名(`../markov-aliasing/` 的 stated-clue 代理被推断难度混淆,见其 README)。

## 5. 旁证(并行小实验,辅助解读)

- **真实数学有可选择余量**(`../realtask-headroom/`,GSM8K base 模型):greedy 0.735、自一致 0.825、oracle-best-of-16 0.915 → **可选择余量 +0.18**、自一致增益 +0.09,k_eff 2.77、parse 1.00。说明 GSM8K 不是 EvalPlus 那种余量≈0 的域,是可用候选域(MATH-500 因 token 预算未出可靠数)。
- **语义别名 / Markov 探测**(`../markov-aliasing/`):stated-vs-clue 的 JS 被"自我推断难度 + oracle-训练-adapter-测 clue 的 OOD"混淆,**不能干净隔离别名**(decoupled JS 偏高是因其 stated 读出极尖 0.018、clue 落后 0.231,非别名更重)。需要纯改写对才能测;当前 stated-clue 代理作废为别名证据。
- **N×T 稳健性 + ECE**(`../sampling-robustness/`):base ECE_determinate=0.517(未校准);decoupled/coupled 的 ECE 与 N 曲线见该目录。
