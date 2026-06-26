# VERDICT — latent-consistency-decode

## 一句话结论

当约束正确时，global decoding 的解码机制本身可用：在一类 per-slot greedy 全错的多变量一致性题上，喂入真约束（oracle edge）后 joint exact-match 从 0.00 升到 marginal 0.88 / viterbi 0.92；而且 calibrated 的逐 slot emission 是这一恢复的必要条件（decoupled 0.88 远高于未训练的 base 0.44）。瓶颈不在解码器，而在约束来源：让语言模型自己判断每条约束，区分"真一致 / 真不一致"的 AUC 只到约 0.67（TinyLlama-1.1B 0.475、Qwen3-4B 0.669，1B–8B 与多种 wording 都到不了更高），把这些判断合成整道题后 decode 仅 0.04–0.08。

## 设置

- **Model / emission.** frozen Qwen3-4B 加一个小 LoRA（rank 16），只训 LoRA。每个 slot 的 emission（= 节点的 unary potential）取答案位置上候选 option 的 softmax 分布。三种 emission：
  - `decoupled`：训练目标是一个 calibrated soft target（保留正确 option 的概率，不塌到局部线索偏好的那个 option）；
  - `coupled`：one-hot cross-entropy，把概率压到单个 option；
  - `base`：不训练，直接读未训练模型的答案位置分布。
- **Carrier（多变量一致性题）.** 每题 T 个 slot（T∈{3,4}），每 slot k 个 option（k∈{3,4}，option 为内容中性的随机名词，无先验偏好）。slot 间有 same / different 约束；每题植入一处局部-全局冲突：只看每个 slot 的局部线索会得到一个全局不一致的解，必须用上约束才对。joint MAP（全局唯一一致解）可精确算出，作为 gold。共 960 题，其中 227 题留出。
- **Edge（约束）来源，共四种.** 每条约束化成一个 (k,k) 的 pairwise log-potential：
  - `oracle`：植入题里的真 hard 约束矩阵（上限对照）；
  - `self`：Qwen3-4B 对自己生成的逐对 yes/no 一致性 query 打分（边权 = logp(yes) − logp(no)）；
  - `cheap`：一个更小、与生成模型 decorrelated 的模型（TinyLlama-1.1B）做同样的逐对 query；
  - `shuffle`：把 oracle 的 hard 约束矩阵跨题打乱（破坏信号的随机 hard 约束对照）。
- **Decoder，三种.** `greedy`（逐 slot 取 unary argmax，忽略约束）；`marginal`（在 hard 约束下逐 slot 取精确后验 marginal 的 argmax，sum-product）；`viterbi`（hard 约束下的精确 MAP，max-product）。
- **Metric.** joint exact-match（整题全部 slot 与 gold 一致才记对）；逐 slot accuracy；item-bootstrap CI。
- **报告范围.** 下面的解码数字来自一个 de-risk 子集：单 seed、cells T=3 k=3 C=2、25 题留出、训练 2–4 epoch；不是完整 (T,k,C)×3-seed 网格。数值见 `results.json`。

## 结果

**(1) 约束正确时解码机制可用，且需要 calibrated emission（oracle edge，decoupled emission）。** per-slot greedy 在这批题上 0/25 全错（carrier 即按此构造）。喂入真约束后大部分被恢复：

| edge 来源 | greedy | marginal (sum-product) | viterbi (max-product) |
|---|---|---|---|
| oracle（真约束） | 0.00 | 0.88 | 0.92 |
| self（Qwen3-4B 判自己的 query） | 0.00 | 0.04 | 0.04 |
| cheap（TinyLlama-1.1B，decorrelated） | 0.00 | 0.00 | 0.00 |
| shuffle（随机 hard 约束） | 0.00 | 0.28 | 0.28 |

**(2) emission 一侧：保留正确 option 的概率是恢复的必要条件（oracle edge）。**

| emission | greedy | marginal / viterbi（oracle） |
|---|---|---|
| decoupled（soft target，正确 option 仍有概率） | 0.00 | 0.88 |
| base（未训练读出） | 0.12 | 0.44 |
| coupled（one-hot 塌缩） | 0.20 | 0.24 |

干净的对照是 decoupled 0.88 远高于 base 0.44：把答案位置的分布训练成一个 calibrated 的局部目标（让正确 option 的概率不为零），相比未训练读出，把真约束之后能恢复的比例大致翻倍。

**coupled 这一格有 confound，需注明：** 这里的 coupled 是 one-hot 训向 gold option，但在被植入冲突的那个 slot 上，gold 恰恰无法从该 slot 的局部 prompt 单独确定（这正是需要全局约束的地方）。所以 coupled 的低值（0.24）同时混入了两件事——分布塌缩，以及一个本就学不到的目标。"在信息量相当的前提下比较铺开 vs 塌缩"的干净对照应当是 one-hot 训向 argmax(local_unary)，而不是训向 gold；该对照尚未重跑。

**(3) viterbi 与 marginal 的差异很小。** decoupled emission + oracle edge 下，viterbi 0.92 对 marginal 0.88（+0.04）。相对 greedy 的提升来自"用上了约束"（globality），而非"取整体单条最优路径"这一形式。

**(4) 约束来源是瓶颈：模型判约束的可靠度有上限。** 直接探针：对每个 option 对，模型的 logp(yes) − logp(no) margin 能否把真一致对排在真不一致对前面（AUC）？

| model | AUC |
|---|---|
| TinyLlama-1.1B | 0.475（≈ 随机） |
| Qwen3-4B | 0.669 |

TinyLlama-1.1B 无论一对 option 是否满足约束，都以相近的置信度说"一致"——没有信号；query 格式本身没问题，因为 Qwen3-4B 在完全相同的 prompt 上能取出方向正确的信号（AUC 约 0.67）。换 wording 或换更大模型也清不掉这道墙：扫了三种 elicitation（M0 当前 query / M1 符号化重述 / M2 few-shot M1）× 多个 decorrelated 模型，1B–8B 与各种措辞的最高都在约 0.67，且 reframe / few-shot 提升不稳定（对 3B 的 M2 甚至把 margin 方向反转）。要定出整道题需把多条逐对判断合成一个联合约束，单格判错就会翻整题，因此即便用最好的来源，合成后 decode 仅 0.04–0.08；同一批题、同一 emission 上的 oracle 仍是 0.88 / 0.92。

## 成立条件与边界

- **解码机制"可用"的条件：约束正确，且 emission 保留了正确 option 的概率。** oracle edge 把 0.00 推到 0.88–0.92 成立的前提是 (a) carrier 确有可被全局解码利用的跨 slot 耦合（预注册的"无耦合"kill 未触发），以及 (b) emission 是 calibrated 的 decoupled 读出。换成 base 读出（0.44）或塌缩的 coupled 读出（0.24，且带上面的 confound），同样的真约束能恢复的就少很多。
- **模型自判约束"不可用"的条件：逐对一致性判断的 AUC 停在约 0.67，且要合成多条约束。** 这一上限在 1B–8B 与多种 wording 下都成立；约 2/3 的逐格正确率经联合合成（C=2）后不足以存活（一格反号即翻整题），decode 因此落到 0.04–0.08。条件之一是"题面把约束直接以文字写出，但 elicitation 让模型逐配置去判断每一格"——这两件事难度不同：读一条已写明的约束接近平凡的 parsing，而逐配置判断才是约 0.67 的墙；若改成"每条约束只问一次、取一个 bit 的关系再用代码施加"，在这种约束写明的 carrier 上更接近读取，而非判断。真正困难、也最相关的，是约束未写明、需从世界知识判断的情形，那属于约 0.67 这一侧。
- **viterbi 与 marginal 的差异条件。** 在这批题上 viterbi 仅比 marginal 高约 0.04；相对 greedy 的增益来自用上约束，而非单条最优路径的形式。只有在对抗性构造的特例下两者才明显分开，这批真实 emission 不落在那里。
- **尚未确立的部分。** 完整 (T,k,C) 网格与多 seed（当前为单 seed、单 cell-group）；coupled 的干净塌缩对照（需用 argmax(local) 目标重跑）；是否存在某个更大的 decorrelated 模型或更好的 elicitation 把 edge AUC 抬过可用线——Qwen3-4B 方向正确的 0.67 说明信号在足够 capacity 下存在，只是单次、带噪的 soft margin 不够干净，且与 emission 同源时既有干扰又冗余。
- **方法上的一点.** 这套"真一致 vs 真不一致"的 AUC 探针，把"约束太弱（scale 问题）"与"约束没有信号（无法 parse）"区分开了——它显示 cheap 失败属于后者（margin 又平又小），不是单纯的规模问题。

数值见 `results.json`。复现脚本见 `core/`。
