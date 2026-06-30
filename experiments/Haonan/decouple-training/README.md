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

## 第三轮:把 verifier 打开(λ>0),给出完整当前状态(报告 `REPORT_round3.md`)

trellis 解码 = 发射 × verifier(`DECODING_MODEL.md`)。上一轮停在 λ=0;这一轮把 λ>0 推到头,核心问题:等 token(可执行底物上为等 exec)的诚实口径下,全局解码能否解出采样基线(best-of-K + 判定)解不出的题——从"不能"到"能"?**结论:instance 层面没有这种效果;只有两个窄正面幸存(path 级重组、低能力 soft-value),且都不是 instance 级胜势。** 完整设置/边界见报告。底物各异(真实 Qwen3-4B Countdown / 合成 merge-lattice(无 LM、等 exec)/ 真实 PRM ProcessBench),逐行标注。

| 报告位置 | 论断(plain) | 关键数字 | 目录 / 文件 |
|---|---|---|---|
| §1 | 域无关 trellis 解码器(发射 log(count/N) × verifier 硬 mask × Φ 合并 × backtrack),所有解码实验共用 | —(纯库;§1 实现引用) | `verifier-decode/decode_core/`(`decode.py` 的 `savi()`) |
| §3.1 | λ=0 全局解码胜 greedy、候选对齐胜 self-consistency,但 token 对齐输给采样+验证 | savi 0.38(配对胜 SC +0.246);best-of-64 0.81@445tok vs savi 0.44@1158tok,配对 −0.375;独解 0 道 | `countdown-decode/`(`results.json`) |
| §3.2 | λ>0 看似翻盘,优势全来自逐步 exact mask、trellis 部件 inert;换学习 value 即反输 | savi−masked_bom = 0/+0.09/0;learned value soft **−0.583**、hard −0.833 vs exact +0.167(k=6) | `verifier-decode/joint-lambda-decode/`(`outputs/t11_*.json`,`t14_value_k6.json`) |
| §5.1 | 能力翻转:低能力区 soft 降权学习 value 胜等 exec selection,未触顶;硬 mask 在边界误差下死、soft 活 | p=0.5 **+0.29…+0.34**(3 模型 CI>0,pass1 .71–.76);无 verifier −0.125;硬 mask +0.354→−0.047 vs soft +0.245→+0.214 | `verifier-decode/merge-noise-phasemap/`(`outputs/phasemap/learned_p0.5.json`、`VERDICT_boundary_transfer.md`) |
| §5.2 | path 级重组在合成 lattice 成立(verifier 门控),但正确路径多 → 升不到 instance 级(效率) | 正确路 **q≈10⁻¹¹**(log₁₀1/q=11.45),stitch 23–36%,无 verifier onlyDec 0.01;同题 ~10³·⁴–10⁴·² rollout 可达 | `verifier-decode/recombination-novelpath/`(`outputs/sweep.json`) |
| §5.2 | 重组不迁移真实 AR:拼接处处为 0;唯一"选择不能"效应来自深档精确 oracle 引导搜索(非重组,学习难复制) | builtin 28:stitch=0、稀疏 exact 0/4;gen k=6 24:stitch=0、稀疏 exact only_decode **3/3**、学习 1/3 | `verifier-decode/mass-harness/`(`outputs/builtin_K64.json`,`gen_k6_K64.json`) |
| §5.3 | 真实 PRM 与 7B critic 误判都集中在 on/off-path 交界(~3.5×),解释"软降权成立、硬剪枝失效" | PRM d=−1 **0.234** vs d≤−5 **0.066**(=3.53×);critic 0.078 vs 0.023(=3.43×) | `verifier-decode/prm-boundary-calibration/`(`results/prm_verdict.json`,`compare.json`) |
| §5.4 | 负对照:有利相关 AR 集成无免费几何相关类比,coverage 天花板=基模型+token | cond−iid @K8 **−0.250**(Holm p=1.0),3.54× token | `correlated-coverage/`(`realprobe/results.json`) |

> 注:§1、§3.2、§5 的实验目录(`verifier-decode/*`)已与本目录合并到同一可复现子集;路径相对 `contrib-savi/experiments/Haonan/`。早期 round-1/2 目录见上文表。

## 目录结构与复现

- 四个主目录（`know-vs-decide` / `know-decide-construct` / `controllable-posterior` / `latent-consistency-decode`）各含 `VERDICT.md`（结论 + 设置 + 结果 + 成立条件）、`PREREG.md`（跑前的 hypothesis + KILL）、`core/`（最小可复现脚本）、`results.json`（结果数值源文件）。
- `marginal-vs-path/` 与 `native-abstain/` 是支撑性证据，含 `VERDICT.md` + `PREREG.md` + `results.json`，不含复现脚本。
- 每个 `core/` 自包含：`latent-consistency-decode/core/` 把跨模块依赖 vendored 到 `core/_deps/`（无需任何外部目录）。复现按各目录 `core/README.md` 的两步说明：先用一次 GPU 生成特征缓存 / 训练 adapter（缓存与 adapter 体积大，不随本目录提交，需本地生成），再用 CPU 跑分析脚本写出结果；`results.json` 为对照参考。模型权重默认从本地 Hugging Face 缓存离线加载（`cache_dir` 读环境变量，未设则用 transformers 默认缓存）。
