# VERDICT — countdown-decode (capstone: decoupled emission → executor-Φ → λ=0 trellis → pass@1)

## 1. 一句话结论

在一个真实的、可执行判定的推理任务(Countdown / 24-game,SymPy-free 精确执行器作 Φ)上,**发射这条流水线确实跑通、且非退化**:decoupled 训练的模型采 N=16 → 执行器 Φ 按意义合并成 canonical 状态 → decode_core 的 λ=0 发射加权 Viterbi,parse 率 0.999、trellis 把 ~89 条别名路径合并到 ~17、把 greedy 的 pass@1 0.08 抬到 savi 0.38、并在**候选数对齐**口径下配对胜过 self-consistency(+0.25/+0.31,CI 不含 0)。**但在诚实的 token 对齐口径下,这个优势消失并反转**:把 selection 基线(best_of_k,采整链 + 执行器判定)给到同等 token 预算,**best_of_64 以不到一半的 token 拿到 0.81,远超 savi 的 0.44**(配对 −0.375,CI[−0.625,−0.125])。

**所以:λ=0 的纯发射 trellis 不足以在等算力下胜过"采样 + 执行器判定"。** 发射(decoupled 训练 + Φ 合并)是**必要**的——它让候选非退化、让 trellis 良定义、并实打实提升了 coverage/SC;但在自验证任务上,**值钱的是"判定"这一步,而 λ=0 把它丢掉了**(它按发射频率选路,不按正确性)。让 trellis 真正赢过 selection,需要 λ>0 的 verifier 给路径打分——那是姊妹线 `verifier-decode` 的区域(其结论:moves 对齐下全局解码胜 selection、且差距随深度增大)。本结论与 `../DECODING_MODEL.md` §4 完全一致:λ=0 只够到 SC 那一档,+0.18 那半归 verifier。

## 2. 设置

- **域**:Countdown(状态 = 可用值的多重集 + 目标;一步合并两个值;canon = (排序值, 目标),**每步严格收缩**,无复现无爆炸,goal 可经 step-trellis 到达)。这是 `decode_core` 的原生域;是从 M2 algebra **pivot** 来的——algebra 的 Φ(lhs−rhs 标量类)让简化步 canon 不变、trellis 退化(见 `../algebra-decode/FINDING_phi_mismatch.md`)。
- **模型/训练**:frozen Qwen3-4B + LoRA(rank 16),plain SFT(completion-only)。**数据级解耦**:coupled = 每状态一个保目标 move;decoupled = 在保目标 move 上均匀采多个(各 207 道可解训练实例 → 621 状态 → 621/1036 行)。fluency anchor on。
- **解码 arms**(`decode_core`,leaf-check = 终态单值 == target):greedy / self_consistency(N 链取众数 canonical 终态)/ **savi**(K=8, N=16, edge_mode="freq", **verifier=False = λ=0**)/ best_of_k(整链,任一执行到 target)/ oracle(绝对可解)。N=16, T=1.0, max_depth=10, seed=0。留出 61 道可解实例(seed 隔离)。
- **iso-compute**:Budget 账本(tokens/candidates/exec)逐 arm。token 对齐的 selection 基线 = best_of_k 在 K=64/128(token ≈ savi 的 1158)。

## 3. 结果

### 主表(留出 61 道,leaf-checked,配对)

| arm | decoupled | coupled |
|---|---|---|
| greedy | 0.082 | 0.066 |
| self_consistency | 0.131 | 0.033 |
| **savi (λ=0)** | **0.377** | **0.344** |
| best_of_16 | 0.328 | 0.262 |
| oracle | 1.000 | 1.000 |
| **savi − SC(配对)** | **+0.246** [+0.115,+0.377],McNemar 18:3 | **+0.311** [+0.197,+0.426],McNemar 19:0 |
| coverage / parse | 0.44 / 0.999 | 0.36 / 0.999 |
| trellis 宽 before→after merge | 89.1 → 16.8 | 83.6 → 14.7 |

- **候选数对齐下主判定成立**:savi 配对胜 SC,两 backend 都显著。Φ 合并真实(~5× 别名路径被并)、parse 近满。
- **次判定不成立**:decoupled-savi vs coupled-savi = **+0.033 [−0.115,+0.180],n.s.**。decoupled 的优势落在更便宜的聚合上(SC 0.131 vs 0.033、coverage 0.44 vs 0.36、best_of_16 0.328 vs 0.262),但 trellis 把训练目标的差异抹平了。

### iso-compute(decoupled,同 16 道,token 对齐)

| 预算轴 | arm | pass@1 | tokens |
|---|---|---|---|
| 候选 N=16 | savi (λ=0) | 0.438 | 1158 |
| 候选 K=16 | best_of_16 | 0.562 | ~110 |
| **token 对齐 K=64** | **best_of_64** | **0.812** | 446 |
| token 对齐 K=128 | best_of_128 | 0.812 | 891 |

- **配对 savi − best_of_64 = −0.375 [−0.625,−0.125]**:savi 用更多 token 输得更惨。
- 同 16 道上,**even 候选数对齐**,best_of_16(0.562)已 ≥ savi(0.438) 且 token 少 ~10×。全 61 道上 savi 0.377 > best_of_16 0.328 那点窄胜,**既是 10× token 代价,也不随 K 放大而保留**。

## 4. 解读:为什么 selection 赢,以及发射的价值在哪

- **best_of_k 在自验证任务上自带"判定"**:采 K 条整链,执行器确认哪条命中 target 就接受——这是合法、可部署的基线,且把执行器当成整链 verifier 用。**savi 在 λ=0 不用这个信号**:它按发射频率选路,而非按正确性。所以 best_of_k 拿到了 savi 丢掉的那半余量。
- **这正是 `DECODING_MODEL.md` 的论点被钉死**:λ=0 = 按发射的 marginal/SC 档选择,在等 token 下打不过"采样+判定";路径优势是 λ>0(verifier 给路径打分、让合并的已验证前缀跨链共享)才有的现象——姊妹线在 moves 对齐下确证、且差距随深度增大。Countdown 只 3 层,深度浅,即便如此 λ=0 也明显落后。
- **发射(decoupled)的价值是真的,但落点不同**:它提升 coverage(0.44 vs 0.36)与 SC——也就是提升"采样+判定"这条基线本身能吃到的东西。**发射让候选更好;但消费这些候选、真正赢过 selection 的,是 verifier-scored 解码(λ>0),不是 λ=0 trellis。**

## 5. 成立条件与边界

- 结论限定:Countdown(4 数,3 层,自验证执行器 Φ);N=16/K=8/T=1.0/单 seed;iso-compute 子集 16 道(配对 CI 不含 0,但绝对值有子集噪声);token 作 iso 轴(非 candidates)。
- **未否定的**:发射流水线跑通、非退化、parse 近满、Φ 合并真实、savi 在候选数对齐胜 SC——这些都成立。**被否定的**:λ=0 纯发射 trellis 在 iso-token 下胜过 selection(不成立)。
- **未覆盖**:λ>0(verifier mask / 打分)下的 savi —— 这是把本线与姊妹线合起来的实验,预测会翻盘(姊妹线已在 verifier 侧确证);更深的任务(层数 >3)token 对齐差距可能不同;模型 scale 提升 coverage 会抬两边天花板。

## 6. 对 scorecard / DECODING_MODEL 的更新建议(master 据此改表)

- row(iii)/发射侧:**"真合并 Φ 上非退化 + 候选可信"已在真实可执行任务上确证(parse 0.999、merge 89→17、savi 候选对齐胜 SC);但 λ=0 纯发射解码在 iso-token 下不敌"采样+执行器判定",值钱的是判定。** 这把"发射 front-end 是 SAVI 有效可训练组件"精确定位为:**发射提供良定义、可信、coverage 更高的候选(必要);全局解码的 pass@1 胜势来自 verifier-scored 路径(λ>0,姊妹线),而非 λ=0 发射选路。**
- 直接的下一步(合线实验):在本 Countdown 域上开 **λ>0 的 savi(verifier=True / 执行器路径打分)**,与 token 对齐的 best_of_k 比——这是把发射线与 verifier 线合起来、检验"trellis 在 iso-token 下胜 selection 且差距随深度增大"在 λ>0 下是否成立的最小实验。
