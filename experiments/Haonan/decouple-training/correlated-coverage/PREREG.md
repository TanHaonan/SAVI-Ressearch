# PREREG — Correlated-coverage: 用相关采样把 i.i.d. best-of-K 的覆盖天花板顶上去

**Status: SPEC(planning only;未执行)。** 这是 emission 线实验。它把组合数学的"概率方法"
(Erdős 1947 → Ma–Shen–Xie 2025 的指数改进)抽象成我们的 P′,并提出一个**构造性方法 S′**
(相关 / 互斥采样)去检验。decode_core 保持 FROZEN;新代码在本目录。iso-compute 轴 = tokens。

---

## 0. 一句话问题

把 K 条独立采样的链(best-of-K = i.i.d. 概率方法 = Shannon 随机码)换成**带"恰当负相关"的
K 条链(构造性 S′)**,在**等 token 预算**下,能不能把**覆盖率 coverage@K 的指数衰减率(= 我们的
"底数")顶上去**——即让"K 条全 miss"的概率比 i.i.d. 的乘积率 (1−c)^K 掉得更快?并且这个增益
是否服从一个由相关强度旋钮 β 控制的"倒 U"(太独立=冗余坍缩,太互斥=被推进死区),正如 sphere
graph 的维度旋钮 k≈ℓ²?

---

## 1. 背景:概率方法,以及它八十年来的天花板被怎么顶破的

**Erdős 1947(概率方法的原型).** 要证"存在一个 Kₙ 的二染色,不含 size-k 单色团"。Erdős 不去
构造,而是**随机均匀染色**,用 union bound:P(出现某个单色 k-团) < 1 当 n < √2^k,故"好染色一定
存在"。这给出对角 Ramsey 下界 R(k,k) > (1+o(1))(k/(e√2))·2^{k/2},**底数 √2 ≈ 1.41**。
关键:**好对象的"质量"= 指数的底数**(n 能相对 k 多大)。

**八十年的天花板.** i.i.d.(Erdős–Rényi G(n,p))模型下,单色团概率**精确**等于 p^{C(r,2)} /
(1−p)^{C(r,2)};对 union bound 优化 p,n,**底数可证就卡在 M_C 上**。78 年里**只有前因子**被改进
(Spencer 1975 用 Lovász Local Lemma),**底数从未被推动**。

**指数改进(Ma–Shen–Xie 2025, arXiv:2507.12926, USTC+清华 YMSC).**
- 对象:**off-diagonal** r(ℓ,Cℓ),C>1(**不是**对角 R(k,k))。Erdős 下界底数 M_C = p_C^{−1/2}
  (C=1 时即 √2)。
- 结果:r(ℓ,Cℓ) ≥ **(M_C + ε)^ℓ**,ε>0(虽小,是二阶修正)。**首次推动了底数本身**;对角 √2 仍
  未被改进。注:坊间"1.41→4"的说法**不准**——4^ℓ 是无关的**上界**(Erdős–Szekeres),本文动的是
  **下界的底数**,意义在于"**i.i.d. 天花板第一次被顶破**",不在量级。
- 引擎:把 i.i.d. 染色换成**相关系综——"random sphere graph"**:n 个点均匀采于单位球 S^k,边
  按 ⟨xᵢ,xⱼ⟩ 的符号染色(标定到每边红概率 = p)。边是**几何相关**的,且相关方向**有利**:红团
  概率被**严格压到独立乘积率 p^{C(r,2)} 之下**,同时**竞争的**蓝团概率仍受控。**一个连续旋钮——
  维度 k**——在"k→∞=独立(无增益)"与"k 太小=过相关(另一侧爆)"之间插值,**甜点在 k≈ℓ²**。
  增益 ε = union-bound 不等式里的**严格余量**(两个底率之和 < 1)。

**Shannon 1948 = 同一招.** 信道编码定理用**随机码系综**证明"达容量的好码一定存在"(非构造)。
之后几十年的编码学是去**构造**逼近这个随机界的码(Turbo/LDPC/Polar)。

**抽出来的母题(你给的 taxonomy 第 1+4 类):** 造一个大随机对象,使坏子结构不出现(或同时回避两
个竞争的坏结构);**顶破 i.i.d. 天花板的办法,是把独立系综换成"有利相关"的系综 + 一个甜点旋钮。**

---

## 2. P→P′,S→S′(Shannon 式抽象)

| | 源(Ramsey/Shannon) | 我们(decode) |
|---|---|---|
| 问题 P/P′ | 存在一个大对象,回避坏子结构(单色团)。质量=指数**底数** | 采样系综里**存在**一条到达目标、回避死路的链。质量=**覆盖衰减率**(miss 概率随 K/token 掉多快) |
| 基线 | i.i.d. 染色(Erdős–Rényi),底数有**可证天花板** | i.i.d. 链采样 = **best-of-K** = Shannon 随机码,经验上**难顶** |
| 解 S/S′ | **相关系综**(sphere graph)+ 维度旋钮 k≈ℓ²,把坏事件压到**独立乘积率之下** → 底数↑ | **相关/互斥链采样** + 相关强度旋钮 β,把 P(K 条全 miss) 压到 **(1−c)^K 之下** → 覆盖底数↑ |
| 两个竞争坏结构 | 红团 vs 蓝团 | **坍缩**(链太相关→都掉同一个错区)vs **弥散**(链太互斥→被推进死区,够不到窄目标) |

**为什么这次能转移(而柯西不能):** 柯西的 load-bearing 条件是解析性,我们的设定可证没有。**这里
的 load-bearing 条件是"一个有正成功概率的随机系综 + 对坏事件的 union bound"——它在 best-of-K 里
精确成立**(coverage>0)。所以这是**同一套数学**(竞争坏事件的联合尾),不是双关。

**为什么这恰好解释我们过去最大的 null(τ=1 下 decoupled≈coupled):** 温度制造的是**边际**散开,
但 K 个样本仍≈**独立(i.i.d.)**,因此冗余、覆盖卡在 Erdős–Rényi 天花板。sphere-graph 的教训是
增益来自**样本之间的有利相关**,不是边际散开。**故 decouple 要起作用,目标应是"样本间的负相关
(joint anti-correlation)",不是逐步边际校准。** 本实验直接检验这一点。

---

## 3. 假设(可证伪;阈值见 §10)

- **H1(相关 > i.i.d.,等 token 下覆盖更高).** 在有头cover-room的实例上,等 token 预算 B 下,
  coverage_corr(β\*) − coverage_iid > 0,配对 bootstrap CI 排除 0。
- **H2(底数被顶起 / 衰减更快).** 把 log(1−coverage) 对 K(或对 token B)作图,**斜率即"底数"**;
  相关臂的衰减斜率 **严格陡于** i.i.d. 臂(CI 排除 0)。这是 Erdős→sphere-graph"底数 +ε"的直译。
- **H3(倒 U / 甜点旋钮).** coverage_corr(β) 关于 β **非单调**:从 β=0(=i.i.d.)升到峰再回落
  (过互斥坍塌),存在最优 β\*>0——sphere 维度旋钮的对应物。
- **H4(是"相关"不是"边际散开"——直接判我们旧 null).** 一个**温度对齐**的 i.i.d. 控制臂
  (把 τ 调高到其**边际熵 = 相关臂的边际熵**)**不能**复现增益:H1 对相关臂成立、对温度臂不成立。
- **H5(regime 依赖,接 verifier 线的 competence 轴).** 增益随 i.i.d. 基线覆盖**递减**:在基线覆盖
  中段(未饱和,= p≈0.5 / frequency⊥correctness)增益大;在基线覆盖饱和(p≈0.9 / 易,如浅 Countdown)
  增益→0。

---

## 4. 方法:构造性 S′ = 顺序-规范态互斥采样(canonical-state repulsion)

复用 decode_core 已有的 `sample()` 与 domain.canon(Φ)。**核心:让第 i 条链在生成时,避开前 1..i−1
条链已经走过的规范态。**

- 维护计数 n(S, t) = 已抽链中,在深度 t 经过规范后继态 S 的条数。
- 抽第 i 条链时,逐步从当前态出发,把该步的转移概率重加权为
  **P̃(S′|S) ∝ P_emission(S′|S) · exp(−β · n(S′, t))**,按 P̃ 取下一步;到目标/死路/深度上限停。
- **β=0 精确退化为 i.i.d.**(写一个等价性测试断言:β=0 时本臂的抽样分布逐位等于 best-of-K 的链分布)
  → 满足 V2 同源。β↑ → 链被推开。

**iso-token 诚实性(binding constraint,本线吃过的亏):** 逐步互斥需要 step-mode 采样(每步 N 个候选),
比 chain-mode 的 best-of-K 贵。**两条应对,都在 §7 计入 token:**
- (主)逐步互斥,**按真实 token 记账**,要求在**等 token**下仍 ≥ i.i.d.(硬测);
- (备)**prompt 级条件化**:chain-mode 便宜地抽,但在 prompt 里附上"已走过的 canonical 前缀,请走不同
  的解法"的简短条件(生成 token 数不变,只增廉价 prompt token)。便宜但依赖模型听话。
- Tier A 用显式系综**与机制成本无关地**证明"原理"(相关能否顶破 i.i.d.);Tier B 才回答"真实廉价
  机制能否在 iso-token 下兑现"。

---

## 5. Arms(全部消费同一 base emission;只换采样过程)

| arm | 做什么 | 角色 |
|---|---|---|
| `iid_bok` | i.i.d. 抽链到 token 预算 B,coverage = 任一到目标 | **headline 基线**(Erdős–Rényi / 随机码) |
| `corr(β)` | §4 互斥采样,β 扫描,到 token 预算 B | **the claim**(S′) |
| `temp_matched_iid` | i.i.d. 但把 τ 提到边际熵 = corr 臂的边际熵 | **关键负控**(H4:边际散开 ≠ 相关) |
| `oracle` | domain.solvable(s0) | 覆盖天花板 / 头cover-room 分子 |

**注意:本实验的 headline 指标是 coverage(存在性),不是 pass@1——刻意把 verifier 排除在方法之外**
(verifier 是 §14 的正交杠杆)。执行器只用于**评测**是否到达目标(ground-truth 打分,两臂共用),
不进入方法。这正是"为什么这是 emission 线该独占的实验":它只动系综/proposal,不碰已判死的 decoder。

---

## 6. 两 tier

- **Tier A(CPU mock,先冻 prereg).** 合成链生成器,带两个旋钮:每步**好步概率 p**(= verifier 线的
  competence 轴,刻意同名同构)与**模式坍缩度**。在此可做**精确** i.i.d. vs 精确互斥采样,扫
  **p × β**,验证 H2/H3 的倒 U、以及 H5(增益在中 p 大、高 p→0)。**这一步把 prereg 阈值冻死,且
  与机制成本无关地证明原理。** 不是 science,是 plumbing + 原理 + freeze。
- **Tier B(GPU,决定性).** 真模型 = frozen Qwen3-4B + LoRA(`adapter_decoupled_s0`,已在盘),更深
  Countdown 的**头cover-room实例**(k=5,6 → depth 4,5;选 i.i.d. coverage@K 落在 0.2–0.9 的题,见 V1)。
  同 harness、同 token 记账,跑 §5 四臂 + β 扫 + τ-match。这产出真 LM 数字。

---

## 7. iso-compute 协议(token,不是 candidates)

- 每臂记 Budget:`tokens`(Σ 生成续写 token)+ `forwards` + `exec`(只在评测覆盖时调 is_goal)。
  **headline 轴 = tokens。**
- 主比较:固定 token 预算 B 的网格,逐 B 比 coverage(配对,同实例同 B)。
- **"底数"度量(H2 的核心):** 对每臂拟合 log(1 − coverage(B)) 的斜率 = miss 概率的**衰减率**;
  报告 corr 与 iid 的斜率差(配对 bootstrap)。这是"底数 +ε"在我们这里的可测对应物。

---

## 8. 仪器(本里程碑新增)

1. **coverage–vs–token 曲线**(headline)+ miss 衰减斜率拟合(H2)。
2. **β 倒 U 扫描**(H3):coverage_corr(β) over β,定位 β\*。
3. **联合多样性表**:每个 K-系综里 distinct canonical 终态/路径数;corr 应在**联合**层显著高于
   iid 与 temp_matched(证明增益来自相关,不是边际散开)。用它给 H4 设定 τ-match(令 temp 臂的
   **边际**熵 = corr 臂的边际熵,再看联合多样性与覆盖是否仍输)。
4. **competence-p × β 热图**(Tier A,H5):增益面,接 verifier 线 phasemap 的同轴。
5. determinism / 同源:β=0 等价测试;process-stable seeds;exec-parity guard 逐臂断言。

---

## 9. 有效性闸(过不了则该 cell 无效,不算 null)

- **V1 头cover-room非空:** 所选实例上 i.i.d. coverage@K ∈ (0.2, 0.9)(太饱和=无可顶;太低=无可采)。
- **V2 同源 + β=0 等价:** corr(β=0) 的链分布逐位 == iid_bok;两臂消费同一 adapter / 同一 per-step
  emission。
- **V3 iso-token:** 各臂 token 预算逐实例匹配在容差内;corr 的逐步开销已计入。
- **V4 评测 oracle 干净:** is_goal/执行器对两臂一致;oracle ceiling > iid coverage(确有可顶空间)。

---

## 10. Prereg 阈值(DRAFT — Tier A 上冻结,再跑 Tier B)

- δ(H1):coverage_corr(β\*) − coverage_iid ≥ **+0.05**,iso-token,头cover-room实例,CI 排除 0。
- H2:miss 衰减斜率 corr − iid > 0,CI 排除 0。
- H3:存在 β\*>0 使 coverage_corr(β\*) > coverage_corr(0)+δ;且 β→大时回落(倒 U,非单调)。
- H4:temp_matched_iid − iid 的覆盖增益 ≈ 0(CI 含 0),**显著小于** corr 的增益。
- H5:增益对 i.i.d. 基线覆盖的回归斜率 < 0(Tier A 的 p 轴上单调)。
- 配对 bootstrap n=10000;H1/H2/H4 间 Holm 校正。所有 δ 在 Tier B 前冻结。

---

## 11. 这settles什么 / 不settles什么

**settles(若 H1–H4 在 Tier B 成立):** 在真 LM + 真可验证域,**等 token 下,带恰当负相关的采样系综
顶破了 i.i.d. best-of-K 的覆盖天花板**,且增益**来自样本间相关(非边际散开)**、服从甜点旋钮——
即把 Erdős→sphere-graph 的"相关系综顶破 i.i.d. 底数"在 LLM decoding 上构造性地实现一次。同时给
emission 线一个**与已判死的 decoder 无关、与 verifier 正交**的真实杠杆。

**不settles:** verifier/选择侧(刻意排除,见 §14);单域(Countdown);自由散文 emission;以及"相关
采样能否随规模/更大 K 继续放大"。Tier A 的 p 是理想旋钮,真 LM 的相关噪声可能让甜点变模糊(承自
verifier 线的同一 caveat)。

---

## 12. Kill criterion(预先写死)

若 corr 在其最优 β\* 下**仍不能**在 iso-token 上顶过 i.i.d.(H1 失败),则 Erdős/sphere-graph 的
转移对 LLM decode **只是隐喻**:i.i.d. 覆盖就是该模型/任务的天花板,LLM 样本的冗余在 iso-token 下
**不可被互斥采样兑现**。这是干净、可发表的负结果,且**闭合整条线**——它会说明覆盖只能靠"更强的
base 模型 / 更多 token",而不是更聪明的系综。

---

## 13. Deliverables(本目录;decode_core FROZEN)

`corr_sampler.py`(顺序-规范态互斥采样 + β;β=0 等价测试)、`mock_backend.py`(competence-p 合成
生成器 + 精确 i.i.d./互斥)、`run_corr.py`(arms + iso-token + β/p 扫 + 仪器 + 闸 + 合成)、
`instruments.py`(coverage 曲线、miss 斜率、联合多样性、τ-match、配对 bootstrap)、tests/
(β=0 等价、iso-token 记账、determinism、倒 U 在 mock 上)、`PREREG.md`(冻结)、`VERDICT.md`、
`results.json`、`EXECUTION_LOG.md`。

---

## 14. 与 SAVI / verifier 线转机的关系(正交、可组合)

verifier 线最新 `merge-noise-phasemap` 证明:decode 只在 **p≈0.5(frequency⊥correctness,即 verifier
不含答案充分统计量)** 的 regime 才有 head cover-room,那里一个**不完美学习 verifier** 把 −0.125 翻成
+0.29…+0.34(合成基底)。概率方法历来有**两半**:存在性(系综/覆盖)与构造化(选择/去随机)。我们
过去全压在**选择(verifier)**上。本实验补上**另一半——系综侧(覆盖)**,且**无 verifier**。两者
组合恰在同一 regime 咬合:p≈0.5 时,(相关采样抬覆盖)×(verifier 选)=(存在)×(选择)。本 PREREG
只负责**存在**那一半,把 pass@1 / verifier 留给 §11 之后的合流实验。
