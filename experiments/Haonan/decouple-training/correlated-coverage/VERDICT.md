# VERDICT — Correlated-coverage:相关采样顶不破 i.i.d. best-of-K 的覆盖天花板(clean NEGATIVE)

**Date:** 2026-06-30。**线:** emission。**判定:** 预注册 §12 kill criterion **触发**——
把 Erdős/sphere-graph 的"相关系综顶破 i.i.d. 底数"迁移到自回归 LLM 解码,在两条可实现路径上**都失败**。
i.i.d. best-of-K 的覆盖是天花板;覆盖只能靠更强的 base 模型 / 更多 token 抬,不能靠"更聪明的(负相关)系综"。

## 一句话

在自回归解码里**诱导有利相关不是免费的**:step 级互斥要 N× 代价(看到候选才能 repel),prompt-
conditioning 虽是 ~1× 生成,却把代价压到 prefill(3.5×)并把**窄 good-漏斗**的链**推进死区**。sphere-graph
的相关来自静态几何(免费),自回归解码没有这个免费对应物。

## 两条路径的结果

### Stage 1 — step 级 canonical-state 互斥(Tier-A,CPU mock,decode_core FROZEN)
诚实 `corr_step`(N=3,全 N-tax 计入)vs iid,在 iso-token、redundant 区(collapse {0.5,0.8,0.95})× p
{0.5–0.8} 共 **12/12 cell 全部 inadmissible**:N-tax 下 corr 无法**同时**留在 iid 头cover-room带(V1)
**且**负担得起 β 起作用所需的 K≥4 系综——两个预条件的预算支撑**不相交**。这是"N-tax 抹掉增益"的最强形式。
V1–V4 全 12/12 通过(harness 良定义,N-tax 诚实计费:corrK 1–3 vs iidK 2–10)。
**注:** mock 的 `corr_cheap` 臂被**隔离**——其 +0.333"win"是 N=3-draw / 1×-bill 的**记账漏水**,不是 repulsion;
verify 门拦下、未作为结果上报。

### Stage 2 — prompt-conditioning(忠实 ~1× 路径,real LLM,决定性)
frozen Qwen3-4B + decoupled LoRA(`adapter_decoupled_s0`),生成 k∈{5,6}(depth 4,5)、V1 选中
(iid coverage@K_ref 分数 ∈(0.2,0.9))的 **8 道**题,arms = iid_bok / conditioned / temp_matched_iid /
oracle,K-grid {2,4,6,8,12,16},τ=0.7,paired bootstrap n=10000。条件臂是**每步 1 forward**(`num_return_sequences=1`
硬编码;无 N-candidate;anti-leak 门通过)——mock 的漏水**未复发**;生成 token ≈ iid,conditioning 只增 prefill。

| arm | K2 | K4 | K6 | K8 | K12 | K16 |
|---|---|---|---|---|---|---|
| iid_bok | 0.500 | 0.750 | 1.000 | 1.000 | 1.000 | 1.000 |
| **conditioned** | 0.375 | 0.375 | 0.750 | 0.750 | 0.750 | 0.875 |
| temp_matched_iid | 0.250 | 0.250 | 0.625 | 0.625 | 0.750 | 0.750 |

- **H1(gen-token 轴,headline):** conditioned − iid @K=8 = **−0.250,95% CI [−0.625, 0.000]**,Holm adj_p=1.0。
  **每个 K 都输** iid。远不及 +0.05 bar。
- **H1(total-token 轴,sensitivity):** 更差。conditioned 在 K=16 花 **3.54×** token(54174 vs 15285)却覆盖更低。
  prefill 代价真实且大,买到的是负覆盖。
- **H2(miss 衰减 / "底数"):** 不显著(Holm adj_p=0.264);gen 轴 cond 名义略陡是 iid 在 K=6 饱和到 1.0 的
  floor 假象,total 轴符号翻转(iid 更陡)。无真实"底数 +ε"。
- **H4(温度控制):** **vacuous**——conditioned 没有增益可归因。temp − iid @K8 = −0.375 [−0.750,−0.125]
  (升温也输 iid);cond − temp = +0.125(n.s.)。所以连"是相关不是去尖化"都无从认证。
- **Funnel(窄漏斗判据):** to_solvable_frac iid 0.364 → **conditioned 0.318(最低)** → temp 0.324。
  conditioning 把链**推进死区**,正是 PREREG 预测的"弥散/过互斥"那一侧,不是有益多样化。
- **有效性:** V1 in-band、V4 oracle=1.0≥iid、anti-leak one-forward/step 全通过。

## 结构性结论(可迁移的部分)

best-of-K = i.i.d. 概率方法(Erdős/Shannon 随机系综),其覆盖天花板 = "K 条里至少一条对"的概率。
Ma–Shen–Xie(arXiv:2507.12926)顶破 i.i.d. 底数靠的是**静态几何给的有利相关**(sphere graph,免费)。
自回归解码**没有免费的相关**:
- **step 级**:要 repel 必须每步看到 N>1 个候选 → N× 生成代价 → 抹掉增益(Stage 1)。
- **prompt-conditioning**:~1× 生成,但 (a) 代价转移到 prefill(3.5×),(b) 把窄 good-漏斗推进死区 → 输(Stage 2)。

**所以 Erdős→sphere-graph 的迁移对 LLM 解码是隐喻,没有免费的自回归对应物。** 这与 verifier 线收口一致:
covered 的天花板由 base 模型 + token 决定;能改的 load-bearing 杠杆仍只有 verifier+depth(且那也需近精确 oracle)。
发射/coverage 这一侧(本线)在 iso-token 下顶不破 i.i.d.。

## 不settles / 诚实 caveat

- **二元 head-room 偏薄。** V1 用 coverage@K_ref **分数**(连续头cover-room代理)选题,但产出指标是**二元** best-of-K
  覆盖,在选中题上 K≈6 即饱和到 1.0 → 只有低 K(2,4)有空间,而那里 conditioned 也明确输(0.375 vs 0.5/0.75)。
  方向决定性,但效应量测得偏欠功率。更紧的版本应按二元 best-of-K@K_ref 选题(需更弱-覆盖层)。
- decoupled 4B adapter 在 depth 4–5 偏弱(多数生成题 0% 覆盖);8 题里仅 1 道 k=6,主要 k=5/depth-4。
- 单域(Countdown)、单 seed 主轴。

## Deliverables
`PREREG.md`(冻结)、`MECHANISM.md`(canonical-state repulsion 详析)、Tier-A harness(`corr_sampler.py`/
`mock_backend.py`/`instruments.py`/`run_corr.py`,141 tests)、real-probe(`realprobe/realprobe_core.py`/
`run_realprobe.py`,22 tests)、`outputs/{tierA.json, results.json, shard_*.json}`、本 `VERDICT.md`。
