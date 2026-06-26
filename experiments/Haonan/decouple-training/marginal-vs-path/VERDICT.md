# VERDICT — 在真实 LLM 的逐步 output 上，Viterbi 路径解码与逐位置 marginal 解码几乎不分高下

**结论。** 这一实验检验：把 single best path（Viterbi / max-product）和逐位置 marginal（per-position marginal / sum-product）这两种 global decoder 拆开比较，路径解码是否能在准确率上独立胜过 marginal。在真实 LLM 的逐步 output 上，答案是基本不能——逐步 output 太尖（top-1 ≈ 0.92），导致两种解码器收敛到同一答案。

**设置。** 载体是一条链式 CRF / HMM 形状的多解任务（多变量、相邻硬互斥约束、posterior 故意多模），unary 用真实 LLM（Qwen2.5-3B）逐位置读出的信念。对照组包含 exact emission（精确 unary）与对抗性构造的特例。指标为整段 exact-match 准确率（MAP 减 MPM）、posterior 多模度（压在 MAP 路径上的质量 pmap）、以及 marginal 解的非法率。数值见 `results.json`。

**结果。** 随机构造实例上，路径解码相对 marginal 的整段准确率差 ≤ 0.05；唯一干净的优势是保持全局合法（marginal 在多模区非法率约 28–38%，路径恒为 0%）。只有对抗性构造的特例（很多条弱解汇聚到某一位置的同一取值，求和后顶过单条最优路径）才让 single best path 在准确率上严格胜过 marginal。换到真实 LLM 的 emission 上，逐步 output 接近单峰、top-1 ≈ 0.92，两种解码器准确率差塌到约 +0.005～+0.010（置信区间触 0）；同批实例若换成 exact emission，则能分开（准确率差 +0.085，pmap 0.24）。

**成立条件与边界。** 路径解码相对 marginal 的独立准确率优势，仅在 posterior 真正多模、且竞争质量被构造成"很多条弱解汇聚到同一位置同一取值"的扇形结构时出现；真实 LLM 的逐步 output 太尖，不落进这种分布，因此其上两种解码器近乎等价。这里测的是合成链式任务上、用真实 LLM 读出信念作 unary 的情形，所测任务存在设计瑕疵（局部词汇线索、整段 exact-match 指标偏脆），方向性结论（尖 output → 不分家）一致，但仍需要一个更干净的多模自然语言任务才能钉死。
