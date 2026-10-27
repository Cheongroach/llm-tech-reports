# 一步步认识 Qwen、DeepSeek 等技术报告

这不是论文摘要合集，也不是模型排行榜。

它只做一件事：帮助已经理解 token、embedding、标准 Transformer 和基本训练流程的读者，逐步读懂 Qwen、DeepSeek 及若干代表性开源模型的技术报告。

本文默认你已经知道：

- token、词表和 embedding 的基本作用；
- decoder-only Transformer 按 token 自回归生成；
- attention、FFN、残差连接和反向传播的基本概念；
- 预训练、SFT 和推理的大致区别。

这些基础不会重新展开。遇到它们时，本文只解释新报告相对标准 Transformer 改了什么，以及这些修改为什么影响训练、推理或部署。

请按顺序阅读。每一步只增加一个主要认知。读完一步后，先回答末尾的问题；答不出来就停下来，不要继续堆术语。

全文截至 2026 年 10 月 27 日，分为 17 个阶段，共 160 步。一次只读一个阶段。

文中的模型指标、训练规模和架构描述来自模型团队公开的技术报告或官方发布材料。它们是论文作者报告的结果，不等于本文独立复现了这些结论。

---

## 第一阶段：先建立读技术报告的坐标系

### 第 1 步：技术报告通常同时回答四类问题

读一份大模型报告时，先把内容分成四条线：

```text
模型结构：每个 token 怎样经过网络
训练方法：参数怎样被学出来
训练系统：大规模训练怎样真正跑起来
评测证据：作者凭什么声称模型更好
```

同一个术语可能只属于其中一条线。比如 MLA 是结构修改，FP8 mixed precision 主要是训练系统修改，GRPO 是后训练方法，AIME 分数是评测结果。

只记住：**不要把结构、训练算法、系统优化和评测结果混成一件事。**

停下来回答：FP8 训练和 AIME 分数分别属于哪条线？

---

### 第 2 步：先分清“模型家族”和“具体 checkpoint”

`Qwen3` 是模型家族，下面可以有：

```text
Qwen3-8B
Qwen3-30B-A3B
Qwen3-235B-A22B
```

它们共享一部分设计，但层数、宽度、总参数和激活参数不同。不能从旗舰模型的结果直接推断小模型也有相同能力。

DeepSeek-R1 也不是所有带 `R1` 名称 checkpoint 的统称。`DeepSeek-R1`、`DeepSeek-R1-Zero` 和 `DeepSeek-R1-Distill-Qwen-32B` 的底座与训练路径都不同。[cite:9]

只记住：**家族名描述一组模型，checkpoint 名才指向一个具体对象。**

停下来回答：`DeepSeek-R1-Distill-Qwen-32B` 是否就是缩小版的原始 R1 权重？

---

### 第 3 步：Base、Instruct 和 Reasoning 模型不是同一训练阶段

可以先建立这条主线：

```text
Base：大规模预训练后的模型
  ↓ SFT / 偏好优化 / RL
Instruct：更会按指令回答的模型
  ↓ 面向可验证任务的推理训练
Reasoning：更倾向于使用长推理轨迹的模型
```

这不是所有团队都严格采用的唯一流程，但足以帮助阅读型号名称。

比较模型时必须对齐阶段。用一个 Base 模型对比另一个 Instruct 模型，常常是在比较不同问题。

只记住：**checkpoint 的后训练阶段是实验条件，不是名称装饰。**

停下来回答：为什么不能只凭参数量比较 Base 与 Instruct 模型？

---

### 第 4 步：Dense 模型只有一套 FFN 参数

标准 dense Transformer 的每一层都会使用该层同一套 attention 和 FFN 权重。

如果模型写作 `32B dense`，通常表示每个 token 前向时都会经过这 32B 参数中的主体计算路径。embedding、输出层等细节会使“参与计算”的严格口径更复杂，但它没有 MoE 的稀疏专家选择。

Llama 3 选择了标准 dense Transformer。报告明确说，这一选择的重要理由是控制系统复杂度并提高训练稳定性。[cite:11]

只记住：**dense 的重点是所有 token 共享同一套主干参数。**

停下来回答：dense 模型是否需要为每个 token 选择不同专家？

---

### 第 5 步：MoE 必须同时看总参数和激活参数

MoE 把 Transformer 中的 FFN 替换为多个专家 FFN。router 为每个 token 只选择其中少数专家。

因此报告常写：

```text
Qwen3-30B-A3B：总参数约 30B，每个 token 激活约 3B
DeepSeek-V3：总参数 671B，每个 token 激活 37B
```

前一个数字更接近权重存储规模，后一个数字更接近单 token 的稀疏计算规模。二者都不能单独代表真实延迟。[cite:4][cite:8]

只记住：**总参数影响容量和存储，激活参数更接近每 token 计算量。**

停下来回答：`671B-A37B` 是否只需要存储 37B 权重？

---

### 第 6 步：论文中的“作者观察”不等于普遍规律

技术报告通常包含三种层次：

```text
设计事实：模型用了 MLA、GQA 或 MoE
作者实验：某设置下 benchmark 或吞吐达到某结果
作者解释：为什么这种设计可能带来结果
```

设计事实可由模型配置和代码核对。实验结果需要对齐模型、硬件、精度和评测协议。机制解释则需要消融或独立实验支持。

只记住：**先判断一句话属于结构事实、实验观察还是机制解释。**

停下来回答：“模型使用 MLA”和“MLA 在所有硬件上都更快”的证据要求是否相同？

---

### 第一阶段检查

你现在应该能对任意模型先问六个问题：

1. 具体是哪个 checkpoint？
2. 它是 Base、Instruct 还是 Reasoning 模型？
3. 它是 dense 还是 MoE？
4. 总参数与激活参数分别是多少？
5. 当前结论属于结构、训练、系统还是评测？
6. 这是实现事实、作者观察还是机制解释？

如果这六个问题还会混淆，请不要继续。

---

## 第二阶段：认识报告共同使用的结构基线

### 第 7 步：现代 LLM 的差异主要集中在 attention 和 FFN

对本文涉及的文本模型，可以把一个 block 简化为：

```text
hidden states
  ↓ normalization
attention
  ↓ residual
normalization
FFN 或 MoE
  ↓ residual
```

Qwen、DeepSeek、Llama、Mistral 和 Gemma 的重要架构差异，大多可以定位到 attention、FFN/MoE、normalization 或位置处理。

只记住：**先找到改动发生在哪个子模块，再讨论作用。**

停下来回答：MLA 和 DeepSeekMoE 分别改动 block 的哪一部分？

---

### 第 8 步：Pre-Norm 把归一化放在子层之前

Pre-Norm 的简化形式是：

```text
x = x + Attention(Norm(x))
x = x + FFN(Norm(x))
```

Qwen2、Qwen2.5、Qwen3 和早期 DeepSeek LLM 都使用 RMSNorm 与 Pre-Norm 组合。[cite:2][cite:3][cite:4][cite:6]

这首先是训练稳定性设计。它不是推理加速算法，也不会直接减少 KV cache。

只记住：**Pre-Norm 描述 normalization 的位置。**

停下来回答：Pre-Norm 是 attention 类型还是残差块组织方式？

---

### 第 9 步：RMSNorm 不减均值

LayerNorm 通常会减均值并按方差缩放。RMSNorm 主要按均方根缩放，不执行完整的均值中心化。

报告中出现 RMSNorm 时，先把它理解为现代 decoder LLM 常见的稳定训练组件。不要因为它出现在新模型中，就把主要能力提升归因于 RMSNorm。

只记住：**RMSNorm 是基础归一化选择，不是 Qwen 或 DeepSeek 的核心独占创新。**

停下来回答：看到多个模型都使用 RMSNorm，能否据此解释它们能力差异？

---

### 第 10 步：SwiGLU 是 FFN 内部的门控形式

普通 FFN 可以写成两次线性变换，中间有非线性激活。

SwiGLU 增加一条门控分支，简化表示为：

```text
FFN(x) = W_down(SiLU(W_gate x) ⊙ (W_up x))
```

Qwen2、Qwen3、DeepSeek LLM 和 Mixtral 都采用或继承了 SwiGLU 类 FFN。[cite:2][cite:4][cite:6][cite:13]

只记住：**SwiGLU 改的是 FFN，不是 token 路由。**

停下来回答：MoE 专家内部能否继续使用 SwiGLU？

---

### 第 11 步：RoPE 把相对位置信息放进 Q 和 K

RoPE 对 query 和 key 的部分维度做与位置相关的旋转。

它影响 attention score 怎样感知相对位置。它不是独立的输入向量相加，也不是 KV cache 压缩。

长上下文报告常在 RoPE 基础上继续修改，例如 YaRN、分段训练或把位置相关部分与压缩部分解耦。

只记住：**RoPE 与 attention score 的位置关系直接相关。**

停下来回答：为什么 MLA 设计需要专门处理 RoPE 与低秩压缩的关系？

---

### 第 12 步：MHA、GQA 和 MQA 的关键差别是 KV 头数量

设 query 头数为 \(H_q\)，KV 头数为 \(H_{kv}\)：

```text
MHA：H_kv = H_q
GQA：1 < H_kv < H_q
MQA：H_kv = 1
```

多个 query 头共享较少的 K/V 头，可以降低解码时的 KV cache 和读取量。Qwen2/2.5/3、Mistral 7B、Llama 3 和 Gemma 2 都采用 GQA。[cite:2][cite:3][cite:4][cite:11][cite:12][cite:14]

只记住：**GQA 用 KV 共享换取更低的缓存与带宽压力。**

停下来回答：32 个 query 头、8 个 KV 头属于 MHA、GQA 还是 MQA？

---

### 第 13 步：KV cache 保存历史 token 的 K 和 V

自回归解码每次只生成一个新 token。历史 token 的 K/V 不变，因此可以缓存，避免每步重复计算。

以常见 GQA 为例，每层每个 token 的 KV 元素数近似为：

\[
2 \times H_{kv} \times d_{head}
\]

整个请求还要乘层数、序列长度、batch 和每元素字节数。

只记住：**长上下文解码不仅受计算限制，也可能受 KV cache 容量和读取带宽限制。**

停下来回答：上下文长度翻倍时，普通 GQA 的 KV cache 大致怎样变化？

---

### 第 14 步：Prefill 和 Decode 是两种不同负载

Prefill 一次处理已知 prompt 中的许多 token，矩阵乘法规模较大，通常更容易利用 GPU 计算吞吐。

Decode 每步只增加一个 token，却要读取大量模型权重和历史 KV cache，常更受显存带宽与调度开销影响。

因此：

```text
训练更快 ≠ prefill 更快 ≠ decode 更快
```

只记住：**任何“推理更快”的结论都要说明是 prefill、decode 还是端到端。**

停下来回答：减少 KV cache 最直接影响哪一阶段？

---

### 第二阶段检查

请闭卷解释：

- GQA 为什么减少 KV cache；
- SwiGLU 位于 block 的哪个位置；
- RoPE 为什么与 Q/K 相关；
- Prefill 和 Decode 为什么不能用同一个瓶颈概括。

---

## 第三阶段：沿 Qwen 报告看模型怎样演进

### 第 15 步：第一代 Qwen 先建立了完整模型家族

第一代 Qwen 报告不只发布 Base 和 Chat 模型，还覆盖代码、数学、视觉语言和 agent/tool-use 方向。[cite:1]

它的意义不是某一个后来仍保留的结构创新，而是建立一条模型家族路线：

```text
通用 Base
  ├─ Chat / RLHF
  ├─ Code
  ├─ Math
  └─ Vision-Language
```

只记住：**Qwen 从一开始就把“通用底座 + 专项后训练”作为家族组织方式。**

停下来回答：Math-Qwen 与 Qwen Base 的关系更接近重新发明架构，还是专项训练分支？

---

### 第 16 步：Qwen 的大词表首先影响压缩率和参数存储

第一代 Qwen 使用约 152K 词表，并强调多语言尤其中文的编码效率。[cite:1]

更大词表可能用更少 token 表示同一文本，但会增加 embedding 和输出 projection 的参数量。它不会自动提高推理能力。

这也是阅读参数量时必须检查 embedding 是否 tied 的原因。

只记住：**词表大小同时影响 token 长度、embedding 参数和输出层成本。**

停下来回答：更高的 tokenizer 压缩率是否等同于更高 benchmark 分数？

---

### 第 17 步：Qwen2 把 GQA 作为 dense 模型主路径

Qwen2 dense 模型采用 GQA，目标是降低 KV cache 并提高推理吞吐。[cite:2]

例如 Qwen2-72B 使用 64 个 query 头和 8 个 KV 头。每 8 个 query 头共享一组 K/V。

这是一项结构选择。实际速度还取决于 kernel、batch、序列长度和硬件。

只记住：**Qwen2 的 attention 主线从标准 MHA 转向 GQA。**

停下来回答：Qwen2-72B 的 64/8 头配置表示怎样的共享关系？

---

### 第 18 步：DCA 和 YaRN 面向长上下文扩展

Qwen2 报告使用 Dual Chunk Attention（DCA）与 YaRN 扩展上下文。[cite:2]

DCA 把长序列的相对位置关系按 chunk 组织；YaRN 调整 RoPE 外推时的位置尺度与 attention scale。

它们解决的是模型在训练长度之外怎样处理更长位置，不等于把全 attention 的理论复杂度变成线性。

只记住：**能接受更长上下文和以更低复杂度处理上下文是两个问题。**

停下来回答：128K context 支持是否自动意味着 attention 计算已经是线性的？

---

### 第 19 步：Qwen2 同时保留 dense 和 MoE 两条产品线

Qwen2 发布 0.5B 到 72B 的 dense 模型，也发布 `57B-A14B` MoE 模型。[cite:2]

这说明 dense 与 MoE 不是简单的前后代关系，而是不同部署约束下的两种选择：

```text
dense：结构规则，部署相对直接
MoE：总容量更大，但路由、通信和权重读取更复杂
```

只记住：**MoE 不是 dense 的必然替代。**

停下来回答：为什么同一模型家族会同时发布 dense 和 MoE？

---

### 第 20 步：Qwen2 MoE 使用细粒度专家

传统 MoE 可以把一个完整 FFN 复制成多个大专家，再选少数专家。

Qwen2 MoE 把专家切得更细，同时激活更多小专家。报告的直觉是：在相近总参数和激活参数下，细粒度设计提供更多专家组合。[cite:2]

这条路线与 DeepSeekMoE 相似，但具体专家数量、共享专家和负载均衡方法并不相同。

只记住：**细粒度 MoE 用更多小专家增加路由组合。**

停下来回答：专家数量更多是否一定意味着每个 token 的激活参数更多？

---

### 第 21 步：Qwen2 MoE 还使用共享专家

共享专家处理所有 token；routed experts 只处理 router 选中的 token。

可以把它理解为：

```text
输出 = 共享能力 + 按 token 选择的专门能力
```

Qwen2 `57B-A14B` 使用 routed experts 与 shared experts 的组合。[cite:2]

只记住：**共享专家始终激活，路由专家稀疏激活。**

停下来回答：共享专家的计算能否通过 top-k router 完全跳过？

---

### 第 22 步：Qwen2.5 的主要变化不在基础 block

Qwen2.5 dense 模型继续使用 Qwen2 的 GQA、SwiGLU、RoPE、QKV bias、RMSNorm 和 Pre-Norm。[cite:3]

因此读 Qwen2.5 报告时，不应假设每一代都必须发明新的 attention。它的重点更多转向数据规模、数据质量、超参数规律和后训练。

只记住：**模型升级可以主要来自数据与训练，而不是结构更换。**

停下来回答：Qwen2.5 分数提高是否能直接归因于一种新 attention？

---

### 第 23 步：Qwen2.5 把预训练数据从 7T 扩展到 18T token

Qwen2.5 报告将预训练数据规模从 Qwen2 的 7T 扩展到 18T，并增加数学、代码、知识与合成数据。[cite:3]

数据规模只是一个维度。报告还描述了：

- 用前代模型做质量评分；
- 调整不同领域的采样比例；
- 用专项模型生成和过滤合成数据。

只记住：**“更多 token”与“更好的数据混合”是两项不同变化。**

停下来回答：只知道 18T token，能否判断数学数据占比和质量？

---

### 第 24 步：Qwen2.5 用 scaling law 选择学习率和 batch

Qwen2.5 不只用 scaling law 预测“模型应该多大”，还用 proxy experiments 估计不同模型规模下的合适学习率和 batch size。[cite:3]

这反映了一种工程方法：

```text
小模型实验
→ 拟合超参数趋势
→ 外推到目标规模
→ 用大模型训练验证
```

只记住：**scaling law 也可以服务于训练超参数选择。**

停下来回答：小模型拟合出的规律在大模型上是否仍需验证？

---

### 第 25 步：Qwen2.5 把后训练拆成多个阶段

Qwen2.5 报告包含超过 1M 条 SFT 数据，以及 offline DPO 和 online GRPO。[cite:3]

可以先这样理解：

```text
SFT：学习高质量示范
DPO：从成对偏好中直接优化
GRPO：对模型自己采样的多条回答做相对强化学习
```

这些阶段改变模型行为，不改变 Qwen2.5 dense block 的基础结构。

只记住：**后训练能力提升不要求重新设计每个 Transformer block。**

停下来回答：GRPO 属于结构优化还是训练目标？

---

### 第 26 步：Qwen3 改用 QK-Norm 并移除 QKV bias

Qwen3 dense 模型总体延续 Qwen2.5，但移除 QKV bias，并在 attention 中加入 QK-Norm，以提高训练稳定性。[cite:4]

QK-Norm 在计算 attention score 前归一化 query 和 key。它主要约束 score 的数值尺度。

只记住：**QK-Norm 是 attention 数值稳定性设计，不是 KV cache 压缩。**

停下来回答：QK-Norm 会直接减少 KV 头数量吗？

---

### 第 27 步：Qwen3 MoE 取消了共享专家

Qwen3 MoE 使用 128 个专家，每个 token 激活 8 个，但不再使用 Qwen2.5 MoE 的 shared experts。[cite:4]

因此不能把“Qwen MoE”当成一套固定结构。至少要说明：

```text
专家总数
top-k
是否有共享专家
负载均衡目标
```

只记住：**同一家族的 MoE 设计也会改变。**

停下来回答：Qwen2 MoE 与 Qwen3 MoE 的共享专家设置是否相同？

---

### 第 28 步：Qwen3 把 thinking 与 non-thinking 合进同一模型

Qwen3 的核心后训练目标之一，是让一个 checkpoint 同时支持：

```text
thinking：允许较长推理过程
non-thinking：快速直接回答
```

模式由 chat template 或输入控制，而不是必须切换两个模型。[cite:4]

只记住：**thinking mode 是同一模型的行为模式，不是另一套 Base 权重。**

停下来回答：切换 Qwen3 thinking mode 是否等于加载另一个模型？

---

### 第 29 步：Thinking budget 把推理长度变成可控资源

Qwen3 引入 thinking budget，让使用者限制推理 token 数量。[cite:4]

它建立了一个显式权衡：

```text
更多 thinking token
→ 可能得到更高任务成功率
→ 同时增加延迟和 token 成本
```

报告观察到多个任务上随 budget 增加而改善，但这不是所有问题上都无限单调提升的数学保证。

只记住：**thinking budget 是 inference-time compute 控制量。**

停下来回答：增加 thinking budget 为什么不能被描述为“免费提升”？

---

### 第 30 步：Qwen3 对小模型更强调蒸馏

Qwen3 报告比较了直接 RL 与从强模型蒸馏，并报告在小模型上，蒸馏在效果和训练效率方面更有优势。[cite:4]

这与后面的 DeepSeek-R1 结论方向一致：大模型发现的推理轨迹可以成为小模型训练数据。

只记住：**小模型不一定适合照搬大模型的完整 RL 探索过程。**

停下来回答：教师蒸馏和学生独立 RL 是否学习自同一数据分布？

---

### 第 31 步：Qwen3.5 是新的延伸线，不应倒灌进 Qwen3 报告

Qwen3.5 官方发布说明描述了 Gated DeltaNet 与标准 attention 的混合结构、更稀疏的 MoE、MTP 和原生多模态训练；首发模型为 `397B-A17B`。[cite:5]

截至 2026-10-27，Qwen3.5-Omni 已有独立技术报告，但 Qwen3.5 纯文本主模型的完整技术报告仍未公开。因此应区分：

```text
Qwen3 报告：完整技术报告
Qwen3.5 文本主模型：官方发布说明中的设计与作者评测
Qwen3.5-Omni：独立的 Omni 技术报告
```

只记住：**最新不等于证据最完整。**

停下来回答：为什么不应把 Qwen3.5 发布页上的所有结果当成已独立复现结论？

---

### 第三阶段检查

请按时间线闭卷说出：

```text
Qwen
→ Qwen2：GQA、dense + 细粒度 MoE、长上下文扩展
→ Qwen2.5：数据与多阶段后训练显著扩展
→ Qwen3：QK-Norm、统一 thinking、thinking budget、蒸馏
→ Qwen3.5：混合线性注意力与原生多模态的延伸线
```

---

## 第四阶段：沿 DeepSeek 报告看架构怎样走向效率优先

### 第 32 步：DeepSeek LLM 的起点是 scaling law

DeepSeek LLM 报告的核心问题之一是：给定计算预算，模型规模、数据规模、batch 和学习率应该怎样增长。[cite:6]

报告还强调，不同数据集拟合出的 scaling law 会不同。这意味着 scaling law 不是脱离数据分布的自然常数。

只记住：**DeepSeek 的早期主线先从“怎样可靠扩展”开始。**

停下来回答：在数据分布改变后，旧 scaling law 是否可不经验证直接复用？

---

### 第 33 步：DeepSeek LLM 仍主要沿用 Llama 式 dense 架构

DeepSeek LLM 使用 Pre-Norm、RMSNorm、SwiGLU 和 RoPE。67B 版本使用 GQA，并通过更深的网络而不是单纯扩大 FFN 宽度来配置参数。[cite:6]

这说明 DeepSeek-V2 的 MLA 与 MoE 不是从第一代就存在。

只记住：**先知道上一代结构，才能看清下一代真正新增了什么。**

停下来回答：可以把 DeepSeek LLM 67B 描述为 MLA 模型吗？

---

### 第 34 步：DeepSeek-V2 同时改了 attention 和 FFN

DeepSeek-V2 的 block 有两项主要变化：

```text
attention：MHA/GQA → MLA
FFN：dense FFN → DeepSeekMoE
```

前者主要针对 KV cache 与推理效率，后者主要针对模型容量与稀疏计算。[cite:7]

只记住：**MLA 与 DeepSeekMoE 解决不同子模块的问题。**

停下来回答：DeepSeek-V2 的 KV cache 降低应主要归因于 MLA 还是 MoE？

---

### 第 35 步：MLA 先把 K/V 联合压缩成 latent

标准 attention 会为每个历史 token 缓存展开后的 K 和 V。

MLA 的核心思路是先得到较小的 latent：

\[
c_t^{KV} = W^{DKV} h_t
\]

需要计算 attention 时，再从 latent 上投影出 K/V 相关表示。缓存对象从完整 K/V 变成更小的 latent 与少量位置相关分量。[cite:7]

只记住：**MLA 缓存压缩后的中间表示，而不是完整多头 K/V。**

停下来回答：MLA 为什么不能简单描述为“更少的 KV 头”？

---

### 第 36 步：MLA 需要把位置部分与可吸收投影分开

如果 K 的所有维度都应用 RoPE，位置相关旋转会阻止一部分投影矩阵在推理时被吸收到其他矩阵中。

DeepSeek-V2 因此使用 decoupled RoPE：把内容相关部分与携带 RoPE 的位置部分分开。[cite:7]

这一步是 MLA 能在保持位置能力的同时压缩缓存的重要配套设计。

只记住：**低秩压缩与位置编码必须共同设计。**

停下来回答：为什么只把 K/V 做低秩投影还不足以完整解释 MLA？

---

### 第 37 步：MLA 的缓存收益不等于所有延迟都同比下降

DeepSeek-V2 报告在自己的模型与系统设置下，给出相对 DeepSeek 67B 的 KV cache、吞吐和成本结果。[cite:7]

这些数字同时受到以下变化影响：

- dense 变为 MoE；
- MHA/GQA 变为 MLA；
- 模型规模与训练数据改变；
- serving 实现和并行策略改变。

只记住：**整模型结果不能自动成为单一组件的纯消融。**

停下来回答：为什么整模型 5.76 倍最大吞吐不能全部写成 MLA 单独贡献？

---

### 第 38 步：DeepSeekMoE 也使用细粒度专家

DeepSeekMoE 把大 FFN 切成更多小专家，让每个 token 选择多个细粒度专家。[cite:7]

与传统“8 个大专家选 2 个”相比，细粒度设计提供更多组合，但也带来更复杂的 token dispatch 和 kernel 组织。

只记住：**专家更细会增加组合自由度，也增加系统实现难度。**

停下来回答：细粒度专家只影响模型统计容量，还是也影响 GPU dispatch？

---

### 第 39 步：DeepSeekMoE 使用 shared expert isolation

DeepSeekMoE 将一部分专家设为共享专家，承载常见知识；其他 routed experts 尝试学习更专门的模式。[cite:7]

它与 Qwen2 MoE 都有共享和路由专家，但不能因此认为两者配置相同。

只记住：**相同 MoE 术语下仍需核对专家数、top-k 和路由规则。**

停下来回答：知道两个模型都有 shared experts，是否足以推断激活参数相同？

---

### 第 40 步：Device-limited routing 约束 token 跨多少设备

如果 top-k 专家散布在很多 GPU 上，一个 token 会触发更多跨设备通信。

DeepSeek-V2 的 device-limited routing 限制每个 token 的候选专家设备范围，以控制通信成本。[cite:7]

这说明 router 的目标不只是“选最匹配专家”，还必须考虑专家放在哪台设备上。

只记住：**MoE 路由同时是模型问题和分布式系统问题。**

停下来回答：为什么只优化 router 准确性可能得到无法高效执行的分配？

---

### 第 41 步：负载均衡损失可能干扰语言建模目标

MoE 需要避免所有 token 集中到少数专家。常见做法是增加 auxiliary balance loss。

但 auxiliary loss 与语言建模 loss 不完全一致。权重太大可能牺牲模型质量，太小又可能导致设备负载不均。

DeepSeek-V2 使用多级辅助损失；DeepSeek-V3 随后把降低这种干扰作为新目标。[cite:7][cite:8]

只记住：**负载均衡不是免费约束。**

停下来回答：把 balance loss 权重不断调大是否一定更好？

---

### 第四阶段检查

你现在应该能画出：

```text
DeepSeek LLM：dense + GQA
        ↓
DeepSeek-V2：
  attention → MLA latent cache
  FFN       → fine-grained DeepSeekMoE
  system    → device-limited routing
```

---

## 第五阶段：读懂 DeepSeek-V3 的算法与系统协同

### 第 42 步：DeepSeek-V3 延续 MLA 与 DeepSeekMoE

DeepSeek-V3 不是重新替换 V2 的整个架构，而是在 MLA 与 DeepSeekMoE 基础上继续扩展到 `671B-A37B`。[cite:8]

它的主要新增点包括：

- auxiliary-loss-free load balancing；
- Multi-Token Prediction；
- FP8 mixed precision training；
- DualPipe 与跨节点 all-to-all 优化。

只记住：**V3 的创新层次同时覆盖模型目标与训练系统。**

停下来回答：MLA 是 V3 首次提出的吗？

---

### 第 43 步：Auxiliary-loss-free 不等于完全没有均衡机制

DeepSeek-V3 为每个专家维护动态 bias，改变路由时的选择分数，以纠正过载或欠载；bias 不直接参与最终专家输出权重。[cite:8]

它还保留了较小的 sequence-wise balance loss 作为补充。

因此准确表述是：

```text
主要负载均衡不依赖会直接干扰主目标的大权重 auxiliary loss
```

而不是“没有任何负载均衡约束”。

只记住：**auxiliary-loss-free 描述主要策略，不是完全取消均衡控制。**

停下来回答：动态 expert bias 的目标是什么？

---

### 第 44 步：Node-limited routing 把通信约束提升到节点级

V3 在跨节点 MoE 训练中限制每个 token 路由到的节点数。[cite:8]

节点内部通常有更高带宽互联，跨节点则依赖较慢网络。限制节点范围可降低 all-to-all 的通信扇出。

只记住：**专家并行性能取决于专家布局与网络拓扑。**

停下来回答：两种专家布局具有相同 FLOPs 时，通信时间是否一定相同？

---

### 第 45 步：MTP 让训练目标预测多个未来 token

标准 next-token prediction 只监督下一个 token。

DeepSeek-V3 的 Multi-Token Prediction（MTP）增加模块，让当前表示同时预测后续多个位置，并把这些损失加入训练目标。[cite:8]

报告把它用于两个方向：

```text
训练：提供更密集的未来 token 监督
推理：MTP 模块可作为 speculative decoding 的 draft
```

只记住：**MTP 首先是训练目标扩展，也可复用于推理加速。**

停下来回答：用了 MTP 是否表示主模型一次就无条件接受多个 token？

---

### 第 46 步：Speculative decoding 仍要由主模型验证

draft 模块先提出多个候选 token，主模型并行验证。

只有验证通过的前缀才能被接受。因此加速取决于：

```text
候选接受率
验证成本
draft 额外成本
batch 与序列形状
```

只记住：**MTP 能提供候选，不会跳过主模型正确性检查。**

停下来回答：draft 连续提出 4 个 token 是否保证每轮都前进 4 个 token？

---

### 第 47 步：FP8 mixed precision 不是把所有状态都存成 FP8

DeepSeek-V3 使用 FP8 执行部分核心 GEMM，并保留更高精度的累加、主权重或敏感操作。[cite:8]

“mixed precision”的关键就是不同对象使用不同精度：

```text
输入与权重块：可量化到 FP8
乘法累加：分阶段提高精度
敏感模块：保留 BF16/FP32 路径
```

只记住：**大规模 FP8 训练依赖精度边界设计，而不是全模型一键转 FP8。**

停下来回答：为什么只看权重 dtype 不能判断完整训练数值路径？

---

### 第 48 步：细粒度量化减少单个 scale 覆盖的动态范围

若一整张大矩阵共享一个 scale，少数异常值会降低其他值的有效精度。

V3 使用 tile-wise 或 block-wise scaling，让较小数据块有独立 scale。[cite:8]

代价是更多 scale 元数据、在线量化和专用 kernel 逻辑。

只记住：**更细的量化粒度通常提高数值适应性，但增加实现复杂度。**

停下来回答：block-wise scaling 为什么比整张矩阵一个 scale 更能适应局部异常值？

---

### 第 49 步：DualPipe 目标是隐藏 pipeline bubble 和通信

Pipeline parallelism 会把模型层分到不同 stage。普通流水线在填充、排空和依赖等待时会出现 bubble。

DualPipe 从流水线两端调度 micro-batch，并重叠前向、反向与 MoE 通信。[cite:8]

只记住：**DualPipe 是训练调度算法，不是模型 attention。**

停下来回答：DualPipe 会改变 checkpoint 的数学结构吗？

---

### 第 50 步：MoE all-to-all 是 token 的跨设备重排

进入 MoE 层时，不同 token 需要被发送到持有目标专家的 GPU；专家计算后再送回原位置。

这形成：

```text
dispatch all-to-all
→ expert GEMM
→ combine all-to-all
```

V3 专门优化跨节点 all-to-all，并利用计算通信重叠降低暴露出来的通信时间。[cite:8]

只记住：**MoE 的稀疏计算收益可能被 token 通信吞掉。**

停下来回答：专家 GEMM 更快时，端到端 step time 是否一定同比下降？

---

### 第 51 步：训练成本数字必须读完整口径

DeepSeek-V3 报告给出完整官方训练约 2.788M H800 GPU hours，并按 2 美元/GPU hour 估算约 5.576M 美元。[cite:8]

报告也明确说明，该数字不包括前期研究、架构试验和数据消融成本。

因此准确结论是：

```text
这是报告口径下最终正式训练的 GPU-hour 与租价换算
```

不是公司完成整个模型研发的总财务成本。

只记住：**成本结论必须包含纳入项、排除项和假设单价。**

停下来回答：5.576M 美元是否包括所有失败实验、人力与数据成本？

---

### 第五阶段检查

请把 V3 新增内容放回正确层次：

| 内容 | 层次 |
|---|---|
| MTP | 训练目标 / 可辅助推理 |
| dynamic expert bias | MoE 路由均衡 |
| FP8 | 数值与训练系统 |
| DualPipe | pipeline 调度 |
| all-to-all kernel | 分布式通信 |

---

## 第六阶段：读懂 DeepSeek-R1 的 reasoning 训练

### 第 52 步：R1 的底座仍是 DeepSeek-V3-Base

DeepSeek-R1 不是另一套全新 attention 或 MoE 架构。

它从 DeepSeek-V3-Base 出发，重点改变后训练过程。[cite:9]

因此 R1 报告的主要问题是：

> 怎样用 reinforcement learning 激发并稳定长推理行为？

只记住：**R1 的主要创新位于 post-training，不是基础 block。**

停下来回答：解释 R1 时是否必须先重新发明 MLA？

---

### 第 53 步：R1-Zero 先测试“没有 SFT 冷启动”的 RL

DeepSeek-R1-Zero 直接在 Base 模型上做大规模 RL，没有先用长 CoT SFT 数据冷启动。[cite:9]

这个实验用于观察：只有可验证 reward 时，模型能否自行形成更长的推理、反思与检查行为。

只记住：**R1-Zero 是验证训练路径的实验对象，不是最终用户体验最好的版本。**

停下来回答：R1-Zero 与最终 R1 的训练起点是否相同？

---

### 第 54 步：GRPO 用同一问题的一组回答建立相对 baseline

对同一个问题，旧策略采样 \(G\) 个回答并得到 reward：

\[
A_i =
\frac{r_i - \operatorname{mean}(r_1,\ldots,r_G)}
{\operatorname{std}(r_1,\ldots,r_G)}
\]

高于组均值的回答得到正 advantage，低于组均值的得到负 advantage。GRPO 不需要训练一个与 policy 同规模的 critic model。[cite:9]

只记住：**GRPO 的 baseline 来自同组样本的相对 reward。**

停下来回答：一个回答的 advantage 只由其绝对 reward 决定吗？

---

### 第 55 步：Rule-based reward 适合可自动验证任务

R1-Zero 的 reward 主要包括：

- accuracy reward：最终答案可由规则检查；
- format reward：输出是否符合要求的格式。

数学题答案、代码测试和形式化证明更容易构造这类 reward。开放式写作则难以仅靠规则覆盖质量。[cite:9]

只记住：**可验证性决定 RL signal 是否可靠。**

停下来回答：为什么同一套 rule reward 不容易直接训练文学写作质量？

---

### 第 56 步：“Aha moment”是观察，不是已证明的内部机制

R1 报告展示了模型在训练中增加思考长度、回看和修正答案的样例，并称为 “aha moment”。[cite:9]

这些样例说明某些行为出现了，但不能单独证明模型内部形成了与人类相同的认知机制。

只记住：**行为样例是现象证据，不是心智机制证明。**

停下来回答：一条自我反思文本能否证明模型具有稳定、通用的自我认知？

---

### 第 57 步：R1-Zero 暴露了可读性和语言混杂问题

纯 reasoning RL 可以提高可验证任务表现，却可能生成难读、重复或语言混杂的长轨迹。[cite:9]

这说明“答案 reward 更高”与“输出适合用户阅读”不是同一目标。

只记住：**推理正确性、表达质量和语言一致性需要分别约束。**

停下来回答：只优化最终答案 reward，是否保证中间推理清晰？

---

### 第 58 步：最终 R1 先加入 cold-start SFT

DeepSeek-R1 在 reasoning RL 前，先使用数千条较高质量的长 CoT 样例进行 cold start。[cite:9]

目的包括：

- 建立更稳定的输出格式；
- 提高可读性；
- 为 RL 提供更好的初始策略。

只记住：**cold start 不替代 RL，它改变 RL 的起点。**

停下来回答：R1 的 cold-start 数据规模是否等同于完整预训练语料？

---

### 第 59 步：R1 不是一次 RL，而是多阶段管线

最终 R1 的简化流程是：

```text
Base
→ cold-start SFT
→ reasoning RL
→ rejection sampling 生成新 SFT 数据
→ reasoning + general SFT
→ general RL
```

报告把它概括为两个 SFT 阶段和两个 RL 阶段。[cite:9]

只记住：**最终能力来自多阶段组合，不能只写成“纯 RL”。**

停下来回答：把 DeepSeek-R1 描述为“完全不用 SFT”是否准确？

---

### 第 60 步：Rejection sampling 把模型探索结果变成监督数据

模型先生成多个候选回答，再用规则或模型筛选较好的轨迹，形成新的 SFT 数据。

这样，RL 期间发现的有效模式可以被整理成更稳定的监督信号。

只记住：**rejection sampling 是从候选中筛数据，不是直接更新 policy 的 RL step。**

停下来回答：被拒绝的样本会作为最终 SFT 正样本吗？

---

### 第 61 步：R1 Distill 模型是“数据蒸馏”，不是权重裁剪

DeepSeek-R1 生成 reasoning 数据，再用这些数据微调 Qwen2.5 或 Llama3 dense 模型，得到多个 Distill checkpoint。[cite:9]

这些模型继承的是教师输出中的推理模式，不是从 671B 权重直接剪枝到 32B。

只记住：**Distill-Qwen 的架构仍是 Qwen，教师主要提供训练数据。**

停下来回答：`R1-Distill-Qwen-32B` 的 MLA 来自哪里？

---

### 第 62 步：报告中的“蒸馏优于小模型直接 RL”有适用范围

R1 报告在特定 Qwen2.5-32B 对照中观察到，直接从 R1 蒸馏优于对小模型应用同类 RL。[cite:9]

这支持“强教师轨迹对小模型很有效”，但不能推出任何教师、任何数据和任何学生上蒸馏都优于 RL。

只记住：**局部实验结论必须保留模型与训练设置。**

停下来回答：该结果能否证明 RL 对所有小模型都无用？

---

### 第六阶段检查

请区分：

```text
R1-Zero：Base → reasoning RL
R1：cold-start SFT → reasoning RL → 新 SFT → general RL
R1-Distill：用 R1 生成的数据训练另一个 dense 底座
```

---

## 第七阶段：把 attention 设计放到同一张图里

### 第 63 步：Full attention 的 prefill 计算随序列长度平方增长

对长度 \(L\) 的 prompt，完整 self-attention 需要构造 token 间关系，核心 score 计算量近似：

\[
O(L^2 d)
\]

这解释了为什么 128K、1M 上下文不能只靠“显存够大”解决。

只记住：**长上下文同时带来 KV 容量问题和 attention 计算问题。**

停下来回答：KV cache 压缩能否自动把 prefill 的 full attention 变成线性复杂度？

---

### 第 64 步：GQA 主要压缩 KV 头，不改变 full attention 的长度阶数

GQA 减少 \(H_{kv}\)，所以缓存和 K/V 读取更小。

但只要每个 query 仍与全部历史位置做 attention，序列长度维度仍是 full attention。

只记住：**GQA 改常数项和缓存规模，不把 \(L^2\) 变成 \(L\)。**

停下来回答：GQA 为什么能明显帮助 decode，却不等于 sparse attention？

---

### 第 65 步：MLA 压缩每个位置保存的表示

MLA 主要把每个历史位置的完整多头 K/V 压缩成 latent cache。[cite:7]

可以把 GQA 与 MLA 的差别先理解为：

```text
GQA：减少独立 KV 头数量
MLA：学习一个低秩 latent，再重建 attention 所需表示
```

只记住：**两者都减少 cache，但压缩参数化方式不同。**

停下来回答：MLA 是否只是把 `num_key_value_heads` 调得更小？

---

### 第 66 步：Sliding Window Attention 限制每层可直接看的位置

Mistral 7B 的每个 token 在一层内只看最近 \(W\) 个 token。随着层数堆叠，信息仍可跨更远距离传播。[cite:12]

计算从“每个 query 看全部历史”变成“每个 query 看固定窗口”，但远距离信息是间接传播，不等价于当前层直接访问任意历史 token。

只记住：**滑窗用局部直接连接换取更低成本。**

停下来回答：最后一层能间接获得远处信息，是否等于它直接对所有历史 token 做 attention？

---

### 第 67 步：Sparse attention 先选位置，再做精确 attention

DeepSeek-V3.2 的 DSA 使用轻量 indexer 为历史位置评分，再只选 top-k KV entries 做主 attention。[cite:10]

主 attention 从：

\[
O(L^2)
\]

降为近似：

\[
O(Lk), \quad k \ll L
\]

indexer 本身仍有成本，但比完整 MLA attention 更轻。

只记住：**sparse attention 的核心是显式减少被访问的位置数。**

停下来回答：DSA 与 GQA 分别稀疏了位置维度还是头维度？

---

### 第 68 步：Linear attention 用状态更新替代完整 token 对

Qwen3.5 官方材料描述的 Gated DeltaNet 以递归状态更新处理序列，大部分层不再显式构造完整 \(L \times L\) attention 矩阵。[cite:5]

这类方法把长序列复杂度降到近线性，但固定状态可能难以像 full attention 那样精确回看任意 token。

因此 Qwen3.5 使用 linear attention 与标准 attention 的 hybrid，而不是全部替换。

只记住：**linear attention 与 sparse attention 是两种不同降复杂度路线。**

停下来回答：top-k 历史检索和递归状态更新是否是同一种算法？

---

### 第 69 步：Hybrid attention 是在召回能力与效率之间分层

混合架构可以让多数层使用便宜的 local/linear attention，少数层保留 full attention。

Gemma 2 交替使用 local sliding-window 和 global attention；Qwen3.5 发布材料则描述 Gated DeltaNet 与标准 attention 混合。[cite:14][cite:5]

只记住：**不同层不必使用同一种 token mixing。**

停下来回答：为什么 hybrid 模型仍可能保留少量昂贵的 full attention 层？

---

### 第 70 步：长上下文能力至少要分成四项验收

一份报告声称支持 128K 或 1M context 时，应分别检查：

1. 模型接口是否接受该长度；
2. 显存和系统是否能执行；
3. 远距离检索 benchmark 是否通过；
4. 复杂长文推理是否保持质量。

Needle-in-a-Haystack 通过主要证明检索，不代表复杂跨段推理全部通过。

只记住：**最大 context length 不是单一质量指标。**

停下来回答：能在 128K 中找回一个唯一字符串，是否证明能正确总结整本书？

---

### 第七阶段检查

| 路线 | 主要减少什么 |
|---|---|
| GQA | KV 头数 |
| MLA | 每个位置缓存的表示维度 |
| Sliding Window | 每层直接访问的位置范围 |
| Sparse Attention | 每个 query 实际选择的位置数 |
| Linear Attention | 不显式保存完整 token-pair 矩阵 |

---

## 第八阶段：真正理解 MoE 的系统代价

### 第 71 步：Router 只决定专家，不负责执行专家

router 为每个 token 计算专家分数并选 top-k。

随后 runtime 还要完成：

```text
统计每个专家收到哪些 token
→ 重排 token
→ 跨设备发送
→ 执行 expert GEMM
→ 把结果送回并恢复顺序
```

只记住：**router 公式只是 MoE 执行链的第一步。**

停下来回答：top-k 选择结束后，为什么还需要 token permutation？

---

### 第 72 步：Expert Parallelism 按专家切分权重

Tensor Parallelism 常把一个矩阵切到多个 GPU；Expert Parallelism 则把不同专家放在不同 GPU。

EP 减少每张卡需要保存的专家权重，但引入 token all-to-all。

只记住：**EP 用通信换取专家权重分片。**

停下来回答：EP 场景中，token 为什么可能离开原 GPU？

---

### 第 73 步：负载不均会形成 straggler

一个 batch 内，如果大量 token 路由到少数专家，这些专家所在 GPU 要处理更多 token。

其他 GPU 即使先完成，也要等待最慢设备。

因此 MoE step time 更接近：

```text
最忙专家/设备的完成时间
```

而不是所有专家平均时间。

只记住：**MoE 的平均负载好看，不代表尾部负载合理。**

停下来回答：为什么专家 token 数的最大值比平均值更能预测 straggler？

---

### 第 74 步：Top-k 增加会同时提高容量和计算

每个 token 选更多专家，可能获得更丰富的组合，也会增加：

- expert GEMM；
- token dispatch；
- combine；
- 激活参数。

因此 `num_experts=128, top_k=8` 与 `num_experts=8, top_k=2` 不能只按专家总数比较。

只记住：**MoE 成本由总专家数、top-k 和专家宽度共同决定。**

停下来回答：总专家数翻倍、top-k 不变时，每 token 计算是否必然翻倍？

---

### 第 75 步：MoE 更少 FLOPs 不保证单请求更低延迟

MoE 的 active FLOPs 可以低于同容量 dense 模型，但仍可能受以下因素限制：

- 更多权重读取；
- 小 GEMM 利用率不足；
- router 与 permutation；
- 跨卡 all-to-all；
- batch 中专家负载不均。

Mixtral 报告也明确区分 active parameter compute 与总权重内存、硬件利用率。[cite:13]

只记住：**FLOPs、显存容量、带宽与通信是不同成本。**

停下来回答：A3B MoE 是否必然比 7B dense 在任意 batch 下延迟更低？

---

### 第 76 步：Shared experts 也会消耗每 token 计算

共享专家服务所有 token，因此它们通常计入 active parameters。

比较 Qwen2 MoE、Qwen3 MoE 和 DeepSeekMoE 时，必须确认 `AxxB` 的统计是否包含共享专家以及 attention 等非专家参数。

只记住：**active parameter 的统计口径也要核对。**

停下来回答：只知道 routed top-k，能否精确算出整模型 active parameters？

---

### 第 77 步：推理框架必须有对应 MoE kernel 和通信实现

模型报告证明一种结构可训练，不代表任意推理框架都能高效运行它。

部署时还需检查：

```text
expert GEMM kernel
token dispatch/combine
EP/TP 组合
all-to-all backend
量化支持
连续批处理下的负载形状
```

只记住：**checkpoint 可加载和模型可高效服务是两项验收。**

停下来回答：模型成功生成一条文本，是否证明 MoE serving 吞吐已经优化？

---

### 第八阶段检查

读到 “MoE 只激活 3B 参数” 时，继续追问：

1. 总权重多大？
2. top-k 与共享专家是多少？
3. 专家怎样放到 GPU？
4. all-to-all 用什么链路？
5. batch 下专家负载是否均匀？
6. kernel 是否能形成足够大的 GEMM？

---

## 第九阶段：用其他报告校准 Qwen 与 DeepSeek

### 第 78 步：Mistral 7B 展示了“少改结构也能形成明确主题”

Mistral 7B 的核心组合是：

```text
GQA：减少 KV cache
Sliding Window Attention：限制直接关注范围
Rolling Buffer Cache：缓存只保留固定窗口
```

它是一份适合学习“结构选择怎样映射到推理成本”的短报告。[cite:12]

只记住：**报告价值不取决于新术语数量。**

停下来回答：rolling buffer cache 为什么能让缓存不随历史无限增长？

---

### 第 79 步：Mixtral 提供最直观的 MoE 入门坐标

Mixtral 每层有 8 个 FFN 专家，每个 token 选 2 个；总参数约 47B，active parameters 约 13B。[cite:13]

它比 DeepSeekMoE 更容易建立第一层直觉：

```text
一个普通 FFN
→ 替换成 8 个 FFN
→ router 每 token 选 2 个
```

只记住：**先用 Mixtral 理解 top-k MoE，再读细粒度专家和共享专家。**

停下来回答：Mixtral 的 “8x7B” 是否等于总参数 56B 且每 token 用 56B？

---

### 第 80 步：Llama 3 展示了 dense 路线的系统性扩展

Llama 3 405B 仍采用 dense Transformer。报告把数据、规模与复杂度控制列为三条主要杠杆，并明确选择 dense 以降低训练复杂性。[cite:11]

它提供了一个重要对照：

```text
Qwen/DeepSeek：积极探索 MoE 与新 attention
Llama 3：大规模 dense + 数据与系统工程
```

只记住：**先进模型不必沿同一架构路线演进。**

停下来回答：Llama 3 没有采用 MoE，是否意味着它没有进行系统优化？

---

### 第 81 步：Gemma 2 展示了“小模型也可以从预训练蒸馏”

Gemma 2 的 2B 和 9B 模型在预训练阶段使用大教师的 token probability distribution，而不只学习 one-hot 下一个 token。[cite:14]

这与 R1 Distill 的重点不同：

```text
Gemma 2：教师分布用于小模型预训练
R1 Distill：教师生成 reasoning 轨迹用于后训练
```

只记住：**“蒸馏”是一个家族术语，必须说明蒸馏信号和阶段。**

停下来回答：教师 logits 蒸馏和教师 CoT 数据 SFT 是否是同一种 loss？

---

### 第 82 步：Qwen3-Omni 展示了文本报告之外的多模态扩展

Qwen3-Omni 使用 Thinker-Talker 架构：Thinker 负责理解与文本生成，Talker 负责流式语音 token；Talker 用多 codebook 表示，并由 MTP 模块补充 residual codebooks。[cite:15]

这条线涉及音频 encoder、视觉 encoder、codec、流式调度和跨 stage handoff，已经超出纯文本 LLM 报告的主线。

只记住：**Omni 模型需要在文本 backbone 之外继续检查 modality encoder、codec 和流水线。**

停下来回答：理解 Qwen3 文本模型是否足以解释 Qwen3-Omni 的实时语音生成？

---

## 第十阶段：学会审查 benchmark，而不是抄排行榜

### 第 83 步：先对齐模型阶段和 prompt protocol

benchmark 分数至少受以下条件影响：

```text
Base / Instruct / Reasoning
zero-shot / few-shot
是否使用 CoT
采样温度
最大输出长度
是否使用工具
答案抽取规则
```

Llama 3 报告的表格会明确标出 0-shot、5-shot、CoT 等设置，这些标注是结果的一部分。[cite:11]

只记住：**同名 benchmark 不保证同一评测协议。**

停下来回答：MMLU 5-shot 与 MMLU 0-shot CoT 能否直接当作同一列比较？

---

### 第 84 步：Pass@1、Pass@k 和 majority vote 回答不同问题

`pass@1` 看单次回答成功率。

`pass@k` 看采样 k 次后至少一次成功的概率。

`maj@k` 或 majority vote 则让多次采样投票。

它们使用不同 inference-time compute，不能只看最大数字。R1 报告同时给出单次与 majority-vote 结果时，应保留口径。[cite:9]

只记住：**采样次数是推理预算。**

停下来回答：maj@64 更高是否证明单次生成也同比提高？

---

### 第 85 步：长输出可能同时提高分数和成本

Reasoning 模型常通过更长 CoT 获得更高可验证任务分数。

比较时至少同时记录：

```text
正确率
平均输出 token
P95 输出 token
time-to-first-token
inter-token latency
每题总成本
```

只记住：**准确率与推理成本需要画在同一条 Pareto 曲线上。**

停下来回答：两个模型准确率相同但平均输出长度相差 10 倍，部署成本是否相同？

---

### 第 86 步：作者自建 benchmark 需要额外透明度

公开 benchmark 也可能有污染和过拟合，但至少更容易由第三方复跑。

内部 benchmark 若没有公开样本、grader 和 prompt，只能作为作者证据，难以独立验证。

只记住：**不可复现不等于结果必然错误，但证据等级更低。**

停下来回答：只公开最终平均分、不公开评测协议，会缺少哪些核查能力？

---

### 第 87 步：Contamination analysis 不是一句“已去重”

应检查：

- 匹配单位是文档、句子还是 n-gram；
- 阈值如何设置；
- 测试题变体是否可被识别；
- 训练数据的时间边界；
- 被污染样本是否单独报告。

Qwen2 与 Llama 3 报告都专门讨论 benchmark contamination。[cite:2][cite:11]

只记住：**数据去重和 benchmark 去污染不是同一个检查。**

停下来回答：训练语料内部去重是否自动删除所有测试题变体？

---

### 第 88 步：Ablation 要只改变一个关键变量

理想消融应尽量保持模型规模、数据、训练 token 和评测协议一致，只改变目标组件。

如果同时从 dense 改 MoE、MHA 改 MLA、数据从 2T 扩到 8.1T，就不能把全部变化归因于 MLA。

只记住：**整代模型对比回答“整套方案怎样”，不回答每个组件贡献多少。**

停下来回答：DeepSeek LLM 67B 与 DeepSeek-V2 的整模型对比是否是纯 MLA 消融？

---

### 第 89 步：负结果比口号更能说明研究边界

R1 报告公开讨论了 Process Reward Model 与 Monte Carlo Tree Search 的未成功尝试。[cite:9]

负结果能说明：

- 哪些假设在当前规模下没有成立；
- 失败来自算法、数据还是系统；
- 后续研究不应重复哪些未经修改的路径。

只记住：**“没有成功”也是技术报告的重要证据。**

停下来回答：一个方法在 R1 设置下失败，是否证明它在所有模型和预算下都不可能有效？

---

### 第 90 步：SOTA 必须绑定任务、协议、日期与可比对象

模型团队常写 “state-of-the-art” 或 “comparable to frontier models”。

准确阅读方式是补全：

```text
在哪些 benchmark
哪个 checkpoint
什么 prompt 和采样配置
对比对象是什么版本
结果由谁运行
发布日期是什么
```

只记住：**SOTA 是条件结论，不是模型永久属性。**

停下来回答：一张跨团队、不同协议拼接的表能否独立证明普遍 SOTA？

---

## 第十一阶段：把论文术语映射到配置与代码

到这里才开始看实现。不要一上来通读整个模型文件。

### 第 91 步：先从 `config.json` 建立结构表

对 Qwen 或 DeepSeek checkpoint，先找：

```text
hidden_size
num_hidden_layers
num_attention_heads
num_key_value_heads
intermediate_size
num_experts
num_experts_per_tok
rope_theta
max_position_embeddings
```

这一步能把型号名映射成真实层数、头数和专家配置。

新认知点：**报告中的架构表怎样进入可加载配置。**

---

### 第 92 步：再沿一个 decoder layer 看数据流

只追踪：

```text
input hidden states
→ norm
→ attention
→ residual
→ norm
→ dense FFN 或 MoE
→ residual
```

先确认模块边界，再进入 MLA、GQA 或 router 内部。

新认知点：**论文中的 block 图怎样变成 forward 调用。**

---

### 第 93 步：读 GQA 时只核对 Q/K/V shape

关注四个量：

```text
Q heads
KV heads
head dimension
KV 怎样扩展或广播到 Q groups
```

不要先陷入 fused kernel。先证明张量契约正确。

新认知点：**GQA 的共享关系怎样体现在 shape。**

---

### 第 94 步：读 MLA 时区分训练表达式与部署表达式

MLA 论文公式可以按展开后的 Q/K/V 描述训练。

高效 decode 实现可能吸收部分 projection，并直接围绕 latent cache 组织 kernel。两条表达式数学等价，但内存布局和计算顺序不同。

新认知点：**同一 attention 的数学图与部署图可以不同。**

---

### 第 95 步：读 MoE 时沿 token 生命周期追踪

按下面顺序查代码：

```text
router logits
→ top-k expert IDs / weights
→ token sort or dispatch
→ expert GEMM
→ weighted combine
→ restore token order
```

再检查 shared experts、load-balance stats 和 expert parallel communication。

新认知点：**MoE 的关键不是一个 `topk()` 调用，而是完整 dispatch-compute-combine。**

---

### 第 96 步：在本仓库先读 Qwen3-Omni pipeline，而不是纯文本 Qwen

当前 `vllm-omni` 仓库的直接重点是 Omni pipeline。可按顺序打开：

```text
recipes/Qwen/Qwen3-Omni.md
vllm_omni/model_executor/models/qwen3_omni/pipeline.py
vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py
vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py
vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_talker.py
vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_code_predictor_mtp.py
```

这里对应报告中的 Thinker、Talker、MoE 与 MTP，而不是 DeepSeek MLA 主线。

新认知点：**先判断仓库实现的是哪一篇报告中的哪部分。**

---

### 第 97 步：部署配置回答的是 stage 怎样放置

继续看：

```text
vllm_omni/deploy/qwen3_omni_moe.yaml
vllm_omni/deploy/qwen3_omni_moe_thinking.yaml
vllm_omni/deploy/qwen3_omni_moe_mrv2.yaml
```

这些文件主要描述 pipeline、并行与 runtime 选择。它们不是模型训练报告，但决定模型如何映射到实际设备。

新认知点：**模型结构和部署拓扑需要分别阅读。**

---

### 第十一阶段检查

现在你应该能按四层读实现：

```text
checkpoint config
→ 单层数学结构
→ kernel / communication
→ pipeline / deployment
```

不要从部署 YAML 反推训练机制，也不要从论文公式假定 runtime 已经高效实现。

---

## 第十二阶段：形成自己的技术报告阅读方法

### 第 98 步：第一次只读摘要、架构图和主表

第一次阅读只回答：

1. 这份报告试图解决什么瓶颈？
2. 相对上一代改了哪三件事？
3. 最主要的证据表在哪里？
4. 哪些结论是作者自报？

不要在第一次阅读就推完所有公式。

---

### 第 99 步：第二次只读一个创新的完整证据链

例如读 MLA：

```text
问题：MHA KV cache 太大
→ 方法：KV joint compression + decoupled RoPE
→ 实现：缓存 latent
→ 消融：与 MHA/GQA 对比
→ 系统结果：显存、吞吐、延迟
→ 边界：硬件、batch、context、精度
```

每次只追一条链。

---

### 第 100 步：第三次核对代码和模型配置

核对：

- 参数是否与表格一致；
- shared experts 是否存在；
- QKV bias、QK-Norm 是否一致；
- 最大 context 是原生训练还是外推；
- dtype 与量化边界是否一致。

报告、config 和代码三者不一致时，先记录版本，不要自行选择最顺眼的解释。

---

### 第 101 步：最后才做跨模型比较

建议使用下表，而不是只列 benchmark：

| 维度 | Qwen3 | DeepSeek-V3 | Llama 3 | Gemma 2 |
|---|---|---|---|---|
| 主干 | dense + MoE | MoE | dense | dense |
| Attention | GQA | MLA | GQA | local/global GQA |
| FFN | dense / fine-grained MoE | DeepSeekMoE | dense | dense |
| 主要后训练 | thinking fusion + RL | SFT + GRPO | SFT + RS + DPO | SFT + RLHF |
| 主要系统主题 | 多规格与蒸馏 | FP8 + DualPipe + EP | 大规模稳定 dense | TPU + distillation |

这张表只给阅读坐标，不代表在未统一协议下给模型排名。

---

### 第 102 步：用一句话准确概括每条技术路线

闭卷尝试说出：

- **Qwen2/2.5：** 用 GQA 与多规格 dense/MoE 建立通用家族，再主要通过数据与多阶段后训练扩展能力。
- **Qwen3：** 在同一 checkpoint 中统一 thinking/non-thinking，并用 thinking budget 和蒸馏覆盖不同推理成本。
- **DeepSeek-V2：** 用 MLA 压缩 KV cache，用细粒度 MoE 扩大容量。
- **DeepSeek-V3：** 在 V2 架构上联合优化负载均衡、MTP、FP8 和分布式训练系统。
- **DeepSeek-R1：** 在 V3 Base 上通过多阶段 RL/SFT 管线训练 reasoning 行为，再把轨迹蒸馏给小模型。
- **Llama 3：** 以稳定 dense 架构承载数据、规模和系统扩展。
- **Mistral/Mixtral：** 分别提供滑窗 GQA 与经典 top-2 MoE 的清晰入门坐标。
- **Gemma 2：** 用 local/global attention 与预训练蒸馏提高实用尺寸模型能力。

---

## 第十三阶段：理解 2025–2026 架构路线怎样重新汇合

### 第 103 步：新一代模型不再只沿一种 attention 路线演进

2025–2026 年的重要报告形成了多条并行路线：

```text
完整 attention：MiniMax-M2
滑窗 + 全局 attention：Gemma 3、gpt-oss、MiMo-V2-Flash
MLA + 稀疏 attention：DeepSeek-V3.2、GLM-5
GDN + 稀疏 attention：Qwen3.8-Next
Mamba + 少量 attention：Nemotron 3
```

这些路线都试图降低长上下文训练或生成成本，但对历史 token 的表示、直接访问能力和 kernel 要求不同。[cite:10][cite:19][cite:20][cite:22][cite:23][cite:24][cite:25][cite:29]

只记住：**“高效 attention”不是一项技术，而是一组互不等价的取舍。**

停下来回答：为什么不能用“线性复杂度”四个字概括所有新模型？

---

### 第 104 步：Muon 改变的是矩阵参数的优化更新

AdamW 为每个参数维护一阶和二阶统计。Muon 则对二维矩阵的 momentum update 做近似正交化，再匹配更新尺度。

Kimi K2、GLM-4.5 和 Qwen3.8-Next 都把 Muon 用于部分矩阵参数，但 embedding、normalization、router 或特殊低秩参数仍可能使用 AdamW。[cite:16][cite:17][cite:23]

只记住：**使用 Muon 不表示全模型所有参数都由同一更新规则处理。**

停下来回答：为什么 embedding 或 router 可能继续使用 AdamW？

---

### 第 105 步：QK-Clip 从参数更新侧控制 attention logit

Kimi K2 报告指出，大规模 Muon 训练更容易出现 attention logit 爆炸。QK-Clip 在 optimizer step 后，根据每个 head 的最大 logit 缩放 Q/K 投影权重。[cite:16]

这与 QK-Norm 不同：

```text
QK-Norm：在 forward 中归一化 Q/K
QK-Clip：在 update 后约束 Q/K 权重增长
```

Kimi K2 还针对 MLA 只缩放不共享的 Q/K 分量，避免影响跨 head 共享部分。

只记住：**QK-Clip 是训练稳定性干预，不是推理时 clipping。**

停下来回答：QK-Clip 与 softmax 前直接裁剪 logits 是否发生在同一位置？

---

### 第 106 步：Attention sink bias 给 softmax 增加“不关注任何 token”的选项

gpt-oss 和 MiMo-V2-Flash 在 softmax 分母中加入一个可学习的 sink bias。[cite:20][cite:25]

如果所有可见 token 都不值得关注，概率质量可以流向这个不携带 value 的额外项，而不必强迫某个历史 token 获得注意力。

MiMo-V2-Flash 报告观察到，在其 128-token 滑窗消融中，加入 sink 后的 hybrid attention 优于未加入版本；这不是所有滑窗模型的普遍定理。[cite:25]

只记住：**attention sink 允许模型选择“不给任何实际 value 分配这部分概率”。**

停下来回答：sink bias 是否等于在序列开头增加一个普通可训练 token？

---

### 第 107 步：QSA 和 DSA 都先索引，但选择粒度不同

DeepSeek Sparse Attention（DSA）由 lightweight indexer 为历史位置打分，再选择 top-k KV entries。[cite:10]

Qwen Sparse Attention（QSA）先压缩历史为 micro-block 表示，用 compressed indexer 选择相关 block，再对 block 内原始 token 做 attention。[cite:23]

可以先记为：

```text
DSA：偏 token 级选择
QSA：先 block 级索引，再访问 block 内 token
```

只记住：**索引器自身的复杂度和选择粒度也是 sparse attention 成本的一部分。**

停下来回答：只比较最终 top-k 数量，能否完整比较 QSA 与 DSA？

---

### 第 108 步：稀疏 attention 常通过 continued pretraining 接入旧模型

DeepSeek-V3.2 先冻结主模型并训练 indexer 模仿 dense attention 分布，再开放全模型适应 sparse pattern。[cite:10]

Qwen3.8-Next 和 GLM-5 也把稀疏 attention 放在后续训练阶段，而不是从第一个 token 就完全使用最终 sparse pattern。[cite:23][cite:24]

这样做的目标是复用已有 dense/hybrid checkpoint，同时降低结构切换造成的质量损失。

只记住：**attention 替换本身也需要训练课程。**

停下来回答：为什么不能把 dense checkpoint 的 attention kernel 直接换成 top-k sparse kernel 就宣布完成？

---

### 第 109 步：Gated Residual 把残差流从一条扩展成多条

Qwen3.8-Next 将 residual stream 扩展为四个分支，并用逐元素 gate 控制每个子层读取和写回哪些分支。[cite:23]

它不同于增加 hidden size：

```text
hidden widening：扩大单个向量
residual branching：保留多条并行状态流
```

报告还把 gate 与高学习率 stress test 下的稳定性联系起来，但这种机制解释来自作者消融，仍应限制在已测结构内。

只记住：**Gated Residual 改的是跨层状态传递方式。**

停下来回答：四分支 residual 是否表示 attention head 数也变成四倍？

---

### 第 110 步：N-gram embedding 把容量放到主干之外

Qwen3.8-Next 使用一个大规模 n-gram embedding 表，根据连续 token 组合查表并注入早期层。[cite:23]

报告中的模型有额外 51B n-gram 参数，表放在 host memory 并提前 prefetch，而不是常驻 accelerator。

这条路线用存储和传输增加模式记忆容量，几乎不增加大矩阵 FLOPs，但会引入：

- 大规模 host memory 占用；
- 查表与预取；
- cache locality；
- CPU–GPU 带宽约束。

只记住：**不计入 active backbone FLOPs 的参数仍然有存储和搬运成本。**

停下来回答：额外 51B n-gram 参数为什么不能被称为“零成本容量”？

---

### 第 111 步：LatentMoE 压缩的是 routed expert 的工作空间

Nemotron 3 的 LatentMoE 先把 token 从 hidden dimension \(d\) 投影到较小 latent dimension \(\ell\)，在 latent 空间进行路由和 expert FFN，再投影回 \(d\)。[cite:22]

这样可以减少：

```text
expert 权重读取
all-to-all payload
```

报告把节省的预算重新投入到更多总专家和更大的 top-k。它与 MLA 都使用“latent”一词，但一个改 attention cache，一个改 MoE expert space。

只记住：**LatentMoE 与 Multi-head Latent Attention 不是同一个模块。**

停下来回答：LatentMoE 能否直接减少 attention 的 KV cache？

---

### 第 112 步：NVFP4 训练需要为敏感层保留更高精度

Nemotron 3 报告使用原生 NVFP4 GEMM 训练大规模 hybrid Mamba-MoE，并采用细粒度 scale、Hadamard transform、stochastic rounding 等方法。[cite:22]

它仍将一部分对象保留为 BF16、FP8 或其他精度，例如：

- 少量 attention 的 QKV 与输出投影；
- Mamba 中容易 flush-to-zero 的投影；
- LatentMoE projection；
- MTP 层和网络尾部。

只记住：**FP4 训练的关键是确定哪些张量不能降到 FP4。**

停下来回答：看到“NVFP4 pretraining”时，为什么不能推断 checkpoint 中每个 tensor 都是 4 bit？

---

### 第 113 步：QAT 与低精度预训练解决不同问题

Gemma 3/4 在完成主要训练后，用 Quantization-Aware Training 适配 int4、Q4 或移动端低比特格式。[cite:29][cite:30]

DeepSeek-V3 的 FP8 和 Nemotron 3 的 NVFP4 则主要服务大规模训练过程。[cite:8][cite:22]

```text
低精度预训练：减少训练算力与显存成本
QAT：让最终低比特部署 checkpoint 适应量化误差
```

只记住：**训练精度和发布权重量化是两个时间点。**

停下来回答：一个 BF16 训练模型能否在训练后通过 QAT 得到 int4 部署版本？

---

### 第 114 步：MTP 已演化出三种部署形态

到 2026 年，报告中至少出现三种 MTP 形态：

1. 主干附加并行未来 token prediction head，如 DeepSeek-V3。
2. 多层轻量 MTP block，直接作为 speculative draft，如 MiMo-V2-Flash。
3. 独立小型 autoregressive drafter，cross-attend 主模型 KV，如 Gemma 4。[cite:8][cite:25][cite:30]

它们都可支持 speculative decoding，但状态复用、训练目标、每步开销和接受长度不同。

只记住：**“支持 MTP”不足以确定 speculative decoding 的实现。**

停下来回答：为什么 MTP 层数越多不一定端到端越快？

---

### 第 115 步：Loss、benchmark、效率和稳定性必须分开验收

Qwen3.8-Next 报告专门展示了这些指标会分歧：

- 更大 n-gram vocabulary 持续降低 loss，但 benchmark 收益饱和；
- 某些 residual 设计 loss 差异很小，却影响下游结果；
- 某些简化在短训练看不出退化，继续训练后才暴露。[cite:23]

因此架构搜索不能只看 validation loss，也不能只看单张 benchmark 表。

只记住：**一个候选架构至少要同时检查能力、训练成本、prefill、decode 和稳定性。**

停下来回答：validation loss 更低是否足以证明新结构更适合生产部署？

---

### 第十三阶段检查

你现在应该能区分：

| 技术 | 主要作用位置 |
|---|---|
| QK-Clip | optimizer update 后 |
| attention sink | softmax denominator |
| QSA / DSA | 长上下文位置选择 |
| Gated Residual | 跨层 residual state |
| n-gram embedding | 主干外部容量 |
| LatentMoE | routed expert latent space |
| NVFP4 | 训练数值路径 |
| QAT | 部署权重适配 |

---

## 第十四阶段：认识 agentic post-training 已成为独立系统

### 第 116 步：Agent 训练样本不再只是 prompt–response

普通 SFT 样本可以是：

```text
prompt → answer
```

Agent 轨迹则包含：

```text
观察
→ 推理
→ tool call
→ 环境返回
→ 再推理
→ 多轮动作
→ 最终产物
```

Kimi K2、GLM-4.5、MiniMax-M2、GLM-5 和 Qwen3.8-Omni 都把环境、工具和长轨迹作为模型训练的一部分。[cite:16][cite:17][cite:19][cite:24][cite:28]

只记住：**agentic data 的基本单位是有环境状态的 trajectory。**

停下来回答：为什么静态答案数据无法完整监督 tool failure 后的恢复行为？

---

### 第 117 步：Kimi K2 把 token efficiency 与 agent training 放在同一报告

Kimi K2 是 `1T-A32B` 的 ultra-sparse MoE，沿用 MLA，并用 MuonClip 在 15.5T token 预训练中控制稳定性。[cite:16]

它的技术主线不是只有更大 MoE，而是：

```text
MuonClip 提高训练 token efficiency
→ 数据重写提高每 token 学习信号
→ agent trajectory synthesis
→ verifiable + self-critic RL
```

只记住：**K2 把预训练 token efficiency 与后训练 agent intelligence 视为连续问题。**

停下来回答：Kimi K2 的 agent 能力是否只来自 MoE 参数量？

---

### 第 118 步：Data rephrasing 试图提高同一知识的可学习性

Kimi K2 将部分知识、数学和高质量语料重写成更适合模型顺序学习的表达，而不是只增加新文档。[cite:16]

它与 Phi-4 的 synthetic curriculum 有相似直觉：同一事实或技能的组织方式会影响 next-token prediction 的学习难度。[cite:36]

但重写也可能引入教师风格、事实错误和分布收缩，需要过滤与原始数据混合。

只记住：**数据 token 数不变时，表达结构仍可能改变 token utility。**

停下来回答：重写后的文本为什么不能完全替代自然语料？

---

### 第 119 步：高质量 agent 数据必须真正执行

Kimi K2 的 pipeline 生成工具、任务和多轮 trajectory，并把 synthetic environments 与真实 execution environments 结合。[cite:16]

只看格式正确的 tool call 不足以证明动作有效。更强的过滤需要检查：

- 工具是否真实返回预期状态；
- 代码是否通过测试；
- 文件或网页产物是否存在；
- 多轮状态是否一致；
- 最终答案是否引用真实 observation。

只记住：**agent 数据质量依赖环境执行证据。**

停下来回答：JSON function call 能成功解析，是否等于任务已经完成？

---

### 第 120 步：Self-critic reward 扩展了不可规则验证的任务

数学答案或代码测试可以使用 rule-based reward。写作、综合分析和开放式 agent 任务常没有唯一答案。

Kimi K2 为这些任务使用 rubric，让模型或 critic 根据多维标准评价回答，并通过 closed-loop refinement 改进 critic。[cite:16]

这种 reward 覆盖面更广，但会引入 judge bias、self-preference 和 reward hacking。

只记住：**model-based reward 扩展可训练任务，也降低 reward 的客观性。**

停下来回答：critic 给出高分是否等同于真实用户一定满意？

---

### 第 121 步：GLM-4.5 把 mid-training 明确成独立阶段

GLM-4.5 在通用 pretraining 后加入 repo-level code、synthetic reasoning、long-context 与 agent trajectory 的 mid-training，并逐步把 context 从 4K 扩到 128K。[cite:17]

mid-training 位于大规模自然语料预训练与 assistant post-training 之间：

```text
pretraining：建立广泛基础
mid-training：定向重塑能力与上下文
post-training：对齐行为与交互
```

只记住：**mid-training 不是 SFT 的另一个名字。**

停下来回答：为什么 repo 级代码与 issue/PR/commit 数据适合放在 mid-training？

---

### 第 122 步：Reasoning RL、Agentic RL 和 General RL 的目标不同

GLM-4.5 和 GLM-5 都把 RL 拆成多个阶段：[cite:17][cite:24]

```text
Reasoning RL：数学、科学、代码等可验证推理
Agentic RL：环境交互、工具选择、长轨迹
General RL：偏好、指令、风格与广泛行为
```

一个阶段的优化可能损害另一个阶段的能力，因此报告会加入 distillation、replay 或混合 reward 防止遗忘。

只记住：**不同 RL 阶段使用不同任务分布和 reward。**

停下来回答：数学 reward 提高后，为什么仍需要 general RL？

---

### 第 123 步：CISPO 裁剪 importance ratio 而不是丢弃 token

MiniMax-M1 提出 CISPO，用 importance sampling weight clipping 稳定 policy update，同时让所有 token 继续参与梯度计算。[cite:18]

它的设计动机是避免某些 PPO/GRPO 变体中，大量超出 clip 范围的 token 对训练不再贡献有效梯度。

报告中的效率结果属于特定 Qwen2.5-32B 与训练实现对照，不能直接推广到任何模型和 reward。

只记住：**RL clipping 规则会改变哪些 token 继续提供学习信号。**

停下来回答：CISPO 的核心差异发生在 sampling、reward 还是 policy objective？

---

### 第 124 步：MiniMax-M1 用 Lightning Attention 降低超长 CoT 成本

MiniMax-M1 每七个 Lightning Attention block 插入一个 full softmax attention block，支持 1M context，并训练 40K/80K thinking budget 模型。[cite:18]

报告给出随生成长度增长的理论 FLOPs 对比，并在其假设下计算出混合线性 attention 在 64K–100K 输出时的计算优势。

这主要回答“长推理怎样更便宜”，不自动回答“固定状态是否保留所有多跳信息”。

只记住：**长 CoT 效率与长 CoT 质量要分别验证。**

停下来回答：理论 FLOPs 更低是否证明多跳 agent 任务一定更强？

---

### 第 125 步：MiniMax-M2 是对混合 attention 路线的重要反例

MiniMax-M2 后来改回每层 full attention。报告公开指出：早期 hybrid/SWA proxy 在常见 benchmark 上接近 full attention，但大规模多跳推理、in-context learning 和超过 32K 的 agent 任务出现差距。[cite:19]

这不证明 linear/SWA 永远无效，而是说明：

```text
小模型 proxy
短 benchmark
短训练 horizon
```

可能无法预测最终 long-horizon agent 质量。

只记住：**架构路线可以根据更强证据被推翻。**

停下来回答：M1 与 M2 的选择冲突为什么不是简单的“其中一篇一定错误”？

---

### 第 126 步：MOPD 让多个领域教师提供 on-policy token reward

MiMo-V2-Flash 先训练数学、代码、agent 等领域教师，再让学生自己采样 trajectory，由教师对学生当前 token 分布提供 dense reward，同时保留 outcome reward。[cite:25]

它与普通离线蒸馏的区别是：

```text
离线蒸馏：学习固定教师数据
on-policy distillation：评价学生当前自己生成的轨迹
```

只记住：**MOPD 试图同时保留学生探索分布和教师细粒度反馈。**

停下来回答：MOPD 是否只对教师预先生成的静态答案做 SFT？

---

### 第 127 步：异步 RL 把 rollout 与 training 解耦

同步 RL 中，trainer 常等待最慢 rollout 完成。长 agent trajectory 的长度差异会造成大量空闲。

GLM-5 与 MiniMax-M2 将 generation、training 和 environment executor 解耦，让 rollout 持续生产数据、trainer 持续消费 batch。[cite:19][cite:24]

收益是更高利用率；代价是 policy staleness 与数据分布偏移。

只记住：**异步 RL 用 off-policy 程度换取系统吞吐。**

停下来回答：rollout 越异步，为什么采样 policy 与当前训练 policy 的差距可能越大？

---

### 第 128 步：Routing replay 解决 train–infer MoE 路由不一致

MiMo-V2-Flash 的 Rollout Routing Replay（R3）记录 rollout 时的 expert routing，并在训练时复用，减少 FP8 inference 与 training 实现差异导致的路由漂移。[cite:25]

对 MoE 而言，轻微数值差异可能改变 top-k expert ID；这不仅改变 logits，还改变实际执行子网络。

只记住：**MoE 的 train–infer consistency 包含 expert route 一致性。**

停下来回答：为什么 dense 模型的轻微 logit 差异与 MoE 的 top-k route 差异影响不同？

---

### 第 129 步：Agent-native RL 系统必须处理长度尾部

MiniMax-M2 的 Forge 使用 windowed-FIFO、prefix-tree merging 和训练/推理/agent 解耦，重点处理长短差异巨大的 trajectory。[cite:19]

它还报告 M2.7 能修改自身 agent scaffold 和调试训练任务。准确表述应是“在报告定义的 ML-engineering 环境中观察到自我改进流程”，不能直接扩展为通用自主递归改进。

只记住：**agent 系统结论必须绑定具体环境、权限和验收器。**

停下来回答：模型能修改一个 scaffold，是否证明它能安全地修改自身全部训练系统？

---

### 第十四阶段检查

你现在应该能画出 agent RL 的完整系统：

```text
task/environment
→ rollout workers
→ tool execution
→ trajectory + artifact
→ verifier / critic / reward
→ replay or scheduler
→ policy training
→ weight synchronization
```

任何一段不可靠，都会让最终 reward 失真。

---

## 第十五阶段：从 VLM 继续走向原生 omni agent

### 第 130 步：多模态模型首先要回答“不同模态在哪里融合”

常见路线包括：

```text
独立 encoder → adapter → LLM
多层 vision feature → 多个 LLM 层
原始 patch/audio chunk → 直接投影到 LLM
离散视觉/音频 token → 自回归 backbone
```

它们分别对应 InternVL、Qwen3-VL DeepStack、Gemma 4 encoder-free 和 Janus/Omni 类路线。[cite:26][cite:30][cite:31][cite:33]

只记住：**“原生多模态”必须进一步说明 encoder、tokenization 和 fusion point。**

停下来回答：所有 VLM 都必须使用独立 ViT 吗？

---

### 第 131 步：Gemma 3 用固定数量视觉 token 控制图像成本

Gemma 3 使用 SigLIP vision encoder，把图像压缩成固定 256 个视觉向量，再交给 language model；高分辨率图像可通过 Pan & Scan 切成多个 crop。[cite:29]

固定 token 数让单图成本可控，但多 crop 会重新增加 context token。

只记住：**视觉分辨率、crop 数和视觉 token 数共同决定 VLM 成本。**

停下来回答：一张超高分辨率图经过 Pan & Scan 后仍一定只有 256 个总视觉 token 吗？

---

### 第 132 步：Interleaved-MRoPE 重新分配时空频率

传统 M-RoPE 把时间、高度、宽度分配到连续的 rotary 维度段。

Qwen3-VL 把 \(t/h/w\) 交错分布到不同频率维度，让每个轴同时覆盖高频与低频位置特征。[cite:26]

只记住：**Interleaved-MRoPE 改的是多模态位置频率分配。**

停下来回答：它是否改变 image patch 本身的视觉内容编码？

---

### 第 133 步：DeepStack 把多个 ViT 层接入 LLM

许多 VLM 只使用 vision encoder 最后一层。

Qwen3-VL 的 DeepStack 将不同 ViT 层的 feature 通过轻量 residual connection 注入对应 LLM 层，保留低层细节与高层语义。[cite:26]

这会增加跨层接口和 feature 对齐复杂度，但可能改善 OCR、细粒度定位和空间理解。

只记住：**DeepStack 是跨层视觉语言融合，不是更大的图像输入。**

停下来回答：只读取 ViT 最后一层与 DeepStack 的信息入口有什么差别？

---

### 第 134 步：Text timestamp 让视频时间成为显式 token

Qwen3-VL、Qwen3.5-Omni 和 Qwen3.8-Omni 都逐步使用文本化 timestamp，把帧或音频片段的绝对时间显式写入序列。[cite:26][cite:27][cite:28]

与只依赖 temporal RoPE 相比，文本 timestamp：

- 更容易跨帧率表达真实秒数；
- 可直接参与语言推理；
- 会增加 context token；
- 依赖训练数据中的时间标注质量。

只记住：**位置编码和显式时间语义可以同时存在。**

停下来回答：为什么视频第 100 帧不一定代表相同的真实时间？

---

### 第 135 步：Visual Resolution Router 按 patch 内容决定压缩率

InternVL3.5-Flash 为每个 image patch 在较高和较低视觉 token 压缩率之间路由。[cite:31]

语义丰富区域可保留更多 token，简单区域可压缩更多。该策略与 Dynamic High Resolution 的区别是：后者主要决定空间切块，前者进一步根据内容决定 patch token 数。

只记住：**动态视觉 token 预算可以细化到 patch。**

停下来回答：ViR 减少视觉 token 后，为什么仍需要一致性训练？

---

### 第 136 步：DvD 把 vision encoder 与 LLM 放到不同 GPU

InternVL3.5 的 Decoupled Vision-Language Deployment 将 ViT 与 LLM 分离部署，试图平衡视觉前处理和语言生成的设备负载。[cite:31]

这是一项 serving topology 优化，不改变 checkpoint 的数学定义。它要求：

- 跨设备传输视觉 feature；
- 独立调度 vision 与 language batch；
- 管理不同阶段的背压。

只记住：**多模态 pipeline 的瓶颈可能位于 encoder，而不是 LLM decode。**

停下来回答：DvD 加速是否需要重新训练模型权重？

---

### 第 137 步：Multimodal RL 需要先稳定 perception 再优化 reasoning

InternVL3.5 的 Cascade RL 先做 offline RL，建立较稳定的 reasoning distribution，再做 online RL 精调。[cite:31]

Seed1.5-VL 等报告也把视觉 reasoning、GUI 和 game task 作为独立后训练对象。[cite:32]

多模态 reward 除答案外还可能验证：

- bounding box 或 point；
- OCR 文本；
- GUI action；
- video timestamp；
- embodied state transition。

只记住：**多模态 RL 的 verifier 必须理解视觉或环境状态。**

停下来回答：纯文本 reward model 能否可靠判断点击位置是否正确？

---

### 第 138 步：Janus-Pro 为理解和生成使用不同视觉编码器

Janus-Pro 用 SigLIP feature 做图像理解，用 VQ tokenizer 的离散 ID 做图像生成，最后由同一 autoregressive Transformer 处理两类序列。[cite:33]

这解决了一个冲突：

```text
理解需要高层语义 feature
生成需要可还原像素的离散 code
```

只记住：**统一 backbone 不要求统一视觉 tokenizer。**

停下来回答：Janus-Pro 为什么不强迫理解和生成共用同一图像表示？

---

### 第 139 步：ARIA 动态对齐文本与语音 token 速率

Qwen3.5-Omni 的 Talker 同时处理文本单元和多 codebook 语音单元。二者 token rate 不一致时，固定 interleave 容易出现漏词、重复或韵律不自然。[cite:27]

ARIA 根据当前文本与语音生成进度动态决定交错关系，目标是在保持低延迟的同时改善 streaming speech stability。

只记住：**流式语音生成不仅是 codec prediction，还需要跨 token-rate 对齐。**

停下来回答：文本 token 和音频 codec frame 为什么不能简单一一对应？

---

### 第 140 步：Qwen3.8-Omni 把多模态能力延伸到 harness

Qwen3.8-Omni 不只发布模型，还发布 Qwen-MM-Plugins 与 Qwen-Live-Harness，用于长视频摘要、lazy loading、实时 tool use、sub-agent 和 persistent memory。[cite:28]

这说明 omni agent 的可用性来自两层：

```text
模型：感知、推理、语音和 tool policy
系统：memory、插件、调度、流式 I/O
```

只记住：**原生多模态模型仍需要 agent runtime。**

停下来回答：模型支持 1M context 是否意味着应该把所有原始视频 token 永久放在上下文中？

---

### 第 141 步：Agent Swarm 把 test-time compute 扩展到并行子 agent

Kimi K2.5 在 text–vision joint pretraining、zero-vision SFT 和 joint RL 基础上，引入 Agent Swarm：主 agent 自行拆任务，并并行调用多个子 agent。[cite:35]

这种 scaling 与单模型长 CoT 不同：

```text
长 CoT：一条串行 trajectory
Agent Swarm：多条并行 trajectory + 汇总
```

收益取决于任务可分解性、子任务独立性、通信成本和最终聚合质量。

只记住：**parallel test-time compute 需要任务分解器。**

停下来回答：把同一个不可分解问题复制给更多 agent 是否一定提高正确率？

---

### 第 142 步：Gemma 4 12B 尝试取消独立视觉与音频 encoder

Gemma 4 12B 直接把原始 image patch 与 40ms audio chunk 通过轻量 projection 映射到 LLM embedding space，不使用独立 ViT 或 audio encoder。[cite:30]

潜在收益包括更少模型组件和更低 memory fragmentation；代价是 LLM 主干需要自己学习更多底层感知变换。

只记住：**encoder-free 是减少模块边界，不是没有任何输入投影。**

停下来回答：原始 RGB patch 能否不经过任何线性映射直接进入语言 hidden state？

---

### 第 143 步：图像生成报告属于 diffusion 路线，不应硬套 LLM 解码

Qwen-Image-2.0 使用 Qwen3-VL 作为条件 encoder、MMDiT 作为 diffusion backbone，并使用高压缩 VAE 在 latent space 生成与编辑图像。[cite:34]

它的执行链是：

```text
文本/图像条件
→ multimodal encoder
→ diffusion transformer 多步去噪
→ VAE decode
→ 像素图像
```

这与 Janus-Pro 自回归生成离散 visual token 是两条不同路线。

只记住：**“多模态生成模型”不等于 autoregressive LLM。**

停下来回答：MMDiT 的 denoising step 是否等同于 next-token decode step？

---

### 第十五阶段检查

请区分五类 multimodal 设计：

| 设计 | 代表报告 |
|---|---|
| ViT–adapter–LLM | InternVL3.5、Seed1.5-VL |
| 多层视觉 feature 注入 | Qwen3-VL DeepStack |
| encoder-free patch/audio projection | Gemma 4 12B |
| 理解/生成双视觉 tokenizer | Janus-Pro |
| encoder + diffusion backbone + VAE | Qwen-Image-2.0 |

---

## 第十六阶段：按开放程度和文档类型判断证据

### 第 144 步：Open-weight 与 fully open 不是同义词

Open-weight 通常只保证最终 checkpoint 可下载。

Fully open 还可能包括：

```text
训练数据或可重建 data mix
训练代码
中间 checkpoint
post-training 数据
评测 harness
失败实验与成本
```

OLMo 3 把这些完整链路称为 model flow。[cite:21]

只记住：**能本地推理不等于能重建训练过程。**

停下来回答：只有 final safetensors 和 config 的模型属于 fully open 吗？

---

### 第 145 步：OLMo 3 的主要价值是可追踪 model flow

OLMo 3 发布 7B/32B Base、Think、Instruct、RL-Zero，以及 pretraining、mid-training、long-context、SFT、DPO、RL 各阶段数据和中间 checkpoint。[cite:21]

它不一定在所有 benchmark 上领先最大 open-weight 模型，但可以回答其他报告很难回答的问题：

- 某能力在哪个 stage 出现；
- benchmark 是否进入训练集；
- RL 增益是否依赖 mid-training；
- cost 分布在哪些阶段；
- 数据 mix 改动带来哪些 trade-off。

只记住：**研究价值不仅来自最终分数，也来自可追溯性。**

停下来回答：为什么 OLMo 3 更适合研究 pretraining data 对 RL 的影响？

---

### 第 146 步：gpt-oss 把量化格式纳入模型发布契约

gpt-oss-120b/20b 是 open-weight MoE reasoning 模型，MoE 权重以 MXFP4 发布，使 120B checkpoint 约 60.8 GiB，并可放入单张 80GB GPU。[cite:20]

模型卡还公开了：

- top-4 experts；
- alternating banded/full attention；
- GQA 与 128K context；
- low/medium/high reasoning effort；
- H100-hour 训练口径；
- tool-use 与 safety evaluations。

只记住：**发布权重格式会直接决定实际可部署硬件。**

停下来回答：120B total parameters 为什么不一定需要约 240GB BF16 权重空间？

---

### 第 147 步：Chat format 也是模型接口的一部分

gpt-oss 使用 Harmony format，显式区分 system、developer、user、assistant、tool roles，以及 analysis、commentary、final channels。[cite:20]

模型卡特别说明，多轮对话中不应把旧 assistant reasoning trace 原样回填。

这说明 checkpoint 正确加载后，仍需匹配：

```text
special tokens
role hierarchy
channel semantics
tool schema
history policy
```

只记住：**chat template 错误会改变模型行为，而不是纯显示问题。**

停下来回答：为什么把 reasoning trace 当普通 assistant content 回填可能破坏多轮行为？

---

### 第 148 步：Technical report、model card 与 system card 回答不同问题

| 文档 | 通常最强的信息 |
|---|---|
| Technical report | 架构、训练、系统、消融 |
| Model card | checkpoint、接口、用途、限制、安全 |
| System card | 产品路由、系统级 safeguard、能力风险 |
| Release note | 新功能与作者主张，细节可能未完备 |

同一模型可能同时有多种文档。不要用 system card 缺少架构细节来断言模型没有新结构，也不要用 release note 代替完整消融。

只记住：**先判断文档类型，再决定它能支持哪类结论。**

停下来回答：system card 最适合回答模型层数还是部署风险？

---

### 第 149 步：GPT-5 是路由系统，不是单一 checkpoint 名

GPT-5 system card 描述了 fast main model、thinking model、mini/nano 变体和实时 router；ChatGPT 还可使用 parallel test-time compute 的 pro 设置。[cite:38]

因此产品结果可能同时包含：

```text
路由决策
模型选择
thinking budget
tool use
system prompt
safety layer
```

只记住：**系统评测不一定能映射到一个固定模型 forward。**

停下来回答：ChatGPT 中 GPT-5 的一次回答是否必然由同一个 checkpoint 处理所有请求？

---

### 第 150 步：闭源 model card 常能说明边界，却不能支持复现

截至 2026-10-27，Anthropic Transparency Hub 和 Gemini model cards 提供模型输入输出、上下文、能力、安全评测与部署 safeguard，但不公开足以重建模型的层数、权重、训练数据和完整训练配方。[cite:39][cite:40]

这类材料适合回答：

- 产品支持哪些模态；
- context/output 上限；
- 已报告的安全风险；
- safeguard 与使用限制；
- 评测协议的部分信息。

只记住：**透明度报告与可复现技术报告不是同一证据等级。**

停下来回答：知道 Gemini 3.7 Flash 支持 1M context，能否据此还原其 attention 架构？

---

### 第 151 步：Phi-4 说明小模型提升可以主要来自数据工程

Phi-4 基本沿用 Phi-3 架构，但把大量高质量 synthetic data、curriculum、数据混合和 post-training 作为主线。[cite:36]

报告还使用训练截止后发布的 AMC 题和 contamination-aware benchmark 检查过拟合。

只记住：**架构变化小不表示技术报告没有可迁移价值。**

停下来回答：Phi-4 的主要研究问题更接近新 attention，还是数据怎样提高 token utility？

---

### 第 152 步：Phi-4-reasoning 展示了 SFT 仍可建立强 reasoning

Phi-4-reasoning 从 14B Phi-4 出发，使用约 1.4M 个精心选择的 “teachable” prompts 与教师 reasoning demonstrations 做 SFT；`reasoning-plus` 再加入少量 outcome RL。[cite:37]

这提供了与 R1-Zero 不同的路线：

```text
R1-Zero：先探索纯 RL 能否涌现 reasoning
Phi-4-reasoning：先用高质量示范高效注入 reasoning
```

报告还重复运行 AIME，展示小测试集上单次分数可能波动 5–10 个百分点。

只记住：**reasoning 能力可以来自 SFT、RL 或两者组合。**

停下来回答：一次 AIME 分数领先 3 分，是否足以证明模型稳定更强？

---

### 第 153 步：Llama 4 公开 model card，但缺少 Llama 3 级别完整报告

Llama 4 官方 model card 给出 Scout/Maverick 的 MoE、17B active parameters、16/128 experts、早期多模态输入和最大 context 等信息。[cite:41]

但它没有像《The Llama 3 Herd of Models》那样公开完整训练系统、数据、长篇消融与可靠性细节。因此适合进入模型地图，不适合承担深入机制结论。

只记住：**同一家族不同代的公开透明度也会变化。**

停下来回答：知道 expert 数与 active parameters，能否推断 Llama 4 的全部 routing 和训练系统？

---

### 第十六阶段检查

看到一项模型主张时，先标记来源：

```text
完整技术报告
官方 model card
官方 system card
官方 release note
第三方复现
```

不同来源可以互补，但不能互相冒充。

---

## 第十七阶段：建立截至 2026-10-27 的报告地图

### 第 154 步：纳入“值得说”的报告需要明确标准

本文正文优先纳入至少满足一项的报告：

1. 提出可迁移的架构或训练方法；
2. 公开系统实现与效率权衡；
3. 有关键消融或负结果；
4. 开辟 reasoning、agent 或 omni 新路线；
5. 提供异常完整的开放 model flow；
6. 对当前主流路线给出有证据的反例。

只发布新 benchmark 高分、缺少方法细节的版本进入索引，不为每个版本单独增加一步。

只记住：**“所有值得说”不等于“所有发布过”。**

停下来回答：为什么纯 API 价格更新不属于本教程的模型技术主线？

---

### 第 155 步：架构与训练系统必读报告

| 报告 | 最值得读的内容 |
|---|---|
| DeepSeek-V3 / V3.2 | MLA、MoE、FP8、DualPipe、DSA |
| Kimi K2 | MuonClip、token efficiency、agent data |
| gpt-oss | MXFP4、banded/full attention、Harmony |
| Nemotron 3 | Mamba–Transformer、LatentMoE、NVFP4 |
| MiMo-V2-Flash | aggressive SWA、sink、MTP、MOPD |
| Qwen3.8-Next | GDN、QSA、Gated Residual、n-gram table |
| MiniMax-M2 | full-attention 反例与 agent-native system |
| Gemma 4 | encoder-free、KV sharing、MTP drafter |

这些报告共同回答“能力、效率与稳定性怎样联合设计”。[cite:8][cite:10][cite:16][cite:19][cite:20][cite:22][cite:23][cite:25][cite:30]

---

### 第 156 步：Reasoning 与 agent RL 必读报告

| 报告 | 最值得读的内容 |
|---|---|
| DeepSeek-R1 | RL-Zero、多阶段 R1、蒸馏 |
| Qwen3 | thinking fusion、budget、strong-to-weak |
| MiniMax-M1 | CISPO、超长 CoT 与 hybrid attention |
| GLM-4.5 | reasoning/agent/general RL 分层 |
| GLM-5 | asynchronous agent RL、环境扩展 |
| Kimi K2 | verifiable + self-critic reward |
| MiniMax-M2 | artifact-aligned reward、Forge |
| Phi-4-reasoning | teachable SFT + small outcome RL |
| OLMo 3 | fully-open SFT/DPO/RL/RL-Zero model flow |

读这些报告时，重点比较 reward、环境、rollout 系统和能力遗忘，而不是只比较最终 AIME。[cite:4][cite:9][cite:16][cite:17][cite:18][cite:19][cite:21][cite:24][cite:37]

---

### 第 157 步：Multimodal 与 omni 必读报告

| 报告 | 最值得读的内容 |
|---|---|
| Gemma 3 | 视觉 token、5:1 local/global attention |
| Qwen3-VL | Interleaved-MRoPE、DeepStack、timestamp |
| InternVL3.5 | Cascade RL、ViR、DvD |
| Seed1.5-VL | dynamic resolution、video/GUI/game |
| Janus-Pro | 理解/生成视觉编码解耦 |
| Qwen3.5-Omni | Thinker–Talker、ARIA、长音视频 |
| Kimi K2.5 | joint text–vision RL、Agent Swarm |
| Qwen3.8-Omni | 1M omni context、plugin/harness |
| Gemma 4 | encoder-free text/image/audio |

这些报告覆盖 perception、reasoning、generation、streaming 与 agent runtime 五层问题。[cite:26][cite:27][cite:28][cite:29][cite:30][cite:31][cite:32][cite:33][cite:35]

---

### 第 158 步：专项报告按需求阅读

| 方向 | 报告 |
|---|---|
| 小模型与 synthetic data | Phi-4 |
| 多模态图像生成 | Qwen-Image / Qwen-Image-2.0 |
| 统一理解与自回归图像生成 | Janus-Pro |
| 多模态检索 | Qwen3-VL-Embedding / Reranker |
| coding agent | Qwen3-Coder-Next、GLM-5 |
| 完整开放研究 | OLMo 3 |
| 开放推理部署 | gpt-oss |

专项报告应在理解对应 backbone 后阅读。例如先懂 Qwen3-Next，再读 Qwen3-Coder-Next；先懂 Qwen3-VL，再读 multimodal embedding。

还有一组不进入主线、但在对应方向值得保留的报告：

| 方向 | 报告与信息增量 |
|---|---|
| Parallel Attention–SSM | Falcon-H1：并行融合 attention 与 state-space head [cite:42] |
| 小激活 MoE | Hunyuan-A13B：20T pretraining 与低 active-parameter 部署 [cite:43] |
| 自回归图像模型 | HunyuanImage 3.0：统一理解、CoT 与 image generation [cite:44] |
| 360-open model flow | LLM360 K2-V2：数据序列、代码、checkpoint 全链开放 [cite:45] |
| 合规多语言开放模型 | Apertus：开放数据、robots exclusion 与多语言覆盖 [cite:46] |
| 欧洲语言 | EuroLLM-22B：24 个欧盟官方语言及扩展语言 [cite:47] |
| Mistral reasoning | Magistral：RL reasoning pipeline 与 Small 开放版本 [cite:48] |
| 端侧 dense 模型 | Ministral 3：3B/8B/14B 蒸馏、vision 与 256K context [cite:49] |
| 专项语音模型 | Qwen3-TTS、Qwen3-ASR：流式 TTS、voice cloning、ASR 与 forced alignment [cite:50][cite:51] |

---

### 第 159 步：闭源 system cards 用于补足系统与风险视角

截至当前日期，值得保留的官方入口包括：

- GPT-5 system card：router、main/thinking 系统与 safe completion；
- Anthropic Transparency Hub：Claude 5.x 的能力、安全和 safeguard；
- Gemini 3.x model cards：模态、context、评测和 frontier safety；
- Llama 4 model card：公开 checkpoint 的架构与接口边界。[cite:38][cite:39][cite:40][cite:41]

这些材料不进入可复现架构排名，但能提醒读者：用户看到的是“模型 + router + tool + safeguard + runtime”。

---

### 第 160 步：以后更新时只追踪真正改变判断链的报告

后续版本出现时，先问：

```text
是否新增架构？
是否新增训练或 RL 算法？
是否公开系统瓶颈与解决方案？
是否提供可信消融或反例？
是否改变 openness 或复现边界？
```

如果五项都没有，只更新报告索引和时间线，不扩写主教程。

只记住：**学习路线应围绕认知增量，而不是版本号增长。**

停下来回答：一个 benchmark 平均提升 0.5 分、方法完全未披露的新版本，应该进入正文还是索引？

---

## 最终技术地图

```text
标准 decoder Transformer
├─ Attention 效率
│  ├─ GQA：Qwen / Llama / Mistral / Gemma
│  ├─ MLA：DeepSeek / Kimi / GLM
│  ├─ Sliding Window：Mistral / gpt-oss / MiMo
│  ├─ Local + Global：Gemma 2/3/4
│  ├─ Sparse Attention：DeepSeek-V3.2 / GLM-5 / Qwen3.8
│  ├─ GDN + Attention：Qwen3.5/3.8
│  └─ Mamba + Attention：Nemotron 3
│
├─ FFN 容量
│  ├─ Dense：Llama 3
│  ├─ Classic top-2 MoE：Mixtral
│  ├─ Fine-grained MoE：Qwen / DeepSeek / MiniMax
│  └─ LatentMoE：Nemotron 3
│
├─ 状态与容量
│  ├─ Gated Residual：Qwen3.8
│  ├─ N-gram host table：Qwen3.8
│  ├─ KV sharing / K=V：Gemma 4
│  └─ Multi-layer vision features：Qwen3-VL
│
├─ 训练目标
│  ├─ Next-token prediction
│  ├─ MTP：DeepSeek / MiMo / Gemma / Nemotron
│  ├─ Logit distillation：Gemma
│  ├─ On-policy distillation：MiMo
│  └─ Reasoning trajectory distillation：R1 / Qwen / Phi
│
├─ 后训练
│  ├─ SFT / DPO
│  ├─ GRPO / CISPO
│  ├─ Thinking mode fusion
│  ├─ Reasoning / Agent / General RL
│  ├─ Asynchronous RL
│  └─ Multi-agent test-time scaling
│
├─ 多模态
│  ├─ ViT–adapter–LLM：InternVL / Seed
│  ├─ DeepStack：Qwen3-VL
│  ├─ Encoder-free：Gemma 4
│  ├─ Thinker–Talker：Qwen Omni
│  ├─ Dual visual tokenizer：Janus-Pro
│  └─ Encoder–MMDiT–VAE：Qwen-Image
│
└─ 系统
   ├─ FP8 / NVFP4 mixed precision
   ├─ QAT / MXFP4 deployment
   ├─ TP / PP / EP
   ├─ DualPipe / asynchronous rollout
   ├─ all-to-all
   ├─ train–infer routing replay
   └─ speculative decoding
```

---

## 建议阅读顺序

不要按发布时间把全部论文从头到尾读完。先走文本主线：

1. [Mistral 7B](https://arxiv.org/abs/2310.06825)：GQA、滑窗与 rolling cache。
2. [Mixtral of Experts](https://arxiv.org/abs/2401.04088)：最直观的 top-2 MoE。
3. [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671)：dense/MoE、GQA、长上下文。
4. [DeepSeek-V2](https://arxiv.org/abs/2405.04434)：MLA 与细粒度 MoE。
5. [DeepSeek-V3](https://arxiv.org/abs/2412.19437)：FP8、DualPipe、MTP 与系统协同。
6. [DeepSeek-R1](https://arxiv.org/abs/2501.12948)：GRPO、多阶段 reasoning 训练与蒸馏。
7. [Qwen2.5](https://arxiv.org/abs/2412.15115)：数据与多阶段后训练。
8. [Qwen3](https://arxiv.org/abs/2505.09388)：thinking fusion、budget 与强弱蒸馏。
9. [Llama 3](https://arxiv.org/abs/2407.21783)：dense 大规模训练的完整对照。
10. [Gemma 2](https://arxiv.org/abs/2408.00118)：预训练蒸馏与 local/global attention。
11. [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)：稀疏 attention 与 agentic RL 延伸。
12. [Kimi K2](https://arxiv.org/abs/2507.20534)：MuonClip 与 agent data/RL。
13. [gpt-oss](https://arxiv.org/abs/2508.10925)：量化 MoE、reasoning effort 与接口契约。
14. [MiniMax-M1](https://arxiv.org/abs/2506.13585) 和 [MiniMax-M2](https://arxiv.org/abs/2605.26494)：连续阅读架构路线修正。
15. [MiMo-V2-Flash](https://arxiv.org/abs/2601.02780)：SWA、MTP 与 MOPD。
16. [Nemotron 3](https://arxiv.org/abs/2512.20856)：Mamba–Transformer、LatentMoE 与 NVFP4。
17. [Qwen3.8-Next](https://arxiv.org/abs/2608.30320)：GDN、QSA、Gated Residual 与 n-gram table。
18. [OLMo 3](https://arxiv.org/abs/2512.13961)：完整开放 model flow。

再选择一条专题支线：

- **Agent / RL：** GLM-4.5 → GLM-5 → MiniMax-M2 → Phi-4-reasoning。
- **Vision：** Gemma 3 → Qwen3-VL → InternVL3.5 → Gemma 4。
- **Omni：** Qwen3-Omni → Qwen3.5-Omni → Qwen3.8-Omni。
- **统一理解与生成：** Janus-Pro → Qwen-Image-2.0。
- **闭源系统：** GPT-5、Claude 和 Gemini system/model cards，只读系统与风险边界。

如果关注本仓库，再补读：

- [Qwen3-Omni Technical Report](https://arxiv.org/abs/2509.17765)。
- [Qwen3.5-Omni Technical Report](https://arxiv.org/abs/2604.15804)。
- [Qwen3.8-Omni](https://arxiv.org/abs/2609.25611)。
- `recipes/Qwen/Qwen3-Omni.md`。
- `vllm_omni/model_executor/models/qwen3_omni/`。

---

## 闭卷检查

能不看文档回答 18 个以上，再进入论文公式与代码：

1. 总参数与激活参数分别影响什么？
2. Base、Instruct、Reasoning checkpoint 有何不同？
3. GQA 与 MLA 分别怎样减少 KV cache？
4. 为什么 KV cache 变小不代表 full attention 复杂度变成线性？
5. 细粒度 MoE 与 Mixtral top-2 MoE 有何差别？
6. shared expert 与 routed expert 有何差别？
7. auxiliary-loss-free 是否表示没有负载均衡？
8. MTP 在训练与推理中分别做什么？
9. FP8 mixed precision 为什么不等于全模型 FP8？
10. DualPipe 解决什么问题？
11. R1-Zero 与最终 R1 的训练路径有何差别？
12. GRPO 为什么不需要同规模 critic？
13. R1 Distill 为什么不是权重裁剪？
14. thinking budget 改变了什么部署权衡？
15. 为什么整代模型比较不能替代组件消融？
16. 为什么 benchmark 分数必须携带 prompt 和采样协议？
17. QK-Clip 与 QK-Norm 有何差别？
18. QSA 与 DSA 的选择粒度有何差别？
19. Gated Residual 与增加 hidden size 有何差别？
20. n-gram host table 为什么不属于零成本参数？
21. LatentMoE 与 MLA 分别压缩什么？
22. MiniMax-M1 与 M2 的 attention 选择为什么相反？
23. MOPD 与离线轨迹蒸馏有何差别？
24. 异步 RL 为什么会引入 policy staleness？
25. Visual Resolution Router 与 Dynamic High Resolution 有何差别？
26. ARIA 解决什么 token-rate 对齐问题？
27. Janus-Pro 为什么使用两个视觉 tokenizer？
28. Open-weight 与 fully open 有何差别？
29. Technical report、model card 和 system card 分别能支持什么结论？
30. GPT-5 这类路由系统为什么不能当作单一 checkpoint 分析？

---

## Sources

[cite:1] Qwen Team, “Qwen Technical Report.” [arXiv:2309.16609](https://arxiv.org/abs/2309.16609)

[cite:2] Qwen Team, “Qwen2 Technical Report.” [arXiv:2407.10671](https://arxiv.org/abs/2407.10671)

[cite:3] Qwen Team, “Qwen2.5 Technical Report.” [arXiv:2412.15115](https://arxiv.org/abs/2412.15115)

[cite:4] Qwen Team, “Qwen3 Technical Report.” [arXiv:2505.09388](https://arxiv.org/abs/2505.09388)

[cite:5] Qwen Team, “Qwen3.5: Towards Native Multimodal Agents,” official release note, 2026-02-16. [Official post](https://qwen.ai/blog?id=qwen3.5)

[cite:6] DeepSeek-AI, “DeepSeek LLM: Scaling Open-Source Language Models with Longtermism.” [arXiv:2401.02954](https://arxiv.org/abs/2401.02954)

[cite:7] DeepSeek-AI, “DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model.” [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)

[cite:8] DeepSeek-AI, “DeepSeek-V3 Technical Report.” [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)

[cite:9] DeepSeek-AI, “DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.” [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)

[cite:10] DeepSeek-AI, “DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models.” [arXiv:2512.02556](https://arxiv.org/abs/2512.02556)

[cite:11] Meta, “The Llama 3 Herd of Models.” [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)

[cite:12] Mistral AI, “Mistral 7B.” [arXiv:2310.06825](https://arxiv.org/abs/2310.06825)

[cite:13] Mistral AI, “Mixtral of Experts.” [arXiv:2401.04088](https://arxiv.org/abs/2401.04088)

[cite:14] Google DeepMind, “Gemma 2: Improving Open Language Models at a Practical Size.” [arXiv:2408.00118](https://arxiv.org/abs/2408.00118)

[cite:15] Qwen Team, “Qwen3-Omni Technical Report.” [arXiv:2509.17765](https://arxiv.org/abs/2509.17765)

[cite:16] Kimi Team, “Kimi K2: Open Agentic Intelligence.” [arXiv:2507.20534](https://arxiv.org/abs/2507.20534)

[cite:17] GLM-4.5 Team, “GLM-4.5: Agentic, Reasoning, and Coding Foundation Models.” [arXiv:2508.06471](https://arxiv.org/abs/2508.06471)

[cite:18] MiniMax, “MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention.” [arXiv:2506.13585](https://arxiv.org/abs/2506.13585)

[cite:19] MiniMax, “The MiniMax-M2 Series: Mini Activations Unleashing Max Real-World Intelligence.” [arXiv:2605.26494](https://arxiv.org/abs/2605.26494)

[cite:20] OpenAI, “gpt-oss-120b & gpt-oss-20b Model Card.” [arXiv:2508.10925](https://arxiv.org/abs/2508.10925)

[cite:21] Olmo Team, “Olmo 3.” [arXiv:2512.13961](https://arxiv.org/abs/2512.13961)

[cite:22] NVIDIA, “NVIDIA Nemotron 3: Efficient and Open Intelligence.” [arXiv:2512.20856](https://arxiv.org/abs/2512.20856)

[cite:23] Qwen Team, “On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability.” [arXiv:2608.30320](https://arxiv.org/abs/2608.30320)

[cite:24] GLM-5 Team, “GLM-5: from Vibe Coding to Agentic Engineering.” [arXiv:2602.15763](https://arxiv.org/abs/2602.15763)

[cite:25] Xiaomi LLM-Core, “MiMo-V2-Flash Technical Report.” [arXiv:2601.02780](https://arxiv.org/abs/2601.02780)

[cite:26] Qwen Team, “Qwen3-VL Technical Report.” [arXiv:2511.21631](https://arxiv.org/abs/2511.21631)

[cite:27] Qwen Team, “Qwen3.5-Omni Technical Report.” [arXiv:2604.15804](https://arxiv.org/abs/2604.15804)

[cite:28] Qwen Team, “Qwen3.8-Omni: Towards Native Omni-Modal Agents.” [arXiv:2609.25611](https://arxiv.org/abs/2609.25611)

[cite:29] Google DeepMind, “Gemma 3 Technical Report.” [arXiv:2503.19786](https://arxiv.org/abs/2503.19786)

[cite:30] Google DeepMind, “Gemma 4 Technical Report.” [arXiv:2607.02770](https://arxiv.org/abs/2607.02770)

[cite:31] InternVL Team, “InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency.” [arXiv:2508.18265](https://arxiv.org/abs/2508.18265)

[cite:32] ByteDance Seed, “Seed1.5-VL Technical Report.” [arXiv:2505.07062](https://arxiv.org/abs/2505.07062)

[cite:33] DeepSeek-AI, “Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling.” [arXiv:2501.17811](https://arxiv.org/abs/2501.17811)

[cite:34] Qwen Team, “Qwen-Image-2.0 Technical Report.” [arXiv:2605.10730](https://arxiv.org/abs/2605.10730)

[cite:35] Kimi Team, “Kimi K2.5: Visual Agentic Intelligence.” [arXiv:2602.02276](https://arxiv.org/abs/2602.02276)

[cite:36] Microsoft Research, “Phi-4 Technical Report.” [arXiv:2412.08905](https://arxiv.org/abs/2412.08905)

[cite:37] Microsoft Research, “Phi-4-reasoning Technical Report.” [arXiv:2504.21318](https://arxiv.org/abs/2504.21318)

[cite:38] OpenAI, “GPT-5 System Card.” [Official publication](https://openai.com/index/gpt-5-system-card/)

[cite:39] Anthropic, “Transparency Hub.” [Official model reports](https://www.anthropic.com/transparency)

[cite:40] Google DeepMind, “Gemini 3.7 Flash Model Card.” [Official model card](https://deepmind.google/models/model-cards/gemini-3-7-flash/)

[cite:41] Meta, “Llama 4 Model Card and Prompt Formats.” [Official documentation](https://www.llama.com/docs/model-cards-and-prompt-formats/llama4/)

[cite:42] Falcon Team, “Falcon-H1: A Family of Hybrid-Head Language Models Redefining Efficiency and Performance.” [arXiv:2507.22448](https://arxiv.org/abs/2507.22448)

[cite:43] Tencent Hunyuan, “Hunyuan-A13B Technical Report.” [arXiv:2609.27284](https://arxiv.org/abs/2609.27284)

[cite:44] Tencent Hunyuan, “HunyuanImage 3.0 Technical Report.” [arXiv:2509.23951](https://arxiv.org/abs/2509.23951)

[cite:45] LLM360 Team, “K2-V2: A 360-Open, Reasoning-Enhanced LLM.” [arXiv:2512.06201](https://arxiv.org/abs/2512.06201)

[cite:46] Apertus Team, “Apertus: Democratizing Open and Compliant LLMs for Global Language Environments.” [arXiv:2509.14233](https://arxiv.org/abs/2509.14233)

[cite:47] EuroLLM Team, “EuroLLM-22B: Technical Report.” [arXiv:2602.05879](https://arxiv.org/abs/2602.05879)

[cite:48] Mistral AI, “Magistral.” [arXiv:2506.10910](https://arxiv.org/abs/2506.10910)

[cite:49] Mistral AI, “Ministral 3.” [arXiv:2601.08584](https://arxiv.org/abs/2601.08584)

[cite:50] Qwen Team, “Qwen3-TTS Technical Report.” [arXiv:2601.15621](https://arxiv.org/abs/2601.15621)

[cite:51] Qwen Team, “Qwen3-ASR Technical Report.” [arXiv:2601.21337](https://arxiv.org/abs/2601.21337)
