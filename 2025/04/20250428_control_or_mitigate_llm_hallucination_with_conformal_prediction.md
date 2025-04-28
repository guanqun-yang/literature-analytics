**[1. [2501.17295] Mitigating Hallucinated Translations in Large Language Models with
  Hallucination-focused Preference Optimization](https://arxiv.org/pdf/2501.17295.pdf)** (2025-01-30)

*Zilu Tang, Rajen Chatterjee, Sarthak Garg*

  Machine Translation (MT) is undergoing a paradigm shift, with systems based
on fine-tuned large language models (LLM) becoming increasingly competitive
with traditional encoder-decoder models trained specifically for translation
tasks. However, LLM-based systems are at a higher risk of generating
hallucinations, which can severely undermine user's trust and safety. Most
prior research on hallucination mitigation focuses on traditional MT models,
with solutions that involve post-hoc mitigation - detecting hallucinated
translations and re-translating them. While effective, this approach introduces
additional complexity in deploying extra tools in production and also increases
latency. To address these limitations, we propose a method that intrinsically
learns to mitigate hallucinations during the model training phase.
Specifically, we introduce a data creation framework to generate hallucination
focused preference datasets. Fine-tuning LLMs on these preference datasets
reduces the hallucination rate by an average of 96% across five language pairs,
while preserving overall translation quality. In a zero-shot setting our
approach reduces hallucinations by 89% on an average across three unseen target
languages.


---

**[2. [2411.09689] LLM Hallucination Reasoning with Zero-shot Knowledge Test](https://arxiv.org/pdf/2411.09689.pdf)** (2024-11-15)

*Seongmin Lee, Hsiang Hsu, Chun-Fu Chen*

  LLM hallucination, where LLMs occasionally generate unfaithful text, poses
significant challenges for their practical applications. Most existing
detection methods rely on external knowledge, LLM fine-tuning, or
hallucination-labeled datasets, and they do not distinguish between different
types of hallucinations, which are crucial for improving detection performance.
We introduce a new task, Hallucination Reasoning, which classifies
LLM-generated text into one of three categories: aligned, misaligned, and
fabricated. Our novel zero-shot method assesses whether LLM has enough
knowledge about a given prompt and text. Our experiments conducted on new
datasets demonstrate the effectiveness of our method in hallucination reasoning
and underscore its importance for enhancing detection performance.


---

**[3. [2502.13416] Detecting LLM Fact-conflicting Hallucinations Enhanced by
  Temporal-logic-based Reasoning](https://arxiv.org/pdf/2502.13416.pdf)** (2025-02-20)

*Ningke Li, Yahui Song, Kailong Wang, Yuekang Li, Ling Shi, Yi Liu, Haoyu Wang*

  Large language models (LLMs) face the challenge of hallucinations -- outputs
that seem coherent but are actually incorrect. A particularly damaging type is
fact-conflicting hallucination (FCH), where generated content contradicts
established facts. Addressing FCH presents three main challenges: 1)
Automatically constructing and maintaining large-scale benchmark datasets is
difficult and resource-intensive; 2) Generating complex and efficient test
cases that the LLM has not been trained on -- especially those involving
intricate temporal features -- is challenging, yet crucial for eliciting
hallucinations; and 3) Validating the reasoning behind LLM outputs is
inherently difficult, particularly with complex logical relationships, as it
requires transparency in the model's decision-making process.
  This paper presents Drowzee, an innovative end-to-end metamorphic testing
framework that utilizes temporal logic to identify fact-conflicting
hallucinations (FCH) in large language models (LLMs). Drowzee builds a
comprehensive factual knowledge base by crawling sources like Wikipedia and
uses automated temporal-logic reasoning to convert this knowledge into a large,
extensible set of test cases with ground truth answers. LLMs are tested using
these cases through template-based prompts, which require them to generate both
answers and reasoning steps. To validate the reasoning, we propose two
semantic-aware oracles that compare the semantic structure of LLM outputs to
the ground truths. Across nine LLMs in nine different knowledge domains,
experimental results show that Drowzee effectively identifies rates of
non-temporal-related hallucinations ranging from 24.7% to 59.8%, and rates of
temporal-related hallucinations ranging from 16.7% to 39.2%.


---

**[4. [2411.04847] Prompt-Guided Internal States for Hallucination Detection of Large
  Language Models](https://arxiv.org/pdf/2411.04847.pdf)** (2025-03-03)

*Fujie Zhang, Peiqi Yu, Biao Yi, Baolei Zhang, Tong Li, Zheli Liu*

  Large Language Models (LLMs) have demonstrated remarkable capabilities across
a variety of tasks in different domains. However, they sometimes generate
responses that are logically coherent but factually incorrect or misleading,
which is known as LLM hallucinations. Data-driven supervised methods train
hallucination detectors by leveraging the internal states of LLMs, but
detectors trained on specific domains often struggle to generalize well to
other domains. In this paper, we aim to enhance the cross-domain performance of
supervised detectors with only in-domain data. We propose a novel framework,
prompt-guided internal states for hallucination detection of LLMs, namely
PRISM. By utilizing appropriate prompts to guide changes to the structure
related to text truthfulness in LLMs' internal states, we make this structure
more salient and consistent across texts from different domains. We integrated
our framework with existing hallucination detection methods and conducted
experiments on datasets from different domains. The experimental results
indicate that our framework significantly enhances the cross-domain
generalization of existing hallucination detection methods.


---

**[5. [2503.21813] OAEI-LLM-T: A TBox Benchmark Dataset for Understanding LLM
  Hallucinations in Ontology Matching Systems](https://arxiv.org/pdf/2503.21813.pdf)** (2025-03-31)

*Zhangcheng Qiang*

  Hallucinations are inevitable in downstream tasks using large language models
(LLMs). While addressing hallucinations becomes a substantial challenge for
LLM-based ontology matching (OM) systems, we introduce a new benchmark dataset
called OAEI-LLM-T. The dataset evolves from the TBox (i.e. schema-matching)
datasets in the Ontology Alignment Evaluation Initiative (OAEI), capturing
hallucinations of different LLMs performing OM tasks. These OM-specific
hallucinations are carefully classified into two primary categories and six
sub-categories. We showcase the usefulness of the dataset in constructing the
LLM leaderboard and fine-tuning foundational LLMs for LLM-based OM systems.


---

**[6. [2407.03282] LLM Internal States Reveal Hallucination Risk Faced With a Query](https://arxiv.org/pdf/2407.03282.pdf)** (2024-10-01)

*Ziwei Ji, Delong Chen, Etsuko Ishii, Samuel Cahyawijaya, Yejin Bang, Bryan Wilie, Pascale Fung*

  The hallucination problem of Large Language Models (LLMs) significantly
limits their reliability and trustworthiness. Humans have a self-awareness
process that allows us to recognize what we don't know when faced with queries.
Inspired by this, our paper investigates whether LLMs can estimate their own
hallucination risk before response generation. We analyze the internal
mechanisms of LLMs broadly both in terms of training data sources and across 15
diverse Natural Language Generation (NLG) tasks, spanning over 700 datasets.
Our empirical analysis reveals two key insights: (1) LLM internal states
indicate whether they have seen the query in training data or not; and (2) LLM
internal states show they are likely to hallucinate or not regarding the query.
Our study explores particular neurons, activation layers, and tokens that play
a crucial role in the LLM perception of uncertainty and hallucination risk. By
a probing estimator, we leverage LLM self-assessment, achieving an average
hallucination estimation accuracy of 84.32\% at run time.


---

**[7. [2410.11414] ReDeEP: Detecting Hallucination in Retrieval-Augmented Generation via
  Mechanistic Interpretability](https://arxiv.org/pdf/2410.11414.pdf)** (2025-01-22)

*Zhongxiang Sun, Xiaoxue Zang, Kai Zheng, Yang Song, Jun Xu, Xiao Zhang, Weijie Yu, Yang Song, Han Li*

  Retrieval-Augmented Generation (RAG) models are designed to incorporate
external knowledge, reducing hallucinations caused by insufficient parametric
(internal) knowledge. However, even with accurate and relevant retrieved
content, RAG models can still produce hallucinations by generating outputs that
conflict with the retrieved information. Detecting such hallucinations requires
disentangling how Large Language Models (LLMs) utilize external and parametric
knowledge. Current detection methods often focus on one of these mechanisms or
without decoupling their intertwined effects, making accurate detection
difficult. In this paper, we investigate the internal mechanisms behind
hallucinations in RAG scenarios. We discover hallucinations occur when the
Knowledge FFNs in LLMs overemphasize parametric knowledge in the residual
stream, while Copying Heads fail to effectively retain or integrate external
knowledge from retrieved content. Based on these findings, we propose ReDeEP, a
novel method that detects hallucinations by decoupling LLM's utilization of
external context and parametric knowledge. Our experiments show that ReDeEP
significantly improves RAG hallucination detection accuracy. Additionally, we
introduce AARF, which mitigates hallucinations by modulating the contributions
of Knowledge FFNs and Copying Heads.


---

**[8. [2409.14038] OAEI-LLM: A Benchmark Dataset for Understanding Large Language Model
  Hallucinations in Ontology Matching](https://arxiv.org/pdf/2409.14038.pdf)** (2025-02-04)

*Zhangcheng Qiang, Kerry Taylor, Weiqing Wang, Jing Jiang*

  Hallucinations of large language models (LLMs) commonly occur in
domain-specific downstream tasks, with no exception in ontology matching (OM).
The prevalence of using LLMs for OM raises the need for benchmarks to better
understand LLM hallucinations. The OAEI-LLM dataset is an extended version of
the Ontology Alignment Evaluation Initiative (OAEI) datasets that evaluate
LLM-specific hallucinations in OM tasks. We outline the methodology used in
dataset construction and schema extension, and provide examples of potential
use cases.


---

**[9. [2402.09733] Do LLMs Know about Hallucination? An Empirical Investigation of LLM's
  Hidden States](https://arxiv.org/pdf/2402.09733.pdf)** (2024-02-16)

*Hanyu Duan, Yi Yang, Kar Yan Tam*

  Large Language Models (LLMs) can make up answers that are not real, and this
is known as hallucination. This research aims to see if, how, and to what
extent LLMs are aware of hallucination. More specifically, we check whether and
how an LLM reacts differently in its hidden states when it answers a question
right versus when it hallucinates. To do this, we introduce an experimental
framework which allows examining LLM's hidden states in different hallucination
situations. Building upon this framework, we conduct a series of experiments
with language models in the LLaMA family (Touvron et al., 2023). Our empirical
findings suggest that LLMs react differently when processing a genuine response
versus a fabricated one. We then apply various model interpretation techniques
to help understand and explain the findings better. Moreover, informed by the
empirical observations, we show great potential of using the guidance derived
from LLM's hidden representation space to mitigate hallucination. We believe
this work provides insights into how LLMs produce hallucinated answers and how
to make them occur less often.


---

**[10. [2501.12975] OnionEval: An Unified Evaluation of Fact-conflicting Hallucination for
  Small-Large Language Models](https://arxiv.org/pdf/2501.12975.pdf)** (2025-01-23)

*Chongren Sun, Yuran Li, Di Wu, Benoit Boulet*

  Large Language Models (LLMs) are highly capable but require significant
computational resources for both training and inference. Within the LLM family,
smaller models (those with fewer than 10 billion parameters) also perform well
across various tasks. However, these smaller models share similar limitations
to their larger counterparts, including the tendency to hallucinate. Despite
the existence of many benchmarks to evaluate hallucination in LLMs, few have
specifically focused on small LLMs (SLLMs). Additionally, SLLMs show widely
varying performance across different benchmarks. In this paper, we introduce
OnionEval, a multi-layer structured framework with a specific metric called the
context-influence score (CI), designed to effectively assess the
fact-conflicting hallucination tendencies of small LLMs across different
contextual levels. Our experimental results reveal a key feature of SLLMs: they
excel in factual analysis but face challenges with context reasoning. Further
investigation shows that a simple Chain-of-Thought strategy can significantly
reduce these limitations, improving the practical usefulness of SLLMs in
real-world applications.


---

**[11. [2407.09417] Mitigating Entity-Level Hallucination in Large Language Models](https://arxiv.org/pdf/2407.09417.pdf)** (2024-07-23)

*Weihang Su, Yichen Tang, Qingyao Ai, Changyue Wang, Zhijing Wu, Yiqun Liu*

  The emergence of Large Language Models (LLMs) has revolutionized how users
access information, shifting from traditional search engines to direct
question-and-answer interactions with LLMs. However, the widespread adoption of
LLMs has revealed a significant challenge known as hallucination, wherein LLMs
generate coherent yet factually inaccurate responses. This hallucination
phenomenon has led to users' distrust in information retrieval systems based on
LLMs. To tackle this challenge, this paper proposes Dynamic Retrieval
Augmentation based on hallucination Detection (DRAD) as a novel method to
detect and mitigate hallucinations in LLMs. DRAD improves upon traditional
retrieval augmentation by dynamically adapting the retrieval process based on
real-time hallucination detection. It features two main components: Real-time
Hallucination Detection (RHD) for identifying potential hallucinations without
external models, and Self-correction based on External Knowledge (SEK) for
correcting these errors using external knowledge. Experiment results show that
DRAD demonstrates superior performance in both detecting and mitigating
hallucinations in LLMs. All of our code and data are open-sourced at
https://github.com/oneal2000/EntityHallucination.


---

**[12. [2310.03951] Chain of Natural Language Inference for Reducing Large Language Model
  Ungrounded Hallucinations](https://arxiv.org/pdf/2310.03951.pdf)** (2023-10-11)

*Deren Lei, Yaxi Li, Mengya Hu, Mingyu Wang, Vincent Yun, Emily Ching, Eslam Kamal*

  Large language models (LLMs) can generate fluent natural language texts when
given relevant documents as background context. This ability has attracted
considerable interest in developing industry applications of LLMs. However,
LLMs are prone to generate hallucinations that are not supported by the
provided sources. In this paper, we propose a hierarchical framework to detect
and mitigate such ungrounded hallucination. Our framework uses Chain of Natural
Language Inference (CoNLI) for hallucination detection and hallucination
reduction via post-editing. Our approach achieves state-of-the-art performance
on hallucination detection and enhances text quality through rewrite, using
LLMs without any fine-tuning or domain-specific prompt engineering. We show
that this simple plug-and-play framework can serve as an effective choice for
hallucination detection and reduction, achieving competitive performance across
various contexts.


---

**[13. [2503.04615] HalluCounter: Reference-free LLM Hallucination Detection in the Wild!](https://arxiv.org/pdf/2503.04615.pdf)** (2025-03-07)

*Ashok Urlana, Gopichand Kanumolu, Charaka Vinayak Kumar, Bala Mallikarjunarao Garlapati, Rahul Mishra*

  Response consistency-based, reference-free hallucination detection (RFHD)
methods do not depend on internal model states, such as generation
probabilities or gradients, which Grey-box models typically rely on but are
inaccessible in closed-source LLMs. However, their inability to capture
query-response alignment patterns often results in lower detection accuracy.
Additionally, the lack of large-scale benchmark datasets spanning diverse
domains remains a challenge, as most existing datasets are limited in size and
scope. To this end, we propose HalluCounter, a novel reference-free
hallucination detection method that utilizes both response-response and
query-response consistency and alignment patterns. This enables the training of
a classifier that detects hallucinations and provides a confidence score and an
optimal response for user queries. Furthermore, we introduce HalluCounterEval,
a benchmark dataset comprising both synthetically generated and human-curated
samples across multiple domains. Our method outperforms state-of-the-art
approaches by a significant margin, achieving over 90\% average confidence in
hallucination detection across datasets.


---

**[14. [2503.17229] FactSelfCheck: Fact-Level Black-Box Hallucination Detection for LLMs](https://arxiv.org/pdf/2503.17229.pdf)** (2025-03-24)

*Albert Sawczyn, Jakub Binkowski, Denis Janiak, Bogdan Gabrys, Tomasz Kajdanowicz*

  Large Language Models (LLMs) frequently generate hallucinated content, posing
significant challenges for applications where factuality is crucial. While
existing hallucination detection methods typically operate at the sentence
level or passage level, we propose FactSelfCheck, a novel black-box
sampling-based method that enables fine-grained fact-level detection. Our
approach represents text as knowledge graphs consisting of facts in the form of
triples. Through analyzing factual consistency across multiple LLM responses,
we compute fine-grained hallucination scores without requiring external
resources or training data. Our evaluation demonstrates that FactSelfCheck
performs competitively with leading sampling-based methods while providing more
detailed insights. Most notably, our fact-level approach significantly improves
hallucination correction, achieving a 35% increase in factual content compared
to the baseline, while sentence-level SelfCheckGPT yields only an 8%
improvement. The granular nature of our detection enables more precise
identification and correction of hallucinated content.


---

**[15. [2412.06007] Hallucination-aware Optimization for Large Language Model-empowered
  Communications](https://arxiv.org/pdf/2412.06007.pdf)** (2024-12-10)

*Yinqiu Liu, Guangyuan Liu, Ruichen Zhang, Dusit Niyato, Zehui Xiong, Dong In Kim, Kaibin Huang, Hongyang Du*

  Large Language Models (LLMs) have significantly advanced communications
fields, such as Telecom Q\&A, mathematical modeling, and coding. However, LLMs
encounter an inherent issue known as hallucination, i.e., generating
fact-conflicting or irrelevant content. This problem critically undermines the
applicability of LLMs in communication systems yet has not been systematically
explored. Hence, this paper provides a comprehensive review of LLM applications
in communications, with a particular emphasis on hallucination mitigation.
Specifically, we analyze hallucination causes and summarize hallucination
mitigation strategies from both model- and system-based perspectives.
Afterward, we review representative LLM-empowered communication schemes,
detailing potential hallucination scenarios and comparing the mitigation
strategies they adopted. Finally, we present a case study of a Telecom-oriented
LLM that utilizes a novel hybrid approach to enhance the hallucination-aware
service experience. On the model side, we publish a Telecom hallucination
dataset and apply direct preference optimization to fine-tune LLMs, resulting
in a 20.6\% correct rate improvement. Moreover, we construct a mobile-edge
mixture-of-experts architecture for optimal LLM expert activation. Our research
aims to propel the field of LLM-empowered communications forward by detecting
and minimizing hallucination impacts.


---

**[16. [2407.20999] MoFO: Momentum-Filtered Optimizer for Mitigating Forgetting in LLM
  Fine-Tuning](https://arxiv.org/pdf/2407.20999.pdf)** (2025-04-21)

*Yupeng Chen, Senmiao Wang, Yushun Zhang, Zhihang Lin, Haozhe Zhang, Weijian Sun, Tian Ding, Ruoyu Sun*

  Large language models (LLMs) have demonstrated remarkable capabilities across
a wide range of tasks. Typically, LLMs are first pre-trained on large corpora
and subsequently fine-tuned on task-specific datasets. However, during
fine-tuning, LLMs may forget some knowledge acquired in the pre-training stage,
leading to a decline in general capabilities. Existing approaches to mitigate
forgetting often rely on access to pre-training data, which may be unavailable
in many real-world scenarios--such as fine-tuning checkpoint-only open-source
LLMs. To address this challenge, we propose a new fine-tuning algorithm termed
Momentum-Filtered Optimizer (MoFO). MoFO is an extension of greedy block
coordinate descent (BCD) methods: in each iteration, MoFO only updates the
model parameters with the largest momentum magnitudes, while keeping all other
parameters fixed. MoFO achieves similar fine-tuning performance to the default
fine-tuning algorithm while effectively mitigating knowledge forgetting. We
validate MoFO through rigorous convergence analysis and extensive experiments,
demonstrating its effectiveness in mitigating forgetting without pre-training
data.


---

**[17. [2410.19385] Investigating the Role of Prompting and External Tools in Hallucination
  Rates of Large Language Models](https://arxiv.org/pdf/2410.19385.pdf)** (2024-10-28)

*Liam Barkley, Brink van der Merwe*

  Large Language Models (LLMs) are powerful computational models trained on
extensive corpora of human-readable text, enabling them to perform
general-purpose language understanding and generation. LLMs have garnered
significant attention in both industry and academia due to their exceptional
performance across various natural language processing (NLP) tasks. Despite
these successes, LLMs often produce inaccuracies, commonly referred to as
hallucinations. Prompt engineering, the process of designing and formulating
instructions for LLMs to perform specific tasks, has emerged as a key approach
to mitigating hallucinations. This paper provides a comprehensive empirical
evaluation of different prompting strategies and frameworks aimed at reducing
hallucinations in LLMs. Various prompting techniques are applied to a broad set
of benchmark datasets to assess the accuracy and hallucination rate of each
method. Additionally, the paper investigates the influence of tool-calling
agents (LLMs augmented with external tools to enhance their capabilities beyond
language generation) on hallucination rates in the same benchmarks. The
findings demonstrate that the optimal prompting technique depends on the type
of problem, and that simpler techniques often outperform more complex methods
in reducing hallucinations. Furthermore, it is shown that LLM agents can
exhibit significantly higher hallucination rates due to the added complexity of
external tool usage.


---

**[18. [2407.04121] Hallucination Detection: Robustly Discerning Reliable Answers in Large
  Language Models](https://arxiv.org/pdf/2407.04121.pdf)** (2024-07-08)

*Yuyan Chen, Qiang Fu, Yichen Yuan, Zhihao Wen, Ge Fan, Dayiheng Liu, Dongmei Zhang, Zhixu Li, Yanghua Xiao*

  Large Language Models (LLMs) have gained widespread adoption in various
natural language processing tasks, including question answering and dialogue
systems. However, a major drawback of LLMs is the issue of hallucination, where
they generate unfaithful or inconsistent content that deviates from the input
source, leading to severe consequences. In this paper, we propose a robust
discriminator named RelD to effectively detect hallucination in LLMs' generated
answers. RelD is trained on the constructed RelQA, a bilingual
question-answering dialogue dataset along with answers generated by LLMs and a
comprehensive set of metrics. Our experimental results demonstrate that the
proposed RelD successfully detects hallucination in the answers generated by
diverse LLMs. Moreover, it performs well in distinguishing hallucination in
LLMs' generated answers from both in-distribution and out-of-distribution
datasets. Additionally, we also conduct a thorough analysis of the types of
hallucinations that occur and present valuable insights. This research
significantly contributes to the detection of reliable answers generated by
LLMs and holds noteworthy implications for mitigating hallucination in the
future work.


---

**[19. [2504.07069] HalluciNot: Hallucination Detection Through Context and Common Knowledge
  Verification](https://arxiv.org/pdf/2504.07069.pdf)** (2025-04-10)

*Bibek Paudel, Alexander Lyzhov, Preetam Joshi, Puneet Anand*

  This paper introduces a comprehensive system for detecting hallucinations in
large language model (LLM) outputs in enterprise settings. We present a novel
taxonomy of LLM responses specific to hallucination in enterprise applications,
categorizing them into context-based, common knowledge, enterprise-specific,
and innocuous statements. Our hallucination detection model HDM-2 validates LLM
responses with respect to both context and generally known facts (common
knowledge). It provides both hallucination scores and word-level annotations,
enabling precise identification of problematic content. To evaluate it on
context-based and common-knowledge hallucinations, we introduce a new dataset
HDMBench. Experimental results demonstrate that HDM-2 out-performs existing
approaches across RagTruth, TruthfulQA, and HDMBench datasets. This work
addresses the specific challenges of enterprise deployment, including
computational efficiency, domain specialization, and fine-grained error
identification. Our evaluation dataset, model weights, and inference code are
publicly available.


---

**[20. [2407.10153] Look Within, Why LLMs Hallucinate: A Causal Perspective](https://arxiv.org/pdf/2407.10153.pdf)** (2024-07-16)

*He Li, Haoang Chi, Mingyu Liu, Wenjing Yang*

  The emergence of large language models (LLMs) is a milestone in generative
artificial intelligence, achieving significant success in text comprehension
and generation tasks. Despite the tremendous success of LLMs in many downstream
tasks, they suffer from severe hallucination problems, posing significant
challenges to the practical applications of LLMs. Most of the works about LLMs'
hallucinations focus on data quality. Self-attention is a core module in
transformer-based LLMs, while its potential relationship with LLMs'
hallucination has been hardly investigated. To fill this gap, we study this
problem from a causal perspective. We propose a method to intervene in LLMs'
self-attention layers and maintain their structures and sizes intact.
Specifically, we disable different self-attention layers in several popular
open-source LLMs and then compare their degrees of hallucination with the
original ones. We evaluate the intervened LLMs on hallucination assessment
benchmarks and conclude that disabling some specific self-attention layers in
the front or tail of the LLMs can alleviate hallucination issues. The study
paves a new way for understanding and mitigating LLMs' hallucinations.


---

**[21. [2502.12964] Trust Me, I'm Wrong: High-Certainty Hallucinations in LLMs](https://arxiv.org/pdf/2502.12964.pdf)** (2025-02-19)

*Adi Simhi, Itay Itzhak, Fazl Barez, Gabriel Stanovsky, Yonatan Belinkov*

  Large Language Models (LLMs) often generate outputs that lack grounding in
real-world facts, a phenomenon known as hallucinations. Prior research has
associated hallucinations with model uncertainty, leveraging this relationship
for hallucination detection and mitigation. In this paper, we challenge the
underlying assumption that all hallucinations are associated with uncertainty.
Using knowledge detection and uncertainty measurement methods, we demonstrate
that models can hallucinate with high certainty even when they have the correct
knowledge. We further show that high-certainty hallucinations are consistent
across models and datasets, distinctive enough to be singled out, and challenge
existing mitigation methods. Our findings reveal an overlooked aspect of
hallucinations, emphasizing the need to understand their origins and improve
mitigation strategies to enhance LLM safety. The code is available at
https://github.com/technion-cs-nlp/Trust_me_Im_wrong .


---

**[22. [2502.11306] Smoothing Out Hallucinations: Mitigating LLM Hallucination with Smoothed
  Knowledge Distillation](https://arxiv.org/pdf/2502.11306.pdf)** (2025-02-18)

*Hieu Nguyen, Zihao He, Shoumik Atul Gandre, Ujjwal Pasupulety, Sharanya Kumari Shivakumar, Kristina Lerman*

  Large language models (LLMs) often suffer from hallucination, generating
factually incorrect or ungrounded content, which limits their reliability in
high-stakes applications. A key factor contributing to hallucination is the use
of hard labels during training, which enforce deterministic supervision,
encourage overconfidence, and disregard the uncertainty inherent in natural
language. To address this, we propose mitigating hallucination through
knowledge distillation (KD), where a teacher model provides smoothed soft
labels to a student model, reducing overconfidence and improving factual
grounding. We apply KD during supervised finetuning on instructional data,
evaluating its effectiveness across LLMs from different families. Experimental
results on summarization benchmarks demonstrate that KD reduces hallucination
compared to standard finetuning while preserving performance on general NLP
tasks. These findings highlight KD as a promising approach for mitigating
hallucination in LLMs and improving model reliability.


---

**[23. [2412.13817] Nullu: Mitigating Object Hallucinations in Large Vision-Language Models
  via HalluSpace Projection](https://arxiv.org/pdf/2412.13817.pdf)** (2025-03-18)

*Le Yang, Ziwei Zheng, Boxu Chen, Zhengyu Zhao, Chenhao Lin, Chao Shen*

  Recent studies have shown that large vision-language models (LVLMs) often
suffer from the issue of object hallucinations (OH). To mitigate this issue, we
introduce an efficient method that edits the model weights based on an unsafe
subspace, which we call HalluSpace in this paper. With truthful and
hallucinated text prompts accompanying the visual content as inputs, the
HalluSpace can be identified by extracting the hallucinated embedding features
and removing the truthful representations in LVLMs. By orthogonalizing the
model weights, input features will be projected into the Null space of the
HalluSpace to reduce OH, based on which we name our method Nullu. We reveal
that HalluSpaces generally contain prior information in the large language
models (LLMs) applied to build LVLMs, which have been shown as essential causes
of OH in previous studies. Therefore, null space projection suppresses the
LLMs' priors to filter out the hallucinated features, resulting in contextually
accurate outputs. Experiments show that our method can effectively mitigate OH
across different LVLM families without extra inference costs and also show
strong performance in general LVLM benchmarks. Code is released at
https://github.com/Ziwei-Zheng/Nullu.


---

**[24. [2407.15441] Developing a Reliable, Fast, General-Purpose Hallucination Detection and
  Mitigation Service](https://arxiv.org/pdf/2407.15441.pdf)** (2025-04-01)

*Song Wang, Xun Wang, Jie Mei, Yujia Xie, Sean Muarray, Zhang Li, Lingfeng Wu, Si-Qing Chen, Wayne Xiong*

  Hallucination, a phenomenon where large language models (LLMs) produce output
that is factually incorrect or unrelated to the input, is a major challenge for
LLM applications that require accuracy and dependability. In this paper, we
introduce a reliable and high-speed production system aimed at detecting and
rectifying the hallucination issue within LLMs. Our system encompasses named
entity recognition (NER), natural language inference (NLI), span-based
detection (SBD), and an intricate decision tree-based process to reliably
detect a wide range of hallucinations in LLM responses. Furthermore, we have
crafted a rewriting mechanism that maintains an optimal mix of precision,
response time, and cost-effectiveness. We detail the core elements of our
framework and underscore the paramount challenges tied to response time,
availability, and performance metrics, which are crucial for real-world
deployment of these technologies. Our extensive evaluation, utilizing offline
data and live production traffic, confirms the efficacy of our proposed
framework and service.


---

**[25. [2402.17811] TruthX: Alleviating Hallucinations by Editing Large Language Models in
  Truthful Space](https://arxiv.org/pdf/2402.17811.pdf)** (2024-06-06)

*Shaolei Zhang, Tian Yu, Yang Feng*

  Large Language Models (LLMs) sometimes suffer from producing hallucinations,
especially LLMs may generate untruthful responses despite knowing the correct
knowledge. Activating the truthfulness within LLM is the key to fully unlocking
LLM's knowledge potential. In this paper, we propose TruthX, an inference-time
intervention method to activate the truthfulness of LLM by identifying and
editing the features within LLM's internal representations that govern the
truthfulness. TruthX employs an auto-encoder to map LLM's representations into
semantic and truthful latent spaces respectively, and applies contrastive
learning to identify a truthful editing direction within the truthful space.
During inference, by editing LLM's internal representations in truthful space,
TruthX effectively enhances the truthfulness of LLM. Experiments show that
TruthX improves the truthfulness of 13 advanced LLMs by an average of 20% on
TruthfulQA benchmark. Further analyses suggest that TruthX can control LLM to
produce truthful or hallucinatory responses via editing only one vector in
LLM's internal representations.


---

**[26. [2504.12314] How to Detect and Defeat Molecular Mirage: A Metric-Driven Benchmark for
  Hallucination in LLM-based Molecular Comprehension](https://arxiv.org/pdf/2504.12314.pdf)** (2025-04-18)

*Hao Li, Liuzhenghao Lv, He Cao, Zijing Liu, Zhiyuan Yan, Yu Wang, Yonghong Tian, Yu Li, Li Yuan*

  Large language models are increasingly used in scientific domains, especially
for molecular understanding and analysis. However, existing models are affected
by hallucination issues, resulting in errors in drug design and utilization. In
this paper, we first analyze the sources of hallucination in LLMs for molecular
comprehension tasks, specifically the knowledge shortcut phenomenon observed in
the PubChem dataset. To evaluate hallucination in molecular comprehension tasks
with computational efficiency, we introduce \textbf{Mol-Hallu}, a novel
free-form evaluation metric that quantifies the degree of hallucination based
on the scientific entailment relationship between generated text and actual
molecular properties. Utilizing the Mol-Hallu metric, we reassess and analyze
the extent of hallucination in various LLMs performing molecular comprehension
tasks. Furthermore, the Hallucination Reduction Post-processing stage~(HRPP) is
proposed to alleviate molecular hallucinations, Experiments show the
effectiveness of HRPP on decoder-only and encoder-decoder molecular LLMs. Our
findings provide critical insights into mitigating hallucination and improving
the reliability of LLMs in scientific applications.


---

**[27. [2503.05757] Uncertainty-Aware Fusion: An Ensemble Framework for Mitigating
  Hallucinations in Large Language Models](https://arxiv.org/pdf/2503.05757.pdf)** (2025-03-11)

*Prasenjit Dey, Srujana Merugu, Sivaramakrishnan Kaveri*

  Large Language Models (LLMs) are known to hallucinate and generate
non-factual outputs which can undermine user trust. Traditional methods to
directly mitigate hallucinations, such as representation editing and
contrastive decoding, often require additional training data and involve high
implementation complexity. While ensemble-based approaches harness multiple
LLMs to tap into the "wisdom of crowds", these methods overlook uncertainties
in individual model responses. Recent studies reveal that uncertainty
estimation can enable LLMs to self-assess the likelihood of generating
hallucinations. In this work, we focus on factoid question answering (QA) and
observe that LLMs accuracy and self-assessment capabilities vary widely with
different models excelling in different scenarios. Leveraging this insight, we
propose Uncertainty-Aware Fusion (UAF), an ensemble framework to reduces
hallucinations by strategically combining multiple LLM based on their accuracy
and self-assessment abilities. Empirical results on several public benchmark
datasets show that UAF outperforms state-of-the-art hallucination mitigation
methods by $8\%$ in factual accuracy, while either narrowing or surpassing the
performance gap with GPT-4.


---

**[28. [2402.03744] INSIDE: LLMs' Internal States Retain the Power of Hallucination
  Detection](https://arxiv.org/pdf/2402.03744.pdf)** (2024-10-22)

*Chao Chen, Kai Liu, Ze Chen, Yi Gu, Yue Wu, Mingyuan Tao, Zhihang Fu, Jieping Ye*

  Knowledge hallucination have raised widespread concerns for the security and
reliability of deployed LLMs. Previous efforts in detecting hallucinations have
been employed at logit-level uncertainty estimation or language-level
self-consistency evaluation, where the semantic information is inevitably lost
during the token-decoding procedure. Thus, we propose to explore the dense
semantic information retained within LLMs' \textbf{IN}ternal \textbf{S}tates
for halluc\textbf{I}nation \textbf{DE}tection (\textbf{INSIDE}). In particular,
a simple yet effective \textbf{EigenScore} metric is proposed to better
evaluate responses' self-consistency, which exploits the eigenvalues of
responses' covariance matrix to measure the semantic consistency/diversity in
the dense embedding space. Furthermore, from the perspective of self-consistent
hallucination detection, a test time feature clipping approach is explored to
truncate extreme activations in the internal states, which reduces
overconfident generations and potentially benefits the detection of
overconfident hallucinations. Extensive experiments and ablation studies are
performed on several popular LLMs and question-answering (QA) benchmarks,
showing the effectiveness of our proposal.


---

**[29. [2412.10246] Detecting LLM Hallucination Through Layer-wise Information Deficiency:
  Analysis of Unanswerable Questions and Ambiguous Prompts](https://arxiv.org/pdf/2412.10246.pdf)** (2024-12-16)

*Hazel Kim, Adel Bibi, Philip Torr, Yarin Gal*

  Large language models (LLMs) frequently generate confident yet inaccurate
responses, introducing significant risks for deployment in safety-critical
domains. We present a novel approach to detecting model hallucination through
systematic analysis of information flow across model layers when processing
inputs with insufficient or ambiguous context. Our investigation reveals that
hallucination manifests as usable information deficiencies in inter-layer
transmissions. While existing approaches primarily focus on final-layer output
analysis, we demonstrate that tracking cross-layer information dynamics
($\mathcal{L}$I) provides robust indicators of model reliability, accounting
for both information gain and loss during computation. $\mathcal{L}$I improves
model reliability by immediately integrating with universal LLMs without
additional training or architectural modifications.


---

**[30. [2407.16908] Generation Constraint Scaling Can Mitigate Hallucination](https://arxiv.org/pdf/2407.16908.pdf)** (2024-07-25)

*Georgios Kollias, Payel Das, Subhajit Chaudhury*

  Addressing the issue of hallucinations in large language models (LLMs) is a
critical challenge. As the cognitive mechanisms of hallucination have been
related to memory, here we explore hallucination for LLM that is enabled with
explicit memory mechanisms. We empirically demonstrate that by simply scaling
the readout vector that constrains generation in a memory-augmented LLM
decoder, hallucination mitigation can be achieved in a training-free manner.
Our method is geometry-inspired and outperforms a state-of-the-art LLM editing
method on the task of generation of Wikipedia-like biography entries both in
terms of generation quality and runtime complexity.


---

**[31. [2502.13490] What are Models Thinking about? Understanding Large Language Model
  Hallucinations "Psychology" through Model Inner State Analysis](https://arxiv.org/pdf/2502.13490.pdf)** (2025-02-20)

*Peiran Wang, Yang Liu, Yunfei Lu, Jue Hong, Ye Wu*

  Large language model (LLM) systems suffer from the models' unstable ability
to generate valid and factual content, resulting in hallucination generation.
Current hallucination detection methods heavily rely on out-of-model
information sources, such as RAG to assist the detection, thus bringing heavy
additional latency. Recently, internal states of LLMs' inference have been
widely used in numerous research works, such as prompt injection detection,
etc. Considering the interpretability of LLM internal states and the fact that
they do not require external information sources, we introduce such states into
LLM hallucination detection. In this paper, we systematically analyze different
internal states' revealing features during inference forward and
comprehensively evaluate their ability in hallucination detection.
Specifically, we cut the forward process of a large language model into three
stages: understanding, query, generation, and extracting the internal state
from these stages. By analyzing these states, we provide a deep understanding
of why the hallucinated content is generated and what happened in the internal
state of the models. Then, we introduce these internal states into
hallucination detection and conduct comprehensive experiments to discuss the
advantages and limitations.


---

**[32. [2410.14748] ETF: An Entity Tracing Framework for Hallucination Detection in Code
  Summaries](https://arxiv.org/pdf/2410.14748.pdf)** (2024-12-20)

*Kishan Maharaj, Vitobha Munigala, Srikanth G. Tamilselvam, Prince Kumar, Sayandeep Sen, Palani Kodeswaran, Abhijit Mishra, Pushpak Bhattacharyya*

  Recent advancements in large language models (LLMs) have significantly
enhanced their ability to understand both natural language and code, driving
their use in tasks like natural language-to-code (NL2Code) and code
summarization. However, LLMs are prone to hallucination-outputs that stray from
intended meanings. Detecting hallucinations in code summarization is especially
difficult due to the complex interplay between programming and natural
languages. We introduce a first-of-its-kind dataset with $\sim$10K samples,
curated specifically for hallucination detection in code summarization. We
further propose a novel Entity Tracing Framework (ETF) that a) utilizes static
program analysis to identify code entities from the program and b) uses LLMs to
map and verify these entities and their intents within generated code
summaries. Our experimental analysis demonstrates the effectiveness of the
framework, leading to a 0.73 F1 score. This approach provides an interpretable
method for detecting hallucinations by grounding entities, allowing us to
evaluate summary accuracy.


---

**[33. [2503.01670] Evaluating LLMs' Assessment of Mixed-Context Hallucination Through the
  Lens of Summarization](https://arxiv.org/pdf/2503.01670.pdf)** (2025-03-04)

*Siya Qi, Rui Cao, Yulan He, Zheng Yuan*

  With the rapid development of large language models (LLMs), LLM-as-a-judge
has emerged as a widely adopted approach for text quality evaluation, including
hallucination evaluation. While previous studies have focused exclusively on
single-context evaluation (e.g., discourse faithfulness or world factuality),
real-world hallucinations typically involve mixed contexts, which remains
inadequately evaluated. In this study, we use summarization as a representative
task to comprehensively evaluate LLMs' capability in detecting mixed-context
hallucinations, specifically distinguishing between factual and non-factual
hallucinations. Through extensive experiments across direct generation and
retrieval-based models of varying scales, our main observations are: (1) LLMs'
intrinsic knowledge introduces inherent biases in hallucination evaluation; (2)
These biases particularly impact the detection of factual hallucinations,
yielding a significant performance bottleneck; (3) The fundamental challenge
lies in effective knowledge utilization, balancing between LLMs' intrinsic
knowledge and external context for accurate mixed-context hallucination
evaluation.


---

**[34. [2401.15449] Learning to Trust Your Feelings: Leveraging Self-awareness in LLMs for
  Hallucination Mitigation](https://arxiv.org/pdf/2401.15449.pdf)** (2024-01-30)

*Yuxin Liang, Zhuoyang Song, Hao Wang, Jiaxing Zhang*

  We evaluate the ability of Large Language Models (LLMs) to discern and
express their internal knowledge state, a key factor in countering factual
hallucination and ensuring reliable application of LLMs. We observe a robust
self-awareness of internal knowledge state in LLMs, evidenced by over 85%
accuracy in knowledge probing. However, LLMs often fail to express their
internal knowledge during generation, leading to factual hallucinations. We
develop an automated hallucination annotation tool, Dreamcatcher, which merges
knowledge probing and consistency checking methods to rank factual preference
data. Using knowledge preference as reward, We propose a Reinforcement Learning
from Knowledge Feedback (RLKF) training framework, leveraging reinforcement
learning to enhance the factuality and honesty of LLMs. Our experiments across
multiple models show that RLKF training effectively enhances the ability of
models to utilize their internal knowledge state, boosting performance in a
variety of knowledge-based and honesty-related tasks.


---

**[35. [2403.02889] InterrogateLLM: Zero-Resource Hallucination Detection in LLM-Generated
  Answers](https://arxiv.org/pdf/2403.02889.pdf)** (2024-08-20)

*Yakir Yehuda, Itzik Malkiel, Oren Barkan, Jonathan Weill, Royi Ronen, Noam Koenigstein*

  Despite the many advances of Large Language Models (LLMs) and their
unprecedented rapid evolution, their impact and integration into every facet of
our daily lives is limited due to various reasons. One critical factor
hindering their widespread adoption is the occurrence of hallucinations, where
LLMs invent answers that sound realistic, yet drift away from factual truth. In
this paper, we present a novel method for detecting hallucinations in large
language models, which tackles a critical issue in the adoption of these models
in various real-world scenarios. Through extensive evaluations across multiple
datasets and LLMs, including Llama-2, we study the hallucination levels of
various recent LLMs and demonstrate the effectiveness of our method to
automatically detect them. Notably, we observe up to 87% hallucinations for
Llama-2 in a specific experiment, where our method achieves a Balanced Accuracy
of 81%, all without relying on external knowledge.


---

**[36. [2412.05223] 100% Elimination of Hallucinations on RAGTruth for GPT-4 and GPT-3.5
  Turbo](https://arxiv.org/pdf/2412.05223.pdf)** (2025-03-27)

*Michael C. Wood, Adam A. Forbes*

  The issue of hallucinations in large language models (LLMs) remains a
critical barrier to the adoption of AI in enterprise and other high-stakes
applications. Despite advancements in retrieval-augmented generation (RAG)
systems, current state-of-the-art methods fail to achieve more than 80%
accuracy in generating faithful and factually correct outputs, even when
provided with relevant and accurate context. In this work, we introduce Acurai,
a novel systematic approach that achieves 100% hallucination-free responses in
LLMs by reformatting queries and context data prior to input. Leveraging a deep
understanding of LLM internal representations, the importance of noun-phrase
dominance, and the role of discrete functional units (DFUs), Acurai ensures
alignment between input context and generated output. We validate this method
using the RAGTruth corpus, demonstrating its ability to eliminate 100%
hallucinations for both GPT-4 and GPT-3.5 Turbo. Acurai sets a new standard for
achieving consistent, accurate, and faithful AI responses, marking a
significant step forward in the development of trustworthy AI systems.


---

**[37. [2503.21157] Real-Time Evaluation Models for RAG: Who Detects Hallucinations Best?](https://arxiv.org/pdf/2503.21157.pdf)** (2025-04-08)

*Ashish Sardana*

  This article surveys Evaluation models to automatically detect hallucinations
in Retrieval-Augmented Generation (RAG), and presents a comprehensive benchmark
of their performance across six RAG applications. Methods included in our study
include: LLM-as-a-Judge, Prometheus, Lynx, the Hughes Hallucination Evaluation
Model (HHEM), and the Trustworthy Language Model (TLM). These approaches are
all reference-free, requiring no ground-truth answers/labels to catch incorrect
LLM responses. Our study reveals that, across diverse RAG applications, some of
these approaches consistently detect incorrect RAG responses with high
precision/recall.


---

**[38. [2406.11514] Counterfactual Debating with Preset Stances for Hallucination
  Elimination of LLMs](https://arxiv.org/pdf/2406.11514.pdf)** (2025-01-16)

*Yi Fang, Moxin Li, Wenjie Wang, Hui Lin, Fuli Feng*

  Large Language Models (LLMs) excel in various natural language processing
tasks but struggle with hallucination issues. Existing solutions have
considered utilizing LLMs' inherent reasoning abilities to alleviate
hallucination, such as self-correction and diverse sampling methods. However,
these methods often overtrust LLMs' initial answers due to inherent biases. The
key to alleviating this issue lies in overriding LLMs' inherent biases for
answer inspection. To this end, we propose a CounterFactual Multi-Agent Debate
(CFMAD) framework. CFMAD presets the stances of LLMs to override their inherent
biases by compelling LLMs to generate justifications for a predetermined
answer's correctness. The LLMs with different predetermined stances are engaged
with a skeptical critic for counterfactual debate on the rationality of
generated justifications. Finally, the debate process is evaluated by a
third-party judge to determine the final answer. Extensive experiments on four
datasets of three tasks demonstrate the superiority of CFMAD over existing
methods.


---

**[39. [2403.06448] Unsupervised Real-Time Hallucination Detection based on the Internal
  States of Large Language Models](https://arxiv.org/pdf/2403.06448.pdf)** (2024-06-11)

*Weihang Su, Changyue Wang, Qingyao Ai, Yiran HU, Zhijing Wu, Yujia Zhou, Yiqun Liu*

  Hallucinations in large language models (LLMs) refer to the phenomenon of
LLMs producing responses that are coherent yet factually inaccurate. This issue
undermines the effectiveness of LLMs in practical applications, necessitating
research into detecting and mitigating hallucinations of LLMs. Previous studies
have mainly concentrated on post-processing techniques for hallucination
detection, which tend to be computationally intensive and limited in
effectiveness due to their separation from the LLM's inference process. To
overcome these limitations, we introduce MIND, an unsupervised training
framework that leverages the internal states of LLMs for real-time
hallucination detection without requiring manual annotations. Additionally, we
present HELM, a new benchmark for evaluating hallucination detection across
multiple LLMs, featuring diverse LLM outputs and the internal states of LLMs
during their inference process. Our experiments demonstrate that MIND
outperforms existing state-of-the-art methods in hallucination detection.


---

**[40. [2408.15533] LRP4RAG: Detecting Hallucinations in Retrieval-Augmented Generation via
  Layer-wise Relevance Propagation](https://arxiv.org/pdf/2408.15533.pdf)** (2024-08-30)

*Haichuan Hu, Yuhan Sun, Quanjun Zhang*

  Retrieval-Augmented Generation (RAG) has become a primary technique for
mitigating hallucinations in large language models (LLMs). However, incomplete
knowledge extraction and insufficient understanding can still mislead LLMs to
produce irrelevant or even contradictory responses, which means hallucinations
persist in RAG. In this paper, we propose LRP4RAG, a method based on the
Layer-wise Relevance Propagation (LRP) algorithm for detecting hallucinations
in RAG. Specifically, we first utilize LRP to compute the relevance between the
input and output of the RAG generator. We then apply further extraction and
resampling to the relevance matrix. The processed relevance data are input into
multiple classifiers to determine whether the output contains hallucinations.
To the best of our knowledge, this is the first time that LRP has been used for
detecting RAG hallucinations, and extensive experiments demonstrate that
LRP4RAG outperforms existing baselines.


---

**[41. [2503.02851] Shakespearean Sparks: The Dance of Hallucination and Creativity in LLMs'
  Decoding Layers](https://arxiv.org/pdf/2503.02851.pdf)** (2025-03-05)

*Zicong He, Boxuan Zhang, Lu Cheng*

  Large language models (LLMs) are known to hallucinate, a phenomenon often
linked to creativity. While previous research has primarily explored this
connection through theoretical or qualitative lenses, our work takes a
quantitative approach to systematically examine the relationship between
hallucination and creativity in LLMs. Given the complex nature of creativity,
we propose a narrow definition tailored to LLMs and introduce an evaluation
framework, HCL, which quantifies Hallucination and Creativity across different
Layers of LLMs during decoding. Our empirical analysis reveals a tradeoff
between hallucination and creativity that is consistent across layer depth,
model type, and model size. Notably, across different model architectures, we
identify a specific layer at each model size that optimally balances this
tradeoff. Additionally, the optimal layer tends to appear in the early layers
of larger models, and the confidence of the model is also significantly higher
at this layer. These findings provide a quantitative perspective that offers
new insights into the interplay between LLM creativity and hallucination. The
code and data for our experiments are available at
https://github.com/ZicongHe2002/HCL-Spark.


---

**[42. [2408.12748] SLM Meets LLM: Balancing Latency, Interpretability and Consistency in
  Hallucination Detection](https://arxiv.org/pdf/2408.12748.pdf)** (2024-08-26)

*Mengya Hu, Rui Xu, Deren Lei, Yaxi Li, Mingyu Wang, Emily Ching, Eslam Kamal, Alex Deng*

  Large language models (LLMs) are highly capable but face latency challenges
in real-time applications, such as conducting online hallucination detection.
To overcome this issue, we propose a novel framework that leverages a small
language model (SLM) classifier for initial detection, followed by a LLM as
constrained reasoner to generate detailed explanations for detected
hallucinated content. This study optimizes the real-time interpretable
hallucination detection by introducing effective prompting techniques that
align LLM-generated explanations with SLM decisions. Empirical experiment
results demonstrate its effectiveness, thereby enhancing the overall user
experience.


---

**[43. [2401.10768] Knowledge Verification to Nip Hallucination in the Bud](https://arxiv.org/pdf/2401.10768.pdf)** (2024-09-24)

*Fanqi Wan, Xinting Huang, Leyang Cui, Xiaojun Quan, Wei Bi, Shuming Shi*

  While large language models (LLMs) have demonstrated exceptional performance
across various tasks following human alignment, they may still generate
responses that sound plausible but contradict factual knowledge, a phenomenon
known as hallucination. In this paper, we demonstrate the feasibility of
mitigating hallucinations by verifying and minimizing the inconsistency between
external knowledge present in the alignment data and the intrinsic knowledge
embedded within foundation LLMs. Specifically, we propose a novel approach
called Knowledge Consistent Alignment (KCA), which employs a well-aligned LLM
to automatically formulate assessments based on external knowledge to evaluate
the knowledge boundaries of foundation LLMs. To address knowledge
inconsistencies in the alignment data, KCA implements several specific
strategies to deal with these data instances. We demonstrate the superior
efficacy of KCA in reducing hallucinations across six benchmarks, utilizing
foundation LLMs of varying backbones and scales. This confirms the
effectiveness of mitigating hallucinations by reducing knowledge inconsistency.
Our code, model weights, and data are openly accessible at
\url{https://github.com/fanqiwan/KCA}.


---

**[44. [2308.15126] Evaluation and Analysis of Hallucination in Large Vision-Language Models](https://arxiv.org/pdf/2308.15126.pdf)** (2023-10-11)

*Junyang Wang, Yiyang Zhou, Guohai Xu, Pengcheng Shi, Chenlin Zhao, Haiyang Xu, Qinghao Ye, Ming Yan, Ji Zhang, Jihua Zhu, Jitao Sang, Haoyu Tang*

  Large Vision-Language Models (LVLMs) have recently achieved remarkable
success. However, LVLMs are still plagued by the hallucination problem, which
limits the practicality in many scenarios. Hallucination refers to the
information of LVLMs' responses that does not exist in the visual input, which
poses potential risks of substantial consequences. There has been limited work
studying hallucination evaluation in LVLMs. In this paper, we propose
Hallucination Evaluation based on Large Language Models (HaELM), an LLM-based
hallucination evaluation framework. HaELM achieves an approximate 95%
performance comparable to ChatGPT and has additional advantages including low
cost, reproducibility, privacy preservation and local deployment. Leveraging
the HaELM, we evaluate the hallucination in current LVLMs. Furthermore, we
analyze the factors contributing to hallucination in LVLMs and offer helpful
suggestions to mitigate the hallucination problem. Our training data and human
annotation hallucination data will be made public soon.


---

**[45. [2409.02976] Hallucination Detection in LLMs: Fast and Memory-Efficient Fine-Tuned
  Models](https://arxiv.org/pdf/2409.02976.pdf)** (2024-12-09)

*Gabriel Y. Arteaga, Thomas B. Schön, Nicolas Pielawski*

  Uncertainty estimation is a necessary component when implementing AI in
high-risk settings, such as autonomous cars, medicine, or insurances. Large
Language Models (LLMs) have seen a surge in popularity in recent years, but
they are subject to hallucinations, which may cause serious harm in high-risk
settings. Despite their success, LLMs are expensive to train and run: they need
a large amount of computations and memory, preventing the use of ensembling
methods in practice. In this work, we present a novel method that allows for
fast and memory-friendly training of LLM ensembles. We show that the resulting
ensembles can detect hallucinations and are a viable approach in practice as
only one GPU is needed for training and inference.


---

**[46. [2309.05922] A Survey of Hallucination in Large Foundation Models](https://arxiv.org/pdf/2309.05922.pdf)** (2023-09-13)

*Vipula Rawte, Amit Sheth, Amitava Das*

  Hallucination in a foundation model (FM) refers to the generation of content
that strays from factual reality or includes fabricated information. This
survey paper provides an extensive overview of recent efforts that aim to
identify, elucidate, and tackle the problem of hallucination, with a particular
focus on ``Large'' Foundation Models (LFMs). The paper classifies various types
of hallucination phenomena that are specific to LFMs and establishes evaluation
criteria for assessing the extent of hallucination. It also examines existing
strategies for mitigating hallucination in LFMs and discusses potential
directions for future research in this area. Essentially, the paper offers a
comprehensive examination of the challenges and solutions related to
hallucination in LFMs.


---

**[47. [2408.13808] Towards Reliable Medical Question Answering: Techniques and Challenges
  in Mitigating Hallucinations in Language Models](https://arxiv.org/pdf/2408.13808.pdf)** (2024-08-27)

*Duy Khoa Pham, Bao Quoc Vo*

  The rapid advancement of large language models (LLMs) has significantly
impacted various domains, including healthcare and biomedicine. However, the
phenomenon of hallucination, where LLMs generate outputs that deviate from
factual accuracy or context, poses a critical challenge, especially in
high-stakes domains. This paper conducts a scoping study of existing techniques
for mitigating hallucinations in knowledge-based task in general and especially
for medical domains. Key methods covered in the paper include
Retrieval-Augmented Generation (RAG)-based techniques, iterative feedback
loops, supervised fine-tuning, and prompt engineering. These techniques, while
promising in general contexts, require further adaptation and optimization for
the medical domain due to its unique demands for up-to-date, specialized
knowledge and strict adherence to medical guidelines. Addressing these
challenges is crucial for developing trustworthy AI systems that enhance
clinical decision-making and patient safety as well as accuracy of biomedical
scientific research.


---

**[48. [2403.18051] Supervisory Prompt Training](https://arxiv.org/pdf/2403.18051.pdf)** (2024-03-28)

*Jean Ghislain Billa, Min Oh, Liang Du*

  The performance of Large Language Models (LLMs) relies heavily on the quality
of prompts, which are often manually engineered and task-specific, making them
costly and non-scalable. We propose a novel approach, Supervisory Prompt
Training (SPT). SPT automates the generation of highly effective prompts using
a dual LLM system. In this system, one LLM, the generator, performs a task
while the other, the corrector, provides feedback and generates improved
prompts. In contrast to earlier techniques, both the generator and corrector
collaboratively and continuously improve their prompts over time. We also
introduce the concept of \textit{impact scores} to measure the sentence-level
effectiveness of the prompts. Our method was tested on four benchmarks, testing
the level of hallucinations in LLMs. Notably, we were able to increase the
accuracy of GPT-4 on GSM8K from 65.8\% to 94.1\% (28.3\% increase). SPT
advances LLMs by refining prompts to enhance performance and reduce
hallucinations, offering an efficient and scalable alternative to traditional
model fine-tuning.


---

**[49. [2411.01696] Conformal Risk Minimization with Variance Reduction](https://arxiv.org/pdf/2411.01696.pdf)** (2025-02-11)

*Sima Noorani, Orlando Romero, Nicolo Dal Fabbro, Hamed Hassani, George J. Pappas*

  Conformal prediction (CP) is a distribution-free framework for achieving
probabilistic guarantees on black-box models. CP is generally applied to a
model post-training. Recent research efforts, on the other hand, have focused
on optimizing CP efficiency during training. We formalize this concept as the
problem of conformal risk minimization (CRM). In this direction, conformal
training (ConfTr) by Stutz et al.(2022) is a technique that seeks to minimize
the expected prediction set size of a model by simulating CP in-between
training updates. Despite its potential, we identify a strong source of sample
inefficiency in ConfTr that leads to overly noisy estimated gradients,
introducing training instability and limiting practical use. To address this
challenge, we propose variance-reduced conformal training (VR-ConfTr), a CRM
method that incorporates a variance reduction technique in the gradient
estimation of the ConfTr objective function. Through extensive experiments on
various benchmark datasets, we demonstrate that VR-ConfTr consistently achieves
faster convergence and smaller prediction sets compared to baselines.


---

**[50. [2412.04235] Addressing Hallucinations with RAG and NMISS in Italian Healthcare LLM
  Chatbots](https://arxiv.org/pdf/2412.04235.pdf)** (2025-02-03)

*Maria Paola Priola*

  I combine detection and mitigation techniques to addresses hallucinations in
Large Language Models (LLMs). Mitigation is achieved in a question-answering
Retrieval-Augmented Generation (RAG) framework while detection is obtained by
introducing the Negative Missing Information Scoring System (NMISS), which
accounts for contextual relevance in responses. While RAG mitigates
hallucinations by grounding answers in external data, NMISS refines the
evaluation by identifying cases where traditional metrics incorrectly flag
contextually accurate responses as hallucinations. I use Italian health news
articles as context to evaluate LLM performance. Results show that Gemma2 and
GPT-4 outperform the other models, with GPT-4 producing answers closely aligned
with reference responses. Mid-tier models, such as Llama2, Llama3, and Mistral
benefit significantly from NMISS, highlighting their ability to provide richer
contextual information. This combined approach offers new insights into the
reduction and more accurate assessment of hallucinations in LLMs, with
applications in real-world healthcare tasks and other domains.


---

**[51. [2309.01219] Siren's Song in the AI Ocean: A Survey on Hallucination in Large
  Language Models](https://arxiv.org/pdf/2309.01219.pdf)** (2023-09-26)

*Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang, Yulong Chen, Longyue Wang, Anh Tuan Luu, Wei Bi, Freda Shi, Shuming Shi*

  While large language models (LLMs) have demonstrated remarkable capabilities
across a range of downstream tasks, a significant concern revolves around their
propensity to exhibit hallucinations: LLMs occasionally generate content that
diverges from the user input, contradicts previously generated context, or
misaligns with established world knowledge. This phenomenon poses a substantial
challenge to the reliability of LLMs in real-world scenarios. In this paper, we
survey recent efforts on the detection, explanation, and mitigation of
hallucination, with an emphasis on the unique challenges posed by LLMs. We
present taxonomies of the LLM hallucination phenomena and evaluation
benchmarks, analyze existing approaches aiming at mitigating LLM hallucination,
and discuss potential directions for future research.


---

**[52. [2411.02712] V-DPO: Mitigating Hallucination in Large Vision Language Models via
  Vision-Guided Direct Preference Optimization](https://arxiv.org/pdf/2411.02712.pdf)** (2024-11-06)

*Yuxi Xie, Guanzhen Li, Xiao Xu, Min-Yen Kan*

  Large vision-language models (LVLMs) suffer from hallucination, resulting in
misalignment between the output textual response and the input visual content.
Recent research indicates that the over-reliance on the Large Language Model
(LLM) backbone, as one cause of the LVLM hallucination, inherently introduces
bias from language priors, leading to insufficient context attention to the
visual inputs.
  We tackle this issue of hallucination by mitigating such over-reliance
through preference learning. We propose Vision-guided Direct Preference
Optimization (V-DPO) to enhance visual context learning at training time. To
interpret the effectiveness and generalizability of V-DPO on different types of
training data, we construct a synthetic dataset containing both response- and
image-contrast preference pairs, compared against existing human-annotated
hallucination samples. Our approach achieves significant improvements compared
with baseline methods across various hallucination benchmarks. Our analysis
indicates that V-DPO excels in learning from image-contrast preference data,
demonstrating its superior ability to elicit and understand nuances of visual
context. Our code is publicly available at https://github.com/YuxiXie/V-DPO.


---

**[53. [2312.15710] Alleviating Hallucinations of Large Language Models through Induced
  Hallucinations](https://arxiv.org/pdf/2312.15710.pdf)** (2024-03-12)

*Yue Zhang, Leyang Cui, Wei Bi, Shuming Shi*

  Despite their impressive capabilities, large language models (LLMs) have been
observed to generate responses that include inaccurate or fabricated
information, a phenomenon commonly known as ``hallucination''. In this work, we
propose a simple \textit{Induce-then-Contrast} Decoding (ICD) strategy to
alleviate hallucinations. We first construct a factually weak LLM by inducing
hallucinations from the original LLMs. Then, we penalize these induced
hallucinations during decoding to enhance the factuality of the generated
content. Concretely, we determine the final next-token predictions by
amplifying the predictions from the original model and downplaying the induced
untruthful predictions via contrastive decoding. Experimental results on both
discrimination-based and generation-based hallucination evaluation benchmarks,
such as TruthfulQA and \textsc{FActScore}, demonstrate that our proposed ICD
methods can effectively enhance the factuality of LLMs across various model
sizes and families. For example, when equipped with ICD, Llama2-7B-Chat and
Mistral-7B-Instruct achieve performance comparable to ChatGPT and GPT4 on
TruthfulQA, respectively.


---

**[54. [2411.10436] Mitigating Hallucination in Multimodal Large Language Model via
  Hallucination-targeted Direct Preference Optimization](https://arxiv.org/pdf/2411.10436.pdf)** (2024-11-18)

*Yuhan Fu, Ruobing Xie, Xingwu Sun, Zhanhui Kang, Xirong Li*

  Multimodal Large Language Models (MLLMs) are known to hallucinate, which
limits their practical applications. Recent works have attempted to apply
Direct Preference Optimization (DPO) to enhance the performance of MLLMs, but
have shown inconsistent improvements in mitigating hallucinations. To address
this issue more effectively, we introduce Hallucination-targeted Direct
Preference Optimization (HDPO) to reduce hallucinations in MLLMs. Unlike
previous approaches, our method tackles hallucinations from their diverse forms
and causes. Specifically, we develop three types of preference pair data
targeting the following causes of MLLM hallucinations: (1) insufficient visual
capabilities, (2) long context generation, and (3) multimodal conflicts.
Experimental results demonstrate that our method achieves superior performance
across multiple hallucination evaluation datasets, surpassing most
state-of-the-art (SOTA) methods and highlighting the potential of our approach.
Ablation studies and in-depth analyses further confirm the effectiveness of our
method and suggest the potential for further improvements through scaling up.


---

**[55. [2502.19209] Bi'an: A Bilingual Benchmark and Model for Hallucination Detection in
  Retrieval-Augmented Generation](https://arxiv.org/pdf/2502.19209.pdf)** (2025-02-27)

*Zhouyu Jiang, Mengshu Sun, Zhiqiang Zhang, Lei Liang*

  Retrieval-Augmented Generation (RAG) effectively reduces hallucinations in
Large Language Models (LLMs) but can still produce inconsistent or unsupported
content. Although LLM-as-a-Judge is widely used for RAG hallucination detection
due to its implementation simplicity, it faces two main challenges: the absence
of comprehensive evaluation benchmarks and the lack of domain-optimized judge
models. To bridge these gaps, we introduce \textbf{Bi'an}, a novel framework
featuring a bilingual benchmark dataset and lightweight judge models. The
dataset supports rigorous evaluation across multiple RAG scenarios, while the
judge models are fine-tuned from compact open-source LLMs. Extensive
experimental evaluations on Bi'anBench show our 14B model outperforms baseline
models with over five times larger parameter scales and rivals state-of-the-art
closed-source LLMs. We will release our data and models soon at
https://github.com/OpenSPG/KAG.


---

**[56. [2312.05200] DelucionQA: Detecting Hallucinations in Domain-specific Question
  Answering](https://arxiv.org/pdf/2312.05200.pdf)** (2023-12-11)

*Mobashir Sadat, Zhengyu Zhou, Lukas Lange, Jun Araki, Arsalan Gundroo, Bingqing Wang, Rakesh R Menon, Md Rizwan Parvez, Zhe Feng*

  Hallucination is a well-known phenomenon in text generated by large language
models (LLMs). The existence of hallucinatory responses is found in almost all
application scenarios e.g., summarization, question-answering (QA) etc. For
applications requiring high reliability (e.g., customer-facing assistants), the
potential existence of hallucination in LLM-generated text is a critical
problem. The amount of hallucination can be reduced by leveraging information
retrieval to provide relevant background information to the LLM. However, LLMs
can still generate hallucinatory content for various reasons (e.g.,
prioritizing its parametric knowledge over the context, failure to capture the
relevant information from the context, etc.). Detecting hallucinations through
automated methods is thus paramount. To facilitate research in this direction,
we introduce a sophisticated dataset, DelucionQA, that captures hallucinations
made by retrieval-augmented LLMs for a domain-specific QA task. Furthermore, we
propose a set of hallucination detection methods to serve as baselines for
future works from the research community. Analysis and case study are also
provided to share valuable insights on hallucination phenomena in the target
scenario.


---

**[57. [2311.07914] Can Knowledge Graphs Reduce Hallucinations in LLMs? : A Survey](https://arxiv.org/pdf/2311.07914.pdf)** (2024-03-19)

*Garima Agrawal, Tharindu Kumarage, Zeyad Alghamdi, Huan Liu*

  The contemporary LLMs are prone to producing hallucinations, stemming mainly
from the knowledge gaps within the models. To address this critical limitation,
researchers employ diverse strategies to augment the LLMs by incorporating
external knowledge, aiming to reduce hallucinations and enhance reasoning
accuracy. Among these strategies, leveraging knowledge graphs as a source of
external information has demonstrated promising results. In this survey, we
comprehensively review these knowledge-graph-based augmentation techniques in
LLMs, focusing on their efficacy in mitigating hallucinations. We
systematically categorize these methods into three overarching groups, offering
methodological comparisons and performance evaluations. Lastly, this survey
explores the current trends and challenges associated with these techniques and
outlines potential avenues for future research in this emerging field.


---

**[58. [2403.03558] Benchmarking Hallucination in Large Language Models based on
  Unanswerable Math Word Problem](https://arxiv.org/pdf/2403.03558.pdf)** (2024-03-07)

*Yuhong Sun, Zhangyue Yin, Qipeng Guo, Jiawen Wu, Xipeng Qiu, Hui Zhao*

  Large language models (LLMs) are highly effective in various natural language
processing (NLP) tasks. However, they are susceptible to producing unreliable
conjectures in ambiguous contexts called hallucination. This paper presents a
new method for evaluating LLM hallucination in Question Answering (QA) based on
the unanswerable math word problem (MWP). To support this approach, we
innovatively develop a dataset called Unanswerable Math Word Problem (UMWP)
which comprises 5200 questions across five categories. We developed an
evaluation methodology combining text similarity and mathematical expression
detection to determine whether LLM considers the question unanswerable. The
results of extensive experiments conducted on 31 LLMs, including GPT-3,
InstructGPT, LLaMA, and Claude, demonstrate that in-context learning and
reinforcement learning with human feedback (RLHF) training significantly
enhance the model's ability to avoid hallucination. We show that utilizing MWP
is a reliable and effective approach to assess hallucination. Our code and data
are available at https://github.com/Yuki-Asuuna/UMWP.


---

**[59. [2408.13184] Can LLM be a Good Path Planner based on Prompt Engineering? Mitigating
  the Hallucination for Path Planning](https://arxiv.org/pdf/2408.13184.pdf)** (2024-08-28)

*Hourui Deng, Hongjie Zhang, Jie Ou, Chaosheng Feng*

  Spatial reasoning in Large Language Models (LLMs) is the foundation for
embodied intelligence. However, even in simple maze environments, LLMs still
encounter challenges in long-term path-planning, primarily influenced by their
spatial hallucination and context inconsistency hallucination by long-term
reasoning. To address this challenge, this study proposes an innovative model,
Spatial-to-Relational Transformation and Curriculum Q-Learning (S2RCQL). To
address the spatial hallucination of LLMs, we propose the Spatial-to-Relational
approach, which transforms spatial prompts into entity relations and paths
representing entity relation chains. This approach fully taps the potential of
LLMs in terms of sequential thinking. As a result, we design a path-planning
algorithm based on Q-learning to mitigate the context inconsistency
hallucination, which enhances the reasoning ability of LLMs. Using the Q-value
of state-action as auxiliary information for prompts, we correct the
hallucinations of LLMs, thereby guiding LLMs to learn the optimal path.
Finally, we propose a reverse curriculum learning technique based on LLMs to
further mitigate the context inconsistency hallucination. LLMs can rapidly
accumulate successful experiences by reducing task difficulty and leveraging
them to tackle more complex tasks. We performed comprehensive experiments based
on Baidu's self-developed LLM: ERNIE-Bot 4.0. The results showed that our
S2RCQL achieved a 23%--40% improvement in both success and optimality rates
compared with advanced prompt engineering.


---

**[60. [2410.06304] FG-PRM: Fine-grained Hallucination Detection and Mitigation in Language
  Model Mathematical Reasoning](https://arxiv.org/pdf/2410.06304.pdf)** (2024-11-19)

*Ruosen Li, Ziming Luo, Xinya Du*

  Hallucinations in large language models (LLMs) pose significant challenges in
tasks requiring complex multi-step reasoning, such as mathematical
problem-solving. Existing approaches primarily detect the presence of
hallucinations but lack a nuanced understanding of their types and
manifestations. In this paper, we first introduce a comprehensive taxonomy that
categorizes the common hallucinations in mathematical reasoning task into six
types: fabrication, factual inconsistency, context inconsistency, instruction
inconsistency, logical inconsistency, and logical error. We then propose FG-PRM
(Fine-Grained Process Reward Model), an augmented model designed to detect and
mitigate hallucinations in a fine-grained, step-level manner. To address the
limitations of manually labeling training data, we propose an automated method
for generating fine-grained hallucination data using LLMs. By injecting
hallucinations into reasoning steps of correct solutions, we create a diverse
and balanced synthetic dataset for training FG-PRM, which consists of six
specialized Process Reward Models (PRMs), each tailored to detect a specific
hallucination type. Our FG-PRM demonstrates superior performance across two key
tasks: 1) Fine-grained hallucination detection: classifying hallucination types
for each reasoning step; and 2) Verification: ranking multiple LLM-generated
outputs to select the most accurate solution, mitigating reasoning
hallucinations. Our experiments show that FG-PRM outperforms ChatGPT-3.5 and
Claude-3 on fine-grained hallucination detection and substantially boosts the
performance of LLMs on GSM8K and MATH benchmarks.


---

**[61. [2306.06085] Trapping LLM Hallucinations Using Tagged Context Prompts](https://arxiv.org/pdf/2306.06085.pdf)** (2023-06-12)

*Philip Feldman, James R. Foulds, Shimei Pan*

  Recent advances in large language models (LLMs), such as ChatGPT, have led to
highly sophisticated conversation agents. However, these models suffer from
"hallucinations," where the model generates false or fabricated information.
Addressing this challenge is crucial, particularly with AI-driven platforms
being adopted across various sectors. In this paper, we propose a novel method
to recognize and flag instances when LLMs perform outside their domain
knowledge, and ensuring users receive accurate information.
  We find that the use of context combined with embedded tags can successfully
combat hallucinations within generative language models. To do this, we
baseline hallucination frequency in no-context prompt-response pairs using
generated URLs as easily-tested indicators of fabricated data. We observed a
significant reduction in overall hallucination when context was supplied along
with question prompts for tested generative engines. Lastly, we evaluated how
placing tags within contexts impacted model responses and were able to
eliminate hallucinations in responses with 98.88% effectiveness.


---

**[62. [2502.16143] The Law of Knowledge Overshadowing: Towards Understanding, Predicting,
  and Preventing LLM Hallucination](https://arxiv.org/pdf/2502.16143.pdf)** (2025-02-25)

*Yuji Zhang, Sha Li, Cheng Qian, Jiateng Liu, Pengfei Yu, Chi Han, Yi R. Fung, Kathleen McKeown, Chengxiang Zhai, Manling Li, Heng Ji*

  Hallucination is a persistent challenge in large language models (LLMs),
where even with rigorous quality control, models often generate distorted
facts. This paradox, in which error generation continues despite high-quality
training data, calls for a deeper understanding of the underlying LLM
mechanisms. To address it, we propose a novel concept: knowledge overshadowing,
where model's dominant knowledge can obscure less prominent knowledge during
text generation, causing the model to fabricate inaccurate details. Building on
this idea, we introduce a novel framework to quantify factual hallucinations by
modeling knowledge overshadowing. Central to our approach is the log-linear
law, which predicts that the rate of factual hallucination increases linearly
with the logarithmic scale of (1) Knowledge Popularity, (2) Knowledge Length,
and (3) Model Size. The law provides a means to preemptively quantify
hallucinations, offering foresight into their occurrence even before model
training or inference. Built on overshadowing effect, we propose a new decoding
strategy CoDa, to mitigate hallucinations, which notably enhance model
factuality on Overshadow (27.9%), MemoTrap (13.1%) and NQ-Swap (18.3%). Our
findings not only deepen understandings of the underlying mechanisms behind
hallucinations but also provide actionable insights for developing more
predictable and controllable language models.


---

**[63. [2401.08358] Hallucination Detection and Hallucination Mitigation: An Investigation](https://arxiv.org/pdf/2401.08358.pdf)** (2024-01-17)

*Junliang Luo, Tianyu Li, Di Wu, Michael Jenkin, Steve Liu, Gregory Dudek*

  Large language models (LLMs), including ChatGPT, Bard, and Llama, have
achieved remarkable successes over the last two years in a range of different
applications. In spite of these successes, there exist concerns that limit the
wide application of LLMs. A key problem is the problem of hallucination.
Hallucination refers to the fact that in addition to correct responses, LLMs
can also generate seemingly correct but factually incorrect responses. This
report aims to present a comprehensive review of the current literature on both
hallucination detection and hallucination mitigation. We hope that this report
can serve as a good reference for both engineers and researchers who are
interested in LLMs and applying them to real world tasks.


---

**[64. [2404.10933] LLMem: Estimating GPU Memory Usage for Fine-Tuning Pre-Trained LLMs](https://arxiv.org/pdf/2404.10933.pdf)** (2024-04-18)

*Taeho Kim, Yanming Wang, Vatshank Chaturvedi, Lokesh Gupta, Seyeon Kim, Yongin Kwon, Sangtae Ha*

  Fine-tuning pre-trained large language models (LLMs) with limited hardware
presents challenges due to GPU memory constraints. Various distributed
fine-tuning methods have been proposed to alleviate memory constraints on GPU.
However, determining the most effective method for achieving rapid fine-tuning
while preventing GPU out-of-memory issues in a given environment remains
unclear. To address this challenge, we introduce LLMem, a solution that
estimates the GPU memory consumption when applying distributed fine-tuning
methods across multiple GPUs and identifies the optimal method. We conduct GPU
memory usage estimation prior to fine-tuning, leveraging the fundamental
structure of transformer-based decoder models and the memory usage distribution
of each method. Experimental results show that LLMem accurately estimates peak
GPU memory usage on a single GPU, with error rates of up to 1.6%. Additionally,
it shows an average error rate of 3.0% when applying distributed fine-tuning
methods to LLMs with more than a billion parameters on multi-GPU setups.


---

**[65. [2406.12053] InternalInspector $I^2$: Robust Confidence Estimation in LLMs through
  Internal States](https://arxiv.org/pdf/2406.12053.pdf)** (2024-06-19)

*Mohammad Beigi, Ying Shen, Runing Yang, Zihao Lin, Qifan Wang, Ankith Mohan, Jianfeng He, Ming Jin, Chang-Tien Lu, Lifu Huang*

  Despite their vast capabilities, Large Language Models (LLMs) often struggle
with generating reliable outputs, frequently producing high-confidence
inaccuracies known as hallucinations. Addressing this challenge, our research
introduces InternalInspector, a novel framework designed to enhance confidence
estimation in LLMs by leveraging contrastive learning on internal states
including attention states, feed-forward states, and activation states of all
layers. Unlike existing methods that primarily focus on the final activation
state, InternalInspector conducts a comprehensive analysis across all internal
states of every layer to accurately identify both correct and incorrect
prediction processes. By benchmarking InternalInspector against existing
confidence estimation methods across various natural language understanding and
generation tasks, including factual question answering, commonsense reasoning,
and reading comprehension, InternalInspector achieves significantly higher
accuracy in aligning the estimated confidence scores with the correctness of
the LLM's predictions and lower calibration error. Furthermore,
InternalInspector excels at HaluEval, a hallucination detection benchmark,
outperforming other internal-based confidence estimation methods in this task.


---

**[66. [2402.19103] Whispers that Shake Foundations: Analyzing and Mitigating False Premise
  Hallucinations in Large Language Models](https://arxiv.org/pdf/2402.19103.pdf)** (2024-03-01)

*Hongbang Yuan, Pengfei Cao, Zhuoran Jin, Yubo Chen, Daojian Zeng, Kang Liu, Jun Zhao*

  Large Language Models (LLMs) have shown impressive capabilities but still
suffer from the issue of hallucinations. A significant type of this issue is
the false premise hallucination, which we define as the phenomenon when LLMs
generate hallucinated text when confronted with false premise questions. In
this paper, we perform a comprehensive analysis of the false premise
hallucination and elucidate its internal working mechanism: a small subset of
attention heads (which we designate as false premise heads) disturb the
knowledge extraction process, leading to the occurrence of false premise
hallucination. Based on our analysis, we propose \textbf{FAITH} (\textbf{F}alse
premise \textbf{A}ttention head constra\textbf{I}ining for mi\textbf{T}igating
\textbf{H}allucinations), a novel and effective method to mitigate false
premise hallucinations. It constrains the false premise attention heads during
the model inference process. Impressively, extensive experiments demonstrate
that constraining only approximately $1\%$ of the attention heads in the model
yields a notable increase of nearly $20\%$ of model performance.


---

**[67. [2311.07397] AMBER: An LLM-free Multi-dimensional Benchmark for MLLMs Hallucination
  Evaluation](https://arxiv.org/pdf/2311.07397.pdf)** (2024-02-26)

*Junyang Wang, Yuhang Wang, Guohai Xu, Jing Zhang, Yukai Gu, Haitao Jia, Jiaqi Wang, Haiyang Xu, Ming Yan, Ji Zhang, Jitao Sang*

  Despite making significant progress in multi-modal tasks, current Multi-modal
Large Language Models (MLLMs) encounter the significant challenge of
hallucinations, which may lead to harmful consequences. Therefore, evaluating
MLLMs' hallucinations is becoming increasingly important in model improvement
and practical application deployment. Previous works are limited in high
evaluation costs (e.g., relying on humans or advanced LLMs) and insufficient
evaluation dimensions (e.g., types of tasks and hallucinations). In this paper,
we propose an LLM-free multi-dimensional benchmark AMBER, which can be used to
evaluate both generative task and discriminative task including existence,
attribute and relation hallucination. Based on AMBER, we design a low-cost and
efficient evaluation pipeline. Additionally, we conduct a comprehensive
evaluation and detailed analysis of mainstream MLLMs including GPT-4V(ision),
and also give guideline suggestions for mitigating hallucinations. The data and
code of AMBER are available at https://github.com/junyangwang0410/AMBER.


---

**[68. [2308.11764] Halo: Estimation and Reduction of Hallucinations in Open-Source Weak
  Large Language Models](https://arxiv.org/pdf/2308.11764.pdf)** (2023-09-15)

*Mohamed Elaraby, Mengyin Lu, Jacob Dunn, Xueying Zhang, Yu Wang, Shizhu Liu, Pingchuan Tian, Yuping Wang, Yuxuan Wang*

  Large Language Models (LLMs) have revolutionized Natural Language Processing
(NLP). Although convenient for research and practical applications, open-source
LLMs with fewer parameters often suffer from severe hallucinations compared to
their larger counterparts. This paper focuses on measuring and reducing
hallucinations in BLOOM 7B, a representative of such weaker open-source LLMs
that are publicly available for research and commercial applications. We
introduce HaloCheck, a lightweight BlackBox knowledge-free framework designed
to quantify the severity of hallucinations in LLMs. Additionally, we explore
techniques like knowledge injection and teacher-student approaches to alleviate
hallucinations in low-parameter LLMs. Our experiments effectively demonstrate
the reduction of hallucinations in challenging domains for these LLMs.


---

**[69. [2502.11948] Can Your Uncertainty Scores Detect Hallucinated Entity?](https://arxiv.org/pdf/2502.11948.pdf)** (2025-02-18)

*Min-Hsuan Yeh, Max Kamachee, Seongheon Park, Yixuan Li*

  To mitigate the impact of hallucination nature of LLMs, many studies propose
detecting hallucinated generation through uncertainty estimation. However,
these approaches predominantly operate at the sentence or paragraph level,
failing to pinpoint specific spans or entities responsible for hallucinated
content. This lack of granularity is especially problematic for long-form
outputs that mix accurate and fabricated information. To address this
limitation, we explore entity-level hallucination detection. We propose a new
data set, HalluEntity, which annotates hallucination at the entity level. Based
on the dataset, we comprehensively evaluate uncertainty-based hallucination
detection approaches across 17 modern LLMs. Our experimental results show that
uncertainty estimation approaches focusing on individual token probabilities
tend to over-predict hallucinations, while context-aware methods show better
but still suboptimal performance. Through an in-depth qualitative study, we
identify relationships between hallucination tendencies and linguistic
properties and highlight important directions for future research.


---

**[70. [2403.01548] In-Context Sharpness as Alerts: An Inner Representation Perspective for
  Hallucination Mitigation](https://arxiv.org/pdf/2403.01548.pdf)** (2024-03-13)

*Shiqi Chen, Miao Xiong, Junteng Liu, Zhengxuan Wu, Teng Xiao, Siyang Gao, Junxian He*

  Large language models (LLMs) frequently hallucinate and produce factual
errors, yet our understanding of why they make these errors remains limited. In
this study, we delve into the underlying mechanisms of LLM hallucinations from
the perspective of inner representations, and discover a salient pattern
associated with hallucinations: correct generations tend to have sharper
context activations in the hidden states of the in-context tokens, compared to
the incorrect ones. Leveraging this insight, we propose an entropy-based metric
to quantify the ``sharpness'' among the in-context hidden states and
incorporate it into the decoding process to formulate a constrained decoding
approach. Experiments on various knowledge-seeking and hallucination benchmarks
demonstrate our approach's consistent effectiveness, for example, achieving up
to an 8.6 point improvement on TruthfulQA. We believe this study can improve
our understanding of hallucinations and serve as a practical solution for
hallucination mitigation.


---

**[71. [2407.08488] Lynx: An Open Source Hallucination Evaluation Model](https://arxiv.org/pdf/2407.08488.pdf)** (2024-07-24)

*Selvan Sunitha Ravi, Bartosz Mielczarek, Anand Kannappan, Douwe Kiela, Rebecca Qian*

  Retrieval Augmented Generation (RAG) techniques aim to mitigate
hallucinations in Large Language Models (LLMs). However, LLMs can still produce
information that is unsupported or contradictory to the retrieved contexts. We
introduce LYNX, a SOTA hallucination detection LLM that is capable of advanced
reasoning on challenging real-world hallucination scenarios. To evaluate LYNX,
we present HaluBench, a comprehensive hallucination evaluation benchmark,
consisting of 15k samples sourced from various real-world domains. Our
experiment results show that LYNX outperforms GPT-4o, Claude-3-Sonnet, and
closed and open-source LLM-as-a-judge models on HaluBench. We release LYNX,
HaluBench and our evaluation code for public access.


---

**[72. [2410.09997] Collu-Bench: A Benchmark for Predicting Language Model Hallucinations in
  Code](https://arxiv.org/pdf/2410.09997.pdf)** (2024-10-15)

*Nan Jiang, Qi Li, Lin Tan, Tianyi Zhang*

  Despite their success, large language models (LLMs) face the critical
challenge of hallucinations, generating plausible but incorrect content. While
much research has focused on hallucinations in multiple modalities including
images and natural language text, less attention has been given to
hallucinations in source code, which leads to incorrect and vulnerable code
that causes significant financial loss. To pave the way for research in LLMs'
hallucinations in code, we introduce Collu-Bench, a benchmark for predicting
code hallucinations of LLMs across code generation (CG) and automated program
repair (APR) tasks. Collu-Bench includes 13,234 code hallucination instances
collected from five datasets and 11 diverse LLMs, ranging from open-source
models to commercial ones. To better understand and predict code
hallucinations, Collu-Bench provides detailed features such as the per-step log
probabilities of LLMs' output, token types, and the execution feedback of LLMs'
generated code for in-depth analysis. In addition, we conduct experiments to
predict hallucination on Collu-Bench, using both traditional machine learning
techniques and neural networks, which achieves 22.03 -- 33.15% accuracy. Our
experiments draw insightful findings of code hallucination patterns, reveal the
challenge of accurately localizing LLMs' hallucinations, and highlight the need
for more sophisticated techniques.


---

**[73. [2405.00648] Drowzee: Metamorphic Testing for Fact-Conflicting Hallucination
  Detection in Large Language Models](https://arxiv.org/pdf/2405.00648.pdf)** (2024-09-04)

*Ningke Li, Yuekang Li, Yi Liu, Ling Shi, Kailong Wang, Haoyu Wang*

  Large language models (LLMs) have transformed the landscape of language
processing, yet struggle with significant challenges in terms of security,
privacy, and the generation of seemingly coherent but factually inaccurate
outputs, commonly referred to as hallucinations. Among these challenges, one
particularly pressing issue is Fact-Conflicting Hallucination (FCH), where LLMs
generate content that directly contradicts established facts. Tackling FCH
poses a formidable task due to two primary obstacles: Firstly, automating the
construction and updating of benchmark datasets is challenging, as current
methods rely on static benchmarks that don't cover the diverse range of FCH
scenarios. Secondly, validating LLM outputs' reasoning process is inherently
complex, especially with intricate logical relations involved.
  In addressing these obstacles, we propose an innovative approach leveraging
logic programming to enhance metamorphic testing for detecting Fact-Conflicting
Hallucinations (FCH). Our method gathers data from sources like Wikipedia,
expands it with logical reasoning to create diverse test cases, assesses LLMs
through structured prompts, and validates their coherence using semantic-aware
assessment mechanisms. Our method generates test cases and detects
hallucinations across six different LLMs spanning nine domains, revealing
hallucination rates ranging from 24.7% to 59.8%. Key observations indicate that
LLMs encounter challenges, particularly with temporal concepts, handling
out-of-distribution knowledge, and exhibiting deficiencies in logical reasoning
capabilities. The outcomes underscore the efficacy of logic-based test cases
generated by our tool in both triggering and identifying hallucinations. These
findings underscore the imperative for ongoing collaborative endeavors within
the community to detect and address LLM hallucinations.


---

**[74. [2409.19492] MedHalu: Hallucinations in Responses to Healthcare Queries by Large
  Language Models](https://arxiv.org/pdf/2409.19492.pdf)** (2024-10-01)

*Vibhor Agarwal, Yiqiao Jin, Mohit Chandra, Munmun De Choudhury, Srijan Kumar, Nishanth Sastry*

  The remarkable capabilities of large language models (LLMs) in language
understanding and generation have not rendered them immune to hallucinations.
LLMs can still generate plausible-sounding but factually incorrect or
fabricated information. As LLM-empowered chatbots become popular, laypeople may
frequently ask health-related queries and risk falling victim to these LLM
hallucinations, resulting in various societal and healthcare implications. In
this work, we conduct a pioneering study of hallucinations in LLM-generated
responses to real-world healthcare queries from patients. We propose MedHalu, a
carefully crafted first-of-its-kind medical hallucination dataset with a
diverse range of health-related topics and the corresponding hallucinated
responses from LLMs with labeled hallucination types and hallucinated text
spans. We also introduce MedHaluDetect framework to evaluate capabilities of
various LLMs in detecting hallucinations. We also employ three groups of
evaluators -- medical experts, LLMs, and laypeople -- to study who are more
vulnerable to these medical hallucinations. We find that LLMs are much worse
than the experts. They also perform no better than laypeople and even worse in
few cases in detecting hallucinations. To fill this gap, we propose
expert-in-the-loop approach to improve hallucination detection through LLMs by
infusing expert reasoning. We observe significant performance gains for all the
LLMs with an average macro-F1 improvement of 6.3 percentage points for GPT-4.


---

**[75. [2402.16211] HypoTermQA: Hypothetical Terms Dataset for Benchmarking Hallucination
  Tendency of LLMs](https://arxiv.org/pdf/2402.16211.pdf)** (2024-02-27)

*Middle East Technical
  University  Cem Uluoglakci, Middle East Technical
  University  Tugba Taskaya Temizel*

  Hallucinations pose a significant challenge to the reliability and alignment
of Large Language Models (LLMs), limiting their widespread acceptance beyond
chatbot applications. Despite ongoing efforts, hallucinations remain a
prevalent challenge in LLMs. The detection of hallucinations itself is also a
formidable task, frequently requiring manual labeling or constrained
evaluations. This paper introduces an automated scalable framework that
combines benchmarking LLMs' hallucination tendencies with efficient
hallucination detection. We leverage LLMs to generate challenging tasks related
to hypothetical phenomena, subsequently employing them as agents for efficient
hallucination detection. The framework is domain-agnostic, allowing the use of
any language model for benchmark creation or evaluation in any domain. We
introduce the publicly available HypoTermQA Benchmarking Dataset, on which
state-of-the-art models' performance ranged between 3% and 11%, and evaluator
agents demonstrated a 6% error rate in hallucination prediction. The proposed
framework provides opportunities to test and improve LLMs. Additionally, it has
the potential to generate benchmarking datasets tailored to specific domains,
such as law, health, and finance.


---

**[76. [2310.02863] Conformal Predictions for Longitudinal Data](https://arxiv.org/pdf/2310.02863.pdf)** (2023-10-05)

*Devesh Batra, Salvatore Mercuri, Raad Khraishi*

  We introduce Longitudinal Predictive Conformal Inference (LPCI), a novel
distribution-free conformal prediction algorithm for longitudinal data. Current
conformal prediction approaches for time series data predominantly focus on the
univariate setting, and thus lack cross-sectional coverage when applied
individually to each time series in a longitudinal dataset. The current
state-of-the-art for longitudinal data relies on creating infinitely-wide
prediction intervals to guarantee both cross-sectional and asymptotic
longitudinal coverage. The proposed LPCI method addresses this by ensuring that
both longitudinal and cross-sectional coverages are guaranteed without
resorting to infinitely wide intervals. In our approach, we model the residual
data as a quantile fixed-effects regression problem, constructing prediction
intervals with a trained quantile regressor. Our extensive experiments
demonstrate that LPCI achieves valid cross-sectional coverage and outperforms
existing benchmarks in terms of longitudinal coverage rates. Theoretically, we
establish LPCI's asymptotic coverage guarantees for both dimensions, with
finite-width intervals. The robust performance of LPCI in generating reliable
prediction intervals for longitudinal data underscores its potential for broad
applications, including in medicine, finance, and supply chain management.


---

**[77. [2407.05474] Enhancing Hallucination Detection through Perturbation-Based Synthetic
  Data Generation in System Responses](https://arxiv.org/pdf/2407.05474.pdf)** (2024-07-09)

*Dongxu Zhang, Varun Gangal, Barrett Martin Lattimer, Yi Yang*

  Detecting hallucinations in large language model (LLM) outputs is pivotal,
yet traditional fine-tuning for this classification task is impeded by the
expensive and quickly outdated annotation process, especially across numerous
vertical domains and in the face of rapid LLM advancements. In this study, we
introduce an approach that automatically generates both faithful and
hallucinated outputs by rewriting system responses. Experimental findings
demonstrate that a T5-base model, fine-tuned on our generated dataset,
surpasses state-of-the-art zero-shot detectors and existing synthetic
generation methods in both accuracy and latency, indicating efficacy of our
approach.


---

**[78. [2010.08098] Agile Robot Navigation through Hallucinated Learning and Sober
  Deployment](https://arxiv.org/pdf/2010.08098.pdf)** (2021-03-09)

*Xuesu Xiao, Bo Liu, Peter Stone*

  Learning from Hallucination (LfH) is a recent machine learning paradigm for
autonomous navigation, which uses training data collected in completely safe
environments and adds numerous imaginary obstacles to make the environment
densely constrained, to learn navigation planners that produce feasible
navigation even in highly constrained (more dangerous) spaces. However, LfH
requires hallucinating the robot perception during deployment to match with the
hallucinated training data, which creates a need for sometimes-infeasible prior
knowledge and tends to generate very conservative planning. In this work, we
propose a new LfH paradigm that does not require runtime hallucination -- a
feature we call "sober deployment" -- and can therefore adapt to more realistic
navigation scenarios. This novel Hallucinated Learning and Sober Deployment
(HLSD) paradigm is tested in a benchmark testbed of 300 simulated navigation
environments with a wide range of difficulty levels, and in the real-world. In
most cases, HLSD outperforms both the original LfH method and a classical
navigation planner.


---

**[79. [2403.00425] HALC: Object Hallucination Reduction via Adaptive Focal-Contrast
  Decoding](https://arxiv.org/pdf/2403.00425.pdf)** (2024-06-11)

*Zhaorun Chen, Zhuokai Zhao, Hongyin Luo, Huaxiu Yao, Bo Li, Jiawei Zhou*

  While large vision-language models (LVLMs) have demonstrated impressive
capabilities in interpreting multi-modal contexts, they invariably suffer from
object hallucinations (OH). We introduce HALC, a novel decoding algorithm
designed to mitigate OH in LVLMs. HALC leverages distinct fine-grained optimal
visual information in vision-language tasks and operates on both local and
global contexts simultaneously. Specifically, HALC integrates a robust
auto-focal grounding mechanism (locally) to correct hallucinated tokens on the
fly, and a specialized beam search algorithm (globally) to significantly reduce
OH while preserving text generation quality. Additionally, HALC can be
integrated into any LVLMs as a plug-and-play module without extra training.
Extensive experimental studies demonstrate the effectiveness of HALC in
reducing OH, outperforming state-of-the-arts across four benchmarks.


---

**[80. [2405.01563] Mitigating LLM Hallucinations via Conformal Abstention](https://arxiv.org/pdf/2405.01563.pdf)** (2024-05-06)

*Yasin Abbasi Yadkori, Ilja Kuzborskij, David Stutz, András György, Adam Fisch, Arnaud Doucet, Iuliya Beloshapka, Wei-Hung Weng, Yao-Yuan Yang, Csaba Szepesvári, Ali Taylan Cemgil, Nenad Tomasev*

  We develop a principled procedure for determining when a large language model
(LLM) should abstain from responding (e.g., by saying "I don't know") in a
general domain, instead of resorting to possibly "hallucinating" a non-sensical
or incorrect answer. Building on earlier approaches that use self-consistency
as a more reliable measure of model confidence, we propose using the LLM itself
to self-evaluate the similarity between each of its sampled responses for a
given query. We then further leverage conformal prediction techniques to
develop an abstention procedure that benefits from rigorous theoretical
guarantees on the hallucination rate (error rate). Experimentally, our
resulting conformal abstention method reliably bounds the hallucination rate on
various closed-book, open-domain generative question answering datasets, while
also maintaining a significantly less conservative abstention rate on a dataset
with long responses (Temporal Sequences) compared to baselines using
log-probability scores to quantify uncertainty, while achieveing comparable
performance on a dataset with short answers (TriviaQA). To evaluate the
experiments automatically, one needs to determine if two responses are
equivalent given a question. Following standard practice, we use a thresholded
similarity function to determine if two responses match, but also provide a
method for calibrating the threshold based on conformal prediction, with
theoretical guarantees on the accuracy of the match prediction, which might be
of independent interest.


---

**[81. [2310.18344] Chainpoll: A high efficacy method for LLM hallucination detection](https://arxiv.org/pdf/2310.18344.pdf)** (2023-10-31)

*Robert Friel, Atindriyo Sanyal*

  Large language models (LLMs) have experienced notable advancements in
generating coherent and contextually relevant responses. However,
hallucinations - incorrect or unfounded claims - are still prevalent, prompting
the creation of automated metrics to detect these in LLM outputs. Our
contributions include: introducing ChainPoll, an innovative hallucination
detection method that excels compared to its counterparts, and unveiling
RealHall, a refined collection of benchmark datasets to assess hallucination
detection metrics from recent studies. While creating RealHall, we assessed
tasks and datasets from previous hallucination detection studies and observed
that many are not suitable for the potent LLMs currently in use. Overcoming
this, we opted for four datasets challenging for modern LLMs and pertinent to
real-world scenarios. Using RealHall, we conducted a comprehensive comparison
of ChainPoll with numerous hallucination metrics from recent studies. Our
findings indicate that ChainPoll outperforms in all RealHall benchmarks,
achieving an overall AUROC of 0.781. This surpasses the next best theoretical
method by 11% and exceeds industry standards by over 23%. Additionally,
ChainPoll is cost-effective and offers greater transparency than other metrics.
We introduce two novel metrics to assess LLM hallucinations: Adherence and
Correctness. Adherence is relevant to Retrieval Augmented Generation workflows,
evaluating an LLM's analytical capabilities within given documents and
contexts. In contrast, Correctness identifies logical and reasoning errors.


---

**[82. [2406.07070] HalluDial: A Large-Scale Benchmark for Automatic Dialogue-Level
  Hallucination Evaluation](https://arxiv.org/pdf/2406.07070.pdf)** (2024-06-12)

*Wen Luo, Tianshu Shen, Wei Li, Guangyue Peng, Richeng Xuan, Houfeng Wang, Xi Yang*

  Large Language Models (LLMs) have significantly advanced the field of Natural
Language Processing (NLP), achieving remarkable performance across diverse
tasks and enabling widespread real-world applications. However, LLMs are prone
to hallucination, generating content that either conflicts with established
knowledge or is unfaithful to the original sources. Existing hallucination
benchmarks primarily focus on sentence- or passage-level hallucination
detection, neglecting dialogue-level evaluation, hallucination localization,
and rationale provision. They also predominantly target factuality
hallucinations while underestimating faithfulness hallucinations, often relying
on labor-intensive or non-specialized evaluators. To address these limitations,
we propose HalluDial, the first comprehensive large-scale benchmark for
automatic dialogue-level hallucination evaluation. HalluDial encompasses both
spontaneous and induced hallucination scenarios, covering factuality and
faithfulness hallucinations. The benchmark includes 4,094 dialogues with a
total of 146,856 samples. Leveraging HalluDial, we conduct a comprehensive
meta-evaluation of LLMs' hallucination evaluation capabilities in
information-seeking dialogues and introduce a specialized judge language model,
HalluJudge. The high data quality of HalluDial enables HalluJudge to achieve
superior or competitive performance in hallucination evaluation, facilitating
the automatic assessment of dialogue-level hallucinations in LLMs and providing
valuable insights into this phenomenon. The dataset and the code are available
at https://github.com/FlagOpen/HalluDial.


---

**[83. [2410.11701] Magnifier Prompt: Tackling Multimodal Hallucination via Extremely Simple
  Instructions](https://arxiv.org/pdf/2410.11701.pdf)** (2025-02-24)

*Yuhan Fu, Ruobing Xie, Jiazhen Liu, Bangxiang Lan, Xingwu Sun, Zhanhui Kang, Xirong Li*

  Hallucinations in multimodal large language models (MLLMs) hinder their
practical applications. To address this, we propose a Magnifier Prompt
(MagPrompt), a simple yet effective method to tackle hallucinations in MLLMs
via extremely simple instructions. MagPrompt is based on the following two key
principles, which guide the design of various effective prompts, demonstrating
robustness: (1) MLLMs should focus more on the image. (2) When there are
conflicts between the image and the model's inner knowledge, MLLMs should
prioritize the image. MagPrompt is training-free and can be applied to
open-source and closed-source models, such as GPT-4o and Gemini-pro. It
performs well across many datasets and its effectiveness is comparable or even
better than more complex methods like VCD. Furthermore, our prompt design
principles and experimental analyses provide valuable insights into multimodal
hallucination.


---

**[84. [2404.00971] Exploring and Evaluating Hallucinations in LLM-Powered Code Generation](https://arxiv.org/pdf/2404.00971.pdf)** (2024-05-14)

*Fang Liu, Yang Liu, Lin Shi, Houkun Huang, Ruifeng Wang, Zhen Yang, Li Zhang, Zhongqi Li, Yuchi Ma*

  The rise of Large Language Models (LLMs) has significantly advanced many
applications on software engineering tasks, particularly in code generation.
Despite the promising performance, LLMs are prone to generate hallucinations,
which means LLMs might produce outputs that deviate from users' intent, exhibit
internal inconsistencies, or misalign with the factual knowledge, making the
deployment of LLMs potentially risky in a wide range of applications. Existing
work mainly focuses on investing the hallucination in the domain of natural
language generation (NLG), leaving a gap in understanding the types and extent
of hallucinations in the context of code generation. To bridge the gap, we
conducted a thematic analysis of the LLM-generated code to summarize and
categorize the hallucinations present in it. Our study established a
comprehensive taxonomy of hallucinations in LLM-generated code, encompassing 5
primary categories of hallucinations depending on the conflicting objectives
and varying degrees of deviation observed in code generation. Furthermore, we
systematically analyzed the distribution of hallucinations, exploring
variations among different LLMs and their correlation with code correctness.
Based on the results, we proposed HalluCode, a benchmark for evaluating the
performance of code LLMs in recognizing hallucinations. Hallucination
recognition and mitigation experiments with HalluCode and HumanEval show
existing LLMs face great challenges in recognizing hallucinations, particularly
in identifying their types, and are hardly able to mitigate hallucinations. We
believe our findings will shed light on future research about hallucination
evaluation, detection, and mitigation, ultimately paving the way for building
more effective and reliable code LLMs in the future.


---

**[85. [2502.07340] Aligning Large Language Models to Follow Instructions and Hallucinate
  Less via Effective Data Filtering](https://arxiv.org/pdf/2502.07340.pdf)** (2025-02-18)

*Shuzheng Si, Haozhe Zhao, Gang Chen, Cheng Gao, Yuzhuo Bai, Zhitong Wang, Kaikai An, Kangyang Luo, Chen Qian, Fanchao Qi, Baobao Chang, Maosong Sun*

  Training LLMs on data containing unfamiliar knowledge during the instruction
tuning stage can encourage hallucinations. To address this challenge, we
introduce NOVA, a novel framework designed to identify high-quality data that
aligns well with the LLM's learned knowledge to reduce hallucinations. NOVA
includes Internal Consistency Probing (ICP) and Semantic Equivalence
Identification (SEI) to measure how familiar the LLM is with instruction data.
Specifically, ICP evaluates the LLM's understanding of the given instruction by
calculating the tailored consistency among multiple self-generated responses.
SEI further assesses the familiarity of the LLM with the target response by
comparing it to the generated responses, using the proposed semantic clustering
and well-designed voting strategy. Finally, to ensure the quality of selected
samples, we introduce an expert-aligned reward model, considering
characteristics beyond just familiarity. By considering data quality and
avoiding unfamiliar data, we can utilize the selected data to effectively align
LLMs to follow instructions and hallucinate less.


---

**[86. [2406.15927] Semantic Entropy Probes: Robust and Cheap Hallucination Detection in
  LLMs](https://arxiv.org/pdf/2406.15927.pdf)** (2024-06-25)

*Jannik Kossen, Jiatong Han, Muhammed Razzak, Lisa Schut, Shreshth Malik, Yarin Gal*

  We propose semantic entropy probes (SEPs), a cheap and reliable method for
uncertainty quantification in Large Language Models (LLMs). Hallucinations,
which are plausible-sounding but factually incorrect and arbitrary model
generations, present a major challenge to the practical adoption of LLMs.
Recent work by Farquhar et al. (2024) proposes semantic entropy (SE), which can
detect hallucinations by estimating uncertainty in the space semantic meaning
for a set of model generations. However, the 5-to-10-fold increase in
computation cost associated with SE computation hinders practical adoption. To
address this, we propose SEPs, which directly approximate SE from the hidden
states of a single generation. SEPs are simple to train and do not require
sampling multiple model generations at test time, reducing the overhead of
semantic uncertainty quantification to almost zero. We show that SEPs retain
high performance for hallucination detection and generalize better to
out-of-distribution data than previous probing methods that directly predict
model accuracy. Our results across models and tasks suggest that model hidden
states capture SE, and our ablation studies give further insights into the
token positions and model layers for which this is the case.


---

**[87. [2312.14346] Don't Believe Everything You Read: Enhancing Summarization
  Interpretability through Automatic Identification of Hallucinations in Large
  Language Models](https://arxiv.org/pdf/2312.14346.pdf)** (2024-04-04)

*Priyesh Vakharia, Devavrat Joshi, Meenal Chavan, Dhananjay Sonawane, Bhrigu Garg, Parsa Mazaheri*

  Large Language Models (LLMs) are adept at text manipulation -- tasks such as
machine translation and text summarization. However, these models can also be
prone to hallucination, which can be detrimental to the faithfulness of any
answers that the model provides. Recent works in combating hallucinations in
LLMs deal with identifying hallucinated sentences and categorizing the
different ways in which models hallucinate. This paper takes a deep dive into
LLM behavior with respect to hallucinations, defines a token-level approach to
identifying different kinds of hallucinations, and further utilizes this
token-level tagging to improve the interpretability and faithfulness of LLMs in
dialogue summarization tasks. Through this, the paper presents a new, enhanced
dataset and a new training paradigm.


---

**[88. [2305.11747] HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large
  Language Models](https://arxiv.org/pdf/2305.11747.pdf)** (2023-10-24)

*Junyi Li, Xiaoxue Cheng, Wayne Xin Zhao, Jian-Yun Nie, Ji-Rong Wen*

  Large language models (LLMs), such as ChatGPT, are prone to generate
hallucinations, i.e., content that conflicts with the source or cannot be
verified by the factual knowledge. To understand what types of content and to
which extent LLMs are apt to hallucinate, we introduce the Hallucination
Evaluation benchmark for Large Language Models (HaluEval), a large collection
of generated and human-annotated hallucinated samples for evaluating the
performance of LLMs in recognizing hallucination. To generate these samples, we
propose a ChatGPT-based two-step framework, i.e., sampling-then-filtering.
Besides, we also hire some human labelers to annotate the hallucinations in
ChatGPT responses. The empirical results suggest that ChatGPT is likely to
generate hallucinated content in specific topics by fabricating unverifiable
information (i.e., about $19.5\%$ responses). Moreover, existing LLMs face
great challenges in recognizing the hallucinations in texts. However, our
experiments also prove that providing external knowledge or adding reasoning
steps can help LLMs recognize hallucinations. Our benchmark can be accessed at
https://github.com/RUCAIBox/HaluEval.


---

**[89. [2410.15460] Hallucination Detox: Sensitivity Dropout (SenD) for Large Language Model
  Training](https://arxiv.org/pdf/2410.15460.pdf)** (2025-01-08)

*Shahrad Mohammadzadeh, Juan David Guerra, Marco Bonizzato, Reihaneh Rabbany, Golnoosh Farnadi*

  As large language models (LLMs) are increasingly deployed across various
industries, concerns regarding their reliability, particularly due to
hallucinations - outputs that are factually inaccurate or irrelevant to user
input - have grown. Our research investigates the relationship between the
training process and the emergence of hallucinations to address a key gap in
existing research that focuses primarily on post hoc detection and mitigation
strategies. Using models from the Pythia suite (70M - 12B parameters) and
several hallucination detection metrics, we analyze hallucination trends
throughout training and explore LLM internal dynamics. We introduce Sensitivity
Dropout (SenD), a novel training protocol designed to mitigate hallucinations
by reducing variance during training. SenD achieves this by deterministically
dropping embedding indices with significant variability, referred to as
Sensitive Embedding Indices. In addition, we develop an unsupervised
hallucination detection metric, Efficient EigenScore (EES), which approximates
the traditional EigenScore at 2x speed. This efficient metric is integrated
into our protocol, allowing SenD to be both computationally scalable and
effective at reducing hallucinations. Our empirical evaluation demonstrates
that our approach improves LLM reliability at test time by up to 40% compared
to normal training while also providing an efficient method to improve factual
accuracy when adapting LLMs to Wikipedia, Medical, and LegalBench domains.


---

**[90. [2312.02798] Weakly Supervised Detection of Hallucinations in LLM Activations](https://arxiv.org/pdf/2312.02798.pdf)** (2023-12-06)

*Miriam Rateike, Celia Cintas, John Wamburu, Tanya Akumu, Skyler Speakman*

  We propose an auditing method to identify whether a large language model
(LLM) encodes patterns such as hallucinations in its internal states, which may
propagate to downstream tasks. We introduce a weakly supervised auditing
technique using a subset scanning approach to detect anomalous patterns in LLM
activations from pre-trained models. Importantly, our method does not need
knowledge of the type of patterns a-priori. Instead, it relies on a reference
dataset devoid of anomalies during testing. Further, our approach enables the
identification of pivotal nodes responsible for encoding these patterns, which
may offer crucial insights for fine-tuning specific sub-networks for bias
mitigation. We introduce two new scanning methods to handle LLM activations for
anomalous sentences that may deviate from the expected distribution in either
direction. Our results confirm prior findings of BERT's limited internal
capacity for encoding hallucinations, while OPT appears capable of encoding
hallucination information internally. Importantly, our scanning approach,
without prior exposure to false statements, performs comparably to a fully
supervised out-of-distribution classifier.


---

**[91. [2504.04151] STEP: Staged Parameter-Efficient Pre-training for Large Language Models](https://arxiv.org/pdf/2504.04151.pdf)** (2025-04-08)

*Kazuki Yano, Takumi Ito, Jun Suzuki*

  Pre-training large language models (LLMs) faces significant memory challenges
due to the large size of model parameters. We introduce STaged
parameter-Efficient Pre-training (STEP), which integrates parameter-efficient
tuning techniques with model growth. We conduct experiments on pre-training
LLMs of various sizes and demonstrate that STEP achieves up to a 53.9%
reduction in maximum memory requirements compared to vanilla pre-training while
maintaining equivalent performance. Furthermore, we show that the model by STEP
performs comparably to vanilla pre-trained models on downstream tasks after
instruction tuning.


---

**[92. [2402.10412] Measuring and Reducing LLM Hallucination without Gold-Standard Answers](https://arxiv.org/pdf/2402.10412.pdf)** (2024-06-10)

*Jiaheng Wei, Yuanshun Yao, Jean-Francois Ton, Hongyi Guo, Andrew Estornell, Yang Liu*

  LLM hallucination, i.e. generating factually incorrect yet seemingly
convincing answers, is currently a major threat to the trustworthiness and
reliability of LLMs. The first step towards solving this complicated problem is
to measure it. However, existing hallucination metrics require having a
benchmark dataset with gold-standard answers, i.e. "best" or "correct" answers
written by humans. Such requirements make hallucination measurement costly and
prone to human errors. In this work, we propose Factualness Evaluations via
Weighting LLMs (FEWL), an innovative hallucination metric that is specifically
designed for the scenario when gold-standard answers are absent. FEWL leverages
the answers from off-the-shelf LLMs that serve as a proxy of gold-standard
answers. The key challenge is how to quantify the expertise of reference LLMs
resourcefully. We show FEWL has certain theoretical guarantees and demonstrate
empirically it gives more accurate hallucination measures than naively using
reference LLMs. We also show how to leverage FEWL to reduce hallucination
through both in-context learning and supervised fine-tuning. Extensive
experiment results on Truthful-QA, CHALE, and HaluEval datasets demonstrate the
effectiveness of FEWL.


---

**[93. [2409.14484] Effectively Enhancing Vision Language Large Models by Prompt
  Augmentation and Caption Utilization](https://arxiv.org/pdf/2409.14484.pdf)** (2024-09-24)

*Minyi Zhao, Jie Wang, Zhaoyang Li, Jiyuan Zhang, Zhenbang Sun, Shuigeng Zhou*

  Recent studies have shown that Vision Language Large Models (VLLMs) may
output content not relevant to the input images. This problem, called the
hallucination phenomenon, undoubtedly degrades VLLM performance. Therefore,
various anti-hallucination techniques have been proposed to make model output
more reasonable and accurate. Despite their successes, from extensive tests we
found that augmenting the prompt (e.g. word appending, rewriting, and spell
error etc.) may change model output and make the output hallucinate again. To
cure this drawback, we propose a new instruct-tuning framework called Prompt
Augmentation and Caption Utilization (PACU) to boost VLLM's generation ability
under the augmented prompt scenario. Concretely, on the one hand, PACU exploits
existing LLMs to augment and evaluate diverse prompts automatically. The
resulting high-quality prompts are utilized to enhance VLLM's ability to
process different prompts. On the other hand, PACU exploits image captions to
jointly work with image features as well as the prompts for response
generation. When the visual feature is inaccurate, LLM can capture useful
information from the image captions for response generation. Extensive
experiments on hallucination evaluation and prompt-augmented datasets
demonstrate that our PACU method can work well with existing schemes to
effectively boost VLLM model performance. Code is available in
https://github.com/zhaominyiz/PACU.


---

**[94. [2501.02699] EAGLE: Enhanced Visual Grounding Minimizes Hallucinations in
  Instructional Multimodal Models](https://arxiv.org/pdf/2501.02699.pdf)** (2025-01-07)

*Andrés Villa, Juan León Alcázar, Motasem Alfarra, Vladimir Araujo, Alvaro Soto, Bernard Ghanem*

  Large language models and vision transformers have demonstrated impressive
zero-shot capabilities, enabling significant transferability in downstream
tasks. The fusion of these models has resulted in multi-modal architectures
with enhanced instructional capabilities. Despite incorporating vast image and
language pre-training, these multi-modal architectures often generate responses
that deviate from the ground truth in the image data. These failure cases are
known as hallucinations. Current methods for mitigating hallucinations
generally focus on regularizing the language component, improving the fusion
module, or ensembling multiple visual encoders to improve visual
representation. In this paper, we address the hallucination issue by directly
enhancing the capabilities of the visual component. Our approach, named EAGLE,
is fully agnostic to the LLM or fusion module and works as a post-pretraining
approach that improves the grounding and language alignment of the visual
encoder. We show that a straightforward reformulation of the original
contrastive pre-training task results in an improved visual encoder that can be
incorporated into the instructional multi-modal architecture without additional
instructional training. As a result, EAGLE achieves a significant reduction in
hallucinations across multiple challenging benchmarks and tasks.


---

**[95. [2402.06647] A Survey on Large Language Model Hallucination via a Creativity
  Perspective](https://arxiv.org/pdf/2402.06647.pdf)** (2024-02-13)

*Xuhui Jiang, Yuxing Tian, Fengrui Hua, Chengjin Xu, Yuanzhuo Wang, Jian Guo*

  Hallucinations in large language models (LLMs) are always seen as
limitations. However, could they also be a source of creativity? This survey
explores this possibility, suggesting that hallucinations may contribute to LLM
application by fostering creativity. This survey begins with a review of the
taxonomy of hallucinations and their negative impact on LLM reliability in
critical applications. Then, through historical examples and recent relevant
theories, the survey explores the potential creative benefits of hallucinations
in LLMs. To elucidate the value and evaluation criteria of this connection, we
delve into the definitions and assessment methods of creativity. Following the
framework of divergent and convergent thinking phases, the survey
systematically reviews the literature on transforming and harnessing
hallucinations for creativity in LLMs. Finally, the survey discusses future
research directions, emphasizing the need to further explore and refine the
application of hallucinations in creative processes within LLMs.


---

**[96. [2504.03579] Hallucination Detection on a Budget: Efficient Bayesian Estimation of
  Semantic Entropy](https://arxiv.org/pdf/2504.03579.pdf)** (2025-04-07)

*Kamil Ciosek, Nicolò Felicioni, Sina Ghiassian*

  Detecting whether an LLM hallucinates is an important research challenge. One
promising way of doing so is to estimate the semantic entropy (Farquhar et al.,
2024) of the distribution of generated sequences. We propose a new algorithm
for doing that, with two main advantages. First, due to us taking the Bayesian
approach, we achieve a much better quality of semantic entropy estimates for a
given budget of samples from the LLM. Second, we are able to tune the number of
samples adaptively so that `harder' contexts receive more samples. We
demonstrate empirically that our approach systematically beats the baselines,
requiring only 59% of samples used by Farquhar et al. (2024) to achieve the
same quality of hallucination detection as measured by AUROC. Moreover, quite
counterintuitively, our estimator is useful even with just one sample from the
LLM.


---

**[97. [2405.19648] Detecting Hallucinations in Large Language Model Generation: A Token
  Probability Approach](https://arxiv.org/pdf/2405.19648.pdf)** (2024-05-31)

*Ernesto Quevedo, Jorge Yero, Rachel Koerner, Pablo Rivas, Tomas Cerny*

  Concerns regarding the propensity of Large Language Models (LLMs) to produce
inaccurate outputs, also known as hallucinations, have escalated. Detecting
them is vital for ensuring the reliability of applications relying on
LLM-generated content. Current methods often demand substantial resources and
rely on extensive LLMs or employ supervised learning with multidimensional
features or intricate linguistic and semantic analyses difficult to reproduce
and largely depend on using the same LLM that hallucinated. This paper
introduces a supervised learning approach employing two simple classifiers
utilizing only four numerical features derived from tokens and vocabulary
probabilities obtained from other LLM evaluators, which are not necessarily the
same. The method yields promising results, surpassing state-of-the-art outcomes
in multiple tasks across three different benchmarks. Additionally, we provide a
comprehensive examination of the strengths and weaknesses of our approach,
highlighting the significance of the features utilized and the LLM employed as
an evaluator. We have released our code publicly at
https://github.com/Baylor-AI/HalluDetect.


---

**[98. [2312.04916] EE-LLM: Large-Scale Training and Inference of Early-Exit Large Language
  Models with 3D Parallelism](https://arxiv.org/pdf/2312.04916.pdf)** (2024-06-18)

*Yanxi Chen, Xuchen Pan, Yaliang Li, Bolin Ding, Jingren Zhou*

  We present EE-LLM, a framework for large-scale training and inference of
early-exit large language models (LLMs). While recent works have shown
preliminary evidence for the efficacy of early exiting in accelerating LLM
inference, EE-LLM makes a foundational step towards scaling up early-exit LLMs
by supporting their training and inference with massive 3D parallelism. Built
upon Megatron-LM, EE-LLM implements a variety of algorithmic innovations and
performance optimizations tailored to early exiting, including a lightweight
method that facilitates backpropagation for the early-exit training objective
with pipeline parallelism, techniques of leveraging idle resources in the
original pipeline schedule for computation related to early-exit layers, and
two approaches of early-exit inference that are compatible with KV caching for
autoregressive generation. Our analytical and empirical study shows that EE-LLM
achieves great training efficiency with negligible computational overhead
compared to standard LLM training, as well as outstanding inference speedup
without compromising output quality. To facilitate further research and
adoption, we release EE-LLM at https://github.com/pan-x-c/EE-LLM.


---

**[99. [2411.07457] DecoPrompt : Decoding Prompts Reduces Hallucinations when Large Language
  Models Meet False Premises](https://arxiv.org/pdf/2411.07457.pdf)** (2025-01-23)

*Nan Xu, Xuezhe Ma*

  While large language models (LLMs) have demonstrated increasing power, they
have also called upon studies on their hallucinated outputs that deviate from
factually correct statements. In this paper, we focus on one important scenario
of false premises, where LLMs are distracted by misaligned claims although the
model possesses the required factual knowledge to answer original questions
accurately. Inspired by the observation that entropy of the false-premise
prompt is closely related to its likelihood to elicit hallucination generation,
we propose a new prompting algorithm, named DecoPrompt, to mitigate
hallucination. DecoPrompt leverages LLMs to "decode" the false-premise prompts
without really eliciting hallucination output from LLMs. We perform experiments
on two datasets, demonstrating that DecoPrompt can reduce hallucinations
effectively on outputs from different LLMs. Moreover, DecoPrompt exhibits
cross-model transferability, which facilitates its applications to scenarios
such as LLMs of large sizes or unavailable model logits.


---

**[100. [2501.19164] Poison as Cure: Visual Noise for Mitigating Object Hallucinations in
  LVMs](https://arxiv.org/pdf/2501.19164.pdf)** (2025-02-24)

*Kejia Zhang, Keda Tao, Jiasheng Tang, Huan Wang*

  Large vision-language models (LVMs) extend large language models (LLMs) with
visual perception capabilities, enabling them to process and interpret visual
information. A major challenge compromising their reliability is object
hallucination that LVMs may generate plausible but factually inaccurate
information. We propose a novel visual adversarial perturbation (VAP) method to
mitigate this hallucination issue. VAP alleviates LVM hallucination by applying
strategically optimized visual noise without altering the base model. Our
approach formulates hallucination suppression as an optimization problem,
leveraging adversarial strategies to generate beneficial visual perturbations
that enhance the model's factual grounding and reduce parametric knowledge
bias. Extensive experimental results demonstrate that our method consistently
reduces object hallucinations across 8 state-of-the-art LVMs, validating its
efficacy across diverse evaluations.


---

**[101. [2311.15296] UHGEval: Benchmarking the Hallucination of Chinese Large Language Models
  via Unconstrained Generation](https://arxiv.org/pdf/2311.15296.pdf)** (2024-10-10)

*Xun Liang, Shichao Song, Simin Niu, Zhiyu Li, Feiyu Xiong, Bo Tang, Yezhaohui Wang, Dawei He, Peng Cheng, Zhonghao Wang, Haiying Deng*

  Large language models (LLMs) have emerged as pivotal contributors in
contemporary natural language processing and are increasingly being applied
across a diverse range of industries. However, these large-scale probabilistic
statistical models cannot currently ensure the requisite quality in
professional content generation. These models often produce hallucinated text,
compromising their practical utility in professional contexts. To assess the
authentic reliability of LLMs in text generation, numerous initiatives have
developed benchmark evaluations for hallucination phenomena. Nevertheless,
these benchmarks frequently utilize constrained generation techniques due to
cost and temporal constraints. These techniques encompass the use of directed
hallucination induction and strategies that deliberately alter authentic text
to produce hallucinations. These approaches are not congruent with the
unrestricted text generation demanded by real-world applications. Furthermore,
a well-established Chinese-language dataset dedicated to the evaluation of
hallucinations in text generation is presently lacking. Consequently, we have
developed an Unconstrained Hallucination Generation Evaluation (UHGEval)
benchmark, designed to compile outputs produced with minimal restrictions by
LLMs. Concurrently, we have established a comprehensive benchmark evaluation
framework to aid subsequent researchers in undertaking scalable and
reproducible experiments. We have also executed extensive experiments,
evaluating prominent Chinese language models and the GPT series models to
derive professional performance insights regarding hallucination challenges.


---

**[102. [2405.00253] CodeHalu: Investigating Code Hallucinations in LLMs via Execution-based
  Verification](https://arxiv.org/pdf/2405.00253.pdf)** (2025-01-22)

*Yuchen Tian, Weixiang Yan, Qian Yang, Xuandong Zhao, Qian Chen, Wen Wang, Ziyang Luo, Lei Ma, Dawn Song*

  Large Language Models (LLMs) have made significant progress in code
generation, offering developers groundbreaking automated programming support.
However, LLMs often generate code that is syntactically correct and even
semantically plausible, but may not execute as expected or fulfill specified
requirements. This phenomenon of hallucinations in the code domain has not been
systematically explored. To advance the community's understanding and research
on this issue, we introduce the concept of code hallucinations and propose a
classification method for code hallucination based on execution verification.
We categorize code hallucinations into four main types: mapping, naming,
resource, and logic hallucinations, with each category further divided into
different subcategories to understand and address the unique challenges faced
by LLMs in code generation with finer granularity. Additionally, we present a
dynamic detection algorithm called CodeHalu designed to detect and quantify
code hallucinations. We also introduce the CodeHaluEval benchmark, which
includes 8,883 samples from 699 tasks, to systematically and quantitatively
evaluate code hallucinations. By evaluating 17 popular LLMs using this
benchmark, we reveal significant differences in their accuracy and reliability
in code generation, offering detailed insights for further improving the code
generation capabilities of LLMs. The CodeHalu benchmark and code are publicly
available at https://github.com/yuchen814/CodeHalu.


---

**[103. [2402.10612] Retrieve Only When It Needs: Adaptive Retrieval Augmentation for
  Hallucination Mitigation in Large Language Models](https://arxiv.org/pdf/2402.10612.pdf)** (2024-10-01)

*Hanxing Ding, Liang Pang, Zihao Wei, Huawei Shen, Xueqi Cheng*

  Hallucinations pose a significant challenge for the practical implementation
of large language models (LLMs). The utilization of parametric knowledge in
generating factual content is constrained by the limited knowledge of LLMs,
potentially resulting in internal hallucinations. While incorporating external
information can help fill knowledge gaps, it also introduces the risk of
irrelevant information, thereby increasing the likelihood of external
hallucinations. A careful and balanced integration of the parametric knowledge
within LLMs with external information is crucial to alleviate hallucinations.
In this study, we present Rowen, a novel approach that enhances LLMs with a
selective retrieval augmentation process tailored to address hallucinated
outputs. This process is governed by a multilingual semantic-aware detection
module, which evaluates the consistency of the perturbed responses across
various languages for the same queries. Upon detecting inconsistencies
indicative of hallucinations, Rowen activates the retrieval of external
information to rectify the model outputs. Rowen adeptly harmonizes the
intrinsic parameters in LLMs with external knowledge sources, effectively
mitigating hallucinations by ensuring a balanced integration of internal
reasoning and external evidence. Through a comprehensive empirical analysis, we
demonstrate that Rowen surpasses the current state-of-the-art in both detecting
and mitigating hallucinated content within the outputs of LLMs.


---

**[104. [2502.06884] Learning Conformal Abstention Policies for Adaptive Risk Management in
  Large Language and Vision-Language Models](https://arxiv.org/pdf/2502.06884.pdf)** (2025-02-12)

*Sina Tayebati, Divake Kumar, Nastaran Darabi, Dinithi Jayasuriya, Ranganath Krishnan, Amit Ranjan Trivedi*

  Large Language and Vision-Language Models (LLMs/VLMs) are increasingly used
in safety-critical applications, yet their opaque decision-making complicates
risk assessment and reliability. Uncertainty quantification (UQ) helps assess
prediction confidence and enables abstention when uncertainty is high.
Conformal prediction (CP), a leading UQ method, provides statistical guarantees
but relies on static thresholds, which fail to adapt to task complexity and
evolving data distributions, leading to suboptimal trade-offs in accuracy,
coverage, and informativeness. To address this, we propose learnable conformal
abstention, integrating reinforcement learning (RL) with CP to optimize
abstention thresholds dynamically. By treating CP thresholds as adaptive
actions, our approach balances multiple objectives, minimizing prediction set
size while maintaining reliable coverage. Extensive evaluations across diverse
LLM/VLM benchmarks show our method outperforms Least Ambiguous Classifiers
(LAC) and Adaptive Prediction Sets (APS), improving accuracy by up to 3.2%,
boosting AUROC for hallucination detection by 22.19%, enhancing
uncertainty-guided selective generation (AUARC) by 21.17%, and reducing
calibration error by 70%-85%. These improvements hold across multiple models
and datasets while consistently meeting the 90% coverage target, establishing
our approach as a more effective and flexible solution for reliable
decision-making in safety-critical applications. The code is available at:
{https://github.com/sinatayebati/vlm-uncertainty}.


---

**[105. [2406.17663] LLM-ARC: Enhancing LLMs with an Automated Reasoning Critic](https://arxiv.org/pdf/2406.17663.pdf)** (2024-07-22)

*Aditya Kalyanpur, Kailash Karthik Saravanakumar, Victor Barres, Jennifer Chu-Carroll, David Melville, David Ferrucci*

  We introduce LLM-ARC, a neuro-symbolic framework designed to enhance the
logical reasoning capabilities of Large Language Models (LLMs), by combining
them with an Automated Reasoning Critic (ARC). LLM-ARC employs an Actor-Critic
method where the LLM Actor generates declarative logic programs along with
tests for semantic correctness, while the Automated Reasoning Critic evaluates
the code, runs the tests and provides feedback on test failures for iterative
refinement. Implemented using Answer Set Programming (ASP), LLM-ARC achieves a
new state-of-the-art accuracy of 88.32% on the FOLIO benchmark which tests
complex logical reasoning capabilities. Our experiments demonstrate significant
improvements over LLM-only baselines, highlighting the importance of logic test
generation and iterative self-refinement. We achieve our best result using a
fully automated self-supervised training loop where the Actor is trained on
end-to-end dialog traces with Critic feedback. We discuss potential
enhancements and provide a detailed error analysis, showcasing the robustness
and efficacy of LLM-ARC for complex natural language reasoning tasks.


---

**[106. [2406.11267] Mitigating Large Language Model Hallucination with Faithful Finetuning](https://arxiv.org/pdf/2406.11267.pdf)** (2024-06-18)

*Minda Hu, Bowei He, Yufei Wang, Liangyou Li, Chen Ma, Irwin King*

  Large language models (LLMs) have demonstrated remarkable performance on
various natural language processing tasks. However, they are prone to
generating fluent yet untruthful responses, known as "hallucinations".
Hallucinations can lead to the spread of misinformation and cause harm in
critical applications. Mitigating hallucinations is challenging as they arise
from factors such as noisy data, model overconfidence, lack of knowledge, and
the generation process itself. Recent efforts have attempted to address this
issue through representation editing and decoding algorithms, reducing
hallucinations without major structural changes or retraining. However, these
approaches either implicitly edit LLMs' behavior in latent space or suppress
the tendency to output unfaithful results during decoding instead of explicitly
modeling on hallucination. In this work, we introduce Faithful Finetuning (F2),
a novel method that explicitly models the process of faithful question
answering through carefully designed loss functions during fine-tuning. We
conduct extensive experiments on popular datasets and demonstrate that F2
achieves significant improvements over vanilla models and baselines.


---

**[107. [2503.14477] Calibrating Verbal Uncertainty as a Linear Feature to Reduce
  Hallucinations](https://arxiv.org/pdf/2503.14477.pdf)** (2025-03-19)

*Ziwei Ji, Lei Yu, Yeskendir Koishekenov, Yejin Bang, Anthony Hartshorn, Alan Schelten, Cheng Zhang, Pascale Fung, Nicola Cancedda*

  LLMs often adopt an assertive language style also when making false claims.
Such ``overconfident hallucinations'' mislead users and erode trust. Achieving
the ability to express in language the actual degree of uncertainty around a
claim is therefore of great importance. We find that ``verbal uncertainty'' is
governed by a single linear feature in the representation space of LLMs, and
show that this has only moderate correlation with the actual ``semantic
uncertainty'' of the model. We apply this insight and show that (1) the
mismatch between semantic and verbal uncertainty is a better predictor of
hallucinations than semantic uncertainty alone and (2) we can intervene on
verbal uncertainty at inference time and reduce hallucinations on short-form
answers, achieving an average relative reduction of 32%.


---

**[108. [2504.07863] Robust Hallucination Detection in LLMs via Adaptive Token Selection](https://arxiv.org/pdf/2504.07863.pdf)** (2025-04-11)

*Mengjia Niu, Hamed Haddadi, Guansong Pang*

  Hallucinations in large language models (LLMs) pose significant safety
concerns that impede their broader deployment. Recent research in hallucination
detection has demonstrated that LLMs' internal representations contain
truthfulness hints, which can be harnessed for detector training. However, the
performance of these detectors is heavily dependent on the internal
representations of predetermined tokens, fluctuating considerably when working
on free-form generations with varying lengths and sparse distributions of
hallucinated entities. To address this, we propose HaMI, a novel approach that
enables robust detection of hallucinations through adaptive selection and
learning of critical tokens that are most indicative of hallucinations. We
achieve this robustness by an innovative formulation of the Hallucination
detection task as Multiple Instance (HaMI) learning over token-level
representations within a sequence, thereby facilitating a joint optimisation of
token selection and hallucination detection on generation sequences of diverse
forms. Comprehensive experimental results on four hallucination benchmarks show
that HaMI significantly outperforms existing state-of-the-art approaches.


---

**[109. [2503.04693] UIPE: Enhancing LLM Unlearning by Removing Knowledge Related to
  Forgetting Targets](https://arxiv.org/pdf/2503.04693.pdf)** (2025-03-07)

*Wenyu Wang, Mengqi Zhang, Xiaotian Ye, Zhaochun Ren, Zhumin Chen, Pengjie Ren*

  Large Language Models (LLMs) inevitably acquire harmful information during
training on massive datasets. LLM unlearning aims to eliminate the influence of
such harmful information while maintaining the model's overall performance.
Existing unlearning methods, represented by gradient ascent-based approaches,
primarily focus on forgetting target data while overlooking the crucial impact
of logically related knowledge on the effectiveness of unlearning. In this
paper, through both theoretical and experimental analyses, we first demonstrate
that a key reason for the suboptimal unlearning performance is that models can
reconstruct the target content through reasoning with logically related
knowledge. To address this issue, we propose Unlearning Improvement via
Parameter Extrapolation (UIPE), a method that removes knowledge highly
correlated with the forgetting targets. Experimental results show that UIPE
significantly enhances the performance of various mainstream LLM unlearning
methods on the TOFU benchmark.


---

**[110. [2411.12591] Thinking Before Looking: Improving Multimodal LLM Reasoning via
  Mitigating Visual Hallucination](https://arxiv.org/pdf/2411.12591.pdf)** (2024-11-20)

*Haojie Zheng, Tianyang Xu, Hanchi Sun, Shu Pu, Ruoxi Chen, Lichao Sun*

  Multimodal large language models (MLLMs) have advanced the integration of
visual and linguistic modalities, establishing themselves as the dominant
paradigm for visual-language tasks. Current approaches like chain of thought
(CoT) reasoning have augmented the cognitive capabilities of large language
models (LLMs), yet their adaptation to MLLMs is hindered by heightened risks of
hallucination in cross-modality comprehension. In this paper, we find that the
thinking while looking paradigm in current multimodal CoT approaches--where
reasoning chains are generated alongside visual input--fails to mitigate
hallucinations caused by misleading images. To address these limitations, we
propose the Visual Inference Chain (VIC) framework, a novel approach that
constructs reasoning chains using textual context alone before introducing
visual input, effectively reducing cross-modal biases and enhancing multimodal
reasoning accuracy. Comprehensive evaluations demonstrate that VIC
significantly improves zero-shot performance across various vision-related
tasks, mitigating hallucinations while refining the reasoning capabilities of
MLLMs. Our code repository can be found at
https://github.com/Terry-Xu-666/visual_inference_chain.


---

**[111. [2503.03032] SAFE: A Sparse Autoencoder-Based Framework for Robust Query Enrichment
  and Hallucination Mitigation in LLMs](https://arxiv.org/pdf/2503.03032.pdf)** (2025-03-06)

*Samir Abdaljalil, Filippo Pallucchini, Andrea Seveso, Hasan Kurban, Fabio Mercorio, Erchin Serpedin*

  Despite the state-of-the-art performance of Large Language Models (LLMs),
these models often suffer from hallucinations, which can undermine their
performance in critical applications. In this work, we propose SAFE, a novel
method for detecting and mitigating hallucinations by leveraging Sparse
Autoencoders (SAEs). While hallucination detection techniques and SAEs have
been explored independently, their synergistic application in a comprehensive
system, particularly for hallucination-aware query enrichment, has not been
fully investigated. To validate the effectiveness of SAFE, we evaluate it on
two models with available SAEs across three diverse cross-domain datasets
designed to assess hallucination problems. Empirical results demonstrate that
SAFE consistently improves query generation accuracy and mitigates
hallucinations across all datasets, achieving accuracy improvements of up to
29.45%.


---

**[112. [2402.08680] Mitigating Object Hallucination in Large Vision-Language Models via
  Classifier-Free Guidance](https://arxiv.org/pdf/2402.08680.pdf)** (2024-02-14)

*Linxi Zhao, Yihe Deng, Weitong Zhang, Quanquan Gu*

  The advancement of Large Vision-Language Models (LVLMs) has increasingly
highlighted the critical issue of their tendency to hallucinate non-existing
objects in the images. To address this issue, previous works focused on using
specially curated datasets or powerful LLMs (e.g., GPT-3.5) to rectify the
outputs of LVLMs. However, these approaches require either expensive
training/fine-tuning or API access to advanced LLMs to correct the model's
output post-generation. In this paper, we tackle this challenge by introducing
a framework called Mitigating hallucinAtion via classifieR-Free guIdaNcE
(MARINE), which is both training-free and API-free, and can effectively and
efficiently reduce object hallucinations during the generation process.
Specifically, MARINE enriches the visual context of LVLMs by integrating
existing open-source vision models, and employs classifier-free guidance to
incorporate the additional object grounding features to improve the precision
of LVLMs' generations. Through comprehensive evaluations across $6$ popular
LVLMs with diverse evaluation metrics, we demonstrate the effectiveness of
MARINE, which even outperforms existing fine-tuning-based methods. Remarkably,
it not only reduces hallucinations but also improves the detailedness of LVLMs'
generations, as assessed by GPT-4V.


---

**[113. [2410.15483] Mitigating Forgetting in LLM Supervised Fine-Tuning and Preference
  Learning](https://arxiv.org/pdf/2410.15483.pdf)** (2025-02-07)

*Heshan Fernando, Han Shen, Parikshit Ram, Yi Zhou, Horst Samulowitz, Nathalie Baracaldo, Tianyi Chen*

  Post-training of pre-trained LLMs, which typically consists of the supervised
fine-tuning (SFT) stage and the preference learning (RLHF or DPO) stage, is
crucial to effective and safe LLM applications. The widely adopted approach in
post-training popular open-source LLMs is to sequentially perform SFT and
RLHF/DPO. However, sequential training is sub-optimal in terms of SFT and
RLHF/DPO trade-off: the LLM gradually forgets about the first stage's training
when undergoing the second stage's training. We theoretically prove the
sub-optimality of sequential post-training. Furthermore, we propose a practical
joint post-training framework with theoretical convergence guarantees and
empirically outperforms sequential post-training framework, while having
similar computational cost. Our code is available at
https://github.com/heshandevaka/XRIGHT.


---

**[114. [2403.18349] Rejection Improves Reliability: Training LLMs to Refuse Unknown
  Questions Using RL from Knowledge Feedback](https://arxiv.org/pdf/2403.18349.pdf)** (2024-08-09)

*Hongshen Xu, Zichen Zhu, Situo Zhang, Da Ma, Shuai Fan, Lu Chen, Kai Yu*

  Large Language Models (LLMs) often generate erroneous outputs, known as
hallucinations, due to their limitations in discerning questions beyond their
knowledge scope. While addressing hallucination has been a focal point in
research, previous efforts primarily concentrate on enhancing correctness
without giving due consideration to the significance of rejection mechanisms.
In this paper, we conduct a comprehensive examination of the role of rejection,
introducing the notion of model reliability along with corresponding metrics.
These metrics measure the model's ability to provide accurate responses while
adeptly rejecting questions exceeding its knowledge boundaries, thereby
minimizing hallucinations. To improve the inherent reliability of LLMs, we
present a novel alignment framework called Reinforcement Learning from
Knowledge Feedback (RLKF). RLKF leverages knowledge feedback to dynamically
determine the model's knowledge boundary and trains a reliable reward model to
encourage the refusal of out-of-knowledge questions. Experimental results on
mathematical questions affirm the substantial efficacy of RLKF in significantly
enhancing LLM reliability.


---

**[115. [2309.11064] Exploring the Relationship between LLM Hallucinations and Prompt
  Linguistic Nuances: Readability, Formality, and Concreteness](https://arxiv.org/pdf/2309.11064.pdf)** (2023-09-21)

*Vipula Rawte, Prachi Priya, S. M Towhidul Islam Tonmoy, S M Mehedi Zaman, Amit Sheth, Amitava Das*

  As Large Language Models (LLMs) have advanced, they have brought forth new
challenges, with one of the prominent issues being LLM hallucination. While
various mitigation techniques are emerging to address hallucination, it is
equally crucial to delve into its underlying causes. Consequently, in this
preliminary exploratory investigation, we examine how linguistic factors in
prompts, specifically readability, formality, and concreteness, influence the
occurrence of hallucinations. Our experimental results suggest that prompts
characterized by greater formality and concreteness tend to result in reduced
hallucination. However, the outcomes pertaining to readability are somewhat
inconclusive, showing a mixed pattern.


---

**[116. [2401.03205] The Dawn After the Dark: An Empirical Study on Factuality Hallucination
  in Large Language Models](https://arxiv.org/pdf/2401.03205.pdf)** (2024-01-09)

*Junyi Li, Jie Chen, Ruiyang Ren, Xiaoxue Cheng, Wayne Xin Zhao, Jian-Yun Nie, Ji-Rong Wen*

  In the era of large language models (LLMs), hallucination (i.e., the tendency
to generate factually incorrect content) poses great challenge to trustworthy
and reliable deployment of LLMs in real-world applications. To tackle the LLM
hallucination, three key questions should be well studied: how to detect
hallucinations (detection), why do LLMs hallucinate (source), and what can be
done to mitigate them (mitigation). To address these challenges, this work
presents a systematic empirical study on LLM hallucination, focused on the the
three aspects of hallucination detection, source and mitigation. Specially, we
construct a new hallucination benchmark HaluEval 2.0, and designs a simple yet
effective detection method for LLM hallucination. Furthermore, we zoom into the
different training or utilization stages of LLMs and extensively analyze the
potential factors that lead to the LLM hallucination. Finally, we implement and
examine a series of widely used techniques to mitigate the hallucinations in
LLMs. Our work has led to several important findings to understand the
hallucination origin and mitigate the hallucinations in LLMs. Our code and data
can be accessed at https://github.com/RUCAIBox/HaluEval-2.0.


---

**[117. [2310.00259] AutoHall: Automated Hallucination Dataset Generation for Large Language
  Models](https://arxiv.org/pdf/2310.00259.pdf)** (2024-07-22)

*Zouying Cao, Yifei Yang, Hai Zhao*

  While Large language models (LLMs) have garnered widespread applications
across various domains due to their powerful language understanding and
generation capabilities, the detection of non-factual or hallucinatory content
generated by LLMs remains scarce. Currently, one significant challenge in
hallucination detection is the laborious task of time-consuming and expensive
manual annotation of the hallucinatory generation. To address this issue, this
paper first introduces a method for automatically constructing model-specific
hallucination datasets based on existing fact-checking datasets called
AutoHall. Furthermore, we propose a zero-resource and black-box hallucination
detection method based on self-contradiction. We conduct experiments towards
prevalent open-/closed-source LLMs, achieving superior hallucination detection
performance compared to extant baselines. Moreover, our experiments reveal
variations in hallucination proportions and types among different models.


---

**[118. [2407.16470] Machine Translation Hallucination Detection for Low and High Resource
  Languages using Large Language Models](https://arxiv.org/pdf/2407.16470.pdf)** (2024-10-22)

*Kenza Benkirane, Laura Gongas, Shahar Pelles, Naomi Fuchs, Joshua Darmon, Pontus Stenetorp, David Ifeoluwa Adelani, Eduardo Sánchez*

  Recent advancements in massively multilingual machine translation systems
have significantly enhanced translation accuracy; however, even the best
performing systems still generate hallucinations, severely impacting user
trust. Detecting hallucinations in Machine Translation (MT) remains a critical
challenge, particularly since existing methods excel with High-Resource
Languages (HRLs) but exhibit substantial limitations when applied to
Low-Resource Languages (LRLs). This paper evaluates sentence-level
hallucination detection approaches using Large Language Models (LLMs) and
semantic similarity within massively multilingual embeddings. Our study spans
16 language directions, covering HRLs, LRLs, with diverse scripts. We find that
the choice of model is essential for performance. On average, for HRLs,
Llama3-70B outperforms the previous state of the art by as much as 0.16 MCC
(Matthews Correlation Coefficient). However, for LRLs we observe that Claude
Sonnet outperforms other LLMs on average by 0.03 MCC. The key takeaway from our
study is that LLMs can achieve performance comparable or even better than
previously proposed models, despite not being explicitly trained for any
machine translation task. However, their advantage is less significant for
LRLs.


---

**[119. [2402.09267] Self-Alignment for Factuality: Mitigating Hallucinations in LLMs via
  Self-Evaluation](https://arxiv.org/pdf/2402.09267.pdf)** (2024-06-12)

*Xiaoying Zhang, Baolin Peng, Ye Tian, Jingyan Zhou, Lifeng Jin, Linfeng Song, Haitao Mi, Helen Meng*

  Despite showing increasingly human-like abilities, large language models
(LLMs) often struggle with factual inaccuracies, i.e. "hallucinations", even
when they hold relevant knowledge. To address these hallucinations, current
approaches typically necessitate high-quality human factuality annotations. In
this work, we explore Self-Alignment for Factuality, where we leverage the
self-evaluation capability of an LLM to provide training signals that steer the
model towards factuality. Specifically, we incorporate Self-Eval, a
self-evaluation component, to prompt an LLM to validate the factuality of its
own generated responses solely based on its internal knowledge. Additionally,
we design Self-Knowledge Tuning (SK-Tuning) to augment the LLM's
self-evaluation ability by improving the model's confidence estimation and
calibration. We then utilize these self-annotated responses to fine-tune the
model via Direct Preference Optimization algorithm. We show that the proposed
self-alignment approach substantially enhances factual accuracy over Llama
family models across three key knowledge-intensive tasks on TruthfulQA and
BioGEN.


---

**[120. [2412.04141] Reducing Tool Hallucination via Reliability Alignment](https://arxiv.org/pdf/2412.04141.pdf)** (2025-02-28)

*Hongshen Xu, Zichen Zhu, Lei Pan, Zihan Wang, Su Zhu, Da Ma, Ruisheng Cao, Lu Chen, Kai Yu*

  Large Language Models (LLMs) have expanded their capabilities beyond language
generation to interact with external tools, enabling automation and real-world
applications. However, tool hallucinations, where models either select
inappropriate tools or misuse them, pose significant challenges, leading to
erroneous task execution, increased computational costs, and reduced system
reliability. To systematically address this issue, we define and categorize
tool hallucinations into two main types, tool selection hallucination and tool
usage hallucination. To evaluate and mitigate these issues, we introduce
RelyToolBench, which integrates specialized test cases and novel metrics to
assess hallucination-aware task success and efficiency. Finally, we propose
Relign, a reliability alignment framework that expands the tool-use action
space to include indecisive actions, allowing LLMs to defer tool use, seek
clarification, or adjust tool selection dynamically. Through extensive
experiments, we demonstrate that Relign significantly reduces tool
hallucinations, improves task reliability, and enhances the efficiency of LLM
tool interactions.


---

**[121. [2504.09482] HalluShift: Measuring Distribution Shifts towards Hallucination
  Detection in LLMs](https://arxiv.org/pdf/2504.09482.pdf)** (2025-04-15)

*Sharanya Dasgupta, Sujoy Nath, Arkaprabha Basu, Pourya Shamsolmoali, Swagatam Das*

  Large Language Models (LLMs) have recently garnered widespread attention due
to their adeptness at generating innovative responses to the given prompts
across a multitude of domains. However, LLMs often suffer from the inherent
limitation of hallucinations and generate incorrect information while
maintaining well-structured and coherent responses. In this work, we
hypothesize that hallucinations stem from the internal dynamics of LLMs. Our
observations indicate that, during passage generation, LLMs tend to deviate
from factual accuracy in subtle parts of responses, eventually shifting toward
misinformation. This phenomenon bears a resemblance to human cognition, where
individuals may hallucinate while maintaining logical coherence, embedding
uncertainty within minor segments of their speech. To investigate this further,
we introduce an innovative approach, HalluShift, designed to analyze the
distribution shifts in the internal state space and token probabilities of the
LLM-generated responses. Our method attains superior performance compared to
existing baselines across various benchmark datasets. Our codebase is available
at https://github.com/sharanya-dasgupta001/hallushift.


---

**[122. [2410.15778] Reducing Hallucinations in Vision-Language Models via Latent Space
  Steering](https://arxiv.org/pdf/2410.15778.pdf)** (2024-10-23)

*Sheng Liu, Haotian Ye, Lei Xing, James Zou*

  Hallucination poses a challenge to the deployment of large vision-language
models (LVLMs) in applications. Unlike in large language models (LLMs),
hallucination in LVLMs often arises from misalignments between visual inputs
and textual outputs. This paper investigates the underlying mechanisms of
hallucination, focusing on the unique structure of LVLMs that distinguishes
them from large language models (LLMs). We identify that hallucinations often
arise from the sensitivity of text decoders to vision inputs, a natural
phenomenon when image encoders and text decoders are pre-trained separately.
Inspired by this, we introduce Visual and Textual Intervention (VTI), a novel
technique designed to reduce hallucinations by steering latent space
representations during inference to enhance the stability of vision features.
As a task-agnostic test-time intervention, VTI can be easily applied to any
problem without additional cost. Extensive experiments demonstrate that it can
effectively reduce hallucinations and outperform baseline methods across
multiple metrics, highlighting the critical role of vision feature stability in
LVLMs.


---

**[123. [2502.15845] Verify when Uncertain: Beyond Self-Consistency in Black Box
  Hallucination Detection](https://arxiv.org/pdf/2502.15845.pdf)** (2025-02-25)

*Yihao Xue, Kristjan Greenewald, Youssef Mroueh, Baharan Mirzasoleiman*

  Large Language Models (LLMs) suffer from hallucination problems, which hinder
their reliability in sensitive applications. In the black-box setting, several
self-consistency-based techniques have been proposed for hallucination
detection. We empirically study these techniques and show that they achieve
performance close to that of a supervised (still black-box) oracle, suggesting
little room for improvement within this paradigm. To address this limitation,
we explore cross-model consistency checking between the target model and an
additional verifier LLM. With this extra information, we observe improved
oracle performance compared to purely self-consistency-based methods. We then
propose a budget-friendly, two-stage detection algorithm that calls the
verifier model only for a subset of cases. It dynamically switches between
self-consistency and cross-consistency based on an uncertainty interval of the
self-consistency classifier. We provide a geometric interpretation of
consistency-based hallucination detection methods through the lens of kernel
mean embeddings, offering deeper theoretical insights. Extensive experiments
show that this approach maintains high detection performance while
significantly reducing computational cost.


---

**[124. [2406.07457] Estimating the Hallucination Rate of Generative AI](https://arxiv.org/pdf/2406.07457.pdf)** (2024-12-10)

*Andrew Jesson, Nicolas Beltran-Velez, Quentin Chu, Sweta Karlekar, Jannik Kossen, Yarin Gal, John P. Cunningham, David Blei*

  This paper presents a method for estimating the hallucination rate for
in-context learning (ICL) with generative AI. In ICL, a conditional generative
model (CGM) is prompted with a dataset and a prediction question and asked to
generate a response. One interpretation of ICL assumes that the CGM computes
the posterior predictive of an unknown Bayesian model, which implicitly defines
a joint distribution over observable datasets and latent mechanisms. This joint
distribution factorizes into two components: the model prior over mechanisms
and the model likelihood of datasets given a mechanism. With this perspective,
we define a hallucination as a generated response to the prediction question
with low model likelihood given the mechanism. We develop a new method that
takes an ICL problem and estimates the probability that a CGM will generate a
hallucination. Our method only requires generating prediction questions and
responses from the CGM and evaluating its response log probability. We
empirically evaluate our method using large language models for synthetic
regression and natural language ICL tasks.


---

**[125. [2501.13824] Hallucinations Can Improve Large Language Models in Drug Discovery](https://arxiv.org/pdf/2501.13824.pdf)** (2025-01-24)

*Shuzhou Yuan, Michael Färber*

  Concerns about hallucinations in Large Language Models (LLMs) have been
raised by researchers, yet their potential in areas where creativity is vital,
such as drug discovery, merits exploration. In this paper, we come up with the
hypothesis that hallucinations can improve LLMs in drug discovery. To verify
this hypothesis, we use LLMs to describe the SMILES string of molecules in
natural language and then incorporate these descriptions as part of the prompt
to address specific tasks in drug discovery. Evaluated on seven LLMs and five
classification tasks, our findings confirm the hypothesis: LLMs can achieve
better performance with text containing hallucinations. Notably, Llama-3.1-8B
achieves an 18.35% gain in ROC-AUC compared to the baseline without
hallucination. Furthermore, hallucinations generated by GPT-4o provide the most
consistent improvements across models. Additionally, we conduct empirical
analyses and a case study to investigate key factors affecting performance and
the underlying reasons. Our research sheds light on the potential use of
hallucinations for LLMs and offers new perspectives for future research
leveraging LLMs in drug discovery.


---

**[126. [2503.06709] Delusions of Large Language Models](https://arxiv.org/pdf/2503.06709.pdf)** (2025-03-11)

*Hongshen Xu, Zixv yang, Zichen Zhu, Kunyao Lan, Zihan Wang, Mengyue Wu, Ziwei Ji, Lu Chen, Pascale Fung, Kai Yu*

  Large Language Models often generate factually incorrect but plausible
outputs, known as hallucinations. We identify a more insidious phenomenon, LLM
delusion, defined as high belief hallucinations, incorrect outputs with
abnormally high confidence, making them harder to detect and mitigate. Unlike
ordinary hallucinations, delusions persist with low uncertainty, posing
significant challenges to model reliability. Through empirical analysis across
different model families and sizes on several Question Answering tasks, we show
that delusions are prevalent and distinct from hallucinations. LLMs exhibit
lower honesty with delusions, which are harder to override via finetuning or
self reflection. We link delusion formation with training dynamics and dataset
noise and explore mitigation strategies such as retrieval augmented generation
and multi agent debating to mitigate delusions. By systematically investigating
the nature, prevalence, and mitigation of LLM delusions, our study provides
insights into the underlying causes of this phenomenon and outlines future
directions for improving model reliability.


---

**[127. [2410.20024] Beyond Fine-Tuning: Effective Strategies for Mitigating Hallucinations
  in Large Language Models for Data Analytics](https://arxiv.org/pdf/2410.20024.pdf)** (2024-10-29)

*Mikhail Rumiantsau, Aliaksei Vertsel, Ilya Hrytsuk, Isaiah Ballah*

  Large Language Models (LLMs) have become increasingly important in natural
language processing, enabling advanced data analytics through natural language
queries. However, these models often generate "hallucinations"-inaccurate or
fabricated information-that can undermine their reliability in critical
data-driven decision-making. Addressing the challenge of hallucinations is
essential to improve the accuracy and trustworthiness of LLMs in processing
natural language queries. This research focuses on mitigating hallucinations in
LLMs, specifically within the context of data analytics. We introduce and
evaluate four targeted strategies: Structured Output Generation, Strict Rules
Enforcement, System Prompt Enhancements, and Semantic Layer Integration. Our
findings show that these methods are more effective than traditional
fine-tuning approaches in reducing hallucinations, offering a more reliable
framework for deploying LLMs in natural language queries for data analytics.
This research demonstrates the potential of these strategies to enhance the
accuracy of LLM-driven data queries, ensuring more dependable results in
data-driven environments.


---

**[128. [2407.09152] The Two Sides of the Coin: Hallucination Generation and Detection with
  LLMs as Evaluators for LLMs](https://arxiv.org/pdf/2407.09152.pdf)** (2024-07-15)

*Anh Thu Maria Bui, Saskia Felizitas Brech, Natalie Hußfeldt, Tobias Jennert, Melanie Ullrich, Timo Breuer, Narjes Nikzad Khasmakhi, Philipp Schaer*

  Hallucination detection in Large Language Models (LLMs) is crucial for
ensuring their reliability. This work presents our participation in the CLEF
ELOQUENT HalluciGen shared task, where the goal is to develop evaluators for
both generating and detecting hallucinated content. We explored the
capabilities of four LLMs: Llama 3, Gemma, GPT-3.5 Turbo, and GPT-4, for this
purpose. We also employed ensemble majority voting to incorporate all four
models for the detection task. The results provide valuable insights into the
strengths and weaknesses of these LLMs in handling hallucination generation and
detection tasks.


---

**[129. [2502.20034] Vision-Encoders (Already) Know What They See: Mitigating Object
  Hallucination via Simple Fine-Grained CLIPScore](https://arxiv.org/pdf/2502.20034.pdf)** (2025-02-28)

*Hongseok Oh, Wonseok Hwang*

  Recently, Large Vision-Language Models (LVLMs) show remarkable performance
across various domains. However, these models suffer from object hallucination.
This study revisits the previous claim that the primary cause of such
hallucination lies in the limited representational capacity of the vision
encoder. Our analysis reveals that the capacity of the vision encoder itself is
already enough for detecting object hallucination. Based on this insight, we
propose a Fine-grained CLIPScore (F-CLIPScore), a simple yet effective
evaluation metric that enhances object-level granularity by incorporating text
embeddings at the noun phrase level. Evaluations on the OHD-Caps benchmark show
that F-CLIPScore significantly outperforms conventional CLIPScore in accuracy
by a large margin of 39.6% without additional training. We further validate
F-CLIPScore by showing that LVLM trained with the data filtered using
F-CLIPScore exhibits reduced hallucination.


---

**[130. [2404.02935] KnowHalu: Hallucination Detection via Multi-Form Knowledge Based Factual
  Checking](https://arxiv.org/pdf/2404.02935.pdf)** (2024-04-05)

*Jiawei Zhang, Chejian Xu, Yu Gai, Freddy Lecue, Dawn Song, Bo Li*

  This paper introduces KnowHalu, a novel approach for detecting hallucinations
in text generated by large language models (LLMs), utilizing step-wise
reasoning, multi-formulation query, multi-form knowledge for factual checking,
and fusion-based detection mechanism. As LLMs are increasingly applied across
various domains, ensuring that their outputs are not hallucinated is critical.
Recognizing the limitations of existing approaches that either rely on the
self-consistency check of LLMs or perform post-hoc fact-checking without
considering the complexity of queries or the form of knowledge, KnowHalu
proposes a two-phase process for hallucination detection. In the first phase,
it identifies non-fabrication hallucinations--responses that, while factually
correct, are irrelevant or non-specific to the query. The second phase,
multi-form based factual checking, contains five key steps: reasoning and query
decomposition, knowledge retrieval, knowledge optimization, judgment
generation, and judgment aggregation. Our extensive evaluations demonstrate
that KnowHalu significantly outperforms SOTA baselines in detecting
hallucinations across diverse tasks, e.g., improving by 15.65% in QA tasks and
5.50% in summarization tasks, highlighting its effectiveness and versatility in
detecting hallucinations in LLM-generated content.


---

**[131. [2212.05765] Information-Theoretic Text Hallucination Reduction for Video-grounded
  Dialogue](https://arxiv.org/pdf/2212.05765.pdf)** (2023-10-10)

*Sunjae Yoon, Eunseop Yoon, Hee Suk Yoon, Junyeong Kim, Chang D. Yoo*

  Video-grounded Dialogue (VGD) aims to decode an answer sentence to a question
regarding a given video and dialogue context. Despite the recent success of
multi-modal reasoning to generate answer sentences, existing dialogue systems
still suffer from a text hallucination problem, which denotes indiscriminate
text-copying from input texts without an understanding of the question. This is
due to learning spurious correlations from the fact that answer sentences in
the dataset usually include the words of input texts, thus the VGD system
excessively relies on copying words from input texts by hoping those words to
overlap with ground-truth texts. Hence, we design Text Hallucination Mitigating
(THAM) framework, which incorporates Text Hallucination Regularization (THR)
loss derived from the proposed information-theoretic text hallucination
measurement approach. Applying THAM with current dialogue systems validates the
effectiveness on VGD benchmarks (i.e., AVSD@DSTC7 and AVSD@DSTC8) and shows
enhanced interpretability.


---

**[132. [2402.12913] OPDAI at SemEval-2024 Task 6: Small LLMs can Accelerate Hallucination
  Detection with Weakly Supervised Data](https://arxiv.org/pdf/2402.12913.pdf)** (2024-02-21)

*Chengcheng Wei, Ze Chen, Songtan Fang, Jiarong He, Max Gao*

  This paper mainly describes a unified system for hallucination detection of
LLMs, which wins the second prize in the model-agnostic track of the
SemEval-2024 Task 6, and also achieves considerable results in the model-aware
track. This task aims to detect hallucination with LLMs for three different
text-generation tasks without labeled training data. We utilize prompt
engineering and few-shot learning to verify the performance of different LLMs
on the validation data. Then we select the LLMs with better performance to
generate high-quality weakly supervised training data, which not only satisfies
the consistency of different LLMs, but also satisfies the consistency of the
optimal LLM with different sampling parameters. Furthermore, we finetune
different LLMs by using the constructed training data, and finding that a
relatively small LLM can achieve a competitive level of performance in
hallucination detection, when compared to the large LLMs and the prompt-based
approaches using GPT-4.


---

**[133. [2410.16251] Can Knowledge Editing Really Correct Hallucinations?](https://arxiv.org/pdf/2410.16251.pdf)** (2025-03-04)

*Baixiang Huang, Canyu Chen, Xiongxiao Xu, Ali Payani, Kai Shu*

  Large Language Models (LLMs) suffer from hallucinations, referring to the
non-factual information in generated content, despite their superior capacities
across tasks. Meanwhile, knowledge editing has been developed as a new popular
paradigm to correct erroneous factual knowledge encoded in LLMs with the
advantage of avoiding retraining from scratch. However, a common issue of
existing evaluation datasets for knowledge editing is that they do not ensure
that LLMs actually generate hallucinated answers to the evaluation questions
before editing. When LLMs are evaluated on such datasets after being edited by
different techniques, it is hard to directly adopt the performance to assess
the effectiveness of different knowledge editing methods in correcting
hallucinations. Thus, the fundamental question remains insufficiently
validated: Can knowledge editing really correct hallucinations in LLMs? We
proposed HalluEditBench to holistically benchmark knowledge editing methods in
correcting real-world hallucinations. First, we rigorously construct a massive
hallucination dataset with 9 domains, 26 topics and more than 6,000
hallucinations. Then, we assess the performance of knowledge editing methods in
a holistic way on five dimensions including Efficacy, Generalization,
Portability, Locality, and Robustness. Through HalluEditBench, we have provided
new insights into the potentials and limitations of different knowledge editing
methods in correcting hallucinations, which could inspire future improvements
and facilitate progress in the field of knowledge editing.


---

**[134. [2503.01921] NCL-UoR at SemEval-2025 Task 3: Detecting Multilingual Hallucination and
  Related Observable Overgeneration Text Spans with Modified RefChecker and
  Modified SeflCheckGPT](https://arxiv.org/pdf/2503.01921.pdf)** (2025-03-05)

*Jiaying Hong, Thanet Markchom, Jianfei Xu, Tong Wu, Huizhi Liang*

  SemEval-2025 Task 3 (Mu-SHROOM) focuses on detecting hallucinations in
content generated by various large language models (LLMs) across multiple
languages. This task involves not only identifying the presence of
hallucinations but also pinpointing their specific occurrences. To tackle this
challenge, this study introduces two methods: modified RefChecker and modified
SelfCheckGPT. The modified RefChecker integrates prompt-based factual
verification into References, structuring them as claim-based tests rather than
single external knowledge sources. The modified SelfCheckGPT incorporates
external knowledge to overcome its reliance on internal knowledge. In addition,
both methods' original prompt designs are enhanced to identify hallucinated
words within LLM-generated texts. Experimental results demonstrate the
effectiveness of the approach, achieving a high ranking on the test dataset in
detecting hallucinations across various languages, with an average IoU of
0.5310 and an average COR of 0.5669.


---

**[135. [2501.02486] LLMPC: Large Language Model Predictive Control](https://arxiv.org/pdf/2501.02486.pdf)** (2025-02-26)

*Gabriel Maher*

  Recent advancements in prompting techniques for Large Language Models (LLMs)
have improved their reasoning, planning, and action abilities. This paper
examines these prompting techniques through the lens of model predictive
control (MPC). We show that LLMs act as implicit planning cost function
minimizers when planning prompts are used. We propose a unified MPC framework
for planning with LLMs and demonstrate improved performance over few shot
prompting on several planning benchmarks.


---

**[136. [2503.01917] How to Steer LLM Latents for Hallucination Detection?](https://arxiv.org/pdf/2503.01917.pdf)** (2025-03-05)

*Seongheon Park, Xuefeng Du, Min-Hsuan Yeh, Haobo Wang, Yixuan Li*

  Hallucinations in LLMs pose a significant concern to their safe deployment in
real-world applications. Recent approaches have leveraged the latent space of
LLMs for hallucination detection, but their embeddings, optimized for
linguistic coherence rather than factual accuracy, often fail to clearly
separate truthful and hallucinated content. To this end, we propose the
Truthfulness Separator Vector (TSV), a lightweight and flexible steering vector
that reshapes the LLM's representation space during inference to enhance the
separation between truthful and hallucinated outputs, without altering model
parameters. Our two-stage framework first trains TSV on a small set of labeled
exemplars to form compact and well-separated clusters. It then augments the
exemplar set with unlabeled LLM generations, employing an optimal
transport-based algorithm for pseudo-labeling combined with a confidence-based
filtering process. Extensive experiments demonstrate that TSV achieves
state-of-the-art performance with minimal labeled data, exhibiting strong
generalization across datasets and providing a practical solution for
real-world LLM applications.


---

**[137. [2408.04664] Mitigating Hallucinations in Large Vision-Language Models (LVLMs) via
  Language-Contrastive Decoding (LCD)](https://arxiv.org/pdf/2408.04664.pdf)** (2024-08-12)

*Avshalom Manevich, Reut Tsarfaty*

  Large Vision-Language Models (LVLMs) are an extension of Large Language
Models (LLMs) that facilitate processing both image and text inputs, expanding
AI capabilities. However, LVLMs struggle with object hallucinations due to
their reliance on text cues and learned object co-occurrence biases. While most
research quantifies these hallucinations, mitigation strategies are still
lacking. Our study introduces a Language Contrastive Decoding (LCD) algorithm
that adjusts LVLM outputs based on LLM distribution confidence levels,
effectively reducing object hallucinations. We demonstrate the advantages of
LCD in leading LVLMs, showing up to %4 improvement in POPE F1 scores and up to
%36 reduction in CHAIR scores on the COCO validation set, while also improving
captioning quality scores. Our method effectively improves LVLMs without
needing complex post-processing or retraining, and is easily applicable to
different models. Our findings highlight the potential of further exploration
of LVLM-specific decoding algorithms.


---

**[138. [2310.18794] Sequence-Level Certainty Reduces Hallucination In Knowledge-Grounded
  Dialogue Generation](https://arxiv.org/pdf/2310.18794.pdf)** (2024-04-16)

*Yixin Wan, Fanyou Wu, Weijie Xu, Srinivasan H. Sengamedu*

  In this work, we propose sequence-level certainty as a common theme over
hallucination in Knowledge Grounded Dialogue Generation (KGDG). We explore the
correlation between the level of hallucination in model responses and two types
of sequence-level certainty: probabilistic certainty and semantic certainty.
Empirical results reveal that higher levels of both types of certainty in model
responses are correlated with lower levels of hallucination. We further propose
Certainty-based Response Ranking (CRR), a decoding-time hallucination
mitigation method that samples several response candidates, ranks them based on
sequence-level certainty, and outputs the response with the highest certainty
level. Aligning with our definitions of sequence-level certainty, we design 2
types of CRR approaches: Probabilistic CRR (P-CRR) and Semantic CRR (S-CRR).
P-CRR ranks individually sampled model responses using the arithmetic mean
log-probability of the entire sequence. S-CRR approaches certainty estimation
from meaning-space, and ranks model response candidates based on their semantic
certainty level as measured by an entailment-based Agreement Score (AS).
Through extensive experiments across 3 KGDG datasets, 3 decoding methods, and 4
KGDG models, we validate the effectiveness of CRR for reducing hallucination in
KGDG task.


---

**[139. [2407.17468] WildHallucinations: Evaluating Long-form Factuality in LLMs with
  Real-World Entity Queries](https://arxiv.org/pdf/2407.17468.pdf)** (2024-07-25)

*Wenting Zhao, Tanya Goyal, Yu Ying Chiu, Liwei Jiang, Benjamin Newman, Abhilasha Ravichander, Khyathi Chandu, Ronan Le Bras, Claire Cardie, Yuntian Deng, Yejin Choi*

  While hallucinations of large language models (LLMs) prevail as a major
challenge, existing evaluation benchmarks on factuality do not cover the
diverse domains of knowledge that the real-world users of LLMs seek information
about. To bridge this gap, we introduce WildHallucinations, a benchmark that
evaluates factuality. It does so by prompting LLMs to generate information
about entities mined from user-chatbot conversations in the wild. These
generations are then automatically fact-checked against a systematically
curated knowledge source collected from web search. Notably, half of these
real-world entities do not have associated Wikipedia pages. We evaluate 118,785
generations from 15 LLMs on 7,919 entities. We find that LLMs consistently
hallucinate more on entities without Wikipedia pages and exhibit varying
hallucination rates across different domains. Finally, given the same base
models, adding a retrieval component only slightly reduces hallucinations but
does not eliminate hallucinations.


---

**[140. [2407.04693] ANAH-v2: Scaling Analytical Hallucination Annotation of Large Language
  Models](https://arxiv.org/pdf/2407.04693.pdf)** (2024-12-20)

*Yuzhe Gu, Ziwei Ji, Wenwei Zhang, Chengqi Lyu, Dahua Lin, Kai Chen*

  Large language models (LLMs) exhibit hallucinations in long-form
question-answering tasks across various domains and wide applications. Current
hallucination detection and mitigation datasets are limited in domains and
sizes, which struggle to scale due to prohibitive labor costs and insufficient
reliability of existing hallucination annotators. To facilitate the scalable
oversight of LLM hallucinations, this paper introduces an iterative
self-training framework that simultaneously and progressively scales up the
hallucination annotation dataset and improves the accuracy of the hallucination
annotator. Based on the Expectation Maximization (EM) algorithm, in each
iteration, the framework first applies a hallucination annotation pipeline to
annotate a scaled dataset and then trains a more accurate hallucination
annotator on the dataset. This new hallucination annotator is adopted in the
hallucination annotation pipeline used for the next iteration. Extensive
experimental results demonstrate that the finally obtained hallucination
annotator with only 7B parameters surpasses the performance of GPT-4 and
obtains new state-of-the-art hallucination detection results on HaluEval and
HalluQA by zero-shot inference. Such an annotator can not only evaluate the
hallucination levels of various LLMs on the large-scale dataset but also help
to mitigate the hallucination of LLMs generations, with the Natural Language
Inference (NLI) metric increasing from 25% to 37% on HaluEval.


---

**[141. [2504.10063] Hallucination Detection in LLMs via Topological Divergence on Attention
  Graphs](https://arxiv.org/pdf/2504.10063.pdf)** (2025-04-15)

*Alexandra Bazarova, Aleksandr Yugay, Andrey Shulga, Alina Ermilova, Andrei Volodichev, Konstantin Polev, Julia Belikova, Rauf Parchiev, Dmitry Simakov, Maxim Savchenko, Andrey Savchenko, Serguei Barannikov, Alexey Zaytsev*

  Hallucination, i.e., generating factually incorrect content, remains a
critical challenge for large language models (LLMs). We introduce TOHA, a
TOpology-based HAllucination detector in the RAG setting, which leverages a
topological divergence metric to quantify the structural properties of graphs
induced by attention matrices. Examining the topological divergence between
prompt and response subgraphs reveals consistent patterns: higher divergence
values in specific attention heads correlate with hallucinated outputs,
independent of the dataset. Extensive experiments, including evaluation on
question answering and data-to-text tasks, show that our approach achieves
state-of-the-art or competitive results on several benchmarks, two of which
were annotated by us and are being publicly released to facilitate further
research. Beyond its strong in-domain performance, TOHA maintains remarkable
domain transferability across multiple open-source LLMs. Our findings suggest
that analyzing the topological structure of attention matrices can serve as an
efficient and robust indicator of factual reliability in LLMs.


---

**[142. [2404.08189] Reducing hallucination in structured outputs via Retrieval-Augmented
  Generation](https://arxiv.org/pdf/2404.08189.pdf)** (2024-12-03)

*Patrice Béchard, Orlando Marquez Ayala*

  A common and fundamental limitation of Generative AI (GenAI) is its
propensity to hallucinate. While large language models (LLM) have taken the
world by storm, without eliminating or at least reducing hallucinations,
real-world GenAI systems may face challenges in user adoption. In the process
of deploying an enterprise application that produces workflows based on natural
language requirements, we devised a system leveraging Retrieval Augmented
Generation (RAG) to greatly improve the quality of the structured output that
represents such workflows. Thanks to our implementation of RAG, our proposed
system significantly reduces hallucinations in the output and improves the
generalization of our LLM in out-of-domain settings. In addition, we show that
using a small, well-trained retriever encoder can reduce the size of the
accompanying LLM, thereby making deployments of LLM-based systems less
resource-intensive.


---

**[143. [2406.09136] Chain of Preference Optimization: Improving Chain-of-Thought Reasoning
  in LLMs](https://arxiv.org/pdf/2406.09136.pdf)** (2024-11-01)

*Xuan Zhang, Chao Du, Tianyu Pang, Qian Liu, Wei Gao, Min Lin*

  The recent development of chain-of-thought (CoT) decoding has enabled large
language models (LLMs) to generate explicit logical reasoning paths for complex
problem-solving. However, research indicates that these paths are not always
deliberate and optimal. The tree-of-thought (ToT) method employs tree-searching
to extensively explore the reasoning space and find better reasoning paths that
CoT decoding might overlook. This deliberation, however, comes at the cost of
significantly increased inference complexity. In this work, we demonstrate that
fine-tuning LLMs leveraging the search tree constructed by ToT allows CoT to
achieve similar or better performance, thereby avoiding the substantial
inference burden. This is achieved through Chain of Preference Optimization
(CPO), where LLMs are fine-tuned to align each step of the CoT reasoning paths
with those of ToT using the inherent preference information in the tree-search
process. Extensive experimental results show that CPO significantly improves
LLM performance in solving a variety of complex problems, including question
answering, fact verification, and arithmetic reasoning, demonstrating its
effectiveness. Our code is available at https://github.com/sail-sg/CPO.


---

**[144. [2409.20550] LLM Hallucinations in Practical Code Generation: Phenomena, Mechanism,
  and Mitigation](https://arxiv.org/pdf/2409.20550.pdf)** (2025-01-20)

*Ziyao Zhang, Yanlin Wang, Chong Wang, Jiachi Chen, Zibin Zheng*

  Code generation aims to automatically generate code from input requirements,
significantly enhancing development efficiency. Recent large language models
(LLMs) based approaches have shown promising results and revolutionized code
generation task. Despite the promising performance, LLMs often generate
contents with hallucinations, especially for the code generation scenario
requiring the handling of complex contextual dependencies in practical
development process. Although previous study has analyzed hallucinations in
LLM-powered code generation, the study is limited to standalone function
generation. In this paper, we conduct an empirical study to study the
phenomena, mechanism, and mitigation of LLM hallucinations within more
practical and complex development contexts in repository-level generation
scenario. First, we manually examine the code generation results from six
mainstream LLMs to establish a hallucination taxonomy of LLM-generated code.
Next, we elaborate on the phenomenon of hallucinations, analyze their
distribution across different models. We then analyze causes of hallucinations
and identify four potential factors contributing to hallucinations. Finally, we
propose an RAG-based mitigation method, which demonstrates consistent
effectiveness in all studied LLMs. The replication package including code,
data, and experimental results is available at
https://github.com/DeepSoftwareAnalytics/LLMCodingHallucination


---

**[145. [2502.08666] Hallucination, Monofacts, and Miscalibration: An Empirical Investigation](https://arxiv.org/pdf/2502.08666.pdf)** (2025-02-14)

*Muqing Miao, Michael Kearns*

  Recent theoretical work by [Kalai and Vempala 2024] proves that a particular
notion of hallucination rate in LLMs must be lower bounded by the training data
monofact rate (related to the classical Good-Turing missing mass estimator)
minus model miscalibration. Through systematic experiments with n-gram models
and in-context learning with LLMs, we empirically investigate and validate this
theory by examining how different underlying data distributions affect the
monofact rate and a model's tendency to hallucinate. We then vary model
miscalibration through controlled upweighting of training samples while holding
monofact rates constant, allowing us to isolate miscalibration's reduction
effect on hallucination. These findings suggest that both the distribution of
fact frequencies in training data and the calibration-hallucination trade-off
are inherent to probabilistic language generation. Our results also suggest
that current practices of aggressive deduplication in training data may need to
be reconsidered, as selective duplication could serve as a principled mechanism
for reducing hallucination.


---

**[146. [2307.16139] User-Controlled Knowledge Fusion in Large Language Models: Balancing
  Creativity and Hallucination](https://arxiv.org/pdf/2307.16139.pdf)** (2023-08-01)

*Chen Zhang*

  In modern dialogue systems, the use of Large Language Models (LLMs) has grown
exponentially due to their capacity to generate diverse, relevant, and creative
responses. Despite their strengths, striking a balance between the LLMs'
creativity and their faithfulness to external knowledge remains a key
challenge. This paper presents an innovative user-controllable mechanism that
modulates the balance between an LLM's imaginative capabilities and its
adherence to factual information. Our approach incorporates a numerical tag
during the fine-tuning phase of the LLM's training, representing the degree of
faithfulness to the reference knowledge in the generated responses. This degree
is computed through an automated process that measures lexical overlap using
ROUGE scores, semantic similarity using Sentence-BERT embeddings, and an LLM's
self-evaluation score. During model inference, users can manipulate this
numerical tag, thus controlling the degree of the LLM's reliance on external
knowledge. We conduct extensive experiments across various scenarios,
demonstrating the adaptability of our method and its efficacy in ensuring the
quality and accuracy of the LLM's responses. The results highlight the
potential of our approach to enhance the versatility of LLMs while maintaining
a balance between creativity and hallucination.


---

**[147. [2502.13622] REFIND at SemEval-2025 Task 3: Retrieval-Augmented Factuality
  Hallucination Detection in Large Language Models](https://arxiv.org/pdf/2502.13622.pdf)** (2025-04-09)

*DongGeon Lee, Hwanjo Yu*

  Hallucinations in large language model (LLM) outputs severely limit their
reliability in knowledge-intensive tasks such as question answering. To address
this challenge, we introduce REFIND (Retrieval-augmented Factuality
hallucINation Detection), a novel framework that detects hallucinated spans
within LLM outputs by directly leveraging retrieved documents. As part of the
REFIND, we propose the Context Sensitivity Ratio (CSR), a novel metric that
quantifies the sensitivity of LLM outputs to retrieved evidence. This
innovative approach enables REFIND to efficiently and accurately detect
hallucinations, setting it apart from existing methods. In the evaluation,
REFIND demonstrated robustness across nine languages, including low-resource
settings, and significantly outperformed baseline models, achieving superior
IoU scores in identifying hallucinated spans. This work highlights the
effectiveness of quantifying context sensitivity for hallucination detection,
thereby paving the way for more reliable and trustworthy LLM applications
across diverse languages. Our code is available at
https://github.com/oneonlee/REFIND.


---

**[148. [2312.00575] Instruction-tuning Aligns LLMs to the Human Brain](https://arxiv.org/pdf/2312.00575.pdf)** (2024-08-12)

*Khai Loong Aw, Syrielle Montariol, Badr AlKhamissi, Martin Schrimpf, Antoine Bosselut*

  Instruction-tuning is a widely adopted finetuning method that enables large
language models (LLMs) to generate output that more closely resembles human
responses. However, no studies have shown that instruction-tuning actually
teaches LLMs to process language in a similar manner as humans. We investigate
the effect of instruction-tuning on aligning LLM and human language processing
mechanisms in two ways: (1) brain alignment, the similarity of LLM internal
representations to neural activity in the human language system, and (2)
behavioral alignment, the similarity of LLM and human behavior on a reading
task. We assess 25 vanilla and instruction-tuned LLMs on three datasets
involving humans reading naturalistic stories and sentences, and find that
instruction-tuning generally enhances brain alignment (~6%), but has no similar
effect on behavioral alignment. To identify factors underlying this improvement
in brain alignment, we compute correlations between brain alignment and various
LLM properties, such as model size, problem-solving, and world knowledge
understanding. Notably, we find a strong positive correlation between brain
alignment and model size (r = 0.95), as well as performance on tasks requiring
world knowledge (r = 0.81). Our results demonstrate that instruction-tuning
LLMs improves both world knowledge representations and brain alignment,
suggesting that the mechanisms that encode world knowledge in LLMs also improve
representational alignment to the human brain.


---

**[149. [2409.10011] HALO: Hallucination Analysis and Learning Optimization to Empower LLMs
  with Retrieval-Augmented Context for Guided Clinical Decision Making](https://arxiv.org/pdf/2409.10011.pdf)** (2024-09-20)

*Sumera Anjum, Hanzhi Zhang, Wenjun Zhou, Eun Jin Paek, Xiaopeng Zhao, Yunhe Feng*

  Large language models (LLMs) have significantly advanced natural language
processing tasks, yet they are susceptible to generating inaccurate or
unreliable responses, a phenomenon known as hallucination. In critical domains
such as health and medicine, these hallucinations can pose serious risks. This
paper introduces HALO, a novel framework designed to enhance the accuracy and
reliability of medical question-answering (QA) systems by focusing on the
detection and mitigation of hallucinations. Our approach generates multiple
variations of a given query using LLMs and retrieves relevant information from
external open knowledge bases to enrich the context. We utilize maximum
marginal relevance scoring to prioritize the retrieved context, which is then
provided to LLMs for answer generation, thereby reducing the risk of
hallucinations. The integration of LangChain further streamlines this process,
resulting in a notable and robust increase in the accuracy of both open-source
and commercial LLMs, such as Llama-3.1 (from 44% to 65%) and ChatGPT (from 56%
to 70%). This framework underscores the critical importance of addressing
hallucinations in medical QA systems, ultimately improving clinical
decision-making and patient care. The open-source HALO is available at:
https://github.com/ResponsibleAILab/HALO.


---

**[150. [2309.02654] Zero-Resource Hallucination Prevention for Large Language Models](https://arxiv.org/pdf/2309.02654.pdf)** (2023-10-10)

*Junyu Luo, Cao Xiao, Fenglong Ma*

  The prevalent use of large language models (LLMs) in various domains has
drawn attention to the issue of "hallucination," which refers to instances
where LLMs generate factually inaccurate or ungrounded information. Existing
techniques for hallucination detection in language assistants rely on intricate
fuzzy, specific free-language-based chain of thought (CoT) techniques or
parameter-based methods that suffer from interpretability issues. Additionally,
the methods that identify hallucinations post-generation could not prevent
their occurrence and suffer from inconsistent performance due to the influence
of the instruction format and model style. In this paper, we introduce a novel
pre-detection self-evaluation technique, referred to as SELF-FAMILIARITY, which
focuses on evaluating the model's familiarity with the concepts present in the
input instruction and withholding the generation of response in case of
unfamiliar concepts. This approach emulates the human ability to refrain from
responding to unfamiliar topics, thus reducing hallucinations. We validate
SELF-FAMILIARITY across four different large language models, demonstrating
consistently superior performance compared to existing techniques. Our findings
propose a significant shift towards preemptive strategies for hallucination
mitigation in LLM assistants, promising improvements in reliability,
applicability, and interpretability.


---

**[151. [2410.18860] DeCoRe: Decoding by Contrasting Retrieval Heads to Mitigate
  Hallucinations](https://arxiv.org/pdf/2410.18860.pdf)** (2024-10-25)

*Aryo Pradipta Gema, Chen Jin, Ahmed Abdulaal, Tom Diethe, Philip Teare, Beatrice Alex, Pasquale Minervini, Amrutha Saseendran*

  Large Language Models (LLMs) often hallucinate, producing unfaithful or
factually incorrect outputs by misrepresenting the provided context or
incorrectly recalling internal knowledge. Recent studies have identified
specific attention heads within the Transformer architecture, known as
retrieval heads, responsible for extracting relevant contextual information. We
hypothesise that masking these retrieval heads can induce hallucinations and
that contrasting the outputs of the base LLM and the masked LLM can reduce
hallucinations. To this end, we propose Decoding by Contrasting Retrieval Heads
(DeCoRe), a novel training-free decoding strategy that amplifies information
found in the context and model parameters. DeCoRe mitigates potentially
hallucinated responses by dynamically contrasting the outputs of the base LLM
and the masked LLM, using conditional entropy as a guide. Our extensive
experiments confirm that DeCoRe significantly improves performance on tasks
requiring high contextual faithfulness, such as summarisation (XSum by 18.6%),
instruction following (MemoTrap by 10.9%), and open-book question answering
(NQ-Open by 2.4% and NQ-Swap by 5.5%).


---

**[152. [2312.15576] Reducing LLM Hallucinations using Epistemic Neural Networks](https://arxiv.org/pdf/2312.15576.pdf)** (2023-12-27)

*Shreyas Verma, Kien Tran, Yusuf Ali, Guangyu Min*

  Reducing and detecting hallucinations in large language models is an open
research problem. In this project, we attempt to leverage recent advances in
the field of uncertainty estimation to reduce hallucinations in frozen large
language models. Epistemic neural networks have recently been proposed to
improve output joint distributions for large pre-trained models. ENNs are small
networks attached to large, frozen models to improve the model's joint
distributions and uncertainty estimates. In this work, we train an epistemic
neural network on top of the Llama-2 7B model combined with a contrastive
decoding feature enhancement technique. We are the first to train an ENN for
the next token prediction task and explore the efficacy of this method in
reducing hallucinations on the TruthfulQA dataset. In essence, we provide a
method that leverages a pre-trained model's latent embeddings to reduce
hallucinations.


---

**[153. [2408.15037] Evidence-Enhanced Triplet Generation Framework for Hallucination
  Alleviation in Generative Question Answering](https://arxiv.org/pdf/2408.15037.pdf)** (2024-08-28)

*Haowei Du, Huishuai Zhang, Dongyan Zhao*

  To address the hallucination in generative question answering (GQA) where the
answer can not be derived from the document, we propose a novel
evidence-enhanced triplet generation framework, EATQA, encouraging the model to
predict all the combinations of (Question, Evidence, Answer) triplet by
flipping the source pair and the target label to understand their logical
relationships, i.e., predict Answer(A), Question(Q), and Evidence(E) given a
QE, EA, and QA pairs, respectively. Furthermore, we bridge the distribution gap
to distill the knowledge from evidence in inference stage. Our framework
ensures the model to learn the logical relation between query, evidence and
answer, which simultaneously improves the evidence generation and query
answering. In this paper, we apply EATQA to LLama and it outperforms other
LLMs-based methods and hallucination mitigation approaches on two challenging
GQA benchmarks. Further analysis shows that our method not only keeps prior
knowledge within LLM, but also mitigates hallucination and generates faithful
answers.


---

**[154. [2405.08619] ALMol: Aligned Language-Molecule Translation LLMs through Offline
  Preference Contrastive Optimisation](https://arxiv.org/pdf/2405.08619.pdf)** (2024-07-16)

*Dimitris Gkoumas*

  The field of chemistry and Artificial Intelligence (AI) intersection is an
area of active research that aims to accelerate scientific discovery. The
integration of large language models (LLMs) with scientific modalities has
shown significant promise in this endeavour. However, challenges persist in
effectively addressing training efficacy and the out-of-distribution problem,
particularly as existing approaches rely on larger models and datasets. In this
context, we focus on machine language-molecule translation and deploy a novel
training approach called contrastive preference optimisation, which avoids
generating translations that are merely adequate but not perfect. To ensure
generalisability and mitigate memorisation effects, we conduct experiments
using only 10% of the data. Our results demonstrate that our models achieve up
to a 32% improvement compared to counterpart models. Finally, we introduce a
fine-grained, domain-agnostic evaluation method to assess hallucination in LLMs
and promote responsible use.


---

**[155. [2410.12787] The Curse of Multi-Modalities: Evaluating Hallucinations of Large
  Multimodal Models across Language, Visual, and Audio](https://arxiv.org/pdf/2410.12787.pdf)** (2024-10-17)

*Sicong Leng, Yun Xing, Zesen Cheng, Yang Zhou, Hang Zhang, Xin Li, Deli Zhao, Shijian Lu, Chunyan Miao, Lidong Bing*

  Recent advancements in large multimodal models (LMMs) have significantly
enhanced performance across diverse tasks, with ongoing efforts to further
integrate additional modalities such as video and audio. However, most existing
LMMs remain vulnerable to hallucinations, the discrepancy between the factual
multimodal input and the generated textual output, which has limited their
applicability in various real-world scenarios. This paper presents the first
systematic investigation of hallucinations in LMMs involving the three most
common modalities: language, visual, and audio. Our study reveals two key
contributors to hallucinations: overreliance on unimodal priors and spurious
inter-modality correlations. To address these challenges, we introduce the
benchmark The Curse of Multi-Modalities (CMM), which comprehensively evaluates
hallucinations in LMMs, providing a detailed analysis of their underlying
issues. Our findings highlight key vulnerabilities, including imbalances in
modality integration and biases from training data, underscoring the need for
balanced cross-modal learning and enhanced hallucination mitigation strategies.
Based on our observations and findings, we suggest potential research
directions that could enhance the reliability of LMMs.


---

**[156. [2311.15548] Deficiency of Large Language Models in Finance: An Empirical Examination
  of Hallucination](https://arxiv.org/pdf/2311.15548.pdf)** (2023-11-28)

*Haoqiang Kang, Xiao-Yang Liu*

  The hallucination issue is recognized as a fundamental deficiency of large
language models (LLMs), especially when applied to fields such as finance,
education, and law. Despite the growing concerns, there has been a lack of
empirical investigation. In this paper, we provide an empirical examination of
LLMs' hallucination behaviors in financial tasks. First, we empirically
investigate LLM model's ability of explaining financial concepts and
terminologies. Second, we assess LLM models' capacity of querying historical
stock prices. Third, to alleviate the hallucination issue, we evaluate the
efficacy of four practical methods, including few-shot learning, Decoding by
Contrasting Layers (DoLa), the Retrieval Augmentation Generation (RAG) method
and the prompt-based tool learning method for a function to generate a query
command. Finally, our major finding is that off-the-shelf LLMs experience
serious hallucination behaviors in financial tasks. Therefore, there is an
urgent need to call for research efforts in mitigating LLMs' hallucination.


---

**[157. [2503.10602] TruthPrInt: Mitigating LVLM Object Hallucination Via Latent
  Truthful-Guided Pre-Intervention](https://arxiv.org/pdf/2503.10602.pdf)** (2025-03-24)

*Jinhao Duan, Fei Kong, Hao Cheng, James Diffenderfer, Bhavya Kailkhura, Lichao Sun, Xiaofeng Zhu, Xiaoshuang Shi, Kaidi Xu*

  Object Hallucination (OH) has been acknowledged as one of the major
trustworthy challenges in Large Vision-Language Models (LVLMs). Recent
advancements in Large Language Models (LLMs) indicate that internal states,
such as hidden states, encode the "overall truthfulness" of generated
responses. However, it remains under-explored how internal states in LVLMs
function and whether they could serve as "per-token" hallucination indicators,
which is essential for mitigating OH. In this paper, we first conduct an
in-depth exploration of LVLM internal states in relation to OH issues and
discover that (1) LVLM internal states are high-specificity per-token
indicators of hallucination behaviors. Moreover, (2) different LVLMs encode
universal patterns of hallucinations in common latent subspaces, indicating
that there exist "generic truthful directions" shared by various LVLMs. Based
on these discoveries, we propose Truthful-Guided Pre-Intervention (TruthPrInt)
that first learns the truthful direction of LVLM decoding and then applies
truthful-guided inference-time intervention during LVLM decoding. We further
propose ComnHallu to enhance both cross-LVLM and cross-data hallucination
detection transferability by constructing and aligning hallucination latent
subspaces. We evaluate TruthPrInt in extensive experimental settings, including
in-domain and out-of-domain scenarios, over popular LVLMs and OH benchmarks.
Experimental results indicate that TruthPrInt significantly outperforms
state-of-the-art methods. Codes will be available at
https://github.com/jinhaoduan/TruthPrInt.


---

**[158. [2409.17504] HaloScope: Harnessing Unlabeled LLM Generations for Hallucination
  Detection](https://arxiv.org/pdf/2409.17504.pdf)** (2024-09-27)

*Xuefeng Du, Chaowei Xiao, Yixuan Li*

  The surge in applications of large language models (LLMs) has prompted
concerns about the generation of misleading or fabricated information, known as
hallucinations. Therefore, detecting hallucinations has become critical to
maintaining trust in LLM-generated content. A primary challenge in learning a
truthfulness classifier is the lack of a large amount of labeled truthful and
hallucinated data. To address the challenge, we introduce HaloScope, a novel
learning framework that leverages the unlabeled LLM generations in the wild for
hallucination detection. Such unlabeled data arises freely upon deploying LLMs
in the open world, and consists of both truthful and hallucinated information.
To harness the unlabeled data, we present an automated membership estimation
score for distinguishing between truthful and untruthful generations within
unlabeled mixture data, thereby enabling the training of a binary truthfulness
classifier on top. Importantly, our framework does not require extra data
collection and human annotations, offering strong flexibility and practicality
for real-world applications. Extensive experiments show that HaloScope can
achieve superior hallucination detection performance, outperforming the
competitive rivals by a significant margin. Code is available at
https://github.com/deeplearningwisc/haloscope.


---

**[159. [2407.21424] Cost-Effective Hallucination Detection for LLMs](https://arxiv.org/pdf/2407.21424.pdf)** (2024-08-12)

*Simon Valentin, Jinmiao Fu, Gianluca Detommaso, Shaoyuan Xu, Giovanni Zappella, Bryan Wang*

  Large language models (LLMs) can be prone to hallucinations - generating
unreliable outputs that are unfaithful to their inputs, external facts or
internally inconsistent. In this work, we address several challenges for
post-hoc hallucination detection in production settings. Our pipeline for
hallucination detection entails: first, producing a confidence score
representing the likelihood that a generated answer is a hallucination; second,
calibrating the score conditional on attributes of the inputs and candidate
response; finally, performing detection by thresholding the calibrated score.
We benchmark a variety of state-of-the-art scoring methods on different
datasets, encompassing question answering, fact checking, and summarization
tasks. We employ diverse LLMs to ensure a comprehensive assessment of
performance. We show that calibrating individual scoring methods is critical
for ensuring risk-aware downstream decision making. Based on findings that no
individual score performs best in all situations, we propose a multi-scoring
framework, which combines different scores and achieves top performance across
all datasets. We further introduce cost-effective multi-scoring, which can
match or even outperform more expensive detection methods, while significantly
reducing computational overhead.


---

**[160. [2410.17021] SG-FSM: A Self-Guiding Zero-Shot Prompting Paradigm for Multi-Hop
  Question Answering Based on Finite State Machine](https://arxiv.org/pdf/2410.17021.pdf)** (2024-10-23)

*Xiaochen Wang, Junqing He, Liang Chen, Reza Haf Zhe Yang, Yiru Wang, Xiangdi Meng, Kunhao Pan, Zhifang Sui*

  Large Language Models with chain-of-thought prompting, such as OpenAI-o1,
have shown impressive capabilities in natural language inference tasks.
However, Multi-hop Question Answering (MHQA) remains challenging for many
existing models due to issues like hallucination, error propagation, and
limited context length. To address these challenges and enhance LLMs'
performance on MHQA, we propose the Self-Guiding prompting Finite State Machine
(SG-FSM), designed to strengthen multi-hop reasoning abilities. Unlike
traditional chain-of-thought methods, SG-FSM tackles MHQA by iteratively
breaking down complex questions into sub-questions, correcting itself to
improve accuracy. It processes one sub-question at a time, dynamically deciding
the next step based on the current context and results, functioning much like
an automaton. Experiments across various benchmarks demonstrate the
effectiveness of our approach, outperforming strong baselines on challenging
datasets such as Musique. SG-FSM reduces hallucination, enabling recovery of
the correct final answer despite intermediate errors. It also improves
adherence to specified output formats, simplifying evaluation significantly.


---

**[161. [2410.15359] A Survey of Hallucination in Large Visual Language Models](https://arxiv.org/pdf/2410.15359.pdf)** (2024-10-22)

*Wei Lan, Wenyi Chen, Qingfeng Chen, Shirui Pan, Huiyu Zhou, Yi Pan*

  The Large Visual Language Models (LVLMs) enhances user interaction and
enriches user experience by integrating visual modality on the basis of the
Large Language Models (LLMs). It has demonstrated their powerful information
processing and generation capabilities. However, the existence of
hallucinations has limited the potential and practical effectiveness of LVLM in
various fields. Although lots of work has been devoted to the issue of
hallucination mitigation and correction, there are few reviews to summary this
issue. In this survey, we first introduce the background of LVLMs and
hallucinations. Then, the structure of LVLMs and main causes of hallucination
generation are introduced. Further, we summary recent works on hallucination
correction and mitigation. In addition, the available hallucination evaluation
benchmarks for LVLMs are presented from judgmental and generative perspectives.
Finally, we suggest some future research directions to enhance the
dependability and utility of LVLMs.


---

**[162. [2503.12908] HICD: Hallucination-Inducing via Attention Dispersion for Contrastive
  Decoding to Mitigate Hallucinations in Large Language Models](https://arxiv.org/pdf/2503.12908.pdf)** (2025-03-18)

*Xinyan Jiang, Hang Ye, Yongxin Zhu, Xiaoying Zheng, Zikang Chen, Jun Gong*

  Large Language Models (LLMs) often generate hallucinations, producing outputs
that are contextually inaccurate or factually incorrect. We introduce HICD, a
novel method designed to induce hallucinations for contrastive decoding to
mitigate hallucinations. Unlike existing contrastive decoding methods, HICD
selects attention heads crucial to the model's prediction as inducing heads,
then induces hallucinations by dispersing attention of these inducing heads and
compares the hallucinated outputs with the original outputs to obtain the final
result. Our approach significantly improves performance on tasks requiring
contextual faithfulness, such as context completion, reading comprehension, and
question answering. It also improves factuality in tasks requiring accurate
knowledge recall. We demonstrate that our inducing heads selection and
attention dispersion method leads to more "contrast-effective" hallucinations
for contrastive decoding, outperforming other hallucination-inducing methods.
Our findings provide a promising strategy for reducing hallucinations by
inducing hallucinations in a controlled manner, enhancing the performance of
LLMs in a wide range of tasks.


---

**[163. [2502.01056] Mitigating Hallucinations in Large Vision-Language Models with Internal
  Fact-based Contrastive Decoding](https://arxiv.org/pdf/2502.01056.pdf)** (2025-02-04)

*Chao Wang, Xuancheng Zhou, Weiwei Fu, Yang Zhou*

  Large Visual Language Models (LVLMs) integrate visual and linguistic
modalities, exhibiting exceptional performance across various multimodal tasks.
Nevertheless, LVLMs remain vulnerable to the issue of object hallucinations.
Previous efforts to mitigate this issue focus on supervised fine-tuning (SFT)
or incorporating external knowledge, both of which entail significant costs
related to training and the acquisition of external data. To address these
challenges, we propose a novel model-agnostic approach termed Internal
Fact-based Contrastive Decoding (IFCD), designed to mitigate and suppress
hallucinations during the inference process of LVLMs by exploiting the LVLMs'
own hallucinations. IFCD is grounded in experimental observations that
alterations to the LVLMs' internal representations tend to amplify
hallucinations caused by language bias. By contrasting disturbed distribution,
IFCD calibrates the LVLMs' output and effectively removes the hallucinatory
logits from the final predictions. Experimental results validate that IFCD
significantly alleviates both object-level and attribute-level hallucinations
while achieving an average 9% accuracy improvement on POPE and 8% accuracy
improvement on MME object hallucinations subset compared with direct decoding,
respectively.


---

**[164. [2406.17642] Banishing LLM Hallucinations Requires Rethinking Generalization](https://arxiv.org/pdf/2406.17642.pdf)** (2024-06-26)

*Johnny Li, Saksham Consul, Eda Zhou, James Wong, Naila Farooqui, Yuxin Ye, Nithyashree Manohar, Zhuxiaona Wei, Tian Wu, Ben Echols, Sharon Zhou, Gregory Diamos*

  Despite their powerful chat, coding, and reasoning abilities, Large Language
Models (LLMs) frequently hallucinate. Conventional wisdom suggests that
hallucinations are a consequence of a balance between creativity and
factuality, which can be mitigated, but not eliminated, by grounding the LLM in
external knowledge sources. Through extensive systematic experiments, we show
that these traditional approaches fail to explain why LLMs hallucinate in
practice. Specifically, we show that LLMs augmented with a massive Mixture of
Memory Experts (MoME) can easily memorize large datasets of random numbers. We
corroborate these experimental findings with a theoretical construction showing
that simple neural networks trained to predict the next token hallucinate when
the training loss is above a threshold as it usually does in practice when
training on internet scale data. We interpret our findings by comparing against
traditional retrieval methods for mitigating hallucinations. We use our
findings to design a first generation model for removing hallucinations --
Lamini-1 -- that stores facts in a massive mixture of millions of memory
experts that are retrieved dynamically.


---

**[165. [2311.05232] A Survey on Hallucination in Large Language Models: Principles,
  Taxonomy, Challenges, and Open Questions](https://arxiv.org/pdf/2311.05232.pdf)** (2024-11-20)

*Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, Ting Liu*

  The emergence of large language models (LLMs) has marked a significant
breakthrough in natural language processing (NLP), fueling a paradigm shift in
information acquisition. Nevertheless, LLMs are prone to hallucination,
generating plausible yet nonfactual content. This phenomenon raises significant
concerns over the reliability of LLMs in real-world information retrieval (IR)
systems and has attracted intensive research to detect and mitigate such
hallucinations. Given the open-ended general-purpose attributes inherent to
LLMs, LLM hallucinations present distinct challenges that diverge from prior
task-specific models. This divergence highlights the urgency for a nuanced
understanding and comprehensive overview of recent advances in LLM
hallucinations. In this survey, we begin with an innovative taxonomy of
hallucination in the era of LLM and then delve into the factors contributing to
hallucinations. Subsequently, we present a thorough overview of hallucination
detection methods and benchmarks. Our discussion then transfers to
representative methodologies for mitigating LLM hallucinations. Additionally,
we delve into the current limitations faced by retrieval-augmented LLMs in
combating hallucinations, offering insights for developing more robust IR
systems. Finally, we highlight the promising research directions on LLM
hallucinations, including hallucination in large vision-language models and
understanding of knowledge boundaries in LLM hallucinations.


---

**[166. [2409.16597] EventHallusion: Diagnosing Event Hallucinations in Video LLMs](https://arxiv.org/pdf/2409.16597.pdf)** (2025-03-25)

*Jiacheng Zhang, Yang Jiao, Shaoxiang Chen, Na Zhao, Zhiyu Tan, Hao Li, Jingjing Chen*

  Recently, Multimodal Large Language Models (MLLMs) have made significant
progress in the video comprehension field. Despite remarkable content reasoning
and instruction following capabilities they demonstrated, the hallucination
problem of these VideoLLMs is less explored compared with its counterpart in
the image domain. To mitigate this gap, we propose EventHallusion, a novel
benchmark that focuses on assessing the VideoLLMs' hallucination toward event,
the crux of video analysis. From a hallucination attribution perspective, our
EventHallusion benchmark is curated to assess a VideoLLM's susceptibility
toward language priors and vision-language biases. On the other hand, we also
propose a simple yet effective method, called Temporal Contrastive Decoding
(TCD), to tackle the hallucination problems of VideoLLMs. The proposed TCD
method rectifies the model's bias toward its priors during the decoding stage
by comparing the original video with a modified version, in which temporal cues
are disrupted. Through comprehensive evaluation of eight open-source and two
closed-source VideoLLMs on the proposed EventHallusion benchmark, we observe
that the open-source models suffer significantly from hallucination problems,
whereas the closed-source ones perform markedly better. By further equipping
open-source VideoLLMs with the proposed TCD approach, evident performance
improvements are achieved across most metrics in the EventHallusion benchmark.
Our codes and benchmark data are available at
https://github.com/Stevetich/EventHallusion.


---

**[167. [2410.12130] Iter-AHMCL: Alleviate Hallucination for Large Language Model via
  Iterative Model-level Contrastive Learning](https://arxiv.org/pdf/2410.12130.pdf)** (2024-10-17)

*Huiwen Wu, Xiaohan Li, Xiaogang Xu, Jiafei Wu, Deyi Zhang, Zhe Liu*

  The development of Large Language Models (LLMs) has significantly advanced
various AI applications in commercial and scientific research fields, such as
scientific literature summarization, writing assistance, and knowledge graph
construction. However, a significant challenge is the high risk of
hallucination during LLM inference, which can lead to security concerns like
factual inaccuracies, inconsistent information, and fabricated content. To
tackle this issue, it is essential to develop effective methods for reducing
hallucination while maintaining the original capabilities of the LLM. This
paper introduces a novel approach called Iterative Model-level Contrastive
Learning (Iter-AHMCL) to address hallucination. This method modifies the
representation layers of pre-trained LLMs by using contrastive `positive' and
`negative' models, trained on data with and without hallucinations. By
leveraging the differences between these two models, we create a more
straightforward pathway to eliminate hallucinations, and the iterative nature
of contrastive learning further enhances performance. Experimental validation
on four pre-trained foundation LLMs (LLaMA2, Alpaca, LLaMA3, and Qwen)
finetuning with a specially designed dataset shows that our approach achieves
an average improvement of 10.1 points on the TruthfulQA benchmark.
Comprehensive experiments demonstrate the effectiveness of Iter-AHMCL in
reducing hallucination while maintaining the general capabilities of LLMs.


---

**[168. [2404.08509] Efficient Interactive LLM Serving with Proxy Model-based Sequence Length
  Prediction](https://arxiv.org/pdf/2404.08509.pdf)** (2024-11-26)

*Haoran Qiu, Weichao Mao, Archit Patke, Shengkun Cui, Saurabh Jha, Chen Wang, Hubertus Franke, Zbigniew T. Kalbarczyk, Tamer Başar, Ravishankar K. Iyer*

  Large language models (LLMs) have been driving a new wave of interactive AI
applications across numerous domains. However, efficiently serving LLM
inference requests is challenging due to their unpredictable execution times
originating from the autoregressive nature of generative models. Existing LLM
serving systems exploit first-come-first-serve (FCFS) scheduling, suffering
from head-of-line blocking issues. To address the non-deterministic nature of
LLMs and enable efficient interactive LLM serving, we present a speculative
shortest-job-first (SSJF) scheduler that uses a light proxy model to predict
LLM output sequence lengths. Our open-source SSJF implementation does not
require changes to memory management or batching strategies. Evaluations on
real-world datasets and production workload traces show that SSJF reduces
average job completion times by 30.5-39.6% and increases throughput by 2.2-3.6x
compared to FCFS schedulers, across no batching, dynamic batching, and
continuous batching settings.


---

**[169. [2310.06827] Teaching Language Models to Hallucinate Less with Synthetic Tasks](https://arxiv.org/pdf/2310.06827.pdf)** (2023-11-08)

*Erik Jones, Hamid Palangi, Clarisse Simões, Varun Chandrasekaran, Subhabrata Mukherjee, Arindam Mitra, Ahmed Awadallah, Ece Kamar*

  Large language models (LLMs) frequently hallucinate on abstractive
summarization tasks such as document-based question-answering, meeting
summarization, and clinical report generation, even though all necessary
information is included in context. However, optimizing LLMs to hallucinate
less on these tasks is challenging, as hallucination is hard to efficiently
evaluate at each optimization step. In this work, we show that reducing
hallucination on a synthetic task can also reduce hallucination on real-world
downstream tasks. Our method, SynTra, first designs a synthetic task where
hallucinations are easy to elicit and measure. It next optimizes the LLM's
system message via prefix-tuning on the synthetic task, and finally transfers
the system message to realistic, hard-to-optimize tasks. Across three realistic
abstractive summarization tasks, SynTra reduces hallucination for two
13B-parameter LLMs using only a synthetic retrieval task for supervision. We
also find that optimizing the system message rather than the model weights can
be critical; fine-tuning the entire model on the synthetic task can
counterintuitively increase hallucination. Overall, SynTra demonstrates that
the extra flexibility of working with synthetic data can help mitigate
undesired behaviors in practice.


---

**[170. [2406.11277] Small Agent Can Also Rock! Empowering Small Language Models as
  Hallucination Detector](https://arxiv.org/pdf/2406.11277.pdf)** (2024-06-18)

*Xiaoxue Cheng, Junyi Li, Wayne Xin Zhao, Hongzhi Zhang, Fuzheng Zhang, Di Zhang, Kun Gai, Ji-Rong Wen*

  Hallucination detection is a challenging task for large language models
(LLMs), and existing studies heavily rely on powerful closed-source LLMs such
as GPT-4. In this paper, we propose an autonomous LLM-based agent framework,
called HaluAgent, which enables relatively smaller LLMs (e.g. Baichuan2-Chat
7B) to actively select suitable tools for detecting multiple hallucination
types such as text, code, and mathematical expression. In HaluAgent, we
integrate the LLM, multi-functional toolbox, and design a fine-grained
three-stage detection framework along with memory mechanism. To facilitate the
effectiveness of HaluAgent, we leverage existing Chinese and English datasets
to synthesize detection trajectories for fine-tuning, which endows HaluAgent
with the capability for bilingual hallucination detection. Extensive
experiments demonstrate that only using 2K samples for tuning LLMs, HaluAgent
can perform hallucination detection on various types of tasks and datasets,
achieving performance comparable to or even higher than GPT-4 without tool
enhancements on both in-domain and out-of-domain datasets. We release our
dataset and code at https://github.com/RUCAIBox/HaluAgent.


---

**[171. [2503.04381] TRACT: Regression-Aware Fine-tuning Meets Chain-of-Thought Reasoning for
  LLM-as-a-Judge](https://arxiv.org/pdf/2503.04381.pdf)** (2025-03-07)

*Cheng-Han Chiang, Hung-yi Lee, Michal Lukasik*

  The LLM-as-a-judge paradigm uses large language models (LLMs) for automated
text evaluation, where a numerical assessment is assigned by an LLM to the
input text following scoring rubrics. Existing methods for LLM-as-a-judge use
cross-entropy (CE) loss for fine-tuning, which neglects the numeric nature of
score prediction. Recent work addresses numerical prediction limitations of LLM
fine-tuning through regression-aware fine-tuning, which, however, does not
consider chain-of-thought (CoT) reasoning for score prediction. In this paper,
we introduce TRACT (Two-stage Regression-Aware fine-tuning with CoT), a method
combining CoT reasoning with regression-aware training. TRACT consists of two
stages: first, seed LLM is fine-tuned to generate CoTs, which serve as
supervision for the second stage fine-tuning. The training objective of TRACT
combines the CE loss for learning the CoT reasoning capabilities, and the
regression-aware loss for the score prediction. Experiments across four
LLM-as-a-judge datasets and two LLMs show that TRACT significantly outperforms
existing methods. Extensive ablation studies validate the importance of each
component in TRACT.


---

**[172. [2504.02865] The Illusionist's Prompt: Exposing the Factual Vulnerabilities of Large
  Language Models with Linguistic Nuances](https://arxiv.org/pdf/2504.02865.pdf)** (2025-04-07)

*Yining Wang, Yuquan Wang, Xi Li, Mi Zhang, Geng Hong, Min Yang*

  As Large Language Models (LLMs) continue to advance, they are increasingly
relied upon as real-time sources of information by non-expert users. To ensure
the factuality of the information they provide, much research has focused on
mitigating hallucinations in LLM responses, but only in the context of formal
user queries, rather than maliciously crafted ones. In this study, we introduce
The Illusionist's Prompt, a novel hallucination attack that incorporates
linguistic nuances into adversarial queries, challenging the factual accuracy
of LLMs against five types of fact-enhancing strategies. Our attack
automatically generates highly transferrable illusory prompts to induce
internal factual errors, all while preserving user intent and semantics.
Extensive experiments confirm the effectiveness of our attack in compromising
black-box LLMs, including commercial APIs like GPT-4o and Gemini-2.0, even with
various defensive mechanisms.


---

**[173. [2306.09782] Full Parameter Fine-tuning for Large Language Models with Limited
  Resources](https://arxiv.org/pdf/2306.09782.pdf)** (2024-06-07)

*Kai Lv, Yuqing Yang, Tengxiao Liu, Qinghui Gao, Qipeng Guo, Xipeng Qiu*

  Large Language Models (LLMs) have revolutionized Natural Language Processing
(NLP) but demand massive GPU resources for training. Lowering the threshold for
LLMs training would encourage greater participation from researchers,
benefiting both academia and society. While existing approaches have focused on
parameter-efficient fine-tuning, which tunes or adds a small number of
parameters, few have addressed the challenge of tuning the full parameters of
LLMs with limited resources. In this work, we propose a new optimizer,
LOw-Memory Optimization (LOMO), which fuses the gradient computation and the
parameter update in one step to reduce memory usage. By integrating LOMO with
existing memory saving techniques, we reduce memory usage to 10.8% compared to
the standard approach (DeepSpeed solution). Consequently, our approach enables
the full parameter fine-tuning of a 65B model on a single machine with 8 RTX
3090, each with 24GB memory.Code and data are available at
https://github.com/OpenLMLab/LOMO.


---

**[174. [2312.01701] Mitigating Fine-Grained Hallucination by Fine-Tuning Large
  Vision-Language Models with Caption Rewrites](https://arxiv.org/pdf/2312.01701.pdf)** (2023-12-05)

*Lei Wang, Jiabang He, Shenshen Li, Ning Liu, Ee-Peng Lim*

  Large language models (LLMs) have shown remarkable performance in natural
language processing (NLP) tasks. To comprehend and execute diverse human
instructions over image data, instruction-tuned large vision-language models
(LVLMs) have been introduced. However, LVLMs may suffer from different types of
object hallucinations. Nevertheless, LVLMs are evaluated for coarse-grained
object hallucinations only (i.e., generated objects non-existent in the input
image). The fine-grained object attributes and behaviors non-existent in the
image may still be generated but not measured by the current evaluation
methods. In this paper, we thus focus on reducing fine-grained hallucinations
of LVLMs. We propose \textit{ReCaption}, a framework that consists of two
components: rewriting captions using ChatGPT and fine-tuning the
instruction-tuned LVLMs on the rewritten captions. We also propose a
fine-grained probing-based evaluation method named \textit{Fine-Grained Object
Hallucination Evaluation} (\textit{FGHE}). Our experiment results demonstrate
that ReCaption effectively reduces fine-grained object hallucination for
different LVLM options and improves their text generation quality. The code can
be found at https://github.com/Anonymousanoy/FOHE.


---

**[175. [2502.05911] GRAIT: Gradient-Driven Refusal-Aware Instruction Tuning for Effective
  Hallucination Mitigation](https://arxiv.org/pdf/2502.05911.pdf)** (2025-02-11)

*Runchuan Zhu, Zinco Jiang, Jiang Wu, Zhipeng Ma, Jiahe Song, Fengshuo Bai, Dahua Lin, Lijun Wu, Conghui He*

  Refusal-Aware Instruction Tuning (RAIT) aims to enhance Large Language Models
(LLMs) by improving their ability to refuse responses to questions beyond their
knowledge, thereby reducing hallucinations and improving reliability. Effective
RAIT must address two key challenges: firstly, effectively reject unknown
questions to minimize hallucinations; secondly, avoid over-refusal to ensure
questions that can be correctly answered are not rejected, thereby maintain the
helpfulness of LLM outputs. In this paper, we address the two challenges by
deriving insightful observations from the gradient-based perspective, and
proposing the Gradient-driven Refusal Aware Instruction Tuning Framework GRAIT:
(1) employs gradient-driven sample selection to effectively minimize
hallucinations and (2) introduces an adaptive weighting mechanism during
fine-tuning to reduce the risk of over-refusal, achieving the balance between
accurate refusals and maintaining useful responses. Experimental evaluations on
open-ended and multiple-choice question answering tasks demonstrate that GRAIT
significantly outperforms existing RAIT methods in the overall performance. The
source code and data will be available at https://github.com/opendatalab/GRAIT .


---

**[176. [2403.09972] Think Twice Before Trusting: Self-Detection for Large Language Models
  through Comprehensive Answer Reflection](https://arxiv.org/pdf/2403.09972.pdf)** (2024-09-30)

*Moxin Li, Wenjie Wang, Fuli Feng, Fengbin Zhu, Qifan Wang, Tat-Seng Chua*

  Self-detection for Large Language Models (LLMs) seeks to evaluate the
trustworthiness of the LLM's output by leveraging its own capabilities, thereby
alleviating the issue of output hallucination. However, existing self-detection
approaches only retrospectively evaluate answers generated by LLM, typically
leading to the over-trust in incorrectly generated answers. To tackle this
limitation, we propose a novel self-detection paradigm that considers the
comprehensive answer space beyond LLM-generated answers. It thoroughly compares
the trustworthiness of multiple candidate answers to mitigate the over-trust in
LLM-generated incorrect answers. Building upon this paradigm, we introduce a
two-step framework, which firstly instructs LLM to reflect and provide
justifications for each candidate answer, and then aggregates the
justifications for comprehensive target answer evaluation. This framework can
be seamlessly integrated with existing approaches for superior self-detection.
Extensive experiments on six datasets spanning three tasks demonstrate the
effectiveness of the proposed framework.


---

**[177. [2504.05946] InstructMPC: A Human-LLM-in-the-Loop Framework for Context-Aware Control](https://arxiv.org/pdf/2504.05946.pdf)** (2025-04-15)

*Ruixiang Wu, Jiahao Ai, Tongxin Li*

  Model Predictive Control (MPC) is a powerful control strategy widely utilized
in domains like energy management, building control, and autonomous systems.
However, its effectiveness in real-world settings is challenged by the need to
incorporate context-specific predictions and expert instructions, which
traditional MPC often neglects. We propose InstructMPC, a novel framework that
addresses this gap by integrating real-time human instructions through a Large
Language Model (LLM) to produce context-aware predictions for MPC. Our method
employs a Language-to-Distribution (L2D) module to translate contextual
information into predictive disturbance trajectories, which are then
incorporated into the MPC optimization. Unlike existing context-aware and
language-based MPC models, InstructMPC enables dynamic human-LLM interaction
and fine-tunes the L2D module in a closed loop with theoretical performance
guarantees, achieving a regret bound of $O(\sqrt{T\log T})$ for linear dynamics
when optimized via advanced fine-tuning methods such as Direct Preference
Optimization (DPO) using a tailored loss function.


---

**[178. [2501.06521] Fine-tuning Large Language Models for Improving Factuality in Legal
  Question Answering](https://arxiv.org/pdf/2501.06521.pdf)** (2025-01-14)

*Yinghao Hu, Leilei Gan, Wenyi Xiao, Kun Kuang, Fei Wu*

  Hallucination, or the generation of incorrect or fabricated information,
remains a critical challenge in large language models (LLMs), particularly in
high-stake domains such as legal question answering (QA). In order to mitigate
the hallucination rate in legal QA, we first introduce a benchmark called
LegalHalBench and three automatic metrics to evaluate the common hallucinations
when LLMs answer legal questions. We then propose a hallucination mitigation
method that integrates behavior cloning and a novel Hard Sample-aware Iterative
Direct Preference Optimization (HIPO). We conduct extensive real-data
experiments to validate the effectiveness of our approach. Our results
demonstrate remarkable improvements in various metrics, including the newly
proposed Non-Hallucinated Statute Rate, Statute Relevance Rate, Legal Claim
Truthfulness, as well as traditional metrics such as METEOR, BERTScore,
ROUGE-L, and win rates.


---

**[179. [2503.21098] Alleviating LLM-based Generative Retrieval Hallucination in Alipay
  Search](https://arxiv.org/pdf/2503.21098.pdf)** (2025-03-28)

*Yedan Shen, Kaixin Wu, Yuechen Ding, Jingyuan Wen, Hong Liu, Mingjie Zhong, Zhouhan Lin, Jia Xu, Linjian Mo*

  Generative retrieval (GR) has revolutionized document retrieval with the
advent of large language models (LLMs), and LLM-based GR is gradually being
adopted by the industry. Despite its remarkable advantages and potential,
LLM-based GR suffers from hallucination and generates documents that are
irrelevant to the query in some instances, severely challenging its credibility
in practical applications. We thereby propose an optimized GR framework
designed to alleviate retrieval hallucination, which integrates knowledge
distillation reasoning in model training and incorporate decision agent to
further improve retrieval precision. Specifically, we employ LLMs to assess and
reason GR retrieved query-document (q-d) pairs, and then distill the reasoning
data as transferred knowledge to the GR model. Moreover, we utilize a decision
agent as post-processing to extend the GR retrieved documents through retrieval
model and select the most relevant ones from multi perspectives as the final
generative retrieval result. Extensive offline experiments on real-world
datasets and online A/B tests on Fund Search and Insurance Search in Alipay
demonstrate our framework's superiority and effectiveness in improving search
quality and conversion gains.


---

**[180. [2504.05324] Hybrid Retrieval for Hallucination Mitigation in Large Language Models:
  A Comparative Analysis](https://arxiv.org/pdf/2504.05324.pdf)** (2025-04-09)

*Chandana Sree Mala, Gizem Gezici, Fosca Giannotti*

  Large Language Models (LLMs) excel in language comprehension and generation
but are prone to hallucinations, producing factually incorrect or unsupported
outputs. Retrieval Augmented Generation (RAG) systems address this issue by
grounding LLM responses with external knowledge. This study evaluates the
relationship between retriever effectiveness and hallucination reduction in
LLMs using three retrieval approaches: sparse retrieval based on BM25 keyword
search, dense retrieval using semantic search with Sentence Transformers, and a
proposed hybrid retrieval module. The hybrid module incorporates query
expansion and combines the results of sparse and dense retrievers through a
dynamically weighted Reciprocal Rank Fusion score. Using the HaluBench dataset,
a benchmark for hallucinations in question answering tasks, we assess retrieval
performance with metrics such as mean average precision and normalised
discounted cumulative gain, focusing on the relevance of the top three
retrieved documents. Results show that the hybrid retriever achieves better
relevance scores, outperforming both sparse and dense retrievers. Further
evaluation of LLM-generated answers against ground truth using metrics such as
accuracy, hallucination rate, and rejection rate reveals that the hybrid
retriever achieves the highest accuracy on fails, the lowest hallucination
rate, and the lowest rejection rate. These findings highlight the hybrid
retriever's ability to enhance retrieval relevance, reduce hallucination rates,
and improve LLM reliability, emphasising the importance of advanced retrieval
techniques in mitigating hallucinations and improving response accuracy.


---

**[181. [2406.09155] DefAn: Definitive Answer Dataset for LLMs Hallucination Evaluation](https://arxiv.org/pdf/2406.09155.pdf)** (2024-06-14)

*A B M Ashikur Rahman, Saeed Anwar, Muhammad Usman, Ajmal Mian*

  Large Language Models (LLMs) have demonstrated remarkable capabilities,
revolutionizing the integration of AI in daily life applications. However, they
are prone to hallucinations, generating claims that contradict established
facts, deviating from prompts, and producing inconsistent responses when the
same prompt is presented multiple times. Addressing these issues is challenging
due to the lack of comprehensive and easily assessable benchmark datasets. Most
existing datasets are small and rely on multiple-choice questions, which are
inadequate for evaluating the generative prowess of LLMs. To measure
hallucination in LLMs, this paper introduces a comprehensive benchmark dataset
comprising over 75,000 prompts across eight domains. These prompts are designed
to elicit definitive, concise, and informative answers. The dataset is divided
into two segments: one publicly available for testing and assessing LLM
performance and a hidden segment for benchmarking various LLMs. In our
experiments, we tested six LLMs-GPT-3.5, LLama 2, LLama 3, Gemini, Mixtral, and
Zephyr-revealing that overall factual hallucination ranges from 59% to 82% on
the public dataset and 57% to 76% in the hidden benchmark. Prompt misalignment
hallucination ranges from 6% to 95% in the public dataset and 17% to 94% in the
hidden counterpart. Average consistency ranges from 21% to 61% and 22% to 63%,
respectively. Domain-wise analysis shows that LLM performance significantly
deteriorates when asked for specific numeric information while performing
moderately with person, location, and date queries. Our dataset demonstrates
its efficacy and serves as a comprehensive benchmark for LLM performance
evaluation. Our dataset and LLMs responses are available at
\href{https://github.com/ashikiut/DefAn}{https://github.com/ashikiut/DefAn}.


---

**[182. [2309.02301] CIEM: Contrastive Instruction Evaluation Method for Better Instruction
  Tuning](https://arxiv.org/pdf/2309.02301.pdf)** (2023-11-27)

*Hongyu Hu, Jiyuan Zhang, Minyi Zhao, Zhenbang Sun*

  Nowadays, the research on Large Vision-Language Models (LVLMs) has been
significantly promoted thanks to the success of Large Language Models (LLM).
Nevertheless, these Vision-Language Models (VLMs) are suffering from the
drawback of hallucination -- due to insufficient understanding of vision and
language modalities, VLMs may generate incorrect perception information when
doing downstream applications, for example, captioning a non-existent entity.
To address the hallucination phenomenon, on the one hand, we introduce a
Contrastive Instruction Evaluation Method (CIEM), which is an automatic
pipeline that leverages an annotated image-text dataset coupled with an LLM to
generate factual/contrastive question-answer pairs for the evaluation of the
hallucination of VLMs. On the other hand, based on CIEM, we further propose a
new instruction tuning method called CIT (the abbreviation of Contrastive
Instruction Tuning) to alleviate the hallucination of VLMs by automatically
producing high-quality factual/contrastive question-answer pairs and
corresponding justifications for model tuning. Through extensive experiments on
CIEM and CIT, we pinpoint the hallucination issues commonly present in existing
VLMs, the disability of the current instruction-tuning dataset to handle the
hallucination phenomenon and the superiority of CIT-tuned VLMs over both CIEM
and public datasets.


---

**[183. [2504.06438] Don't Let It Hallucinate: Premise Verification via Retrieval-Augmented
  Logical Reasoning](https://arxiv.org/pdf/2504.06438.pdf)** (2025-04-10)

*Yuehan Qin, Shawn Li, Yi Nian, Xinyan Velocity Yu, Yue Zhao, Xuezhe Ma*

  Large language models (LLMs) have shown substantial capacity for generating
fluent, contextually appropriate responses. However, they can produce
hallucinated outputs, especially when a user query includes one or more false
premises-claims that contradict established facts. Such premises can mislead
LLMs into offering fabricated or misleading details. Existing approaches
include pretraining, fine-tuning, and inference-time techniques that often rely
on access to logits or address hallucinations after they occur. These methods
tend to be computationally expensive, require extensive training data, or lack
proactive mechanisms to prevent hallucination before generation, limiting their
efficiency in real-time applications. We propose a retrieval-based framework
that identifies and addresses false premises before generation. Our method
first transforms a user's query into a logical representation, then applies
retrieval-augmented generation (RAG) to assess the validity of each premise
using factual sources. Finally, we incorporate the verification results into
the LLM's prompt to maintain factual consistency in the final output.
Experiments show that this approach effectively reduces hallucinations,
improves factual accuracy, and does not require access to model logits or
large-scale fine-tuning.


---

**[184. [2406.06950] A Probabilistic Framework for LLM Hallucination Detection via Belief
  Tree Propagation](https://arxiv.org/pdf/2406.06950.pdf)** (2025-02-11)

*Bairu Hou, Yang Zhang, Jacob Andreas, Shiyu Chang*

  This paper focuses on the task of hallucination detection, which aims to
determine the truthfulness of LLM-generated statements. To address this
problem, a popular class of methods utilize the LLM's self-consistencies in its
beliefs in a set of logically related augmented statements generated by the
LLM, which does not require external knowledge databases and can work with both
white-box and black-box LLMs. However, in many existing approaches, the
augmented statements tend to be very monotone and unstructured, which makes it
difficult to integrate meaningful information from the LLM beliefs in these
statements. Also, many methods work with the binarized version of the LLM's
belief, instead of the continuous version, which significantly loses
information. To overcome these limitations, in this paper, we propose Belief
Tree Propagation (BTProp), a probabilistic framework for LLM hallucination
detection. BTProp introduces a belief tree of logically related statements by
recursively decomposing a parent statement into child statements with three
decomposition strategies, and builds a hidden Markov tree model to integrate
the LLM's belief scores in these statements in a principled way. Experiment
results show that our method improves baselines by 3%-9% (evaluated by AUROC
and AUC-PR) on multiple hallucination detection benchmarks. Code is available
at https://github.com/UCSB-NLP-Chang/BTProp.


---

**[185. [2409.15548] Beyond Conformal Predictors: Adaptive Conformal Inference with
  Confidence Predictors](https://arxiv.org/pdf/2409.15548.pdf)** (2024-10-28)

*Johan Hallberg Szabadváry*

  Conformal prediction (CP) is a robust framework for distribution-free
uncertainty quantification, but it requires exchangeable data to ensure valid
prediction sets at a user-specified significance level. When this assumption is
violated, as in time-series or other structured data, the validity guarantees
of CP no longer hold. Adaptive conformal inference (ACI) was introduced to
address this limitation by adjusting the significance level dynamically,
ensuring finite-sample coverage guarantees even for non-exchangeable data. In
this paper, we show that ACI does not require the use of conformal predictors;
instead, it can be implemented with the more general confidence predictors,
which are computationally simpler and still maintain the crucial property of
nested prediction sets. Through experiments on synthetic and real-world data,
we demonstrate that confidence predictors can perform comparably to, or even
better than, conformal predictors, particularly in terms of computational
efficiency. These findings suggest that confidence predictors represent a
viable and efficient alternative to conformal predictors in non-exchangeable
data settings, although further studies are needed to identify when one method
is superior.


---

**[186. [2503.09153] Is LLMs Hallucination Usable? LLM-based Negative Reasoning for Fake News
  Detection](https://arxiv.org/pdf/2503.09153.pdf)** (2025-03-13)

*Chaowei Zhang, Zongling Feng, Zewei Zhang, Jipeng Qiang, Guandong Xu, Yun Li*

  The questionable responses caused by knowledge hallucination may lead to
LLMs' unstable ability in decision-making. However, it has never been
investigated whether the LLMs' hallucination is possibly usable to generate
negative reasoning for facilitating the detection of fake news. This study
proposes a novel supervised self-reinforced reasoning rectification approach -
SR$^3$ that yields both common reasonable reasoning and wrong understandings
(negative reasoning) for news via LLMs reflection for semantic consistency
learning. Upon that, we construct a negative reasoning-based news learning
model called - \emph{NRFE}, which leverages positive or negative news-reasoning
pairs for learning the semantic consistency between them. To avoid the impact
of label-implicated reasoning, we deploy a student model - \emph{NRFE-D} that
only takes news content as input to inspect the performance of our method by
distilling the knowledge from \emph{NRFE}. The experimental results verified on
three popular fake news datasets demonstrate the superiority of our method
compared with three kinds of baselines including prompting on LLMs, fine-tuning
on pre-trained SLMs, and other representative fake news detection methods.


---

**[187. [2501.09997] Attention-guided Self-reflection for Zero-shot Hallucination Detection
  in Large Language Models](https://arxiv.org/pdf/2501.09997.pdf)** (2025-02-13)

*Qiang Liu, Xinlong Chen, Yue Ding, Shizhen Xu, Shu Wu, Liang Wang*

  Hallucination has emerged as a significant barrier to the effective
application of Large Language Models (LLMs). In this work, we introduce a novel
Attention-Guided SElf-Reflection (AGSER) approach for zero-shot hallucination
detection in LLMs. The AGSER method utilizes attention contributions to
categorize the input query into attentive and non-attentive queries. Each query
is then processed separately through the LLMs, allowing us to compute
consistency scores between the generated responses and the original answer. The
difference between the two consistency scores serves as a hallucination
estimator. In addition to its efficacy in detecting hallucinations, AGSER
notably reduces computational overhead, requiring only three passes through the
LLM and utilizing two sets of tokens. We have conducted extensive experiments
with four widely-used LLMs across three different hallucination benchmarks,
demonstrating that our approach significantly outperforms existing methods in
zero-shot hallucination detection.


---

**[188. [2403.04307] HaluEval-Wild: Evaluating Hallucinations of Language Models in the Wild](https://arxiv.org/pdf/2403.04307.pdf)** (2024-09-17)

*Zhiying Zhu, Yiming Yang, Zhiqing Sun*

  Hallucinations pose a significant challenge to the reliability of large
language models (LLMs) in critical domains. Recent benchmarks designed to
assess LLM hallucinations within conventional NLP tasks, such as
knowledge-intensive question answering (QA) and summarization, are insufficient
for capturing the complexities of user-LLM interactions in dynamic, real-world
settings. To address this gap, we introduce HaluEval-Wild, the first benchmark
specifically designed to evaluate LLM hallucinations in the wild. We
meticulously collect challenging (adversarially filtered by Alpaca) user
queries from ShareGPT, an existing real-world user-LLM interaction datasets, to
evaluate the hallucination rates of various LLMs. Upon analyzing the collected
queries, we categorize them into five distinct types, which enables a
fine-grained analysis of the types of hallucinations LLMs exhibit, and
synthesize the reference answers with the powerful GPT-4 model and
retrieval-augmented generation (RAG). Our benchmark offers a novel approach
towards enhancing our comprehension of and improving LLM reliability in
scenarios reflective of real-world interactions. Our benchmark is available at
https://github.com/HaluEval-Wild/HaluEval-Wild.


---

**[189. [2412.07965] HalluCana: Fixing LLM Hallucination with A Canary Lookahead](https://arxiv.org/pdf/2412.07965.pdf)** (2024-12-12)

*Tianyi Li, Erenay Dayanik, Shubhi Tyagi, Andrea Pierleoni*

  In this paper, we present HalluCana, a canary lookahead to detect and correct
factuality hallucinations of Large Language Models (LLMs) in long-form
generation. HalluCana detects and intervenes as soon as traces of hallucination
emerge, during and even before generation. To support timely detection, we
exploit the internal factuality representation in the LLM hidden space, where
we investigate various proxies to the LLMs' factuality self-assessment, and
discuss its relation to the models' context familiarity from their
pre-training. On biography generation, our method improves generation quality
by up to 2.5x, while consuming over 6 times less compute.


---

**[190. [2502.16872] Mitigating Hallucinations in Diffusion Models through Adaptive Attention
  Modulation](https://arxiv.org/pdf/2502.16872.pdf)** (2025-02-25)

*Trevine Oorloff, Yaser Yacoob, Abhinav Shrivastava*

  Diffusion models, while increasingly adept at generating realistic images,
are notably hindered by hallucinations -- unrealistic or incorrect features
inconsistent with the trained data distribution. In this work, we propose
Adaptive Attention Modulation (AAM), a novel approach to mitigate
hallucinations by analyzing and modulating the self-attention mechanism in
diffusion models. We hypothesize that self-attention during early denoising
steps may inadvertently amplify or suppress features, contributing to
hallucinations. To counter this, AAM introduces a temperature scaling mechanism
within the softmax operation of the self-attention layers, dynamically
modulating the attention distribution during inference. Additionally, AAM
employs a masked perturbation technique to disrupt early-stage noise that may
otherwise propagate into later stages as hallucinations. Extensive experiments
demonstrate that AAM effectively reduces hallucinatory artifacts, enhancing
both the fidelity and reliability of generated images. For instance, the
proposed approach improves the FID score by 20.8% and reduces the percentage of
hallucinated images by 12.9% (in absolute terms) on the Hands dataset.


---

**[191. [2502.18342] BRIDO: Bringing Democratic Order to Abstractive Summarization](https://arxiv.org/pdf/2502.18342.pdf)** (2025-02-26)

*Junhyun Lee, Harshith Goka, Hyeonmok Ko*

  Hallucination refers to the inaccurate, irrelevant, and inconsistent text
generated from large language models (LLMs). While the LLMs have shown great
promise in a variety of tasks, the issue of hallucination still remains a major
challenge for many practical uses. In this paper, we tackle the issue of
hallucination in abstract text summarization by mitigating exposure bias.
Existing models targeted for exposure bias mitigation, namely BRIO, aim for
better summarization quality in the ROUGE score. We propose a model that uses a
similar exposure bias mitigation strategy but with a goal that is aligned with
less hallucination. We conjecture that among a group of candidate outputs, ones
with hallucinations will comprise the minority of the whole group. That is,
candidates with less similarity with others will have a higher chance of
containing hallucinated content. Our method uses this aspect and utilizes
contrastive learning, incentivizing candidates with high inter-candidate ROUGE
scores. We performed experiments on the XSum and CNN/DM summarization datasets,
and our method showed 6.25% and 3.82% improvement, respectively, on the
consistency G-Eval score over BRIO.


---

**[192. [2502.08904] MIH-TCCT: Mitigating Inconsistent Hallucinations in LLMs via
  Event-Driven Text-Code Cyclic Training](https://arxiv.org/pdf/2502.08904.pdf)** (2025-02-28)

*Xinxin You, Xien Liu, Qixin Sun, Huan Zhang, Kaiyin Zhou, Shaohui Liu, GuoPing Hu, ShiJin Wang, Si Liu, Ji Wu*

  Recent methodologies utilizing synthetic datasets have aimed to address
inconsistent hallucinations in large language models (LLMs); however,these
approaches are primarily tailored to specific tasks, limiting their
generalizability. Inspired by the strong performance of code-trained models in
logic-intensive domains, we propose a novel framework that leverages
event-based text to generate corresponding code and employs cyclic training to
transfer the logical consistency of code to natural language effectively. Our
method significantly reduces inconsistent hallucinations across three leading
LLMs and two categories of natural language tasks while maintaining overall
performance. This framework effectively alleviates hallucinations without
necessitating adaptation to downstream tasks, demonstrating generality and
providing new perspectives to tackle the challenge of inconsistent
hallucinations.


---

**[193. [2403.10492] Mitigating Dialogue Hallucination for Large Vision Language Models via
  Adversarial Instruction Tuning](https://arxiv.org/pdf/2403.10492.pdf)** (2024-10-07)

*Dongmin Park, Zhaofang Qian, Guangxing Han, Ser-Nam Lim*

  Mitigating hallucinations of Large Vision Language Models,(LVLMs) is crucial
to enhance their reliability for general-purpose assistants. This paper shows
that such hallucinations of LVLMs can be significantly exacerbated by preceding
user-system dialogues. To precisely measure this, we first present an
evaluation benchmark by extending popular multi-modal benchmark datasets with
prepended hallucinatory dialogues powered by our novel Adversarial Question
Generator (AQG), which can automatically generate image-related yet adversarial
dialogues by adopting adversarial attacks on LVLMs. On our benchmark, the
zero-shot performance of state-of-the-art LVLMs drops significantly for both
the VQA and Captioning tasks. Next, we further reveal this hallucination is
mainly due to the prediction bias toward preceding dialogues rather than visual
content. To reduce this bias, we propose Adversarial Instruction Tuning (AIT)
that robustly fine-tunes LVLMs against hallucinatory dialogues. Extensive
experiments show our proposed approach successfully reduces dialogue
hallucination while maintaining performance.


---

**[194. [2407.04831] Code Hallucination](https://arxiv.org/pdf/2407.04831.pdf)** (2024-08-09)

*Mirza Masfiqur Rahman, Ashish Kundu*

  Generative models such as large language models are extensively used as code
copilots and for whole program generation. However, the programs they generate
often have questionable correctness, authenticity and reliability in terms of
integration as they might not follow the user requirements, provide incorrect
and/or nonsensical outputs, or even contain semantic/syntactic errors - overall
known as LLM hallucination. In this work, we present several types of code
hallucination. We have generated such hallucinated code manually using large
language models. We also present a technique - HallTrigger, in order to
demonstrate efficient ways of generating arbitrary code hallucination. Our
method leverages 3 different dynamic attributes of LLMs to craft prompts that
can successfully trigger hallucinations from models without the need to access
model architecture or parameters. Results from popular blackbox models suggest
that HallTrigger is indeed effective and the pervasive LLM hallucination have
sheer impact on software development.


---

**[195. [2405.18027] TimeChara: Evaluating Point-in-Time Character Hallucination of
  Role-Playing Large Language Models](https://arxiv.org/pdf/2405.18027.pdf)** (2024-05-29)

*Jaewoo Ahn, Taehyun Lee, Junyoung Lim, Jin-Hwa Kim, Sangdoo Yun, Hwaran Lee, Gunhee Kim*

  While Large Language Models (LLMs) can serve as agents to simulate human
behaviors (i.e., role-playing agents), we emphasize the importance of
point-in-time role-playing. This situates characters at specific moments in the
narrative progression for three main reasons: (i) enhancing users' narrative
immersion, (ii) avoiding spoilers, and (iii) fostering engagement in fandom
role-playing. To accurately represent characters at specific time points,
agents must avoid character hallucination, where they display knowledge that
contradicts their characters' identities and historical timelines. We introduce
TimeChara, a new benchmark designed to evaluate point-in-time character
hallucination in role-playing LLMs. Comprising 10,895 instances generated
through an automated pipeline, this benchmark reveals significant hallucination
issues in current state-of-the-art LLMs (e.g., GPT-4o). To counter this
challenge, we propose Narrative-Experts, a method that decomposes the reasoning
steps and utilizes narrative experts to reduce point-in-time character
hallucinations effectively. Still, our findings with TimeChara highlight the
ongoing challenges of point-in-time character hallucination, calling for
further study.


---

**[196. [2411.11919] VL-Uncertainty: Detecting Hallucination in Large Vision-Language Model
  via Uncertainty Estimation](https://arxiv.org/pdf/2411.11919.pdf)** (2024-12-03)

*Ruiyang Zhang, Hu Zhang, Zhedong Zheng*

  Given the higher information load processed by large vision-language models
(LVLMs) compared to single-modal LLMs, detecting LVLM hallucinations requires
more human and time expense, and thus rise a wider safety concerns. In this
paper, we introduce VL-Uncertainty, the first uncertainty-based framework for
detecting hallucinations in LVLMs. Different from most existing methods that
require ground-truth or pseudo annotations, VL-Uncertainty utilizes uncertainty
as an intrinsic metric. We measure uncertainty by analyzing the prediction
variance across semantically equivalent but perturbed prompts, including visual
and textual data. When LVLMs are highly confident, they provide consistent
responses to semantically equivalent queries. However, when uncertain, the
responses of the target LVLM become more random. Considering semantically
similar answers with different wordings, we cluster LVLM responses based on
their semantic content and then calculate the cluster distribution entropy as
the uncertainty measure to detect hallucination. Our extensive experiments on
10 LVLMs across four benchmarks, covering both free-form and multi-choice
tasks, show that VL-Uncertainty significantly outperforms strong baseline
methods in hallucination detection.


---

**[197. [2407.00499] ConU: Conformal Uncertainty in Large Language Models with Correctness
  Coverage Guarantees](https://arxiv.org/pdf/2407.00499.pdf)** (2024-11-19)

*Zhiyuan Wang, Jinhao Duan, Lu Cheng, Yue Zhang, Qingni Wang, Xiaoshuang Shi, Kaidi Xu, Hengtao Shen, Xiaofeng Zhu*

  Uncertainty quantification (UQ) in natural language generation (NLG) tasks
remains an open challenge, exacerbated by the closed-source nature of the
latest large language models (LLMs). This study investigates applying conformal
prediction (CP), which can transform any heuristic uncertainty notion into
rigorous prediction sets, to black-box LLMs in open-ended NLG tasks. We
introduce a novel uncertainty measure based on self-consistency theory, and
then develop a conformal uncertainty criterion by integrating the uncertainty
condition aligned with correctness into the CP algorithm. Empirical evaluations
indicate that our uncertainty measure outperforms prior state-of-the-art
methods. Furthermore, we achieve strict control over the correctness coverage
rate utilizing 7 popular LLMs on 4 free-form NLG datasets, spanning
general-purpose and medical scenarios. Additionally, the calibrated prediction
sets with small size further highlights the efficiency of our method in
providing trustworthy guarantees for practical open-ended NLG applications.


---

**[198. [2406.06211] iMotion-LLM: Motion Prediction Instruction Tuning](https://arxiv.org/pdf/2406.06211.pdf)** (2024-06-12)

*Abdulwahab Felemban, Eslam Mohamed Bakr, Xiaoqian Shen, Jian Ding, Abduallah Mohamed, Mohamed Elhoseiny*

  We introduce iMotion-LLM: a Multimodal Large Language Models (LLMs) with
trajectory prediction, tailored to guide interactive multi-agent scenarios.
Different from conventional motion prediction approaches, iMotion-LLM
capitalizes on textual instructions as key inputs for generating contextually
relevant trajectories. By enriching the real-world driving scenarios in the
Waymo Open Dataset with textual motion instructions, we created InstructWaymo.
Leveraging this dataset, iMotion-LLM integrates a pretrained LLM, fine-tuned
with LoRA, to translate scene features into the LLM input space. iMotion-LLM
offers significant advantages over conventional motion prediction models.
First, it can generate trajectories that align with the provided instructions
if it is a feasible direction. Second, when given an infeasible direction, it
can reject the instruction, thereby enhancing safety. These findings act as
milestones in empowering autonomous navigation systems to interpret and predict
the dynamics of multi-agent environments, laying the groundwork for future
advancements in this field.


---
