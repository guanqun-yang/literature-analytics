**[1. [2408.15533] LRP4RAG: Detecting Hallucinations in Retrieval-Augmented Generation via
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

**[2. [2410.11414] ReDeEP: Detecting Hallucination in Retrieval-Augmented Generation via
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

**[3. [2502.17125] LettuceDetect: A Hallucination Detection Framework for RAG Applications](https://arxiv.org/pdf/2502.17125.pdf)** (2025-02-25)

*Ádám Kovács, Gábor Recski*

  Retrieval Augmented Generation (RAG) systems remain vulnerable to
hallucinated answers despite incorporating external knowledge sources. We
present LettuceDetect a framework that addresses two critical limitations in
existing hallucination detection methods: (1) the context window constraints of
traditional encoder-based methods, and (2) the computational inefficiency of
LLM based approaches. Building on ModernBERT's extended context capabilities
(up to 8k tokens) and trained on the RAGTruth benchmark dataset, our approach
outperforms all previous encoder-based models and most prompt-based models,
while being approximately 30 times smaller than the best models. LettuceDetect
is a token-classification model that processes context-question-answer triples,
allowing for the identification of unsupported claims at the token level.
Evaluations on the RAGTruth corpus demonstrate an F1 score of 79.22% for
example-level detection, which is a 14.8% improvement over Luna, the previous
state-of-the-art encoder-based architecture. Additionally, the system can
process 30 to 60 examples per second on a single GPU, making it more practical
for real-world RAG applications.


---

**[4. [2412.05223] 100% Elimination of Hallucinations on RAGTruth for GPT-4 and GPT-3.5
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

**[5. [2502.19209] Bi'an: A Bilingual Benchmark and Model for Hallucination Detection in
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

**[6. [2407.19794] Introducing a new hyper-parameter for RAG: Context Window Utilization](https://arxiv.org/pdf/2407.19794.pdf)** (2024-08-20)

*Kush Juvekar, Anupam Purwar*

  This paper introduces a new hyper-parameter for Retrieval-Augmented
Generation (RAG) systems called Context Window Utilization. RAG systems enhance
generative models by incorporating relevant information retrieved from external
knowledge bases, improving the factual accuracy and contextual relevance of
generated responses. The size of the text chunks retrieved and processed is a
critical factor influencing RAG performance. This study aims to identify the
optimal chunk size that maximizes answer generation quality. Through systematic
experimentation, we analyze the effects of varying chunk sizes on the
efficiency and effectiveness of RAG frameworks. Our findings reveal that an
optimal chunk size balances the trade-off between providing sufficient context
and minimizing irrelevant information. These insights are crucial for enhancing
the design and implementation of RAG systems, underscoring the importance of
selecting an appropriate chunk size to achieve superior performance.


---

**[7. [2412.04235] Addressing Hallucinations with RAG and NMISS in Italian Healthcare LLM
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

**[8. [2503.21157] Real-Time Evaluation Models for RAG: Who Detects Hallucinations Best?](https://arxiv.org/pdf/2503.21157.pdf)** (2025-04-08)

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

**[9. [2309.05922] A Survey of Hallucination in Large Foundation Models](https://arxiv.org/pdf/2309.05922.pdf)** (2023-09-13)

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

**[10. [2504.10198] DioR: Adaptive Cognitive Detection and Contextual Retrieval Optimization
  for Dynamic Retrieval-Augmented Generation](https://arxiv.org/pdf/2504.10198.pdf)** (2025-04-15)

*Hanghui Guo, Jia Zhu, Shimin Di, Weijie Shi, Zhangze Chen, Jiajie Xu*

  Dynamic Retrieval-augmented Generation (RAG) has shown great success in
mitigating hallucinations in large language models (LLMs) during generation.
However, existing dynamic RAG methods face significant limitations in two key
aspects: 1) Lack of an effective mechanism to control retrieval triggers, and
2) Lack of effective scrutiny of retrieval content. To address these
limitations, we propose an innovative dynamic RAG method, DioR (Adaptive
Cognitive Detection and Contextual Retrieval Optimization), which consists of
two main components: adaptive cognitive detection and contextual retrieval
optimization, specifically designed to determine when retrieval is needed and
what to retrieve for LLMs is useful. Experimental results demonstrate that DioR
achieves superior performance on all tasks, demonstrating the effectiveness of
our work.


---

**[11. [2407.08488] Lynx: An Open Source Hallucination Evaluation Model](https://arxiv.org/pdf/2407.08488.pdf)** (2024-07-24)

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

**[12. [2503.21813] OAEI-LLM-T: A TBox Benchmark Dataset for Understanding LLM
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

**[13. [2502.13416] Detecting LLM Fact-conflicting Hallucinations Enhanced by
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

**[15. [2401.00396] RAGTruth: A Hallucination Corpus for Developing Trustworthy
  Retrieval-Augmented Language Models](https://arxiv.org/pdf/2401.00396.pdf)** (2024-05-20)

*Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu, Kashun Shum, Randy Zhong, Juntong Song, Tong Zhang*

  Retrieval-augmented generation (RAG) has become a main technique for
alleviating hallucinations in large language models (LLMs). Despite the
integration of RAG, LLMs may still present unsupported or contradictory claims
to the retrieved contents. In order to develop effective hallucination
prevention strategies under RAG, it is important to create benchmark datasets
that can measure the extent of hallucination. This paper presents RAGTruth, a
corpus tailored for analyzing word-level hallucinations in various domains and
tasks within the standard RAG frameworks for LLM applications. RAGTruth
comprises nearly 18,000 naturally generated responses from diverse LLMs using
RAG. These responses have undergone meticulous manual annotations at both the
individual cases and word levels, incorporating evaluations of hallucination
intensity. We not only benchmark hallucination frequencies across different
LLMs, but also critically assess the effectiveness of several existing
hallucination detection methodologies. Furthermore, we show that using a
high-quality dataset such as RAGTruth, it is possible to finetune a relatively
small LLM and achieve a competitive level of performance in hallucination
detection when compared to the existing prompt-based approaches using
state-of-the-art large language models such as GPT-4.


---

**[16. [2412.14905] Dehallucinating Parallel Context Extension for Retrieval-Augmented
  Generation](https://arxiv.org/pdf/2412.14905.pdf)** (2024-12-20)

*Zexiong Ma, Shengnan An, Zeqi Lin, Yanzhen Zou, Jian-Guang Lou, Bing Xie*

  Large language models (LLMs) are susceptible to generating hallucinated
information, despite the integration of retrieval-augmented generation (RAG).
Parallel context extension (PCE) is a line of research attempting to
effectively integrating parallel (unordered) contexts, while it still suffers
from hallucinations when adapted to RAG scenarios. In this paper, we propose
DePaC (Dehallucinating Parallel Context Extension), which alleviates the
hallucination problem with context-aware negative training and
information-calibrated aggregation. DePaC is designed to alleviate two types of
in-context hallucination: fact fabrication (i.e., LLMs present claims that are
not supported by the contexts) and fact omission (i.e., LLMs fail to present
claims that can be supported by the contexts). Specifically, (1) for fact
fabrication, we apply the context-aware negative training that fine-tunes the
LLMs with negative supervisions, thus explicitly guiding the LLMs to refuse to
answer when contexts are not related to questions; (2) for fact omission, we
propose the information-calibrated aggregation which prioritizes context
windows with higher information increment from their contexts. The experimental
results on nine RAG tasks demonstrate that DePaC significantly alleviates the
two types of hallucination and consistently achieves better performances on
these tasks.


---

**[17. [2503.01921] NCL-UoR at SemEval-2025 Task 3: Detecting Multilingual Hallucination and
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

**[18. [2504.07103] FG-RAG: Enhancing Query-Focused Summarization with Context-Aware
  Fine-Grained Graph RAG](https://arxiv.org/pdf/2504.07103.pdf)** (2025-04-11)

*Yubin Hong, Chaofan Li, Jingyi Zhang, Yingxia Shao*

  Retrieval-Augmented Generation (RAG) enables large language models to provide
more precise and pertinent responses by incorporating external knowledge. In
the Query-Focused Summarization (QFS) task, GraphRAG-based approaches have
notably enhanced the comprehensiveness and diversity of generated responses.
However, existing GraphRAG-based approaches predominantly focus on
coarse-grained information summarization without being aware of the specific
query, and the retrieved content lacks sufficient contextual information to
generate comprehensive responses. To address the deficiencies of current RAG
systems, we propose Context-Aware Fine-Grained Graph RAG (FG-RAG) to enhance
the performance of the QFS task. FG-RAG employs Context-Aware Entity Expansion
in graph retrieval to expand the coverage of retrieved entities in the graph,
thus providing enough contextual information for the retrieved content.
Furthermore, FG-RAG utilizes Query-Level Fine-Grained Summarization to
incorporate fine-grained details during response generation, enhancing query
awareness for the generated summarization. Our evaluation demonstrates that
FG-RAG outperforms other RAG systems in multiple metrics of comprehensiveness,
diversity, and empowerment when handling the QFS task. Our implementation is
available at https://github.com/BuptWululu/FG-RAG.


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

**[20. [2409.14038] OAEI-LLM: A Benchmark Dataset for Understanding Large Language Model
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

**[21. [2504.05163] Evaluating Knowledge Graph Based Retrieval Augmented Generation Methods
  under Knowledge Incompleteness](https://arxiv.org/pdf/2504.05163.pdf)** (2025-04-08)

*Dongzhuoran Zhou, Yuqicheng Zhu, Yuan He, Jiaoyan Chen, Evgeny Kharlamov, Steffen Staab*

  Knowledge Graph based Retrieval-Augmented Generation (KG-RAG) is a technique
that enhances Large Language Model (LLM) inference in tasks like Question
Answering (QA) by retrieving relevant information from knowledge graphs (KGs).
However, real-world KGs are often incomplete, meaning that essential
information for answering questions may be missing. Existing benchmarks do not
adequately capture the impact of KG incompleteness on KG-RAG performance. In
this paper, we systematically evaluate KG-RAG methods under incomplete KGs by
removing triples using different methods and analyzing the resulting effects.
We demonstrate that KG-RAG methods are sensitive to KG incompleteness,
highlighting the need for more robust approaches in realistic settings.


---

**[22. [2501.12975] OnionEval: An Unified Evaluation of Fact-conflicting Hallucination for
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

**[23. [2412.19494] Retrieval-augmented Generation for GenAI-enabled Semantic Communications](https://arxiv.org/pdf/2412.19494.pdf)** (2024-12-30)

*Shunpu Tang, Ruichen Zhang, Yuxuan Yan, Qianqian Yang, Dusit Niyato, Xianbin Wang, Shiwen Mao*

  Semantic communication (SemCom) is an emerging paradigm aiming at
transmitting only task-relevant semantic information to the receiver, which can
significantly improve communication efficiency. Recent advancements in
generative artificial intelligence (GenAI) have empowered GenAI-enabled SemCom
(GenSemCom) to further expand its potential in various applications. However,
current GenSemCom systems still face challenges such as semantic inconsistency,
limited adaptability to diverse tasks and dynamic environments, and the
inability to leverage insights from past transmission. Motivated by the success
of retrieval-augmented generation (RAG) in the domain of GenAI, this paper
explores the integration of RAG in GenSemCom systems. Specifically, we first
provide a comprehensive review of existing GenSemCom systems and the
fundamentals of RAG techniques. We then discuss how RAG can be integrated into
GenSemCom. Following this, we conduct a case study on semantic image
transmission using an RAG-enabled diffusion-based SemCom system, demonstrating
the effectiveness of the proposed integration. Finally, we outline future
directions for advancing RAG-enabled GenSemCom systems.


---

**[24. [2408.05141] A Hybrid RAG System with Comprehensive Enhancement on Complex Reasoning](https://arxiv.org/pdf/2408.05141.pdf)** (2024-09-04)

*Ye Yuan, Chengwu Liu, Jingyang Yuan, Gongbo Sun, Siqi Li, Ming Zhang*

  Retrieval-augmented generation (RAG) is a framework enabling large language
models (LLMs) to enhance their accuracy and reduce hallucinations by
integrating external knowledge bases. In this paper, we introduce a hybrid RAG
system enhanced through a comprehensive suite of optimizations that
significantly improve retrieval quality, augment reasoning capabilities, and
refine numerical computation ability. We refined the text chunks and tables in
web pages, added attribute predictors to reduce hallucinations, conducted LLM
Knowledge Extractor and Knowledge Graph Extractor, and finally built a
reasoning strategy with all the references. We evaluated our system on the CRAG
dataset through the Meta CRAG KDD Cup 2024 Competition. Both the local and
online evaluations demonstrate that our system significantly enhances complex
reasoning capabilities. In local evaluations, we have significantly improved
accuracy and reduced error rates compared to the baseline model, achieving a
notable increase in scores. In the meanwhile, we have attained outstanding
results in online assessments, demonstrating the performance and generalization
capabilities of the proposed system. The source code for our system is released
in \url{https://gitlab.aicrowd.com/shizueyy/crag-new}.


---

**[25. [2501.15098] CFT-RAG: An Entity Tree Based Retrieval Augmented Generation Algorithm
  With Cuckoo Filter](https://arxiv.org/pdf/2501.15098.pdf)** (2025-01-28)

*Zihang Li, Yangdong Ruan, Wenjun Liu, Zhengyang Wang, Tong Yang*

  Although retrieval-augmented generation(RAG) significantly improves
generation quality by retrieving external knowledge bases and integrating
generated content, it faces computational efficiency bottlenecks, particularly
in knowledge retrieval tasks involving hierarchical structures for Tree-RAG.
This paper proposes a Tree-RAG acceleration method based on the improved Cuckoo
Filter, which optimizes entity localization during the retrieval process to
achieve significant performance improvements. Tree-RAG effectively organizes
entities through the introduction of a hierarchical tree structure, while the
Cuckoo Filter serves as an efficient data structure that supports rapid
membership queries and dynamic updates. The experiment results demonstrate that
our method is much faster than naive Tree-RAG while maintaining high levels of
generative quality. When the number of trees is large, our method is hundreds
of times faster than naive Tree-RAG. Our work is available at
https://github.com/TUPYP7180/CFT-RAG-2025.


---

**[26. [2409.17648] Efficient In-Domain Question Answering for Resource-Constrained
  Environments](https://arxiv.org/pdf/2409.17648.pdf)** (2024-10-18)

*Isaac Chung, Phat Vo, Arman C. Kizilkale, Aaron Reite*

  Retrieval Augmented Generation (RAG) is a common method for integrating
external knowledge into pretrained Large Language Models (LLMs) to enhance
accuracy and relevancy in question answering (QA) tasks. However, prompt
engineering and resource efficiency remain significant bottlenecks in
developing optimal and robust RAG solutions for real-world QA applications.
Recent studies have shown success in using fine tuning to address these
problems; in particular, Retrieval Augmented Fine Tuning (RAFT) applied to
smaller 7B models has demonstrated superior performance compared to RAG setups
with much larger models such as GPT-3.5. The combination of RAFT with
parameter-efficient fine tuning (PEFT) techniques, such as Low-Rank Adaptation
(LoRA), promises an even more efficient solution, yet remains an unexplored
area. In this work, we combine RAFT with LoRA to reduce fine tuning and storage
requirements and gain faster inference times while maintaining comparable RAG
performance. This results in a more compute-efficient RAFT, or CRAFT, which is
particularly useful for knowledge-intensive QA tasks in resource-constrained
environments where internet access may be restricted and hardware resources
limited.


---

**[27. [2503.20757] MCTS-RAG: Enhancing Retrieval-Augmented Generation with Monte Carlo Tree
  Search](https://arxiv.org/pdf/2503.20757.pdf)** (2025-03-27)

*Yunhai Hu, Yilun Zhao, Chen Zhao, Arman Cohan*

  We introduce MCTS-RAG, a novel approach that enhances the reasoning
capabilities of small language models on knowledge-intensive tasks by
leveraging retrieval-augmented generation (RAG) to provide relevant context and
Monte Carlo Tree Search (MCTS) to refine reasoning paths. MCTS-RAG dynamically
integrates retrieval and reasoning through an iterative decision-making
process. Unlike standard RAG methods, which typically retrieve information
independently from reasoning and thus integrate knowledge suboptimally, or
conventional MCTS reasoning, which depends solely on internal model knowledge
without external facts, MCTS-RAG combines structured reasoning with adaptive
retrieval. This integrated approach enhances decision-making, reduces
hallucinations, and ensures improved factual accuracy and response consistency.
The experimental results on multiple reasoning and knowledge-intensive datasets
datasets (i.e., ComplexWebQA, GPQA, and FoolMeTwice) show that our method
enables small-scale LMs to achieve performance comparable to frontier LLMs like
GPT-4o by effectively scaling inference-time compute, setting a new standard
for reasoning in small-scale models.


---

**[28. [2501.03995] RAG-Check: Evaluating Multimodal Retrieval Augmented Generation
  Performance](https://arxiv.org/pdf/2501.03995.pdf)** (2025-01-08)

*Matin Mortaheb, Mohammad A. Amir Khojastepour, Srimat T. Chakradhar, Sennur Ulukus*

  Retrieval-augmented generation (RAG) improves large language models (LLMs) by
using external knowledge to guide response generation, reducing hallucinations.
However, RAG, particularly multi-modal RAG, can introduce new hallucination
sources: (i) the retrieval process may select irrelevant pieces (e.g.,
documents, images) as raw context from the database, and (ii) retrieved images
are processed into text-based context via vision-language models (VLMs) or
directly used by multi-modal language models (MLLMs) like GPT-4o, which may
hallucinate. To address this, we propose a novel framework to evaluate the
reliability of multi-modal RAG using two performance measures: (i) the
relevancy score (RS), assessing the relevance of retrieved entries to the
query, and (ii) the correctness score (CS), evaluating the accuracy of the
generated response. We train RS and CS models using a ChatGPT-derived database
and human evaluator samples. Results show that both models achieve ~88%
accuracy on test data. Additionally, we construct a 5000-sample human-annotated
database evaluating the relevancy of retrieved pieces and the correctness of
response statements. Our RS model aligns with human preferences 20% more often
than CLIP in retrieval, and our CS model matches human preferences ~91% of the
time. Finally, we assess various RAG systems' selection and generation
performances using RS and CS.


---

**[29. [2311.01740] SAC3: Reliable Hallucination Detection in Black-Box Language Models via
  Semantic-aware Cross-check Consistency](https://arxiv.org/pdf/2311.01740.pdf)** (2024-02-20)

*Jiaxin Zhang, Zhuohang Li, Kamalika Das, Bradley A. Malin, Sricharan Kumar*

  Hallucination detection is a critical step toward understanding the
trustworthiness of modern language models (LMs). To achieve this goal, we
re-examine existing detection approaches based on the self-consistency of LMs
and uncover two types of hallucinations resulting from 1) question-level and 2)
model-level, which cannot be effectively identified through self-consistency
check alone. Building upon this discovery, we propose a novel sampling-based
method, i.e., semantic-aware cross-check consistency (SAC3) that expands on the
principle of self-consistency checking. Our SAC3 approach incorporates
additional mechanisms to detect both question-level and model-level
hallucinations by leveraging advances including semantically equivalent
question perturbation and cross-model response consistency checking. Through
extensive and systematic empirical analysis, we demonstrate that SAC3
outperforms the state of the art in detecting both non-factual and factual
statements across multiple question-answering and open-domain generation
benchmarks.


---

**[30. [2504.10063] Hallucination Detection in LLMs via Topological Divergence on Attention
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

**[31. [2503.13514] RAG-KG-IL: A Multi-Agent Hybrid Framework for Reducing Hallucinations
  and Enhancing LLM Reasoning through RAG and Incremental Knowledge Graph
  Learning Integration](https://arxiv.org/pdf/2503.13514.pdf)** (2025-03-19)

*University of Derby  Hong Qing Yu, Bloc Digital  Frank McQuade*

  This paper presents RAG-KG-IL, a novel multi-agent hybrid framework designed
to enhance the reasoning capabilities of Large Language Models (LLMs) by
integrating Retrieval-Augmented Generation (RAG) and Knowledge Graphs (KGs)
with an Incremental Learning (IL) approach. Despite recent advancements, LLMs
still face significant challenges in reasoning with structured data, handling
dynamic knowledge evolution, and mitigating hallucinations, particularly in
mission-critical domains. Our proposed RAG-KG-IL framework addresses these
limitations by employing a multi-agent architecture that enables continuous
knowledge updates, integrates structured knowledge, and incorporates autonomous
agents for enhanced explainability and reasoning. The framework utilizes RAG to
ensure the generated responses are grounded in verifiable information, while
KGs provide structured domain knowledge for improved consistency and depth of
understanding. The Incremental Learning approach allows for dynamic updates to
the knowledge base without full retraining, significantly reducing
computational overhead and improving the model's adaptability. We evaluate the
framework using real-world case studies involving health-related queries,
comparing it to state-of-the-art models like GPT-4o and a RAG-only baseline.
Experimental results demonstrate that our approach significantly reduces
hallucination rates and improves answer completeness and reasoning accuracy.
The results underscore the potential of combining RAG, KGs, and multi-agent
systems to create intelligent, adaptable systems capable of real-time knowledge
integration and reasoning in complex domains.


---

**[32. [2503.01670] Evaluating LLMs' Assessment of Mixed-Context Hallucination Through the
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

**[33. [2305.11747] HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large
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

**[34. [2406.15927] Semantic Entropy Probes: Robust and Cheap Hallucination Detection in
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

**[35. [2403.04307] HaluEval-Wild: Evaluating Hallucinations of Language Models in the Wild](https://arxiv.org/pdf/2403.04307.pdf)** (2024-09-17)

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

**[36. [2403.11116] PhD: A ChatGPT-Prompted Visual hallucination Evaluation Dataset](https://arxiv.org/pdf/2403.11116.pdf)** (2025-04-15)

*Jiazhen Liu, Yuhan Fu, Ruobing Xie, Runquan Xie, Xingwu Sun, Fengzong Lian, Zhanhui Kang, Xirong Li*

  Multimodal Large Language Models (MLLMs) hallucinate, resulting in an
emerging topic of visual hallucination evaluation (VHE). This paper contributes
a ChatGPT-Prompted visual hallucination evaluation Dataset (PhD) for objective
VHE at a large scale. The essence of VHE is to ask an MLLM questions about
specific images to assess its susceptibility to hallucination. Depending on
what to ask (objects, attributes, sentiment, etc.) and how the questions are
asked, we structure PhD along two dimensions, i.e. task and mode. Five visual
recognition tasks, ranging from low-level (object / attribute recognition) to
middle-level (sentiment / position recognition and counting), are considered.
Besides a normal visual QA mode, which we term PhD-base, PhD also asks
questions with specious context (PhD-sec) or with incorrect context ({PhD-icc),
or with AI-generated counter common sense images (PhD-ccs). We construct PhD by
a ChatGPT-assisted semi-automated pipeline, encompassing four pivotal modules:
task-specific hallucinatory item (hitem) selection, hitem-embedded question
generation, specious / incorrect context generation, and counter-common-sense
(CCS) image generation. With over 14k daily images, 750 CCS images and 102k VQA
triplets in total, PhD reveals considerable variability in MLLMs' performance
across various modes and tasks, offering valuable insights into the nature of
hallucination. As such, PhD stands as a potent tool not only for VHE but may
also play a significant role in the refinement of MLLMs.


---

**[37. [2502.13490] What are Models Thinking about? Understanding Large Language Model
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

**[38. [2404.16032] Studying Large Language Model Behaviors Under Context-Memory Conflicts
  With Real Documents](https://arxiv.org/pdf/2404.16032.pdf)** (2024-10-10)

*Evgenii Kortukov, Alexander Rubinstein, Elisa Nguyen, Seong Joon Oh*

  Retrieval-augmented generation (RAG) mitigates many problems of fully
parametric language models, such as temporal degradation, hallucinations, and
lack of grounding. In RAG, the model's knowledge can be updated from documents
provided in context. This leads to cases of conflict between the model's
parametric knowledge and the contextual information, where the model may not
always update its knowledge. Previous work studied context-memory knowledge
conflicts by creating synthetic documents that contradict the model's correct
parametric answers. We present a framework for studying such knowledge
conflicts in a realistic setup. We update incorrect parametric knowledge using
real conflicting documents. This reflects how knowledge conflicts arise in
practice. In this realistic scenario, we find that knowledge updates fail less
often than previously reported. In cases where the models still fail to update
their answers, we find a parametric bias: the incorrect parametric answer
appearing in context makes the knowledge update likelier to fail. These results
suggest that the factual parametric knowledge of LLMs can negatively influence
their reading abilities and behaviors. Our code is available at
https://github.com/kortukov/realistic_knowledge_conflicts/ .


---

**[39. [2407.16833] Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive
  Study and Hybrid Approach](https://arxiv.org/pdf/2407.16833.pdf)** (2024-10-18)

*Zhuowan Li, Cheng Li, Mingyang Zhang, Qiaozhu Mei, Michael Bendersky*

  Retrieval Augmented Generation (RAG) has been a powerful tool for Large
Language Models (LLMs) to efficiently process overly lengthy contexts. However,
recent LLMs like Gemini-1.5 and GPT-4 show exceptional capabilities to
understand long contexts directly. We conduct a comprehensive comparison
between RAG and long-context (LC) LLMs, aiming to leverage the strengths of
both. We benchmark RAG and LC across various public datasets using three latest
LLMs. Results reveal that when resourced sufficiently, LC consistently
outperforms RAG in terms of average performance. However, RAG's significantly
lower cost remains a distinct advantage. Based on this observation, we propose
Self-Route, a simple yet effective method that routes queries to RAG or LC
based on model self-reflection. Self-Route significantly reduces the
computation cost while maintaining a comparable performance to LC. Our findings
provide a guideline for long-context applications of LLMs using RAG and LC.


---

**[40. [2411.09689] LLM Hallucination Reasoning with Zero-shot Knowledge Test](https://arxiv.org/pdf/2411.09689.pdf)** (2024-11-15)

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

**[41. [2405.13008] Control Token with Dense Passage Retrieval](https://arxiv.org/pdf/2405.13008.pdf)** (2024-05-24)

*Juhwan Lee, Jisu Kim*

  This study addresses the hallucination problem in large language models
(LLMs). We adopted Retrieval-Augmented Generation(RAG) (Lewis et al., 2020), a
technique that involves embedding relevant information in the prompt to obtain
accurate answers. However, RAG also faced inherent issues in retrieving correct
information. To address this, we employed the Dense Passage Retrieval(DPR)
(Karpukhin et al., 2020) model for fetching domain-specific documents related
to user queries. Despite this, the DPR model still lacked accuracy in document
retrieval. We enhanced the DPR model by incorporating control tokens, achieving
significantly superior performance over the standard DPR model, with a 13%
improvement in Top-1 accuracy and a 4% improvement in Top-20 accuracy.


---

**[42. [2208.06313] Voxels Intersecting along Orthogonal Levels Attention U-Net for
  Intracerebral Haemorrhage Segmentation in Head CT](https://arxiv.org/pdf/2208.06313.pdf)** (2023-04-26)

*Qinghui Liu, Bradley J MacIntosh, Till Schellhorn, Karoline Skogen, KyrreEeg Emblem, Atle Bjørnerud*

  We propose a novel and flexible attention based U-Net architecture referred
to as "Voxels-Intersecting Along Orthogonal Levels Attention U-Net"
(viola-Unet), for intracranial hemorrhage (ICH) segmentation task in the
INSTANCE 2022 Data Challenge on non-contrast computed tomography (CT). The
performance of ICH segmentation was improved by efficiently incorporating fused
spatially orthogonal and cross-channel features via our proposed Viola
attention plugged into the U-Net decoding branches. The viola-Unet outperformed
the strong baseline nnU-Net models during both 5-fold cross validation and
online validation. Our solution was the winner of the challenge validation
phase in terms of all four performance metrics (i.e., DSC, HD, NSD, and RVD).
The code base, pretrained weights, and docker image of the viola-Unet AI tool
are publicly available at \url{https://github.com/samleoqh/Viola-Unet}.


---

**[43. [2411.01696] Conformal Risk Minimization with Variance Reduction](https://arxiv.org/pdf/2411.01696.pdf)** (2025-02-11)

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

**[44. [2108.04926] First Order Locally Orderless Registration](https://arxiv.org/pdf/2108.04926.pdf)** (2021-08-12)

*Sune Darkner, Jose D Tascon, Francois Lauze*

  First Order Locally Orderless Registration (FLOR) is a scale-space framework
for image density estimation used for defining image similarity, mainly for
Image Registration. The Locally Orderless Registration framework was designed
in principle to use zeroth-order information, providing image density estimates
over three scales: image scale, intensity scale, and integration scale. We
extend it to take first-order information into account and hint at higher-order
information. We show how standard similarity measures extend into the
framework. We study especially Sum of Squared Differences (SSD) and Normalized
Cross-Correlation (NCC) but present the theory of how Normalised Mutual
Information (NMI) can be included.


---

**[45. [2504.08020] Learning Fine-grained Domain Generalization via Hyperbolic State Space
  Hallucination](https://arxiv.org/pdf/2504.08020.pdf)** (2025-04-14)

*Qi Bi, Jingjun Yi, Haolan Zhan, Wei Ji, Gui-Song Xia*

  Fine-grained domain generalization (FGDG) aims to learn a fine-grained
representation that can be well generalized to unseen target domains when only
trained on the source domain data. Compared with generic domain generalization,
FGDG is particularly challenging in that the fine-grained category can be only
discerned by some subtle and tiny patterns. Such patterns are particularly
fragile under the cross-domain style shifts caused by illumination, color and
etc. To push this frontier, this paper presents a novel Hyperbolic State Space
Hallucination (HSSH) method. It consists of two key components, namely, state
space hallucination (SSH) and hyperbolic manifold consistency (HMC). SSH
enriches the style diversity for the state embeddings by firstly extrapolating
and then hallucinating the source images. Then, the pre- and post- style
hallucinate state embeddings are projected into the hyperbolic manifold. The
hyperbolic state space models the high-order statistics, and allows a better
discernment of the fine-grained patterns. Finally, the hyperbolic distance is
minimized, so that the impact of style variation on fine-grained patterns can
be eliminated. Experiments on three FGDG benchmarks demonstrate its
state-of-the-art performance.


---

**[46. [2212.05765] Information-Theoretic Text Hallucination Reduction for Video-grounded
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

**[47. [2310.08157] MCRepair: Multi-Chunk Program Repair via Patch Optimization with Buggy
  Block](https://arxiv.org/pdf/2310.08157.pdf)** (2023-11-08)

*Jisung Kim, Byeongjung Lee*

  Automated program repair (APR) is a technology that identifies and repairs
bugs automatically. However, repairing multi-chunk bugs remains a long-standing
and challenging problem because an APR technique must consider dependencies and
then reduce the large patch space. In addition, little is known about how to
combine individual candidate patches even though multi-chunk bugs require
combinations. Therefore, we propose a novel APR technique called multi-code
repair (MCRepair), which applies a buggy block, patch optimization, and
CodeBERT to target multi-chunk bugs. A buggy block is a novel method that binds
buggy chunks into a multi-buggy chunk and preprocesses the chunk with its buggy
contexts for patch space reduction and dependency problems. Patch optimization
is a novel strategy that effectively combines the generated candidate patches
with patch space reduction. In addition, CodeBERT, a BERT for source code
datasets, is fine-tuned to address the lack of datasets and out-of-vocabulary
problems. We conducted several experiments to evaluate our approach on six
project modules of Defects4J. In the experiments using Defects4J, MCRepair
repaired 65 bugs, including 21 multi-chunk bugs. Moreover, it fixed 18 unique
bugs, including eight multi-chunk bugs, and improved 40 to 250 percent
performance than the baselines.


---

**[48. [2504.08758] Hyper-RAG: Combating LLM Hallucinations using Hypergraph-Driven
  Retrieval-Augmented Generation](https://arxiv.org/pdf/2504.08758.pdf)** (2025-04-15)

*Yifan Feng, Hao Hu, Xingliang Hou, Shiquan Liu, Shihui Ying, Shaoyi Du, Han Hu, Yue Gao*

  Large language models (LLMs) have transformed various sectors, including
education, finance, and medicine, by enhancing content generation and
decision-making processes. However, their integration into the medical field is
cautious due to hallucinations, instances where generated content deviates from
factual accuracy, potentially leading to adverse outcomes. To address this, we
introduce Hyper-RAG, a hypergraph-driven Retrieval-Augmented Generation method
that comprehensively captures both pairwise and beyond-pairwise correlations in
domain-specific knowledge, thereby mitigating hallucinations. Experiments on
the NeurologyCrop dataset with six prominent LLMs demonstrated that Hyper-RAG
improves accuracy by an average of 12.3% over direct LLM use and outperforms
Graph RAG and Light RAG by 6.3% and 6.0%, respectively. Additionally, Hyper-RAG
maintained stable performance with increasing query complexity, unlike existing
methods which declined. Further validation across nine diverse datasets showed
a 35.5% performance improvement over Light RAG using a selection-based
assessment. The lightweight variant, Hyper-RAG-Lite, achieved twice the
retrieval speed and a 3.3% performance boost compared with Light RAG. These
results confirm Hyper-RAG's effectiveness in enhancing LLM reliability and
reducing hallucinations, making it a robust solution for high-stakes
applications like medical diagnostics.


---

**[49. [2410.00004] Retro-li: Small-Scale Retrieval Augmented Generation Supporting Noisy
  Similarity Searches and Domain Shift Generalization](https://arxiv.org/pdf/2410.00004.pdf)** (2025-03-27)

*Gentiana Rashiti, Geethan Karunaratne, Mrinmaya Sachan, Abu Sebastian, Abbas Rahimi*

  The retrieval augmented generation (RAG) system such as Retro has been shown
to improve language modeling capabilities and reduce toxicity and
hallucinations by retrieving from a database of non-parametric memory
containing trillions of entries. We introduce Retro-li that shows retrieval can
also help using a small-scale database, but it demands more accurate and better
neighbors when searching in a smaller hence sparser non-parametric memory. This
can be met by using a proper semantic similarity search. We further propose
adding a regularization to the non-parametric memory for the first time: it
significantly reduces perplexity when the neighbor search operations are noisy
during inference, and it improves generalization when a domain shift occurs. We
also show that Retro-li's non-parametric memory can potentially be implemented
on analog in-memory computing hardware, exhibiting O(1) search time while
causing noise in retrieving neighbors, with minimal (<1%) performance loss. Our
code is available at:
https://github.com/IBM/Retrieval-Enhanced-Transformer-Little.


---

**[50. [2410.10869] Application of NotebookLM, a Large Language Model with
  Retrieval-Augmented Generation, for Lung Cancer Staging](https://arxiv.org/pdf/2410.10869.pdf)** (2024-12-02)

*Ryota Tozuka, Hisashi Johno, Akitomo Amakawa, Junichi Sato, Mizuki Muto, Shoichiro Seki, Atsushi Komaba, Hiroshi Onishi*

  Purpose: In radiology, large language models (LLMs), including ChatGPT, have
recently gained attention, and their utility is being rapidly evaluated.
However, concerns have emerged regarding their reliability in clinical
applications due to limitations such as hallucinations and insufficient
referencing. To address these issues, we focus on the latest technology,
retrieval-augmented generation (RAG), which enables LLMs to reference reliable
external knowledge (REK). Specifically, this study examines the utility and
reliability of a recently released RAG-equipped LLM (RAG-LLM), NotebookLM, for
staging lung cancer.
  Materials and methods: We summarized the current lung cancer staging
guideline in Japan and provided this as REK to NotebookLM. We then tasked
NotebookLM with staging 100 fictional lung cancer cases based on CT findings
and evaluated its accuracy. For comparison, we performed the same task using a
gold-standard LLM, GPT-4 Omni (GPT-4o), both with and without the REK.
  Results: NotebookLM achieved 86% diagnostic accuracy in the lung cancer
staging experiment, outperforming GPT-4o, which recorded 39% accuracy with the
REK and 25% without it. Moreover, NotebookLM demonstrated 95% accuracy in
searching reference locations within the REK.
  Conclusion: NotebookLM successfully performed lung cancer staging by
utilizing the REK, demonstrating superior performance compared to GPT-4o.
Additionally, it provided highly accurate reference locations within the REK,
allowing radiologists to efficiently evaluate the reliability of NotebookLM's
responses and detect possible hallucinations. Overall, this study highlights
the potential of NotebookLM, a RAG-LLM, in image diagnosis.


---

**[51. [2009.09703] The High-Quality Wide Multi-Channel Attack (HQ-WMCA) database](https://arxiv.org/pdf/2009.09703.pdf)** (2020-09-22)

*Zohreh Mostaani, Anjith George, Guillaume Heusch, David Geissbuhler, Sebastien Marcel*

  The High-Quality Wide Multi-Channel Attack database (HQ-WMCA) database
extends the previous Wide Multi-Channel Attack database(WMCA), with more
channels including color, depth, thermal, infrared (spectra), and short-wave
infrared (spectra), and also a wide variety of attacks.


---

**[52. [2502.06864] Knowledge Graph-Guided Retrieval Augmented Generation](https://arxiv.org/pdf/2502.06864.pdf)** (2025-02-12)

*Xiangrong Zhu, Yuexiang Xie, Yi Liu, Yaliang Li, Wei Hu*

  Retrieval-augmented generation (RAG) has emerged as a promising technology
for addressing hallucination issues in the responses generated by large
language models (LLMs). Existing studies on RAG primarily focus on applying
semantic-based approaches to retrieve isolated relevant chunks, which ignore
their intrinsic relationships. In this paper, we propose a novel Knowledge
Graph-Guided Retrieval Augmented Generation (KG$^2$RAG) framework that utilizes
knowledge graphs (KGs) to provide fact-level relationships between chunks,
improving the diversity and coherence of the retrieved results. Specifically,
after performing a semantic-based retrieval to provide seed chunks, KG$^2$RAG
employs a KG-guided chunk expansion process and a KG-based chunk organization
process to deliver relevant and important knowledge in well-organized
paragraphs. Extensive experiments conducted on the HotpotQA dataset and its
variants demonstrate the advantages of KG$^2$RAG compared to existing RAG-based
approaches, in terms of both response quality and retrieval quality.


---

**[53. [2406.12449] Retrieval-Augmented Generation for Generative Artificial Intelligence in
  Medicine](https://arxiv.org/pdf/2406.12449.pdf)** (2024-06-19)

*Rui Yang, Yilin Ning, Emilia Keppo, Mingxuan Liu, Chuan Hong, Danielle S Bitterman, Jasmine Chiat Ling Ong, Daniel Shu Wei Ting, Nan Liu*

  Generative artificial intelligence (AI) has brought revolutionary innovations
in various fields, including medicine. However, it also exhibits limitations.
In response, retrieval-augmented generation (RAG) provides a potential
solution, enabling models to generate more accurate contents by leveraging the
retrieval of external knowledge. With the rapid advancement of generative AI,
RAG can pave the way for connecting this transformative technology with medical
applications and is expected to bring innovations in equity, reliability, and
personalization to health care.


---

**[54. [2402.12177] Mafin: Enhancing Black-Box Embeddings with Model Augmented Fine-Tuning](https://arxiv.org/pdf/2402.12177.pdf)** (2024-03-13)

*Mingtian Zhang, Shawn Lan, Peter Hayes, David Barber*

  Retrieval Augmented Generation (RAG) has emerged as an effective solution for
mitigating hallucinations in Large Language Models (LLMs). The retrieval stage
in RAG typically involves a pre-trained embedding model, which converts queries
and passages into vectors to capture their semantics. However, a standard
pre-trained embedding model may exhibit sub-optimal performance when applied to
specific domain knowledge, necessitating fine-tuning. This paper addresses
scenarios where the embeddings are only available from a black-box model. We
introduce Model augmented fine-tuning (Mafin) -- a novel approach for
fine-tuning a black-box embedding model by augmenting it with a trainable
embedding model. Our results demonstrate that Mafin significantly enhances the
performance of the black-box embeddings by only requiring the training of a
small augmented model. We validate the effectiveness of our method on both
labeled and unlabeled datasets, illustrating its broad applicability and
efficiency.


---

**[55. [2410.10594] VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality
  Documents](https://arxiv.org/pdf/2410.10594.pdf)** (2025-03-04)

*Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang, Xu Han, Zhiyuan Liu, Maosong Sun*

  Retrieval-augmented generation (RAG) is an effective technique that enables
large language models (LLMs) to utilize external knowledge sources for
generation. However, current RAG systems are solely based on text, rendering it
impossible to utilize vision information like layout and images that play
crucial roles in real-world multi-modality documents. In this paper, we
introduce VisRAG, which tackles this issue by establishing a vision-language
model (VLM)-based RAG pipeline. In this pipeline, instead of first parsing the
document to obtain text, the document is directly embedded using a VLM as an
image and then retrieved to enhance the generation of a VLM. Compared to
traditional text-based RAG, VisRAG maximizes the retention and utilization of
the data information in the original documents, eliminating the information
loss introduced during the parsing process. We collect both open-source and
synthetic data to train the retriever in VisRAG and explore a variety of
generation methods. Experiments demonstrate that VisRAG outperforms traditional
RAG in both the retrieval and generation stages, achieving a 20--40% end-to-end
performance gain over traditional text-based RAG pipeline. Further analysis
reveals that VisRAG is efficient in utilizing training data and demonstrates
strong generalization capability, positioning it as a promising solution for
RAG on multi-modality documents. Our code and data are available at
https://github.com/openbmb/visrag.


---

**[56. [2501.03468] MTRAG: A Multi-Turn Conversational Benchmark for Evaluating
  Retrieval-Augmented Generation Systems](https://arxiv.org/pdf/2501.03468.pdf)** (2025-01-08)

*Yannis Katsis, Sara Rosenthal, Kshitij Fadnis, Chulaka Gunasekara, Young-Suk Lee, Lucian Popa, Vraj Shah, Huaiyu Zhu, Danish Contractor, Marina Danilevsky*

  Retrieval-augmented generation (RAG) has recently become a very popular task
for Large Language Models (LLMs). Evaluating them on multi-turn RAG
conversations, where the system is asked to generate a response to a question
in the context of a preceding conversation is an important and often overlooked
task with several additional challenges. We present MTRAG: an end-to-end
human-generated multi-turn RAG benchmark that reflects several real-world
properties across diverse dimensions for evaluating the full RAG pipeline.
MTRAG contains 110 conversations averaging 7.7 turns each across four domains
for a total of 842 tasks. We also explore automation paths via synthetic data
and LLM-as-a-Judge evaluation. Our human and automatic evaluations show that
even state-of-the-art LLM RAG systems struggle on MTRAG. We demonstrate the
need for strong retrieval and generation systems that can handle later turns,
unanswerable questions, non-standalone questions, and multiple domains. MTRAG
is available at https://github.com/ibm/mt-rag-benchmark.


---

**[57. [2410.19572] ChunkRAG: Novel LLM-Chunk Filtering Method for RAG Systems](https://arxiv.org/pdf/2410.19572.pdf)** (2024-11-20)

*Ishneet Sukhvinder Singh, Ritvik Aggarwal, Ibrahim Allahverdiyev, Muhammad Taha, Aslihan Akalin, Kevin Zhu, Sean O'Brien*

  Retrieval-Augmented Generation (RAG) systems using large language models
(LLMs) often generate inaccurate responses due to the retrieval of irrelevant
or loosely related information. Existing methods, which operate at the document
level, fail to effectively filter out such content. We propose LLM-driven chunk
filtering, ChunkRAG, a framework that enhances RAG systems by evaluating and
filtering retrieved information at the chunk level. Our approach employs
semantic chunking to divide documents into coherent sections and utilizes
LLM-based relevance scoring to assess each chunk's alignment with the user's
query. By filtering out less pertinent chunks before the generation phase, we
significantly reduce hallucinations and improve factual accuracy. Experiments
show that our method outperforms existing RAG models, achieving higher accuracy
on tasks requiring precise information retrieval. This advancement enhances the
reliability of RAG systems, making them particularly beneficial for
applications like fact-checking and multi-hop reasoning.


---

**[58. [2411.19528] RAGDiffusion: Faithful Cloth Generation via External Knowledge
  Assimilation](https://arxiv.org/pdf/2411.19528.pdf)** (2024-12-02)

*Xianfeng Tan, Yuhan Li, Wenxiang Shang, Yubo Wu, Jian Wang, Xuanhong Chen, Yi Zhang, Ran Lin, Bingbing Ni*

  Standard clothing asset generation involves creating forward-facing flat-lay
garment images displayed on a clear background by extracting clothing
information from diverse real-world contexts, which presents significant
challenges due to highly standardized sampling distributions and precise
structural requirements in the generated images. Existing models have limited
spatial perception and often exhibit structural hallucinations in this
high-specification generative task. To address this issue, we propose a novel
Retrieval-Augmented Generation (RAG) framework, termed RAGDiffusion, to enhance
structure determinacy and mitigate hallucinations by assimilating external
knowledge from LLM and databases. RAGDiffusion consists of two core processes:
(1) Retrieval-based structure aggregation, which employs contrastive learning
and a Structure Locally Linear Embedding (SLLE) to derive global structure and
spatial landmarks, providing both soft and hard guidance to counteract
structural ambiguities; and (2) Omni-level faithful garment generation, which
introduces a three-level alignment that ensures fidelity in structural,
pattern, and decoding components within the diffusing. Extensive experiments on
challenging real-world datasets demonstrate that RAGDiffusion synthesizes
structurally and detail-faithful clothing assets with significant performance
improvements, representing a pioneering effort in high-specification faithful
generation with RAG to confront intrinsic hallucinations and enhance fidelity.


---

**[59. [2410.09962] LongHalQA: Long-Context Hallucination Evaluation for MultiModal Large
  Language Models](https://arxiv.org/pdf/2410.09962.pdf)** (2024-10-16)

*Han Qiu, Jiaxing Huang, Peng Gao, Qin Qi, Xiaoqin Zhang, Ling Shao, Shijian Lu*

  Hallucination, a phenomenon where multimodal large language models~(MLLMs)
tend to generate textual responses that are plausible but unaligned with the
image, has become one major hurdle in various MLLM-related applications.
Several benchmarks have been created to gauge the hallucination levels of
MLLMs, by either raising discriminative questions about the existence of
objects or introducing LLM evaluators to score the generated text from MLLMs.
However, the discriminative data largely involve simple questions that are not
aligned with real-world text, while the generative data involve LLM evaluators
that are computationally intensive and unstable due to their inherent
randomness. We propose LongHalQA, an LLM-free hallucination benchmark that
comprises 6K long and complex hallucination text. LongHalQA is featured by
GPT4V-generated hallucinatory data that are well aligned with real-world
scenarios, including object/image descriptions and multi-round conversations
with 14/130 words and 189 words, respectively, on average. It introduces two
new tasks, hallucination discrimination and hallucination completion, unifying
both discriminative and generative evaluations in a single
multiple-choice-question form and leading to more reliable and efficient
evaluations without the need for LLM evaluators. Further, we propose an
advanced pipeline that greatly facilitates the construction of future
hallucination benchmarks with long and complex questions and descriptions.
Extensive experiments over multiple recent MLLMs reveal various new challenges
when they are handling hallucinations with long and complex textual data.
Dataset and evaluation code are available at
https://github.com/hanqiu-hq/LongHalQA.


---

**[60. [2503.23895] Better wit than wealth: Dynamic Parametric Retrieval Augmented
  Generation for Test-time Knowledge Enhancement](https://arxiv.org/pdf/2503.23895.pdf)** (2025-04-01)

*Yuqiao Tan, Shizhu He, Huanxuan Liao, Jun Zhao, Kang Liu*

  Retrieval-augmented generation (RAG) enhances large language models (LLMs) by
retrieving relevant documents from external sources and incorporating them into
the context. While it improves reliability by providing factual texts, it
significantly increases inference costs as context length grows and introduces
challenging issue of RAG hallucination, primarily caused by the lack of
corresponding parametric knowledge in LLMs. An efficient solution is to enhance
the knowledge of LLMs at test-time. Parametric RAG (PRAG) addresses this by
embedding document into LLMs parameters to perform test-time knowledge
enhancement, effectively reducing inference costs through offline training.
However, its high training and storage costs, along with limited generalization
ability, significantly restrict its practical adoption. To address these
challenges, we propose Dynamic Parametric RAG (DyPRAG), a novel framework that
leverages a lightweight parameter translator model to efficiently convert
documents into parametric knowledge. DyPRAG not only reduces inference,
training, and storage costs but also dynamically generates parametric
knowledge, seamlessly enhancing the knowledge of LLMs and resolving knowledge
conflicts in a plug-and-play manner at test-time. Extensive experiments on
multiple datasets demonstrate the effectiveness and generalization capabilities
of DyPRAG, offering a powerful and practical RAG paradigm which enables
superior knowledge fusion and mitigates RAG hallucination in real-world
applications. Our code is available at https://github.com/Trae1ounG/DyPRAG.


---

**[61. [2409.09916] SFR-RAG: Towards Contextually Faithful LLMs](https://arxiv.org/pdf/2409.09916.pdf)** (2024-09-17)

*Xuan-Phi Nguyen, Shrey Pandit, Senthil Purushwalkam, Austin Xu, Hailin Chen, Yifei Ming, Zixuan Ke, Silvio Savarese, Caiming Xong, Shafiq Joty*

  Retrieval Augmented Generation (RAG), a paradigm that integrates external
contextual information with large language models (LLMs) to enhance factual
accuracy and relevance, has emerged as a pivotal area in generative AI. The
LLMs used in RAG applications are required to faithfully and completely
comprehend the provided context and users' questions, avoid hallucination,
handle unanswerable, counterfactual or otherwise low-quality and irrelevant
contexts, perform complex multi-hop reasoning and produce reliable citations.
In this paper, we introduce SFR-RAG, a small LLM that is instruction-tuned with
an emphasis on context-grounded generation and hallucination minimization. We
also present ContextualBench, a new evaluation framework compiling multiple
popular and diverse RAG benchmarks, such as HotpotQA and TriviaQA, with
consistent RAG settings to ensure reproducibility and consistency in model
assessments. Experimental results demonstrate that our SFR-RAG-9B model
outperforms leading baselines such as Command-R+ (104B) and GPT-4o, achieving
state-of-the-art results in 3 out of 7 benchmarks in ContextualBench with
significantly fewer parameters. The model is also shown to be resilient to
alteration in the contextual information and behave appropriately when relevant
context is removed. Additionally, the SFR-RAG model maintains competitive
performance in general instruction-following tasks and function-calling
capabilities.


---

**[62. [2410.08431] oRetrieval Augmented Generation for 10 Large Language Models and its
  Generalizability in Assessing Medical Fitness](https://arxiv.org/pdf/2410.08431.pdf)** (2024-10-14)

*Yu He Ke, Liyuan Jin, Kabilan Elangovan, Hairil Rizal Abdullah, Nan Liu, Alex Tiong Heng Sia, Chai Rick Soh, Joshua Yi Min Tung, Jasmine Chiat Ling Ong, Chang-Fu Kuo, Shao-Chun Wu, Vesela P. Kovacheva, Daniel Shu Wei Ting*

  Large Language Models (LLMs) show potential for medical applications but
often lack specialized clinical knowledge. Retrieval Augmented Generation (RAG)
allows customization with domain-specific information, making it suitable for
healthcare. This study evaluates the accuracy, consistency, and safety of RAG
models in determining fitness for surgery and providing preoperative
instructions. We developed LLM-RAG models using 35 local and 23 international
preoperative guidelines and tested them against human-generated responses. A
total of 3,682 responses were evaluated. Clinical documents were processed
using Llamaindex, and 10 LLMs, including GPT3.5, GPT4, and Claude-3, were
assessed. Fourteen clinical scenarios were analyzed, focusing on seven aspects
of preoperative instructions. Established guidelines and expert judgment were
used to determine correct responses, with human-generated answers serving as
comparisons. The LLM-RAG models generated responses within 20 seconds,
significantly faster than clinicians (10 minutes). The GPT4 LLM-RAG model
achieved the highest accuracy (96.4% vs. 86.6%, p=0.016), with no
hallucinations and producing correct instructions comparable to clinicians.
Results were consistent across both local and international guidelines. This
study demonstrates the potential of LLM-RAG models for preoperative healthcare
tasks, highlighting their efficiency, scalability, and reliability.


---

**[63. [2412.05447] TOBUGraph: Knowledge Graph-Based Retrieval for Enhanced LLM Performance
  Beyond RAG](https://arxiv.org/pdf/2412.05447.pdf)** (2025-04-02)

*Savini Kashmira, Jayanaka L. Dantanarayana, Joshua Brodsky, Ashish Mahendra, Yiping Kang, Krisztian Flautner, Lingjia Tang, Jason Mars*

  Retrieval-Augmented Generation (RAG) is one of the leading and most widely
used techniques for enhancing LLM retrieval capabilities, but it still faces
significant limitations in commercial use cases. RAG primarily relies on the
query-chunk text-to-text similarity in the embedding space for retrieval and
can fail to capture deeper semantic relationships across chunks, is highly
sensitive to chunking strategies, and is prone to hallucinations. To address
these challenges, we propose TOBUGraph, a graph-based retrieval framework that
first constructs the knowledge graph from unstructured data dynamically and
automatically. Using LLMs, TOBUGraph extracts structured knowledge and diverse
relationships among data, going beyond RAG's text-to-text similarity. Retrieval
is achieved through graph traversal, leveraging the extracted relationships and
structures to enhance retrieval accuracy, eliminating the need for chunking
configurations while reducing hallucination. We demonstrate TOBUGraph's
effectiveness in TOBU, a real-world application in production for personal
memory organization and retrieval. Our evaluation using real user data
demonstrates that TOBUGraph outperforms multiple RAG implementations in both
precision and recall, significantly improving user experience through improved
retrieval accuracy.


---

**[64. [2204.02548] Style-Hallucinated Dual Consistency Learning for Domain Generalized
  Semantic Segmentation](https://arxiv.org/pdf/2204.02548.pdf)** (2022-07-20)

*Yuyang Zhao, Zhun Zhong, Na Zhao, Nicu Sebe, Gim Hee Lee*

  In this paper, we study the task of synthetic-to-real domain generalized
semantic segmentation, which aims to learn a model that is robust to unseen
real-world scenes using only synthetic data. The large domain shift between
synthetic and real-world data, including the limited source environmental
variations and the large distribution gap between synthetic and real-world
data, significantly hinders the model performance on unseen real-world scenes.
In this work, we propose the Style-HAllucinated Dual consistEncy learning
(SHADE) framework to handle such domain shift. Specifically, SHADE is
constructed based on two consistency constraints, Style Consistency (SC) and
Retrospection Consistency (RC). SC enriches the source situations and
encourages the model to learn consistent representation across
style-diversified samples. RC leverages real-world knowledge to prevent the
model from overfitting to synthetic data and thus largely keeps the
representation consistent between the synthetic and real-world models.
Furthermore, we present a novel style hallucination module (SHM) to generate
style-diversified samples that are essential to consistency learning. SHM
selects basis styles from the source distribution, enabling the model to
dynamically generate diverse and realistic samples during training. Experiments
show that our SHADE yields significant improvement and outperforms
state-of-the-art methods by 5.05% and 8.35% on the average mIoU of three
real-world datasets on single- and multi-source settings, respectively.


---

**[65. [2403.19113] FACTOID: FACtual enTailment fOr hallucInation Detection](https://arxiv.org/pdf/2403.19113.pdf)** (2024-03-29)

*Vipula Rawte, S. M Towhidul Islam Tonmoy, Krishnav Rajbangshi, Shravani Nag, Aman Chadha, Amit P. Sheth, Amitava Das*

  The widespread adoption of Large Language Models (LLMs) has facilitated
numerous benefits. However, hallucination is a significant concern. In
response, Retrieval Augmented Generation (RAG) has emerged as a highly
promising paradigm to improve LLM outputs by grounding them in factual
information. RAG relies on textual entailment (TE) or similar methods to check
if the text produced by LLMs is supported or contradicted, compared to
retrieved documents. This paper argues that conventional TE methods are
inadequate for spotting hallucinations in content generated by LLMs. For
instance, consider a prompt about the 'USA's stance on the Ukraine war''. The
AI-generated text states, ...U.S. President Barack Obama says the U.S. will not
put troops in Ukraine...'' However, during the war the U.S. president is Joe
Biden which contradicts factual reality. Moreover, current TE systems are
unable to accurately annotate the given text and identify the exact portion
that is contradicted. To address this, we introduces a new type of TE called
``Factual Entailment (FE).'', aims to detect factual inaccuracies in content
generated by LLMs while also highlighting the specific text segment that
contradicts reality. We present FACTOID (FACTual enTAILment for hallucInation
Detection), a benchmark dataset for FE. We propose a multi-task learning (MTL)
framework for FE, incorporating state-of-the-art (SoTA) long text embeddings
such as e5-mistral-7b-instruct, along with GPT-3, SpanBERT, and RoFormer. The
proposed MTL architecture for FE achieves an avg. 40\% improvement in accuracy
on the FACTOID benchmark compared to SoTA TE methods. As FE automatically
detects hallucinations, we assessed 15 modern LLMs and ranked them using our
proposed Auto Hallucination Vulnerability Index (HVI_auto). This index
quantifies and offers a comparative scale to evaluate and rank LLMs according
to their hallucinations.


---

**[66. [2502.09073] Enhancing RAG with Active Learning on Conversation Records: Reject
  Incapables and Answer Capables](https://arxiv.org/pdf/2502.09073.pdf)** (2025-02-14)

*Xuzhao Geng, Haozhao Wang, Jun Wang, Wei Liu, Ruixuan Li*

  Retrieval-augmented generation (RAG) is a key technique for leveraging
external knowledge and reducing hallucinations in large language models (LLMs).
However, RAG still struggles to fully prevent hallucinated responses. To
address this, it is essential to identify samples prone to hallucination or
guide LLMs toward correct responses, which experts then annotate to develop
high-quality datasets for refining LLMs. However, the growing scarcity of such
datasets makes their creation challenging. This paper proposes using the vast
amount of conversations from widespread LLM usage to build these datasets,
training LLMs to avoid hallucination-prone questions while accurately
responding to manageable ones. Given the impracticality of expert-annotating
all conversation records, the paper introduces AL4RAG, which uses active
learning to select the most suitable conversation samples for annotation,
optimizing performance within an annotation budget. Additionally, recognizing
that traditional active learning methods are not fully compatible with RAG due
to unsuitable distance metrics, we develop a novel sample distance measurement
for RAG active learning. Extensive experiments show that our method
consistently outperforms baselines across multiple metrics.


---

**[67. [2409.15566] GEM-RAG: Graphical Eigen Memories For Retrieval Augmented Generation](https://arxiv.org/pdf/2409.15566.pdf)** (2024-09-25)

*Brendan Hogan Rappazzo, Yingheng Wang, Aaron Ferber, Carla Gomes*

  The ability to form, retrieve, and reason about memories in response to
stimuli serves as the cornerstone for general intelligence - shaping entities
capable of learning, adaptation, and intuitive insight. Large Language Models
(LLMs) have proven their ability, given the proper memories or context, to
reason and respond meaningfully to stimuli. However, they are still unable to
optimally encode, store, and retrieve memories - the ability to do this would
unlock their full ability to operate as AI agents, and to specialize to niche
domains. To remedy this, one promising area of research is Retrieval Augmented
Generation (RAG), which aims to augment LLMs by providing them with rich
in-context examples and information. In question-answering (QA) applications,
RAG methods embed the text of interest in chunks, and retrieve the most
relevant chunks for a prompt using text embeddings. Motivated by human memory
encoding and retrieval, we aim to improve over standard RAG methods by
generating and encoding higher-level information and tagging the chunks by
their utility to answer questions. We introduce Graphical Eigen Memories For
Retrieval Augmented Generation (GEM-RAG). GEM-RAG works by tagging each chunk
of text in a given text corpus with LLM generated ``utility'' questions,
connecting chunks in a graph based on the similarity of both their text and
utility questions, and then using the eigendecomposition of the memory graph to
build higher level summary nodes that capture the main themes of the text. We
evaluate GEM-RAG, using both UnifiedQA and GPT-3.5 Turbo as the LLMs, with
SBERT, and OpenAI's text encoders on two standard QA tasks, showing that
GEM-RAG outperforms other state-of-the-art RAG methods on these tasks. We also
discuss the implications of having a robust RAG system and future directions.


---

**[68. [2312.11361] "Knowing When You Don't Know": A Multilingual Relevance Assessment
  Dataset for Robust Retrieval-Augmented Generation](https://arxiv.org/pdf/2312.11361.pdf)** (2024-11-12)

*Nandan Thakur, Luiz Bonifacio, Xinyu Zhang, Odunayo Ogundepo, Ehsan Kamalloo, David Alfonso-Hermelo, Xiaoguang Li, Qun Liu, Boxing Chen, Mehdi Rezagholizadeh, Jimmy Lin*

  Retrieval-Augmented Generation (RAG) grounds Large Language Model (LLM)
output by leveraging external knowledge sources to reduce factual
hallucinations. However, prior work lacks a comprehensive evaluation of
different language families, making it challenging to evaluate LLM robustness
against errors in external retrieved knowledge. To overcome this, we establish
NoMIRACL, a human-annotated dataset for evaluating LLM robustness in RAG across
18 typologically diverse languages. NoMIRACL includes both a non-relevant and a
relevant subset. Queries in the non-relevant subset contain passages judged as
non-relevant, whereas queries in the relevant subset include at least a single
judged relevant passage. We measure relevance assessment using: (i)
hallucination rate, measuring model tendency to hallucinate, when the answer is
not present in passages in the non-relevant subset, and (ii) error rate,
measuring model inaccuracy to recognize relevant passages in the relevant
subset.In our work, we observe that most models struggle to balance the two
capacities. Models such as LLAMA-2 and Orca-2 achieve over 88% hallucination
rate on the non-relevant subset. Mistral and LLAMA-3 hallucinate less but can
achieve up to a 74.9% error rate on the relevant subset. Overall, GPT-4 is
observed to provide the best tradeoff on both subsets, highlighting future work
necessary to improve LLM robustness. NoMIRACL dataset and evaluation code are
available at: https://github.com/project-miracl/nomiracl.


---

**[69. [2504.06475] Successive randomized compression: A randomized algorithm for the
  compressed MPO-MPS product](https://arxiv.org/pdf/2504.06475.pdf)** (2025-04-10)

*Chris Camaño, Ethan N. Epperly, Joel A. Tropp*

  Tensor networks like matrix product states (MPSs) and matrix product
operators (MPOs) are powerful tools for representing exponentially large states
and operators, with applications in quantum many-body physics, machine
learning, numerical analysis, and other areas. In these applications, computing
a compressed representation of the MPO--MPS product is a fundamental
computational primitive. For this operation, this paper introduces a new
single-pass, randomized algorithm, called successive randomized compression
(SRC), that improves on existing approaches in speed or in accuracy. The
performance of the new algorithm is evaluated on synthetic problems and unitary
time evolution problems for quantum spin systems.


---

**[70. [2402.09906] Generative Representational Instruction Tuning](https://arxiv.org/pdf/2402.09906.pdf)** (2025-03-04)

*Niklas Muennighoff, Hongjin Su, Liang Wang, Nan Yang, Furu Wei, Tao Yu, Amanpreet Singh, Douwe Kiela*

  All text-based language problems can be reduced to either generation or
embedding. Current models only perform well at one or the other. We introduce
generative representational instruction tuning (GRIT) whereby a large language
model is trained to handle both generative and embedding tasks by
distinguishing between them through instructions. Compared to other open
models, our resulting GritLM 7B sets a new state of the art on the Massive Text
Embedding Benchmark (MTEB) and outperforms all models up to its size on a range
of generative tasks. By scaling up further, GritLM 8x7B outperforms all open
generative language models that we tried while still being among the best
embedding models. Notably, we find that GRIT matches training on only
generative or embedding data, thus we can unify both at no performance loss.
Among other benefits, the unification via GRIT speeds up Retrieval-Augmented
Generation (RAG) by > 60% for long documents, by no longer requiring separate
retrieval and generation models. Models, code, etc. are freely available at
https://github.com/ContextualAI/gritlm.


---

**[71. [2502.20034] Vision-Encoders (Already) Know What They See: Mitigating Object
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

**[72. [2411.18948] RevPRAG: Revealing Poisoning Attacks in Retrieval-Augmented Generation
  through LLM Activation Analysis](https://arxiv.org/pdf/2411.18948.pdf)** (2025-02-20)

*Xue Tan, Hao Luan, Mingyu Luo, Xiaoyan Sun, Ping Chen, Jun Dai*

  Retrieval-Augmented Generation (RAG) enriches the input to LLMs by retrieving
information from the relevant knowledge database, enabling them to produce
responses that are more accurate and contextually appropriate. It is worth
noting that the knowledge database, being sourced from publicly available
channels such as Wikipedia, inevitably introduces a new attack surface. RAG
poisoning involves injecting malicious texts into the knowledge database,
ultimately leading to the generation of the attacker's target response (also
called poisoned response). However, there are currently limited methods
available for detecting such poisoning attacks. We aim to bridge the gap in
this work. Particularly, we introduce RevPRAG, a flexible and automated
detection pipeline that leverages the activations of LLMs for poisoned response
detection. Our investigation uncovers distinct patterns in LLMs' activations
when generating correct responses versus poisoned responses. Our results on
multiple benchmark datasets and RAG architectures show our approach could
achieve 98% true positive rate, while maintaining false positive rates close to
1%.


---

**[73. [2409.01151] Understanding Multimodal Hallucination with Parameter-Free
  Representation Alignment](https://arxiv.org/pdf/2409.01151.pdf)** (2024-09-04)

*Yueqian Wang, Jianxin Liang, Yuxuan Wang, Huishuai Zhang, Dongyan Zhao*

  Hallucination is a common issue in Multimodal Large Language Models (MLLMs),
yet the underlying principles remain poorly understood. In this paper, we
investigate which components of MLLMs contribute to object hallucinations. To
analyze image representations while completely avoiding the influence of all
other factors other than the image representation itself, we propose a
parametric-free representation alignment metric (Pfram) that can measure the
similarities between any two representation systems without requiring
additional training parameters. Notably, Pfram can also assess the alignment of
a neural representation system with the human representation system,
represented by ground-truth annotations of images. By evaluating the alignment
with object annotations, we demonstrate that this metric shows strong and
consistent correlations with object hallucination across a wide range of
state-of-the-art MLLMs, spanning various model architectures and sizes.
Furthermore, using this metric, we explore other key issues related to image
representations in MLLMs, such as the role of different modules, the impact of
textual instructions, and potential improvements including the use of
alternative visual encoders. Our code is available at:
https://github.com/yellow-binary-tree/Pfram.


---

**[74. [2403.15048] Make VLM Recognize Visual Hallucination on Cartoon Character Image with
  Pose Information](https://arxiv.org/pdf/2403.15048.pdf)** (2025-01-23)

*Bumsoo Kim, Wonseop Shin, Kyuchul Lee, Yonghoon Jung, Sanghyun Seo*

  Leveraging large-scale Text-to-Image (TTI) models have become a common
technique for generating exemplar or training dataset in the fields of image
synthesis, video editing, 3D reconstruction. However, semantic structural
visual hallucinations involving perceptually severe defects remain a concern,
especially in the domain of non-photorealistic rendering (NPR) such as cartoons
and pixelization-style character. To detect these hallucinations in NPR, We
propose a novel semantic structural hallucination detection system using
Vision-Language Model (VLM). Our approach is to leverage the emerging
capability of large language model, in-context learning which denotes that VLM
has seen some examples by user for specific downstream task, here hallucination
detection. Based on in-context learning, we introduce pose-aware in-context
visual learning (PA-ICVL) which improve the overall performance of VLM by
further inputting visual data beyond prompts, RGB images and pose information.
By incorporating pose guidance, we enable VLMs to make more accurate decisions.
Experimental results demonstrate significant improvements in identifying visual
hallucinations compared to baseline methods relying solely on RGB images.
Within selected two VLMs, GPT-4v, Gemini pro vision, our proposed PA-ICVL
improves the hallucination detection with 50% to 78%, 57% to 80%, respectively.
This research advances a capability of TTI models toward real-world
applications by mitigating visual hallucinations via in-context visual
learning, expanding their potential in non-photorealistic domains. In addition,
it showcase how users can boost the downstream-specialized capability of open
VLM by harnessing additional conditions. We collect synthetic
cartoon-hallucination dataset with TTI models, this dataset and final tuned VLM
will be publicly available.


---

**[75. [2405.19285] MASSIVE Multilingual Abstract Meaning Representation: A Dataset and
  Baselines for Hallucination Detection](https://arxiv.org/pdf/2405.19285.pdf)** (2024-05-30)

*Michael Regan, Shira Wein, George Baker, Emilio Monti*

  Abstract Meaning Representation (AMR) is a semantic formalism that captures
the core meaning of an utterance. There has been substantial work developing
AMR corpora in English and more recently across languages, though the limited
size of existing datasets and the cost of collecting more annotations are
prohibitive. With both engineering and scientific questions in mind, we
introduce MASSIVE-AMR, a dataset with more than 84,000 text-to-graph
annotations, currently the largest and most diverse of its kind: AMR graphs for
1,685 information-seeking utterances mapped to 50+ typologically diverse
languages. We describe how we built our resource and its unique features before
reporting on experiments using large language models for multilingual AMR and
SPARQL parsing as well as applying AMRs for hallucination detection in the
context of knowledge base question answering, with results shedding light on
persistent issues using LLMs for structured parsing.


---

**[76. [2502.15040] Reducing Hallucinations of Medical Multimodal Large Language Models with
  Visual Retrieval-Augmented Generation](https://arxiv.org/pdf/2502.15040.pdf)** (2025-02-24)

*Yun-Wei Chu, Kai Zhang, Christopher Malon, Martin Renqiang Min*

  Multimodal Large Language Models (MLLMs) have shown impressive performance in
vision and text tasks. However, hallucination remains a major challenge,
especially in fields like healthcare where details are critical. In this work,
we show how MLLMs may be enhanced to support Visual RAG (V-RAG), a
retrieval-augmented generation framework that incorporates both text and visual
data from retrieved images. On the MIMIC-CXR chest X-ray report generation and
Multicare medical image caption generation datasets, we show that Visual RAG
improves the accuracy of entity probing, which asks whether a medical entities
is grounded by an image. We show that the improvements extend both to frequent
and rare entities, the latter of which may have less positive training data.
Downstream, we apply V-RAG with entity probing to correct hallucinations and
generate more clinically accurate X-ray reports, obtaining a higher RadGraph-F1
score.


---

**[77. [2409.09337] Wave-U-Mamba: An End-To-End Framework For High-Quality And Efficient
  Speech Super Resolution](https://arxiv.org/pdf/2409.09337.pdf)** (2025-02-04)

*Yongjoon Lee, Chanwoo Kim*

  Speech Super-Resolution (SSR) is a task of enhancing low-resolution speech
signals by restoring missing high-frequency components. Conventional approaches
typically reconstruct log-mel features, followed by a vocoder that generates
high-resolution speech in the waveform domain. However, as mel features lack
phase information, this can result in performance degradation during the
reconstruction phase. Motivated by recent advances with Selective State Spaces
Models (SSMs), we propose a method, referred to as Wave-U-Mamba that directly
performs SSR in time domain. In our comparative study, including models such as
WSRGlow, NU-Wave 2, and AudioSR, Wave-U-Mamba demonstrates superior
performance, achieving the lowest Log-Spectral Distance (LSD) across various
low-resolution sampling rates, ranging from 8 to 24 kHz. Additionally,
subjective human evaluations, scored using Mean Opinion Score (MOS) reveal that
our method produces SSR with natural and human-like quality. Furthermore,
Wave-U-Mamba achieves these results while generating high-resolution speech
over nine times faster than baseline models on a single A100 GPU, with
parameter sizes less than 2\% of those in the baseline models.


---

**[78. [2410.11701] Magnifier Prompt: Tackling Multimodal Hallucination via Extremely Simple
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

**[79. [2210.02627] Improving the Domain Adaptation of Retrieval Augmented Generation (RAG)
  Models for Open Domain Question Answering](https://arxiv.org/pdf/2210.02627.pdf)** (2024-10-08)

*Shamane Siriwardhana, Rivindu Weerasekera, Elliott Wen, Tharindu Kaluarachchi, Rajib Rana, Suranga Nanayakkara*

  Retrieval Augment Generation (RAG) is a recent advancement in Open-Domain
Question Answering (ODQA). RAG has only been trained and explored with a
Wikipedia-based external knowledge base and is not optimized for use in other
specialized domains such as healthcare and news. In this paper, we evaluate the
impact of joint training of the retriever and generator components of RAG for
the task of domain adaptation in ODQA. We propose \textit{RAG-end2end}, an
extension to RAG, that can adapt to a domain-specific knowledge base by
updating all components of the external knowledge base during training. In
addition, we introduce an auxiliary training signal to inject more
domain-specific knowledge. This auxiliary signal forces \textit{RAG-end2end} to
reconstruct a given sentence by accessing the relevant information from the
external knowledge base. Our novel contribution is unlike RAG, RAG-end2end does
joint training of the retriever and generator for the end QA task and domain
adaptation. We evaluate our approach with datasets from three domains:
COVID-19, News, and Conversations, and achieve significant performance
improvements compared to the original RAG model. Our work has been open-sourced
through the Huggingface Transformers library, attesting to our work's
credibility and technical consistency.


---

**[80. [2410.20753] Plan*RAG: Efficient Test-Time Planning for Retrieval Augmented
  Generation](https://arxiv.org/pdf/2410.20753.pdf)** (2025-02-05)

*Prakhar Verma, Sukruta Prakash Midigeshi, Gaurav Sinha, Arno Solin, Nagarajan Natarajan, Amit Sharma*

  We introduce Plan*RAG, a novel framework that enables structured multi-hop
reasoning in retrieval-augmented generation (RAG) through test-time reasoning
plan generation. While existing approaches such as ReAct maintain reasoning
chains within the language model's context window, we observe that this often
leads to plan fragmentation and execution failures. Our key insight is that by
isolating the reasoning plan as a directed acyclic graph (DAG) outside the LM's
working memory, we can enable (1) systematic exploration of reasoning paths,
(2) atomic subqueries enabling precise retrievals and grounding, and (3)
efficiency through parallel execution and bounded context window utilization.
Moreover, Plan*RAG's modular design allows it to be integrated with existing
RAG methods, thus providing a practical solution to improve current RAG
systems. On standard multi-hop reasoning benchmarks, Plan*RAG consistently
achieves improvements over recently proposed methods such as RQ-RAG and
Self-RAG, while maintaining comparable computational costs.


---

**[81. [2411.04847] Prompt-Guided Internal States for Hallucination Detection of Large
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

**[82. [2410.19493] Conditional Hallucinations for Image Compression](https://arxiv.org/pdf/2410.19493.pdf)** (2025-03-07)

*Till Aczel, Roger Wattenhofer*

  In lossy image compression, models face the challenge of either hallucinating
details or generating out-of-distribution samples due to the information
bottleneck. This implies that at times, introducing hallucinations is necessary
to generate in-distribution samples. The optimal level of hallucination varies
depending on image content, as humans are sensitive to small changes that alter
the semantic meaning. We propose a novel compression method that dynamically
balances the degree of hallucination based on content. We collect data and
train a model to predict user preferences on hallucinations. By using this
prediction to adjust the perceptual weight in the reconstruction loss, we
develop a Conditionally Hallucinating compression model (ConHa) that
outperforms state-of-the-art image compression methods. Code and images are
available at https://polybox.ethz.ch/index.php/s/owS1k5JYs4KD4TA.


---

**[83. [2402.09717] Visually Dehallucinative Instruction Generation: Know What You Don't
  Know](https://arxiv.org/pdf/2402.09717.pdf)** (2024-02-16)

*Sungguk Cha, Jusung Lee, Younghyun Lee, Cheoljong Yang*

  "When did the emperor Napoleon invented iPhone?" Such hallucination-inducing
question is well known challenge in generative language modeling. In this
study, we present an innovative concept of visual hallucination, referred to as
"I Know (IK)" hallucination, to address scenarios where "I Don't Know" is the
desired response. To effectively tackle this issue, we propose the VQAv2-IDK
benchmark, the subset of VQAv2 comprising unanswerable image-question pairs as
determined by human annotators. Stepping further, we present the visually
dehallucinative instruction generation method for IK hallucination and
introduce the IDK-Instructions visual instruction database. Our experiments
show that current methods struggle with IK hallucination. Yet, our approach
effectively reduces these hallucinations, proving its versatility across
different frameworks and datasets.


---

**[84. [2410.10360] Parenting: Optimizing Knowledge Selection of Retrieval-Augmented
  Language Models with Parameter Decoupling and Tailored Tuning](https://arxiv.org/pdf/2410.10360.pdf)** (2024-10-22)

*Yongxin Xu, Ruizhe Zhang, Xinke Jiang, Yujie Feng, Yuzhen Xiao, Xinyu Ma, Runchuan Zhu, Xu Chu, Junfeng Zhao, Yasha Wang*

  Retrieval-Augmented Generation (RAG) offers an effective solution to the
issues faced by Large Language Models (LLMs) in hallucination generation and
knowledge obsolescence by incorporating externally retrieved knowledge.
However, existing methods lack effective control mechanisms for integrating
internal and external knowledge. Inspired by human cognitive processes, we
propose Parenting, a novel framework that decouples, identifies, and
purposefully optimizes parameter subspaces related to adherence and robustness.
Specifically, Parenting utilizes a key parameter mining method that combines
forward and backward propagation signals to localize subspaces representing
different capabilities. Then, Parenting employs a type-tailored tuning
strategy, applying specific and appropriate optimizations to different
subspaces, aiming to achieve a balanced enhancement of both adherence and
robustness. Extensive experiments on various datasets and models validate the
effectiveness and generalizability of our method.


---

**[85. [2201.12528] SupWMA: Consistent and Efficient Tractography Parcellation of
  Superficial White Matter with Deep Learning](https://arxiv.org/pdf/2201.12528.pdf)** (2022-02-01)

*Tengfei Xue, Fan Zhang, Chaoyi Zhang, Yuqian Chen, Yang Song, Nikos Makris, Yogesh Rathi, Weidong Cai, Lauren J. O'Donnell*

  White matter parcellation classifies tractography streamlines into clusters
or anatomically meaningful tracts to enable quantification and visualization.
Most parcellation methods focus on the deep white matter (DWM), while fewer
methods address the superficial white matter (SWM) due to its complexity. We
propose a deep-learning-based framework, Superficial White Matter Analysis
(SupWMA), that performs an efficient and consistent parcellation of 198 SWM
clusters from whole-brain tractography. A point-cloud-based network is modified
for our SWM parcellation task, and supervised contrastive learning enables more
discriminative representations between plausible streamlines and outliers. We
perform evaluation on a large tractography dataset with ground truth labels and
on three independently acquired testing datasets from individuals across ages
and health conditions. Compared to several state-of-the-art methods, SupWMA
obtains a highly consistent and accurate SWM parcellation result. In addition,
the computational speed of SupWMA is much faster than other methods.


---

**[86. [2403.00425] HALC: Object Hallucination Reduction via Adaptive Focal-Contrast
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

**[87. [2410.07590] TurboRAG: Accelerating Retrieval-Augmented Generation with Precomputed
  KV Caches for Chunked Text](https://arxiv.org/pdf/2410.07590.pdf)** (2024-10-11)

*Songshuo Lu, Hua Wang, Yutian Rong, Zhi Chen, Yaohua Tang*

  Current Retrieval-Augmented Generation (RAG) systems concatenate and process
numerous retrieved document chunks for prefill which requires a large volume of
computation, therefore leading to significant latency in time-to-first-token
(TTFT). To reduce the computation overhead as well as TTFT, we introduce
TurboRAG, a novel RAG system that redesigns the inference paradigm of the
current RAG system by first pre-computing and storing the key-value (KV) caches
of documents offline, and then directly retrieving the saved KV cache for
prefill. Hence, online computation of KV caches is eliminated during inference.
In addition, we provide a number of insights into the mask matrix and
positional embedding mechanisms, plus fine-tune a pretrained language model to
maintain model accuracy of TurboRAG. Our approach is applicable to most
existing large language models and their applications without any requirement
in modification of models and inference systems. Experimental results across a
suite of RAG benchmarks demonstrate that TurboRAG reduces TTFT by up to 9.4x
compared to the conventional RAG systems (on an average of 8.6x), but reserving
comparable performance to the standard RAG systems.


---

**[88. [2502.01549] VideoRAG: Retrieval-Augmented Generation with Extreme Long-Context
  Videos](https://arxiv.org/pdf/2502.01549.pdf)** (2025-02-04)

*Xubin Ren, Lingrui Xu, Long Xia, Shuaiqiang Wang, Dawei Yin, Chao Huang*

  Retrieval-Augmented Generation (RAG) has demonstrated remarkable success in
enhancing Large Language Models (LLMs) through external knowledge integration,
yet its application has primarily focused on textual content, leaving the rich
domain of multi-modal video knowledge predominantly unexplored. This paper
introduces VideoRAG, the first retrieval-augmented generation framework
specifically designed for processing and understanding extremely long-context
videos. Our core innovation lies in its dual-channel architecture that
seamlessly integrates (i) graph-based textual knowledge grounding for capturing
cross-video semantic relationships, and (ii) multi-modal context encoding for
efficiently preserving visual features. This novel design empowers VideoRAG to
process unlimited-length videos by constructing precise knowledge graphs that
span multiple videos while maintaining semantic dependencies through
specialized multi-modal retrieval paradigms. Through comprehensive empirical
evaluation on our proposed LongerVideos benchmark-comprising over 160 videos
totaling 134+ hours across lecture, documentary, and entertainment
categories-VideoRAG demonstrates substantial performance compared to existing
RAG alternatives and long video understanding methods. The source code of
VideoRAG implementation and the benchmark dataset are openly available at:
https://github.com/HKUDS/VideoRAG.


---

**[89. [2502.17297] Benchmarking Retrieval-Augmented Generation in Multi-Modal Contexts](https://arxiv.org/pdf/2502.17297.pdf)** (2025-02-25)

*Zhenghao Liu, Xingsheng Zhu, Tianshuo Zhou, Xinyi Zhang, Xiaoyuan Yi, Yukun Yan, Yu Gu, Ge Yu, Maosong Sun*

  This paper introduces Multi-Modal Retrieval-Augmented Generation (M^2RAG), a
benchmark designed to evaluate the effectiveness of Multi-modal Large Language
Models (MLLMs) in leveraging knowledge from multi-modal retrieval documents.
The benchmark comprises four tasks: image captioning, multi-modal question
answering, multi-modal fact verification, and image reranking. All tasks are
set in an open-domain setting, requiring RAG models to retrieve query-relevant
information from a multi-modal document collection and use it as input context
for RAG modeling. To enhance the context utilization capabilities of MLLMs, we
also introduce Multi-Modal Retrieval-Augmented Instruction Tuning (MM-RAIT), an
instruction tuning method that optimizes MLLMs within multi-modal contexts. Our
experiments show that MM-RAIT improves the performance of RAG systems by
enabling them to effectively learn from multi-modal contexts. All data and code
are available at https://github.com/NEUIR/M2RAG.


---

**[90. [2311.01477] FaithScore: Fine-grained Evaluations of Hallucinations in Large
  Vision-Language Models](https://arxiv.org/pdf/2311.01477.pdf)** (2024-09-30)

*Liqiang Jing, Ruosen Li, Yunmo Chen, Xinya Du*

  We introduce FaithScore (Faithfulness to Atomic Image Facts Score), a
reference-free and fine-grained evaluation metric that measures the
faithfulness of the generated free-form answers from large vision-language
models (LVLMs). The FaithScore evaluation first identifies sub-sentences
containing descriptive statements that need to be verified, then extracts a
comprehensive list of atomic facts from these sub-sentences, and finally
conducts consistency verification between fine-grained atomic facts and the
input image. Meta-evaluation demonstrates that our metric highly correlates
with human judgments of faithfulness. We collect two benchmark datasets (i.e.
LLaVA-1k and MSCOCO-Cap) for evaluating LVLMs instruction-following
hallucinations. We measure hallucinations in state-of-the-art LVLMs with
FaithScore on the datasets. Results reveal that current systems are prone to
generate hallucinated content unfaithful to the image, which leaves room for
future improvements. We hope our metric FaithScore can help evaluate future
LVLMs in terms of faithfulness and provide insightful advice for enhancing
LVLMs' faithfulness.


---

**[91. [2501.02699] EAGLE: Enhanced Visual Grounding Minimizes Hallucinations in
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

**[92. [2312.05200] DelucionQA: Detecting Hallucinations in Domain-specific Question
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

**[93. [2503.06950] CtrlRAG: Black-box Adversarial Attacks Based on Masked Language Models
  in Retrieval-Augmented Language Generation](https://arxiv.org/pdf/2503.06950.pdf)** (2025-03-11)

*Runqi Sui*

  Retrieval-Augmented Generation (RAG) systems enhance Large Language Models
(LLMs) by integrating external knowledge bases. However, this integration
introduces a new security threat: adversaries can exploit the retrieval
mechanism to inject malicious content into the knowledge base, thereby
influencing the generated responses. Based on this attack vector, we propose
CtrlRAG, a novel attack method designed for RAG system in the black-box
setting, which aligns with real-world scenarios. Unlike existing attack
methods, CtrlRAG introduces a perturbation mechanism using Masked Language
Model (MLM) to dynamically optimize malicious content in response to changes in
the retrieved context. Experimental results demonstrate that CtrlRAG
outperforms three baseline methods in both Emotional Manipulation and
Hallucination Amplification objectives. Furthermore, we evaluate three existing
defense mechanisms, revealing their limited effectiveness against CtrlRAG and
underscoring the urgent need for more robust defenses.


---

**[94. [2402.09801] EFUF: Efficient Fine-grained Unlearning Framework for Mitigating
  Hallucinations in Multimodal Large Language Models](https://arxiv.org/pdf/2402.09801.pdf)** (2024-09-24)

*Shangyu Xing, Fei Zhao, Zhen Wu, Tuo An, Weihao Chen, Chunhui Li, Jianbing Zhang, Xinyu Dai*

  Multimodal large language models (MLLMs) have attracted increasing attention
in the past few years, but they may still generate descriptions that include
objects not present in the corresponding images, a phenomenon known as object
hallucination. To eliminate hallucinations, existing methods manually annotate
paired responses with and without hallucinations, and then employ various
alignment algorithms to improve the alignment capability between images and
text. However, they not only demand considerable computation resources during
the finetuning stage but also require expensive human annotation to construct
paired data needed by the alignment algorithms. To address these issues, we
borrow the idea of unlearning and propose an efficient fine-grained unlearning
framework (EFUF), which can eliminate hallucinations without the need for
paired data. Extensive experiments show that our method consistently reduces
hallucinations while preserving the generation quality with modest
computational overhead. Our code and datasets will be publicly available.


---

**[95. [2410.10293] FunnelRAG: A Coarse-to-Fine Progressive Retrieval Paradigm for RAG](https://arxiv.org/pdf/2410.10293.pdf)** (2025-02-18)

*Xinping Zhao, Yan Zhong, Zetian Sun, Xinshuo Hu, Zhenyu Liu, Dongfang Li, Baotian Hu, Min Zhang*

  Retrieval-Augmented Generation (RAG) prevails in Large Language Models. It
mainly consists of retrieval and generation. The retrieval modules (a.k.a.
retrievers) aim to find useful information used to facilitate the generation
modules (a.k.a. generators). As such, generators' performance largely depends
on the effectiveness and efficiency of retrievers. However, the widely used
retrieval paradigm remains flat. It treats retrieval procedures as a one-off
deal with constant granularity. Despite effectiveness, we argue that they
suffer from two limitations: (1) flat retrieval exerts a significant burden on
one retriever; (2) constant granularity limits the ceiling of retrieval
performance. In this work, we propose a progressive retrieval paradigm with
coarse-to-fine granularity for RAG, termed FunnelRAG, so as to balance
effectiveness and efficiency. Specifically, FunnelRAG establishes a progressive
retrieval pipeline by collaborating coarse-to-fine granularity, large-to-small
quantity, and low-to-high capacity, which can relieve the burden on one
retriever and also promote the ceiling of retrieval performance. Extensive
experiments manifest that FunnelRAG achieves comparable retrieval performance
while the time overhead is reduced by nearly 40 percent.


---

**[96. [2308.06382] Phoneme Hallucinator: One-shot Voice Conversion via Set Expansion](https://arxiv.org/pdf/2308.06382.pdf)** (2024-01-02)

*Siyuan Shan, Yang Li, Amartya Banerjee, Junier B. Oliva*

  Voice conversion (VC) aims at altering a person's voice to make it sound
similar to the voice of another person while preserving linguistic content.
Existing methods suffer from a dilemma between content intelligibility and
speaker similarity; i.e., methods with higher intelligibility usually have a
lower speaker similarity, while methods with higher speaker similarity usually
require plenty of target speaker voice data to achieve high intelligibility. In
this work, we propose a novel method \textit{Phoneme Hallucinator} that
achieves the best of both worlds. Phoneme Hallucinator is a one-shot VC model;
it adopts a novel model to hallucinate diversified and high-fidelity target
speaker phonemes based just on a short target speaker voice (e.g. 3 seconds).
The hallucinated phonemes are then exploited to perform neighbor-based voice
conversion. Our model is a text-free, any-to-any VC model that requires no text
annotations and supports conversion to any unseen speaker. Objective and
subjective evaluations show that \textit{Phoneme Hallucinator} outperforms
existing VC methods for both intelligibility and speaker similarity.


---

**[97. [2406.07070] HalluDial: A Large-Scale Benchmark for Automatic Dialogue-Level
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

**[98. [2402.03181] C-RAG: Certified Generation Risks for Retrieval-Augmented Language
  Models](https://arxiv.org/pdf/2402.03181.pdf)** (2024-07-31)

*Mintong Kang, Nezihe Merve Gürel, Ning Yu, Dawn Song, Bo Li*

  Despite the impressive capabilities of large language models (LLMs) across
diverse applications, they still suffer from trustworthiness issues, such as
hallucinations and misalignments. Retrieval-augmented language models (RAG)
have been proposed to enhance the credibility of generations by grounding
external knowledge, but the theoretical understandings of their generation
risks remains unexplored. In this paper, we answer: 1) whether RAG can indeed
lead to low generation risks, 2) how to provide provable guarantees on the
generation risks of RAG and vanilla LLMs, and 3) what sufficient conditions
enable RAG models to reduce generation risks. We propose C-RAG, the first
framework to certify generation risks for RAG models. Specifically, we provide
conformal risk analysis for RAG models and certify an upper confidence bound of
generation risks, which we refer to as conformal generation risk. We also
provide theoretical guarantees on conformal generation risks for general
bounded risk functions under test distribution shifts. We prove that RAG
achieves a lower conformal generation risk than that of a single LLM when the
quality of the retrieval model and transformer is non-trivial. Our intensive
empirical results demonstrate the soundness and tightness of our conformal
generation risk guarantees across four widely-used NLP datasets on four
state-of-the-art retrieval models.


---

**[99. [2212.09068] Style-Hallucinated Dual Consistency Learning: A Unified Framework for
  Visual Domain Generalization](https://arxiv.org/pdf/2212.09068.pdf)** (2023-11-28)

*Yuyang Zhao, Zhun Zhong, Na Zhao, Nicu Sebe, Gim Hee Lee*

  Domain shift widely exists in the visual world, while modern deep neural
networks commonly suffer from severe performance degradation under domain shift
due to the poor generalization ability, which limits the real-world
applications. The domain shift mainly lies in the limited source environmental
variations and the large distribution gap between source and unseen target
data. To this end, we propose a unified framework, Style-HAllucinated Dual
consistEncy learning (SHADE), to handle such domain shift in various visual
tasks. Specifically, SHADE is constructed based on two consistency constraints,
Style Consistency (SC) and Retrospection Consistency (RC). SC enriches the
source situations and encourages the model to learn consistent representation
across style-diversified samples. RC leverages general visual knowledge to
prevent the model from overfitting to source data and thus largely keeps the
representation consistent between the source and general visual models.
Furthermore, we present a novel style hallucination module (SHM) to generate
style-diversified samples that are essential to consistency learning. SHM
selects basis styles from the source distribution, enabling the model to
dynamically generate diverse and realistic samples during training. Extensive
experiments demonstrate that our versatile SHADE can significantly enhance the
generalization in various visual recognition tasks, including image
classification, semantic segmentation and object detection, with different
models, i.e., ConvNets and Transformer.


---

**[100. [2411.00299] RadFlag: A Black-Box Hallucination Detection Method for Medical Vision
  Language Models](https://arxiv.org/pdf/2411.00299.pdf)** (2024-11-19)

*Serena Zhang, Sraavya Sambara, Oishi Banerjee, Julian Acosta, L. John Fahrner, Pranav Rajpurkar*

  Generating accurate radiology reports from medical images is a clinically
important but challenging task. While current Vision Language Models (VLMs)
show promise, they are prone to generating hallucinations, potentially
compromising patient care. We introduce RadFlag, a black-box method to enhance
the accuracy of radiology report generation. Our method uses a sampling-based
flagging technique to find hallucinatory generations that should be removed. We
first sample multiple reports at varying temperatures and then use a Large
Language Model (LLM) to identify claims that are not consistently supported
across samples, indicating that the model has low confidence in those claims.
Using a calibrated threshold, we flag a fraction of these claims as likely
hallucinations, which should undergo extra review or be automatically rejected.
Our method achieves high precision when identifying both individual
hallucinatory sentences and reports that contain hallucinations. As an
easy-to-use, black-box system that only requires access to a model's
temperature parameter, RadFlag is compatible with a wide range of radiology
report generation models and has the potential to broadly improve the quality
of automated radiology reporting.


---

**[101. [2503.13563] MES-RAG: Bringing Multi-modal, Entity-Storage, and Secure Enhancements
  to RAG](https://arxiv.org/pdf/2503.13563.pdf)** (2025-03-19)

*Pingyu Wu, Daiheng Gao, Jing Tang, Huimin Chen, Wenbo Zhou, Weiming Zhang, Nenghai Yu*

  Retrieval-Augmented Generation (RAG) improves Large Language Models (LLMs) by
using external knowledge, but it struggles with precise entity information
retrieval. In this paper, we proposed MES-RAG framework, which enhances
entity-specific query handling and provides accurate, secure, and consistent
responses. MES-RAG introduces proactive security measures that ensure system
integrity by applying protections prior to data access. Additionally, the
system supports real-time multi-modal outputs, including text, images, audio,
and video, seamlessly integrating into existing RAG architectures. Experimental
results demonstrate that MES-RAG significantly improves both accuracy and
recall, highlighting its effectiveness in advancing the security and utility of
question-answering, increasing accuracy to 0.83 (+0.25) on targeted task. Our
code and data are available at https://github.com/wpydcr/MES-RAG.


---

**[102. [2308.12587] Grounded Entity-Landmark Adaptive Pre-training for Vision-and-Language
  Navigation](https://arxiv.org/pdf/2308.12587.pdf)** (2023-08-25)

*Yibo Cui, Liang Xie, Yakun Zhang, Meishan Zhang, Ye Yan, Erwei Yin*

  Cross-modal alignment is one key challenge for Vision-and-Language Navigation
(VLN). Most existing studies concentrate on mapping the global instruction or
single sub-instruction to the corresponding trajectory. However, another
critical problem of achieving fine-grained alignment at the entity level is
seldom considered. To address this problem, we propose a novel Grounded
Entity-Landmark Adaptive (GELA) pre-training paradigm for VLN tasks. To achieve
the adaptive pre-training paradigm, we first introduce grounded entity-landmark
human annotations into the Room-to-Room (R2R) dataset, named GEL-R2R.
Additionally, we adopt three grounded entity-landmark adaptive pre-training
objectives: 1) entity phrase prediction, 2) landmark bounding box prediction,
and 3) entity-landmark semantic alignment, which explicitly supervise the
learning of fine-grained cross-modal alignment between entity phrases and
environment landmarks. Finally, we validate our model on two downstream
benchmarks: VLN with descriptive instructions (R2R) and dialogue instructions
(CVDN). The comprehensive experiments show that our GELA model achieves
state-of-the-art results on both tasks, demonstrating its effectiveness and
generalizability.


---

**[103. [2403.18920] CPR: Retrieval Augmented Generation for Copyright Protection](https://arxiv.org/pdf/2403.18920.pdf)** (2024-03-29)

*Aditya Golatkar, Alessandro Achille, Luca Zancato, Yu-Xiang Wang, Ashwin Swaminathan, Stefano Soatto*

  Retrieval Augmented Generation (RAG) is emerging as a flexible and robust
technique to adapt models to private users data without training, to handle
credit attribution, and to allow efficient machine unlearning at scale.
However, RAG techniques for image generation may lead to parts of the retrieved
samples being copied in the model's output. To reduce risks of leaking private
information contained in the retrieved set, we introduce Copy-Protected
generation with Retrieval (CPR), a new method for RAG with strong copyright
protection guarantees in a mixed-private setting for diffusion models.CPR
allows to condition the output of diffusion models on a set of retrieved
images, while also guaranteeing that unique identifiable information about
those example is not exposed in the generated outputs. In particular, it does
so by sampling from a mixture of public (safe) distribution and private (user)
distribution by merging their diffusion scores at inference. We prove that CPR
satisfies Near Access Freeness (NAF) which bounds the amount of information an
attacker may be able to extract from the generated images. We provide two
algorithms for copyright protection, CPR-KL and CPR-Choose. Unlike previously
proposed rejection-sampling-based NAF methods, our methods enable efficient
copyright-protected sampling with a single run of backward diffusion. We show
that our method can be applied to any pre-trained conditional diffusion model,
such as Stable Diffusion or unCLIP. In particular, we empirically show that
applying CPR on top of unCLIP improves quality and text-to-image alignment of
the generated results (81.4 to 83.17 on TIFA benchmark), while enabling credit
attribution, copy-right protection, and deterministic, constant time,
unlearning.


---

**[104. [2406.13805] WikiContradict: A Benchmark for Evaluating LLMs on Real-World Knowledge
  Conflicts from Wikipedia](https://arxiv.org/pdf/2406.13805.pdf)** (2024-06-21)

*Yufang Hou, Alessandra Pascale, Javier Carnerero-Cano, Tigran Tchrakian, Radu Marinescu, Elizabeth Daly, Inkit Padhi, Prasanna Sattigeri*

  Retrieval-augmented generation (RAG) has emerged as a promising solution to
mitigate the limitations of large language models (LLMs), such as
hallucinations and outdated information. However, it remains unclear how LLMs
handle knowledge conflicts arising from different augmented retrieved passages,
especially when these passages originate from the same source and have equal
trustworthiness. In this work, we conduct a comprehensive evaluation of
LLM-generated answers to questions that have varying answers based on
contradictory passages from Wikipedia, a dataset widely regarded as a
high-quality pre-training resource for most LLMs. Specifically, we introduce
WikiContradict, a benchmark consisting of 253 high-quality, human-annotated
instances designed to assess LLM performance when augmented with retrieved
passages containing real-world knowledge conflicts. We benchmark a diverse
range of both closed and open-source LLMs under different QA scenarios,
including RAG with a single passage, and RAG with 2 contradictory passages.
Through rigorous human evaluations on a subset of WikiContradict instances
involving 5 LLMs and over 3,500 judgements, we shed light on the behaviour and
limitations of these models. For instance, when provided with two passages
containing contradictory facts, all models struggle to generate answers that
accurately reflect the conflicting nature of the context, especially for
implicit conflicts requiring reasoning. Since human evaluation is costly, we
also introduce an automated model that estimates LLM performance using a strong
open-source language model, achieving an F-score of 0.8. Using this automated
metric, we evaluate more than 1,500 answers from seven LLMs across all
WikiContradict instances. To facilitate future work, we release WikiContradict
on: https://ibm.biz/wikicontradict.


---
