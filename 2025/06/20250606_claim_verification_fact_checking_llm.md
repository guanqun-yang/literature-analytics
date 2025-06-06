**[1. [2503.07937] LLM-based Corroborating and Refuting Evidence Retrieval for Scientific
  Claim Verification](https://arxiv.org/pdf/2503.07937.pdf)** (2025-03-12)

*Siyuan Wang, James R. Foulds, Md Osman Gani, Shimei Pan*

  In this paper, we introduce CIBER (Claim Investigation Based on Evidence
Retrieval), an extension of the Retrieval-Augmented Generation (RAG) framework
designed to identify corroborating and refuting documents as evidence for
scientific claim verification. CIBER addresses the inherent uncertainty in
Large Language Models (LLMs) by evaluating response consistency across diverse
interrogation probes. By focusing on the behavioral analysis of LLMs without
requiring access to their internal information, CIBER is applicable to both
white-box and black-box models. Furthermore, CIBER operates in an unsupervised
manner, enabling easy generalization across various scientific domains.
Comprehensive evaluations conducted using LLMs with varying levels of
linguistic proficiency reveal CIBER's superior performance compared to
conventional RAG approaches. These findings not only highlight the
effectiveness of CIBER but also provide valuable insights for future
advancements in LLM-based scientific claim verification.


---

**[2. [2402.12566] GenAudit: Fixing Factual Errors in Language Model Outputs with Evidence](https://arxiv.org/pdf/2402.12566.pdf)** (2025-01-22)

*Kundan Krishna, Sanjana Ramprasad, Prakhar Gupta, Byron C. Wallace, Zachary C. Lipton, Jeffrey P. Bigham*

  LLMs can generate factually incorrect statements even when provided access to
reference documents. Such errors can be dangerous in high-stakes applications
(e.g., document-grounded QA for healthcare or finance). We present GenAudit --
a tool intended to assist fact-checking LLM responses for document-grounded
tasks. GenAudit suggests edits to the LLM response by revising or removing
claims that are not supported by the reference document, and also presents
evidence from the reference for facts that do appear to have support. We train
models to execute these tasks, and design an interactive interface to present
suggested edits and evidence to users. Comprehensive evaluation by human raters
shows that GenAudit can detect errors in 8 different LLM outputs when
summarizing documents from diverse domains. User studies demonstrate that using
GenAudit can substantially improve the performance of humans at finding errors
in LLM-generated summaries. We release our tool (GenAudit) and fact-checking
model for public use.


---

**[3. [2402.05904] FACT-GPT: Fact-Checking Augmentation via Claim Matching with LLMs](https://arxiv.org/pdf/2402.05904.pdf)** (2024-02-09)

*Eun Cheol Choi, Emilio Ferrara*

  Our society is facing rampant misinformation harming public health and trust.
To address the societal challenge, we introduce FACT-GPT, a system leveraging
Large Language Models (LLMs) to automate the claim matching stage of
fact-checking. FACT-GPT, trained on a synthetic dataset, identifies social
media content that aligns with, contradicts, or is irrelevant to previously
debunked claims. Our evaluation shows that our specialized LLMs can match the
accuracy of larger models in identifying related claims, closely mirroring
human judgment. This research provides an automated solution for efficient
claim matching, demonstrates the potential of LLMs in supporting fact-checkers,
and offers valuable resources for further research in the field.


---

**[4. [2410.23526] LEAF: Learning and Evaluation Augmented by Fact-Checking to Improve
  Factualness in Large Language Models](https://arxiv.org/pdf/2410.23526.pdf)** (2024-11-01)

*Hieu Tran, Junda Wang, Yujan Ting, Weijing Huang, Terrence Chen*

  Large language models (LLMs) have shown remarkable capabilities in various
natural language processing tasks, yet they often struggle with maintaining
factual accuracy, particularly in knowledge-intensive domains like healthcare.
This study introduces LEAF: Learning and Evaluation Augmented by Fact-Checking,
a novel approach designed to enhance the factual reliability of LLMs, with a
focus on medical question answering (QA). LEAF utilizes a dual strategy to
enhance the factual accuracy of responses from models such as Llama 3 70B
Instruct and Llama 3 8B Instruct. The first strategy, Fact-Check-Then-RAG,
improves Retrieval-Augmented Generation (RAG) by incorporating fact-checking
results to guide the retrieval process without updating model parameters. The
second strategy, Learning from Fact-Checks via Self-Training, involves
supervised fine-tuning (SFT) on fact-checked responses or applying Simple
Preference Optimization (SimPO) with fact-checking as a ranking mechanism, both
updating LLM parameters from supervision. These findings suggest that
integrating fact-checked responses whether through RAG enhancement or
self-training enhances the reliability and factual correctness of LLM outputs,
offering a promising solution for applications where information accuracy is
crucial.


---

**[5. [2502.13416] Detecting LLM Fact-conflicting Hallucinations Enhanced by
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

**[6. [2310.13549] The Perils & Promises of Fact-checking with Large Language Models](https://arxiv.org/pdf/2310.13549.pdf)** (2024-02-08)

*Dorian Quelle, Alexandre Bovet*

  Automated fact-checking, using machine learning to verify claims, has grown
vital as misinformation spreads beyond human fact-checking capacity. Large
Language Models (LLMs) like GPT-4 are increasingly trusted to write academic
papers, lawsuits, and news articles and to verify information, emphasizing
their role in discerning truth from falsehood and the importance of being able
to verify their outputs. Understanding the capacities and limitations of LLMs
in fact-checking tasks is therefore essential for ensuring the health of our
information ecosystem. Here, we evaluate the use of LLM agents in fact-checking
by having them phrase queries, retrieve contextual data, and make decisions.
Importantly, in our framework, agents explain their reasoning and cite the
relevant sources from the retrieved context. Our results show the enhanced
prowess of LLMs when equipped with contextual information. GPT-4 outperforms
GPT-3, but accuracy varies based on query language and claim veracity. While
LLMs show promise in fact-checking, caution is essential due to inconsistent
accuracy. Our investigation calls for further research, fostering a deeper
comprehension of when agents succeed and when they fail.


---

**[7. [2405.05583] OpenFactCheck: Building, Benchmarking Customized Fact-Checking Systems
  and Evaluating the Factuality of Claims and LLMs](https://arxiv.org/pdf/2405.05583.pdf)** (2024-12-17)

*Yuxia Wang, Minghan Wang, Hasan Iqbal, Georgi Georgiev, Jiahui Geng, Preslav Nakov*

  The increased use of large language models (LLMs) across a variety of
real-world applications calls for mechanisms to verify the factual accuracy of
their outputs. Difficulties lie in assessing the factuality of free-form
responses in open domains. Also, different papers use disparate evaluation
benchmarks and measurements, which renders them hard to compare and hampers
future progress. To mitigate these issues, we propose OpenFactCheck, a unified
framework for building customized automatic fact-checking systems, benchmarking
their accuracy, evaluating factuality of LLMs, and verifying claims in a
document. OpenFactCheck consists of three modules: (i) CUSTCHECKER allows users
to easily customize an automatic fact-checker and verify the factual
correctness of documents and claims, (ii) LLMEVAL, a unified evaluation
framework assesses LLM's factuality ability from various perspectives fairly,
and (iii) CHECKEREVAL is an extensible solution for gauging the reliability of
automatic fact-checkers' verification results using human-annotated datasets.
Data and code are publicly available at
https://github.com/yuxiaw/openfactcheck.


---

**[8. [2306.13781] Retrieving Supporting Evidence for LLMs Generated Answers](https://arxiv.org/pdf/2306.13781.pdf)** (2023-06-27)

*Siqing Huo, Negar Arabzadeh, Charles L. A. Clarke*

  Current large language models (LLMs) can exhibit near-human levels of
performance on many natural language tasks, including open-domain question
answering. Unfortunately, they also convincingly hallucinate incorrect answers,
so that responses to questions must be verified against external sources before
they can be accepted at face value. In this paper, we report a simple
experiment to automatically verify generated answers against a corpus. After
presenting a question to an LLM and receiving a generated answer, we query the
corpus with the combination of the question + generated answer. We then present
the LLM with the combination of the question + generated answer + retrieved
answer, prompting it to indicate if the generated answer can be supported by
the retrieved answer. We base our experiment on questions and passages from the
MS MARCO (V1) test collection, exploring three retrieval approaches ranging
from standard BM25 to a full question answering stack, including a reader based
on the LLM. For a large fraction of questions, we find that an LLM is capable
of verifying its generated answer if appropriate supporting material is
provided. However, with an accuracy of 70-80%, this approach cannot be fully
relied upon to detect hallucinations.


---

**[9. [2412.10689] Learning to Verify Summary Facts with Fine-Grained LLM Feedback](https://arxiv.org/pdf/2412.10689.pdf)** (2024-12-17)

*Jihwan Oh, Jeonghwan Choi, Nicole Hee-Yeon Kim, Taewon Yun, Hwanjun Song*

  Training automatic summary fact verifiers often faces the challenge of a lack
of human-labeled data. In this paper, we explore alternative way of leveraging
Large Language Model (LLM) generated feedback to address the inherent
limitation of using human-labeled data. We introduce FineSumFact, a large-scale
dataset containing fine-grained factual feedback on summaries. We employ 10
distinct LLMs for diverse summary generation and Llama-3-70B-Instruct for
feedback. We utilize this dataset to fine-tune the lightweight open-source
model Llama-3-8B-Instruct, optimizing resource efficiency while maintaining
high performance. Our experimental results reveal that the model trained on
extensive LLM-generated datasets surpasses that trained on smaller
human-annotated datasets when evaluated using human-generated test sets.
Fine-tuning fact verification models with LLM feedback can be more effective
and cost-efficient than using human feedback. The dataset is available at
https://github.com/DISL-Lab/FineSumFact.


---

**[10. [2404.00942] Evaluating the Factuality of Large Language Models using Large-Scale
  Knowledge Graphs](https://arxiv.org/pdf/2404.00942.pdf)** (2024-04-02)

*Xiaoze Liu, Feijie Wu, Tianyang Xu, Zhuo Chen, Yichi Zhang, Xiaoqian Wang, Jing Gao*

  The advent of Large Language Models (LLMs) has significantly transformed the
AI landscape, enhancing machine learning and AI capabilities. Factuality issue
is a critical concern for LLMs, as they may generate factually incorrect
responses. In this paper, we propose GraphEval to evaluate an LLM's performance
using a substantially large test dataset. Specifically, the test dataset is
retrieved from a large knowledge graph with more than 10 million facts without
expensive human efforts. Unlike conventional methods that evaluate LLMs based
on generated responses, GraphEval streamlines the evaluation process by
creating a judge model to estimate the correctness of the answers given by the
LLM. Our experiments demonstrate that the judge model's factuality assessment
aligns closely with the correctness of the LLM's generated outputs, while also
substantially reducing evaluation costs. Besides, our findings offer valuable
insights into LLM performance across different metrics and highlight the
potential for future improvements in ensuring the factual integrity of LLM
outputs. The code is publicly available at https://github.com/xz-liu/GraphEval.


---

**[11. [2408.12060] Evidence-backed Fact Checking using RAG and Few-Shot In-Context Learning
  with LLMs](https://arxiv.org/pdf/2408.12060.pdf)** (2024-10-08)

*Ronit Singhal, Pransh Patwa, Parth Patwa, Aman Chadha, Amitava Das*

  Given the widespread dissemination of misinformation on social media,
implementing fact-checking mechanisms for online claims is essential. Manually
verifying every claim is very challenging, underscoring the need for an
automated fact-checking system. This paper presents our system designed to
address this issue. We utilize the Averitec dataset (Schlichtkrull et al.,
2023) to assess the performance of our fact-checking system. In addition to
veracity prediction, our system provides supporting evidence, which is
extracted from the dataset. We develop a Retrieve and Generate (RAG) pipeline
to extract relevant evidence sentences from a knowledge base, which are then
inputted along with the claim into a large language model (LLM) for
classification. We also evaluate the few-shot In-Context Learning (ICL)
capabilities of multiple LLMs. Our system achieves an 'Averitec' score of 0.33,
which is a 22% absolute improvement over the baseline. Our Code is publicly
available on
https://github.com/ronit-singhal/evidence-backed-fact-checking-using-rag-and-few-shot-in-context-learning-with-llms.


---

**[12. [2406.13805] WikiContradict: A Benchmark for Evaluating LLMs on Real-World Knowledge
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

**[13. [2406.17663] LLM-ARC: Enhancing LLMs with an Automated Reasoning Critic](https://arxiv.org/pdf/2406.17663.pdf)** (2024-07-22)

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

**[14. [2310.04535] LLM4DV: Using Large Language Models for Hardware Test Stimuli Generation](https://arxiv.org/pdf/2310.04535.pdf)** (2025-03-26)

*Zixi Zhang, Balint Szekely, Pedro Gimenes, Greg Chadwick, Hugo McNally, Jianyi Cheng, Robert Mullins, Yiren Zhao*

  Hardware design verification (DV) is a process that checks the functional
equivalence of a hardware design against its specifications, improving hardware
reliability and robustness. A key task in the DV process is the test stimuli
generation, which creates a set of conditions or inputs for testing. These test
conditions are often complex and specific to the given hardware design,
requiring substantial human engineering effort to optimize. We seek a solution
of automated and efficient testing for arbitrary hardware designs that takes
advantage of large language models (LLMs). LLMs have already shown promising
results for improving hardware design automation, but remain under-explored for
hardware DV. In this paper, we propose an open-source benchmarking framework
named LLM4DV that efficiently orchestrates LLMs for automated hardware test
stimuli generation. Our analysis evaluates six different LLMs involving six
prompting improvements over eight hardware designs and provides insight for
future work on LLMs development for efficient automated DV.


---

**[15. [2501.16672] VeriFact: Verifying Facts in LLM-Generated Clinical Text with Electronic
  Health Records](https://arxiv.org/pdf/2501.16672.pdf)** (2025-01-29)

*Philip Chung, Akshay Swaminathan, Alex J. Goodell, Yeasul Kim, S. Momsen Reincke, Lichy Han, Ben Deverett, Mohammad Amin Sadeghi, Abdel-Badih Ariss, Marc Ghanem, David Seong, Andrew A. Lee, Caitlin E. Coombes, Brad Bradshaw, Mahir A. Sufian, Hyo Jung Hong, Teresa P. Nguyen, Mohammad R. Rasouli, Komal Kamra, Mark A. Burbridge, James C. McAvoy, Roya Saffary, Stephen P. Ma, Dev Dash, James Xie, Ellen Y. Wang, Clifford A. Schmiesing, Nigam Shah, Nima Aghaeepour*

  Methods to ensure factual accuracy of text generated by large language models
(LLM) in clinical medicine are lacking. VeriFact is an artificial intelligence
system that combines retrieval-augmented generation and LLM-as-a-Judge to
verify whether LLM-generated text is factually supported by a patient's medical
history based on their electronic health record (EHR). To evaluate this system,
we introduce VeriFact-BHC, a new dataset that decomposes Brief Hospital Course
narratives from discharge summaries into a set of simple statements with
clinician annotations for whether each statement is supported by the patient's
EHR clinical notes. Whereas highest agreement between clinicians was 88.5%,
VeriFact achieves up to 92.7% agreement when compared to a denoised and
adjudicated average human clinican ground truth, suggesting that VeriFact
exceeds the average clinician's ability to fact-check text against a patient's
medical record. VeriFact may accelerate the development of LLM-based EHR
applications by removing current evaluation bottlenecks.


---

**[16. [2502.19954] Collaborative Stance Detection via Small-Large Language Model
  Consistency Verification](https://arxiv.org/pdf/2502.19954.pdf)** (2025-02-28)

*Yu Yan, Sheng Sun, Zixiang Tang, Teli Liu, Min Liu*

  Stance detection on social media aims to identify attitudes expressed in
tweets towards specific targets. Current studies prioritize Large Language
Models (LLMs) over Small Language Models (SLMs) due to the overwhelming
performance improving provided by LLMs. However, heavily relying on LLMs for
stance detection, regardless of the cost, is impractical for real-world social
media monitoring systems that require vast data analysis. To this end, we
propose \textbf{\underline{Co}}llaborative Stance Detection via Small-Large
Language Model Consistency \textbf{\underline{Ver}}ification (\textbf{CoVer})
framework, which enhances LLM utilization via context-shared batch reasoning
and logical verification between LLM and SLM. Specifically, instead of
processing each text individually, CoVer processes texts batch-by-batch,
obtaining stance predictions and corresponding explanations via LLM reasoning
in a shared context. Then, to exclude the bias caused by context noises, CoVer
introduces the SLM for logical consistency verification. Finally, texts that
repeatedly exhibit low logical consistency are classified using
consistency-weighted aggregation of prior LLM stance predictions. Our
experiments show that CoVer outperforms state-of-the-art methods across
multiple benchmarks in the zero-shot setting, achieving 0.54 LLM queries per
tweet while significantly enhancing performance. Our CoVer offers a more
practical solution for LLM deploying for social media stance detection.


---

**[17. [2410.18359] Improving Model Factuality with Fine-grained Critique-based Evaluator](https://arxiv.org/pdf/2410.18359.pdf)** (2025-03-25)

*Yiqing Xie, Wenxuan Zhou, Pradyot Prakash, Di Jin, Yuning Mao, Quintin Fettes, Arya Talebzadeh, Sinong Wang, Han Fang, Carolyn Rose, Daniel Fried, Hejia Zhang*

  Factuality evaluation aims to detect factual errors produced by language
models (LMs) and hence guide the development of more factual models. Towards
this goal, we train a factuality evaluator, FenCE, that provides LM generators
with claim-level factuality feedback. We conduct data augmentation on a
combination of public judgment datasets to train FenCE to (1) generate textual
critiques along with scores and (2) make claim-level judgment based on diverse
source documents obtained by various tools. We then present a framework that
leverages FenCE to improve the factuality of LM generators by constructing
training data. Specifically, we generate a set of candidate responses, leverage
FenCE to revise and score each response without introducing lesser-known facts,
and train the generator by preferring highly scored revised responses.
Experiments show that our data augmentation methods improve the evaluator's
accuracy by 2.9% on LLM-AggreFact. With FenCE, we improve Llama2-7B-chat and
Llama3-8B-chat's factuality rate by 16.86% and 14.45% on FActScore,
outperforming state-of-the-art factuality finetuning methods by 8.83% and
6.96%.


---

**[18. [2501.12975] OnionEval: An Unified Evaluation of Fact-conflicting Hallucination for
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

**[19. [2410.16848] ETHIC: Evaluating Large Language Models on Long-Context Tasks with High
  Information Coverage](https://arxiv.org/pdf/2410.16848.pdf)** (2025-02-28)

*Taewhoo Lee, Chanwoong Yoon, Kyochul Jang, Donghyeon Lee, Minju Song, Hyunjae Kim, Jaewoo Kang*

  Recent advancements in large language models (LLM) capable of processing
extremely long texts highlight the need for a dedicated evaluation benchmark to
assess their long-context capabilities. However, existing methods, like the
needle-in-a-haystack test, do not effectively assess whether these models fully
utilize contextual information, raising concerns about the reliability of
current evaluation techniques. To thoroughly examine the effectiveness of
existing benchmarks, we introduce a new metric called information coverage
(IC), which quantifies the proportion of the input context necessary for
answering queries. Our findings indicate that current benchmarks exhibit low
IC; although the input context may be extensive, the actual usable context is
often limited. To address this, we present ETHIC, a novel benchmark designed to
assess LLMs' ability to leverage the entire context. Our benchmark comprises
1,986 test instances spanning four long-context tasks with high IC scores in
the domains of books, debates, medicine, and law. Our evaluations reveal
significant performance drops in contemporary LLMs, highlighting a critical
challenge in managing long contexts. Our benchmark is available at
https://github.com/dmis-lab/ETHIC.


---

**[20. [2310.00741] FELM: Benchmarking Factuality Evaluation of Large Language Models](https://arxiv.org/pdf/2310.00741.pdf)** (2023-11-29)

*Shiqi Chen, Yiran Zhao, Jinghan Zhang, I-Chun Chern, Siyang Gao, Pengfei Liu, Junxian He*

  Assessing factuality of text generated by large language models (LLMs) is an
emerging yet crucial research area, aimed at alerting users to potential errors
and guiding the development of more reliable LLMs. Nonetheless, the evaluators
assessing factuality necessitate suitable evaluation themselves to gauge
progress and foster advancements. This direction remains under-explored,
resulting in substantial impediments to the progress of factuality evaluators.
To mitigate this issue, we introduce a benchmark for Factuality Evaluation of
large Language Models, referred to as felm. In this benchmark, we collect
responses generated from LLMs and annotate factuality labels in a fine-grained
manner. Contrary to previous studies that primarily concentrate on the
factuality of world knowledge (e.g.~information from Wikipedia), felm focuses
on factuality across diverse domains, spanning from world knowledge to math and
reasoning. Our annotation is based on text segments, which can help pinpoint
specific factual errors. The factuality annotations are further supplemented by
predefined error types and reference links that either support or contradict
the statement. In our experiments, we investigate the performance of several
LLM-based factuality evaluators on felm, including both vanilla LLMs and those
augmented with retrieval mechanisms and chain-of-thought processes. Our
findings reveal that while retrieval aids factuality evaluation, current LLMs
are far from satisfactory to faithfully detect factual errors.


---

**[21. [2311.10733] Proceedings of the 3rd International Workshop on Mining and Learning in
  the Legal Domain (MLLD-23)](https://arxiv.org/pdf/2311.10733.pdf)** (2023-11-21)

*Masoud Makrehchi, Dell Zhang, Alina Petrova, John Armour*

  This is the Proceedings of the 3rd International Workshop on Mining and
Learning in the Legal Domain (MLLD-23) which took place in conjunction with the
32nd ACM International Conference on Information and Knowledge Management
(CIKM-2023) at the University of Birmingham, Birmingham, UK on Sunday 22nd
October 2023.


---

**[22. [2503.17229] FactSelfCheck: Fact-Level Black-Box Hallucination Detection for LLMs](https://arxiv.org/pdf/2503.17229.pdf)** (2025-03-24)

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

**[23. [2502.07912] Elevating Legal LLM Responses: Harnessing Trainable Logical Structures
  and Semantic Knowledge with Legal Reasoning](https://arxiv.org/pdf/2502.07912.pdf)** (2025-02-13)

*Rujing Yao, Yang Wu, Chenghao Wang, Jingwei Xiong, Fang Wang, Xiaozhong Liu*

  Large Language Models (LLMs) have achieved impressive results across numerous
domains, yet they experience notable deficiencies in legal question-answering
tasks. LLMs often generate generalized responses that lack the logical
specificity required for expert legal advice and are prone to hallucination,
providing answers that appear correct but are unreliable. Retrieval-Augmented
Generation (RAG) techniques offer partial solutions to address this challenge,
but existing approaches typically focus only on semantic similarity, neglecting
the logical structure essential to legal reasoning. In this paper, we propose
the Logical-Semantic Integration Model (LSIM), a novel supervised framework
that bridges semantic and logical coherence. LSIM comprises three components:
reinforcement learning predicts a structured fact-rule chain for each question,
a trainable Deep Structured Semantic Model (DSSM) retrieves the most relevant
candidate questions by integrating semantic and logical features, and
in-context learning generates the final answer using the retrieved content. Our
experiments on a real-world legal QA dataset-validated through both automated
metrics and human evaluation-demonstrate that LSIM significantly enhances
accuracy and reliability compared to existing methods.


---

**[24. [2410.04838] Rationale-Aware Answer Verification by Pairwise Self-Evaluation](https://arxiv.org/pdf/2410.04838.pdf)** (2024-10-28)

*Akira Kawabata, Saku Sugawara*

  Answer verification identifies correct solutions among candidates generated
by large language models (LLMs). Current approaches typically train verifier
models by labeling solutions as correct or incorrect based solely on whether
the final answer matches the gold answer. However, this approach neglects any
flawed rationale in the solution yielding the correct answer, undermining the
verifier's ability to distinguish between sound and flawed rationales. We
empirically show that in StrategyQA, only 19% of LLM-generated solutions with
correct answers have valid rationales, thus leading to an unreliable verifier.
Furthermore, we demonstrate that training a verifier on valid rationales
significantly improves its ability to distinguish valid and flawed rationale.
To make a better verifier without extra human supervision, we introduce REPS
(Rationale Enhancement through Pairwise Selection), a method for selecting
valid rationales from candidates by iteratively applying pairwise
self-evaluation using the same LLM that generates the solutions. Verifiers
trained on solutions selected by REPS outperform those trained using
conventional training methods on three reasoning benchmarks (ARC-Challenge,
DROP, and StrategyQA). Our results suggest that training reliable verifiers
requires ensuring the validity of rationales in addition to the correctness of
the final answers, which would be critical for models assisting humans in
solving complex reasoning tasks.


---

**[25. [2402.17097] Re-Ex: Revising after Explanation Reduces the Factual Errors in LLM
  Responses](https://arxiv.org/pdf/2402.17097.pdf)** (2025-04-15)

*Juyeon Kim, Jeongeun Lee, Yoonho Chang, Chanyeol Choi, Junseong Kim, Jy-yong Sohn*

  Mitigating hallucination issues is a key challenge that must be overcome to
reliably deploy large language models (LLMs) in real-world scenarios. Recently,
various methods have been proposed to detect and revise factual errors in
LLM-generated texts, in order to reduce hallucination. In this paper, we
propose Re-Ex, a method for post-editing LLM-generated responses. Re-Ex
introduces a novel reasoning step dubbed as the factual error explanation step.
Re-Ex revises the initial response of LLMs using 3-steps : first, external
tools are used to retrieve the evidences of the factual errors in the initial
LLM response; next, LLM is instructed to explain the problematic parts of the
response based on the gathered evidence; finally, LLM revises the initial
response using the explanations provided in the previous step. In addition to
the explanation step, Re-Ex also incorporates new prompting techniques to
reduce the token count and inference time required for the response revision
process. Compared with existing methods including FacTool, CoVE, and RARR,
Re-Ex provides better detection and revision performance with less inference
time and fewer tokens in multiple benchmarks.


---

**[26. [2405.00648] Drowzee: Metamorphic Testing for Fact-Conflicting Hallucination
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

**[27. [2410.12377] HerO at AVeriTeC: The Herd of Open Large Language Models for Verifying
  Real-World Claims](https://arxiv.org/pdf/2410.12377.pdf)** (2024-10-22)

*Yejun Yoon, Jaeyoon Jung, Seunghyun Yoon, Kunwoo Park*

  To tackle the AVeriTeC shared task hosted by the FEVER-24, we introduce a
system that only employs publicly available large language models (LLMs) for
each step of automated fact-checking, dubbed the Herd of Open LLMs for
verifying real-world claims (HerO). For evidence retrieval, a language model is
used to enhance a query by generating hypothetical fact-checking documents. We
prompt pretrained and fine-tuned LLMs for question generation and veracity
prediction by crafting prompts with retrieved in-context samples. HerO achieved
2nd place on the leaderboard with the AVeriTeC score of 0.57, suggesting the
potential of open LLMs for verifying real-world claims. For future research, we
make our code publicly available at https://github.com/ssu-humane/HerO.


---

**[28. [2504.06438] Don't Let It Hallucinate: Premise Verification via Retrieval-Augmented
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

**[29. [2408.11832] OpenFactCheck: A Unified Framework for Factuality Evaluation of LLMs](https://arxiv.org/pdf/2408.11832.pdf)** (2024-11-07)

*Hasan Iqbal, Yuxia Wang, Minghan Wang, Georgi Georgiev, Jiahui Geng, Iryna Gurevych, Preslav Nakov*

  The increased use of large language models (LLMs) across a variety of
real-world applications calls for automatic tools to check the factual accuracy
of their outputs, as LLMs often hallucinate. This is difficult as it requires
assessing the factuality of free-form open-domain responses. While there has
been a lot of research on this topic, different papers use different evaluation
benchmarks and measures, which makes them hard to compare and hampers future
progress. To mitigate these issues, we developed OpenFactCheck, a unified
framework, with three modules: (i) RESPONSEEVAL, which allows users to easily
customize an automatic fact-checking system and to assess the factuality of all
claims in an input document using that system, (ii) LLMEVAL, which assesses the
overall factuality of an LLM, and (iii) CHECKEREVAL, a module to evaluate
automatic fact-checking systems. OpenFactCheck is open-sourced
(https://github.com/mbzuai-nlp/openfactcheck) and publicly released as a Python
library (https://pypi.org/project/openfactcheck/) and also as a web service
(http://app.openfactcheck.com). A video describing the system is available at
https://youtu.be/-i9VKL0HleI.


---

**[30. [2504.00374] When Persuasion Overrides Truth in Multi-Agent LLM Debates: Introducing
  a Confidence-Weighted Persuasion Override Rate (CW-POR)](https://arxiv.org/pdf/2504.00374.pdf)** (2025-04-02)

*Mahak Agarwal, Divyam Khanna*

  In many real-world scenarios, a single Large Language Model (LLM) may
encounter contradictory claims-some accurate, others forcefully incorrect-and
must judge which is true. We investigate this risk in a single-turn,
multi-agent debate framework: one LLM-based agent provides a factual answer
from TruthfulQA, another vigorously defends a falsehood, and the same LLM
architecture serves as judge. We introduce the Confidence-Weighted Persuasion
Override Rate (CW-POR), which captures not only how often the judge is deceived
but also how strongly it believes the incorrect choice. Our experiments on five
open-source LLMs (3B-14B parameters), where we systematically vary agent
verbosity (30-300 words), reveal that even smaller models can craft persuasive
arguments that override truthful answers-often with high confidence. These
findings underscore the importance of robust calibration and adversarial
testing to prevent LLMs from confidently endorsing misinformation.


---

**[31. [2403.07557] SIFiD: Reassess Summary Factual Inconsistency Detection with LLM](https://arxiv.org/pdf/2403.07557.pdf)** (2024-03-13)

*Jiuding Yang, Hui Liu, Weidong Guo, Zhuwei Rao, Yu Xu, Di Niu*

  Ensuring factual consistency between the summary and the original document is
paramount in summarization tasks. Consequently, considerable effort has been
dedicated to detecting inconsistencies. With the advent of Large Language
Models (LLMs), recent studies have begun to leverage their advanced language
understanding capabilities for inconsistency detection. However, early attempts
have shown that LLMs underperform traditional models due to their limited
ability to follow instructions and the absence of an effective detection
methodology. In this study, we reassess summary inconsistency detection with
LLMs, comparing the performances of GPT-3.5 and GPT-4. To advance research in
LLM-based inconsistency detection, we propose SIFiD (Summary Inconsistency
Detection with Filtered Document) that identify key sentences within documents
by either employing natural language inference or measuring semantic similarity
between summaries and documents.


---

**[32. [2503.18293] Fact-checking AI-generated news reports: Can LLMs catch their own lies?](https://arxiv.org/pdf/2503.18293.pdf)** (2025-03-25)

*Jiayi Yao, Haibo Sun, Nianwen Xue*

  In this paper, we evaluate the ability of Large Language Models (LLMs) to
assess the veracity of claims in ''news reports'' generated by themselves or
other LLMs. Our goal is to determine whether LLMs can effectively fact-check
their own content, using methods similar to those used to verify claims made by
humans. Our findings indicate that LLMs are more effective at assessing claims
in national or international news stories than in local news stories, better at
evaluating static information than dynamic information, and better at verifying
true claims compared to false ones. We hypothesize that this disparity arises
because the former types of claims are better represented in the training data.
Additionally, we find that incorporating retrieved results from a search engine
in a Retrieval-Augmented Generation (RAG) setting significantly reduces the
number of claims an LLM cannot assess. However, this approach also increases
the occurrence of incorrect assessments, partly due to irrelevant or
low-quality search results. This diagnostic study highlights the need for
future research on fact-checking machine-generated reports to prioritize
improving the precision and relevance of retrieved information to better
support fact-checking efforts. Furthermore, claims about dynamic events and
local news may require human-in-the-loop fact-checking systems to ensure
accuracy and reliability.


---

**[33. [2402.00386] AssertLLM: Generating and Evaluating Hardware Verification Assertions
  from Design Specifications via Multi-LLMs](https://arxiv.org/pdf/2402.00386.pdf)** (2024-11-05)

*Wenji Fang, Mengming Li, Min Li, Zhiyuan Yan, Shang Liu, Zhiyao Xie, Hongce Zhang*

  Assertion-based verification (ABV) is a critical method for ensuring design
circuits comply with their architectural specifications, which are typically
described in natural language. This process often requires human interpretation
by verification engineers to convert these specifications into functional
verification assertions. Existing methods for generating assertions from
natural language specifications are limited to sentences extracted by
engineers, discouraging its practical application. In this work, we present
AssertLLM, an automatic assertion generation framework that processes complete
specification files. AssertLLM breaks down the complex task into three phases,
incorporating three customized Large Language Models (LLMs) for extracting
structural specifications, mapping signal definitions, and generating
assertions. Our evaluation of AssertLLM on a full design, encompassing 23 I/O
signals, demonstrates that 89\% of the generated assertions are both
syntactically and functionally accurate.


---

**[34. [2308.09267] GraphReason: Enhancing Reasoning Capabilities of Large Language Models
  through A Graph-Based Verification Approach](https://arxiv.org/pdf/2308.09267.pdf)** (2024-04-23)

*Lang Cao*

  Large Language Models (LLMs) have showcased impressive reasoning
capabilities, particularly when guided by specifically designed prompts in
complex reasoning tasks such as math word problems. These models typically
solve tasks using a chain-of-thought approach, which not only bolsters their
reasoning abilities but also provides valuable insights into their
problem-solving process. However, there is still significant room for enhancing
the reasoning abilities of LLMs. Some studies suggest that the integration of
an LLM output verifier can boost reasoning accuracy without necessitating
additional model training. In this paper, we follow these studies and introduce
a novel graph-based method to further augment the reasoning capabilities of
LLMs. We posit that multiple solutions to a reasoning task, generated by an
LLM, can be represented as a reasoning graph due to the logical connections
between intermediate steps from different reasoning paths. Therefore, we
propose the Reasoning Graph Verifier (GraphReason) to analyze and verify the
solutions generated by LLMs. By evaluating these graphs, models can yield more
accurate and reliable results.Our experimental results show that our
graph-based verification method not only significantly enhances the reasoning
abilities of LLMs but also outperforms existing verifier methods in terms of
improving these models' reasoning performance.


---

**[35. [2311.07838] LLatrieval: LLM-Verified Retrieval for Verifiable Generation](https://arxiv.org/pdf/2311.07838.pdf)** (2024-03-28)

*Xiaonan Li, Changtai Zhu, Linyang Li, Zhangyue Yin, Tianxiang Sun, Xipeng Qiu*

  Verifiable generation aims to let the large language model (LLM) generate
text with supporting documents, which enables the user to flexibly verify the
answer and makes the LLM's output more reliable. Retrieval plays a crucial role
in verifiable generation. Specifically, the retrieved documents not only
supplement knowledge to help the LLM generate correct answers, but also serve
as supporting evidence for the user to verify the LLM's output. However, the
widely used retrievers become the bottleneck of the entire pipeline and limit
the overall performance. Their capabilities are usually inferior to LLMs since
they often have much fewer parameters than the large language model and have
not been demonstrated to scale well to the size of LLMs. If the retriever does
not correctly find the supporting documents, the LLM can not generate the
correct and verifiable answer, which overshadows the LLM's remarkable
abilities. To address these limitations, we propose \LLatrieval (Large Language
Model Verified Retrieval), where the LLM updates the retrieval result until it
verifies that the retrieved documents can sufficiently support answering the
question. Thus, the LLM can iteratively provide feedback to retrieval and
facilitate the retrieval result to fully support verifiable generation.
Experiments show that LLatrieval significantly outperforms extensive baselines
and achieves state-of-the-art results.


---

**[36. [2404.12174] Claim Check-Worthiness Detection: How Well do LLMs Grasp Annotation
  Guidelines?](https://arxiv.org/pdf/2404.12174.pdf)** (2024-10-22)

*Laura Majer, Jan Šnajder*

  The increasing threat of disinformation calls for automating parts of the
fact-checking pipeline. Identifying text segments requiring fact-checking is
known as claim detection (CD) and claim check-worthiness detection (CW), the
latter incorporating complex domain-specific criteria of worthiness and often
framed as a ranking task. Zero- and few-shot LLM prompting is an attractive
option for both tasks, as it bypasses the need for labeled datasets and allows
verbalized claim and worthiness criteria to be directly used for prompting. We
evaluate the LLMs' predictive and calibration accuracy on five CD/CW datasets
from diverse domains, each utilizing a different worthiness criterion. We
investigate two key aspects: (1) how best to distill factuality and worthiness
criteria into a prompt and (2) what amount of context to provide for each
claim. To this end, we experiment with varying the level of prompt verbosity
and the amount of contextual information provided to the model. Our results
show that optimal prompt verbosity is domain-dependent, adding context does not
improve performance, and confidence scores can be directly used to produce
reliable check-worthiness rankings.


---

**[37. [2411.00784] FIRE: Fact-checking with Iterative Retrieval and Verification](https://arxiv.org/pdf/2411.00784.pdf)** (2025-02-13)

*Zhuohan Xie, Rui Xing, Yuxia Wang, Jiahui Geng, Hasan Iqbal, Dhruv Sahnan, Iryna Gurevych, Preslav Nakov*

  Fact-checking long-form text is challenging, and it is therefore common
practice to break it down into multiple atomic claims. The typical approach to
fact-checking these atomic claims involves retrieving a fixed number of pieces
of evidence, followed by a verification step. However, this method is usually
not cost-effective, as it underutilizes the verification model's internal
knowledge of the claim and fails to replicate the iterative reasoning process
in human search strategies. To address these limitations, we propose FIRE, a
novel agent-based framework that integrates evidence retrieval and claim
verification in an iterative manner. Specifically, FIRE employs a unified
mechanism to decide whether to provide a final answer or generate a subsequent
search query, based on its confidence in the current judgment. We compare FIRE
with other strong fact-checking frameworks and find that it achieves slightly
better performance while reducing large language model (LLM) costs by an
average of 7.6 times and search costs by 16.5 times. These results indicate
that FIRE holds promise for application in large-scale fact-checking
operations. Our code is available at https://github.com/mbzuai-nlp/fire.git.


---

**[38. [2402.01733] Development and Testing of Retrieval Augmented Generation in Large
  Language Models -- A Case Study Report](https://arxiv.org/pdf/2402.01733.pdf)** (2024-02-06)

*YuHe Ke, Liyuan Jin, Kabilan Elangovan, Hairil Rizal Abdullah, Nan Liu, Alex Tiong Heng Sia, Chai Rick Soh, Joshua Yi Min Tung, Jasmine Chiat Ling Ong, Daniel Shu Wei Ting*

  Purpose: Large Language Models (LLMs) hold significant promise for medical
applications. Retrieval Augmented Generation (RAG) emerges as a promising
approach for customizing domain knowledge in LLMs. This case study presents the
development and evaluation of an LLM-RAG pipeline tailored for healthcare,
focusing specifically on preoperative medicine.
  Methods: We developed an LLM-RAG model using 35 preoperative guidelines and
tested it against human-generated responses, with a total of 1260 responses
evaluated. The RAG process involved converting clinical documents into text
using Python-based frameworks like LangChain and Llamaindex, and processing
these texts into chunks for embedding and retrieval. Vector storage techniques
and selected embedding models to optimize data retrieval, using Pinecone for
vector storage with a dimensionality of 1536 and cosine similarity for loss
metrics. Human-generated answers, provided by junior doctors, were used as a
comparison.
  Results: The LLM-RAG model generated answers within an average of 15-20
seconds, significantly faster than the 10 minutes typically required by humans.
Among the basic LLMs, GPT4.0 exhibited the best accuracy of 80.1%. This
accuracy was further increased to 91.4% when the model was enhanced with RAG.
Compared to the human-generated instructions, which had an accuracy of 86.3%,
the performance of the GPT4.0 RAG model demonstrated non-inferiority (p=0.610).
  Conclusions: In this case study, we demonstrated a LLM-RAG model for
healthcare implementation. The pipeline shows the advantages of grounded
knowledge, upgradability, and scalability as important aspects of healthcare
LLM deployment.


---

**[39. [2502.08909] Towards Automated Fact-Checking of Real-World Claims: Exploring Task
  Formulation and Assessment with LLMs](https://arxiv.org/pdf/2502.08909.pdf)** (2025-02-14)

*Premtim Sahitaj, Iffat Maab, Junichi Yamagishi, Jawan Kolanowski, Sebastian Möller, Vera Schmitt*

  Fact-checking is necessary to address the increasing volume of
misinformation. Traditional fact-checking relies on manual analysis to verify
claims, but it is slow and resource-intensive. This study establishes baseline
comparisons for Automated Fact-Checking (AFC) using Large Language Models
(LLMs) across multiple labeling schemes (binary, three-class, five-class) and
extends traditional claim verification by incorporating analysis, verdict
classification, and explanation in a structured setup to provide comprehensive
justifications for real-world claims. We evaluate Llama-3 models of varying
sizes (3B, 8B, 70B) on 17,856 claims collected from PolitiFact (2007-2024)
using evidence retrieved via restricted web searches. We utilize TIGERScore as
a reference-free evaluation metric to score the justifications. Our results
show that larger LLMs consistently outperform smaller LLMs in classification
accuracy and justification quality without fine-tuning. We find that smaller
LLMs in a one-shot scenario provide comparable task performance to fine-tuned
Small Language Models (SLMs) with large context sizes, while larger LLMs
consistently surpass them. Evidence integration improves performance across all
models, with larger LLMs benefiting most. Distinguishing between nuanced labels
remains challenging, emphasizing the need for further exploration of labeling
schemes and alignment with evidences. Our findings demonstrate the potential of
retrieval-augmented AFC with LLMs.


---

**[40. [2410.08431] oRetrieval Augmented Generation for 10 Large Language Models and its
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

**[41. [2502.17898] VeriPlan: Integrating Formal Verification and LLMs into End-User
  Planning](https://arxiv.org/pdf/2502.17898.pdf)** (2025-02-26)

*Christine Lee, David Porfirio, Xinyu Jessica Wang, Kevin Zhao, Bilge Mutlu*

  Automated planning is traditionally the domain of experts, utilized in fields
like manufacturing and healthcare with the aid of expert planning tools. Recent
advancements in LLMs have made planning more accessible to everyday users due
to their potential to assist users with complex planning tasks. However, LLMs
face several application challenges within end-user planning, including
consistency, accuracy, and user trust issues. This paper introduces VeriPlan, a
system that applies formal verification techniques, specifically model
checking, to enhance the reliability and flexibility of LLMs for end-user
planning. In addition to the LLM planner, VeriPlan includes three additional
core features -- a rule translator, flexibility sliders, and a model checker --
that engage users in the verification process. Through a user study (n=12), we
evaluate VeriPlan, demonstrating improvements in the perceived quality,
usability, and user satisfaction of LLMs. Our work shows the effective
integration of formal verification and user-control features with LLMs for
end-user planning tasks.


---

**[42. [2411.14436] AssertLLM: Generating Hardware Verification Assertions from Design
  Specifications via Multi-LLMs](https://arxiv.org/pdf/2411.14436.pdf)** (2024-11-25)

*Zhiyuan Yan, Wenji Fang, Mengming Li, Min Li, Shang Liu, Zhiyao Xie, Hongce Zhang*

  Assertion-based verification (ABV) is a critical method to ensure logic
designs comply with their architectural specifications. ABV requires
assertions, which are generally converted from specifications through human
interpretation by verification engineers. Existing methods for generating
assertions from specification documents are limited to sentences extracted by
engineers, discouraging their practical applications. In this work, we present
AssertLLM, an automatic assertion generation framework that processes complete
specification documents. AssertLLM can generate assertions from both natural
language and waveform diagrams in specification files. It first converts
unstructured specification sentences and waveforms into structured descriptions
using natural language templates. Then, a customized Large Language Model (LLM)
generates the final assertions based on these descriptions. Our evaluation
demonstrates that AssertLLM can generate more accurate and higher-quality
assertions compared to GPT-4o and GPT-3.5.


---

**[43. [2503.02003] HoT: Highlighted Chain of Thought for Referencing Supporting Facts from
  Inputs](https://arxiv.org/pdf/2503.02003.pdf)** (2025-03-06)

*Tin Nguyen, Logan Bolton, Mohammad Reza Taesiri, Anh Totti Nguyen*

  An Achilles heel of Large Language Models (LLMs) is their tendency to
hallucinate non-factual statements. A response mixed of factual and non-factual
statements poses a challenge for humans to verify and accurately base their
decisions on. To combat this problem, we propose Highlighted Chain-of-Thought
Prompting (HoT), a technique for prompting LLMs to generate responses with XML
tags that ground facts to those provided in the query. That is, given an input
question, LLMs would first re-format the question to add XML tags highlighting
key facts, and then, generate a response with highlights over the facts
referenced from the input. Interestingly, in few-shot settings, HoT outperforms
vanilla chain of thought prompting (CoT) on a wide range of 17 tasks from
arithmetic, reading comprehension to logical reasoning. When asking humans to
verify LLM responses, highlights help time-limited participants to more
accurately and efficiently recognize when LLMs are correct. Yet, surprisingly,
when LLMs are wrong, HoTs tend to make users believe that an answer is correct.


---

**[44. [2402.14690] UFO: a Unified and Flexible Framework for Evaluating Factuality of Large
  Language Models](https://arxiv.org/pdf/2402.14690.pdf)** (2024-02-23)

*Zhaoheng Huang, Zhicheng Dou, Yutao Zhu, Ji-rong Wen*

  Large language models (LLMs) may generate text that lacks consistency with
human knowledge, leading to factual inaccuracies or \textit{hallucination}.
Existing research for evaluating the factuality of LLMs involves extracting
fact claims using an LLM and verifying them against a predefined fact source.
However, these evaluation metrics are task-specific, and not scalable, and the
substitutability of fact sources in different tasks is under-explored. To
address these challenges, we categorize four available fact sources:
human-written evidence, reference documents, search engine results, and LLM
knowledge, along with five text generation tasks containing six representative
datasets. Then, we propose \texttt{UFO}, an LLM-based unified and flexible
evaluation framework to verify facts against plug-and-play fact sources. We
implement five evaluation scenarios based on this framework. Experimental
results show that for most QA tasks, human-written evidence and reference
documents are crucial, and they can substitute for each other in
retrieval-augmented QA tasks. In news fact generation tasks, search engine
results and LLM knowledge are essential. Our dataset and code are available at
\url{https://github.com/WaldenRUC/UFO}.


---

**[45. [2502.17924] FACT-AUDIT: An Adaptive Multi-Agent Framework for Dynamic Fact-Checking
  Evaluation of Large Language Models](https://arxiv.org/pdf/2502.17924.pdf)** (2025-03-04)

*Hongzhan Lin, Yang Deng, Yuxuan Gu, Wenxuan Zhang, Jing Ma, See-Kiong Ng, Tat-Seng Chua*

  Large Language Models (LLMs) have significantly advanced the fact-checking
studies. However, existing automated fact-checking evaluation methods rely on
static datasets and classification metrics, which fail to automatically
evaluate the justification production and uncover the nuanced limitations of
LLMs in fact-checking. In this work, we introduce FACT-AUDIT, an agent-driven
framework that adaptively and dynamically assesses LLMs' fact-checking
capabilities. Leveraging importance sampling principles and multi-agent
collaboration, FACT-AUDIT generates adaptive and scalable datasets, performs
iterative model-centric evaluations, and updates assessments based on
model-specific responses. By incorporating justification production alongside
verdict prediction, this framework provides a comprehensive and evolving audit
of LLMs' factual reasoning capabilities, to investigate their trustworthiness.
Extensive experiments demonstrate that FACT-AUDIT effectively differentiates
among state-of-the-art LLMs, providing valuable insights into model strengths
and limitations in model-centric fact-checking analysis.


---

**[46. [2402.18045] Multi-FAct: Assessing Factuality of Multilingual LLMs using FActScore](https://arxiv.org/pdf/2402.18045.pdf)** (2024-10-04)

*Sheikh Shafayat, Eunsu Kim, Juhyun Oh, Alice Oh*

  Evaluating the factuality of long-form large language model (LLM)-generated
text is an important challenge. Recently there has been a surge of interest in
factuality evaluation for English, but little is known about the factuality
evaluation of multilingual LLMs, specially when it comes to long-form
generation. %This paper systematically evaluates multilingual LLMs' factual
accuracy across languages and geographic regions. We introduce a simple
pipeline for multilingual factuality evaluation, by applying FActScore (Min et
al., 2023) for diverse languages. In addition to evaluating multilingual
factual generation, we evaluate the factual accuracy of long-form text
generation in topics that reflect regional diversity. We also examine the
feasibility of running the FActScore pipeline using non-English Wikipedia and
provide comprehensive guidelines on multilingual factual evaluation for
regionally diverse topics.


---

**[47. [2404.03623] Unveiling LLMs: The Evolution of Latent Representations in a Dynamic
  Knowledge Graph](https://arxiv.org/pdf/2404.03623.pdf)** (2024-08-07)

*Marco Bronzini, Carlo Nicolini, Bruno Lepri, Jacopo Staiano, Andrea Passerini*

  Large Language Models (LLMs) demonstrate an impressive capacity to recall a
vast range of factual knowledge. However, understanding their underlying
reasoning and internal mechanisms in exploiting this knowledge remains a key
research area. This work unveils the factual information an LLM represents
internally for sentence-level claim verification. We propose an end-to-end
framework to decode factual knowledge embedded in token representations from a
vector space to a set of ground predicates, showing its layer-wise evolution
using a dynamic knowledge graph. Our framework employs activation patching, a
vector-level technique that alters a token representation during inference, to
extract encoded knowledge. Accordingly, we neither rely on training nor
external models. Using factual and common-sense claims from two claim
verification datasets, we showcase interpretability analyses at local and
global levels. The local analysis highlights entity centrality in LLM
reasoning, from claim-related information and multi-hop reasoning to
representation errors causing erroneous evaluation. On the other hand, the
global reveals trends in the underlying evolution, such as word-based knowledge
evolving into claim-related facts. By interpreting semantics from LLM latent
representations and enabling graph-related analyses, this work enhances the
understanding of the factual knowledge resolution process.


---

**[48. [2410.15737] Who's Who: Large Language Models Meet Knowledge Conflicts in Practice](https://arxiv.org/pdf/2410.15737.pdf)** (2024-10-22)

*Quang Hieu Pham, Hoang Ngo, Anh Tuan Luu, Dat Quoc Nguyen*

  Retrieval-augmented generation (RAG) methods are viable solutions for
addressing the static memory limits of pre-trained language models.
Nevertheless, encountering conflicting sources of information within the
retrieval context is an inevitable practical challenge. In such situations, the
language models are recommended to transparently inform users about the
conflicts rather than autonomously deciding what to present based on their
inherent biases. To analyze how current large language models (LLMs) align with
our recommendation, we introduce WhoQA, a public benchmark dataset to examine
model's behavior in knowledge conflict situations. We induce conflicts by
asking about a common property among entities having the same name, resulting
in questions with up to 8 distinctive answers. WhoQA evaluation set includes 5K
questions across 13 Wikidata property types and 150K Wikipedia entities. Our
experiments show that despite the simplicity of WhoQA questions, knowledge
conflicts significantly degrades LLMs' performance in RAG settings.


---

**[49. [2410.05801] Retrieving, Rethinking and Revising: The Chain-of-Verification Can
  Improve Retrieval Augmented Generation](https://arxiv.org/pdf/2410.05801.pdf)** (2024-10-10)

*Bolei He, Nuo Chen, Xinran He, Lingyong Yan, Zhenkai Wei, Jinchang Luo, Zhen-Hua Ling*

  Recent Retrieval Augmented Generation (RAG) aims to enhance Large Language
Models (LLMs) by incorporating extensive knowledge retrieved from external
sources. However, such approach encounters some challenges: Firstly, the
original queries may not be suitable for precise retrieval, resulting in
erroneous contextual knowledge; Secondly, the language model can easily
generate inconsistent answer with external references due to their knowledge
boundary limitation. To address these issues, we propose the
chain-of-verification (CoV-RAG) to enhance the external retrieval correctness
and internal generation consistency. Specifically, we integrate the
verification module into the RAG, engaging in scoring, judgment, and rewriting.
To correct external retrieval errors, CoV-RAG retrieves new knowledge using a
revised query. To correct internal generation errors, we unify QA and
verification tasks with a Chain-of-Thought (CoT) reasoning during training. Our
comprehensive experiments across various LLMs demonstrate the effectiveness and
adaptability compared with other strong baselines. Especially, our CoV-RAG can
significantly surpass the state-of-the-art baselines using different LLM
backbones.


---

**[50. [2312.16374] LLM Factoscope: Uncovering LLMs' Factual Discernment through Inner
  States Analysis](https://arxiv.org/pdf/2312.16374.pdf)** (2024-07-19)

*Jinwen He, Yujia Gong, Kai Chen, Zijin Lin, Chengan Wei, Yue Zhao*

  Large Language Models (LLMs) have revolutionized various domains with
extensive knowledge and creative capabilities. However, a critical issue with
LLMs is their tendency to produce outputs that diverge from factual reality.
This phenomenon is particularly concerning in sensitive applications such as
medical consultation and legal advice, where accuracy is paramount. In this
paper, we introduce the LLM factoscope, a novel Siamese network-based model
that leverages the inner states of LLMs for factual detection. Our
investigation reveals distinguishable patterns in LLMs' inner states when
generating factual versus non-factual content. We demonstrate the LLM
factoscope's effectiveness across various architectures, achieving over 96%
accuracy in factual detection. Our work opens a new avenue for utilizing LLMs'
inner states for factual detection and encourages further exploration into
LLMs' inner workings for enhanced reliability and transparency.


---

**[51. [2404.11086] ViLLM-Eval: A Comprehensive Evaluation Suite for Vietnamese Large
  Language Models](https://arxiv.org/pdf/2404.11086.pdf)** (2024-04-19)

*Trong-Hieu Nguyen, Anh-Cuong Le, Viet-Cuong Nguyen*

  The rapid advancement of large language models (LLMs) necessitates the
development of new benchmarks to accurately assess their capabilities. To
address this need for Vietnamese, this work aims to introduce ViLLM-Eval, the
comprehensive evaluation suite designed to measure the advanced knowledge and
reasoning abilities of foundation models within a Vietnamese context.
ViLLM-Eval consists of multiple-choice questions and predict next word tasks
spanning various difficulty levels and diverse disciplines, ranging from
humanities to science and engineering. A thorough evaluation of the most
advanced LLMs on ViLLM-Eval revealed that even the best performing models have
significant room for improvement in understanding and responding to Vietnamese
language tasks. ViLLM-Eval is believed to be instrumental in identifying key
strengths and weaknesses of foundation models, ultimately promoting their
development and enhancing their performance for Vietnamese users. This paper
provides a thorough overview of ViLLM-Eval as part of the Vietnamese Large
Language Model shared task, held within the 10th International Workshop on
Vietnamese Language and Speech Processing (VLSP 2023).


---

**[52. [2503.09011] Word2winners at SemEval-2025 Task 7: Multilingual and Crosslingual
  Fact-Checked Claim Retrieval](https://arxiv.org/pdf/2503.09011.pdf)** (2025-03-13)

*Amirmohammad Azadi, Sina Zamani, Mohammadmostafa Rostamkhani, Sauleh Eetemadi*

  This paper describes our system for SemEval 2025 Task 7: Previously
Fact-Checked Claim Retrieval. The task requires retrieving relevant fact-checks
for a given input claim from the extensive, multilingual MultiClaim dataset,
which comprises social media posts and fact-checks in several languages. To
address this challenge, we first evaluated zero-shot performance using
state-of-the-art English and multilingual retrieval models and then fine-tuned
the most promising systems, leveraging machine translation to enhance
crosslingual retrieval. Our best model achieved an accuracy of 85% on
crosslingual data and 92% on monolingual data.


---

**[53. [2312.01858] Evaluating Dependencies in Fact Editing for Language Models: Specificity
  and Implication Awareness](https://arxiv.org/pdf/2312.01858.pdf)** (2023-12-05)

*Zichao Li, Ines Arous, Siva Reddy, Jackie C. K. Cheung*

  The potential of using a large language model (LLM) as a knowledge base (KB)
has sparked significant interest. To manage the knowledge acquired by LLMs, we
need to ensure that the editing of learned facts respects internal logical
constraints, which are known as dependency of knowledge. Existing work on
editing LLMs has partially addressed the issue of dependency, when the editing
of a fact should apply to its lexical variations without disrupting irrelevant
ones. However, they neglect the dependency between a fact and its logical
implications. We propose an evaluation protocol with an accompanying
question-answering dataset, DepEdit, that provides a comprehensive assessment
of the editing process considering the above notions of dependency. Our
protocol involves setting up a controlled environment in which we edit facts
and monitor their impact on LLMs, along with their implications based on
If-Then rules. Extensive experiments on DepEdit show that existing knowledge
editing methods are sensitive to the surface form of knowledge, and that they
have limited performance in inferring the implications of edited facts.


---

**[54. [2311.09000] Factcheck-Bench: Fine-Grained Evaluation Benchmark for Automatic
  Fact-checkers](https://arxiv.org/pdf/2311.09000.pdf)** (2024-04-17)

*Yuxia Wang, Revanth Gangi Reddy, Zain Muhammad Mujahid, Arnav Arora, Aleksandr Rubashevskii, Jiahui Geng, Osama Mohammed Afzal, Liangming Pan, Nadav Borenstein, Aditya Pillai, Isabelle Augenstein, Iryna Gurevych, Preslav Nakov*

  The increased use of large language models (LLMs) across a variety of
real-world applications calls for mechanisms to verify the factual accuracy of
their outputs. In this work, we present a holistic end-to-end solution for
annotating the factuality of LLM-generated responses, which encompasses a
multi-stage annotation scheme designed to yield detailed labels concerning the
verifiability and factual inconsistencies found in LLM outputs. We further
construct an open-domain document-level factuality benchmark in three-level
granularity: claim, sentence and document, aiming to facilitate the evaluation
of automatic fact-checking systems. Preliminary experiments show that FacTool,
FactScore and Perplexity.ai are struggling to identify false claims, with the
best F1=0.63 by this annotation solution based on GPT-4. Annotation tool,
benchmark and code are available at https://github.com/yuxiaw/Factcheck-GPT.


---

**[55. [2409.13082] AutoVerus: Automated Proof Generation for Rust Code](https://arxiv.org/pdf/2409.13082.pdf)** (2025-02-11)

*Chenyuan Yang, Xuheng Li, Md Rakib Hossain Misu, Jianan Yao, Weidong Cui, Yeyun Gong, Chris Hawblitzel, Shuvendu Lahiri, Jacob R. Lorch, Shuai Lu, Fan Yang, Ziqiao Zhou, Shan Lu*

  Generative AI has shown its values for many software engineering tasks. Still
in its infancy, large language model (LLM)-based proof generation lags behind
LLM-based code generation. In this paper, we present AutoVerus. AutoVerus uses
LLM to automatically generate correctness proof for Rust code. AutoVerus is
designed to match the unique features of Verus, a verification tool that can
prove the correctness of Rust code using proofs and specifications also written
in Rust. AutoVerus consists of a network of LLM agents that are crafted and
orchestrated to mimic human experts' three phases of proof construction:
preliminary proof generation, proof refinement guided by generic tips, and
proof debugging guided by verification errors. To thoroughly evaluate AutoVerus
and help foster future research in this direction, we have built a benchmark
suite of 150 non-trivial proof tasks, based on existing code-generation
benchmarks and verification benchmarks. Our evaluation shows that AutoVerus can
automatically generate correct proof for more than 90% of them, with more than
half of them tackled in less than 30 seconds or 3 LLM calls.


---

**[56. [2310.17119] FLEEK: Factual Error Detection and Correction with Evidence Retrieved
  from External Knowledge](https://arxiv.org/pdf/2310.17119.pdf)** (2023-10-27)

*Farima Fatahi Bayat, Kun Qian, Benjamin Han, Yisi Sang, Anton Belyi, Samira Khorshidi, Fei Wu, Ihab F. Ilyas, Yunyao Li*

  Detecting factual errors in textual information, whether generated by large
language models (LLM) or curated by humans, is crucial for making informed
decisions. LLMs' inability to attribute their claims to external knowledge and
their tendency to hallucinate makes it difficult to rely on their responses.
Humans, too, are prone to factual errors in their writing. Since manual
detection and correction of factual errors is labor-intensive, developing an
automatic approach can greatly reduce human effort. We present FLEEK, a
prototype tool that automatically extracts factual claims from text, gathers
evidence from external knowledge sources, evaluates the factuality of each
claim, and suggests revisions for identified errors using the collected
evidence. Initial empirical evaluation on fact error detection (77-85\% F1)
shows the potential of FLEEK. A video demo of FLEEK can be found at
https://youtu.be/NapJFUlkPdQ.


---

**[57. [2411.01022] Provenance: A Light-weight Fact-checker for Retrieval Augmented LLM
  Generation Output](https://arxiv.org/pdf/2411.01022.pdf)** (2024-11-25)

*Hithesh Sankararaman, Mohammed Nasheed Yasin, Tanner Sorensen, Alessandro Di Bari, Andreas Stolcke*

  We present a light-weight approach for detecting nonfactual outputs from
retrieval-augmented generation (RAG). Given a context and putative output, we
compute a factuality score that can be thresholded to yield a binary decision
to check the results of LLM-based question-answering, summarization, or other
systems. Unlike factuality checkers that themselves rely on LLMs, we use
compact, open-source natural language inference (NLI) models that yield a
freely accessible solution with low latency and low cost at run-time, and no
need for LLM fine-tuning. The approach also enables downstream mitigation and
correction of hallucinations, by tracing them back to specific context chunks.
Our experiments show high area under the ROC curve (AUC) across a wide range of
relevant open source datasets, indicating the effectiveness of our method for
fact-checking RAG output.


---

**[58. [2410.22257] FactBench: A Dynamic Benchmark for In-the-Wild Language Model Factuality
  Evaluation](https://arxiv.org/pdf/2410.22257.pdf)** (2025-01-09)

*Farima Fatahi Bayat, Lechen Zhang, Sheza Munir, Lu Wang*

  The rapid adoption of language models (LMs) across diverse applications has
raised concerns about their factuality, i.e., their consistency with real-world
facts. We first present VERIFY (Verification and Evidence RetrIeval for
FactualitY evaluation), a pipeline to evaluate LMs' factuality in real-world
user interactions. VERIFY considers the verifiability of LM-generated content
and categorizes content units as supported, unsupported, or undecidable based
on Web-retrieved evidence. Importantly, factuality judgment by VERIFY
correlates better with human evaluations than existing methods. Using VERIFY,
we identify "hallucination prompts" across diverse topics, i.e., those
eliciting the highest rates of incorrect (unsupported) and inconclusive
(undecidable) LM responses. These prompts form FACTBENCH, a dataset of 1K
prompts across 150 fine-grained topics. Our dataset captures emerging
factuality challenges in real-world LM interactions and can be regularly
updated with new prompts. We benchmark widely-used LMs from GPT, Gemini, and
Llama families on FACTBENCH, yielding the following key findings: (i)
Proprietary models exhibit better factuality, with decreased performance from
Easy to Hard hallucination prompts. (ii) Llama3.1-405B-Instruct shows
comparable or lower factual precision than Llama3.1-70B-Instruct across all
evaluation methods due to its higher subjectivity that leads to more content
labeled as undecidable. (iii) Gemini1.5-Pro shows a significantly higher
refusal rate, with over-refusal in 25% of cases.


---

**[59. [2411.04424] Bayesian Calibration of Win Rate Estimation with LLM Evaluators](https://arxiv.org/pdf/2411.04424.pdf)** (2024-12-25)

*Yicheng Gao, Gonghan Xu, Zhe Wang, Arman Cohan*

  Recent advances in large language models (LLMs) show the potential of using
LLMs as evaluators for assessing the quality of text generations from LLMs.
However, applying LLM evaluators naively to compare or judge between different
systems can lead to unreliable results due to the intrinsic win rate estimation
bias of LLM evaluators. In order to mitigate this problem, we propose two
calibration methods, Bayesian Win Rate Sampling (BWRS) and Bayesian
Dawid-Skene, both of which leverage Bayesian inference to more accurately infer
the true win rate of generative language models. We empirically validate our
methods on six datasets covering story generation, summarization, and
instruction following tasks. We show that both our methods are effective in
improving the accuracy of win rate estimation using LLMs as evaluators,
offering a promising direction for reliable automatic text quality evaluation.


---

**[60. [2407.18367] Robust Claim Verification Through Fact Detection](https://arxiv.org/pdf/2407.18367.pdf)** (2024-07-29)

*Nazanin Jafari, James Allan*

  Claim verification can be a challenging task. In this paper, we present a
method to enhance the robustness and reasoning capabilities of automated claim
verification through the extraction of short facts from evidence. Our novel
approach, FactDetect, leverages Large Language Models (LLMs) to generate
concise factual statements from evidence and label these facts based on their
semantic relevance to the claim and evidence. The generated facts are then
combined with the claim and evidence. To train a lightweight supervised model,
we incorporate a fact-detection task into the claim verification process as a
multitasking approach to improve both performance and explainability. We also
show that augmenting FactDetect in the claim verification prompt enhances
performance in zero-shot claim verification using LLMs. Our method demonstrates
competitive results in the supervised claim verification model by 15% on the F1
score when evaluated for challenging scientific claim verification datasets. We
also demonstrate that FactDetect can be augmented with claim and evidence for
zero-shot prompting (AugFactDetect) in LLMs for verdict prediction. We show
that AugFactDetect outperforms the baseline with statistical significance on
three challenging scientific claim verification datasets with an average of
17.3% performance gain compared to the best performing baselines.


---

**[61. [2310.07521] Survey on Factuality in Large Language Models: Knowledge, Retrieval and
  Domain-Specificity](https://arxiv.org/pdf/2310.07521.pdf)** (2023-12-19)

*Cunxiang Wang, Xiaoze Liu, Yuanhao Yue, Xiangru Tang, Tianhang Zhang, Cheng Jiayang, Yunzhi Yao, Wenyang Gao, Xuming Hu, Zehan Qi, Yidong Wang, Linyi Yang, Jindong Wang, Xing Xie, Zheng Zhang, Yue Zhang*

  This survey addresses the crucial issue of factuality in Large Language
Models (LLMs). As LLMs find applications across diverse domains, the
reliability and accuracy of their outputs become vital. We define the
Factuality Issue as the probability of LLMs to produce content inconsistent
with established facts. We first delve into the implications of these
inaccuracies, highlighting the potential consequences and challenges posed by
factual errors in LLM outputs. Subsequently, we analyze the mechanisms through
which LLMs store and process facts, seeking the primary causes of factual
errors. Our discussion then transitions to methodologies for evaluating LLM
factuality, emphasizing key metrics, benchmarks, and studies. We further
explore strategies for enhancing LLM factuality, including approaches tailored
for specific domains. We focus two primary LLM configurations standalone LLMs
and Retrieval-Augmented LLMs that utilizes external data, we detail their
unique challenges and potential enhancements. Our survey offers a structured
guide for researchers aiming to fortify the factual reliability of LLMs.


---

**[62. [2407.01212] EconNLI: Evaluating Large Language Models on Economics Reasoning](https://arxiv.org/pdf/2407.01212.pdf)** (2024-07-02)

*Yue Guo, Yi Yang*

  Large Language Models (LLMs) are widely used for writing economic analysis
reports or providing financial advice, but their ability to understand economic
knowledge and reason about potential results of specific economic events lacks
systematic evaluation. To address this gap, we propose a new dataset, natural
language inference on economic events (EconNLI), to evaluate LLMs' knowledge
and reasoning abilities in the economic domain. We evaluate LLMs on (1) their
ability to correctly classify whether a premise event will cause a hypothesis
event and (2) their ability to generate reasonable events resulting from a
given premise. Our experiments reveal that LLMs are not sophisticated in
economic reasoning and may generate wrong or hallucinated answers. Our study
raises awareness of the limitations of using LLMs for critical decision-making
involving economic reasoning and analysis. The dataset and codes are available
at https://github.com/Irenehere/EconNLI.


---

**[63. [2404.04302] CBR-RAG: Case-Based Reasoning for Retrieval Augmented Generation in LLMs
  for Legal Question Answering](https://arxiv.org/pdf/2404.04302.pdf)** (2024-04-09)

*Nirmalie Wiratunga, Ramitha Abeyratne, Lasal Jayawardena, Kyle Martin, Stewart Massie, Ikechukwu Nkisi-Orji, Ruvan Weerasinghe, Anne Liret, Bruno Fleisch*

  Retrieval-Augmented Generation (RAG) enhances Large Language Model (LLM)
output by providing prior knowledge as context to input. This is beneficial for
knowledge-intensive and expert reliant tasks, including legal
question-answering, which require evidence to validate generated text outputs.
We highlight that Case-Based Reasoning (CBR) presents key opportunities to
structure retrieval as part of the RAG process in an LLM. We introduce CBR-RAG,
where CBR cycle's initial retrieval stage, its indexing vocabulary, and
similarity knowledge containers are used to enhance LLM queries with
contextually relevant cases. This integration augments the original LLM query,
providing a richer prompt. We present an evaluation of CBR-RAG, and examine
different representations (i.e. general and domain-specific embeddings) and
methods of comparison (i.e. inter, intra and hybrid similarity) on the task of
legal question-answering. Our results indicate that the context provided by
CBR's case reuse enforces similarity between relevant components of the
questions and the evidence base leading to significant improvements in the
quality of generated answers.


---

**[64. [2410.09623] Quebec Automobile Insurance Question-Answering With Retrieval-Augmented
  Generation](https://arxiv.org/pdf/2410.09623.pdf)** (2024-10-15)

*David Beauchemin, Zachary Gagnon, Ricahrd Khoury*

  Large Language Models (LLMs) perform outstandingly in various downstream
tasks, and the use of the Retrieval-Augmented Generation (RAG) architecture has
been shown to improve performance for legal question answering (Nuruzzaman and
Hussain, 2020; Louis et al., 2024). However, there are limited applications in
insurance questions-answering, a specific type of legal document. This paper
introduces two corpora: the Quebec Automobile Insurance Expertise Reference
Corpus and a set of 82 Expert Answers to Layperson Automobile Insurance
Questions. Our study leverages both corpora to automatically and manually
assess a GPT4-o, a state-of-the-art LLM, to answer Quebec automobile insurance
questions. Our results demonstrate that, on average, using our expertise
reference corpus generates better responses on both automatic and manual
evaluation metrics. However, they also highlight that LLM QA is unreliable
enough for mass utilization in critical areas. Indeed, our results show that
between 5% to 13% of answered questions include a false statement that could
lead to customer misunderstanding.


---

**[65. [2410.21330] LLM Robustness Against Misinformation in Biomedical Question Answering](https://arxiv.org/pdf/2410.21330.pdf)** (2024-10-30)

*Alexander Bondarenko, Adrian Viehweger*

  The retrieval-augmented generation (RAG) approach is used to reduce the
confabulation of large language models (LLMs) for question answering by
retrieving and providing additional context coming from external knowledge
sources (e.g., by adding the context to the prompt). However, injecting
incorrect information can mislead the LLM to generate an incorrect answer.
  In this paper, we evaluate the effectiveness and robustness of four LLMs
against misinformation - Gemma 2, GPT-4o-mini, Llama~3.1, and Mixtral - in
answering biomedical questions. We assess the answer accuracy on yes-no and
free-form questions in three scenarios: vanilla LLM answers (no context is
provided), "perfect" augmented generation (correct context is provided), and
prompt-injection attacks (incorrect context is provided). Our results show that
Llama 3.1 (70B parameters) achieves the highest accuracy in both vanilla
(0.651) and "perfect" RAG (0.802) scenarios. However, the accuracy gap between
the models almost disappears with "perfect" RAG, suggesting its potential to
mitigate the LLM's size-related effectiveness differences.
  We further evaluate the ability of the LLMs to generate malicious context on
one hand and the LLM's robustness against prompt-injection attacks on the other
hand, using metrics such as attack success rate (ASR), accuracy under attack,
and accuracy drop. As adversaries, we use the same four LLMs (Gemma 2,
GPT-4o-mini, Llama 3.1, and Mixtral) to generate incorrect context that is
injected in the target model's prompt. Interestingly, Llama is shown to be the
most effective adversary, causing accuracy drops of up to 0.48 for vanilla
answers and 0.63 for "perfect" RAG across target models. Our analysis reveals
that robustness rankings vary depending on the evaluation measure, highlighting
the complexity of assessing LLM resilience to adversarial attacks.


---

**[66. [2503.22877] Understanding Inequality of LLM Fact-Checking over Geographic Regions
  with Agent and Retrieval models](https://arxiv.org/pdf/2503.22877.pdf)** (2025-04-01)

*Bruno Coelho, Shujaat Mirza, Yuyuan Cui, Christina Pöpper, Damon McCoy*

  Fact-checking is a potentially useful application of Large Language Models
(LLMs) to combat the growing dissemination of disinformation. However, the
performance of LLMs varies across geographic regions. In this paper, we
evaluate the factual accuracy of open and private models across a diverse set
of regions and scenarios.
  Using a dataset containing 600 fact-checked statements balanced across six
global regions we examine three experimental setups of fact-checking a
statement: (1) when just the statement is available, (2) when an LLM-based
agent with Wikipedia access is utilized, and (3) as a best case scenario when a
Retrieval-Augmented Generation (RAG) system provided with the official fact
check is employed. Our findings reveal that regardless of the scenario and LLM
used, including GPT-4, Claude Sonnet, and LLaMA, statements from the Global
North perform substantially better than those from the Global South.
Furthermore, this gap is broadened for the more realistic case of a Wikipedia
agent-based system, highlighting that overly general knowledge bases have a
limited ability to address region-specific nuances. These results underscore
the urgent need for better dataset balancing and robust retrieval strategies to
enhance LLM fact-checking capabilities, particularly in geographically diverse
contexts.


---

**[67. [2411.05897] Humans and Large Language Models in Clinical Decision Support: A Study
  with Medical Calculators](https://arxiv.org/pdf/2411.05897.pdf)** (2025-03-25)

*Nicholas Wan, Qiao Jin, Joey Chan, Guangzhi Xiong, Serina Applebaum, Aidan Gilson, Reid McMurry, R. Andrew Taylor, Aidong Zhang, Qingyu Chen, Zhiyong Lu*

  Although large language models (LLMs) have been assessed for general medical
knowledge using licensing exams, their ability to support clinical
decision-making, such as selecting medical calculators, remains uncertain. We
assessed nine LLMs, including open-source, proprietary, and domain-specific
models, with 1,009 multiple-choice question-answer pairs across 35 clinical
calculators and compared LLMs to humans on a subset of questions. While the
highest-performing LLM, OpenAI o1, provided an answer accuracy of 66.0% (CI:
56.7-75.3%) on the subset of 100 questions, two human annotators nominally
outperformed LLMs with an average answer accuracy of 79.5% (CI: 73.5-85.0%).
Ultimately, we evaluated medical trainees and LLMs in recommending medical
calculators across clinical scenarios like risk stratification and diagnosis.
With error analysis showing that the highest-performing LLMs continue to make
mistakes in comprehension (49.3% of errors) and calculator knowledge (7.1% of
errors), our findings highlight that LLMs are not superior to humans in
calculator recommendation.


---

**[68. [2407.02301] CFinBench: A Comprehensive Chinese Financial Benchmark for Large
  Language Models](https://arxiv.org/pdf/2407.02301.pdf)** (2024-07-03)

*Ying Nie, Binwei Yan, Tianyu Guo, Hao Liu, Haoyu Wang, Wei He, Binfan Zheng, Weihao Wang, Qiang Li, Weijian Sun, Yunhe Wang, Dacheng Tao*

  Large language models (LLMs) have achieved remarkable performance on various
NLP tasks, yet their potential in more challenging and domain-specific task,
such as finance, has not been fully explored. In this paper, we present
CFinBench: a meticulously crafted, the most comprehensive evaluation benchmark
to date, for assessing the financial knowledge of LLMs under Chinese context.
In practice, to better align with the career trajectory of Chinese financial
practitioners, we build a systematic evaluation from 4 first-level categories:
(1) Financial Subject: whether LLMs can memorize the necessary basic knowledge
of financial subjects, such as economics, statistics and auditing. (2)
Financial Qualification: whether LLMs can obtain the needed financial qualified
certifications, such as certified public accountant, securities qualification
and banking qualification. (3) Financial Practice: whether LLMs can fulfill the
practical financial jobs, such as tax consultant, junior accountant and
securities analyst. (4) Financial Law: whether LLMs can meet the requirement of
financial laws and regulations, such as tax law, insurance law and economic
law. CFinBench comprises 99,100 questions spanning 43 second-level categories
with 3 question types: single-choice, multiple-choice and judgment. We conduct
extensive experiments of 50 representative LLMs with various model size on
CFinBench. The results show that GPT4 and some Chinese-oriented models lead the
benchmark, with the highest average accuracy being 60.16%, highlighting the
challenge presented by CFinBench. The dataset and evaluation code are available
at https://cfinbench.github.io/.


---

**[69. [2212.09561] Large Language Models are Better Reasoners with Self-Verification](https://arxiv.org/pdf/2212.09561.pdf)** (2023-10-20)

*Yixuan Weng, Minjun Zhu, Fei Xia, Bin Li, Shizhu He, Shengping Liu, Bin Sun, Kang Liu, Jun Zhao*

  Recently, with the chain of thought (CoT) prompting, large language models
(LLMs), e.g., GPT-3, have shown strong reasoning ability in several natural
language processing tasks such as arithmetic, commonsense, and logical
reasoning. However, LLMs with CoT require multi-step prompting and multi-token
prediction, which is highly sensitive to individual mistakes and vulnerable to
error accumulation. The above issues make the LLMs need the ability to verify
the answers. In fact, after inferring conclusions in some thinking decision
tasks, people often check them by re-verifying steps to avoid some mistakes. In
this paper, we propose and prove that LLMs also have similar self-verification
abilities. We take the conclusion obtained by CoT as one of the conditions for
solving the original problem. By performing a backward verification of the
answers that LLM deduced for itself, we can obtain interpretable answer
validation scores to select the candidate answer with the highest score.
Experimental results demonstrate that the proposed method can improve the
reasoning performance on various arithmetic, commonsense, and logical reasoning
datasets. Our code is publicly available at:
https://github.com/WENGSYX/Self-Verification.


---

**[70. [2504.05163] Evaluating Knowledge Graph Based Retrieval Augmented Generation Methods
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

**[71. [2407.01796] Ground Every Sentence: Improving Retrieval-Augmented LLMs with
  Interleaved Reference-Claim Generation](https://arxiv.org/pdf/2407.01796.pdf)** (2024-07-03)

*Sirui Xia, Xintao Wang, Jiaqing Liang, Yifei Zhang, Weikang Zhou, Jiaji Deng, Fei Yu, Yanghua Xiao*

  Retrieval-Augmented Generation (RAG) has been widely adopted to enhance Large
Language Models (LLMs) in knowledge-intensive tasks. Recently, Attributed Text
Generation (ATG) has attracted growing attention, which provides citations to
support the model's responses in RAG, so as to enhance the credibility of
LLM-generated content and facilitate verification. Prior methods mainly adopt
coarse-grained attributions, linking to passage-level references or providing
paragraph-level citations. However, these methods still fall short in
verifiability and require certain time costs for fact checking. This paper
proposes a fine-grained ATG method called ReClaim(Refer & Claim), which
alternates the generation of references and answers step by step. Unlike
traditional coarse-grained attribution, ReClaim allows the model to add
sentence-level fine-grained citations to each answer sentence in long-form
question-answering tasks. Our experiments encompass various training and
inference methods and multiple LLMs, verifying the effectiveness of our
approach.


---

**[72. [2412.15265] Chinese SafetyQA: A Safety Short-form Factuality Benchmark for Large
  Language Models](https://arxiv.org/pdf/2412.15265.pdf)** (2024-12-24)

*Yingshui Tan, Boren Zheng, Baihui Zheng, Kerui Cao, Huiyun Jing, Jincheng Wei, Jiaheng Liu, Yancheng He, Wenbo Su, Xiangyong Zhu, Bo Zheng, Kaifu Zhang*

  With the rapid advancement of Large Language Models (LLMs), significant
safety concerns have emerged. Fundamentally, the safety of large language
models is closely linked to the accuracy, comprehensiveness, and clarity of
their understanding of safety knowledge, particularly in domains such as law,
policy and ethics. This factuality ability is crucial in determining whether
these models can be deployed and applied safely and compliantly within specific
regions. To address these challenges and better evaluate the factuality ability
of LLMs to answer short questions, we introduce the Chinese SafetyQA benchmark.
Chinese SafetyQA has several properties (i.e., Chinese, Diverse, High-quality,
Static, Easy-to-evaluate, Safety-related, Harmless). Based on Chinese SafetyQA,
we perform a comprehensive evaluation on the factuality abilities of existing
LLMs and analyze how these capabilities relate to LLM abilities, e.g., RAG
ability and robustness against attacks.


---

**[73. [2404.09932] Foundational Challenges in Assuring Alignment and Safety of Large
  Language Models](https://arxiv.org/pdf/2404.09932.pdf)** (2024-09-09)

*Usman Anwar, Abulhair Saparov, Javier Rando, Daniel Paleka, Miles Turpin, Peter Hase, Ekdeep Singh Lubana, Erik Jenner, Stephen Casper, Oliver Sourbut, Benjamin L. Edelman, Zhaowei Zhang, Mario Günther, Anton Korinek, Jose Hernandez-Orallo, Lewis Hammond, Eric Bigelow, Alexander Pan, Lauro Langosco, Tomasz Korbak, Heidi Zhang, Ruiqi Zhong, Seán Ó hÉigeartaigh, Gabriel Recchia, Giulio Corsi, Alan Chan, Markus Anderljung, Lilian Edwards, Aleksandar Petrov, Christian Schroeder de Witt, Sumeet Ramesh Motwan, Yoshua Bengio, Danqi Chen, Philip H. S. Torr, Samuel Albanie, Tegan Maharaj, Jakob Foerster, Florian Tramer, He He, Atoosa Kasirzadeh, Yejin Choi, David Krueger*

  This work identifies 18 foundational challenges in assuring the alignment and
safety of large language models (LLMs). These challenges are organized into
three different categories: scientific understanding of LLMs, development and
deployment methods, and sociotechnical challenges. Based on the identified
challenges, we pose $200+$ concrete research questions.


---

**[74. [2406.09136] Chain of Preference Optimization: Improving Chain-of-Thought Reasoning
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

**[75. [2406.13990] Inference-Time Decontamination: Reusing Leaked Benchmarks for Large
  Language Model Evaluation](https://arxiv.org/pdf/2406.13990.pdf)** (2024-06-25)

*Qin Zhu, Qingyuan Cheng, Runyu Peng, Xiaonan Li, Tengxiao Liu, Ru Peng, Xipeng Qiu, Xuanjing Huang*

  The training process of large language models (LLMs) often involves varying
degrees of test data contamination. Although current LLMs are achieving
increasingly better performance on various benchmarks, their performance in
practical applications does not always match their benchmark results. Leakage
of benchmarks can prevent the accurate assessment of LLMs' true performance.
However, constructing new benchmarks is costly, labor-intensive and still
carries the risk of leakage. Therefore, in this paper, we ask the question, Can
we reuse these leaked benchmarks for LLM evaluation? We propose Inference-Time
Decontamination (ITD) to address this issue by detecting and rewriting leaked
samples without altering their difficulties. ITD can mitigate performance
inflation caused by memorizing leaked benchmarks. Our proof-of-concept
experiments demonstrate that ITD reduces inflated accuracy by 22.9% on GSM8K
and 19.0% on MMLU. On MMLU, using Inference-time Decontamination can lead to a
decrease in the results of Phi3 and Mistral by 6.7% and 3.6% respectively. We
hope that ITD can provide more truthful evaluation results for large language
models.


---

**[76. [2403.03888] FaaF: Facts as a Function for the evaluation of generated text](https://arxiv.org/pdf/2403.03888.pdf)** (2024-09-25)

*Vasileios Katranidis, Gabor Barany*

  The demand for accurate and efficient verification of information in texts
generated by large language models (LMs) is at an all-time high, but remains
unresolved. Recent efforts have focused on extracting and verifying atomic
facts from these texts via prompting LM evaluators. However, we demonstrate
that this method of prompting is unreliable when faced with incomplete or
inaccurate reference information. We introduce Facts as a Function (FaaF), a
new approach to the fact verification task that leverages the function-calling
capabilities of LMs. FaaF significantly enhances the ability of LMs to identify
unsupported facts in texts, while also improving efficiency and significantly
lowering costs compared to prompt-based methods. Additionally, we propose a
framework for evaluating factual recall in Retrieval Augmented Generation (RAG)
systems, which we employ to compare prompt-based and FaaF methods using various
LMs under challenging conditions.


---

**[77. [2310.10049] FATE-LLM: A Industrial Grade Federated Learning Framework for Large
  Language Models](https://arxiv.org/pdf/2310.10049.pdf)** (2023-10-17)

*Tao Fan, Yan Kang, Guoqiang Ma, Weijing Chen, Wenbin Wei, Lixin Fan, Qiang Yang*

  Large Language Models (LLMs), such as ChatGPT, LLaMA, GLM, and PaLM, have
exhibited remarkable performances across various tasks in recent years.
However, LLMs face two main challenges in real-world applications. One
challenge is that training LLMs consumes vast computing resources, preventing
LLMs from being adopted by small and medium-sized enterprises with limited
computing resources. Another is that training LLM requires a large amount of
high-quality data, which are often scattered among enterprises. To address
these challenges, we propose FATE-LLM, an industrial-grade federated learning
framework for large language models. FATE-LLM (1) facilitates federated
learning for large language models (coined FedLLM); (2) promotes efficient
training of FedLLM using parameter-efficient fine-tuning methods; (3) protects
the intellectual property of LLMs; (4) preserves data privacy during training
and inference through privacy-preserving mechanisms. We release the code of
FATE-LLM at https://github.com/FederatedAI/FATE-LLM to facilitate the research
of FedLLM and enable a broad range of industrial applications.


---

**[78. [2305.14540] LLMs as Factual Reasoners: Insights from Existing Benchmarks and Beyond](https://arxiv.org/pdf/2305.14540.pdf)** (2023-05-25)

*Philippe Laban, Wojciech Kryściński, Divyansh Agarwal, Alexander R. Fabbri, Caiming Xiong, Shafiq Joty, Chien-Sheng Wu*

  With the recent appearance of LLMs in practical settings, having methods that
can effectively detect factual inconsistencies is crucial to reduce the
propagation of misinformation and improve trust in model outputs. When testing
on existing factual consistency benchmarks, we find that a few large language
models (LLMs) perform competitively on classification benchmarks for factual
inconsistency detection compared to traditional non-LLM methods. However, a
closer analysis reveals that most LLMs fail on more complex formulations of the
task and exposes issues with existing evaluation benchmarks, affecting
evaluation precision. To address this, we propose a new protocol for
inconsistency detection benchmark creation and implement it in a 10-domain
benchmark called SummEdits. This new benchmark is 20 times more cost-effective
per sample than previous benchmarks and highly reproducible, as we estimate
inter-annotator agreement at about 0.9. Most LLMs struggle on SummEdits, with
performance close to random chance. The best-performing model, GPT-4, is still
8\% below estimated human performance, highlighting the gaps in LLMs' ability
to reason about facts and detect inconsistencies when they occur.


---

**[79. [2401.17839] Global-Liar: Factuality of LLMs over Time and Geographic Regions](https://arxiv.org/pdf/2401.17839.pdf)** (2024-02-01)

*Shujaat Mirza, Bruno Coelho, Yuyuan Cui, Christina Pöpper, Damon McCoy*

  The increasing reliance on AI-driven solutions, particularly Large Language
Models (LLMs) like the GPT series, for information retrieval highlights the
critical need for their factuality and fairness, especially amidst the rampant
spread of misinformation and disinformation online. Our study evaluates the
factual accuracy, stability, and biases in widely adopted GPT models, including
GPT-3.5 and GPT-4, contributing to reliability and integrity of AI-mediated
information dissemination.
  We introduce 'Global-Liar,' a dataset uniquely balanced in terms of
geographic and temporal representation, facilitating a more nuanced evaluation
of LLM biases. Our analysis reveals that newer iterations of GPT models do not
always equate to improved performance. Notably, the GPT-4 version from March
demonstrates higher factual accuracy than its subsequent June release.
Furthermore, a concerning bias is observed, privileging statements from the
Global North over the Global South, thus potentially exacerbating existing
informational inequities. Regions such as Africa and the Middle East are at a
disadvantage, with much lower factual accuracy. The performance fluctuations
over time suggest that model updates may not consistently benefit all regions
equally.
  Our study also offers insights into the impact of various LLM configuration
settings, such as binary decision forcing, model re-runs and temperature, on
model's factuality. Models constrained to binary (true/false) choices exhibit
reduced factuality compared to those allowing an 'unclear' option. Single
inference at a low temperature setting matches the reliability of majority
voting across various configurations. The insights gained highlight the need
for culturally diverse and geographically inclusive model training and
evaluation. This approach is key to achieving global equity in technology,
distributing AI benefits fairly worldwide.


---

**[80. [2411.05764] FinDVer: Explainable Claim Verification over Long and Hybrid-Content
  Financial Documents](https://arxiv.org/pdf/2411.05764.pdf)** (2024-11-11)

*Yilun Zhao, Yitao Long, Yuru Jiang, Chengye Wang, Weiyuan Chen, Hongjun Liu, Yiming Zhang, Xiangru Tang, Chen Zhao, Arman Cohan*

  We introduce FinDVer, a comprehensive benchmark specifically designed to
evaluate the explainable claim verification capabilities of LLMs in the context
of understanding and analyzing long, hybrid-content financial documents.
FinDVer contains 2,400 expert-annotated examples, divided into three subsets:
information extraction, numerical reasoning, and knowledge-intensive reasoning,
each addressing common scenarios encountered in real-world financial contexts.
We assess a broad spectrum of LLMs under long-context and RAG settings. Our
results show that even the current best-performing system, GPT-4o, still lags
behind human experts. We further provide in-depth analysis on long-context and
RAG setting, Chain-of-Thought reasoning, and model reasoning errors, offering
insights to drive future advancements. We believe that FinDVer can serve as a
valuable benchmark for evaluating LLMs in claim verification over complex,
expert-domain documents.


---

**[81. [2412.00543] Evaluating the Consistency of LLM Evaluators](https://arxiv.org/pdf/2412.00543.pdf)** (2024-12-03)

*Noah Lee, Jiwoo Hong, James Thorne*

  Large language models (LLMs) have shown potential as general evaluators along
with the evident benefits of speed and cost. While their correlation against
human annotators has been widely studied, consistency as evaluators is still
understudied, raising concerns about the reliability of LLM evaluators. In this
paper, we conduct extensive studies on the two aspects of consistency in LLM
evaluations, Self-Consistency (SC) and Inter-scale Consistency (IC), on
different scoring scales and criterion granularity with open-source and
proprietary models. Our comprehensive analysis demonstrates that strong
proprietary models are not necessarily consistent evaluators, highlighting the
importance of considering consistency in assessing the capability of LLM
evaluators.


---

**[82. [2402.11243] Can Large Language Models perform Relation-based Argument Mining?](https://arxiv.org/pdf/2402.11243.pdf)** (2024-02-20)

*Deniz Gorur, Antonio Rago, Francesca Toni*

  Argument mining (AM) is the process of automatically extracting arguments,
their components and/or relations amongst arguments and components from text.
As the number of platforms supporting online debate increases, the need for AM
becomes ever more urgent, especially in support of downstream tasks.
Relation-based AM (RbAM) is a form of AM focusing on identifying agreement
(support) and disagreement (attack) relations amongst arguments. RbAM is a
challenging classification task, with existing methods failing to perform
satisfactorily. In this paper, we show that general-purpose Large Language
Models (LLMs), appropriately primed and prompted, can significantly outperform
the best performing (RoBERTa-based) baseline. Specifically, we experiment with
two open-source LLMs (Llama-2 and Mistral) with ten datasets.


---

**[83. [2403.09972] Think Twice Before Trusting: Self-Detection for Large Language Models
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

**[84. [2407.05868] KG-FPQ: Evaluating Factuality Hallucination in LLMs with Knowledge
  Graph-based False Premise Questions](https://arxiv.org/pdf/2407.05868.pdf)** (2024-12-24)

*Yanxu Zhu, Jinlin Xiao, Yuhang Wang, Jitao Sang*

  Recent studies have demonstrated that large language models (LLMs) are
susceptible to being misled by false premise questions (FPQs), leading to
errors in factual knowledge, know as factuality hallucination. Existing
benchmarks that assess this vulnerability primarily rely on manual
construction, resulting in limited scale and lack of scalability. In this work,
we introduce an automated, scalable pipeline to create FPQs based on knowledge
graphs (KGs). The first step is modifying true triplets extracted from KGs to
create false premises. Subsequently, utilizing the state-of-the-art
capabilities of GPTs, we generate semantically rich FPQs. Based on the proposed
method, we present a comprehensive benchmark, the Knowledge Graph-based False
Premise Questions (KG-FPQ), which contains approximately 178k FPQs across three
knowledge domains, at six levels of confusability, and in two task formats.
Using KG-FPQ, we conduct extensive evaluations on several representative LLMs
and provide valuable insights. The KG-FPQ dataset and code are available
at~https://github.com/yanxuzhu/KG-FPQ.


---

**[85. [2407.10582] Boosting Zero-Shot Crosslingual Performance using LLM-Based
  Augmentations with Effective Data Selection](https://arxiv.org/pdf/2407.10582.pdf)** (2024-07-16)

*Barah Fazili, Ashish Sunil Agrawal, Preethi Jyothi*

  Large language models (LLMs) are very proficient text generators. We leverage
this capability of LLMs to generate task-specific data via zero-shot prompting
and promote cross-lingual transfer for low-resource target languages. Given
task-specific data in a source language and a teacher model trained on this
data, we propose using this teacher to label LLM generations and employ a set
of simple data selection strategies that use the teacher's label probabilities.
Our data selection strategies help us identify a representative subset of
diverse generations that help boost zero-shot accuracies while being efficient,
in comparison to using all the LLM generations (without any subset selection).
We also highlight other important design choices that affect cross-lingual
performance such as the use of translations of source data and what labels are
best to use for the LLM generations. We observe significant performance gains
across sentiment analysis and natural language inference tasks (of up to a
maximum of 7.13 absolute points and 1.5 absolute points on average) across a
number of target languages (Hindi, Marathi, Urdu, Swahili) and domains.


---

**[86. [2410.04616] Can LLMs Improve Multimodal Fact-Checking by Asking Relevant Questions?](https://arxiv.org/pdf/2410.04616.pdf)** (2025-02-24)

*Alimohammad Beigi, Bohan Jiang, Dawei Li, Zhen Tan, Pouya Shaeri, Tharindu Kumarage, Amrita Bhattacharjee, Huan Liu*

  Traditional fact-checking relies on humans to formulate relevant and targeted
fact-checking questions (FCQs), search for evidence, and verify the factuality
of claims. While Large Language Models (LLMs) have been commonly used to
automate evidence retrieval and factuality verification at scale, their
effectiveness for fact-checking is hindered by the absence of FCQ formulation.
To bridge this gap, we seek to answer two research questions: (1) Can LLMs
generate relevant FCQs? (2) Can LLM-generated FCQs improve multimodal
fact-checking? We therefore introduce a framework LRQ-FACT for using LLMs to
generate relevant FCQs to facilitate evidence retrieval and enhance
fact-checking by probing information across multiple modalities. Through
extensive experiments, we verify if LRQ-FACT can generate relevant FCQs of
different types and if LRQ-FACT can consistently outperform baseline methods in
multimodal fact-checking. Further analysis illustrates how each component in
LRQ-FACT works toward improving the fact-checking performance.


---

**[87. [2501.10860] Zero-shot and Few-shot Learning with Instruction-following LLMs for
  Claim Matching in Automated Fact-checking](https://arxiv.org/pdf/2501.10860.pdf)** (2025-03-04)

*Dina Pisarevskaya, Arkaitz Zubiaga*

  The claim matching (CM) task can benefit an automated fact-checking pipeline
by putting together claims that can be resolved with the same fact-check. In
this work, we are the first to explore zero-shot and few-shot learning
approaches to the task. We consider CM as a binary classification task and
experiment with a set of instruction-following large language models
(GPT-3.5-turbo, Gemini-1.5-flash, Mistral-7B-Instruct, and
Llama-3-8B-Instruct), investigating prompt templates. We introduce a new CM
dataset, ClaimMatch, which will be released upon acceptance. We put LLMs to the
test in the CM task and find that it can be tackled by leveraging more mature
yet similar tasks such as natural language inference or paraphrase detection.
We also propose a pipeline for CM, which we evaluate on texts of different
lengths.


---

**[88. [2502.14765] Step-by-Step Fact Verification System for Medical Claims with
  Explainable Reasoning](https://arxiv.org/pdf/2502.14765.pdf)** (2025-02-21)

*Juraj Vladika, Ivana Hacajová, Florian Matthes*

  Fact verification (FV) aims to assess the veracity of a claim based on
relevant evidence. The traditional approach for automated FV includes a
three-part pipeline relying on short evidence snippets and encoder-only
inference models. More recent approaches leverage the multi-turn nature of LLMs
to address FV as a step-by-step problem where questions inquiring additional
context are generated and answered until there is enough information to make a
decision. This iterative method makes the verification process rational and
explainable. While these methods have been tested for encyclopedic claims,
exploration on domain-specific and realistic claims is missing. In this work,
we apply an iterative FV system on three medical fact-checking datasets and
evaluate it with multiple settings, including different LLMs, external web
search, and structured reasoning using logic predicates. We demonstrate
improvements in the final performance over traditional approaches and the high
potential of step-by-step FV systems for domain-specific claims.


---

**[89. [2502.02896] A Benchmark for the Detection of Metalinguistic Disagreements between
  LLMs and Knowledge Graphs](https://arxiv.org/pdf/2502.02896.pdf)** (2025-02-06)

*Bradley P. Allen, Paul T. Groth*

  Evaluating large language models (LLMs) for tasks like fact extraction in
support of knowledge graph construction frequently involves computing accuracy
metrics using a ground truth benchmark based on a knowledge graph (KG). These
evaluations assume that errors represent factual disagreements. However, human
discourse frequently features metalinguistic disagreement, where agents differ
not on facts but on the meaning of the language used to express them. Given the
complexity of natural language processing and generation using LLMs, we ask: do
metalinguistic disagreements occur between LLMs and KGs? Based on an
investigation using the T-REx knowledge alignment dataset, we hypothesize that
metalinguistic disagreement does in fact occur between LLMs and KGs, with
potential relevance for the practice of knowledge graph engineering. We propose
a benchmark for evaluating the detection of factual and metalinguistic
disagreements between LLMs and KGs. An initial proof of concept of such a
benchmark is available on Github.


---

**[90. [2410.23850] The Automated Verification of Textual Claims (AVeriTeC) Shared Task](https://arxiv.org/pdf/2410.23850.pdf)** (2024-11-01)

*Michael Schlichtkrull, Yulong Chen, Chenxi Whitehouse, Zhenyun Deng, Mubashara Akhtar, Rami Aly, Zhijiang Guo, Christos Christodoulopoulos, Oana Cocarascu, Arpit Mittal, James Thorne, Andreas Vlachos*

  The Automated Verification of Textual Claims (AVeriTeC) shared task asks
participants to retrieve evidence and predict veracity for real-world claims
checked by fact-checkers. Evidence can be found either via a search engine, or
via a knowledge store provided by the organisers. Submissions are evaluated
using AVeriTeC score, which considers a claim to be accurately verified if and
only if both the verdict is correct and retrieved evidence is considered to
meet a certain quality threshold. The shared task received 21 submissions, 18
of which surpassed our baseline. The winning team was TUDA_MAI with an AVeriTeC
score of 63%. In this paper we describe the shared task, present the full
results, and highlight key takeaways from the shared task.


---

**[91. [2501.06211] FLAME: Financial Large-Language Model Assessment and Metrics Evaluation](https://arxiv.org/pdf/2501.06211.pdf)** (2025-01-14)

*Jiayu Guo, Yu Guo, Martha Li, Songtao Tan*

  LLMs have revolutionized NLP and demonstrated potential across diverse
domains. More and more financial LLMs have been introduced for finance-specific
tasks, yet comprehensively assessing their value is still challenging. In this
paper, we introduce FLAME, a comprehensive financial LLMs evaluation system in
Chinese, which includes two core evaluation benchmarks: FLAME-Cer and
FLAME-Sce. FLAME-Cer covers 14 types of authoritative financial certifications,
including CPA, CFA, and FRM, with a total of approximately 16,000 carefully
selected questions. All questions have been manually reviewed to ensure
accuracy and representativeness. FLAME-Sce consists of 10 primary core
financial business scenarios, 21 secondary financial business scenarios, and a
comprehensive evaluation set of nearly 100 tertiary financial application
tasks. We evaluate 6 representative LLMs, including GPT-4o, GLM-4, ERNIE-4.0,
Qwen2.5, XuanYuan3, and the latest Baichuan4-Finance, revealing
Baichuan4-Finance excels other LLMs in most tasks. By establishing a
comprehensive and professional evaluation system, FLAME facilitates the
advancement of financial LLMs in Chinese contexts. Instructions for
participating in the evaluation are available on GitHub:
https://github.com/FLAME-ruc/FLAME.


---

**[92. [2411.09689] LLM Hallucination Reasoning with Zero-shot Knowledge Test](https://arxiv.org/pdf/2411.09689.pdf)** (2024-11-15)

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

**[93. [2411.05641] Evaluating Large Language Model Capability in Vietnamese Fact-Checking
  Data Generation](https://arxiv.org/pdf/2411.05641.pdf)** (2024-11-11)

*Long Truong To, Hung Tuan Le, Dat Van-Thanh Nguyen, Manh Trong Nguyen, Tri Thien Nguyen, Tin Van Huynh, Kiet Van Nguyen*

  Large Language Models (LLMs), with gradually improving reading comprehension
and reasoning capabilities, are being applied to a range of complex language
tasks, including the automatic generation of language data for various
purposes. However, research on applying LLMs for automatic data generation in
low-resource languages like Vietnamese is still underdeveloped and lacks
comprehensive evaluation. In this paper, we explore the use of LLMs for
automatic data generation for the Vietnamese fact-checking task, which faces
significant data limitations. Specifically, we focus on fact-checking data
where claims are synthesized from multiple evidence sentences to assess the
information synthesis capabilities of LLMs. We develop an automatic data
construction process using simple prompt techniques on LLMs and explore several
methods to improve the quality of the generated data. To evaluate the quality
of the data generated by LLMs, we conduct both manual quality assessments and
performance evaluations using language models. Experimental results and manual
evaluations illustrate that while the quality of the generated data has
significantly improved through fine-tuning techniques, LLMs still cannot match
the data quality produced by humans.


---

**[94. [2502.17947] DeepSeek-R1 Outperforms Gemini 2.0 Pro, OpenAI o1, and o3-mini in
  Bilingual Complex Ophthalmology Reasoning](https://arxiv.org/pdf/2502.17947.pdf)** (2025-02-26)

*Pusheng Xu, Yue Wu, Kai Jin, Xiaolan Chen, Mingguang He, Danli Shi*

  Purpose: To evaluate the accuracy and reasoning ability of DeepSeek-R1 and
three other recently released large language models (LLMs) in bilingual complex
ophthalmology cases. Methods: A total of 130 multiple-choice questions (MCQs)
related to diagnosis (n = 39) and management (n = 91) were collected from the
Chinese ophthalmology senior professional title examination and categorized
into six topics. These MCQs were translated into English using DeepSeek-R1. The
responses of DeepSeek-R1, Gemini 2.0 Pro, OpenAI o1 and o3-mini were generated
under default configurations between February 15 and February 20, 2025.
Accuracy was calculated as the proportion of correctly answered questions, with
omissions and extra answers considered incorrect. Reasoning ability was
evaluated through analyzing reasoning logic and the causes of reasoning error.
Results: DeepSeek-R1 demonstrated the highest overall accuracy, achieving 0.862
in Chinese MCQs and 0.808 in English MCQs. Gemini 2.0 Pro, OpenAI o1, and
OpenAI o3-mini attained accuracies of 0.715, 0.685, and 0.692 in Chinese MCQs
(all P<0.001 compared with DeepSeek-R1), and 0.746 (P=0.115), 0.723 (P=0.027),
and 0.577 (P<0.001) in English MCQs, respectively. DeepSeek-R1 achieved the
highest accuracy across five topics in both Chinese and English MCQs. It also
excelled in management questions conducted in Chinese (all P<0.05). Reasoning
ability analysis showed that the four LLMs shared similar reasoning logic.
Ignoring key positive history, ignoring key positive signs, misinterpretation
medical data, and too aggressive were the most common causes of reasoning
errors. Conclusion: DeepSeek-R1 demonstrated superior performance in bilingual
complex ophthalmology reasoning tasks than three other state-of-the-art LLMs.
While its clinical applicability remains challenging, it shows promise for
supporting diagnosis and clinical decision-making.


---

**[95. [2403.03883] SaulLM-7B: A pioneering Large Language Model for Law](https://arxiv.org/pdf/2403.03883.pdf)** (2024-03-08)

*Pierre Colombo, Telmo Pessoa Pires, Malik Boudiaf, Dominic Culver, Rui Melo, Caio Corro, Andre F. T. Martins, Fabrizio Esposito, Vera Lúcia Raposo, Sofia Morgado, Michael Desa*

  In this paper, we introduce SaulLM-7B, a large language model (LLM) tailored
for the legal domain. With 7 billion parameters, SaulLM-7B is the first LLM
designed explicitly for legal text comprehension and generation. Leveraging the
Mistral 7B architecture as its foundation, SaulLM-7B is trained on an English
legal corpus of over 30 billion tokens. SaulLM-7B exhibits state-of-the-art
proficiency in understanding and processing legal documents. Additionally, we
present a novel instructional fine-tuning method that leverages legal datasets
to further enhance SaulLM-7B's performance in legal tasks. SaulLM-7B is
released under the MIT License.


---

**[96. [2411.05980] FactLens: Benchmarking Fine-Grained Fact Verification](https://arxiv.org/pdf/2411.05980.pdf)** (2024-11-12)

*Kushan Mitra, Dan Zhang, Sajjadur Rahman, Estevam Hruschka*

  Large Language Models (LLMs) have shown impressive capability in language
generation and understanding, but their tendency to hallucinate and produce
factually incorrect information remains a key limitation. To verify
LLM-generated contents and claims from other sources, traditional verification
approaches often rely on holistic models that assign a single factuality label
to complex claims, potentially obscuring nuanced errors. In this paper, we
advocate for a shift toward fine-grained verification, where complex claims are
broken down into smaller sub-claims for individual verification, allowing for
more precise identification of inaccuracies, improved transparency, and reduced
ambiguity in evidence retrieval. However, generating sub-claims poses
challenges, such as maintaining context and ensuring semantic equivalence with
respect to the original claim. We introduce FactLens, a benchmark for
evaluating fine-grained fact verification, with metrics and automated
evaluators of sub-claim quality. The benchmark data is manually curated to
ensure high-quality ground truth. Our results show alignment between automated
FactLens evaluators and human judgments, and we discuss the impact of sub-claim
characteristics on the overall verification performance.


---

**[97. [2502.11959] STRIVE: Structured Reasoning for Self-Improvement in Claim Verification](https://arxiv.org/pdf/2502.11959.pdf)** (2025-02-18)

*Haisong Gong, Jing Li, Junfei Wu, Qiang Liu, Shu Wu, Liang Wang*

  Claim verification is the task of determining whether a claim is supported or
refuted by evidence. Self-improvement methods, where reasoning chains are
generated and those leading to correct results are selected for training, have
succeeded in tasks like mathematical problem solving. However, in claim
verification, this approach struggles. Low-quality reasoning chains may falsely
match binary truth labels, introducing faulty reasoning into the
self-improvement process and ultimately degrading performance. To address this,
we propose STRIVE: Structured Reasoning for Self-Improved Verification. Our
method introduces a structured reasoning design with Claim Decomposition,
Entity Analysis, and Evidence Grounding Verification. These components improve
reasoning quality, reduce errors, and provide additional supervision signals
for self-improvement. STRIVE begins with a warm-up phase, where the base model
is fine-tuned on a small number of annotated examples to learn the structured
reasoning design. It is then applied to generate reasoning chains for all
training examples, selecting only those that are correct and structurally sound
for subsequent self-improvement training. We demonstrate that STRIVE achieves
significant improvements over baseline models, with a 31.4% performance gain
over the base model and 20.7% over Chain of Thought on the HOVER datasets,
highlighting its effectiveness.


---

**[98. [2407.11833] LoFTI: Localization and Factuality Transfer to Indian Locales](https://arxiv.org/pdf/2407.11833.pdf)** (2024-07-17)

*Sona Elza Simon, Soumen Kumar Mondal, Abhishek Singhania, Sayambhu Sen, Preethi Jyothi*

  Large language models (LLMs) encode vast amounts of world knowledge acquired
via training on large web-scale datasets crawled from the internet. However,
these datasets typically exhibit a geographical bias towards English-speaking
Western countries. This results in LLMs producing biased or hallucinated
responses to queries that require answers localized to other geographical
regions. In this work, we introduce a new benchmark named LoFTI (Localization
and Factuality Transfer to Indian Locales) that can be used to evaluate an
LLM's localization and factual text transfer capabilities. LoFTI consists of
factual statements about entities in source and target locations; the source
locations are spread across the globe and the target locations are all within
India with varying degrees of hyperlocality (country, states, cities). The
entities span a wide variety of categories. We use LoFTI to evaluate Mixtral,
GPT-4 and two other Mixtral-based approaches well-suited to the task of
localized factual transfer. We demonstrate that LoFTI is a high-quality
evaluation benchmark and all the models, including GPT-4, produce skewed
results across varying levels of hyperlocality.


---

**[99. [2211.08412] Evaluating the Factual Consistency of Large Language Models Through News
  Summarization](https://arxiv.org/pdf/2211.08412.pdf)** (2023-12-05)

*Derek Tam, Anisha Mascarenhas, Shiyue Zhang, Sarah Kwan, Mohit Bansal, Colin Raffel*

  While large language models (LLMs) have proven to be effective on a large
variety of tasks, they are also known to hallucinate information. To measure
whether an LLM prefers factually consistent continuations of its input, we
propose a new benchmark called FIB(Factual Inconsistency Benchmark) that
focuses on the task of summarization. Specifically, our benchmark involves
comparing the scores an LLM assigns to a factually consistent versus a
factually inconsistent summary for an input news article. For factually
consistent summaries, we use human-written reference summaries that we manually
verify as factually consistent. To generate summaries that are factually
inconsistent, we generate summaries from a suite of summarization models that
we have manually annotated as factually inconsistent. A model's factual
consistency is then measured according to its accuracy, i.e.\ the proportion of
documents where it assigns a higher score to the factually consistent summary.
To validate the usefulness of FIB, we evaluate 23 large language models ranging
from 1B to 176B parameters from six different model families including BLOOM
and OPT. We find that existing LLMs generally assign a higher score to
factually consistent summaries than to factually inconsistent summaries.
However, if the factually inconsistent summaries occur verbatim in the
document, then LLMs assign a higher score to these factually inconsistent
summaries than factually consistent summaries. We validate design choices in
our benchmark including the scoring method and source of distractor summaries.
Our code and benchmark data can be found at https://github.com/r-three/fib.


---

**[100. [2407.17468] WildHallucinations: Evaluating Long-form Factuality in LLMs with
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

**[101. [2404.16032] Studying Large Language Model Behaviors Under Context-Memory Conflicts
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

**[102. [2502.14678] How to Get Your LLM to Generate Challenging Problems for Evaluation](https://arxiv.org/pdf/2502.14678.pdf)** (2025-02-21)

*Arkil Patel, Siva Reddy, Dzmitry Bahdanau*

  The pace of evolution of Large Language Models (LLMs) necessitates new
approaches for rigorous and comprehensive evaluation. Traditional human
annotation is increasingly impracticable due to the complexities and costs
involved in generating high-quality, challenging problems. In this work, we
introduce CHASE, a unified framework to synthetically generate challenging
problems using LLMs without human involvement. For a given task, our approach
builds a hard problem in a bottom-up manner from simpler components. Moreover,
our framework decomposes the generation process into independently verifiable
sub-tasks, thereby ensuring a high level of quality and correctness. We
implement CHASE to create evaluation benchmarks across three diverse domains:
(1) document-based question answering, (2) repository-level code completion,
and (3) math reasoning. The performance of state-of-the-art LLMs on these
synthetic benchmarks lies in the range of 40-60% accuracy, thereby
demonstrating the effectiveness of our framework at generating challenging
problems. We publicly release our benchmarks and code.


---

**[103. [2409.14961] UELLM: A Unified and Efficient Approach for LLM Inference Serving](https://arxiv.org/pdf/2409.14961.pdf)** (2024-09-25)

*Yiyuan He, Minxian Xu, Jingfeng Wu, Wanyi Zheng, Kejiang Ye, Chengzhong Xu*

  In the context of Machine Learning as a Service (MLaaS) clouds, the extensive
use of Large Language Models (LLMs) often requires efficient management of
significant query loads. When providing real-time inference services, several
challenges arise. Firstly, increasing the number of GPUs may lead to a decrease
in inference speed due to heightened communication overhead, while an
inadequate number of GPUs can lead to out-of-memory errors. Secondly, different
deployment strategies need to be evaluated to guarantee optimal utilization and
minimal inference latency. Lastly, inefficient orchestration of inference
queries can easily lead to significant Service Level Objective (SLO)
violations. Lastly, inefficient orchestration of inference queries can easily
lead to significant Service Level Objective (SLO) violations. To address these
challenges, we propose a Unified and Efficient approach for Large Language
Model inference serving (UELLM), which consists of three main components: 1)
resource profiler, 2) batch scheduler, and 3) LLM deployer. UELLM minimizes
resource overhead, reduces inference latency, and lowers SLO violation rates.
Compared with state-of-the-art (SOTA) techniques, UELLM reduces the inference
latency by 72.3% to 90.3%, enhances GPU utilization by 1.2X to 4.1X, and
increases throughput by 1.92X to 4.98X, it can also serve without violating the
inference latency SLO.


---

**[104. [2311.06697] Trusted Source Alignment in Large Language Models](https://arxiv.org/pdf/2311.06697.pdf)** (2023-11-14)

*Vasilisa Bashlovkina, Zhaobin Kuang, Riley Matthews, Edward Clifford, Yennie Jun, William W. Cohen, Simon Baumgartner*

  Large language models (LLMs) are trained on web-scale corpora that inevitably
include contradictory factual information from sources of varying reliability.
In this paper, we propose measuring an LLM property called trusted source
alignment (TSA): the model's propensity to align with content produced by
trusted publishers in the face of uncertainty or controversy. We present
FactCheckQA, a TSA evaluation dataset based on a corpus of fact checking
articles. We describe a simple protocol for evaluating TSA and offer a detailed
analysis of design considerations including response extraction, claim
contextualization, and bias in prompt formulation. Applying the protocol to
PaLM-2, we find that as we scale up the model size, the model performance on
FactCheckQA improves from near-random to up to 80% balanced accuracy in
aligning with trusted sources.


---

**[105. [2310.11511] Self-RAG: Learning to Retrieve, Generate, and Critique through
  Self-Reflection](https://arxiv.org/pdf/2310.11511.pdf)** (2023-10-19)

*Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi*

  Despite their remarkable capabilities, large language models (LLMs) often
produce responses containing factual inaccuracies due to their sole reliance on
the parametric knowledge they encapsulate. Retrieval-Augmented Generation
(RAG), an ad hoc approach that augments LMs with retrieval of relevant
knowledge, decreases such issues. However, indiscriminately retrieving and
incorporating a fixed number of retrieved passages, regardless of whether
retrieval is necessary, or passages are relevant, diminishes LM versatility or
can lead to unhelpful response generation. We introduce a new framework called
Self-Reflective Retrieval-Augmented Generation (Self-RAG) that enhances an LM's
quality and factuality through retrieval and self-reflection. Our framework
trains a single arbitrary LM that adaptively retrieves passages on-demand, and
generates and reflects on retrieved passages and its own generations using
special tokens, called reflection tokens. Generating reflection tokens makes
the LM controllable during the inference phase, enabling it to tailor its
behavior to diverse task requirements. Experiments show that Self-RAG (7B and
13B parameters) significantly outperforms state-of-the-art LLMs and
retrieval-augmented models on a diverse set of tasks. Specifically, Self-RAG
outperforms ChatGPT and retrieval-augmented Llama2-chat on Open-domain QA,
reasoning and fact verification tasks, and it shows significant gains in
improving factuality and citation accuracy for long-form generations relative
to these models.


---

**[106. [2504.00180] Contradiction Detection in RAG Systems: Evaluating LLMs as Context
  Validators for Improved Information Consistency](https://arxiv.org/pdf/2504.00180.pdf)** (2025-04-02)

*Vignesh Gokul, Srikanth Tenneti, Alwarappan Nakkiran*

  Retrieval Augmented Generation (RAG) systems have emerged as a powerful
method for enhancing large language models (LLMs) with up-to-date information.
However, the retrieval step in RAG can sometimes surface documents containing
contradictory information, particularly in rapidly evolving domains such as
news. These contradictions can significantly impact the performance of LLMs,
leading to inconsistent or erroneous outputs. This study addresses this
critical challenge in two ways. First, we present a novel data generation
framework to simulate different types of contradictions that may occur in the
retrieval stage of a RAG system. Second, we evaluate the robustness of
different LLMs in performing as context validators, assessing their ability to
detect contradictory information within retrieved document sets. Our
experimental results reveal that context validation remains a challenging task
even for state-of-the-art LLMs, with performance varying significantly across
different types of contradictions. While larger models generally perform better
at contradiction detection, the effectiveness of different prompting strategies
varies across tasks and model architectures. We find that chain-of-thought
prompting shows notable improvements for some models but may hinder performance
in others, highlighting the complexity of the task and the need for more robust
approaches to context validation in RAG systems.


---

**[107. [2408.02964] Accuracy and Consistency of LLMs in the Registered Dietitian Exam: The
  Impact of Prompt Engineering and Knowledge Retrieval](https://arxiv.org/pdf/2408.02964.pdf)** (2024-08-09)

*Iman Azimi, Mohan Qi, Li Wang, Amir M. Rahmani, Youlin Li*

  Large language models (LLMs) are fundamentally transforming human-facing
applications in the health and well-being domains: boosting patient engagement,
accelerating clinical decision-making, and facilitating medical education.
Although state-of-the-art LLMs have shown superior performance in several
conversational applications, evaluations within nutrition and diet applications
are still insufficient. In this paper, we propose to employ the Registered
Dietitian (RD) exam to conduct a standard and comprehensive evaluation of
state-of-the-art LLMs, GPT-4o, Claude 3.5 Sonnet, and Gemini 1.5 Pro, assessing
both accuracy and consistency in nutrition queries. Our evaluation includes
1050 RD exam questions encompassing several nutrition topics and proficiency
levels. In addition, for the first time, we examine the impact of Zero-Shot
(ZS), Chain of Thought (CoT), Chain of Thought with Self Consistency (CoT-SC),
and Retrieval Augmented Prompting (RAP) on both accuracy and consistency of the
responses. Our findings revealed that while these LLMs obtained acceptable
overall performance, their results varied considerably with different prompts
and question domains. GPT-4o with CoT-SC prompting outperformed the other
approaches, whereas Gemini 1.5 Pro with ZS recorded the highest consistency.
For GPT-4o and Claude 3.5, CoT improved the accuracy, and CoT-SC improved both
accuracy and consistency. RAP was particularly effective for GPT-4o to answer
Expert level questions. Consequently, choosing the appropriate LLM and
prompting technique, tailored to the proficiency level and specific domain, can
mitigate errors and potential risks in diet and nutrition chatbots.


---

**[108. [2305.13252] "According to ...": Prompting Language Models Improves Quoting from
  Pre-Training Data](https://arxiv.org/pdf/2305.13252.pdf)** (2024-02-28)

*Orion Weller, Marc Marone, Nathaniel Weir, Dawn Lawrie, Daniel Khashabi, Benjamin Van Durme*

  Large Language Models (LLMs) may hallucinate and generate fake information,
despite pre-training on factual data. Inspired by the journalistic device of
"according to sources", we propose according-to prompting: directing LLMs to
ground responses against previously observed text. To quantify this grounding,
we propose a novel evaluation metric (QUIP-Score) that measures the extent to
which model-produced answers are directly found in underlying text corpora. We
illustrate with experiments on three corpora (Wikipedia, PubMed, and the U.S.
legal tax code) that these prompts improve grounding under our metrics, with
the additional benefit of often improving end-task performance. Furthermore,
prompts that ask the model to decrease grounding (or to ground to other
corpora) indeed decrease QUIP-Score, indicating the ability of LLMs to increase
or decrease grounded generations on request.


---

**[109. [2305.12295] Logic-LM: Empowering Large Language Models with Symbolic Solvers for
  Faithful Logical Reasoning](https://arxiv.org/pdf/2305.12295.pdf)** (2023-10-20)

*Liangming Pan, Alon Albalak, Xinyi Wang, William Yang Wang*

  Large Language Models (LLMs) have shown human-like reasoning abilities but
still struggle with complex logical problems. This paper introduces a novel
framework, Logic-LM, which integrates LLMs with symbolic solvers to improve
logical problem-solving. Our method first utilizes LLMs to translate a natural
language problem into a symbolic formulation. Afterward, a deterministic
symbolic solver performs inference on the formulated problem. We also introduce
a self-refinement module, which utilizes the symbolic solver's error messages
to revise symbolic formalizations. We demonstrate Logic-LM's effectiveness on
five logical reasoning datasets: ProofWriter, PrOntoQA, FOLIO,
LogicalDeduction, and AR-LSAT. On average, Logic-LM achieves a significant
performance boost of 39.2% over using LLM alone with standard prompting and
18.4% over LLM with chain-of-thought prompting. Our findings suggest that
Logic-LM, by combining LLMs with symbolic logic, offers a promising avenue for
faithful logical reasoning. Code and data are publicly available at
https://github.com/teacherpeterpan/Logic-LLM.


---

**[110. [2408.12188] Reasoning Factual Knowledge in Structured Data with Large Language
  Models](https://arxiv.org/pdf/2408.12188.pdf)** (2024-08-23)

*Sirui Huang, Yanggan Gu, Xuming Hu, Zhonghao Li, Qing Li, Guandong Xu*

  Large language models (LLMs) have made remarkable progress in various natural
language processing tasks as a benefit of their capability to comprehend and
reason with factual knowledge. However, a significant amount of factual
knowledge is stored in structured data, which possesses unique characteristics
that differ from the unstructured texts used for pretraining. This difference
can introduce imperceptible inference parameter deviations, posing challenges
for LLMs in effectively utilizing and reasoning with structured data to
accurately infer factual knowledge. To this end, we propose a benchmark named
StructFact, to evaluate the structural reasoning capabilities of LLMs in
inferring factual knowledge. StructFact comprises 8,340 factual questions
encompassing various tasks, domains, timelines, and regions. This benchmark
allows us to investigate the capability of LLMs across five factual tasks
derived from the unique characteristics of structural facts. Extensive
experiments on a set of LLMs with different training strategies reveal the
limitations of current LLMs in inferring factual knowledge from structured
data. We present this benchmark as a compass to navigate the strengths and
weaknesses of LLMs in reasoning with structured data for knowledge-sensitive
tasks, and to encourage advancements in related real-world applications. Please
find our code at https://github.com/EganGu/StructFact.


---

**[111. [2407.20999] MoFO: Momentum-Filtered Optimizer for Mitigating Forgetting in LLM
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

**[112. [2504.02881] Better Bill GPT: Comparing Large Language Models against Legal Invoice
  Reviewers](https://arxiv.org/pdf/2504.02881.pdf)** (2025-04-07)

*Nick Whitehouse, Nicole Lincoln, Stephanie Yiu, Lizzie Catterson, Rivindu Perera*

  Legal invoice review is a costly, inconsistent, and time-consuming process,
traditionally performed by Legal Operations, Lawyers or Billing Specialists who
scrutinise billing compliance line by line. This study presents the first
empirical comparison of Large Language Models (LLMs) against human invoice
reviewers - Early-Career Lawyers, Experienced Lawyers, and Legal Operations
Professionals-assessing their accuracy, speed, and cost-effectiveness.
Benchmarking state-of-the-art LLMs against a ground truth set by expert legal
professionals, our empirically substantiated findings reveal that LLMs
decisively outperform humans across every metric. In invoice approval
decisions, LLMs achieve up to 92% accuracy, surpassing the 72% ceiling set by
experienced lawyers. On a granular level, LLMs dominate line-item
classification, with top models reaching F-scores of 81%, compared to just 43%
for the best-performing human group. Speed comparisons are even more striking -
while lawyers take 194 to 316 seconds per invoice, LLMs are capable of
completing reviews in as fast as 3.6 seconds. And cost? AI slashes review
expenses by 99.97%, reducing invoice processing costs from an average of $4.27
per invoice for human invoice reviewers to mere cents. These results highlight
the evolving role of AI in legal spend management. As law firms and corporate
legal departments struggle with inefficiencies, this study signals a seismic
shift: The era of LLM-powered legal spend management is not on the horizon, it
has arrived. The challenge ahead is not whether AI can perform as well as human
reviewers, but how legal teams will strategically incorporate it, balancing
automation with human discretion.


---

**[113. [2410.13352] LAR-ECHR: A New Legal Argument Reasoning Task and Dataset for Cases of
  the European Court of Human Rights](https://arxiv.org/pdf/2410.13352.pdf)** (2024-10-18)

*Odysseas S. Chlapanis, Dimitrios Galanis, Ion Androutsopoulos*

  We present Legal Argument Reasoning (LAR), a novel task designed to evaluate
the legal reasoning capabilities of Large Language Models (LLMs). The task
requires selecting the correct next statement (from multiple choice options) in
a chain of legal arguments from court proceedings, given the facts of the case.
We constructed a dataset (LAR-ECHR) for this task using cases from the European
Court of Human Rights (ECHR). We evaluated seven general-purpose LLMs on
LAR-ECHR and found that (a) the ranking of the models is aligned with that of
LegalBench, an established US-based legal reasoning benchmark, even though
LAR-ECHR is based on EU law, (b) LAR-ECHR distinguishes top models more
clearly, compared to LegalBench, (c) even the best model (GPT-4o) obtains 75.8%
accuracy on LAR-ECHR, indicating significant potential for further model
improvement. The process followed to construct LAR-ECHR can be replicated with
cases from other legal systems.


---

**[114. [2407.05015] How do you know that? Teaching Generative Language Models to Reference
  Answers to Biomedical Questions](https://arxiv.org/pdf/2407.05015.pdf)** (2024-07-09)

*Bojana Bašaragin, Adela Ljajić, Darija Medvecki, Lorenzo Cassano, Miloš Košprdić, Nikola Milošević*

  Large language models (LLMs) have recently become the leading source of
answers for users' questions online. Despite their ability to offer eloquent
answers, their accuracy and reliability can pose a significant challenge. This
is especially true for sensitive domains such as biomedicine, where there is a
higher need for factually correct answers. This paper introduces a biomedical
retrieval-augmented generation (RAG) system designed to enhance the reliability
of generated responses. The system is based on a fine-tuned LLM for the
referenced question-answering, where retrieved relevant abstracts from PubMed
are passed to LLM's context as input through a prompt. Its output is an answer
based on PubMed abstracts, where each statement is referenced accordingly,
allowing the users to verify the answer. Our retrieval system achieves an
absolute improvement of 23% compared to the PubMed search engine. Based on the
manual evaluation on a small sample, our fine-tuned LLM component achieves
comparable results to GPT-4 Turbo in referencing relevant abstracts. We make
the dataset used to fine-tune the models and the fine-tuned models based on
Mistral-7B-instruct-v0.1 and v0.2 publicly available.


---

**[115. [2411.19655] Truth or Mirage? Towards End-to-End Factuality Evaluation with LLM-Oasis](https://arxiv.org/pdf/2411.19655.pdf)** (2025-04-01)

*Alessandro Scirè, Andrei Stefan Bejgu, Simone Tedeschi, Karim Ghonim, Federico Martelli, Roberto Navigli*

  After the introduction of Large Language Models (LLMs), there have been
substantial improvements in the performance of Natural Language Generation
(NLG) tasks, including Text Summarization and Machine Translation. However,
LLMs still produce outputs containing hallucinations, that is, content not
grounded in factual information. Therefore, developing methods to assess the
factuality of LLMs has become urgent.
  Indeed, resources for factuality evaluation have recently emerged. Although
challenging, these resources face one or more of the following limitations: (i)
they are tailored to a specific task or domain; (ii) they are limited in size,
thereby preventing the training of new factuality evaluators; (iii) they are
designed for simpler verification tasks, such as claim verification.
  To address these issues, we introduce LLM-Oasis, to the best of our knowledge
the largest resource for training end-to-end factuality evaluators. LLM-Oasis
is constructed by extracting claims from Wikipedia, falsifying a subset of
these claims, and generating pairs of factual and unfactual texts. We then rely
on human annotators to both validate the quality of our dataset and to create a
gold standard test set for benchmarking factuality evaluation systems.
  Our experiments demonstrate that LLM-Oasis presents a significant challenge
for state-of-the-art LLMs, with GPT-4o achieving up to 60% accuracy in our
proposed end-to-end factuality evaluation task, highlighting its potential to
drive future research in the field.


---

**[116. [2312.04916] EE-LLM: Large-Scale Training and Inference of Early-Exit Large Language
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

**[117. [2310.00305] Towards LLM-based Fact Verification on News Claims with a Hierarchical
  Step-by-Step Prompting Method](https://arxiv.org/pdf/2310.00305.pdf)** (2023-10-03)

*Xuan Zhang, Wei Gao*

  While large pre-trained language models (LLMs) have shown their impressive
capabilities in various NLP tasks, they are still under-explored in the
misinformation domain. In this paper, we examine LLMs with in-context learning
(ICL) for news claim verification, and find that only with 4-shot demonstration
examples, the performance of several prompting methods can be comparable with
previous supervised models. To further boost performance, we introduce a
Hierarchical Step-by-Step (HiSS) prompting method which directs LLMs to
separate a claim into several subclaims and then verify each of them via
multiple questions-answering steps progressively. Experiment results on two
public misinformation datasets show that HiSS prompting outperforms
state-of-the-art fully-supervised approach and strong few-shot ICL-enabled
baselines.


---

**[118. [2411.18019] A Real-World Benchmark for Evaluating Fine-Grained Issue Solving
  Capabilities of Large Language Models](https://arxiv.org/pdf/2411.18019.pdf)** (2024-11-28)

*Ruida Hu, Chao Peng, Jingyi Ren, Bo Jiang, Xiangxin Meng, Qinyun Wu, Pengfei Gao, Xinchen Wang, Cuiyun Gao*

  Automatically resolving software issues is crucial for software development
in practice, impacting the software quality and user experience. The process of
resolving real-world issues encompasses tasks such as question-answering (QA),
fault localization, and code editing. Existing benchmarks such as HumanEval
fall short in their ability to assess LLMs' proficiency in solving issues
within a codebase. Although benchmarks like SWE-Bench are designed to evaluate
the LLMs' capability to handle real-world GitHub issues, the end-to-end
evaluation method cannot provide granular insights on the performance of
subtasks involved in issue solving. To address existing deficiencies in
benchmarking LLMs for practical software engineering tasks, we introduce
FAUN-Eval, a benchmark specifically designed to evaluate the Fine-grAined issUe
solviNg capabilities of LLMs. FAUN-Eval systematically assesses LLMs across
three distinct tasks: QA, fault localization, and code editing. This benchmark
is constructed using a dataset curated from 30 well-known GitHub repositories.
For each entry, issue and pull request (PR) pairs are meticulously compiled and
validated using cross-referencing and keyword verification methods. FAUN-Eval
includes 300 entries and employs both LLM and manual checks to ensure data
quality. We evaluate ten LLMs with FAUN-Eval, including four closed-source and
six open-source models. Our experimental results reveal several key findings.
We find that the top-performing LLMs differ across the different tasks.
Additionally, features in issues may lead LLMs to generate incorrect
information. Moreover, models may vary in their proficiency with texts of
different lengths.


---

**[119. [2406.11514] Counterfactual Debating with Preset Stances for Hallucination
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

**[120. [2412.11142] AD-LLM: Benchmarking Large Language Models for Anomaly Detection](https://arxiv.org/pdf/2412.11142.pdf)** (2024-12-17)

*Tiankai Yang, Yi Nian, Shawn Li, Ruiyao Xu, Yuangang Li, Jiaqi Li, Zhuo Xiao, Xiyang Hu, Ryan Rossi, Kaize Ding, Xia Hu, Yue Zhao*

  Anomaly detection (AD) is an important machine learning task with many
real-world uses, including fraud detection, medical diagnosis, and industrial
monitoring. Within natural language processing (NLP), AD helps detect issues
like spam, misinformation, and unusual user activity. Although large language
models (LLMs) have had a strong impact on tasks such as text generation and
summarization, their potential in AD has not been studied enough. This paper
introduces AD-LLM, the first benchmark that evaluates how LLMs can help with
NLP anomaly detection. We examine three key tasks: (i) zero-shot detection,
using LLMs' pre-trained knowledge to perform AD without tasks-specific
training; (ii) data augmentation, generating synthetic data and category
descriptions to improve AD models; and (iii) model selection, using LLMs to
suggest unsupervised AD models. Through experiments with different datasets, we
find that LLMs can work well in zero-shot AD, that carefully designed
augmentation methods are useful, and that explaining model selection for
specific datasets remains challenging. Based on these results, we outline six
future research directions on LLMs for AD.


---

**[121. [2411.16238] UVLLM: An Automated Universal RTL Verification Framework using LLMs](https://arxiv.org/pdf/2411.16238.pdf)** (2024-11-26)

*Yuchen Hu, Junhao Ye, Ke Xu, Jialin Sun, Shiyue Zhang, Xinyao Jiao, Dingrong Pan, Jie Zhou, Ning Wang, Weiwei Shan, Xinwei Fang, Xi Wang, Nan Guan, Zhe Jiang*

  Verifying hardware designs in embedded systems is crucial but often
labor-intensive and time-consuming. While existing solutions have improved
automation, they frequently rely on unrealistic assumptions. To address these
challenges, we introduce a novel framework, UVLLM, which combines Large
Language Models (LLMs) with the Universal Verification Methodology (UVM) to
relax these assumptions. UVLLM significantly enhances the automation of testing
and repairing error-prone Register Transfer Level (RTL) codes, a critical
aspect of verification development. Unlike existing methods, UVLLM ensures that
all errors are triggered during verification, achieving a syntax error fix rate
of 86.99% and a functional error fix rate of 71.92% on our proposed benchmark.
These results demonstrate a substantial improvement in verification efficiency.
Additionally, our study highlights the current limitations of LLM applications,
particularly their reliance on extensive training data. We emphasize the
transformative potential of LLMs in hardware design verification and suggest
promising directions for future research in AI-driven hardware design
methodologies. The Repo. of dataset and code:
https://anonymous.4open.science/r/UVLLM/.


---

**[122. [2407.07666] A Proposed S.C.O.R.E. Evaluation Framework for Large Language Models :
  Safety, Consensus, Objectivity, Reproducibility and Explainability](https://arxiv.org/pdf/2407.07666.pdf)** (2024-07-11)

*Ting Fang Tan, Kabilan Elangovan, Jasmine Ong, Nigam Shah, Joseph Sung, Tien Yin Wong, Lan Xue, Nan Liu, Haibo Wang, Chang Fu Kuo, Simon Chesterman, Zee Kin Yeong, Daniel SW Ting*

  A comprehensive qualitative evaluation framework for large language models
(LLM) in healthcare that expands beyond traditional accuracy and quantitative
metrics needed. We propose 5 key aspects for evaluation of LLMs: Safety,
Consensus, Objectivity, Reproducibility and Explainability (S.C.O.R.E.). We
suggest that S.C.O.R.E. may form the basis for an evaluation framework for
future LLM-based models that are safe, reliable, trustworthy, and ethical for
healthcare and clinical applications.


---

**[123. [2503.21717] CLAIMCHECK: How Grounded are LLM Critiques of Scientific Papers?](https://arxiv.org/pdf/2503.21717.pdf)** (2025-03-28)

*Jiefu Ou, William Gantt Walden, Kate Sanders, Zhengping Jiang, Kaiser Sun, Jeffrey Cheng, William Jurayj, Miriam Wanner, Shaobo Liang, Candice Morgan, Seunghoon Han, Weiqi Wang, Chandler May, Hannah Recknor, Daniel Khashabi, Benjamin Van Durme*

  A core part of scientific peer review involves providing expert critiques
that directly assess the scientific claims a paper makes. While it is now
possible to automatically generate plausible (if generic) reviews, ensuring
that these reviews are sound and grounded in the papers' claims remains
challenging. To facilitate LLM benchmarking on these challenges, we introduce
CLAIMCHECK, an annotated dataset of NeurIPS 2023 and 2024 submissions and
reviews mined from OpenReview. CLAIMCHECK is richly annotated by ML experts for
weakness statements in the reviews and the paper claims that they dispute, as
well as fine-grained labels of the validity, objectivity, and type of the
identified weaknesses. We benchmark several LLMs on three claim-centric tasks
supported by CLAIMCHECK, requiring models to (1) associate weaknesses with the
claims they dispute, (2) predict fine-grained labels for weaknesses and rewrite
the weaknesses to enhance their specificity, and (3) verify a paper's claims
with grounded reasoning. Our experiments reveal that cutting-edge LLMs, while
capable of predicting weakness labels in (2), continue to underperform relative
to human experts on all other tasks.


---

**[124. [2501.11929] ALoFTRAG: Automatic Local Fine Tuning for Retrieval Augmented Generation](https://arxiv.org/pdf/2501.11929.pdf)** (2025-01-22)

*Peter Devine*

  Retrieval Augmented Generation (RAG) systems have been shown to improve the
accuracy of Large Language Model (LLM) outputs. However, these models can often
achieve low accuracy when applied to new data domains.
  We introduce the Automatic Local Fine Tuning of Retrieval Augmented
Generation models (ALoFTRAG) framework, designed to improve the accuracy of RAG
systems on a given domain by training LLMs without manually labeled data or
using larger teacher models.
  By generating and filtering synthetic training data and performing LoRA
fine-tuning, ALoFTRAG improves citation and answer accuracy across 20 datasets
in 26 languages by, on average, 8.3% and 3.0% respectively.
  Our results demonstrate that ALoFTRAG offers a practical, cost-effective, and
data-secure solution for improving RAG accuracy, making it particularly
applicable to sensitive domains such as healthcare and finance.


---

**[125. [2411.08254] VALTEST: Automated Validation of Language Model Generated Test Cases](https://arxiv.org/pdf/2411.08254.pdf)** (2024-11-14)

*Hamed Taherkhani, Hadi Hemmati*

  Large Language Models (LLMs) have demonstrated significant potential in
automating software testing, specifically in generating unit test cases.
However, the validation of LLM-generated test cases remains a challenge,
particularly when the ground truth is unavailable. This paper introduces
VALTEST, a novel framework designed to automatically validate test cases
generated by LLMs by leveraging token probabilities. We evaluate VALTEST using
nine test suites generated from three datasets (HumanEval, MBPP, and LeetCode)
across three LLMs (GPT-4o, GPT-3.5-turbo, and LLama3.1 8b). By extracting
statistical features from token probabilities, we train a machine learning
model to predict test case validity. VALTEST increases the validity rate of
test cases by 6.2% to 24%, depending on the dataset and LLM. Our results
suggest that token probabilities are reliable indicators for distinguishing
between valid and invalid test cases, which provides a robust solution for
improving the correctness of LLM-generated test cases in software testing. In
addition, we found that replacing the identified invalid test cases by VALTEST,
using a Chain-of-Thought prompting results in a more effective test suite while
keeping the high validity rates.


---

**[126. [2401.17703] WSC+: Enhancing The Winograd Schema Challenge Using Tree-of-Experts](https://arxiv.org/pdf/2401.17703.pdf)** (2024-02-01)

*Pardis Sadat Zahraei, Ali Emami*

  The Winograd Schema Challenge (WSC) serves as a prominent benchmark for
evaluating machine understanding. While Large Language Models (LLMs) excel at
answering WSC questions, their ability to generate such questions remains less
explored. In this work, we propose Tree-of-Experts (ToE), a novel prompting
method which enhances the generation of WSC instances (50% valid cases vs. 10%
in recent methods). Using this approach, we introduce WSC+, a novel dataset
comprising 3,026 LLM-generated sentences. Notably, we extend the WSC framework
by incorporating new 'ambiguous' and 'offensive' categories, providing a deeper
insight into model overconfidence and bias. Our analysis reveals nuances in
generation-evaluation consistency, suggesting that LLMs may not always
outperform in evaluating their own generated questions when compared to those
crafted by other models. On WSC+, GPT-4, the top-performing LLM, achieves an
accuracy of 68.7%, significantly below the human benchmark of 95.1%.


---

**[127. [2503.16515] Highlighting Case Studies in LLM Literature Review of Interdisciplinary
  System Science](https://arxiv.org/pdf/2503.16515.pdf)** (2025-03-24)

*Lachlan McGinness, Peter Baumgartner*

  Large Language Models (LLMs) were used to assist four Commonwealth Scientific
and Industrial Research Organisation (CSIRO) researchers to perform systematic
literature reviews (SLR). We evaluate the performance of LLMs for SLR tasks in
these case studies. In each, we explore the impact of changing parameters on
the accuracy of LLM responses. The LLM was tasked with extracting evidence from
chosen academic papers to answer specific research questions. We evaluate the
models' performance in faithfully reproducing quotes from the literature and
subject experts were asked to assess the model performance in answering the
research questions. We developed a semantic text highlighting tool to
facilitate expert review of LLM responses.
  We found that state of the art LLMs were able to reproduce quotes from texts
with greater than 95% accuracy and answer research questions with an accuracy
of approximately 83%. We use two methods to determine the correctness of LLM
responses; expert review and the cosine similarity of transformer embeddings of
LLM and expert answers. The correlation between these methods ranged from 0.48
to 0.77, providing evidence that the latter is a valid metric for measuring
semantic similarity.


---

**[128. [2402.13758] Factual consistency evaluation of summarization in the Era of large
  language models](https://arxiv.org/pdf/2402.13758.pdf)** (2025-02-28)

*Zheheng Luo, Qianqian Xie, Sophia Ananiadou*

  Factual inconsistency with source documents in automatically generated
summaries can lead to misinformation or pose risks. Existing factual
consistency (FC) metrics are constrained by their performance, efficiency, and
explainability. Recent advances in Large language models (LLMs) have
demonstrated remarkable potential in text evaluation but their effectiveness in
assessing FC in summarization remains underexplored. Prior research has mostly
focused on proprietary LLMs, leaving essential factors that affect their
assessment capabilities unexplored. Additionally, current FC evaluation
benchmarks are restricted to news articles, casting doubt on the generality of
the FC methods tested on them. In this paper, we first address the gap by
introducing TreatFact a dataset of LLM-generated summaries of clinical texts,
annotated for FC by domain experts. Moreover, we benchmark 11 LLMs for FC
evaluation across news and clinical domains and analyse the impact of model
size, prompts, pre-training and fine-tuning data. Our findings reveal that
despite proprietary models prevailing on the task, open-source LLMs lag behind.
Nevertheless, there is potential for enhancing the performance of open-source
LLMs through increasing model size, expanding pre-training data, and developing
well-curated fine-tuning data. Experiments on TreatFact suggest that both
previous methods and LLM-based evaluators are unable to capture factual
inconsistencies in clinical summaries, posing a new challenge for FC
evaluation.


---

**[129. [2401.11467] Over-Reasoning and Redundant Calculation of Large Language Models](https://arxiv.org/pdf/2401.11467.pdf)** (2024-03-21)

*Cheng-Han Chiang, Hung-yi Lee*

  Large language models (LLMs) can solve problems step-by-step. While this
chain-of-thought (CoT) reasoning boosts LLMs' performance, it is unclear if
LLMs \textit{know} when to use CoT and whether those CoT are always necessary
to answer the question. This paper shows that LLMs tend to generate redundant
calculations and reasoning on a manually constructed math QA dataset,
GSM8K-Zero. GSM8K-Zero is constructed such that the questions can be answered
without any calculations, but LLMs, including Llama-2 models and Claude-2, tend
to generate lengthy and unnecessary calculations to answer the questions. We
also conduct experiments to explain why LLMs generate redundant calculations
and reasonings. GSM8K-Zero is publicly available at
https://github.com/d223302/Over-Reasoning-of-LLMs and
https://huggingface.co/datasets/dcml0714/GSM8K-Zero.


---

**[130. [2406.03075] Towards Detecting LLMs Hallucination via Markov Chain-based Multi-agent
  Debate Framework](https://arxiv.org/pdf/2406.03075.pdf)** (2024-06-06)

*Xiaoxi Sun, Jinpeng Li, Yan Zhong, Dongyan Zhao, Rui Yan*

  The advent of large language models (LLMs) has facilitated the development of
natural language text generation. It also poses unprecedented challenges, with
content hallucination emerging as a significant concern. Existing solutions
often involve expensive and complex interventions during the training process.
Moreover, some approaches emphasize problem disassembly while neglecting the
crucial validation process, leading to performance degradation or limited
applications. To overcome these limitations, we propose a Markov Chain-based
multi-agent debate verification framework to enhance hallucination detection
accuracy in concise claims. Our method integrates the fact-checking process,
including claim detection, evidence retrieval, and multi-agent verification. In
the verification stage, we deploy multiple agents through flexible Markov
Chain-based debates to validate individual claims, ensuring meticulous
verification outcomes. Experimental results across three generative tasks
demonstrate that our approach achieves significant improvements over baselines.


---

**[131. [2503.24307] A Systematic Evaluation of LLM Strategies for Mental Health Text
  Analysis: Fine-tuning vs. Prompt Engineering vs. RAG](https://arxiv.org/pdf/2503.24307.pdf)** (2025-04-01)

*Arshia Kermani, Veronica Perez-Rosas, Vangelis Metsis*

  This study presents a systematic comparison of three approaches for the
analysis of mental health text using large language models (LLMs): prompt
engineering, retrieval augmented generation (RAG), and fine-tuning. Using LLaMA
3, we evaluate these approaches on emotion classification and mental health
condition detection tasks across two datasets. Fine-tuning achieves the highest
accuracy (91% for emotion classification, 80% for mental health conditions) but
requires substantial computational resources and large training sets, while
prompt engineering and RAG offer more flexible deployment with moderate
performance (40-68% accuracy). Our findings provide practical insights for
implementing LLM-based solutions in mental health applications, highlighting
the trade-offs between accuracy, computational requirements, and deployment
flexibility.


---

**[132. [2308.08090] Separate the Wheat from the Chaff: Model Deficiency Unlearning via
  Parameter-Efficient Module Operation](https://arxiv.org/pdf/2308.08090.pdf)** (2024-01-19)

*Xinshuo Hu, Dongfang Li, Baotian Hu, Zihao Zheng, Zhenyu Liu, Min Zhang*

  Large language models (LLMs) have been widely used in various applications
but are known to suffer from issues related to untruthfulness and toxicity.
While parameter-efficient modules (PEMs) have demonstrated their effectiveness
in equipping models with new skills, leveraging PEMs for deficiency unlearning
remains underexplored. In this work, we propose a PEMs operation approach,
namely Extraction-before-Subtraction (Ext-Sub), to enhance the truthfulness and
detoxification of LLMs through the integration of ``expert'' PEM and
``anti-expert'' PEM. Remarkably, even anti-expert PEM possess valuable
capabilities due to their proficiency in generating fabricated content, which
necessitates language modeling and logical narrative competence. Rather than
merely negating the parameters, our approach involves extracting and
eliminating solely the deficiency capability within anti-expert PEM while
preserving the general capabilities. To evaluate the effectiveness of our
approach in terms of truthfulness and detoxification, we conduct extensive
experiments on LLMs, encompassing additional abilities such as language
modeling and mathematical reasoning. Our empirical results demonstrate that our
approach effectively improves truthfulness and detoxification, while largely
preserving the fundamental abilities of LLMs.


---

**[133. [2402.00093] ChIRAAG: ChatGPT Informed Rapid and Automated Assertion Generation](https://arxiv.org/pdf/2402.00093.pdf)** (2024-07-01)

*Bhabesh Mali, Karthik Maddala, Vatsal Gupta, Sweeya Reddy, Chandan Karfa, Ramesh Karri*

  System Verilog Assertion (SVA) formulation -- a critical yet complex task is
a prerequisite in the Assertion Based Verification (ABV) process.
Traditionally, SVA formulation involves expert-driven interpretation of
specifications, which is time-consuming and prone to human error. Recently,
LLM-informed automatic assertion generation is gaining interest. We designed a
novel framework called ChIRAAG, based on OpenAI GPT4, to generate SVA from
natural language specifications of a design. ChIRAAG constitutes the systematic
breakdown of design specifications into a standardized format, further
generating assertions from formatted specifications using LLM. Furthermore, we
used few test cases to validate the LLM-generated assertions. Automatic
feedback of log messages from the simulation tool to the LLM ensures that the
framework can generate correct SVAs. In our experiments, only 27% of
LLM-generated raw assertions had errors, which was rectified in few iterations
based on the simulation log. Our results on OpenTitan designs show that LLMs
can streamline and assist engineers in the assertion generation process,
reshaping verification workflows.


---

**[134. [2404.15515] ToM-LM: Delegating Theory of Mind Reasoning to External Symbolic
  Executors in Large Language Models](https://arxiv.org/pdf/2404.15515.pdf)** (2024-06-27)

*Weizhi Tang, Vaishak Belle*

  Theory of Mind (ToM) refers to the ability of individuals to attribute mental
states to others. While Large Language Models (LLMs) have shown some promise
with ToM ability, they still struggle with complex ToM reasoning. Our approach
leverages an external symbolic executor, specifically the SMCDEL model checker,
and fine-tuning to improve the ToM reasoning ability of LLMs. In our approach,
an LLM is first fine-tuned through pairs of natural language and symbolic
formulation representation of ToM problems and is then instructed to generate
the symbolic formulation with a one-shot in-context example. The generated
symbolic formulation is then executed by the SMCDEL model checker to perform
transparent and verifiable ToM reasoning and give the final result. We
demonstrate that our approach, ToM-LM, shows a significant improvement over all
the constructed baselines. Our study proposes a novel view about externalizing
a particular component of ToM reasoning, mainly reasoning about beliefs, and
suggests generalizing it to other aspects of ToM reasoning.


---

**[135. [2412.04947] C$^2$LEVA: Toward Comprehensive and Contamination-Free Language Model
  Evaluation](https://arxiv.org/pdf/2412.04947.pdf)** (2024-12-17)

*Yanyang Li, Tin Long Wong, Cheung To Hung, Jianqiao Zhao, Duo Zheng, Ka Wai Liu, Michael R. Lyu, Liwei Wang*

  Recent advances in large language models (LLMs) have shown significant
promise, yet their evaluation raises concerns, particularly regarding data
contamination due to the lack of access to proprietary training data. To
address this issue, we present C$^2$LEVA, a comprehensive bilingual benchmark
featuring systematic contamination prevention. C$^2$LEVA firstly offers a
holistic evaluation encompassing 22 tasks, each targeting a specific
application or ability of LLMs, and secondly a trustworthy assessment due to
our contamination-free tasks, ensured by a systematic contamination prevention
strategy that fully automates test data renewal and enforces data protection
during benchmark data release. Our large-scale evaluation of 15 open-source and
proprietary models demonstrates the effectiveness of C$^2$LEVA.


---

**[136. [2408.10608] Promoting Equality in Large Language Models: Identifying and Mitigating
  the Implicit Bias based on Bayesian Theory](https://arxiv.org/pdf/2408.10608.pdf)** (2024-08-21)

*Yongxin Deng, Xihe Qiu, Xiaoyu Tan, Jing Pan, Chen Jue, Zhijun Fang, Yinghui Xu, Wei Chu, Yuan Qi*

  Large language models (LLMs) are trained on extensive text corpora, which
inevitably include biased information. Although techniques such as Affective
Alignment can mitigate some negative impacts of these biases, existing
prompt-based attack methods can still extract these biases from the model's
weights. Moreover, these biases frequently appear subtly when LLMs are prompted
to perform identical tasks across different demographic groups, thereby
camouflaging their presence. To address this issue, we have formally defined
the implicit bias problem and developed an innovative framework for bias
removal based on Bayesian theory, Bayesian-Theory based Bias Removal (BTBR).
BTBR employs likelihood ratio screening to pinpoint data entries within
publicly accessible biased datasets that represent biases inadvertently
incorporated during the LLM training phase. It then automatically constructs
relevant knowledge triples and expunges bias information from LLMs using model
editing techniques. Through extensive experimentation, we have confirmed the
presence of the implicit bias problem in LLMs and demonstrated the
effectiveness of our BTBR approach.


---

**[137. [2502.18532] CuDIP: Enhancing Theorem Proving in LLMs via Curriculum Learning-based
  Direct Preference Optimization](https://arxiv.org/pdf/2502.18532.pdf)** (2025-02-27)

*Shuming Shi, Ruobing Zuo, Gaolei He, Jianlin Wang, Chenyang Xu, Zhengfeng Yang*

  Automated theorem proving (ATP) is one of the most challenging mathematical
reasoning tasks for Large Language Models (LLMs). Most existing LLM-based ATP
methods rely on supervised fine-tuning, which results in a limited alignment
between the theorem proving process and human preferences. Direct Preference
Optimization (DPO), which aligns LLMs with human preferences, has shown
positive effects for certain tasks. However, the lack of high-quality
preference data for theorem proving presents a significant challenge. In this
paper, we innovatively apply DPO to formal automated theorem proving and
introduces a Curriculum Learning-based DPO Iterative Theorem Proving (CuDIP)
method. Specifically, we propose a method for constructing preference data
which utilizes LLMs and existing theorem proving data to enhance the diversity
of the preference data while reducing the reliance on human preference
annotations. We then integrate this preference data construction method with
curriculum learning to iteratively fine-tune the theorem proving model through
DPO. Experimental results on the MiniF2F and ProofNet datasets demonstrate the
effectiveness of the proposed method.


---

**[138. [2502.06737] VersaPRM: Multi-Domain Process Reward Model via Synthetic Reasoning Data](https://arxiv.org/pdf/2502.06737.pdf)** (2025-02-11)

*Thomas Zeng, Shuibai Zhang, Shutong Wu, Christian Classen, Daewon Chae, Ethan Ewer, Minjae Lee, Heeju Kim, Wonjun Kang, Jackson Kunde, Ying Fan, Jungtaek Kim, Hyung Il Koo, Kannan Ramchandran, Dimitris Papailiopoulos, Kangwook Lee*

  Process Reward Models (PRMs) have proven effective at enhancing mathematical
reasoning for Large Language Models (LLMs) by leveraging increased
inference-time computation. However, they are predominantly trained on
mathematical data and their generalizability to non-mathematical domains has
not been rigorously studied. In response, this work first shows that current
PRMs have poor performance in other domains. To address this limitation, we
introduce VersaPRM, a multi-domain PRM trained on synthetic reasoning data
generated using our novel data generation and annotation method. VersaPRM
achieves consistent performance gains across diverse domains. For instance, in
the MMLU-Pro category of Law, VersaPRM via weighted majority voting, achieves a
7.9% performance gain over the majority voting baseline -- surpassing
Qwen2.5-Math-PRM's gain of 1.3%. We further contribute to the community by
open-sourcing all data, code and models for VersaPRM.


---

**[139. [2408.03010] Fact Finder -- Enhancing Domain Expertise of Large Language Models by
  Incorporating Knowledge Graphs](https://arxiv.org/pdf/2408.03010.pdf)** (2024-08-07)

*Daniel Steinigen, Roman Teucher, Timm Heine Ruland, Max Rudat, Nicolas Flores-Herr, Peter Fischer, Nikola Milosevic, Christopher Schymura, Angelo Ziletti*

  Recent advancements in Large Language Models (LLMs) have showcased their
proficiency in answering natural language queries. However, their effectiveness
is hindered by limited domain-specific knowledge, raising concerns about the
reliability of their responses. We introduce a hybrid system that augments LLMs
with domain-specific knowledge graphs (KGs), thereby aiming to enhance factual
correctness using a KG-based retrieval approach. We focus on a medical KG to
demonstrate our methodology, which includes (1) pre-processing, (2) Cypher
query generation, (3) Cypher query processing, (4) KG retrieval, and (5)
LLM-enhanced response generation. We evaluate our system on a curated dataset
of 69 samples, achieving a precision of 78\% in retrieving correct KG nodes.
Our findings indicate that the hybrid system surpasses a standalone LLM in
accuracy and completeness, as verified by an LLM-as-a-Judge evaluation method.
This positions the system as a promising tool for applications that demand
factual correctness and completeness, such as target identification -- a
critical process in pinpointing biological entities for disease treatment or
crop enhancement. Moreover, its intuitive search interface and ability to
provide accurate responses within seconds make it well-suited for
time-sensitive, precision-focused research contexts. We publish the source code
together with the dataset and the prompt templates used.


---

**[140. [2407.12853] Automated Justification Production for Claim Veracity in Fact Checking:
  A Survey on Architectures and Approaches](https://arxiv.org/pdf/2407.12853.pdf)** (2025-04-08)

*Islam Eldifrawi, Shengrui Wang, Amine Trabelsi*

  Automated Fact-Checking (AFC) is the automated verification of claim
accuracy. AFC is crucial in discerning truth from misinformation, especially
given the huge amounts of content are generated online daily. Current research
focuses on predicting claim veracity through metadata analysis and language
scrutiny, with an emphasis on justifying verdicts. This paper surveys recent
methodologies, proposing a comprehensive taxonomy and presenting the evolution
of research in that landscape. A comparative analysis of methodologies and
future directions for improving fact-checking explainability are also
discussed.


---

**[141. [2502.11142] NavRAG: Generating User Demand Instructions for Embodied Navigation
  through Retrieval-Augmented LLM](https://arxiv.org/pdf/2502.11142.pdf)** (2025-03-10)

*Zihan Wang, Yaohui Zhu, Gim Hee Lee, Yachun Fan*

  Vision-and-Language Navigation (VLN) is an essential skill for embodied
agents, allowing them to navigate in 3D environments following natural language
instructions. High-performance navigation models require a large amount of
training data, the high cost of manually annotating data has seriously hindered
this field. Therefore, some previous methods translate trajectory videos into
step-by-step instructions for expanding data, but such instructions do not
match well with users' communication styles that briefly describe destinations
or state specific needs. Moreover, local navigation trajectories overlook
global context and high-level task planning. To address these issues, we
propose NavRAG, a retrieval-augmented generation (RAG) framework that generates
user demand instructions for VLN. NavRAG leverages LLM to build a hierarchical
scene description tree for 3D scene understanding from global layout to local
details, then simulates various user roles with specific demands to retrieve
from the scene tree, generating diverse instructions with LLM. We annotate over
2 million navigation instructions across 861 scenes and evaluate the data
quality and navigation performance of trained models.


---

**[142. [2312.05834] Evidence-based Interpretable Open-domain Fact-checking with Large
  Language Models](https://arxiv.org/pdf/2312.05834.pdf)** (2023-12-12)

*Xin Tan, Bowei Zou, Ai Ti Aw*

  Universal fact-checking systems for real-world claims face significant
challenges in gathering valid and sufficient real-time evidence and making
reasoned decisions. In this work, we introduce the Open-domain Explainable
Fact-checking (OE-Fact) system for claim-checking in real-world scenarios. The
OE-Fact system can leverage the powerful understanding and reasoning
capabilities of large language models (LLMs) to validate claims and generate
causal explanations for fact-checking decisions. To adapt the traditional
three-module fact-checking framework to the open domain setting, we first
retrieve claim-related information as relevant evidence from open websites.
After that, we retain the evidence relevant to the claim through LLM and
similarity calculation for subsequent verification. We evaluate the performance
of our adapted three-module OE-Fact system on the Fact Extraction and
Verification (FEVER) dataset. Experimental results show that our OE-Fact system
outperforms general fact-checking baseline systems in both closed- and
open-domain scenarios, ensuring stable and accurate verdicts while providing
concise and convincing real-time explanations for fact-checking decisions.


---

**[143. [2410.14567] ELOQ: Resources for Enhancing LLM Detection of Out-of-Scope Questions](https://arxiv.org/pdf/2410.14567.pdf)** (2025-04-10)

*Zhiyuan Peng, Jinming Nian, Alexandre Evfimievski, Yi Fang*

  Large Language Models (LLMs) are widely used in Conversational AI systems to
generate responses to user inquiries. However, many natural questions lack
well-defined answers. While existing studies primarily focus on question types
such as false premises, they often overlook out-of-scope questions, where the
provided document is semantically highly similar to the query but does not
contain the required answer. In this paper, we propose a guided
hallucination-based method to efficiently generate a diverse set of
out-of-scope questions from a given document corpus. We then evaluate multiple
LLMs based on their effectiveness in confusion detection and appropriate
response generation. Furthermore, we introduce an improved method for detecting
such out-of-scope questions, enhancing the reliability of LLM-based
question-answering systems.


---

**[144. [2503.00032] Detecting LLM-Generated Korean Text through Linguistic Feature Analysis](https://arxiv.org/pdf/2503.00032.pdf)** (2025-03-05)

*Shinwoo Park, Shubin Kim, Do-Kyung Kim, Yo-Sub Han*

  The rapid advancement of large language models (LLMs) increases the
difficulty of distinguishing between human-written and LLM-generated text.
Detecting LLM-generated text is crucial for upholding academic integrity,
preventing plagiarism, protecting copyrights, and ensuring ethical research
practices. Most prior studies on detecting LLM-generated text focus primarily
on English text. However, languages with distinct morphological and syntactic
characteristics require specialized detection approaches. Their unique
structures and usage patterns can hinder the direct application of methods
primarily designed for English. Among such languages, we focus on Korean, which
has relatively flexible spacing rules, a rich morphological system, and less
frequent comma usage compared to English. We introduce KatFish, the first
benchmark dataset for detecting LLM-generated Korean text. The dataset consists
of text written by humans and generated by four LLMs across three genres.
  By examining spacing patterns, part-of-speech diversity, and comma usage, we
illuminate the linguistic differences between human-written and LLM-generated
Korean text. Building on these observations, we propose KatFishNet, a detection
method specifically designed for the Korean language. KatFishNet achieves an
average of 19.78% higher AUROC compared to the best-performing existing
detection method. Our code and data are available at
https://github.com/Shinwoo-Park/detecting_llm_generated_korean_text_through_linguistic_analysis.


---

**[145. [2410.12608] Not All Votes Count! Programs as Verifiers Improve Self-Consistency of
  Language Models for Math Reasoning](https://arxiv.org/pdf/2410.12608.pdf)** (2024-12-18)

*Vernon Y. H. Toh, Deepanway Ghosal, Soujanya Poria*

  Large language models (LLMs) have shown increasing competence in solving
mathematical reasoning problems. However, many open-source LLMs still struggle
with errors in calculation and semantic understanding during intermediate
reasoning steps. In this work, we introduce Prove, a simple yet effective
framework that leverages translated programs derived from natural language
solutions as a verification mechanism to filter out potentially incorrect
reasoning paths before aggregating final answers. Unlike vanilla majority
voting, our approach filters out solutions whose corresponding program output
is inconsistent with the generated solution, aggregating only those that pass
verification. We conducted extensive experiments using 13 open-source LLMs from
various model families and sizes, ranging from 0.5B to 13B parameters, across
eight mathematical benchmarks. Our results show that Prove consistently
outperforms vanilla majority voting as a heuristic for solving mathematical
reasoning tasks across all model sizes and datasets, achieving improvements of
up to 18% on GSM8K and 8% on MATH-500. Our codes are available at
https://github.com/declare-lab/prove.


---

**[146. [2309.11495] Chain-of-Verification Reduces Hallucination in Large Language Models](https://arxiv.org/pdf/2309.11495.pdf)** (2023-09-26)

*Shehzaad Dhuliawala, Mojtaba Komeili, Jing Xu, Roberta Raileanu, Xian Li, Asli Celikyilmaz, Jason Weston*

  Generation of plausible yet incorrect factual information, termed
hallucination, is an unsolved issue in large language models. We study the
ability of language models to deliberate on the responses they give in order to
correct their mistakes. We develop the Chain-of-Verification (CoVe) method
whereby the model first (i) drafts an initial response; then (ii) plans
verification questions to fact-check its draft; (iii) answers those questions
independently so the answers are not biased by other responses; and (iv)
generates its final verified response. In experiments, we show CoVe decreases
hallucinations across a variety of tasks, from list-based questions from
Wikidata, closed book MultiSpanQA and longform text generation.


---

**[147. [2306.11489] Give Us the Facts: Enhancing Large Language Models with Knowledge Graphs
  for Fact-aware Language Modeling](https://arxiv.org/pdf/2306.11489.pdf)** (2024-01-31)

*Linyao Yang, Hongyang Chen, Zhao Li, Xiao Ding, Xindong Wu*

  Recently, ChatGPT, a representative large language model (LLM), has gained
considerable attention due to its powerful emergent abilities. Some researchers
suggest that LLMs could potentially replace structured knowledge bases like
knowledge graphs (KGs) and function as parameterized knowledge bases. However,
while LLMs are proficient at learning probabilistic language patterns based on
large corpus and engaging in conversations with humans, they, like previous
smaller pre-trained language models (PLMs), still have difficulty in recalling
facts while generating knowledge-grounded contents. To overcome these
limitations, researchers have proposed enhancing data-driven PLMs with
knowledge-based KGs to incorporate explicit factual knowledge into PLMs, thus
improving their performance to generate texts requiring factual knowledge and
providing more informed responses to user queries. This paper reviews the
studies on enhancing PLMs with KGs, detailing existing knowledge graph enhanced
pre-trained language models (KGPLMs) as well as their applications. Inspired by
existing studies on KGPLM, this paper proposes to enhance LLMs with KGs by
developing knowledge graph-enhanced large language models (KGLLMs). KGLLM
provides a solution to enhance LLMs' factual reasoning ability, opening up new
avenues for LLM research.


---

**[148. [2502.10440] Towards Copyright Protection for Knowledge Bases of Retrieval-augmented
  Language Models via Ownership Verification with Reasoning](https://arxiv.org/pdf/2502.10440.pdf)** (2025-02-18)

*Junfeng Guo, Yiming Li, Ruibo Chen, Yihan Wu, Chenxi Liu, Yanshuo Chen, Heng Huang*

  Large language models (LLMs) are increasingly integrated into real-world
applications through retrieval-augmented generation (RAG) mechanisms to
supplement their responses with up-to-date and domain-specific knowledge.
However, the valuable and often proprietary nature of the knowledge bases used
in RAG introduces the risk of unauthorized usage by adversaries. Existing
methods that can be generalized as watermarking techniques to protect these
knowledge bases typically involve poisoning attacks. However, these methods
require to alter the results of verification samples (\eg, generating incorrect
outputs), inevitably making them susceptible to anomaly detection and even
introduce new security risks. To address these challenges, we propose \name{}
for `harmless' copyright protection of knowledge bases. Instead of manipulating
LLM's final output, \name{} implants distinct verification behaviors in the
space of chain-of-thought (CoT) reasoning, maintaining the correctness of the
final answer. Our method has three main stages: (1) \textbf{Generating CoTs}:
For each verification question, we generate two CoTs, including a target CoT
for building watermark behaviors; (2) \textbf{Optimizing Watermark Phrases and
Target CoTs}: We optimize them to minimize retrieval errors under the black-box
setting of suspicious LLM, ensuring that the watermarked verification queries
activate the target CoTs without being activated in non-watermarked ones; (3)
\textbf{Ownership Verification}: We exploit a pairwise Wilcoxon test to
statistically verify whether a suspicious LLM is augmented with the protected
knowledge base by comparing its responses to watermarked and benign
verification queries. Our experiments on diverse benchmarks demonstrate that
\name{} effectively protects knowledge bases against unauthorized usage while
preserving the integrity and performance of the RAG.


---

**[149. [2404.10774] MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents](https://arxiv.org/pdf/2404.10774.pdf)** (2024-10-02)

*Liyan Tang, Philippe Laban, Greg Durrett*

  Recognizing if LLM output can be grounded in evidence is central to many
tasks in NLP: retrieval-augmented generation, summarization, document-grounded
dialogue, and more. Current approaches to this kind of fact-checking are based
on verifying each piece of a model generation against potential evidence using
an LLM. However, this process can be very computationally expensive, requiring
many calls to a model to check a single response. In this work, we show how to
build small fact-checking models that have GPT-4-level performance but for 400x
lower cost. We do this by constructing synthetic training data with GPT-4,
which involves creating realistic yet challenging instances of factual errors
via a structured generation procedure. Training on this data teaches models to
check each fact in the claim and recognize synthesis of information across
sentences. For evaluation, we unify datasets from recent work on fact-checking
and grounding LLM generations into a new benchmark, LLM-AggreFact. Our best
system MiniCheck-FT5 (770M parameters) outperforms all systems of comparable
size and reaches GPT-4 accuracy. We release LLM-AggreFact, code for data
synthesis, and models.


---

**[150. [2408.14317] Claim Verification in the Age of Large Language Models: A Survey](https://arxiv.org/pdf/2408.14317.pdf)** (2025-02-12)

*Alphaeus Dmonte, Roland Oruche, Marcos Zampieri, Prasad Calyam, Isabelle Augenstein*

  The large and ever-increasing amount of data available on the Internet
coupled with the laborious task of manual claim and fact verification has
sparked the interest in the development of automated claim verification
systems. Several deep learning and transformer-based models have been proposed
for this task over the years. With the introduction of Large Language Models
(LLMs) and their superior performance in several NLP tasks, we have seen a
surge of LLM-based approaches to claim verification along with the use of novel
methods such as Retrieval Augmented Generation (RAG). In this survey, we
present a comprehensive account of recent claim verification frameworks using
LLMs. We describe the different components of the claim verification pipeline
used in these frameworks in detail including common approaches to retrieval,
prompting, and fine-tuning. Finally, we describe publicly available English
datasets created for this task.


---

**[151. [2407.02351] Generative Large Language Models in Automated Fact-Checking: A Survey](https://arxiv.org/pdf/2407.02351.pdf)** (2024-10-31)

*Ivan Vykopal, Matúš Pikuliak, Simon Ostermann, Marián Šimko*

  The dissemination of false information on online platforms presents a serious
societal challenge. While manual fact-checking remains crucial, Large Language
Models (LLMs) offer promising opportunities to support fact-checkers with their
vast knowledge and advanced reasoning capabilities. This survey explores the
application of generative LLMs in fact-checking, highlighting various
approaches and techniques for prompting or fine-tuning these models. By
providing an overview of existing methods and their limitations, the survey
aims to enhance the understanding of how LLMs can be used in fact-checking and
to facilitate further progress in their integration into the fact-checking
process.


---

**[152. [2410.13153] Better to Ask in English: Evaluation of Large Language Models on
  English, Low-resource and Cross-Lingual Settings](https://arxiv.org/pdf/2410.13153.pdf)** (2024-10-18)

*Krishno Dey, Prerona Tarannum, Md. Arid Hasan, Imran Razzak, Usman Naseem*

  Large Language Models (LLMs) are trained on massive amounts of data, enabling
their application across diverse domains and tasks. Despite their remarkable
performance, most LLMs are developed and evaluated primarily in English.
Recently, a few multi-lingual LLMs have emerged, but their performance in
low-resource languages, especially the most spoken languages in South Asia, is
less explored. To address this gap, in this study, we evaluate LLMs such as
GPT-4, Llama 2, and Gemini to analyze their effectiveness in English compared
to other low-resource languages from South Asia (e.g., Bangla, Hindi, and
Urdu). Specifically, we utilized zero-shot prompting and five different prompt
settings to extensively investigate the effectiveness of the LLMs in
cross-lingual translated prompts. The findings of the study suggest that GPT-4
outperformed Llama 2 and Gemini in all five prompt settings and across all
languages. Moreover, all three LLMs performed better for English language
prompts than other low-resource language prompts. This study extensively
investigates LLMs in low-resource language contexts to highlight the
improvements required in LLMs and language-specific resources to develop more
generally purposed NLP applications.


---

**[153. [2401.16212] Better Call GPT, Comparing Large Language Models Against Lawyers](https://arxiv.org/pdf/2401.16212.pdf)** (2024-01-30)

*Onit AI Centre of Excellence  Lauren Martin, Onit AI Centre of Excellence  Nick Whitehouse, Onit AI Centre of Excellence  Stephanie Yiu, Onit AI Centre of Excellence  Lizzie Catterson, Onit AI Centre of Excellence  Rivindu Perera*

  This paper presents a groundbreaking comparison between Large Language Models
and traditional legal contract reviewers, Junior Lawyers and Legal Process
Outsourcers. We dissect whether LLMs can outperform humans in accuracy, speed,
and cost efficiency during contract review. Our empirical analysis benchmarks
LLMs against a ground truth set by Senior Lawyers, uncovering that advanced
models match or exceed human accuracy in determining legal issues. In speed,
LLMs complete reviews in mere seconds, eclipsing the hours required by their
human counterparts. Cost wise, LLMs operate at a fraction of the price,
offering a staggering 99.97 percent reduction in cost over traditional methods.
These results are not just statistics, they signal a seismic shift in legal
practice. LLMs stand poised to disrupt the legal industry, enhancing
accessibility and efficiency of legal services. Our research asserts that the
era of LLM dominance in legal contract review is upon us, challenging the
status quo and calling for a reimagined future of legal workflows.


---

**[154. [2410.04463] Wrong-of-Thought: An Integrated Reasoning Framework with
  Multi-Perspective Verification and Wrong Information](https://arxiv.org/pdf/2410.04463.pdf)** (2024-10-08)

*Yongheng Zhang, Qiguang Chen, Jingxuan Zhou, Peng Wang, Jiasheng Si, Jin Wang, Wenpeng Lu, Libo Qin*

  Chain-of-Thought (CoT) has become a vital technique for enhancing the
performance of Large Language Models (LLMs), attracting increasing attention
from researchers. One stream of approaches focuses on the iterative enhancement
of LLMs by continuously verifying and refining their reasoning outputs for
desired quality. Despite its impressive results, this paradigm faces two
critical issues: (1) Simple verification methods: The current paradigm relies
solely on a single verification method. (2) Wrong Information Ignorance:
Traditional paradigms directly ignore wrong information during reasoning and
refine the logic paths from scratch each time. To address these challenges, we
propose Wrong-of-Thought (WoT), which includes two core modules: (1)
Multi-Perspective Verification: A multi-perspective verification method for
accurately refining the reasoning process and result, and (2) Wrong Information
Utilization: Utilizing wrong information to alert LLMs and reduce the
probability of LLMs making same mistakes. Experiments on 8 popular datasets and
5 LLMs demonstrate that WoT surpasses all previous baselines. In addition, WoT
exhibits powerful capabilities in difficult computation tasks.


---

**[155. [2503.04830] Cite Before You Speak: Enhancing Context-Response Grounding in
  E-commerce Conversational LLM-Agents](https://arxiv.org/pdf/2503.04830.pdf)** (2025-03-11)

*Jingying Zeng, Hui Liu, Zhenwei Dai, Xianfeng Tang, Chen Luo, Samarth Varshney, Zhen Li, Qi He*

  With the advancement of conversational large language models (LLMs), several
LLM-based Conversational Shopping Agents (CSA) have been developed to help
customers answer questions and smooth their shopping journey in e-commerce
domain. The primary objective in building a trustworthy CSA is to ensure the
agent's responses are accurate and factually grounded, which is essential for
building customer trust and encouraging continuous engagement. However, two
challenges remain. First, LLMs produce hallucinated or unsupported claims. Such
inaccuracies risk spreading misinformation and diminishing customer trust.
Second, without providing knowledge source attribution in CSA response,
customers struggle to verify LLM-generated information. To address these
challenges, we present an easily productionized solution that enables a
"citation experience" utilizing In-context Learning (ICL) and
Multi-UX-Inference (MUI) to generate responses with citations to attribute its
original sources without interfering other existing UX features. With proper UX
design, these citation marks can be linked to the related product information
and display the source to our customers. In this work, we also build
auto-metrics and scalable benchmarks to holistically evaluate LLM's grounding
and attribution capabilities. Our experiments demonstrate that incorporating
this citation generation paradigm can substantially enhance the grounding of
LLM responses by 13.83% on the real-world data. As such, our solution not only
addresses the immediate challenges of LLM grounding issues but also adds
transparency to conversational AI.


---

**[156. [2406.18326] PaCoST: Paired Confidence Significance Testing for Benchmark
  Contamination Detection in Large Language Models](https://arxiv.org/pdf/2406.18326.pdf)** (2025-03-19)

*Huixuan Zhang, Yun Lin, Xiaojun Wan*

  Large language models (LLMs) are known to be trained on vast amounts of data,
which may unintentionally or intentionally include data from commonly used
benchmarks. This inclusion can lead to cheatingly high scores on model
leaderboards, yet result in disappointing performance in real-world
applications. To address this benchmark contamination problem, we first propose
a set of requirements that practical contamination detection methods should
follow. Following these proposed requirements, we introduce PaCoST, a Paired
Confidence Significance Testing to effectively detect benchmark contamination
in LLMs. Our method constructs a counterpart for each piece of data with the
same distribution, and performs statistical analysis of the corresponding
confidence to test whether the model is significantly more confident under the
original benchmark. We validate the effectiveness of PaCoST and apply it on
popular open-source models and benchmarks. We find that almost all models and
benchmarks we tested are suspected contaminated more or less. We finally call
for new LLM evaluation methods.


---

**[157. [2410.03727] FaithEval: Can Your Language Model Stay Faithful to Context, Even If
  "The Moon is Made of Marshmallows"](https://arxiv.org/pdf/2410.03727.pdf)** (2024-10-10)

*Yifei Ming, Senthil Purushwalkam, Shrey Pandit, Zixuan Ke, Xuan-Phi Nguyen, Caiming Xiong, Shafiq Joty*

  Ensuring faithfulness to context in large language models (LLMs) and
retrieval-augmented generation (RAG) systems is crucial for reliable deployment
in real-world applications, as incorrect or unsupported information can erode
user trust. Despite advancements on standard benchmarks, faithfulness
hallucination-where models generate responses misaligned with the provided
context-remains a significant challenge. In this work, we introduce FaithEval,
a novel and comprehensive benchmark tailored to evaluate the faithfulness of
LLMs in contextual scenarios across three diverse tasks: unanswerable,
inconsistent, and counterfactual contexts. These tasks simulate real-world
challenges where retrieval mechanisms may surface incomplete, contradictory, or
fabricated information. FaithEval comprises 4.9K high-quality problems in
total, validated through a rigorous four-stage context construction and
validation framework, employing both LLM-based auto-evaluation and human
validation. Our extensive study across a wide range of open-source and
proprietary models reveals that even state-of-the-art models often struggle to
remain faithful to the given context, and that larger models do not necessarily
exhibit improved faithfulness.Project is available at:
\url{https://github.com/SalesforceAIResearch/FaithEval}.


---

**[158. [2411.12764] SEFD: Semantic-Enhanced Framework for Detecting LLM-Generated Text](https://arxiv.org/pdf/2411.12764.pdf)** (2024-11-21)

*Weiqing He, Bojian Hou, Tianqi Shang, Davoud Ataee Tarzanagh, Qi Long, Li Shen*

  The widespread adoption of large language models (LLMs) has created an urgent
need for robust tools to detect LLM-generated text, especially in light of
\textit{paraphrasing} techniques that often evade existing detection methods.
To address this challenge, we present a novel semantic-enhanced framework for
detecting LLM-generated text (SEFD) that leverages a retrieval-based mechanism
to fully utilize text semantics. Our framework improves upon existing detection
methods by systematically integrating retrieval-based techniques with
traditional detectors, employing a carefully curated retrieval mechanism that
strikes a balance between comprehensive coverage and computational efficiency.
We showcase the effectiveness of our approach in sequential text scenarios
common in real-world applications, such as online forums and Q\&A platforms.
Through comprehensive experiments across various LLM-generated texts and
detection methods, we demonstrate that our framework substantially enhances
detection accuracy in paraphrasing scenarios while maintaining robustness for
standard LLM-generated content.


---

**[159. [2309.11392] Retrieving Supporting Evidence for Generative Question Answering](https://arxiv.org/pdf/2309.11392.pdf)** (2023-09-29)

*Siqing Huo, Negar Arabzadeh, Charles L. A. Clarke*

  Current large language models (LLMs) can exhibit near-human levels of
performance on many natural language-based tasks, including open-domain
question answering. Unfortunately, at this time, they also convincingly
hallucinate incorrect answers, so that responses to questions must be verified
against external sources before they can be accepted at face value. In this
paper, we report two simple experiments to automatically validate generated
answers against a corpus. We base our experiments on questions and passages
from the MS MARCO (V1) test collection, and a retrieval pipeline consisting of
sparse retrieval, dense retrieval and neural rerankers. In the first
experiment, we validate the generated answer in its entirety. After presenting
a question to an LLM and receiving a generated answer, we query the corpus with
the combination of the question + generated answer. We then present the LLM
with the combination of the question + generated answer + retrieved answer,
prompting it to indicate if the generated answer can be supported by the
retrieved answer. In the second experiment, we consider the generated answer at
a more granular level, prompting the LLM to extract a list of factual
statements from the answer and verifying each statement separately. We query
the corpus with each factual statement and then present the LLM with the
statement and the corresponding retrieved evidence. The LLM is prompted to
indicate if the statement can be supported and make necessary edits using the
retrieved material. With an accuracy of over 80%, we find that an LLM is
capable of verifying its generated answer when a corpus of supporting material
is provided. However, manual assessment of a random sample of questions reveals
that incorrect generated answers are missed by this verification process. While
this verification process can reduce hallucinations, it can not entirely
eliminate them.


---

**[160. [2410.12265] An Automatic and Cost-Efficient Peer-Review Framework for Language
  Generation Evaluation](https://arxiv.org/pdf/2410.12265.pdf)** (2024-10-17)

*Junjie Chen, Weihang Su, Zhumin Chu, Haitao Li, Qinyao Ai, Yiqun Liu, Min Zhang, Shaoping Ma*

  With the rapid development of large language models (LLMs), how to
efficiently evaluate them has become an important research question. Existing
evaluation methods often suffer from high costs, limited test formats, the need
of human references, and systematic evaluation biases. To address these
limitations, our study introduces the Auto-PRE, an automatic LLM evaluation
framework based on peer review. In contrast to previous studies that rely on
human annotations, Auto-PRE selects evaluator LLMs automatically based on their
inherent traits including consistency, self-confidence, and pertinence. We
conduct extensive experiments on three tasks: summary generation, non-factoid
question-answering, and dialogue generation. Experimental results indicate our
Auto-PRE achieves state-of-the-art performance at a lower cost. Moreover, our
study highlights the impact of prompt strategies and evaluation formats on
evaluation performance, offering guidance for method optimization in the
future.


---

**[161. [2406.12277] What Matters in Memorizing and Recalling Facts? Multifaceted Benchmarks
  for Knowledge Probing in Language Models](https://arxiv.org/pdf/2406.12277.pdf)** (2024-10-10)

*Xin Zhao, Naoki Yoshinaga, Daisuke Oba*

  Language models often struggle with handling factual knowledge, exhibiting
factual hallucination issue. This makes it vital to evaluate the models'
ability to recall its parametric knowledge about facts. In this study, we
introduce a knowledge probing benchmark, BELIEF(ICL), to evaluate the knowledge
recall ability of both encoder- and decoder-based pre-trained language models
(PLMs) from diverse perspectives. BELIEFs utilize a multi-prompt dataset to
evaluate PLM's accuracy, consistency, and reliability in factual knowledge
recall. To enable a more reliable evaluation with BELIEFs, we
semi-automatically create MyriadLAMA, which has massively diverse prompts. We
validate the effectiveness of BELIEFs in comprehensively evaluating PLM's
knowledge recall ability on diverse PLMs, including recent large language
models (LLMs). We then investigate key factors in memorizing and recalling
facts in PLMs, such as model size, pretraining strategy and corpora,
instruction-tuning process and in-context learning settings. Finally, we reveal
the limitation of the prompt-based knowledge probing. The MyriadLAMA is
publicized.


---

**[162. [2402.02549] Are Large Language Models Table-based Fact-Checkers?](https://arxiv.org/pdf/2402.02549.pdf)** (2024-11-14)

*Hanwen Zhang, Qingyi Si, Peng Fu, Zheng Lin, Weiping Wang*

  Table-based Fact Verification (TFV) aims to extract the entailment relation
between statements and structured tables. Existing TFV methods based on
small-scaled models suffer from insufficient labeled data and weak zero-shot
ability. Recently, the appearance of Large Language Models (LLMs) has gained
lots of attraction in research fields. They have shown powerful zero-shot and
in-context learning abilities on several NLP tasks, but their potential on TFV
is still unknown. In this work, we implement a preliminary study about whether
LLMs are table-based fact-checkers. In detail, we design diverse prompts to
explore how the in-context learning can help LLMs in TFV, i.e., zero-shot and
few-shot TFV capability. Besides, we carefully design and construct TFV
instructions to study the performance gain brought by the instruction tuning of
LLMs. Experimental results demonstrate that LLMs can achieve acceptable results
on zero-shot and few-shot TFV with prompt engineering, while instruction-tuning
can stimulate the TFV capability significantly. We also make some valuable
findings about the format of zero-shot prompts and the number of in-context
examples. Finally, we analyze some possible directions to promote the accuracy
of TFV via LLMs, which is beneficial to further research of table reasoning.


---

**[163. [2304.14732] Search-in-the-Chain: Interactively Enhancing Large Language Models with
  Search for Knowledge-intensive Tasks](https://arxiv.org/pdf/2304.14732.pdf)** (2024-02-27)

*Shicheng Xu, Liang Pang, Huawei Shen, Xueqi Cheng, Tat-Seng Chua*

  Making the content generated by Large Language Model (LLM), accurate,
credible and traceable is crucial, especially in complex knowledge-intensive
tasks that require multi-step reasoning and each step needs knowledge to solve.
Retrieval-augmented generation is good potential to solve this problem.
However, where and how to introduce Information Retrieval (IR) to LLM is a big
challenge. Previous work has the problems that wrong knowledge retrieved by IR
misleads the LLM and interaction between IR and LLM breaks the reasoning chain
of LLM. This paper proposes a novel framework named
\textbf{Search-in-the-Chain} (SearChain) for the interaction between LLM and IR
to solve the challenges. First, LLM generates the reasoning chain named
Chain-of-Query (CoQ) where each node consists of an IR-oriented query-answer
pair. Second, IR verifies the answer of each node of CoQ. It corrects the
answer that is not consistent with the retrieved information when IR gives high
confidence, which improves the credibility. Third, LLM can indicate its missing
knowledge in CoQ and rely on IR to provide this knowledge to LLM. These
operations improve the accuracy in terms of reasoning and knowledge. Finally,
SearChain generates the reasoning process and marks references to supporting
documents for each reasoning step, which improves traceability. Interaction
with IR in SearChain forms a novel reasoning path based on a tree, which
enables LLM to dynamically modify the direction of reasoning. Experiments show
that SearChain outperforms state-of-the-art baselines on complex
knowledge-intensive tasks including multi-hop Q\&A, slot filling, fact
checking, and long-form Q\&A.


---

**[164. [2405.03170] Oracle-Checker Scheme for Evaluating a Generative Large Language Model](https://arxiv.org/pdf/2405.03170.pdf)** (2024-05-07)

*Yueling Jenny Zeng, Li-C. Wang, Thomas Ibbetson*

  This work presents a novel approach called oracle-checker scheme for
evaluating the answer given by a generative large language model (LLM). Two
types of checkers are presented. The first type of checker follows the idea of
property testing. The second type of checker follows the idea of program
checking. Their applications are demonstrated in two separate contexts, entity
extraction and paraphrase decision, respectively.


---

**[165. [2309.16621] Stress Testing Chain-of-Thought Prompting for Large Language Models](https://arxiv.org/pdf/2309.16621.pdf)** (2023-09-29)

*Aayush Mishra, Karan Thakkar*

  This report examines the effectiveness of Chain-of-Thought (CoT) prompting in
improving the multi-step reasoning abilities of large language models (LLMs).
Inspired by previous studies \cite{Min2022RethinkingWork}, we analyze the
impact of three types of CoT prompt perturbations, namely CoT order, CoT
values, and CoT operators on the performance of GPT-3 on various tasks. Our
findings show that incorrect CoT prompting leads to poor performance on
accuracy metrics. Correct values in the CoT is crucial for predicting correct
answers. Moreover, incorrect demonstrations, where the CoT operators or the CoT
order are wrong, do not affect the performance as drastically when compared to
the value based perturbations. This research deepens our understanding of CoT
prompting and opens some new questions regarding the capability of LLMs to
learn reasoning in context.


---

**[166. [2404.00216] Is Factuality Enhancement a Free Lunch For LLMs? Better Factuality Can
  Lead to Worse Context-Faithfulness](https://arxiv.org/pdf/2404.00216.pdf)** (2024-10-07)

*Baolong Bi, Shenghua Liu, Yiwei Wang, Lingrui Mei, Junfeng Fang, Hongcheng Gao, Shiyu Ni, Xueqi Cheng*

  As the modern tools of choice for text understanding and generation, large
language models (LLMs) are expected to accurately output answers by leveraging
the input context. This requires LLMs to possess both context-faithfulness and
factual accuracy. Extensive efforts have been made to enable better outputs
from LLMs by mitigating hallucinations through factuality enhancement methods.
However, they also pose risks of hindering context-faithfulness, as factuality
enhancement can lead LLMs to become overly confident in their parametric
knowledge, causing them to overlook the relevant input context. In this work,
we argue that current factuality enhancement methods can significantly
undermine the context-faithfulness of LLMs. We first revisit the current
factuality enhancement methods and evaluate their effectiveness in enhancing
factual accuracy. Next, we evaluate their performance on knowledge editing
tasks to assess the potential impact on context-faithfulness. The experimental
results reveal that while these methods may yield inconsistent improvements in
factual accuracy, they also cause a more severe decline in
context-faithfulness, with the largest decrease reaching a striking 69.7\%. To
explain these declines, we analyze the hidden states and logit distributions
for the tokens representing new knowledge and parametric knowledge
respectively, highlighting the limitations of current approaches. Our finding
highlights the complex trade-offs inherent in enhancing LLMs. Therefore, we
recommend that more research on LLMs' factuality enhancement make efforts to
reduce the sacrifice of context-faithfulness.


---

**[167. [2504.10326] AlayaDB: The Data Foundation for Efficient and Effective Long-context
  LLM Inference](https://arxiv.org/pdf/2504.10326.pdf)** (2025-04-15)

*Yangshen Deng, Zhengxin You, Long Xiang, Qilong Li, Peiqi Yuan, Zhaoyang Hong, Yitao Zheng, Wanting Li, Runzhong Li, Haotian Liu, Kyriakos Mouratidis, Man Lung Yiu, Huan Li, Qiaomu Shen, Rui Mao, Bo Tang*

  AlayaDB is a cutting-edge vector database system natively architected for
efficient and effective long-context inference for Large Language Models (LLMs)
at AlayaDB AI. Specifically, it decouples the KV cache and attention
computation from the LLM inference systems, and encapsulates them into a novel
vector database system. For the Model as a Service providers (MaaS), AlayaDB
consumes fewer hardware resources and offers higher generation quality for
various workloads with different kinds of Service Level Objectives (SLOs), when
comparing with the existing alternative solutions (e.g., KV cache
disaggregation, retrieval-based sparse attention). The crux of AlayaDB is that
it abstracts the attention computation and cache management for LLM inference
into a query processing procedure, and optimizes the performance via a native
query optimizer. In this work, we demonstrate the effectiveness of AlayaDB via
(i) three use cases from our industry partners, and (ii) extensive experimental
results on LLM inference benchmarks.


---

**[168. [2503.12505] MPBench: A Comprehensive Multimodal Reasoning Benchmark for Process
  Errors Identification](https://arxiv.org/pdf/2503.12505.pdf)** (2025-03-18)

*Zhaopan Xu, Pengfei Zhou, Jiaxin Ai, Wangbo Zhao, Kai Wang, Xiaojiang Peng, Wenqi Shao, Hongxun Yao, Kaipeng Zhang*

  Reasoning is an essential capacity for large language models (LLMs) to
address complex tasks, where the identification of process errors is vital for
improving this ability. Recently, process-level reward models (PRMs) were
proposed to provide step-wise rewards that facilitate reinforcement learning
and data production during training and guide LLMs toward correct steps during
inference, thereby improving reasoning accuracy. However, existing benchmarks
of PRMs are text-based and focus on error detection, neglecting other scenarios
like reasoning search. To address this gap, we introduce MPBench, a
comprehensive, multi-task, multimodal benchmark designed to systematically
assess the effectiveness of PRMs in diverse scenarios. MPBench employs three
evaluation paradigms, each targeting a specific role of PRMs in the reasoning
process: (1) Step Correctness, which assesses the correctness of each
intermediate reasoning step; (2) Answer Aggregation, which aggregates multiple
solutions and selects the best one; and (3) Reasoning Process Search, which
guides the search for optimal reasoning steps during inference. Through these
paradigms, MPBench makes comprehensive evaluations and provides insights into
the development of multimodal PRMs.


---

**[169. [2402.10735] Assessing the Reasoning Capabilities of LLMs in the context of
  Evidence-based Claim Verification](https://arxiv.org/pdf/2402.10735.pdf)** (2025-02-21)

*John Dougrez-Lewis, Mahmud Elahi Akhter, Federico Ruggeri, Sebastian Löbbers, Yulan He, Maria Liakata*

  Although LLMs have shown great performance on Mathematics and Coding related
reasoning tasks, the reasoning capabilities of LLMs regarding other forms of
reasoning are still an open problem. Here, we examine the issue of reasoning
from the perspective of claim verification. We propose a framework designed to
break down any claim paired with evidence into atomic reasoning types that are
necessary for verification. We use this framework to create Reasoning in
Evidence-based Claim Verification (RECV), the first claim verification
benchmark, incorporating real-world claims, to assess the deductive and
abductive reasoning capabilities of LLMs. The benchmark comprises of three
datasets, covering reasoning problems of increasing complexity. We evaluate
three state-of-the-art proprietary LLMs under multiple prompt settings. Our
results show that while LLMs can address deductive reasoning problems, they
consistently fail in cases of abductive reasoning. Moreover, we observe that
enhancing LLMs with rationale generation is not always beneficial. Nonetheless,
we find that generated rationales are semantically similar to those provided by
humans, especially in deductive reasoning cases.


---

**[170. [2311.08596] Are You Sure? Challenging LLMs Leads to Performance Drops in The
  FlipFlop Experiment](https://arxiv.org/pdf/2311.08596.pdf)** (2024-02-22)

*Philippe Laban, Lidiya Murakhovs'ka, Caiming Xiong, Chien-Sheng Wu*

  The interactive nature of Large Language Models (LLMs) theoretically allows
models to refine and improve their answers, yet systematic analysis of the
multi-turn behavior of LLMs remains limited. In this paper, we propose the
FlipFlop experiment: in the first round of the conversation, an LLM completes a
classification task. In a second round, the LLM is challenged with a follow-up
phrase like "Are you sure?", offering an opportunity for the model to reflect
on its initial answer, and decide whether to confirm or flip its answer. A
systematic study of ten LLMs on seven classification tasks reveals that models
flip their answers on average 46% of the time and that all models see a
deterioration of accuracy between their first and final prediction, with an
average drop of 17% (the FlipFlop effect). We conduct finetuning experiments on
an open-source LLM and find that finetuning on synthetically created data can
mitigate - reducing performance deterioration by 60% - but not resolve
sycophantic behavior entirely. The FlipFlop experiment illustrates the
universality of sycophantic behavior in LLMs and provides a robust framework to
analyze model behavior and evaluate future models.


---

**[171. [2405.16792] Laurel: Unblocking Automated Verification with Large Language Models](https://arxiv.org/pdf/2405.16792.pdf)** (2025-03-05)

*Eric Mugnier, Emmanuel Anaya Gonzalez, Ranjit Jhala, Nadia Polikarpova, Yuanyuan Zhou*

  Program verifiers such as Dafny automate proofs by outsourcing them to an SMT
solver. This automation is not perfect, however, and the solver often requires
hints in the form of assertions, creating a burden for the proof engineer. In
this paper, we propose Laurel, a tool that alleviates this burden by
automatically generating assertions using large language models (LLMs). To
improve the success rate of LLMs in this task, we design two domain-specific
prompting techniques. First, we help the LLM determine the location of the
missing assertion by analyzing the verifier's error message and inserting an
assertion placeholder at that location. Second, we provide the LLM with example
assertions from the same codebase, which we select based on a new proof
similarity metric. We evaluate our techniques on our new benchmark DafnyGym, a
dataset of complex lemmas we extracted from three real-world Dafny codebases.
Our evaluation shows that Laurel is able to generate over 56.6\% of the
required assertions given only a few attempts, making LLMs an affordable tool
for unblocking program verifiers without human intervention.


---

**[172. [2502.17535] The Lottery LLM Hypothesis, Rethinking What Abilities Should LLM
  Compression Preserve?](https://arxiv.org/pdf/2502.17535.pdf)** (2025-02-26)

*Zhenheng Tang, Xiang Liu, Qian Wang, Peijie Dong, Bingsheng He, Xiaowen Chu, Bo Li*

  Motivated by reducing the computational and storage costs of LLMs, model
compression and KV cache compression have attracted much attention from
researchers. However, current methods predominantly emphasize maintaining the
performance of compressed LLMs, as measured by perplexity or simple accuracy on
tasks of common sense knowledge QA and basic arithmetic reasoning. In this
blog, we present a brief review of recent advancements in LLMs related to
retrieval-augmented generation, multi-step reasoning, external tools, and
computational expressivity, all of which substantially enhance LLM performance.
Then, we propose a lottery LLM hypothesis suggesting that for a given LLM and
task, there exists a smaller lottery LLM capable of producing the same
performance as the original LLM with the assistance of multi-step reasoning and
external tools. Based on the review of current progress in LLMs, we discuss and
summarize the essential capabilities that the lottery LLM and KV cache
compression must possess, which are currently overlooked in existing methods.


---

**[173. [2504.12982] Accommodate Knowledge Conflicts in Retrieval-augmented LLMs: Towards
  Reliable Response Generation in the Wild](https://arxiv.org/pdf/2504.12982.pdf)** (2025-04-18)

*Jiatai Wang, Zhiwei Xu, Di Jin, Xuewen Yang, Tao Li*

  The proliferation of large language models (LLMs) has significantly advanced
information retrieval systems, particularly in response generation (RG).
Unfortunately, LLMs often face knowledge conflicts between internal memory and
retrievaled external information, arising from misinformation, biases, or
outdated knowledge. These conflicts undermine response reliability and
introduce uncertainty in decision-making. In this work, we analyze how LLMs
navigate knowledge conflicts from an information-theoretic perspective and
reveal that when conflicting and supplementary information exhibit significant
differences, LLMs confidently resolve their preferences. However, when the
distinction is ambiguous, LLMs experience heightened uncertainty. Based on this
insight, we propose Swin-VIB, a novel framework that integrates a pipeline of
variational information bottleneck models into adaptive augmentation of
retrieved information and guiding LLM preference in response generation.
Extensive experiments on single-choice, open-ended question-answering (QA), and
retrieval augmented generation (RAG) validate our theoretical findings and
demonstrate the efficacy of Swin-VIB. Notably, our method improves
single-choice task accuracy by at least 7.54\% over competitive baselines.


---

**[174. [2305.14623] Self-Checker: Plug-and-Play Modules for Fact-Checking with Large
  Language Models](https://arxiv.org/pdf/2305.14623.pdf)** (2024-04-02)

*Miaoran Li, Baolin Peng, Michel Galley, Jianfeng Gao, Zhu Zhang*

  Fact-checking is an essential task in NLP that is commonly utilized for
validating the factual accuracy of claims. Prior work has mainly focused on
fine-tuning pre-trained languages models on specific datasets, which can be
computationally intensive and time-consuming. With the rapid development of
large language models (LLMs), such as ChatGPT and GPT-3, researchers are now
exploring their in-context learning capabilities for a wide range of tasks. In
this paper, we aim to assess the capacity of LLMs for fact-checking by
introducing Self-Checker, a framework comprising a set of plug-and-play modules
that facilitate fact-checking by purely prompting LLMs in an almost zero-shot
setting. This framework provides a fast and efficient way to construct
fact-checking systems in low-resource environments. Empirical results
demonstrate the potential of Self-Checker in utilizing LLMs for fact-checking.
However, there is still significant room for improvement compared to SOTA
fine-tuned models, which suggests that LLM adoption could be a promising
approach for future fact-checking research.


---

**[175. [2305.13711] LLM-Eval: Unified Multi-Dimensional Automatic Evaluation for Open-Domain
  Conversations with Large Language Models](https://arxiv.org/pdf/2305.13711.pdf)** (2023-05-24)

*Yen-Ting Lin, Yun-Nung Chen*

  We propose LLM-Eval, a unified multi-dimensional automatic evaluation method
for open-domain conversations with large language models (LLMs). Existing
evaluation methods often rely on human annotations, ground-truth responses, or
multiple LLM prompts, which can be expensive and time-consuming. To address
these issues, we design a single prompt-based evaluation method that leverages
a unified evaluation schema to cover multiple dimensions of conversation
quality in a single model call. We extensively evaluate the performance of
LLM-Eval on various benchmark datasets, demonstrating its effectiveness,
efficiency, and adaptability compared to state-of-the-art evaluation methods.
Our analysis also highlights the importance of choosing suitable LLMs and
decoding strategies for accurate evaluation results. LLM-Eval offers a
versatile and robust solution for evaluating open-domain conversation systems,
streamlining the evaluation process and providing consistent performance across
diverse scenarios.


---

**[176. [2411.03079] Utilizing Precise and Complete Code Context to Guide LLM in Automatic
  False Positive Mitigation](https://arxiv.org/pdf/2411.03079.pdf)** (2024-11-06)

*University of Science and Technology of China  Jinbao Chen, University of Science and Technology of China  Hongjing Xiang, University of Science and Technology of China  Luhao Li, University of Science and Technology of China  Yu Zhang, University of Science and Technology of China  Boyao Ding, University of Science and Technology of China  Qingwei Li*

  Static Application Security Testing(SAST) tools are crucial for early bug
detection and code quality but often generate false positives that slow
development. Automating false positive mitigation is thus essential for
advancing SAST tools. Past efforts use static/dynamic analysis or machine
learning. The advent of Large Language Models, adept at understanding natural
language and code, offers promising ways to improve the accuracy and usability
of SAST tools. However, existing LLM-based methods need improvement in two key
areas: first, extracted code snippets related to warnings are often cluttered
with irrelevant control and data flows, reducing precision; second, critical
code contexts are often missing, leading to incomplete representations that can
mislead LLMs and cause inaccurate assessments. To ensure the use of precise and
complete code context, thereby avoiding misguidance and enabling LLMs to reach
accurate conclusions, we propose LLM4FPM. One of its core components is
eCPG-Slicer, which builds an extended code property graph and extracts
line-level, precise code context. Moreover, LLM4FPM incorporates FARF
algorithm, which builds a file reference graph and then efficiently detects all
files related to a warning in linear time, enabling eCPG-Slicer to gather
complete code context across these files. We evaluate LLM4FPM on Juliet
dataset, where it comprehensively outperforms the baseline, achieving an F1
score above 99% across various CWEs. LLM4FPM leverages a free, open-source
model, avoiding costly alternatives and reducing inspection costs by up to
$2758 per run on Juliet, with an average inspection time of 4.7 seconds per
warning. Our work emphasizes the critical impact of precise and complete code
context and highlights the potential of combining program analysis with LLMs,
improving the quality and efficiency of software development.


---

**[177. [2405.14092] Large Language Models Can Self-Correct with Key Condition Verification](https://arxiv.org/pdf/2405.14092.pdf)** (2024-10-04)

*Zhenyu Wu, Qingkai Zeng, Zhihan Zhang, Zhaoxuan Tan, Chao Shen, Meng Jiang*

  Intrinsic self-correct was a method that instructed large language models
(LLMs) to verify and correct their responses without external feedback.
Unfortunately, the study concluded that the LLMs could not self-correct
reasoning yet. We find that a simple yet effective verification method can
unleash inherent capabilities of the LLMs. That is to mask a key condition in
the question, add the current response to construct a verification question,
and predict the condition to verify the response. The condition can be an
entity in an open-domain question or a numeric value in a math question, which
requires minimal effort (via prompting) to identify. We propose an iterative
verify-then-correct framework to progressively identify and correct (probably)
false responses, named ProCo. We conduct experiments on three reasoning tasks.
On average, ProCo, with GPT-3.5-Turbo as the backend LLM, yields $+6.8$ exact
match on four open-domain question answering datasets, $+14.1$ accuracy on
three arithmetic reasoning datasets, and $+9.6$ accuracy on a commonsense
reasoning dataset, compared to Self-Correct. Our implementation is made
publicly available at https://wzy6642.github.io/proco.github.io/.


---

**[178. [2310.09820] Assessing the Reliability of Large Language Model Knowledge](https://arxiv.org/pdf/2310.09820.pdf)** (2023-10-17)

*Weixuan Wang, Barry Haddow, Alexandra Birch, Wei Peng*

  Large language models (LLMs) have been treated as knowledge bases due to
their strong performance in knowledge probing tasks. LLMs are typically
evaluated using accuracy, yet this metric does not capture the vulnerability of
LLMs to hallucination-inducing factors like prompt and context variability. How
do we evaluate the capabilities of LLMs to consistently produce factually
correct answers? In this paper, we propose MOdel kNowledge relIabiliTy scORe
(MONITOR), a novel metric designed to directly measure LLMs' factual
reliability. MONITOR computes the distance between the probability
distributions of a valid output and its counterparts produced by the same LLM
probing the same fact using different styles of prompts and
contexts.Experiments on a comprehensive range of 12 LLMs demonstrate the
effectiveness of MONITOR in evaluating the factual reliability of LLMs while
maintaining a low computational overhead. In addition, we release the FKTC
(Factual Knowledge Test Corpus) test set, containing 210,158 prompts in total
to foster research along this line (https://github.com/Vicky-Wil/MONITOR).


---

**[179. [2407.07321] Examining Long-Context Large Language Models for Environmental Review
  Document Comprehension](https://arxiv.org/pdf/2407.07321.pdf)** (2024-10-17)

*Hung Phan, Anurag Acharya, Rounak Meyur, Sarthak Chaturvedi, Shivam Sharma, Mike Parker, Dan Nally, Ali Jannesari, Karl Pazdernik, Mahantesh Halappanavar, Sai Munikoti, Sameera Horawalavithana*

  As LLMs become increasingly ubiquitous, researchers have tried various
techniques to augment the knowledge provided to these models. Long context and
retrieval-augmented generation (RAG) are two such methods that have recently
gained popularity. In this work, we examine the benefits of both of these
techniques by utilizing question answering (QA) task in a niche domain. While
the effectiveness of LLM-based QA systems has already been established at an
acceptable level in popular domains such as trivia and literature, it has not
often been established in niche domains that traditionally require specialized
expertise. We construct the NEPAQuAD1.0 benchmark to evaluate the performance
of five long-context LLMs -- Claude Sonnet, Gemini, GPT-4, Llama 3.1, and
Mistral -- when answering questions originating from Environmental Impact
Statements prepared by U.S. federal government agencies in accordance with the
National Environmental Environmental Act (NEPA). We specifically measure the
ability of LLMs to understand the nuances of legal, technical, and
compliance-related information present in NEPA documents in different
contextual scenarios. We test the LLMs' internal prior NEPA knowledge by
providing questions without any context, as well as assess how LLMs synthesize
the contextual information present in long NEPA documents to facilitate the
question/answering task. We compare the performance of the models in handling
different types of questions (e.g., problem-solving, divergent, etc.). Our
results suggest that RAG powered models significantly outperform those provided
with only the PDF context in terms of answer accuracy, regardless of the choice
of the LLM. Our further analysis reveals that many models perform better
answering closed type questions (Yes/No) than divergent and problem-solving
questions.


---

**[180. [2308.10443] Using Large Language Models for Cybersecurity Capture-The-Flag
  Challenges and Certification Questions](https://arxiv.org/pdf/2308.10443.pdf)** (2023-08-22)

*Wesley Tann, Yuancheng Liu, Jun Heng Sim, Choon Meng Seah, Ee-Chien Chang*

  The assessment of cybersecurity Capture-The-Flag (CTF) exercises involves
participants finding text strings or ``flags'' by exploiting system
vulnerabilities. Large Language Models (LLMs) are natural-language models
trained on vast amounts of words to understand and generate text; they can
perform well on many CTF challenges. Such LLMs are freely available to
students. In the context of CTF exercises in the classroom, this raises
concerns about academic integrity. Educators must understand LLMs' capabilities
to modify their teaching to accommodate generative AI assistance. This research
investigates the effectiveness of LLMs, particularly in the realm of CTF
challenges and questions. Here we evaluate three popular LLMs, OpenAI ChatGPT,
Google Bard, and Microsoft Bing. First, we assess the LLMs' question-answering
performance on five Cisco certifications with varying difficulty levels. Next,
we qualitatively study the LLMs' abilities in solving CTF challenges to
understand their limitations. We report on the experience of using the LLMs for
seven test cases in all five types of CTF challenges. In addition, we
demonstrate how jailbreak prompts can bypass and break LLMs' ethical
safeguards. The paper concludes by discussing LLM's impact on CTF exercises and
its implications.


---

**[181. [2411.16732] Multi-Reranker: Maximizing performance of retrieval-augmented generation
  in the FinanceRAG challenge](https://arxiv.org/pdf/2411.16732.pdf)** (2024-11-28)

*Joohyun Lee, Minji Roh*

  As Large Language Models (LLMs) increasingly address domain-specific
problems, their application in the financial sector has expanded rapidly. Tasks
that are both highly valuable and time-consuming, such as analyzing financial
statements, disclosures, and related documents, are now being effectively
tackled using LLMs. This paper details the development of a high-performance,
finance-specific Retrieval-Augmented Generation (RAG) system for the ACM-ICAIF
'24 FinanceRAG competition. We optimized performance through ablation studies
on query expansion and corpus refinement during the pre-retrieval phase. To
enhance retrieval accuracy, we employed multiple reranker models. Notably, we
introduced an efficient method for managing long context sizes during the
generation phase, significantly improving response quality without sacrificing
performance. We ultimately achieve 2nd place in the FinanceRAG Challenge. Our
key contributions include: (1) pre-retrieval ablation analysis, (2) an enhanced
retrieval algorithm, and (3) a novel approach for long-context management. This
work demonstrates the potential of LLMs in effectively processing and analyzing
complex financial data to generate accurate and valuable insights. The source
code and further details are available at https://github.com/cv-lee/FinanceRAG.


---

**[182. [2410.05318] Improving LLM Reasoning through Scaling Inference Computation with
  Collaborative Verification](https://arxiv.org/pdf/2410.05318.pdf)** (2024-10-10)

*Zhenwen Liang, Ye Liu, Tong Niu, Xiangliang Zhang, Yingbo Zhou, Semih Yavuz*

  Despite significant advancements in the general capability of large language
models (LLMs), they continue to struggle with consistent and accurate
reasoning, especially in complex tasks such as mathematical and code reasoning.
One key limitation is that LLMs are trained primarily on correct solutions,
reducing their ability to detect and learn from errors, which hampers their
ability to reliably verify and rank outputs. To address this, we scale up the
inference-time computation by generating multiple reasoning paths and employing
verifiers to assess and rank the generated outputs by correctness. To
facilitate this, we introduce a comprehensive dataset consisting of correct and
incorrect solutions for math and code tasks, generated by multiple LLMs. This
diverse set of solutions enables verifiers to more effectively distinguish and
rank correct answers from erroneous outputs. The training methods for building
verifiers were selected based on an extensive comparison of existing
approaches. Moreover, to leverage the unique strengths of different reasoning
strategies, we propose a novel collaborative method integrating
Chain-of-Thought (CoT) and Program-of-Thought (PoT) solutions for verification.
CoT provides a clear, step-by-step reasoning process that enhances
interpretability, while PoT, being executable, offers a precise and
error-sensitive validation mechanism. By taking both of their strengths, our
approach significantly improves the accuracy and reliability of reasoning
verification. Our verifiers, Math-Rev and Code-Rev, demonstrate substantial
performance gains to existing LLMs, achieving state-of-the-art results on
benchmarks such as GSM8k and MATH and even outperforming GPT-4o with
Qwen-72B-Instruct as the reasoner.


---

**[183. [2502.17638] Towards Robust Legal Reasoning: Harnessing Logical LLMs in Law](https://arxiv.org/pdf/2502.17638.pdf)** (2025-02-26)

*Manuj Kant, Sareh Nabi, Manav Kant, Roland Scharrer, Megan Ma, Marzieh Nabi*

  Legal services rely heavily on text processing. While large language models
(LLMs) show promise, their application in legal contexts demands higher
accuracy, repeatability, and transparency. Logic programs, by encoding legal
concepts as structured rules and facts, offer reliable automation, but require
sophisticated text extraction. We propose a neuro-symbolic approach that
integrates LLMs' natural language understanding with logic-based reasoning to
address these limitations.
  As a legal document case study, we applied neuro-symbolic AI to
coverage-related queries in insurance contracts using both closed and
open-source LLMs. While LLMs have improved in legal reasoning, they still lack
the accuracy and consistency required for complex contract analysis. In our
analysis, we tested three methodologies to evaluate whether a specific claim is
covered under a contract: a vanilla LLM, an unguided approach that leverages
LLMs to encode both the contract and the claim, and a guided approach that uses
a framework for the LLM to encode the contract. We demonstrated the promising
capabilities of LLM + Logic in the guided approach.


---

**[184. [2502.18573] FactReasoner: A Probabilistic Approach to Long-Form Factuality
  Assessment for Large Language Models](https://arxiv.org/pdf/2502.18573.pdf)** (2025-02-27)

*Radu Marinescu, Debarun Bhattacharjya, Junkyu Lee, Tigran Tchrakian, Javier Carnerero Cano, Yufang Hou, Elizabeth Daly, Alessandra Pascale*

  Large language models (LLMs) have demonstrated vast capabilities on
generative tasks in recent years, yet they struggle with guaranteeing the
factual correctness of the generated content. This makes these models
unreliable in realistic situations where factually accurate responses are
expected. In this paper, we propose FactReasoner, a new factuality assessor
that relies on probabilistic reasoning to assess the factuality of a long-form
generated response. Specifically, FactReasoner decomposes the response into
atomic units, retrieves relevant contexts for them from an external knowledge
source, and constructs a joint probability distribution over the atoms and
contexts using probabilistic encodings of the logical relationships
(entailment, contradiction) between the textual utterances corresponding to the
atoms and contexts. FactReasoner then computes the posterior probability of
whether atomic units in the response are supported by the retrieved contexts.
Our experiments on labeled and unlabeled benchmark datasets demonstrate clearly
that FactReasoner improves considerably over state-of-the-art prompt-based
approaches in terms of both factual precision and recall.


---

**[185. [2405.00253] CodeHalu: Investigating Code Hallucinations in LLMs via Execution-based
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

**[186. [2501.08200] CWEval: Outcome-driven Evaluation on Functionality and Security of LLM
  Code Generation](https://arxiv.org/pdf/2501.08200.pdf)** (2025-01-15)

*Jinjun Peng, Leyi Cui, Kele Huang, Junfeng Yang, Baishakhi Ray*

  Large Language Models (LLMs) have significantly aided developers by
generating or assisting in code writing, enhancing productivity across various
tasks. While identifying incorrect code is often straightforward, detecting
vulnerabilities in functionally correct code is more challenging, especially
for developers with limited security knowledge, which poses considerable
security risks of using LLM-generated code and underscores the need for robust
evaluation benchmarks that assess both functional correctness and security.
Current benchmarks like CyberSecEval and SecurityEval attempt to solve it but
are hindered by unclear and impractical specifications, failing to assess both
functionality and security accurately. To tackle these deficiencies, we
introduce CWEval, a novel outcome-driven evaluation framework designed to
enhance the evaluation of secure code generation by LLMs. This framework not
only assesses code functionality but also its security simultaneously with
high-quality task specifications and outcome-driven test oracles which provides
high accuracy. Coupled with CWEval-bench, a multilingual, security-critical
coding benchmark, CWEval provides a rigorous empirical security evaluation on
LLM-generated code, overcoming previous benchmarks' shortcomings. Through our
evaluations, CWEval reveals a notable portion of functional but insecure code
produced by LLMs, and shows a serious inaccuracy of previous evaluations,
ultimately contributing significantly to the field of secure code generation.
We open-source our artifact at: https://github.com/Co1lin/CWEval .


---

**[187. [2405.01593] Large Language Model Agent for Fake News Detection](https://arxiv.org/pdf/2405.01593.pdf)** (2024-05-06)

*Xinyi Li, Yongfeng Zhang, Edward C. Malthouse*

  In the current digital era, the rapid spread of misinformation on online
platforms presents significant challenges to societal well-being, public trust,
and democratic processes, influencing critical decision making and public
opinion. To address these challenges, there is a growing need for automated
fake news detection mechanisms. Pre-trained large language models (LLMs) have
demonstrated exceptional capabilities across various natural language
processing (NLP) tasks, prompting exploration into their potential for
verifying news claims. Instead of employing LLMs in a non-agentic way, where
LLMs generate responses based on direct prompts in a single shot, our work
introduces FactAgent, an agentic approach of utilizing LLMs for fake news
detection. FactAgent enables LLMs to emulate human expert behavior in verifying
news claims without any model training, following a structured workflow. This
workflow breaks down the complex task of news veracity checking into multiple
sub-steps, where LLMs complete simple tasks using their internal knowledge or
external tools. At the final step of the workflow, LLMs integrate all findings
throughout the workflow to determine the news claim's veracity. Compared to
manual human verification, FactAgent offers enhanced efficiency. Experimental
studies demonstrate the effectiveness of FactAgent in verifying claims without
the need for any training process. Moreover, FactAgent provides transparent
explanations at each step of the workflow and during final decision-making,
offering insights into the reasoning process of fake news detection for end
users. FactAgent is highly adaptable, allowing for straightforward updates to
its tools that LLMs can leverage within the workflow, as well as updates to the
workflow itself using domain knowledge. This adaptability enables FactAgent's
application to news verification across various domains.


---

**[188. [2402.05130] LB-KBQA: Large-language-model and BERT based Knowledge-Based Question
  and Answering System](https://arxiv.org/pdf/2402.05130.pdf)** (2024-02-12)

*Yan Zhao, Zhongyun Li, Yushan Pan, Jiaxing Wang, Yihong Wang*

  Generative Artificial Intelligence (AI), because of its emergent abilities,
has empowered various fields, one typical of which is large language models
(LLMs). One of the typical application fields of Generative AI is large
language models (LLMs), and the natural language understanding capability of
LLM is dramatically improved when compared with conventional AI-based methods.
The natural language understanding capability has always been a barrier to the
intent recognition performance of the Knowledge-Based-Question-and-Answer
(KBQA) system, which arises from linguistic diversity and the newly appeared
intent. Conventional AI-based methods for intent recognition can be divided
into semantic parsing-based and model-based approaches. However, both of the
methods suffer from limited resources in intent recognition. To address this
issue, we propose a novel KBQA system based on a Large Language Model(LLM) and
BERT (LB-KBQA). With the help of generative AI, our proposed method could
detect newly appeared intent and acquire new knowledge. In experiments on
financial domain question answering, our model has demonstrated superior
effectiveness.


---

**[189. [2411.18948] RevPRAG: Revealing Poisoning Attacks in Retrieval-Augmented Generation
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

**[190. [2502.19669] Investigating Neurons and Heads in Transformer-based LLMs for
  Typographical Errors](https://arxiv.org/pdf/2502.19669.pdf)** (2025-02-28)

*Kohei Tsuji, Tatsuya Hiraoka, Yuchang Cheng, Eiji Aramaki, Tomoya Iwakura*

  This paper investigates how LLMs encode inputs with typos. We hypothesize
that specific neurons and attention heads recognize typos and fix them
internally using local and global contexts. We introduce a method to identify
typo neurons and typo heads that work actively when inputs contain typos. Our
experimental results suggest the following: 1) LLMs can fix typos with local
contexts when the typo neurons in either the early or late layers are
activated, even if those in the other are not. 2) Typo neurons in the middle
layers are responsible for the core of typo-fixing with global contexts. 3)
Typo heads fix typos by widely considering the context not focusing on specific
tokens. 4) Typo neurons and typo heads work not only for typo-fixing but also
for understanding general contexts.


---

**[191. [2411.17309] PIM-AI: A Novel Architecture for High-Efficiency LLM Inference](https://arxiv.org/pdf/2411.17309.pdf)** (2024-12-02)

*Cristobal Ortega, Yann Falevoz, Renaud Ayrignac*

  Large Language Models (LLMs) have become essential in a variety of
applications due to their advanced language understanding and generation
capabilities. However, their computational and memory requirements pose
significant challenges to traditional hardware architectures.
Processing-in-Memory (PIM), which integrates computational units directly into
memory chips, offers several advantages for LLM inference, including reduced
data transfer bottlenecks and improved power efficiency.
  This paper introduces PIM-AI, a novel DDR5/LPDDR5 PIM architecture designed
for LLM inference without modifying the memory controller or DDR/LPDDR memory
PHY. We have developed a simulator to evaluate the performance of PIM-AI in
various scenarios and demonstrate its significant advantages over conventional
architectures. In cloud-based scenarios, PIM-AI reduces the 3-year TCO per
queries-per-second by up to 6.94x compared to state-of-the-art GPUs, depending
on the LLM model used. In mobile scenarios, PIM-AI achieves a 10- to 20-fold
reduction in energy per token compared to state-of-the-art mobile SoCs,
resulting in 25 to 45~\% more queries per second and 6.9x to 13.4x less energy
per query, extending battery life and enabling more inferences per charge.
These results highlight PIM-AI's potential to revolutionize LLM deployments,
making them more efficient, scalable, and sustainable.


---

**[192. [2504.07069] HalluciNot: Hallucination Detection Through Context and Common Knowledge
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

**[193. [2410.04698] MathHay: An Automated Benchmark for Long-Context Mathematical Reasoning
  in LLMs](https://arxiv.org/pdf/2410.04698.pdf)** (2024-10-08)

*Lei Wang, Shan Dong, Yuhui Xu, Hanze Dong, Yalu Wang, Amrita Saha, Ee-Peng Lim, Caiming Xiong, Doyen Sahoo*

  Recent large language models (LLMs) have demonstrated versatile capabilities
in long-context scenarios. Although some recent benchmarks have been developed
to evaluate the long-context capabilities of LLMs, there is a lack of
benchmarks evaluating the mathematical reasoning abilities of LLMs over long
contexts, which is crucial for LLMs' application in real-world scenarios. In
this paper, we introduce MathHay, an automated benchmark designed to assess the
long-context mathematical reasoning capabilities of LLMs. Unlike previous
benchmarks like Needle in a Haystack, which focus primarily on information
retrieval within long texts, MathHay demands models with both
information-seeking and complex mathematical reasoning abilities. We conduct
extensive experiments on MathHay to assess the long-context mathematical
reasoning abilities of eight top-performing LLMs. Even the best-performing
model, Gemini-1.5-Pro-002, still struggles with mathematical reasoning over
long contexts, achieving only 51.26% accuracy at 128K tokens. This highlights
the significant room for improvement on the MathHay benchmark.


---

**[194. [2404.11262] Sampling-based Pseudo-Likelihood for Membership Inference Attacks](https://arxiv.org/pdf/2404.11262.pdf)** (2024-04-18)

*Masahiro Kaneko, Youmi Ma, Yuki Wata, Naoaki Okazaki*

  Large Language Models (LLMs) are trained on large-scale web data, which makes
it difficult to grasp the contribution of each text. This poses the risk of
leaking inappropriate data such as benchmarks, personal information, and
copyrighted texts in the training data. Membership Inference Attacks (MIA),
which determine whether a given text is included in the model's training data,
have been attracting attention. Previous studies of MIAs revealed that
likelihood-based classification is effective for detecting leaks in LLMs.
However, the existing methods cannot be applied to some proprietary models like
ChatGPT or Claude 3 because the likelihood is unavailable to the user. In this
study, we propose a Sampling-based Pseudo-Likelihood (\textbf{SPL}) method for
MIA (\textbf{SaMIA}) that calculates SPL using only the text generated by an
LLM to detect leaks. The SaMIA treats the target text as the reference text and
multiple outputs from the LLM as text samples, calculates the degree of
$n$-gram match as SPL, and determines the membership of the text in the
training data. Even without likelihoods, SaMIA performed on par with existing
likelihood-based methods.


---

**[195. [2403.18802] Long-form factuality in large language models](https://arxiv.org/pdf/2403.18802.pdf)** (2024-11-08)

*Jerry Wei, Chengrun Yang, Xinying Song, Yifeng Lu, Nathan Hu, Jie Huang, Dustin Tran, Daiyi Peng, Ruibo Liu, Da Huang, Cosmo Du, Quoc V. Le*

  Large language models (LLMs) often generate content that contains factual
errors when responding to fact-seeking prompts on open-ended topics. To
benchmark a model's long-form factuality in open domains, we first use GPT-4 to
generate LongFact, a prompt set comprising thousands of questions spanning 38
topics. We then propose that LLM agents can be used as automated evaluators for
long-form factuality through a method which we call Search-Augmented Factuality
Evaluator (SAFE). SAFE utilizes an LLM to break down a long-form response into
a set of individual facts and to evaluate the accuracy of each fact using a
multi-step reasoning process comprising sending search queries to Google Search
and determining whether a fact is supported by the search results. Furthermore,
we propose extending F1 score as an aggregated metric for long-form factuality.
To do so, we balance the percentage of supported facts in a response
(precision) with the percentage of provided facts relative to a hyperparameter
representing a user's preferred response length (recall).
  Empirically, we demonstrate that LLM agents can outperform crowdsourced human
annotators - on a set of ~16k individual facts, SAFE agrees with crowdsourced
human annotators 72% of the time, and on a random subset of 100 disagreement
cases, SAFE wins 76% of the time. At the same time, SAFE is more than 20 times
cheaper than human annotators. We also benchmark thirteen language models on
LongFact across four model families (Gemini, GPT, Claude, and PaLM-2), finding
that larger language models generally achieve better long-form factuality.
LongFact, SAFE, and all experimental code are available at
https://github.com/google-deepmind/long-form-factuality.


---

**[196. [2502.19320] Shh, don't say that! Domain Certification in LLMs](https://arxiv.org/pdf/2502.19320.pdf)** (2025-03-10)

*Cornelius Emde, Alasdair Paren, Preetham Arvind, Maxime Kayser, Tom Rainforth, Thomas Lukasiewicz, Bernard Ghanem, Philip H. S. Torr, Adel Bibi*

  Large language models (LLMs) are often deployed to perform constrained tasks,
with narrow domains. For example, customer support bots can be built on top of
LLMs, relying on their broad language understanding and capabilities to enhance
performance. However, these LLMs are adversarially susceptible, potentially
generating outputs outside the intended domain. To formalize, assess, and
mitigate this risk, we introduce domain certification; a guarantee that
accurately characterizes the out-of-domain behavior of language models. We then
propose a simple yet effective approach, which we call VALID that provides
adversarial bounds as a certificate. Finally, we evaluate our method across a
diverse set of datasets, demonstrating that it yields meaningful certificates,
which bound the probability of out-of-domain samples tightly with minimum
penalty to refusal behavior.


---
