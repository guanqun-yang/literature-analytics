**[1. [2402.00707] Non-Exchangeable Conformal Language Generation with Nearest Neighbors](https://arxiv.org/pdf/2402.00707.pdf)** (2024-02-02)

*Dennis Ulmer, Chrysoula Zerva, André F. T. Martins*

  Quantifying uncertainty in automatically generated text is important for
letting humans check potential hallucinations and making systems more reliable.
Conformal prediction is an attractive framework to provide predictions imbued
with statistical guarantees, however, its application to text generation is
challenging since any i.i.d. assumptions are not realistic. In this paper, we
bridge this gap by leveraging recent results on non-exchangeable conformal
prediction, which still ensures bounds on coverage. The result,
non-exchangeable conformal nucleus sampling, is a novel extension of the
conformal prediction framework to generation based on nearest neighbors. Our
method can be used post-hoc for an arbitrary model without extra training and
supplies token-level, calibrated prediction sets equipped with statistical
guarantees. Experiments in machine translation and language modeling show
encouraging results in generation quality. By also producing tighter prediction
sets with good coverage, we thus give a more theoretically principled way to
perform sampling with conformal guarantees.


---

**[2. [2305.06599] Structured Chain-of-Thought Prompting for Code Generation](https://arxiv.org/pdf/2305.06599.pdf)** (2023-09-08)

*Jia Li, Ge Li, Yongmin Li, Zhi Jin*

  Large Language Models (LLMs) (e.g., ChatGPT) have shown impressive
performance in code generation. LLMs take prompts as inputs, and
Chain-of-Thought (CoT) prompting is the state-of-the-art prompting technique.
CoT prompting asks LLMs first to generate CoTs (i.e., intermediate natural
language reasoning steps) and then output the code. However, CoT prompting is
designed for natural language generation and has low accuracy in code
generation.
  In this paper, we propose Structured CoTs (SCoTs) and present a novel
prompting technique for code generation, named SCoT prompting. Our motivation
is source code contains rich structural information and any code can be
composed of three program structures (i.e., sequence, branch, and loop
structures). Intuitively, structured intermediate reasoning steps make for
structured source code. Thus, we ask LLMs to use program structures to build
CoTs, obtaining SCoTs. Then, LLMs generate the final code based on SCoTs.
Compared to CoT prompting, SCoT prompting explicitly constrains LLMs to think
about how to solve requirements from the view of source code and further the
performance of LLMs in code generation. We apply SCoT prompting to two LLMs
(i.e., ChatGPT and Codex) and evaluate it on three benchmarks (i.e., HumanEval,
MBPP, and MBCPP). (1) SCoT prompting outperforms the state-of-the-art baseline
- CoT prompting by up to 13.79% in Pass@1. (2) Human evaluation shows human
developers prefer programs from SCoT prompting. (3) SCoT prompting is robust to
examples and achieves substantial improvements.


---

**[3. [2411.01696] Conformal Risk Minimization with Variance Reduction](https://arxiv.org/pdf/2411.01696.pdf)** (2025-02-11)

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

**[4. [2503.08340] Online Conformal Compression for Zero-Delay Communication with
  Distortion Guarantees](https://arxiv.org/pdf/2503.08340.pdf)** (2025-03-12)

*Unnikrishnan Kunnath Ganesan, Giuseppe Durisi, Matteo Zecchin, Petar Popovski, Osvaldo Simeone*

  We investigate a lossy source compression problem in which both the encoder
and decoder are equipped with a pre-trained sequence predictor. We propose an
online lossy compression scheme that, under a 0-1 loss distortion function,
ensures a deterministic, per-sequence upper bound on the distortion (outage)
level for any time instant. The outage guarantees apply irrespective of any
assumption on the distribution of the sequences to be encoded or on the quality
of the predictor at the encoder and decoder. The proposed method, referred to
as online conformal compression (OCC), is built upon online conformal
prediction--a novel method for constructing confidence intervals for arbitrary
predictors. Numerical results show that OCC achieves a compression rate
comparable to that of an idealized scheme in which the encoder, with hindsight,
selects the optimal subset of symbols to describe to the decoder, while
satisfying the overall outage constraint.


---

**[5. [2407.12504] Case2Code: Scalable Synthetic Data for Code Generation](https://arxiv.org/pdf/2407.12504.pdf)** (2025-02-11)

*Yunfan Shao, Linyang Li, Yichuan Ma, Peiji Li, Demin Song, Qinyuan Cheng, Shimin Li, Xiaonan Li, Pengyu Wang, Qipeng Guo, Hang Yan, Xipeng Qiu, Xuanjing Huang, Dahua Lin*

  Large Language Models (LLMs) have shown outstanding breakthroughs in code
generation. Recent work improves code LLMs by training on synthetic data
generated by some powerful LLMs, which can be challenging to scale due to the
dependence on a teacher model and high generation costs. In this paper, we
focus on synthesizing code data at scale and propose a \textbf{Case2Code} task
by exploiting the expressiveness and correctness of programs.
\textbf{Case2Code} is an inductive inference task that aims to infer underlying
code implementations by observing input-output examples or program behaviors,
By incorporating LLMs to generate program inputs, and executing the program
with these inputs to obtain the program outputs, we can synthesize diverse and
high-quality \textbf{Case2Code} data at scale for training and evaluating code
LLMs. Experimental results show that case-to-code induction is challenging for
current representative LLMs if they are untrained. Models trained with
\textbf{Case2Code} improve performance not only on distribution case-to-code
induction but also on various coding-generation tasks, demonstrating the great
potential of large-scale synthetic data and inductive learning.


---

**[6. [2410.06494] Conformal Prediction: A Data Perspective](https://arxiv.org/pdf/2410.06494.pdf)** (2025-03-12)

*Xiaofan Zhou, Baiting Chen, Yu Gui, Lu Cheng*

  Conformal prediction (CP), a distribution-free uncertainty quantification
(UQ) framework, reliably provides valid predictive inference for black-box
models. CP constructs prediction sets that contain the true output with a
specified probability. However, modern data science diverse modalities, along
with increasing data and model complexity, challenge traditional CP methods.
These developments have spurred novel approaches to address evolving scenarios.
This survey reviews the foundational concepts of CP and recent advancements
from a data-centric perspective, including applications to structured,
unstructured, and dynamic data. We also discuss the challenges and
opportunities CP faces in large-scale data and models.


---

**[7. [2211.16823] Algebraic-geometric codes with many automorphisms arising from Galois
  points](https://arxiv.org/pdf/2211.16823.pdf)** (2022-12-01)

*Satoru Fukasawa*

  A method of constructing algebraic-geometric codes with many automorphisms
arising from Galois points for algebraic curves is presented.


---

**[8. [2405.02140] An Information Theoretic Perspective on Conformal Prediction](https://arxiv.org/pdf/2405.02140.pdf)** (2025-02-18)

*Alvaro H. C. Correia, Fabio Valerio Massoli, Christos Louizos, Arash Behboodi*

  Conformal Prediction (CP) is a distribution-free uncertainty estimation
framework that constructs prediction sets guaranteed to contain the true answer
with a user-specified probability. Intuitively, the size of the prediction set
encodes a general notion of uncertainty, with larger sets associated with
higher degrees of uncertainty. In this work, we leverage information theory to
connect conformal prediction to other notions of uncertainty. More precisely,
we prove three different ways to upper bound the intrinsic uncertainty, as
described by the conditional entropy of the target variable given the inputs,
by combining CP with information theoretical inequalities. Moreover, we
demonstrate two direct and useful applications of such connection between
conformal prediction and information theory: (i) more principled and effective
conformal training objectives that generalize previous approaches and enable
end-to-end training of machine learning models from scratch, and (ii) a natural
mechanism to incorporate side information into conformal prediction. We
empirically validate both applications in centralized and federated learning
settings, showing our theoretical results translate to lower inefficiency
(average prediction set size) for popular CP methods.


---

**[9. [2104.01885] Conformal testing in a binary model situation](https://arxiv.org/pdf/2104.01885.pdf)** (2021-04-06)

*Vladimir Vovk*

  Conformal testing is a way of testing the IID assumption based on conformal
prediction. The topic of this note is computational evaluation of the
performance of conformal testing in a model situation in which IID binary
observations generated from a Bernoulli distribution are followed by IID binary
observations generated from another Bernoulli distribution, with the parameters
of the distributions and changepoint unknown. Existing conformal test
martingales can be used for this task and work well in simple cases, but their
efficiency can be improved greatly.


---

**[10. [2502.18905] Automated Code Generation and Validation for Software Components of
  Microcontrollers](https://arxiv.org/pdf/2502.18905.pdf)** (2025-02-27)

*Sebastian Haug, Christoph Böhm, Daniel Mayer*

  This paper proposes a method for generating software components for embedded
systems, integrating seamlessly into existing implementations without developer
intervention. We demonstrate this by automatically generating hardware
abstraction layer (HAL) code for GPIO operations on the STM32F407
microcontroller. Using Abstract Syntax Trees (AST) for code analysis and
Retrieval-Augmented Generation (RAG) for component generation, our approach
enables autonomous code completion for embedded applications.


---

**[11. [2305.00418] Using Large Language Models to Generate JUnit Tests: An Empirical Study](https://arxiv.org/pdf/2305.00418.pdf)** (2024-08-29)

*Mohammed Latif Siddiq, Joanna C. S. Santos, Ridwanul Hasan Tanvir, Noshin Ulfat, Fahmid Al Rifat, Vinicius Carvalho Lopes*

  A code generation model generates code by taking a prompt from a code
comment, existing code, or a combination of both. Although code generation
models (e.g., GitHub Copilot) are increasingly being adopted in practice, it is
unclear whether they can successfully be used for unit test generation without
fine-tuning for a strongly typed language like Java. To fill this gap, we
investigated how well three models (Codex, GPT-3.5-Turbo, and StarCoder) can
generate unit tests. We used two benchmarks (HumanEval and Evosuite SF110) to
investigate the effect of context generation on the unit test generation
process. We evaluated the models based on compilation rates, test correctness,
test coverage, and test smells. We found that the Codex model achieved above
80% coverage for the HumanEval dataset, but no model had more than 2% coverage
for the EvoSuite SF110 benchmark. The generated tests also suffered from test
smells, such as Duplicated Asserts and Empty Tests.


---

**[12. [2501.17584] GLLM: Self-Corrective G-Code Generation using Large Language Models with
  User Feedback](https://arxiv.org/pdf/2501.17584.pdf)** (2025-01-30)

*Mohamed Abdelaal, Samuel Lokadjaja, Gilbert Engert*

  This paper introduces GLLM, an innovative tool that leverages Large Language
Models (LLMs) to automatically generate G-code from natural language
instructions for Computer Numerical Control (CNC) machining. GLLM addresses the
challenges of manual G-code writing by bridging the gap between human-readable
task descriptions and machine-executable code. The system incorporates a
fine-tuned StarCoder-3B model, enhanced with domain-specific training data and
a Retrieval-Augmented Generation (RAG) mechanism. GLLM employs advanced
prompting strategies and a novel self-corrective code generation approach to
ensure both syntactic and semantic correctness of the generated G-code. The
architecture includes robust validation mechanisms, including syntax checks,
G-code-specific verifications, and functional correctness evaluations using
Hausdorff distance. By combining these techniques, GLLM aims to democratize CNC
programming, making it more accessible to users without extensive programming
experience while maintaining high accuracy and reliability in G-code
generation.


---

**[13. [2007.02609] Relevance Transformer: Generating Concise Code Snippets with Relevance
  Feedback](https://arxiv.org/pdf/2007.02609.pdf)** (2020-12-09)

*Carlos Gemmell, Federico Rossetto, Jeffrey Dalton*

  Tools capable of automatic code generation have the potential to augment
programmer's capabilities. While straightforward code retrieval is incorporated
into many IDEs, an emerging area is explicit code generation. Code generation
is currently approached as a Machine Translation task, with Recurrent Neural
Network (RNN) based encoder-decoder architectures trained on code-description
pairs. In this work we introduce and study modern Transformer architectures for
this task. We further propose a new model called the Relevance Transformer that
incorporates external knowledge using pseudo-relevance feedback. The Relevance
Transformer biases the decoding process to be similar to existing retrieved
code while enforcing diversity. We perform experiments on multiple standard
benchmark datasets for code generation including Django, Hearthstone, and
CoNaLa. The results show improvements over state-of-the-art methods based on
BLEU evaluation. The Relevance Transformer model shows the potential of
Transformer-based architectures for code generation and introduces a method of
incorporating pseudo-relevance feedback during inference.


---

**[14. [2411.10599] Generating Energy-efficient code with LLMs](https://arxiv.org/pdf/2411.10599.pdf)** (2024-11-19)

*Tom Cappendijk, Pepijn de Reus, Ana Oprescu*

  The increasing electricity demands of personal computers, communication
networks, and data centers contribute to higher atmospheric greenhouse gas
emissions, which in turn lead to global warming and climate change. Therefore
the energy consumption of code must be minimized. Code can be generated by
large language models. We look at the influence of prompt modification on the
energy consumption of the code generated. We use three different Python code
problems of varying difficulty levels. Prompt modification is done by adding
the sentence ``Give me an energy-optimized solution for this problem'' or by
using two Python coding best practices. The large language models used are
CodeLlama-70b, CodeLlama-70b-Instruct, CodeLlama-70b-Python,
DeepSeek-Coder-33b-base, and DeepSeek-Coder-33b-instruct. We find a decrease in
energy consumption for a specific combination of prompt optimization, LLM, and
Python code problem. However, no single optimization prompt consistently
decreases energy consumption for the same LLM across the different Python code
problems.


---

**[15. [2201.08810] GAP-Gen: Guided Automatic Python Code Generation](https://arxiv.org/pdf/2201.08810.pdf)** (2023-05-11)

*Junchen Zhao, Yurun Song, Junlin Wang, Ian G. Harris*

  Automatic code generation from natural language descriptions can be highly
beneficial during the process of software development. In this work, we propose
GAP-Gen, a Guided Automatic Python Code Generation method based on Python
syntactic constraints and semantic constraints. We first introduce Python
syntactic constraints in the form of Syntax-Flow, which is a simplified version
of Abstract Syntax Tree (AST) reducing the size and high complexity of Abstract
Syntax Tree but maintaining crucial syntactic information of Python code. In
addition to Syntax-Flow, we introduce Variable-Flow which abstracts variable
and function names consistently through out the code. In our work, rather than
pretraining, we focus on modifying the finetuning process which reduces
computational requirements but retains high generation performance on automatic
Python code generation task. GAP-Gen fine-tunes the transformer based language
models T5 and CodeT5 using the Code-to-Docstring datasets CodeSearchNet,
CodeSearchNet AdvTest and Code-Docstring Corpus from EdinburghNLP. Our
experiments show that GAP-Gen achieves better results on automatic Python code
generation task than previous works.


---

**[16. [2407.04831] Code Hallucination](https://arxiv.org/pdf/2407.04831.pdf)** (2024-08-09)

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

**[17. [2502.12601] COPU: Conformal Prediction for Uncertainty Quantification in Natural
  Language Generation](https://arxiv.org/pdf/2502.12601.pdf)** (2025-04-09)

*Sean Wang, Yicheng Jiang, Yuxin Tang, Lu Cheng, Hanjie Chen*

  Uncertainty Quantification (UQ) for Natural Language Generation (NLG) is
crucial for assessing the performance of Large Language Models (LLMs), as it
reveals confidence in predictions, identifies failure modes, and gauges output
reliability. Conformal Prediction (CP), a model-agnostic method that generates
prediction sets with a specified error rate, has been adopted for UQ in
classification tasks, where the size of the prediction set indicates the
model's uncertainty. However, when adapting CP to NLG, the sampling-based
method for generating candidate outputs cannot guarantee the inclusion of the
ground truth, limiting its applicability across a wide range of error rates. To
address this, we propose \ourmethod, a method that explicitly adds the ground
truth to the candidate outputs and uses logit scores to measure nonconformity.
Our experiments with six LLMs on four NLG tasks show that \ourmethod
outperforms baseline methods in calibrating error rates and empirical cover
rates, offering accurate UQ across a wide range of user-specified error rates.


---

**[18. [2407.17377] Entropy Reweighted Conformal Classification](https://arxiv.org/pdf/2407.17377.pdf)** (2024-07-25)

*Rui Luo, Nicolo Colombo*

  Conformal Prediction (CP) is a powerful framework for constructing prediction
sets with guaranteed coverage. However, recent studies have shown that
integrating confidence calibration with CP can lead to a degradation in
efficiency. In this paper, We propose an adaptive approach that considers the
classifier's uncertainty and employs entropy-based reweighting to enhance the
efficiency of prediction sets for conformal classification. Our experimental
results demonstrate that this method significantly improves efficiency.


---

**[19. [2401.11974] Cross-Validation Conformal Risk Control](https://arxiv.org/pdf/2401.11974.pdf)** (2024-05-02)

*Shitz  Kfir M. Cohen, Shitz  Sangwoo Park, Shitz  Osvaldo Simeone, Shitz  Shlomo Shamai*

  Conformal risk control (CRC) is a recently proposed technique that applies
post-hoc to a conventional point predictor to provide calibration guarantees.
Generalizing conformal prediction (CP), with CRC, calibration is ensured for a
set predictor that is extracted from the point predictor to control a risk
function such as the probability of miscoverage or the false negative rate. The
original CRC requires the available data set to be split between training and
validation data sets. This can be problematic when data availability is
limited, resulting in inefficient set predictors. In this paper, a novel CRC
method is introduced that is based on cross-validation, rather than on
validation as the original CRC. The proposed cross-validation CRC (CV-CRC)
extends a version of the jackknife-minmax from CP to CRC, allowing for the
control of a broader range of risk functions. CV-CRC is proved to offer
theoretical guarantees on the average risk of the set predictor. Furthermore,
numerical experiments show that CV-CRC can reduce the average set size with
respect to CRC when the available data are limited.


---

**[20. [2406.06818] Conformal Prediction for Class-wise Coverage via Augmented Label Rank
  Calibration](https://arxiv.org/pdf/2406.06818.pdf)** (2024-12-09)

*Yuanjie Shi, Subhankar Ghosh, Taha Belkhouja, Janardhan Rao Doppa, Yan Yan*

  Conformal prediction (CP) is an emerging uncertainty quantification framework
that allows us to construct a prediction set to cover the true label with a
pre-specified marginal or conditional probability. Although the valid coverage
guarantee has been extensively studied for classification problems, CP often
produces large prediction sets which may not be practically useful. This issue
is exacerbated for the setting of class-conditional coverage on imbalanced
classification tasks with many and/or imbalanced classes. This paper proposes
the Rank Calibrated Class-conditional CP (RC3P) algorithm to reduce the
prediction set sizes to achieve class-conditional coverage, where the valid
coverage holds for each class. In contrast to the standard class-conditional CP
(CCP) method that uniformly thresholds the class-wise conformity score for each
class, the augmented label rank calibration step allows RC3P to selectively
iterate this class-wise thresholding subroutine only for a subset of classes
whose class-wise top-k error is small. We prove that agnostic to the classifier
and data distribution, RC3P achieves class-wise coverage. We also show that
RC3P reduces the size of prediction sets compared to the CCP method.
Comprehensive experiments on multiple real-world datasets demonstrate that RC3P
achieves class-wise coverage and 26.25% reduction in prediction set sizes on
average.


---

**[21. [2205.11708] A Proof-Generating C Code Generator for ACL2 Based on a Shallow
  Embedding of C in ACL2](https://arxiv.org/pdf/2205.11708.pdf)** (2022-05-25)

*Kestrel Institute  Alessandro Coglio*

  This paper describes a C code generator for ACL2 that recognizes ACL2
representations of C constructs, according to a shallow embedding of C in ACL2,
and translates those representations to the represented C constructs. The code
generator also generates ACL2 theorems asserting the correctness of the C code
with respect to the ACL2 code. The code generator currently supports a limited
but growing subset of C that already suffices for some interesting programs.
This paper also offers a general perspective on language embedding and code
generation.


---

**[22. [2502.06631] Conformal Predictions for Human Action Recognition with Vision-Language
  Models](https://arxiv.org/pdf/2502.06631.pdf)** (2025-02-11)

*Bary Tim, Fuchs Clément, Macq Benoît*

  Human-In-The-Loop (HITL) frameworks are integral to many real-world computer
vision systems, enabling human operators to make informed decisions with AI
assistance. Conformal Predictions (CP), which provide label sets with rigorous
guarantees on ground truth inclusion probabilities, have recently gained
traction as a valuable tool in HITL settings. One key application area is video
surveillance, closely associated with Human Action Recognition (HAR). This
study explores the application of CP on top of state-of-the-art HAR methods
that utilize extensively pre-trained Vision-Language Models (VLMs). Our
findings reveal that CP can significantly reduce the average number of
candidate classes without modifying the underlying VLM. However, these
reductions often result in distributions with long tails. To address this, we
introduce a method based on tuning the temperature parameter of the VLMs to
minimize these tails without requiring additional calibration data. Our code is
made available on GitHub at the address https://github.com/tbary/CP4VLM.


---

**[23. [2402.13720] Ouroboros: Generating Longer Drafts Phrase by Phrase for Faster
  Speculative Decoding](https://arxiv.org/pdf/2402.13720.pdf)** (2024-10-16)

*Weilin Zhao, Yuxiang Huang, Xu Han, Wang Xu, Chaojun Xiao, Xinrong Zhang, Yewei Fang, Kaihuo Zhang, Zhiyuan Liu, Maosong Sun*

  Speculative decoding is a widely used method that accelerates the generation
process of large language models (LLMs) with no compromise in model
performance. It achieves this goal by using an existing smaller model for
drafting and then employing the target LLM to verify the draft in a low-cost
parallel manner. Under such a drafting-verification framework, drafting
efficiency has become a bottleneck in the final speedup of speculative
decoding. Therefore, generating longer drafts at less cost can lead to better
decoding speedup. To achieve this, we introduce Ouroboros, which can generate
draft phrases to parallelize the drafting process and meanwhile lengthen drafts
in a training-free manner. The experimental results on various typical text
generation tasks show that Ouroboros can achieve speedups of up to $2.8\times$
over speculative decoding and $3.9\times$ over vanilla decoding, without
fine-tuning draft and target models. The source code of Ouroboros is available
at https://github.com/thunlp/Ouroboros.


---

**[24. [2503.10512] Conformal Prediction Sets for Deep Generative Models via Reduction to
  Conformal Regression](https://arxiv.org/pdf/2503.10512.pdf)** (2025-03-14)

*Hooman Shahrokhi, Devjeet Raj Roy, Yan Yan, Venera Arnaoudova, Janaradhan Rao Doppa*

  We consider the problem of generating valid and small prediction sets by
sampling outputs (e.g., software code and natural language text) from a
black-box deep generative model for a given input (e.g., textual prompt). The
validity of a prediction set is determined by a user-defined binary
admissibility function depending on the target application. For example,
requiring at least one program in the set to pass all test cases in code
generation application. To address this problem, we develop a simple and
effective conformal inference algorithm referred to as Generative Prediction
Sets (GPS). Given a set of calibration examples and black-box access to a deep
generative model, GPS can generate prediction sets with provable guarantees.
The key insight behind GPS is to exploit the inherent structure within the
distribution over the minimum number of samples needed to obtain an admissible
output to develop a simple conformal regression approach over the minimum
number of samples. Experiments on multiple datasets for code and math word
problems using different large language models demonstrate the efficacy of GPS
over state-of-the-art methods.


---

**[25. [2408.08333] CodeMirage: Hallucinations in Code Generated by Large Language Models](https://arxiv.org/pdf/2408.08333.pdf)** (2024-08-19)

*Vibhor Agarwal, Yulong Pei, Salwa Alamir, Xiaomo Liu*

  Large Language Models (LLMs) have shown promising potentials in program
generation and no-code automation. However, LLMs are prone to generate
hallucinations, i.e., they generate text which sounds plausible but is
incorrect. Although there has been a recent surge in research on LLM
hallucinations for text generation, similar hallucination phenomenon can happen
in code generation. Sometimes the generated code can have syntactical or
logical errors as well as more advanced issues like security vulnerabilities,
memory leaks, etc. Given the wide adaptation of LLMs to enhance efficiency in
code generation and development in general, it becomes imperative to
investigate hallucinations in code generation. To the best of our knowledge,
this is the first attempt at studying hallucinations in the code generated by
LLMs. We start by introducing the code hallucination definition and a
comprehensive taxonomy of code hallucination types. We propose the first
benchmark CodeMirage dataset for code hallucinations. The benchmark contains
1,137 GPT-3.5 generated hallucinated code snippets for Python programming
problems from two base datasets - HumanEval and MBPP. We then propose the
methodology for code hallucination detection and experiment with open source
LLMs such as CodeLLaMA as well as OpenAI's GPT-3.5 and GPT-4 models using
one-shot prompt. We find that GPT-4 performs the best on HumanEval dataset and
gives comparable results to the fine-tuned CodeBERT baseline on MBPP dataset.
Towards the end, we discuss various mitigation strategies for code
hallucinations and conclude our work.


---

**[26. [2308.08784] CodeCoT: Tackling Code Syntax Errors in CoT Reasoning for Code
  Generation](https://arxiv.org/pdf/2308.08784.pdf)** (2024-02-26)

*Dong Huang, Qingwen Bu, Yuhao Qing, Heming Cui*

  Chain-of-thought (CoT) has emerged as a groundbreaking tool in NLP, notably
for its efficacy in complex reasoning tasks, such as mathematical proofs.
However, its application in code generation faces a distinct challenge, i.e.,
although the code generated with CoT reasoning is logically correct, it faces
the problem of syntax error (e.g., invalid syntax error report) during code
execution, which causes the CoT result's pass@1 in HumanEval even lower than
the zero-shot result.
  In this paper, we present Code Chain-of-Thought (CodeCoT) that integrates CoT
with a self-examination process for code generation. CodeCoT begins with the
LLMs using CoT for initial code development to ensure the generated code
follows the correct logic flow. Then, CodeCoT will generate test cases to
validate whether the code has syntax errors during the execution. CodeCoT then
employs a self-examination phase, in which the generated code is executed
against these test cases in the local environment. If the local environment
raises error information (e.g., invalid syntax error), CodeCoT will iteratively
refine the code based on the feedback information. Within this loop, CodeCoT
can make sure their generated codes not only follow the logic flow of the code
description, but the syntax error will also be addressed with the
self-examination process. Our evaluation results reveal that CodeCoT improves
the effectiveness of code generation. For example, CodeCoT increases pass@1
from 75.6% to 79.3% for the HumanEval dataset.


---

**[27. [2403.19082] Enhancing Conformal Prediction Using E-Test Statistics](https://arxiv.org/pdf/2403.19082.pdf)** (2024-03-29)

*A. A. Balinsky, A. D. Balinsky*

  Conformal Prediction (CP) serves as a robust framework that quantifies
uncertainty in predictions made by Machine Learning (ML) models. Unlike
traditional point predictors, CP generates statistically valid prediction
regions, also known as prediction intervals, based on the assumption of data
exchangeability. Typically, the construction of conformal predictions hinges on
p-values. This paper, however, ventures down an alternative path, harnessing
the power of e-test statistics to augment the efficacy of conformal predictions
by introducing a BB-predictor (bounded from the below predictor).


---

**[28. [2504.02361] MG-Gen: Single Image to Motion Graphics Generation with Layer
  Decomposition](https://arxiv.org/pdf/2504.02361.pdf)** (2025-04-07)

*Takahiro Shirakawa, Tomoyuki Suzuki, Daichi Haraguchi*

  General image-to-video generation methods often produce suboptimal animations
that do not meet the requirements of animated graphics, as they lack active
text motion and exhibit object distortion. Also, code-based animation
generation methods typically require layer-structured vector data which are
often not readily available for motion graphic generation. To address these
challenges, we propose a novel framework named MG-Gen that reconstructs data in
vector format from a single raster image to extend the capabilities of
code-based methods to enable motion graphics generation from a raster image in
the framework of general image-to-video generation. MG-Gen first decomposes the
input image into layer-wise elements, reconstructs them as HTML format data and
then generates executable JavaScript code for the reconstructed HTML data. We
experimentally confirm that MG-Gen generates motion graphics while preserving
text readability and input consistency. These successful results indicate that
combining layer decomposition and animation code generation is an effective
strategy for motion graphics generation.


---

**[29. [2312.01524] Code Swarm: A Code Generation Tool Based on the Automatic Derivation of
  Transformation Rule Set](https://arxiv.org/pdf/2312.01524.pdf)** (2023-12-05)

*Hina Mahmood, Atif Aftab Jilani, Abdul Rauf*

  Automatic generation of software code from system design models remains an
actively explored research area for the past several years. A number of tools
are currently available to facilitate and automate the task of generating code
from software models. To the best of our knowledge, existing software tools
rely on an explicitly defined transformation rule set to perform the
model-to-code transformation process. In this paper, we introduce a novel tool
named Code Swarm, abbreviated as CodS, that automatically generates
implementation code from system design models by utilizing a swarm-based
approach. Specifically, CodS is capable of generating Java code from the class
and state models of the software system by making use of the previously solved
model-to-code transformation examples. Our tool enables the designers to
specify behavioural actions in the input models using the Action Specification
Language (ASL). We use an industrial case study of the Elevator Control System
(ECS) to perform the experimental validation of our tool. Our results indicate
that the code generated by CodS is correct and consistent with the input design
models. CodS performs the process of automatic code generation without taking
the explicit transformation rule set or languages metamodels information as
input, which distinguishes it from all the existing automatic code generation
tools.


---

**[30. [2409.15548] Beyond Conformal Predictors: Adaptive Conformal Inference with
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

**[31. [2402.14658] OpenCodeInterpreter: Integrating Code Generation with Execution and
  Refinement](https://arxiv.org/pdf/2402.14658.pdf)** (2025-01-08)

*Tianyu Zheng, Ge Zhang, Tianhao Shen, Xueling Liu, Bill Yuchen Lin, Jie Fu, Wenhu Chen, Xiang Yue*

  The introduction of large language models has significantly advanced code
generation. However, open-source models often lack the execution capabilities
and iterative refinement of advanced systems like the GPT-4 Code Interpreter.
To address this, we introduce OpenCodeInterpreter, a family of open-source code
systems designed for generating, executing, and iteratively refining code.
Supported by Code-Feedback, a dataset featuring 68K multi-turn interactions,
OpenCodeInterpreter integrates execution and human feedback for dynamic code
refinement. Our comprehensive evaluation of OpenCodeInterpreter across key
benchmarks such as HumanEval, MBPP, and their enhanced versions from EvalPlus
reveals its exceptional performance. Notably, OpenCodeInterpreter-33B achieves
an accuracy of 83.2 (76.4) on the average (and plus versions) of HumanEval and
MBPP, closely rivaling GPT-4's 84.2 (76.2) and further elevates to 91.6 (84.6)
with synthesized human feedback from GPT-4. OpenCodeInterpreter brings the gap
between open-source code generation models and proprietary systems like GPT-4
Code Interpreter.


---

**[32. [2407.20563] Pyramid Coder: Hierarchical Code Generator for Compositional Visual
  Question Answering](https://arxiv.org/pdf/2407.20563.pdf)** (2024-07-31)

*Ruoyue Shen, Nakamasa Inoue, Koichi Shinoda*

  Visual question answering (VQA) is the task of providing accurate answers to
natural language questions based on visual input. Programmatic VQA (PVQA)
models have been gaining attention recently. These use large language models
(LLMs) to formulate executable programs that address questions requiring
complex visual reasoning. However, there are challenges in enabling LLMs to
comprehend the usage of image processing modules and generate relevant code. To
overcome these challenges, this paper introduces PyramidCoder, a novel
prompting framework for PVQA models. PyramidCoder consists of three
hierarchical levels, each serving a distinct purpose: query rephrasing, code
generation, and answer aggregation. Notably, PyramidCoder utilizes a single
frozen LLM and pre-defined prompts at each level, eliminating the need for
additional training and ensuring flexibility across various LLM architectures.
Compared to the state-of-the-art PVQA model, our approach improves accuracy by
at least 0.5% on the GQA dataset, 1.4% on the VQAv2 dataset, and 2.9% on the
NLVR2 dataset.


---

**[33. [2502.17139] CodeSwift: Accelerating LLM Inference for Efficient Code Generation](https://arxiv.org/pdf/2502.17139.pdf)** (2025-02-25)

*Qianhui Zhao, Li Zhang, Fang Liu, Xiaoli Lian, Qiaoyuanhe Meng, Ziqian Jiao, Zetong Zhou, Borui Zhang, Runlin Guo, Jia Li*

  Code generation is a latency-sensitive task that demands high timeliness, but
the autoregressive decoding mechanism of Large Language Models (LLMs) leads to
poor inference efficiency. Existing LLM inference acceleration methods mainly
focus on standalone functions using only built-in components. Moreover, they
treat code like natural language sequences, ignoring its unique syntax and
semantic characteristics. As a result, the effectiveness of these approaches in
code generation tasks remains limited and fails to align with real-world
programming scenarios. To alleviate this issue, we propose CodeSwift, a simple
yet highly efficient inference acceleration approach specifically designed for
code generation, without comprising the quality of the output. CodeSwift
constructs a multi-source datastore, providing access to both general and
project-specific knowledge, facilitating the retrieval of high-quality draft
sequences. Moreover, CodeSwift reduces retrieval cost by controlling retrieval
timing, and enhances efficiency through parallel retrieval and a context- and
LLM preference-aware cache. Experimental results show that CodeSwift can reach
up to 2.53x and 2.54x speedup compared to autoregressive decoding in
repository-level and standalone code generation tasks, respectively,
outperforming state-of-the-art inference acceleration approaches by up to 88%.


---

**[34. [2406.04712] AICoderEval: Improving AI Domain Code Generation of Large Language
  Models](https://arxiv.org/pdf/2406.04712.pdf)** (2024-06-10)

*Yinghui Xia, Yuyan Chen, Tianyu Shi, Jun Wang, Jinsong Yang*

  Automated code generation is a pivotal capability of large language models
(LLMs). However, assessing this capability in real-world scenarios remains
challenging. Previous methods focus more on low-level code generation, such as
model loading, instead of generating high-level codes catering for real-world
tasks, such as image-to-text, text classification, in various domains.
Therefore, we construct AICoderEval, a dataset focused on real-world tasks in
various domains based on HuggingFace, PyTorch, and TensorFlow, along with
comprehensive metrics for evaluation and enhancing LLMs' task-specific code
generation capability. AICoderEval contains test cases and complete programs
for automated evaluation of these tasks, covering domains such as natural
language processing, computer vision, and multimodal learning. To facilitate
research in this area, we open-source the AICoderEval dataset at
\url{https://huggingface.co/datasets/vixuowis/AICoderEval}. After that, we
propose CoderGen, an agent-based framework, to help LLMs generate codes related
to real-world tasks on the constructed AICoderEval. Moreover, we train a more
powerful task-specific code generation model, named AICoder, which is refined
on llama-3 based on AICoderEval. Our experiments demonstrate the effectiveness
of CoderGen in improving LLMs' task-specific code generation capability (by
12.00\% on pass@1 for original model and 9.50\% on pass@1 for ReAct Agent).
AICoder also outperforms current code generation LLMs, indicating the great
quality of the AICoderEval benchmark.


---

**[35. [2502.05150] CodeSCM: Causal Analysis for Multi-Modal Code Generation](https://arxiv.org/pdf/2502.05150.pdf)** (2025-02-10)

*Mukur Gupta, Noopur Bhatt, Suman Jana*

  In this paper, we propose CodeSCM, a Structural Causal Model (SCM) for
analyzing multi-modal code generation using large language models (LLMs). By
applying interventions to CodeSCM, we measure the causal effects of different
prompt modalities, such as natural language, code, and input-output examples,
on the model. CodeSCM introduces latent mediator variables to separate the code
and natural language semantics of a multi-modal code generation prompt. Using
the principles of Causal Mediation Analysis on these mediators we quantify
direct effects representing the model's spurious leanings. We find that, in
addition to natural language instructions, input-output examples significantly
influence code generation.


---

**[36. [2106.12879] Decoding a class of maximum Hermitian rank metric codes](https://arxiv.org/pdf/2106.12879.pdf)** (2021-06-25)

*Wrya K. Kadir, Chunlei Li, Ferdinando Zullo*

  Maximum Hermitian rank metric codes were introduced by Schmidt in 2018 and in
this paper we propose both interpolation-based encoding and decoding algorithms
for this family of codes when the length and the minimum distance of the code
are both odd.


---

**[37. [2409.20424] World to Code: Multi-modal Data Generation via Self-Instructed
  Compositional Captioning and Filtering](https://arxiv.org/pdf/2409.20424.pdf)** (2024-10-01)

*Jiacong Wang, Bohong Wu, Haiyong Jiang, Xun Zhou, Xin Xiao, Haoyuan Guo, Jun Xiao*

  Recent advances in Vision-Language Models (VLMs) and the scarcity of
high-quality multi-modal alignment data have inspired numerous researches on
synthetic VLM data generation. The conventional norm in VLM data construction
uses a mixture of specialists in caption and OCR, or stronger VLM APIs and
expensive human annotation. In this paper, we present World to Code (W2C), a
meticulously curated multi-modal data construction pipeline that organizes the
final generation output into a Python code format. The pipeline leverages the
VLM itself to extract cross-modal information via different prompts and filter
the generated outputs again via a consistency filtering strategy. Experiments
have demonstrated the high quality of W2C by improving various existing visual
question answering and visual grounding benchmarks across different VLMs.
Further analysis also demonstrates that the new code parsing ability of VLMs
presents better cross-modal equivalence than the commonly used detail caption
ability. Our code is available at
https://github.com/foundation-multimodal-models/World2Code.


---

**[38. [2407.13945] FANTAstic SEquences and Where to Find Them: Faithful and Efficient API
  Call Generation through State-tracked Constrained Decoding and Reranking](https://arxiv.org/pdf/2407.13945.pdf)** (2024-07-22)

*Zhuoer Wang, Leonardo F. R. Ribeiro, Alexandros Papangelis, Rohan Mukherjee, Tzu-Yen Wang, Xinyan Zhao, Arijit Biswas, James Caverlee, Angeliki Metallinou*

  API call generation is the cornerstone of large language models' tool-using
ability that provides access to the larger world. However, existing supervised
and in-context learning approaches suffer from high training costs, poor data
efficiency, and generated API calls that can be unfaithful to the API
documentation and the user's request. To address these limitations, we propose
an output-side optimization approach called FANTASE. Two of the unique
contributions of FANTASE are its State-Tracked Constrained Decoding (SCD) and
Reranking components. SCD dynamically incorporates appropriate API constraints
in the form of Token Search Trie for efficient and guaranteed generation
faithfulness with respect to the API documentation. The Reranking component
efficiently brings in the supervised signal by leveraging a lightweight model
as the discriminator to rerank the beam-searched candidate generations of the
large language model. We demonstrate the superior performance of FANTASE in API
call generation accuracy, inference efficiency, and context efficiency with
DSTC8 and API Bank datasets.


---

**[39. [2405.00253] CodeHalu: Investigating Code Hallucinations in LLMs via Execution-based
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

**[40. [2409.07368] Demo: SGCode: A Flexible Prompt-Optimizing System for Secure Generation
  of Code](https://arxiv.org/pdf/2409.07368.pdf)** (2024-09-26)

*Khiem Ton, Nhi Nguyen, Mahmoud Nazzal, Abdallah Khreishah, Cristian Borcea, NhatHai Phan, Ruoming Jin, Issa Khalil, Yelong Shen*

  This paper introduces SGCode, a flexible prompt-optimizing system to generate
secure code with large language models (LLMs). SGCode integrates recent
prompt-optimization approaches with LLMs in a unified system accessible through
front-end and back-end APIs, enabling users to 1) generate secure code, which
is free of vulnerabilities, 2) review and share security analysis, and 3)
easily switch from one prompt optimization approach to another, while providing
insights on model and system performance. We populated SGCode on an AWS server
with PromSec, an approach that optimizes prompts by combining an LLM and
security tools with a lightweight generative adversarial graph neural network
to detect and fix security vulnerabilities in the generated code. Extensive
experiments show that SGCode is practical as a public tool to gain insights
into the trade-offs between model utility, secure code generation, and system
cost. SGCode has only a marginal cost compared with prompting LLMs. SGCode is
available at: https://sgcode.codes/.


---

**[41. [2009.09311] Differential Codes on Higher Dimensional Varieties Via Grothendieck's
  Residue Symbol](https://arxiv.org/pdf/2009.09311.pdf)** (2024-02-07)

*David Grant, John D. Massman, III, S. Srimathy*

  We give a new construction of linear codes over finite fields on higher
dimensional varieties using Grothendieck's theory of residues. This generalizes
the construction of differential codes over curves to varieties of higher
dimensions.


---

**[42. [2204.12120] Automated Generation of High-Performance Computational Fluid Dynamics
  Codes](https://arxiv.org/pdf/2204.12120.pdf)** (2022-04-28)

*Sandra Macià, Pedro J. Martıínez-Ferrer, Eduard Ayguadé, Vicenç Beltran*

  Domain-Specific Languages (DSLs) improve programmers productivity by
decoupling problem descriptions from algorithmic implementations. However, DSLs
for High-Performance Computing (HPC) have two additional critical requirements:
performance and scalability. This paper presents the automated process of
generating, from abstract mathematical specifications of Computational Fluid
Dynamics (CFD) problems, optimised parallel codes that perform and scale as
manually optimised ones. We consciously combine within Saiph, a DSL for solving
CFD problems, low-level optimisations and parallelisation strategies, enabling
high-performance single-core executions which effectively scale to multi-core
and distributed environments. Our results demonstrate how high-level DSLs can
offer competitive performance by transparently leveraging state-of-the-art HPC
techniques.


---

**[43. [2308.05673] Algorithms for Encoding and Decoding 3D Hilbert Orderings](https://arxiv.org/pdf/2308.05673.pdf)** (2023-09-26)

*David Walker*

  This paper presents algorithms and pseudocode for encoding and decoding 3D
Hilbert orderings.


---

**[44. [2206.06584] Probabilistic Conformal Prediction Using Conditional Random Samples](https://arxiv.org/pdf/2206.06584.pdf)** (2022-06-22)

*Zhendong Wang, Ruijiang Gao, Mingzhang Yin, Mingyuan Zhou, David M. Blei*

  This paper proposes probabilistic conformal prediction (PCP), a predictive
inference algorithm that estimates a target variable by a discontinuous
predictive set. Given inputs, PCP construct the predictive set based on random
samples from an estimated generative model. It is efficient and compatible with
either explicit or implicit conditional generative models. Theoretically, we
show that PCP guarantees correct marginal coverage with finite samples.
Empirically, we study PCP on a variety of simulated and real datasets. Compared
to existing methods for conformal inference, PCP provides sharper predictive
sets.


---

**[45. [2402.04486] Outer Code Designs for Augmented and Local-Global Polar Code
  Architectures](https://arxiv.org/pdf/2402.04486.pdf)** (2024-05-15)

*Ziyuan Zhu, Paul H. Siegel*

  In this paper, we introduce two novel methods to design outer polar codes for
two previously proposed concatenated polar code architectures: augmented polar
codes and local-global polar codes. These methods include a stopping set (SS)
construction and a nonstationary density evolution (NDE) construction.
Simulation results demonstrate the advantage of these methods over previously
proposed constructions based on density evolution (DE) and LLR evolution.


---

**[46. [2403.19115] HiRoPE: Length Extrapolation for Code Models Using Hierarchical Position](https://arxiv.org/pdf/2403.19115.pdf)** (2024-08-12)

*Kechi Zhang, Ge Li, Huangzhao Zhang, Zhi Jin*

  Addressing the limitation of context length in large language models for
code-related tasks is the primary focus of this paper. Existing LLMs are
constrained by their pre-trained context lengths, leading to performance issues
in handling long complex code sequences. Inspired by how human programmers
navigate code, we introduce Hierarchical Rotary Position Embedding (HiRoPE), a
novel approach that enhances the traditional rotary position embedding into a
hierarchical format based on the hierarchical structure of source code. HiRoPE
offers easy integration into existing LLMs without extra training costs. Our
method is extensively evaluated with various LLMs, demonstrating stable
performance in tasks such as language modeling and long code completion. We
also introduce a new long code understanding task with real-world code
projects, in hopes of promoting further development in this code-related field.
Theoretically and experimentally, we find that HiRoPE also addresses the
out-of-distribution issue in position encoding. Our HiRoPE significantly
expands the context length capabilities of LLMs, enabling inference at lengths
exponentially greater than the training length.


---

**[47. [2407.00499] ConU: Conformal Uncertainty in Large Language Models with Correctness
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

**[48. [2407.01638] LASSI: An LLM-based Automated Self-Correcting Pipeline for Translating
  Parallel Scientific Codes](https://arxiv.org/pdf/2407.01638.pdf)** (2024-07-03)

*Matthew T. Dearing, Yiheng Tao, Xingfu Wu, Zhiling Lan, Valerie Taylor*

  This paper addresses the problem of providing a novel approach to sourcing
significant training data for LLMs focused on science and engineering. In
particular, a crucial challenge is sourcing parallel scientific codes in the
ranges of millions to billions of codes. To tackle this problem, we propose an
automated pipeline framework, called LASSI, designed to translate between
parallel programming languages by bootstrapping existing closed- or open-source
LLMs. LASSI incorporates autonomous enhancement through self-correcting loops
where errors encountered during compilation and execution of generated code are
fed back to the LLM through guided prompting for debugging and refactoring. We
highlight the bi-directional translation of existing GPU benchmarks between
OpenMP target offload and CUDA to validate LASSI.
  The results of evaluating LASSI with different application codes across four
LLMs demonstrate the effectiveness of LASSI for generating executable parallel
codes, with 80% of OpenMP to CUDA translations and 85% of CUDA to OpenMP
translations producing the expected output. We also observe approximately 78%
of OpenMP to CUDA translations and 62% of CUDA to OpenMP translations execute
within 10% of or at a faster runtime than the original benchmark code in the
same language.


---

**[49. [2005.04151] Swarm Programming Using Moth-Flame Optimization and Whale Optimization
  Algorithms](https://arxiv.org/pdf/2005.04151.pdf)** (2020-05-11)

*Tapas Si*

  Automatic programming (AP) is an important area of Machine Learning (ML)
where computer programs are generated automatically. Swarm Programming (SP), a
newly emerging research area in AP, automatically generates the computer
programs using Swarm Intelligence (SI) algorithms. This paper presents two
grammar-based SP methods named as Grammatical Moth-Flame Optimizer (GMFO) and
Grammatical Whale Optimizer (GWO). The Moth-Flame Optimizer and Whale
Optimization algorithm are used as search engines or learning algorithms in
GMFO and GWO respectively. The proposed methods are tested on Santa Fe Ant
Trail, quartic symbolic regression, and 3-input multiplexer problems. The
results are compared with Grammatical Bee Colony (GBC) and Grammatical
Fireworks algorithm (GFWA). The experimental results demonstrate that the
proposed SP methods can be used in automatic computer program generation.


---

**[50. [2504.06475] Successive randomized compression: A randomized algorithm for the
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

**[51. [2309.01051] On Galois self-orthogonal algebraic geometry codes](https://arxiv.org/pdf/2309.01051.pdf)** (2024-04-01)

*Yun Ding, Shixin Zhu, Xiaoshan Kai, Yang Li*

  Galois self-orthogonal (SO) codes are generalizations of Euclidean and
Hermitian SO codes. Algebraic geometry (AG) codes are the first known class of
linear codes exceeding the Gilbert-Varshamov bound. Both of them have attracted
much attention for their rich algebraic structures and wide applications in
these years. In this paper, we consider them together and study Galois SO AG
codes. A criterion for an AG code being Galois SO is presented. Based on this
criterion, we construct several new classes of maximum distance separable (MDS)
Galois SO AG codes from projective lines and several new classes of Galois SO
AG codes from projective elliptic curves, hyper-elliptic curves and hermitian
curves. In addition, we give an embedding method that allows us to obtain more
MDS Galois SO codes from known MDS Galois SO AG codes.


---

**[52. [2405.15383] Generating Code World Models with Large Language Models Guided by Monte
  Carlo Tree Search](https://arxiv.org/pdf/2405.15383.pdf)** (2024-10-31)

*Nicola Dainese, Matteo Merler, Minttu Alakuijala, Pekka Marttinen*

  In this work we consider Code World Models, world models generated by a Large
Language Model (LLM) in the form of Python code for model-based Reinforcement
Learning (RL). Calling code instead of LLMs for planning has potential to be
more precise, reliable, interpretable, and extremely efficient. However,
writing appropriate Code World Models requires the ability to understand
complex instructions, to generate exact code with non-trivial logic and to
self-debug a long program with feedback from unit tests and environment
trajectories. To address these challenges, we propose Generate, Improve and Fix
with Monte Carlo Tree Search (GIF-MCTS), a new code generation strategy for
LLMs. To test our approach in an offline RL setting, we introduce the Code
World Models Benchmark (CWMB), a suite of program synthesis and planning tasks
comprised of 18 diverse RL environments paired with corresponding textual
descriptions and curated trajectories. GIF-MCTS surpasses all baselines on the
CWMB and two other benchmarks, and we show that the Code World Models
synthesized with it can be successfully used for planning, resulting in
model-based RL agents with greatly improved sample efficiency and inference
speed.


---

**[53. [2305.04087] Self-Edit: Fault-Aware Code Editor for Code Generation](https://arxiv.org/pdf/2305.04087.pdf)** (2023-09-12)

*Kechi Zhang, Zhuo Li, Jia Li, Ge Li, Zhi Jin*

  Large language models (LLMs) have demonstrated an impressive ability to
generate codes on competitive programming tasks. However, with limited sample
numbers, LLMs still suffer from poor accuracy. Inspired by the process of human
programming, we propose a generate-and-edit approach named Self-Edit that
utilizes execution results of the generated code from LLMs to improve the code
quality on the competitive programming task. We execute the generated code on
the example test case provided in the question and wrap execution results into
a supplementary comment. Utilizing this comment as guidance, our fault-aware
code editor is employed to correct errors in the generated code. We perform
extensive evaluations across two competitive programming datasets with nine
different LLMs. Compared to directly generating from LLMs, our approach can
improve the average of pass@1 by 89\% on APPS-dev, 31\% on APPS-test, and 48\%
on HumanEval over nine popular code generation LLMs with parameter sizes
ranging from 110M to 175B. Compared to other post-processing methods, our
method demonstrates superior accuracy and efficiency.


---

**[54. [2204.07537] Unconditional Image-Text Pair Generation with Multimodal Cross Quantizer](https://arxiv.org/pdf/2204.07537.pdf)** (2022-10-17)

*Hyungyung Lee, Sungjin Park, Joonseok Lee, Edward Choi*

  Although deep generative models have gained a lot of attention, most of the
existing works are designed for unimodal generation. In this paper, we explore
a new method for unconditional image-text pair generation. We design Multimodal
Cross-Quantization VAE (MXQ-VAE), a novel vector quantizer for joint image-text
representations, with which we discover that a joint image-text representation
space is effective for semantically consistent image-text pair generation. To
learn a multimodal semantic correlation in a quantized space, we combine VQ-VAE
with a Transformer encoder and apply an input masking strategy. Specifically,
MXQ-VAE accepts a masked image-text pair as input and learns a quantized joint
representation space, so that the input can be converted to a unified code
sequence, then we perform unconditional image-text pair generation with the
code sequence. Extensive experiments show the correlation between the quantized
joint space and the multimodal generation capability on synthetic and
real-world datasets. In addition, we demonstrate the superiority of our
approach in these two aspects over several baselines. The source code is
publicly available at: https://github.com/ttumyche/MXQ-VAE.


---

**[55. [2402.15674] Formally Verified C Code Generation from Hybrid Communicating Sequential
  Processes](https://arxiv.org/pdf/2402.15674.pdf)** (2024-02-28)

*Shuling Wang, Zekun Ji, Bohua Zhan, Xiong Xu, Qiang Gao, Naijun Zhan*

  Hybrid Communicating Sequential Processes (HCSP) is a formal model for hybrid
systems, including primitives for evolution along an ordinary differential
equation (ODE), communication, and parallel composition. Code generation is
needed to convert HCSP models into code that can be executed in practice, and
the correctness of this conversion is essential to ensure that the generated
code accurately reflects the formal model. In this paper, we propose a code
generation algorithm from HCSP to C with POSIX library for concurrency. The
main difficulties include how to bridge the gap between the synchronized
communication model in HCSP and the use of mutexes for synchronization in C,
and how to discretize evolution along ODEs and support interrupt of ODE
evolution by communication. To prove the correctness of code generation, we
define a formal semantics for POSIX C, and build transition system models for
both HCSP and C programs. We then define an approximate bisimulation relation
between traces of transition systems, and show that under certain robustness
conditions for HCSP, the generated C program is approximately bisimilar to the
original model. Finally, we evaluate the code generation algorithm on a
detailed model for automatic cruise control, showing its utility on real-world
examples.


---

**[56. [2001.04711] Partial MDS Codes with Local Regeneration](https://arxiv.org/pdf/2001.04711.pdf)** (2020-05-11)

*Lukas Holzbaur, Sven Puchinger, Eitan Yaakobi, Antonia Wachter-Zeh*

  Partial MDS (PMDS) and sector-disk (SD) codes are classes of erasure codes
that combine locality with strong erasure correction capabilities. We construct
PMDS and SD codes where each local code is a bandwidth-optimal regenerating MDS
code. The constructions require significantly smaller field size than the only
other construction known in literature.


---

**[57. [2503.12483] Modularization is Better: Effective Code Generation with Modular
  Prompting](https://arxiv.org/pdf/2503.12483.pdf)** (2025-03-18)

*Ruwei Pan, Hongyu Zhang*

  Large Language Models are transforming software development by automatically
generating code. Current prompting techniques such as Chain-of-Thought (CoT)
suggest tasks step by step and the reasoning process follows a linear
structure, which hampers the understanding of complex programming problems,
particularly those requiring hierarchical solutions. Inspired by the principle
of modularization in software development, in this work, we propose a novel
prompting technique, called MoT, to enhance the code generation performance of
LLMs. At first, MoT exploits modularization principles to decompose complex
programming problems into smaller, independent reasoning steps, enabling a more
structured and interpretable problem-solving process. This hierarchical
structure improves the LLM's ability to comprehend complex programming
problems. Then, it structures the reasoning process using an MLR Graph
(Multi-Level Reasoning Graph), which hierarchically organizes reasoning steps.
This approach enhances modular understanding and ensures better alignment
between reasoning steps and the generated code, significantly improving code
generation performance. Our experiments on two advanced LLMs (GPT-4o-mini and
DeepSeek-R1), comparing MoT to six baseline prompting techniques across six
widely used datasets, HumanEval, HumanEval-ET, HumanEval+, MBPP, MBPP-ET, and
MBPP+, demonstrate that MoT significantly outperforms existing baselines (e.g.,
CoT and SCoT), achieving Pass@1 scores ranging from 58.1% to 95.1%. The
experimental results confirm that MoT significantly enhances the performance of
LLM-based code generation.


---

**[58. [2207.04632] SkexGen: Autoregressive Generation of CAD Construction Sequences with
  Disentangled Codebooks](https://arxiv.org/pdf/2207.04632.pdf)** (2022-07-12)

*Xiang Xu, Karl D. D. Willis, Joseph G. Lambourne, Chin-Yi Cheng, Pradeep Kumar Jayaraman, Yasutaka Furukawa*

  We present SkexGen, a novel autoregressive generative model for
computer-aided design (CAD) construction sequences containing
sketch-and-extrude modeling operations. Our model utilizes distinct Transformer
architectures to encode topological, geometric, and extrusion variations of
construction sequences into disentangled codebooks. Autoregressive Transformer
decoders generate CAD construction sequences sharing certain properties
specified by the codebook vectors. Extensive experiments demonstrate that our
disentangled codebook representation generates diverse and high-quality CAD
models, enhances user control, and enables efficient exploration of the design
space. The code is available at https://samxuxiang.github.io/skexgen.


---

**[59. [2207.06803] FFTc: An MLIR Dialect for Developing HPC Fast Fourier Transform
  Libraries](https://arxiv.org/pdf/2207.06803.pdf)** (2022-07-27)

*Yifei He, Artur Podobas, Måns I. Andersson, Stefano Markidis*

  Discrete Fourier Transform (DFT) libraries are one of the most critical
software components for scientific computing. Inspired by FFTW, a widely used
library for DFT HPC calculations, we apply compiler technologies for the
development of HPC Fourier transform libraries. In this work, we introduce
FFTc, a domain-specific language, based on Multi-Level Intermediate
Representation (MLIR), for expressing Fourier Transform algorithms. We present
the initial design, implementation, and preliminary results of FFTc.


---

**[60. [2504.12189] Leave-One-Out Stable Conformal Prediction](https://arxiv.org/pdf/2504.12189.pdf)** (2025-04-17)

*Kiljae Lee, Yuan Zhang*

  Conformal prediction (CP) is an important tool for distribution-free
predictive uncertainty quantification. Yet, a major challenge is to balance
computational efficiency and prediction accuracy, particularly for multiple
predictions. We propose Leave-One-Out Stable Conformal Prediction (LOO-StabCP),
a novel method to speed up full conformal using algorithmic stability without
sample splitting. By leveraging leave-one-out stability, our method is much
faster in handling a large number of prediction requests compared to existing
method RO-StabCP based on replace-one stability. We derived stability bounds
for several popular machine learning tools: regularized loss minimization (RLM)
and stochastic gradient descent (SGD), as well as kernel method, neural
networks and bagging. Our method is theoretically justified and demonstrates
superior numerical performance on synthetic and real-world data. We applied our
method to a screening problem, where its effective exploitation of training
data led to improved test power compared to state-of-the-art method based on
split conformal.


---

**[61. [2405.18723] Conformal Depression Prediction](https://arxiv.org/pdf/2405.18723.pdf)** (2024-08-28)

*Yonghong Li, Xiuzhuang Zhou*

  While existing depression prediction methods based on deep learning show
promise, their practical application is hindered by the lack of
trustworthiness, as these deep models are often deployed as black box models,
leaving us uncertain on the confidence of their predictions. For high-risk
clinical applications like depression prediction, uncertainty quantification is
essential in decision-making. In this paper, we introduce conformal depression
prediction (CDP), a depression prediction method with uncertainty
quantification based on conformal prediction (CP), giving valid confidence
intervals with theoretical coverage guarantees for the model predictions. CDP
is a plug-and-play module that requires neither model retraining nor an
assumption about the depression data distribution. As CDP provides only an
average coverage guarantee across all inputs rather than per-input performance
guarantee, we further propose CDP-ACC, an improved conformal prediction with
approximate conditional coverage. CDP-ACC firstly estimates the prediction
distribution through neighborhood relaxation, and then introduces a conformal
score function by constructing nested sequences, so as to provide a tighter
prediction interval adaptive to specific input. We empirically demonstrate the
application of CDP in uncertainty-aware facial depression prediction, as well
as the effectiveness and superiority of CDP-ACC on the AVEC 2013 and AVEC 2014
datasets. Our code is publicly available at https://github.com/PushineLee/CDP.


---

**[62. [2307.15337] Skeleton-of-Thought: Prompting LLMs for Efficient Parallel Generation](https://arxiv.org/pdf/2307.15337.pdf)** (2024-03-05)

*Xuefei Ning, Zinan Lin, Zixuan Zhou, Zifu Wang, Huazhong Yang, Yu Wang*

  This work aims at decreasing the end-to-end generation latency of large
language models (LLMs). One of the major causes of the high generation latency
is the sequential decoding approach adopted by almost all state-of-the-art
LLMs. In this work, motivated by the thinking and writing process of humans, we
propose Skeleton-of-Thought (SoT), which first guides LLMs to generate the
skeleton of the answer, and then conducts parallel API calls or batched
decoding to complete the contents of each skeleton point in parallel. Not only
does SoT provide considerable speed-ups across 12 LLMs, but it can also
potentially improve the answer quality on several question categories. SoT is
an initial attempt at data-centric optimization for inference efficiency, and
showcases the potential of eliciting high-quality answers by explicitly
planning the answer structure in language.


---

**[63. [2104.13100] Shellcode_IA32: A Dataset for Automatic Shellcode Generation](https://arxiv.org/pdf/2104.13100.pdf)** (2022-03-21)

*Pietro Liguori, Erfan Al-Hossami, Domenico Cotroneo, Roberto Natella, Bojan Cukic, Samira Shaikh*

  We take the first step to address the task of automatically generating
shellcodes, i.e., small pieces of code used as a payload in the exploitation of
a software vulnerability, starting from natural language comments. We assemble
and release a novel dataset (Shellcode_IA32), consisting of challenging but
common assembly instructions with their natural language descriptions. We
experiment with standard methods in neural machine translation (NMT) to
establish baseline performance levels on this task.


---

**[64. [2409.20550] LLM Hallucinations in Practical Code Generation: Phenomena, Mechanism,
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

**[65. [2309.14049] How Novices Use LLM-Based Code Generators to Solve CS1 Coding Tasks in a
  Self-Paced Learning Environment](https://arxiv.org/pdf/2309.14049.pdf)** (2023-09-26)

*Majeed Kazemitabaar, Xinying Hou, Austin Henley, Barbara J. Ericson, David Weintrop, Tovi Grossman*

  As Large Language Models (LLMs) gain in popularity, it is important to
understand how novice programmers use them. We present a thematic analysis of
33 learners, aged 10-17, independently learning Python through 45
code-authoring tasks using Codex, an LLM-based code generator. We explore
several questions related to how learners used these code generators and
provide an analysis of the properties of the written prompts and the generated
code. Specifically, we explore (A) the context in which learners use Codex, (B)
what learners are asking from Codex, (C) properties of their prompts in terms
of relation to task description, language, and clarity, and prompt crafting
patterns, (D) the correctness, complexity, and accuracy of the AI-generated
code, and (E) how learners utilize AI-generated code in terms of placement,
verification, and manual modifications. Furthermore, our analysis reveals four
distinct coding approaches when writing code with an AI code generator: AI
Single Prompt, where learners prompted Codex once to generate the entire
solution to a task; AI Step-by-Step, where learners divided the problem into
parts and used Codex to generate each part; Hybrid, where learners wrote some
of the code themselves and used Codex to generate others; and Manual coding,
where learners wrote the code themselves. The AI Single Prompt approach
resulted in the highest correctness scores on code-authoring tasks, but the
lowest correctness scores on subsequent code-modification tasks during
training. Our results provide initial insight into how novice learners use AI
code generators and the challenges and opportunities associated with
integrating them into self-paced learning environments. We conclude with
various signs of over-reliance and self-regulation, as well as opportunities
for curriculum and tool development.


---

**[66. [2406.16441] UniCoder: Scaling Code Large Language Model via Universal Code](https://arxiv.org/pdf/2406.16441.pdf)** (2024-06-25)

*Tao Sun, Linzheng Chai, Jian Yang, Yuwei Yin, Hongcheng Guo, Jiaheng Liu, Bing Wang, Liqun Yang, Zhoujun Li*

  Intermediate reasoning or acting steps have successfully improved large
language models (LLMs) for handling various downstream natural language
processing (NLP) tasks. When applying LLMs for code generation, recent works
mainly focus on directing the models to articulate intermediate
natural-language reasoning steps, as in chain-of-thought (CoT) prompting, and
then output code with the natural language or other structured intermediate
steps. However, such output is not suitable for code translation or generation
tasks since the standard CoT has different logical structures and forms of
expression with the code. In this work, we introduce the universal code
(UniCode) as the intermediate representation. It is a description of algorithm
steps using a mix of conventions of programming languages, such as assignment
operator, conditional operator, and loop. Hence, we collect an instruction
dataset UniCoder-Instruct to train our model UniCoder on multi-task learning
objectives. UniCoder-Instruct comprises natural-language questions, code
solutions, and the corresponding universal code. The alignment between the
intermediate universal code representation and the final code solution
significantly improves the quality of the generated code. The experimental
results demonstrate that UniCoder with the universal code significantly
outperforms the previous prompting methods by a large margin, showcasing the
effectiveness of the structural clues in pseudo-code.


---

**[67. [2412.03578] PerfCodeGen: Improving Performance of LLM Generated Code with Execution
  Feedback](https://arxiv.org/pdf/2412.03578.pdf)** (2024-12-06)

*Yun Peng, Akhilesh Deepak Gotmare, Michael Lyu, Caiming Xiong, Silvio Savarese, Doyen Sahoo*

  Large Language Models (LLMs) are widely adopted for assisting in software
development tasks, yet their performance evaluations have narrowly focused on
the functional correctness of generated code. Human programmers, however,
require LLM-generated code to be not only correct but also optimally efficient.
We propose PerfCodeGen, a training-free framework that enhances the performance
of LLM-generated code by incorporating feedback based on runtime during test
case execution into the self-refinement iterations. With PerfCodeGen, we
achieve speedups for a significantly higher proportion of problems compared to
using the base LLM with sophisticated prompting techniques. Applied to open
language models like Phi-3-mini, PerfCodeGen achieves runtime efficiency
comparable to prompting powerful closed models like GPT-4. We achieve
state-of-the-art runtime efficiency on benchmarks such as HumanEval, MBPP, and
APPS, frequently surpassing the ground truth reference solutions with
PerfCodeGen using GPT-3.5 and GPT-4. Additionally, we demonstrate the
effectiveness of our approach in enhancing code quality across a range of open
LLMs of varying sizes including Phi-3-mini, Llama 3 8B, Mixtral 8x7B, Command
R, and Llama 3 70B.


---

**[68. [2111.02592] Conformal prediction for text infilling and part-of-speech prediction](https://arxiv.org/pdf/2111.02592.pdf)** (2021-11-05)

*Neil Dey, Jing Ding, Jack Ferrell, Carolina Kapper, Maxwell Lovig, Emiliano Planchon, Jonathan P Williams*

  Modern machine learning algorithms are capable of providing remarkably
accurate point-predictions; however, questions remain about their statistical
reliability. Unlike conventional machine learning methods, conformal prediction
algorithms return confidence sets (i.e., set-valued predictions) that
correspond to a given significance level. Moreover, these confidence sets are
valid in the sense that they guarantee finite sample control over type 1 error
probabilities, allowing the practitioner to choose an acceptable error rate. In
our paper, we propose inductive conformal prediction (ICP) algorithms for the
tasks of text infilling and part-of-speech (POS) prediction for natural
language data. We construct new conformal prediction-enhanced bidirectional
encoder representations from transformers (BERT) and bidirectional long
short-term memory (BiLSTM) algorithms for POS tagging and a new conformal
prediction-enhanced BERT algorithm for text infilling. We analyze the
performance of the algorithms in simulations using the Brown Corpus, which
contains over 57,000 sentences. Our results demonstrate that the ICP algorithms
are able to produce valid set-valued predictions that are small enough to be
applicable in real-world applications. We also provide a real data example for
how our proposed set-valued predictions can improve machine generated audio
transcriptions.


---

**[69. [2304.10778] Evaluating the Code Quality of AI-Assisted Code Generation Tools: An
  Empirical Study on GitHub Copilot, Amazon CodeWhisperer, and ChatGPT](https://arxiv.org/pdf/2304.10778.pdf)** (2023-10-24)

*Burak Yetiştiren, Işık Özsoy, Miray Ayerdem, Eray Tüzün*

  Context: AI-assisted code generation tools have become increasingly prevalent
in software engineering, offering the ability to generate code from natural
language prompts or partial code inputs. Notable examples of these tools
include GitHub Copilot, Amazon CodeWhisperer, and OpenAI's ChatGPT.
  Objective: This study aims to compare the performance of these prominent code
generation tools in terms of code quality metrics, such as Code Validity, Code
Correctness, Code Security, Code Reliability, and Code Maintainability, to
identify their strengths and shortcomings.
  Method: We assess the code generation capabilities of GitHub Copilot, Amazon
CodeWhisperer, and ChatGPT using the benchmark HumanEval Dataset. The generated
code is then evaluated based on the proposed code quality metrics.
  Results: Our analysis reveals that the latest versions of ChatGPT, GitHub
Copilot, and Amazon CodeWhisperer generate correct code 65.2%, 46.3%, and 31.1%
of the time, respectively. In comparison, the newer versions of GitHub CoPilot
and Amazon CodeWhisperer showed improvement rates of 18% for GitHub Copilot and
7% for Amazon CodeWhisperer. The average technical debt, considering code
smells, was found to be 8.9 minutes for ChatGPT, 9.1 minutes for GitHub
Copilot, and 5.6 minutes for Amazon CodeWhisperer.
  Conclusions: This study highlights the strengths and weaknesses of some of
the most popular code generation tools, providing valuable insights for
practitioners. By comparing these generators, our results may assist
practitioners in selecting the optimal tool for specific tasks, enhancing their
decision-making process.


---

**[70. [2403.01216] API Is Enough: Conformal Prediction for Large Language Models Without
  Logit-Access](https://arxiv.org/pdf/2403.01216.pdf)** (2024-04-05)

*Jiayuan Su, Jing Luo, Hongwei Wang, Lu Cheng*

  This study aims to address the pervasive challenge of quantifying uncertainty
in large language models (LLMs) without logit-access. Conformal Prediction
(CP), known for its model-agnostic and distribution-free features, is a desired
approach for various LLMs and data distributions. However, existing CP methods
for LLMs typically assume access to the logits, which are unavailable for some
API-only LLMs. In addition, logits are known to be miscalibrated, potentially
leading to degraded CP performance. To tackle these challenges, we introduce a
novel CP method that (1) is tailored for API-only LLMs without logit-access;
(2) minimizes the size of prediction sets; and (3) ensures a statistical
guarantee of the user-defined coverage. The core idea of this approach is to
formulate nonconformity measures using both coarse-grained (i.e., sample
frequency) and fine-grained uncertainty notions (e.g., semantic similarity).
Experimental results on both close-ended and open-ended Question Answering
tasks show our approach can mostly outperform the logit-based CP baselines.


---
