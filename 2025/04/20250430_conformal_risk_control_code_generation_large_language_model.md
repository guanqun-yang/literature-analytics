**[1. [2501.02486] LLMPC: Large Language Model Predictive Control](https://arxiv.org/pdf/2501.02486.pdf)** (2025-02-26)

*Gabriel Maher*

  Recent advancements in prompting techniques for Large Language Models (LLMs)
have improved their reasoning, planning, and action abilities. This paper
examines these prompting techniques through the lens of model predictive
control (MPC). We show that LLMs act as implicit planning cost function
minimizers when planning prompts are used. We propose a unified MPC framework
for planning with LLMs and demonstrate improved performance over few shot
prompting on several planning benchmarks.


---

**[2. [2405.06237] Risks of Practicing Large Language Models in Smart Grid: Threat Modeling
  and Validation](https://arxiv.org/pdf/2405.06237.pdf)** (2024-11-19)

*Jiangnan Li, Yingyuan Yang, Jinyuan Sun*

  Large language models (LLMs) represent significant breakthroughs in
artificial intelligence and hold considerable potential for applications within
smart grids. However, as demonstrated in previous literature, AI technologies
are susceptible to various types of attacks. It is crucial to investigate and
evaluate the risks associated with LLMs before deploying them in critical
infrastructure like smart grids. In this paper, we systematically evaluated the
risks of LLMs and identified two major types of attacks relevant to potential
smart grid LLM applications, presenting the corresponding threat models. We
also validated these attacks using popular LLMs and real smart grid data. Our
validation demonstrates that attackers are capable of injecting bad data and
retrieving domain knowledge from LLMs employed in different smart grid
applications.


---

**[3. [2308.02678] Ethical Considerations and Policy Implications for Large Language
  Models: Guiding Responsible Development and Deployment](https://arxiv.org/pdf/2308.02678.pdf)** (2023-08-08)

*Jianyi Zhang, Xu Ji, Zhangchi Zhao, Xiali Hei, Kim-Kwang Raymond Choo*

  This paper examines the ethical considerations and implications of large
language models (LLMs) in generating content. It highlights the potential for
both positive and negative uses of generative AI programs and explores the
challenges in assigning responsibility for their outputs. The discussion
emphasizes the need for proactive ethical frameworks and policy measures to
guide the responsible development and deployment of LLMs.


---

**[4. [2306.03081] Sequential Monte Carlo Steering of Large Language Models using
  Probabilistic Programs](https://arxiv.org/pdf/2306.03081.pdf)** (2023-11-28)

*Alexander K. Lew, Tan Zhi-Xuan, Gabriel Grand, Vikash K. Mansinghka*

  Even after fine-tuning and reinforcement learning, large language models
(LLMs) can be difficult, if not impossible, to control reliably with prompts
alone. We propose a new inference-time approach to enforcing syntactic and
semantic constraints on the outputs of LLMs, called sequential Monte Carlo
(SMC) steering. The key idea is to specify language generation tasks as
posterior inference problems in a class of discrete probabilistic sequence
models, and replace standard decoding with sequential Monte Carlo inference.
For a computational cost similar to that of beam search, SMC can steer LLMs to
solve diverse tasks, including infilling, generation under syntactic
constraints, and prompt intersection. To facilitate experimentation with SMC
steering, we present a probabilistic programming library, LLaMPPL
(https://github.com/probcomp/hfppl), for concisely specifying new generation
tasks as language model probabilistic programs, and automating steering of
LLaMA-family Transformers.


---

**[5. [2401.11974] Cross-Validation Conformal Risk Control](https://arxiv.org/pdf/2401.11974.pdf)** (2024-05-02)

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

**[6. [2501.05764] Controlling Large Language Models Through Concept Activation Vectors](https://arxiv.org/pdf/2501.05764.pdf)** (2025-01-13)

*Hanyu Zhang, Xiting Wang, Chengao Li, Xiang Ao, Qing He*

  As large language models (LLMs) are widely deployed across various domains,
the ability to control their generated outputs has become more critical. This
control involves aligning LLMs outputs with human values and ethical principles
or customizing LLMs on specific topics or styles for individual users. Existing
controlled generation methods either require significant computational
resources and extensive trial-and-error or provide coarse-grained control. In
this paper, we propose Generation with Concept Activation Vector (GCAV), a
lightweight model control framework that ensures accurate control without
requiring resource-extensive fine-tuning. Specifically, GCAV first trains a
concept activation vector for specified concepts to be controlled, such as
toxicity. During inference, GCAV steers the concept vector in LLMs, for
example, by removing the toxicity concept vector from the activation layers.
Control experiments from different perspectives, including toxicity reduction,
sentiment control, linguistic style, and topic control, demonstrate that our
framework achieves state-of-the-art performance with granular control, allowing
for fine-grained adjustments of both the steering layers and the steering
magnitudes for individual samples.


---

**[7. [2412.05625] Can Large Language Models Help Developers with Robotic Finite State
  Machine Modification?](https://arxiv.org/pdf/2412.05625.pdf)** (2024-12-10)

*Xiangyu Robin Gan, Yuxin Ray Song, Nick Walker, Maya Cakmak*

  Finite state machines (FSMs) are widely used to manage robot behavior logic,
particularly in real-world applications that require a high degree of
reliability and structure. However, traditional manual FSM design and
modification processes can be time-consuming and error-prone. We propose that
large language models (LLMs) can assist developers in editing FSM code for
real-world robotic use cases. LLMs, with their ability to use context and
process natural language, offer a solution for FSM modification with high
correctness, allowing developers to update complex control logic through
natural language instructions. Our approach leverages few-shot prompting and
language-guided code generation to reduce the amount of time it takes to edit
an FSM. To validate this approach, we evaluate it on a real-world robotics
dataset, demonstrating its effectiveness in practical scenarios.


---

**[8. [2404.09932] Foundational Challenges in Assuring Alignment and Safety of Large
  Language Models](https://arxiv.org/pdf/2404.09932.pdf)** (2024-09-09)

*Usman Anwar, Abulhair Saparov, Javier Rando, Daniel Paleka, Miles Turpin, Peter Hase, Ekdeep Singh Lubana, Erik Jenner, Stephen Casper, Oliver Sourbut, Benjamin L. Edelman, Zhaowei Zhang, Mario Günther, Anton Korinek, Jose Hernandez-Orallo, Lewis Hammond, Eric Bigelow, Alexander Pan, Lauro Langosco, Tomasz Korbak, Heidi Zhang, Ruiqi Zhong, Seán Ó hÉigeartaigh, Gabriel Recchia, Giulio Corsi, Alan Chan, Markus Anderljung, Lilian Edwards, Aleksandar Petrov, Christian Schroeder de Witt, Sumeet Ramesh Motwan, Yoshua Bengio, Danqi Chen, Philip H. S. Torr, Samuel Albanie, Tegan Maharaj, Jakob Foerster, Florian Tramer, He He, Atoosa Kasirzadeh, Yejin Choi, David Krueger*

  This work identifies 18 foundational challenges in assuring the alignment and
safety of large language models (LLMs). These challenges are organized into
three different categories: scientific understanding of LLMs, development and
deployment methods, and sociotechnical challenges. Based on the identified
challenges, we pose $200+$ concrete research questions.


---

**[9. [2503.18460] ModiGen: A Large Language Model-Based Workflow for Multi-Task Modelica
  Code Generation](https://arxiv.org/pdf/2503.18460.pdf)** (2025-03-25)

*Jiahui Xiang, Tong Ye, Peiyu Liu, Yinan Zhang, Wenhai Wang*

  Modelica is a widely adopted language for simulating complex physical
systems, yet effective model creation and optimization require substantial
domain expertise. Although large language models (LLMs) have demonstrated
promising capabilities in code generation, their application to modeling
remains largely unexplored. To address this gap, we have developed benchmark
datasets specifically designed to evaluate the performance of LLMs in
generating Modelica component models and test cases. Our evaluation reveals
substantial limitations in current LLMs, as the generated code often fails to
simulate successfully. To overcome these challenges, we propose a specialized
workflow that integrates supervised fine-tuning, graph retrieval-augmented
generation, and feedback optimization to improve the accuracy and reliability
of Modelica code generation. The evaluation results demonstrate significant
performance gains: the maximum improvement in pass@1 reached 0.3349 for the
component generation task and 0.2457 for the test case generation task. This
research underscores the potential of LLMs to advance intelligent modeling
tools and offers valuable insights for future developments in system modeling
and engineering applications.


---

**[10. [2403.01216] API Is Enough: Conformal Prediction for Large Language Models Without
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

**[11. [2404.19048] A Framework for Real-time Safeguarding the Text Generation of Large
  Language Model](https://arxiv.org/pdf/2404.19048.pdf)** (2024-05-03)

*Ximing Dong, Dayi Lin, Shaowei Wang, Ahmed E. Hassan*

  Large Language Models (LLMs) have significantly advanced natural language
processing (NLP) tasks but also pose ethical and societal risks due to their
propensity to generate harmful content. To address this, various approaches
have been developed to safeguard LLMs from producing unsafe content. However,
existing methods have limitations, including the need for training specific
control models and proactive intervention during text generation, that lead to
quality degradation and increased computational overhead. To mitigate those
limitations, we propose LLMSafeGuard, a lightweight framework to safeguard LLM
text generation in real-time. LLMSafeGuard integrates an external validator
into the beam search algorithm during decoding, rejecting candidates that
violate safety constraints while allowing valid ones to proceed. We introduce a
similarity based validation approach, simplifying constraint introduction and
eliminating the need for control model training. Additionally, LLMSafeGuard
employs a context-wise timing selection strategy, intervening LLMs only when
necessary. We evaluate LLMSafeGuard on two tasks, detoxification and copyright
safeguarding, and demonstrate its superior performance over SOTA baselines. For
instance, LLMSafeGuard reduces the average toxic score of. LLM output by 29.7%
compared to the best baseline meanwhile preserving similar linguistic quality
as natural output in detoxification task. Similarly, in the copyright task,
LLMSafeGuard decreases the Longest Common Subsequence (LCS) by 56.2% compared
to baselines. Moreover, our context-wise timing selection strategy reduces
inference time by at least 24% meanwhile maintaining comparable effectiveness
as validating each time step. LLMSafeGuard also offers tunable parameters to
balance its effectiveness and efficiency.


---

**[12. [2501.18883] Predictive Prompt Analysis](https://arxiv.org/pdf/2501.18883.pdf)** (2025-03-14)

*Jae Yong Lee, Sungmin Kang, Shin Yoo*

  Large Language Models (LLMs) are machine learning models that have seen
widespread adoption due to their capability of handling previously difficult
tasks. LLMs, due to their training, are sensitive to how exactly a question is
presented, also known as prompting. However, prompting well is challenging, as
it has been difficult to uncover principles behind prompting -- generally,
trial-and-error is the most common way of improving prompts, despite its
significant computational cost. In this context, we argue it would be useful to
perform `predictive prompt analysis', in which an automated technique would
perform a quick analysis of a prompt and predict how the LLM would react to it,
relative to a goal provided by the user. As a demonstration of the concept, we
present Syntactic Prevalence Analyzer (SPA), a predictive prompt analysis
approach based on sparse autoencoders (SAEs). SPA accurately predicted how
often an LLM would generate target syntactic structures during code synthesis,
with up to 0.994 Pearson correlation between the predicted and actual
prevalence of the target structure. At the same time, SPA requires only 0.4\%
of the time it takes to run the LLM on a benchmark. As LLMs are increasingly
used during and integrated into modern software development, our proposed
predictive prompt analysis concept has the potential to significantly ease the
use of LLMs for both practitioners and researchers.


---

**[13. [2502.05150] CodeSCM: Causal Analysis for Multi-Modal Code Generation](https://arxiv.org/pdf/2502.05150.pdf)** (2025-02-10)

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

**[14. [2406.04306] Semantically Diverse Language Generation for Uncertainty Estimation in
  Language Models](https://arxiv.org/pdf/2406.04306.pdf)** (2024-06-07)

*Lukas Aichberger, Kajetan Schweighofer, Mykyta Ielanskyi, Sepp Hochreiter*

  Large language models (LLMs) can suffer from hallucinations when generating
text. These hallucinations impede various applications in society and industry
by making LLMs untrustworthy. Current LLMs generate text in an autoregressive
fashion by predicting and appending text tokens. When an LLM is uncertain about
the semantic meaning of the next tokens to generate, it is likely to start
hallucinating. Thus, it has been suggested that hallucinations stem from
predictive uncertainty. We introduce Semantically Diverse Language Generation
(SDLG) to quantify predictive uncertainty in LLMs. SDLG steers the LLM to
generate semantically diverse yet likely alternatives for an initially
generated text. This approach provides a precise measure of aleatoric semantic
uncertainty, detecting whether the initial text is likely to be hallucinated.
Experiments on question-answering tasks demonstrate that SDLG consistently
outperforms existing methods while being the most computationally efficient,
setting a new standard for uncertainty estimation in LLMs.


---

**[15. [2408.15625] CBF-LLM: Safe Control for LLM Alignment](https://arxiv.org/pdf/2408.15625.pdf)** (2024-10-08)

*Yuya Miyaoka, Masaki Inoue*

  This paper proposes a control-based framework for aligning large language
models (LLMs) by leveraging a control barrier function (CBF) to ensure
user-desirable text generation. The presented framework applies the safety
filter, designed based on the CBF, to the output generation of the baseline
LLM, i.e., the sequence of the token, with the aim of intervening in the
generated text. The overall text-generation system is implemented with Llama 3
and a RoBERTa model, and the source code is available at
https://github.com/Mya-Mya/CBF-LLM. The experiment demonstrates its control
ability and effectiveness in reducing the number of interventions needed for
user-specified alignment tasks.


---

**[16. [2411.01696] Conformal Risk Minimization with Variance Reduction](https://arxiv.org/pdf/2411.01696.pdf)** (2025-02-11)

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

**[17. [2310.10049] FATE-LLM: A Industrial Grade Federated Learning Framework for Large
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

**[18. [2402.03181] C-RAG: Certified Generation Risks for Retrieval-Augmented Language
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

**[19. [2404.19368] Exploring Multi-Lingual Bias of Large Code Models in Code Generation](https://arxiv.org/pdf/2404.19368.pdf)** (2024-05-01)

*Chaozheng Wang, Zongjie Li, Cuiyun Gao, Wenxuan Wang, Ting Peng, Hailiang Huang, Yuetang Deng, Shuai Wang, Michael R. Lyu*

  Code generation aims to synthesize code and fulfill functional requirements
based on natural language (NL) specifications, which can greatly improve
development efficiency. In the era of large language models (LLMs), large code
models (LCMs) have been recently proposed to generate source code. LCMs can
generate highly feasible solutions for programming problems described in
natural language. Despite the effectiveness, we observe a noticeable
multilingual bias in the generation performance of LCMs. Specifically, LCMs
demonstrate proficiency in generating solutions when provided with instructions
in English, yet may falter when faced with semantically equivalent instructions
in other NLs such as Chinese. Moreover, the ability of LCMs to generate code
exhibits variety across different programming languages (PLs), such as Python
and C++. The observed phenomenon indicates the presence of multi-lingual bias
within the generative capabilities of LCMs, which has remained unexplored.
  In this paper, we aim to investigate the multi-lingual bias that exists in
current LCMs. First, we initiate our investigation by constructing the first
multi-lingual evaluation benchmark X-HumanEval-X, enabling us to systematically
evaluate the extent of multi-lingual bias that exists in current LCMs. In our
large-scale experiments on nine popular LCMs, we observe a pronounced
multi-lingual bias of LCMs in code generation, including multi-NL and multi-PL
bias. Specifically, when using Chinese instructions, the code generation
capabilities of LCMs decrease by at least 13% in terms of the Pass@1 metric.
Furthermore, LCMs perform variously across different programming languages,
e.g., the performance gap between Python and C++ reaches as high as 20.9%. ...


---

**[20. [2503.05852] Evaluating Large Language Models in Code Generation: INFINITE
  Methodology for Defining the Inference Index](https://arxiv.org/pdf/2503.05852.pdf)** (2025-04-10)

*Nicholas Christakis, Dimitris Drikakis*

  This study introduces a new methodology for an Inference Index (InI), called
INFerence INdex In Testing model Effectiveness methodology (INFINITE), aiming
to evaluate the performance of Large Language Models (LLMs) in code generation
tasks. The InI index provides a comprehensive assessment focusing on three key
components: efficiency, consistency, and accuracy. This approach encapsulates
time-based efficiency, response quality, and the stability of model outputs,
offering a thorough understanding of LLM performance beyond traditional
accuracy metrics. We applied this methodology to compare OpenAI's GPT-4o (GPT),
OpenAI-o1 pro (OAI1), and OpenAI-o3 mini-high (OAI3) in generating Python code
for the Long-Short-Term-Memory (LSTM) model to forecast meteorological
variables such as temperature, relative humidity and wind velocity. Our
findings demonstrate that GPT outperforms OAI1 and performs comparably to OAI3
regarding accuracy and workflow efficiency. The study reveals that LLM-assisted
code generation can produce results similar to expert-designed models with
effective prompting and refinement. GPT's performance advantage highlights the
benefits of widespread use and user feedback.


---

**[21. [2407.07666] A Proposed S.C.O.R.E. Evaluation Framework for Large Language Models :
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

**[22. [2403.05156] On Protecting the Data Privacy of Large Language Models (LLMs): A Survey](https://arxiv.org/pdf/2403.05156.pdf)** (2024-03-15)

*Biwei Yan, Kun Li, Minghui Xu, Yueyan Dong, Yue Zhang, Zhaochun Ren, Xiuzhen Cheng*

  Large language models (LLMs) are complex artificial intelligence systems
capable of understanding, generating and translating human language. They learn
language patterns by analyzing large amounts of text data, allowing them to
perform writing, conversation, summarizing and other language tasks. When LLMs
process and generate large amounts of data, there is a risk of leaking
sensitive information, which may threaten data privacy. This paper concentrates
on elucidating the data privacy concerns associated with LLMs to foster a
comprehensive understanding. Specifically, a thorough investigation is
undertaken to delineate the spectrum of data privacy threats, encompassing both
passive privacy leakage and active privacy attacks within LLMs. Subsequently,
we conduct an assessment of the privacy protection mechanisms employed by LLMs
at various stages, followed by a detailed examination of their efficacy and
constraints. Finally, the discourse extends to delineate the challenges
encountered and outline prospective directions for advancement in the realm of
LLM privacy protection.


---

**[23. [2412.07992] Concept Bottleneck Large Language Models](https://arxiv.org/pdf/2412.07992.pdf)** (2025-04-04)

*Chung-En Sun, Tuomas Oikarinen, Berk Ustun, Tsui-Wei Weng*

  We introduce Concept Bottleneck Large Language Models (CB-LLMs), a novel
framework for building inherently interpretable Large Language Models (LLMs).
In contrast to traditional black-box LLMs that rely on limited post-hoc
interpretations, CB-LLMs integrate intrinsic interpretability directly into the
LLMs -- allowing accurate explanations with scalability and transparency. We
build CB-LLMs for two essential NLP tasks: text classification and text
generation. In text classification, CB-LLMs is competitive with, and at times
outperforms, traditional black-box models while providing explicit and
interpretable reasoning. For the more challenging task of text generation,
interpretable neurons in CB-LLMs enable precise concept detection, controlled
generation, and safer outputs. The embedded interpretability empowers users to
transparently identify harmful content, steer model behavior, and unlearn
undesired concepts -- significantly enhancing the safety, reliability, and
trustworthiness of LLMs, which are critical capabilities notably absent in
existing models. Our code is available at
https://github.com/Trustworthy-ML-Lab/CB-LLMs.


---

**[24. [2404.11086] ViLLM-Eval: A Comprehensive Evaluation Suite for Vietnamese Large
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

**[25. [2412.18989] How Propense Are Large Language Models at Producing Code Smells? A
  Benchmarking Study](https://arxiv.org/pdf/2412.18989.pdf)** (2025-01-22)

*Alejandro Velasco, Daniel Rodriguez-Cardenas, Luftar Rahman Alif, David N. Palacio, Denys Poshyvanyk*

  Large Language Models (LLMs) have shown significant potential in automating
software engineering tasks, particularly in code generation. However, current
evaluation benchmarks, which primarily focus on accuracy, fall short in
assessing the quality of the code generated by these models, specifically their
tendency to produce code smells. To address this limitation, we introduce
CodeSmellEval, a benchmark designed to evaluate the propensity of LLMs for
generating code smells. Our benchmark includes a novel metric: Propensity
Smelly Score (PSC), and a curated dataset of method-level code smells:
CodeSmellData. To demonstrate the use of CodeSmellEval, we conducted a case
study with two state-of-the-art LLMs, CodeLlama and Mistral. The results reveal
that both models tend to generate code smells, such as simplifiable-condition
and consider-merging-isinstance. These findings highlight the effectiveness of
our benchmark in evaluating LLMs, providing valuable insights into their
reliability and their propensity to introduce code smells in code generation
tasks.


---

**[26. [2404.00600] AI Act and Large Language Models (LLMs): When critical issues and
  privacy impact require human and ethical oversight](https://arxiv.org/pdf/2404.00600.pdf)** (2024-04-03)

*Nicola Fabiano*

  The imposing evolution of artificial intelligence systems and, specifically,
of Large Language Models (LLM) makes it necessary to carry out assessments of
their level of risk and the impact they may have in the area of privacy,
personal data protection and at an ethical level, especially on the weakest and
most vulnerable. This contribution addresses human oversight, ethical
oversight, and privacy impact assessment.


---

**[27. [2407.03387] ConCodeEval: Evaluating Large Language Models for Code Constraints in
  Domain-Specific Languages](https://arxiv.org/pdf/2407.03387.pdf)** (2025-03-25)

*Mehant Kammakomati, Sameer Pimparkhede, Srikanth Tamilselvam, Prince Kumar, Pushpak Bhattacharyya*

  Recent work shows Large Language Models (LLMs) struggle to understand natural
language constraints for various text generation tasks in zero- and few-shot
settings. While, in the code domain, there is wide usage of constraints in code
format to maintain the integrity of code written in Domain-Specific Languages
(DSLs) like JSON and YAML which are widely used for system-level programming
tasks in enterprises. Given that LLMs are increasingly used for system-level
code tasks, evaluating if they can comprehend these code constraints is
crucial. However, no work has been done to evaluate their controllability over
code constraints. Hence, we introduce ConCodeEval, a first-of-its-kind
benchmark having two novel tasks for code constraints across five
representations. Our findings suggest that language models struggle with code
constraints. Code languages that perform excellently for normal code tasks do
not perform well when the same languages represent fine-grained constraints.


---

**[28. [2504.04151] STEP: Staged Parameter-Efficient Pre-training for Large Language Models](https://arxiv.org/pdf/2504.04151.pdf)** (2025-04-08)

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

**[29. [2404.11338] LLMs for Cyber Security: New Opportunities](https://arxiv.org/pdf/2404.11338.pdf)** (2024-04-18)

*Dinil Mon Divakaran, Sai Teja Peddinti*

  Large language models (LLMs) are a class of powerful and versatile models
that are beneficial to many industries. With the emergence of LLMs, we take a
fresh look at cyber security, specifically exploring and summarizing the
potential of LLMs in addressing challenging problems in the security and safety
domains.


---

**[30. [2311.15786] YUAN 2.0: A Large Language Model with Localized Filtering-based
  Attention](https://arxiv.org/pdf/2311.15786.pdf)** (2023-12-19)

*Shaohua Wu, Xudong Zhao, Shenling Wang, Jiangang Luo, Lingjun Li, Xi Chen, Bing Zhao, Wei Wang, Tong Yu, Rongguo Zhang, Jiahua Zhang, Chao Wang*

  In this work, we develop and release Yuan 2.0, a series of large language
models with parameters ranging from 2.1 billion to 102.6 billion. The Localized
Filtering-based Attention (LFA) is introduced to incorporate prior knowledge of
local dependencies of natural language into Attention. A data filtering and
generating system is presented to build pre-training and fine-tuning dataset in
high quality. A distributed training method with non-uniform pipeline parallel,
data parallel, and optimizer parallel is proposed, which greatly reduces the
bandwidth requirements of intra-node communication, and achieves good
performance in large-scale distributed training. Yuan 2.0 models display
impressive ability in code generation, math problem-solving, and chatting
compared with existing models. The latest version of YUAN 2.0, including model
weights and source code, is accessible at Github.


---

**[31. [2303.09384] LLMSecEval: A Dataset of Natural Language Prompts for Security
  Evaluations](https://arxiv.org/pdf/2303.09384.pdf)** (2023-03-17)

*Catherine Tony, Markus Mutas, Nicolás E. Díaz Ferreyra, Riccardo Scandariato*

  Large Language Models (LLMs) like Codex are powerful tools for performing
code completion and code generation tasks as they are trained on billions of
lines of code from publicly available sources. Moreover, these models are
capable of generating code snippets from Natural Language (NL) descriptions by
learning languages and programming practices from public GitHub repositories.
Although LLMs promise an effortless NL-driven deployment of software
applications, the security of the code they generate has not been extensively
investigated nor documented. In this work, we present LLMSecEval, a dataset
containing 150 NL prompts that can be leveraged for assessing the security
performance of such models. Such prompts are NL descriptions of code snippets
prone to various security vulnerabilities listed in MITRE's Top 25 Common
Weakness Enumeration (CWE) ranking. Each prompt in our dataset comes with a
secure implementation example to facilitate comparative evaluations against
code produced by LLMs. As a practical application, we show how LLMSecEval can
be used for evaluating the security of snippets automatically generated from NL
descriptions.


---

**[32. [2310.15205] DISC-FinLLM: A Chinese Financial Large Language Model based on Multiple
  Experts Fine-tuning](https://arxiv.org/pdf/2310.15205.pdf)** (2023-10-26)

*Wei Chen, Qiushi Wang, Zefei Long, Xianyin Zhang, Zhongtian Lu, Bingxuan Li, Siyuan Wang, Jiarong Xu, Xiang Bai, Xuanjing Huang, Zhongyu Wei*

  We propose Multiple Experts Fine-tuning Framework to build a financial large
language model (LLM), DISC-FinLLM. Our methodology improves general LLMs by
endowing them with multi-turn question answering abilities, domain text
processing capabilities, mathematical computation skills, and
retrieval-enhanced generation capabilities. We build a financial
instruction-tuning dataset named DISC-FIN-SFT, including instruction samples of
four categories (consulting, NLP tasks, computing and retrieval-augmented
generation). Evaluations conducted on multiple benchmarks demonstrate that our
model performs better than baseline models in various financial scenarios.
Further resources can be found at https://github.com/FudanDISC/DISC-FinLLM.


---

**[33. [2303.15324] Can Large Language Models design a Robot?](https://arxiv.org/pdf/2303.15324.pdf)** (2023-03-28)

*Francesco Stella, Cosimo Della Santina, Josie Hughes*

  Large Language Models can lead researchers in the design of robots.


---

**[34. [2406.04244] Benchmark Data Contamination of Large Language Models: A Survey](https://arxiv.org/pdf/2406.04244.pdf)** (2024-06-07)

*Cheng Xu, Shuhao Guan, Derek Greene, M-Tahar Kechadi*

  The rapid development of Large Language Models (LLMs) like GPT-4, Claude-3,
and Gemini has transformed the field of natural language processing. However,
it has also resulted in a significant issue known as Benchmark Data
Contamination (BDC). This occurs when language models inadvertently incorporate
evaluation benchmark information from their training data, leading to
inaccurate or unreliable performance during the evaluation phase of the
process. This paper reviews the complex challenge of BDC in LLM evaluation and
explores alternative assessment methods to mitigate the risks associated with
traditional benchmarks. The paper also examines challenges and future
directions in mitigating BDC risks, highlighting the complexity of the issue
and the need for innovative solutions to ensure the reliability of LLM
evaluation in real-world applications.


---

**[35. [2405.15383] Generating Code World Models with Large Language Models Guided by Monte
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

**[36. [2501.02018] Safeguarding Large Language Models in Real-time with Tunable
  Safety-Performance Trade-offs](https://arxiv.org/pdf/2501.02018.pdf)** (2025-01-07)

*Joao Fonseca, Andrew Bell, Julia Stoyanovich*

  Large Language Models (LLMs) have been shown to be susceptible to jailbreak
attacks, or adversarial attacks used to illicit high risk behavior from a
model. Jailbreaks have been exploited by cybercriminals and blackhat actors to
cause significant harm, highlighting the critical need to safeguard
widely-deployed models. Safeguarding approaches, which include fine-tuning
models or having LLMs "self-reflect", may lengthen the inference time of a
model, incur a computational penalty, reduce the semantic fluency of an output,
and restrict ``normal'' model behavior. Importantly, these Safety-Performance
Trade-offs (SPTs) remain an understudied area. In this work, we introduce a
novel safeguard, called SafeNudge, that combines Controlled Text Generation
with "nudging", or using text interventions to change the behavior of a model.
SafeNudge triggers during text-generation while a jailbreak attack is being
executed, and can reduce successful jailbreak attempts by 30% by guiding the
LLM towards a safe responses. It adds minimal latency to inference and has a
negligible impact on the semantic fluency of outputs. Further, we allow for
tunable SPTs. SafeNudge is open-source and available through https://pypi.org/,
and is compatible with models loaded with the Hugging Face "transformers"
library.


---

**[37. [2312.07910] PromptBench: A Unified Library for Evaluation of Large Language Models](https://arxiv.org/pdf/2312.07910.pdf)** (2024-08-21)

*Kaijie Zhu, Qinlin Zhao, Hao Chen, Jindong Wang, Xing Xie*

  The evaluation of large language models (LLMs) is crucial to assess their
performance and mitigate potential security risks. In this paper, we introduce
PromptBench, a unified library to evaluate LLMs. It consists of several key
components that are easily used and extended by researchers: prompt
construction, prompt engineering, dataset and model loading, adversarial prompt
attack, dynamic evaluation protocols, and analysis tools. PromptBench is
designed to be an open, general, and flexible codebase for research purposes
that can facilitate original study in creating new benchmarks, deploying
downstream applications, and designing new evaluation protocols. The code is
available at: https://github.com/microsoft/promptbench and will be continuously
supported.


---

**[38. [2503.13505] Ensemble Learning for Large Language Models in Text and Code Generation:
  A Survey](https://arxiv.org/pdf/2503.13505.pdf)** (2025-03-19)

*Mari Ashiga, Wei Jie, Fan Wu, Vardan Voskanyan, Fateme Dinmohammadi, Paul Brookes, Jingzhi Gong, Zheng Wang*

  Generative pretrained transformers (GPT) are the common large language models
(LLMs) used for generating text from natural language inputs. However, the
fixed properties of language parameters in individual LLMs can lead to
inconsistencies in the generated outputs. This limitation also restricts the
models' ability to represent diverse language patterns due to inherent biases.
Moreover, many powerful LLMs are closed-source. This prevents organizations
from integrating their data into these systems, raising concerns about data
privacy and limiting industry applications. Inspired by the successful
application of LLM ensemble models in text generation, recent literature has
also investigated their potential in code generation. This article reviews
these emerging LLM ensemble approaches. Our goal is to enhance readers'
understanding of existing techniques and encourage further research and
practical implementation, aiming to expand the real-world applications of LLM
ensemble models in both text and code generation. We categorize these
approaches into seven main methods: weight merging, knowledge fusion, mixture
of experts, reward ensemble, output ensemble, routing, and cascading. From this
list, we focus on four methods and models that show strong performance and
potential for broader applications. We analyze their modeling steps, training
methods, and output features to provide a clear understanding of their
capabilities. Our findings highlight the benefits of LLM ensemble techniques.
These include better representation of diversity, improved output quality, and
greater flexibility in applications. This information offers valuable insights
for selecting models for various real-world tasks involving text and code
generation, and potentially applying methods to multimodal LLMs.


---

**[39. [2305.06599] Structured Chain-of-Thought Prompting for Code Generation](https://arxiv.org/pdf/2305.06599.pdf)** (2023-09-08)

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

**[40. [2501.18657] Enhancing Large Language Model Efficiencyvia Symbolic Compression: A
  Formal Approach Towards Interpretability](https://arxiv.org/pdf/2501.18657.pdf)** (2025-02-03)

*Lumen AI, Tengzhou No. 1 Middle School, Shihao Ji, Zihui Song, Fucheng Zhong, Jisen Jia, Zhaobo Wu, Zheyi Cao, Tianhao Xu*

  Large language models (LLMs) face significant token efficiency bottlenecks in
code generation and logical reasoning tasks, a challenge that directly impacts
inference cost and model interpretability. This paper proposes a formal
framework based on symbolic compression,integrating combinatory logic,
information-theoretic optimal encoding, and context-aware inference techniques
to achieve a step-change improvement in token efficiency while preserving
semantic integrity. We establish a mathematical framework within a functional
programming paradigm, derive the quantitative relationship between symbolic
density and model interpretability, and propose a differentiable compression
factor metric to evaluate encoding efficiency. Furthermore, we leverage
parameter-efficient fine-tuning (PEFT) techniques to achieve a low-cost
application of the GAEL language. Experimental results show that this method
achieves a 78.3% token compression rate in code generation tasks while
improving logical traceability by 62% through structural explicitness. This
research provides new theoretical tools for efficient inference in LLMs and
opens a symbolic path for modelinterpretability research.


---

**[41. [2307.14936] PanGu-Coder2: Boosting Large Language Models for Code with Ranking
  Feedback](https://arxiv.org/pdf/2307.14936.pdf)** (2023-07-28)

*Bo Shen, Jiaxin Zhang, Taihong Chen, Daoguang Zan, Bing Geng, An Fu, Muhan Zeng, Ailun Yu, Jichuan Ji, Jingyang Zhao, Yuenan Guo, Qianxiang Wang*

  Large Language Models for Code (Code LLM) are flourishing. New and powerful
models are released on a weekly basis, demonstrating remarkable performance on
the code generation task. Various approaches have been proposed to boost the
code generation performance of pre-trained Code LLMs, such as supervised
fine-tuning, instruction tuning, reinforcement learning, etc. In this paper, we
propose a novel RRTF (Rank Responses to align Test&Teacher Feedback) framework,
which can effectively and efficiently boost pre-trained large language models
for code generation. Under this framework, we present PanGu-Coder2, which
achieves 62.20% pass@1 on the OpenAI HumanEval benchmark. Furthermore, through
an extensive evaluation on CoderEval and LeetCode benchmarks, we show that
PanGu-Coder2 consistently outperforms all previous Code LLMs.


---

**[42. [2406.04712] AICoderEval: Improving AI Domain Code Generation of Large Language
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

**[43. [2409.02026] Foundations of Large Language Model Compression -- Part 1: Weight
  Quantization](https://arxiv.org/pdf/2409.02026.pdf)** (2024-10-04)

*Sean I. Young*

  In recent years, compression of large language models (LLMs) has emerged as
an important problem to enable language model deployment on
resource-constrained devices, reduce computational costs, and mitigate the
environmental footprint of large-scale AI infrastructure. In this paper, we lay
down the foundation for LLM quantization from a convex optimization perspective
and propose a quantization technique that builds on this foundation for optimum
quantization outcomes. Our quantization framework, CVXQ, scales to models
containing hundreds of billions of weight parameters and provides users with
the flexibility to compress models to any specified model size, post-training.
A reference implementation of CVXQ can be obtained from github.com/seannz/cvxq.


---

**[44. [2502.20285] Conformal Tail Risk Control for Large Language Model Alignment](https://arxiv.org/pdf/2502.20285.pdf)** (2025-02-28)

*Catherine Yu-Chi Chen, Jingyan Shen, Zhun Deng, Lihua Lei*

  Recent developments in large language models (LLMs) have led to their
widespread usage for various tasks. The prevalence of LLMs in society implores
the assurance on the reliability of their performance. In particular,
risk-sensitive applications demand meticulous attention to unexpectedly poor
outcomes, i.e., tail events, for instance, toxic answers, humiliating language,
and offensive outputs. Due to the costly nature of acquiring human annotations,
general-purpose scoring models have been created to automate the process of
quantifying these tail events. This phenomenon introduces potential
human-machine misalignment between the respective scoring mechanisms. In this
work, we present a lightweight calibration framework for blackbox models that
ensures the alignment of humans and machines with provable guarantees. Our
framework provides a rigorous approach to controlling any distortion risk
measure that is characterized by a weighted average of quantiles of the loss
incurred by the LLM with high confidence. The theoretical foundation of our
method relies on the connection between conformal risk control and a
traditional family of statistics, i.e., L-statistics. To demonstrate the
utility of our framework, we conduct comprehensive experiments that address the
issue of human-machine misalignment.


---

**[45. [2503.16252] Fin-R1: A Large Language Model for Financial Reasoning through
  Reinforcement Learning](https://arxiv.org/pdf/2503.16252.pdf)** (2025-03-24)

*Zhaowei Liu, Xin Guo, Fangqi Lou, Lingfeng Zeng, Jinyi Niu, Zixuan Wang, Jiajie Xu, Weige Cai, Ziwei Yang, Xueqian Zhao, Chao Li, Sheng Xu, Dezhi Chen, Yun Chen, Zuo Bai, Liwen Zhang*

  Reasoning large language models are rapidly evolving across various domains.
However, their capabilities in handling complex financial tasks still require
in-depth exploration. In this paper, we introduce Fin-R1, a reasoning large
language model specifically designed for the financial sector. Fin-R1 is built
using a two-stage architecture, leveraging a financial reasoning dataset
distilled and processed based on DeepSeek-R1. Through supervised fine-tuning
(SFT) and reinforcement learning (RL) training, it demonstrates performance
close to DeepSeek-R1 with a parameter size of 7 billion across a range of
financial reasoning tasks. It achieves the state-of-the-art (SOTA) in the FinQA
and ConvFinQA tasks between those LLMs in our evaluation, surpassing larger
models in other tasks as well. Fin-R1 showcases strong reasoning and
decision-making capabilities, providing solutions to various problems
encountered in the financial domain. Our code is available at
https://github.com/SUFE-AIFLM-Lab/Fin-R1.


---

**[46. [2310.05736] LLMLingua: Compressing Prompts for Accelerated Inference of Large
  Language Models](https://arxiv.org/pdf/2310.05736.pdf)** (2023-12-07)

*Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, Lili Qiu*

  Large language models (LLMs) have been applied in various applications due to
their astonishing capabilities. With advancements in technologies such as
chain-of-thought (CoT) prompting and in-context learning (ICL), the prompts fed
to LLMs are becoming increasingly lengthy, even exceeding tens of thousands of
tokens. To accelerate model inference and reduce cost, this paper presents
LLMLingua, a coarse-to-fine prompt compression method that involves a budget
controller to maintain semantic integrity under high compression ratios, a
token-level iterative compression algorithm to better model the interdependence
between compressed contents, and an instruction tuning based method for
distribution alignment between language models. We conduct experiments and
analysis over four datasets from different scenarios, i.e., GSM8K, BBH,
ShareGPT, and Arxiv-March23; showing that the proposed approach yields
state-of-the-art performance and allows for up to 20x compression with little
performance loss. Our code is available at https://aka.ms/LLMLingua.


---

**[47. [2407.12504] Case2Code: Scalable Synthetic Data for Code Generation](https://arxiv.org/pdf/2407.12504.pdf)** (2025-02-11)

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

**[48. [2502.12601] COPU: Conformal Prediction for Uncertainty Quantification in Natural
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

**[49. [2503.01245] Large Language Models for Code Generation: A Comprehensive Survey of
  Challenges, Techniques, Evaluation, and Applications](https://arxiv.org/pdf/2503.01245.pdf)** (2025-04-03)

*Nam Huynh, Beiyu Lin*

  Large Language Models (LLMs) have demonstrated their remarkable capabilities
in numerous fields. This survey focuses on how LLMs empower users, regardless
of their technical background, to use human languages to automatically generate
executable code. We begin with understanding LLMs' limitations and challenges
in automated code generation. Subsequently, we review various fine-tuning
techniques designed to enhance both the performance and adaptability of LLMs in
code generation tasks. We then review the existing metrics and benchmarks for
evaluations to assess model performance based on fine-tuning techniques.
Finally, we explore the applications of LLMs (e.g. CodeLlama, GitHub Copilot,
ToolGen) in code generation tasks to illustrate their roles and
functionalities. This survey provides a comprehensive overview of LLMs for code
generation, helps researchers in diverse fields better understand the current
state-of-the-art technologies, and offers the potential of effectively
leveraging LLMs for code generation tasks.


---

**[50. [2309.16621] Stress Testing Chain-of-Thought Prompting for Large Language Models](https://arxiv.org/pdf/2309.16621.pdf)** (2023-09-29)

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

**[51. [2407.00499] ConU: Conformal Uncertainty in Large Language Models with Correctness
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

**[52. [2408.04392] Open-domain Implicit Format Control for Large Language Model Generation](https://arxiv.org/pdf/2408.04392.pdf)** (2024-08-09)

*Yiqun Yao, Wenjia Ma, Xuezhi Fang, Xin Jiang, Xiang Li, Xuying Meng, Peng Han, Jing Li, Aixin Sun, Yequan Wang*

  Controlling the format of outputs generated by large language models (LLMs)
is a critical functionality in various applications. Current methods typically
employ constrained decoding with rule-based automata or fine-tuning with
manually crafted format instructions, both of which struggle with open-domain
format requirements. To address this limitation, we introduce a novel framework
for controlled generation in LLMs, leveraging user-provided, one-shot QA pairs.
This study investigates LLMs' capabilities to follow open-domain, one-shot
constraints and replicate the format of the example answers. We observe that
this is a non-trivial problem for current LLMs. We also develop a dataset
collection methodology for supervised fine-tuning that enhances the open-domain
format control of LLMs without degrading output quality, as well as a benchmark
on which we evaluate both the helpfulness and format correctness of LLM
outputs. The resulting datasets, named OIFC-SFT, along with the related code,
will be made publicly available at https://github.com/cofe-ai/OIFC.


---

**[53. [2502.20747] Measuring Determinism in Large Language Models for Software Code Review](https://arxiv.org/pdf/2502.20747.pdf)** (2025-03-03)

*Eugene Klishevich, Yegor Denisov-Blanch, Simon Obstbaum, Igor Ciobanu, Michal Kosinski*

  Large Language Models (LLMs) promise to streamline software code reviews, but
their ability to produce consistent assessments remains an open question. In
this study, we tested four leading LLMs -- GPT-4o mini, GPT-4o, Claude 3.5
Sonnet, and LLaMA 3.2 90B Vision -- on 70 Java commits from both private and
public repositories. By setting each model's temperature to zero, clearing
context, and repeating the exact same prompts five times, we measured how
consistently each model generated code-review assessments. Our results reveal
that even with temperature minimized, LLM responses varied to different
degrees. These findings highlight a consideration about the inherently limited
consistency (test-retest reliability) of LLMs -- even when the temperature is
set to zero -- and the need for caution when using LLM-generated code reviews
to make real-world decisions.


---

**[54. [2410.00361] PclGPT: A Large Language Model for Patronizing and Condescending
  Language Detection](https://arxiv.org/pdf/2410.00361.pdf)** (2024-10-02)

*Hongbo Wang, Mingda Li, Junyu Lu, Hebin Xia, Liang Yang, Bo Xu, Ruizhu Liu, Hongfei Lin*

  Disclaimer: Samples in this paper may be harmful and cause discomfort!
  Patronizing and condescending language (PCL) is a form of speech directed at
vulnerable groups. As an essential branch of toxic language, this type of
language exacerbates conflicts and confrontations among Internet communities
and detrimentally impacts disadvantaged groups. Traditional pre-trained
language models (PLMs) perform poorly in detecting PCL due to its implicit
toxicity traits like hypocrisy and false sympathy. With the rise of large
language models (LLMs), we can harness their rich emotional semantics to
establish a paradigm for exploring implicit toxicity. In this paper, we
introduce PclGPT, a comprehensive LLM benchmark designed specifically for PCL.
We collect, annotate, and integrate the Pcl-PT/SFT dataset, and then develop a
bilingual PclGPT-EN/CN model group through a comprehensive pre-training and
supervised fine-tuning staircase process to facilitate implicit toxic
detection. Group detection results and fine-grained detection from PclGPT and
other models reveal significant variations in the degree of bias in PCL towards
different vulnerable groups, necessitating increased societal attention to
protect them.


---

**[55. [2312.06149] Unlocking Anticipatory Text Generation: A Constrained Approach for Large
  Language Models Decoding](https://arxiv.org/pdf/2312.06149.pdf)** (2024-10-07)

*Lifu Tu, Semih Yavuz, Jin Qu, Jiacheng Xu, Rui Meng, Caiming Xiong, Yingbo Zhou*

  Large Language Models (LLMs) have demonstrated a powerful ability for text
generation. However, achieving optimal results with a given prompt or
instruction can be challenging, especially for billion-sized models.
Additionally, undesired behaviors such as toxicity or hallucinations can
manifest. While much larger models (e.g., ChatGPT) may demonstrate strength in
mitigating these issues, there is still no guarantee of complete prevention. In
this work, we propose formalizing text generation as a future-constrained
generation problem to minimize undesirable behaviors and enforce faithfulness
to instructions. The estimation of future constraint satisfaction, accomplished
using LLMs, guides the text generation process. Our extensive experiments
demonstrate the effectiveness of the proposed approach across three distinct
text generation tasks: keyword-constrained generation (Lin et al., 2020),
toxicity reduction (Gehman et al., 2020), and factual correctness in
question-answering (Gao et al., 2023).


---

**[56. [2411.01414] A Deep Dive Into Large Language Model Code Generation Mistakes: What and
  Why?](https://arxiv.org/pdf/2411.01414.pdf)** (2025-03-21)

*QiHong Chen, Jiachen Yu, Jiawei Li, Jiecheng Deng, Justin Tian Jin Chen, Iftekhar Ahmed*

  Recent advancements in Large Language Models (LLMs) have led to their
widespread application in automated code generation. However, these models can
still generate defective code that deviates from the specification. Previous
research has mainly focused on the mistakes in LLM-generated standalone
functions, overlooking real-world software development situations where the
successful generation of the code requires software contexts such as external
dependencies. In this paper, we considered both of these code generation
situations and identified a range of \textit{non-syntactic mistakes} arising
from LLMs' misunderstandings of coding question specifications. Seven
categories of non-syntactic mistakes were identified through extensive manual
analyses, four of which were missed by previous works. To better understand
these mistakes, we proposed six reasons behind these mistakes from various
perspectives. Moreover, we explored the effectiveness of LLMs in detecting
mistakes and their reasons. Our evaluation demonstrated that GPT-4 with the
ReAct prompting technique can achieve an F1 score of up to 0.65 when
identifying reasons for LLM's mistakes, such as misleading function signatures.
We believe that these findings offer valuable insights into enhancing the
quality of LLM-generated code.


---

**[57. [2412.04947] C$^2$LEVA: Toward Comprehensive and Contamination-Free Language Model
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

**[58. [2403.12503] Securing Large Language Models: Threats, Vulnerabilities and Responsible
  Practices](https://arxiv.org/pdf/2403.12503.pdf)** (2024-03-20)

*Sara Abdali, Richard Anarfi, CJ Barberan, Jia He*

  Large language models (LLMs) have significantly transformed the landscape of
Natural Language Processing (NLP). Their impact extends across a diverse
spectrum of tasks, revolutionizing how we approach language understanding and
generations. Nevertheless, alongside their remarkable utility, LLMs introduce
critical security and risk considerations. These challenges warrant careful
examination to ensure responsible deployment and safeguard against potential
vulnerabilities. This research paper thoroughly investigates security and
privacy concerns related to LLMs from five thematic perspectives: security and
privacy concerns, vulnerabilities against adversarial attacks, potential harms
caused by misuses of LLMs, mitigation strategies to address these challenges
while identifying limitations of current strategies. Lastly, the paper
recommends promising avenues for future research to enhance the security and
risk management of LLMs.


---

**[59. [2501.17584] GLLM: Self-Corrective G-Code Generation using Large Language Models with
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

**[60. [2407.02402] Assessing the Code Clone Detection Capability of Large Language Models](https://arxiv.org/pdf/2407.02402.pdf)** (2024-07-03)

*Zixian Zhang, Takfarinas Saber*

  This study aims to assess the performance of two advanced Large Language
Models (LLMs), GPT-3.5 and GPT-4, in the task of code clone detection. The
evaluation involves testing the models on a variety of code pairs of different
clone types and levels of similarity, sourced from two datasets: BigCloneBench
(human-made) and GPTCloneBench (LLM-generated). Findings from the study
indicate that GPT-4 consistently surpasses GPT-3.5 across all clone types. A
correlation was observed between the GPTs' accuracy at identifying code clones
and code similarity, with both GPT models exhibiting low effectiveness in
detecting the most complex Type-4 code clones. Additionally, GPT models
demonstrate a higher performance identifying code clones in LLM-generated code
compared to humans-generated code. However, they do not reach impressive
accuracy. These results emphasize the imperative for ongoing enhancements in
LLM capabilities, particularly in the recognition of code clones and in
mitigating their predisposition towards self-generated code clones--which is
likely to become an issue as software engineers are more numerous to leverage
LLM-enabled code generation and code refactoring tools.


---

**[61. [2405.20132] LLaMEA: A Large Language Model Evolutionary Algorithm for Automatically
  Generating Metaheuristics](https://arxiv.org/pdf/2405.20132.pdf)** (2025-01-31)

*Niki van Stein, Thomas Bäck*

  Large Language Models (LLMs) such as GPT-4 have demonstrated their ability to
understand natural language and generate complex code snippets. This paper
introduces a novel Large Language Model Evolutionary Algorithm (LLaMEA)
framework, leveraging GPT models for the automated generation and refinement of
algorithms. Given a set of criteria and a task definition (the search space),
LLaMEA iteratively generates, mutates and selects algorithms based on
performance metrics and feedback from runtime evaluations. This framework
offers a unique approach to generating optimized algorithms without requiring
extensive prior expertise. We show how this framework can be used to generate
novel black-box metaheuristic optimization algorithms automatically. LLaMEA
generates multiple algorithms that outperform state-of-the-art optimization
algorithms (Covariance Matrix Adaptation Evolution Strategy and Differential
Evolution) on the five dimensional black box optimization benchmark (BBOB). The
algorithms also show competitive performance on the 10- and 20-dimensional
instances of the test functions, although they have not seen such instances
during the automated generation process. The results demonstrate the
feasibility of the framework and identify future directions for automated
generation and optimization of algorithms via LLMs.


---

**[62. [2403.03883] SaulLM-7B: A pioneering Large Language Model for Law](https://arxiv.org/pdf/2403.03883.pdf)** (2024-03-08)

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

**[63. [2503.13733] CoDet-M4: Detecting Machine-Generated Code in Multi-Lingual,
  Multi-Generator and Multi-Domain Settings](https://arxiv.org/pdf/2503.13733.pdf)** (2025-03-19)

*Daniil Orel, Dilshod Azizov, Preslav Nakov*

  Large language models (LLMs) have revolutionized code generation, automating
programming with remarkable efficiency. However, these advancements challenge
programming skills, ethics, and assessment integrity, making the detection of
LLM-generated code essential for maintaining accountability and standards.
While, there has been some research on this problem, it generally lacks domain
coverage and robustness, and only covers a small number of programming
languages. To this end, we propose a framework capable of distinguishing
between human- and LLM-written code across multiple programming languages, code
generators, and domains. We use a large-scale dataset from renowned platforms
and LLM-based code generators, alongside applying rigorous data quality checks,
feature engineering, and comparative analysis using evaluation of traditional
machine learning models, pre-trained language models (PLMs), and LLMs for code
detection. We perform an evaluation on out-of-domain scenarios, such as
detecting the authorship and hybrid authorship of generated code and
generalizing to unseen models, domains, and programming languages. Moreover,
our extensive experiments show that our framework effectively distinguishes
human- from LLM-written code and sets a new benchmark for this task.


---

**[64. [2402.05624] Efficient Models for the Detection of Hate, Abuse and Profanity](https://arxiv.org/pdf/2402.05624.pdf)** (2024-02-09)

*Christoph Tillmann, Aashka Trivedi, Bishwaranjan Bhattacharjee*

  Large Language Models (LLMs) are the cornerstone for many Natural Language
Processing (NLP) tasks like sentiment analysis, document classification, named
entity recognition, question answering, summarization, etc. LLMs are often
trained on data which originates from the web. This data is prone to having
content with Hate, Abuse and Profanity (HAP). For a detailed definition of HAP,
please refer to the Appendix. Due to the LLMs being exposed to HAP content
during training, the models learn it and may then generate hateful or profane
content. For example, when the open-source RoBERTa model (specifically, the
RoBERTA base model) from the HuggingFace (HF) Transformers library is prompted
to replace the mask token in `I do not know that Persian people are that MASK`
it returns the word `stupid` with the highest score. This is unacceptable in
civil discourse.The detection of Hate, Abuse and Profanity in text is a vital
component of creating civil and unbiased LLMs, which is needed not only for
English, but for all languages. In this article, we briefly describe the
creation of HAP detectors and various ways of using them to make models civil
and acceptable in the output they generate.


---

**[65. [2308.10443] Using Large Language Models for Cybersecurity Capture-The-Flag
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

**[66. [2412.06843] Semantic Loss Guided Data Efficient Supervised Fine Tuning for Safe
  Responses in LLMs](https://arxiv.org/pdf/2412.06843.pdf)** (2024-12-12)

*Yuxiao Lu, Arunesh Sinha, Pradeep Varakantham*

  Large Language Models (LLMs) generating unsafe responses to toxic prompts is
a significant issue in their applications. While various efforts aim to address
this safety concern, previous approaches often demand substantial human data
collection or rely on the less dependable option of using another LLM to
generate corrective data. In this paper, we aim to take this problem and
overcome limitations of requiring significant high-quality human data. Our
method requires only a small set of unsafe responses to toxic prompts, easily
obtained from the unsafe LLM itself. By employing a semantic cost combined with
a negative Earth Mover Distance (EMD) loss, we guide the LLM away from
generating unsafe responses. Additionally, we propose a novel lower bound for
EMD loss, enabling more efficient optimization. Our results demonstrate
superior performance and data efficiency compared to baselines, and we further
examine the nuanced effects of over-alignment and potential degradation of
language capabilities when using contrastive data.


---

**[67. [2406.00244] Controlling Large Language Model Agents with Entropic Activation
  Steering](https://arxiv.org/pdf/2406.00244.pdf)** (2024-10-14)

*Nate Rahn, Pierluca D'Oro, Marc G. Bellemare*

  The rise of large language models (LLMs) has prompted increasing interest in
their use as in-context learning agents. At the core of agentic behavior is the
capacity for exploration, or the ability to actively gather information about
the environment. But how do LLM agents explore, and how can we control their
exploratory behaviors? To answer these questions, we take a
representation-level perspective, and introduce Entropic Activation Steering
(EAST), an activation steering method for in-context LLM agents. Firstly, we
demonstrate that EAST can effectively manipulate an LLM agent's exploration by
directly affecting the high-level actions parsed from the outputs of the LLM,
in contrast to token-level temperature sampling. Secondly, we reveal how
applying this control modulates the uncertainty exhibited in the LLM's
thoughts, guiding the agent towards more exploratory actions. Finally, we
demonstrate that the steering vectors obtained by EAST generalize across task
variants. In total, these results show that LLM agents explicitly encode
uncertainty over their actions in their representation space. Our work paves
the way for a new understanding of the functioning of LLM agents and to
effective control of their decision-making behaviors.


---

**[68. [2404.05499] Guiding Large Language Models to Generate Computer-Parsable Content](https://arxiv.org/pdf/2404.05499.pdf)** (2024-04-23)

*Jiaye Wang*

  We propose a method to guide Large Language Models (LLMs) in generating
structured content adhering to specific conventions without fine-tuning. By
utilizing coroutine-based content generation constraints through a pre-agreed
context-free grammar (CFG), LLMs are directed during decoding to produce formal
language compliant outputs. This enhances stability and consistency in
generating target data structures, types, or instructions, reducing application
development complexities. Experimentally, error rates of GPT-2 and Gemma exceed
95% for DSLs longer than 36 and 282 tokens, respectively. We introduce
YieldLang, a coroutine-based DSL generation framework, and evaluate it with
LLMs on various tasks including JSON and Mermaid flowchart generation. Compared
to benchmarks, our approach improves accuracy by 1.09 to 11.6 times, with LLMs
requiring only about 16.5% of the samples to generate JSON effectively. This
enhances usability of LLM-generated content for computer programs.


---

**[69. [2411.18010] JPPO: Joint Power and Prompt Optimization for Accelerated Large Language
  Model Services](https://arxiv.org/pdf/2411.18010.pdf)** (2025-02-25)

*Feiran You, Hongyang Du, Kaibin Huang, Abbas Jamalipour*

  Large Language Models (LLMs) have demonstrated remarkable capabilities in
various tasks, leading to their increasing deployment in wireless networks for
a wide variety of user services. However, the growing longer prompt setting
highlights the crucial issue of computational resource demands and huge
communication load. To address this challenge, we propose Joint Power and
Prompt Optimization (JPPO), a framework that combines Small Language Model
(SLM)-based prompt compression with wireless power allocation optimization. By
deploying SLM at user devices for prompt compression and employing Deep
Reinforcement Learning for joint optimization of compression ratio and
transmission power, JPPO effectively balances service quality with resource
efficiency. Experimental results demonstrate that our framework achieves high
service fidelity and low bit error rates while optimizing power usage in
wireless LLM services. The system reduces response time by about 17%, with the
improvement varying based on the length of the original prompt.


---

**[70. [2504.12357] Replicating ReLM Results: Validating Large Language Models with ReLM](https://arxiv.org/pdf/2504.12357.pdf)** (2025-04-18)

*Reece Adamson, Erin Song*

  Validating Large Language Models with ReLM explores the application of formal
languages to evaluate and control Large Language Models (LLMs) for
memorization, bias, and zero-shot performance. Current approaches for
evaluating these types behavior are often slow, imprecise, costly, or introduce
biases of their own, but are necessary due to the importance of this behavior
when productionizing LLMs. This project reproduces key results from the
original ReLM paper and expounds on the approach and applications with an
emphasis on the relevance to the field of systems for machine learning.


---

**[71. [2502.16691] Toward Responsible Federated Large Language Models: Leveraging a Safety
  Filter and Constitutional AI](https://arxiv.org/pdf/2502.16691.pdf)** (2025-02-25)

*Eunchung Noh, Jeonghun Baek*

  Recent research has increasingly focused on training large language models
(LLMs) using federated learning, known as FedLLM. However, responsible AI
(RAI), which aims to ensure safe responses, remains underexplored in the
context of FedLLM. In FedLLM, client data used for training may contain harmful
content, leading to unsafe LLMs that generate harmful responses. Aggregating
such unsafe LLMs into the global model and distributing them to clients may
result in the widespread deployment of unsafe LLMs. To address this issue, we
incorporate two well-known RAI methods into FedLLM: the safety filter and
constitutional AI. Our experiments demonstrate that these methods significantly
enhance the safety of the LLM, achieving over a 20% improvement on AdvBench, a
benchmark for evaluating safety performance.


---

**[72. [2310.03951] Chain of Natural Language Inference for Reducing Large Language Model
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

**[73. [2405.06650] Large Language Models as Planning Domain Generators](https://arxiv.org/pdf/2405.06650.pdf)** (2024-05-14)

*James Oswald, Kavitha Srinivas, Harsha Kokel, Junkyu Lee, Michael Katz, Shirin Sohrabi*

  Developing domain models is one of the few remaining places that require
manual human labor in AI planning. Thus, in order to make planning more
accessible, it is desirable to automate the process of domain model generation.
To this end, we investigate if large language models (LLMs) can be used to
generate planning domain models from simple textual descriptions. Specifically,
we introduce a framework for automated evaluation of LLM-generated domains by
comparing the sets of plans for domain instances. Finally, we perform an
empirical analysis of 7 large language models, including coding and chat models
across 9 different planning domains, and under three classes of natural
language domain descriptions. Our results indicate that LLMs, particularly
those with high parameter counts, exhibit a moderate level of proficiency in
generating correct planning domains from natural language descriptions. Our
code is available at https://github.com/IBM/NL2PDDL.


---

**[74. [2405.03170] Oracle-Checker Scheme for Evaluating a Generative Large Language Model](https://arxiv.org/pdf/2405.03170.pdf)** (2024-05-07)

*Yueling Jenny Zeng, Li-C. Wang, Thomas Ibbetson*

  This work presents a novel approach called oracle-checker scheme for
evaluating the answer given by a generative large language model (LLM). Two
types of checkers are presented. The first type of checker follows the idea of
property testing. The second type of checker follows the idea of program
checking. Their applications are demonstrated in two separate contexts, entity
extraction and paraphrase decision, respectively.


---

**[75. [2410.15037] mHumanEval -- A Multilingual Benchmark to Evaluate Large Language Models
  for Code Generation](https://arxiv.org/pdf/2410.15037.pdf)** (2025-01-28)

*Nishat Raihan, Antonios Anastasopoulos, Marcos Zampieri*

  Recent advancements in large language models (LLMs) have significantly
enhanced code generation from natural language prompts. The HumanEval
Benchmark, developed by OpenAI, remains the most widely used code generation
benchmark. However, this and other Code LLM benchmarks face critical
limitations, particularly in task diversity, test coverage, and linguistic
scope. Current evaluations primarily focus on English-to-Python conversion
tasks with limited test cases, potentially overestimating model performance.
While recent works have addressed test coverage and programming language (PL)
diversity, code generation from low-resource language prompts remains largely
unexplored. To address this gap, we introduce mHumanEval, an extended benchmark
supporting prompts in over 200 natural languages. We employ established machine
translation methods to compile the benchmark, coupled with a quality assurance
process. Furthermore, we provide expert human translations for 15 diverse
natural languages (NLs). We conclude by analyzing the multilingual code
generation capabilities of state-of-the-art (SOTA) Code LLMs, offering insights
into the current landscape of cross-lingual code generation.


---

**[76. [2405.02828] Trojans in Large Language Models of Code: A Critical Review through a
  Trigger-Based Taxonomy](https://arxiv.org/pdf/2405.02828.pdf)** (2024-05-07)

*Aftab Hussain, Md Rafiqul Islam Rabin, Toufique Ahmed, Bowen Xu, Premkumar Devanbu, Mohammad Amin Alipour*

  Large language models (LLMs) have provided a lot of exciting new capabilities
in software development. However, the opaque nature of these models makes them
difficult to reason about and inspect. Their opacity gives rise to potential
security risks, as adversaries can train and deploy compromised models to
disrupt the software development process in the victims' organization.
  This work presents an overview of the current state-of-the-art trojan attacks
on large language models of code, with a focus on triggers -- the main design
point of trojans -- with the aid of a novel unifying trigger taxonomy
framework. We also aim to provide a uniform definition of the fundamental
concepts in the area of trojans in Code LLMs. Finally, we draw implications of
findings on how code models learn on trigger design.


---

**[77. [2402.01722] Enhancing Large Language Model Performance To Answer Questions and
  Extract Information More Accurately](https://arxiv.org/pdf/2402.01722.pdf)** (2024-02-06)

*Liang Zhang, Katherine Jijo, Spurthi Setty, Eden Chung, Fatima Javid, Natan Vidra, Tommy Clifford*

  Large Language Models (LLMs) generate responses to questions; however, their
effectiveness is often hindered by sub-optimal quality of answers and
occasional failures to provide accurate responses to questions. To address
these challenges, a fine-tuning process is employed, involving feedback and
examples to refine models. The objective is to enhance AI models through
continuous feedback loops, utilizing metrics such as cosine similarity, LLM
evaluation and Rouge-L scores to evaluate the models. Leveraging LLMs like
GPT-3.5, GPT4ALL, and LLaMA2, and Claude, this approach is benchmarked on
financial datasets, including the FinanceBench and RAG Instruct Benchmark
Tester Dataset, illustrating the necessity of fine-tuning. The results showcase
the capability of fine-tuned models to surpass the accuracy of zero-shot LLMs,
providing superior question and answering capabilities. Notably, the
combination of fine-tuning the LLM with a process known as Retrieval Augmented
Generation (RAG) proves to generate responses with improved accuracy.


---

**[78. [2401.01620] Large Language Model Capabilities in Perioperative Risk Prediction and
  Prognostication](https://arxiv.org/pdf/2401.01620.pdf)** (2024-01-04)

*Philip Chung, Christine T Fong, Andrew M Walters, Nima Aghaeepour, Meliha Yetisgen, Vikas N O'Reilly-Shah*

  We investigate whether general-domain large language models such as GPT-4
Turbo can perform risk stratification and predict post-operative outcome
measures using a description of the procedure and a patient's clinical notes
derived from the electronic health record. We examine predictive performance on
8 different tasks: prediction of ASA Physical Status Classification, hospital
admission, ICU admission, unplanned admission, hospital mortality, PACU Phase 1
duration, hospital duration, and ICU duration. Few-shot and chain-of-thought
prompting improves predictive performance for several of the tasks. We achieve
F1 scores of 0.50 for ASA Physical Status Classification, 0.81 for ICU
admission, and 0.86 for hospital mortality. Performance on duration prediction
tasks were universally poor across all prompt strategies. Current generation
large language models can assist clinicians in perioperative risk
stratification on classification tasks and produce high-quality natural
language summaries and explanations.


---

**[79. [2012.00413] CPM: A Large-scale Generative Chinese Pre-trained Language Model](https://arxiv.org/pdf/2012.00413.pdf)** (2020-12-02)

*Zhengyan Zhang, Xu Han, Hao Zhou, Pei Ke, Yuxian Gu, Deming Ye, Yujia Qin, Yusheng Su, Haozhe Ji, Jian Guan, Fanchao Qi, Xiaozhi Wang, Yanan Zheng, Guoyang Zeng, Huanqi Cao, Shengqi Chen, Daixuan Li, Zhenbo Sun, Zhiyuan Liu, Minlie Huang, Wentao Han, Jie Tang, Juanzi Li, Xiaoyan Zhu, Maosong Sun*

  Pre-trained Language Models (PLMs) have proven to be beneficial for various
downstream NLP tasks. Recently, GPT-3, with 175 billion parameters and 570GB
training data, drew a lot of attention due to the capacity of few-shot (even
zero-shot) learning. However, applying GPT-3 to address Chinese NLP tasks is
still challenging, as the training corpus of GPT-3 is primarily English, and
the parameters are not publicly available. In this technical report, we release
the Chinese Pre-trained Language Model (CPM) with generative pre-training on
large-scale Chinese training data. To the best of our knowledge, CPM, with 2.6
billion parameters and 100GB Chinese training data, is the largest Chinese
pre-trained language model, which could facilitate several downstream Chinese
NLP tasks, such as conversation, essay generation, cloze test, and language
understanding. Extensive experiments demonstrate that CPM achieves strong
performance on many NLP tasks in the settings of few-shot (even zero-shot)
learning. The code and parameters are available at
https://github.com/TsinghuaAI/CPM-Generate.


---

**[80. [2402.13720] Ouroboros: Generating Longer Drafts Phrase by Phrase for Faster
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

**[81. [2502.19361] Can Large Language Models Detect Errors in Long Chain-of-Thought
  Reasoning?](https://arxiv.org/pdf/2502.19361.pdf)** (2025-04-01)

*Yancheng He, Shilong Li, Jiaheng Liu, Weixun Wang, Xingyuan Bu, Ge Zhang, Zhongyuan Peng, Zhaoxiang Zhang, Zhicheng Zheng, Wenbo Su, Bo Zheng*

  Recently, o1-like models have drawn significant attention, where these models
produce the long Chain-of-Thought (CoT) reasoning steps to improve the
reasoning abilities of existing Large Language Models (LLMs). In this paper, to
understand the qualities of these long CoTs and measure the critique abilities
of existing LLMs on these long CoTs, we introduce the DeltaBench, including the
generated long CoTs from different o1-like models (e.g., QwQ, DeepSeek-R1) for
different reasoning tasks (e.g., Math, Code, General Reasoning), to measure the
ability to detect errors in long CoT reasoning. Based on DeltaBench, we first
perform fine-grained analysis of the generated long CoTs to discover the
effectiveness and efficiency of different o1-like models. Then, we conduct
extensive evaluations of existing process reward models (PRMs) and critic
models to detect the errors of each annotated process, which aims to
investigate the boundaries and limitations of existing PRMs and critic models.
Finally, we hope that DeltaBench could guide developers to better understand
the long CoT reasoning abilities of their models.


---

**[82. [2306.08018] Mol-Instructions: A Large-Scale Biomolecular Instruction Dataset for
  Large Language Models](https://arxiv.org/pdf/2306.08018.pdf)** (2024-03-05)

*Yin Fang, Xiaozhuan Liang, Ningyu Zhang, Kangwei Liu, Rui Huang, Zhuo Chen, Xiaohui Fan, Huajun Chen*

  Large Language Models (LLMs), with their remarkable task-handling
capabilities and innovative outputs, have catalyzed significant advancements
across a spectrum of fields. However, their proficiency within specialized
domains such as biomolecular studies remains limited. To address this
challenge, we introduce Mol-Instructions, a comprehensive instruction dataset
designed for the biomolecular domain. Mol-Instructions encompasses three key
components: molecule-oriented instructions, protein-oriented instructions, and
biomolecular text instructions. Each component aims to improve the
understanding and prediction capabilities of LLMs concerning biomolecular
features and behaviors. Through extensive instruction tuning experiments on
LLMs, we demonstrate the effectiveness of Mol-Instructions in enhancing large
models' performance in the intricate realm of biomolecular studies, thus
fostering progress in the biomolecular research community. Mol-Instructions is
publicly available for ongoing research and will undergo regular updates to
enhance its applicability.


---

**[83. [2310.17838] Real-time Animation Generation and Control on Rigged Models via Large
  Language Models](https://arxiv.org/pdf/2310.17838.pdf)** (2024-02-16)

*Han Huang, Fernanda De La Torre, Cathy Mengying Fang, Andrzej Banburski-Fahey, Judith Amores, Jaron Lanier*

  We introduce a novel method for real-time animation control and generation on
rigged models using natural language input. First, we embed a large language
model (LLM) in Unity to output structured texts that can be parsed into diverse
and realistic animations. Second, we illustrate LLM's potential to enable
flexible state transition between existing animations. We showcase the
robustness of our approach through qualitative results on various rigged models
and motions.


---

**[84. [2502.14425] A Survey on Data Contamination for Large Language Models](https://arxiv.org/pdf/2502.14425.pdf)** (2025-02-21)

*Yuxing Cheng, Yi Chang, Yuan Wu*

  Recent advancements in Large Language Models (LLMs) have demonstrated
significant progress in various areas, such as text generation and code
synthesis. However, the reliability of performance evaluation has come under
scrutiny due to data contamination-the unintended overlap between training and
test datasets. This overlap has the potential to artificially inflate model
performance, as LLMs are typically trained on extensive datasets scraped from
publicly available sources. These datasets often inadvertently overlap with the
benchmarks used for evaluation, leading to an overestimation of the models'
true generalization capabilities. In this paper, we first examine the
definition and impacts of data contamination. Secondly, we review methods for
contamination-free evaluation, focusing on three strategies: data
updating-based methods, data rewriting-based methods, and prevention-based
methods. Specifically, we highlight dynamic benchmarks and LLM-driven
evaluation methods. Finally, we categorize contamination detecting methods
based on model information dependency: white-Box, gray-Box, and black-Box
detection approaches. Our survey highlights the requirements for more rigorous
evaluation protocols and proposes future directions for addressing data
contamination challenges.


---

**[85. [2411.09601] Accelerating Knowledge Graph and Ontology Engineering with Large
  Language Models](https://arxiv.org/pdf/2411.09601.pdf)** (2024-11-15)

*Cogan Shimizu, Pascal Hitzler*

  Large Language Models bear the promise of significant acceleration of key
Knowledge Graph and Ontology Engineering tasks, including ontology modeling,
extension, modification, population, alignment, as well as entity
disambiguation. We lay out LLM-based Knowledge Graph and Ontology Engineering
as a new and coming area of research, and argue that modular approaches to
ontologies will be of central importance.


---

**[86. [2406.13787] LIT: Large Language Model Driven Intention Tracking for Proactive
  Human-Robot Collaboration -- A Robot Sous-Chef Application](https://arxiv.org/pdf/2406.13787.pdf)** (2024-06-21)

*Zhe Huang, John Pohovey, Ananya Yammanuru, Katherine Driggs-Campbell*

  Large Language Models (LLM) and Vision Language Models (VLM) enable robots to
ground natural language prompts into control actions to achieve tasks in an
open world. However, when applied to a long-horizon collaborative task, this
formulation results in excessive prompting for initiating or clarifying robot
actions at every step of the task. We propose Language-driven Intention
Tracking (LIT), leveraging LLMs and VLMs to model the human user's long-term
behavior and to predict the next human intention to guide the robot for
proactive collaboration. We demonstrate smooth coordination between a LIT-based
collaborative robot and the human user in collaborative cooking tasks.


---

**[87. [2305.01639] Privacy-Preserving In-Context Learning for Large Language Models](https://arxiv.org/pdf/2305.01639.pdf)** (2023-10-03)

*Tong Wu, Ashwinee Panda, Jiachen T. Wang, Prateek Mittal*

  In-context learning (ICL) is an important capability of Large Language Models
(LLMs), enabling these models to dynamically adapt based on specific,
in-context exemplars, thereby improving accuracy and relevance. However, LLM's
responses may leak the sensitive private information contained in in-context
exemplars. To address this challenge, we propose Differentially Private
In-context Learning (DP-ICL), a general paradigm for privatizing ICL tasks. The
key idea for DP-ICL paradigm is generating differentially private responses
through a noisy consensus among an ensemble of LLM's responses based on
disjoint exemplar sets. Based on the general paradigm of DP-ICL, we instantiate
several techniques showing how to privatize ICL for text classification and
language generation. We evaluate DP-ICL on four text classification benchmarks
and two language generation tasks, and our empirical results show that DP-ICL
achieves a strong utility-privacy tradeoff.


---

**[88. [2408.10577] Optimizing Large Language Model Hyperparameters for Code Generation](https://arxiv.org/pdf/2408.10577.pdf)** (2024-08-21)

*Chetan Arora, Ahnaf Ibn Sayeed, Sherlock Licorish, Fanyu Wang, Christoph Treude*

  Large Language Models (LLMs), such as GPT models, are increasingly used in
software engineering for various tasks, such as code generation, requirements
management, and debugging. While automating these tasks has garnered
significant attention, a systematic study on the impact of varying
hyperparameters on code generation outcomes remains unexplored. This study aims
to assess LLMs' code generation performance by exhaustively exploring the
impact of various hyperparameters. Hyperparameters for LLMs are adjustable
settings that affect the model's behaviour and performance. Specifically, we
investigated how changes to the hyperparameters: temperature, top probability
(top_p), frequency penalty, and presence penalty affect code generation
outcomes. We systematically adjusted all hyperparameters together, exploring
every possible combination by making small increments to each hyperparameter at
a time. This exhaustive approach was applied to 13 Python code generation
tasks, yielding one of four outcomes for each hyperparameter combination: no
output from the LLM, non executable code, code that fails unit tests, or
correct and functional code. We analysed these outcomes for a total of 14,742
generated Python code segments, focusing on correctness, to determine how the
hyperparameters influence the LLM to arrive at each outcome. Using correlation
coefficient and regression tree analyses, we ascertained which hyperparameters
influence which aspect of the LLM. Our results indicate that optimal
performance is achieved with a temperature below 0.5, top probability below
0.75, frequency penalty above -1 and below 1.5, and presence penalty above -1.
We make our dataset and results available to facilitate replication.


---

**[89. [2405.14191] S-Eval: Towards Automated and Comprehensive Safety Evaluation for Large
  Language Models](https://arxiv.org/pdf/2405.14191.pdf)** (2025-04-08)

*Xiaohan Yuan, Jinfeng Li, Dongxia Wang, Yuefeng Chen, Xiaofeng Mao, Longtao Huang, Jialuo Chen, Hui Xue, Xiaoxia Liu, Wenhai Wang, Kui Ren, Jingyi Wang*

  Generative large language models (LLMs) have revolutionized natural language
processing with their transformative and emergent capabilities. However, recent
evidence indicates that LLMs can produce harmful content that violates social
norms, raising significant concerns regarding the safety and ethical
ramifications of deploying these advanced models. Thus, it is both critical and
imperative to perform a rigorous and comprehensive safety evaluation of LLMs
before deployment. Despite this need, owing to the extensiveness of LLM
generation space, it still lacks a unified and standardized risk taxonomy to
systematically reflect the LLM content safety, as well as automated safety
assessment techniques to explore the potential risk efficiently.
  To bridge the striking gap, we propose S-Eval, a novel LLM-based automated
Safety Evaluation framework with a newly defined comprehensive risk taxonomy.
S-Eval incorporates two key components, i.e., an expert testing LLM ${M}_t$ and
a novel safety critique LLM ${M}_c$. ${M}_t$ is responsible for automatically
generating test cases in accordance with the proposed risk taxonomy. ${M}_c$
can provide quantitative and explainable safety evaluations for better risk
awareness of LLMs. In contrast to prior works, S-Eval is efficient and
effective in test generation and safety evaluation. Moreover, S-Eval can be
flexibly configured and adapted to the rapid evolution of LLMs and accompanying
new safety threats, test generation methods and safety critique methods thanks
to the LLM-based architecture. S-Eval has been deployed in our industrial
partner for the automated safety evaluation of multiple LLMs serving millions
of users, demonstrating its effectiveness in real-world scenarios. Our
benchmark is publicly available at https://github.com/IS2Lab/S-Eval.


---

**[90. [2411.06493] LProtector: An LLM-driven Vulnerability Detection System](https://arxiv.org/pdf/2411.06493.pdf)** (2024-11-15)

*Ze Sheng, Fenghua Wu, Xiangwu Zuo, Chao Li, Yuxin Qiao, Lei Hang*

  This paper presents LProtector, an automated vulnerability detection system
for C/C++ codebases driven by the large language model (LLM) GPT-4o and
Retrieval-Augmented Generation (RAG). As software complexity grows, traditional
methods face challenges in detecting vulnerabilities effectively. LProtector
leverages GPT-4o's powerful code comprehension and generation capabilities to
perform binary classification and identify vulnerabilities within target
codebases. We conducted experiments on the Big-Vul dataset, showing that
LProtector outperforms two state-of-the-art baselines in terms of F1 score,
demonstrating the potential of integrating LLMs with vulnerability detection.


---

**[91. [2306.11507] TrustGPT: A Benchmark for Trustworthy and Responsible Large Language
  Models](https://arxiv.org/pdf/2306.11507.pdf)** (2023-06-21)

*Yue Huang, Qihui Zhang, Philip S. Y, Lichao Sun*

  Large Language Models (LLMs) such as ChatGPT, have gained significant
attention due to their impressive natural language processing capabilities. It
is crucial to prioritize human-centered principles when utilizing these models.
Safeguarding the ethical and moral compliance of LLMs is of utmost importance.
However, individual ethical issues have not been well studied on the latest
LLMs. Therefore, this study aims to address these gaps by introducing a new
benchmark -- TrustGPT. TrustGPT provides a comprehensive evaluation of LLMs in
three crucial areas: toxicity, bias, and value-alignment. Initially, TrustGPT
examines toxicity in language models by employing toxic prompt templates
derived from social norms. It then quantifies the extent of bias in models by
measuring quantifiable toxicity values across different groups. Lastly,
TrustGPT assesses the value of conversation generation models from both active
value-alignment and passive value-alignment tasks. Through the implementation
of TrustGPT, this research aims to enhance our understanding of the performance
of conversation generation models and promote the development of language
models that are more ethical and socially responsible.


---

**[92. [2501.03112] LangFair: A Python Package for Assessing Bias and Fairness in Large
  Language Model Use Cases](https://arxiv.org/pdf/2501.03112.pdf)** (2025-01-30)

*Dylan Bouchard, Mohit Singh Chauhan, David Skarbrevik, Viren Bajaj, Zeya Ahmad*

  Large Language Models (LLMs) have been observed to exhibit bias in numerous
ways, potentially creating or worsening outcomes for specific groups identified
by protected attributes such as sex, race, sexual orientation, or age. To help
address this gap, we introduce LangFair, an open-source Python package that
aims to equip LLM practitioners with the tools to evaluate bias and fairness
risks relevant to their specific use cases. The package offers functionality to
easily generate evaluation datasets, comprised of LLM responses to
use-case-specific prompts, and subsequently calculate applicable metrics for
the practitioner's use case. To guide in metric selection, LangFair offers an
actionable decision framework.


---

**[93. [2410.08431] oRetrieval Augmented Generation for 10 Large Language Models and its
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

**[94. [2403.07865] CodeAttack: Revealing Safety Generalization Challenges of Large Language
  Models via Code Completion](https://arxiv.org/pdf/2403.07865.pdf)** (2024-09-17)

*Qibing Ren, Chang Gao, Jing Shao, Junchi Yan, Xin Tan, Wai Lam, Lizhuang Ma*

  The rapid advancement of Large Language Models (LLMs) has brought about
remarkable generative capabilities but also raised concerns about their
potential misuse. While strategies like supervised fine-tuning and
reinforcement learning from human feedback have enhanced their safety, these
methods primarily focus on natural languages, which may not generalize to other
domains. This paper introduces CodeAttack, a framework that transforms natural
language inputs into code inputs, presenting a novel environment for testing
the safety generalization of LLMs. Our comprehensive studies on
state-of-the-art LLMs including GPT-4, Claude-2, and Llama-2 series reveal a
new and universal safety vulnerability of these models against code input:
CodeAttack bypasses the safety guardrails of all models more than 80\% of the
time. We find that a larger distribution gap between CodeAttack and natural
language leads to weaker safety generalization, such as encoding natural
language input with data structures. Furthermore, we give our hypotheses about
the success of CodeAttack: the misaligned bias acquired by LLMs during code
training, prioritizing code completion over avoiding the potential safety risk.
Finally, we analyze potential mitigation measures. These findings highlight new
safety risks in the code domain and the need for more robust safety alignment
algorithms to match the code capabilities of LLMs.


---

**[95. [2502.11466] GiFT: Gibbs Fine-Tuning for Code Generation](https://arxiv.org/pdf/2502.11466.pdf)** (2025-02-18)

*Haochen Li, Wanjin Feng, Xin Zhou, Zhiqi Shen*

  Training Large Language Models (LLMs) with synthetic data is a prevalent
practice in code generation. A key approach is self-training, where LLMs are
iteratively trained on self-generated correct code snippets. In this case, the
self-generated codes are drawn from a conditional distribution, conditioned on
a specific seed description. However, the seed description is not the only
valid representation that aligns with its intended meaning. With all valid
descriptions and codes forming a joint space, codes drawn from the conditional
distribution would lead to an underrepresentation of the full description-code
space. As such, we propose Gibbs Fine-Tuning (GiFT), a novel self-training
method inspired by Gibbs sampling. GiFT allows self-generated data to be drawn
from the marginal distribution of the joint space, thereby mitigating the
biases inherent in conditional sampling. We provide a theoretical analysis
demonstrating the potential benefits of fine-tuning LLMs with code derived from
the marginal distribution. Furthermore, we propose a perplexity-based code
selection method to mitigate the imbalanced long-tail distribution of the
self-generated codes. Empirical evaluation of two LLMs across four datasets
demonstrates that GiFT achieves superior performance, particularly on more
challenging benchmarks.


---

**[96. [2404.12038] Uncovering Safety Risks of Large Language Models through Concept
  Activation Vector](https://arxiv.org/pdf/2404.12038.pdf)** (2024-12-03)

*Zhihao Xu, Ruixuan Huang, Changyu Chen, Xiting Wang*

  Despite careful safety alignment, current large language models (LLMs) remain
vulnerable to various attacks. To further unveil the safety risks of LLMs, we
introduce a Safety Concept Activation Vector (SCAV) framework, which
effectively guides the attacks by accurately interpreting LLMs' safety
mechanisms. We then develop an SCAV-guided attack method that can generate both
attack prompts and embedding-level attacks with automatically selected
perturbation hyperparameters. Both automatic and human evaluations demonstrate
that our attack method significantly improves the attack success rate and
response quality while requiring less training data. Additionally, we find that
our generated attack prompts may be transferable to GPT-4, and the
embedding-level attacks may also be transferred to other white-box LLMs whose
parameters are known. Our experiments further uncover the safety risks present
in current LLMs. For example, in our evaluation of seven open-source LLMs, we
observe an average attack success rate of 99.14%, based on the classic
keyword-matching criterion. Finally, we provide insights into the safety
mechanism of LLMs. The code is available at
https://github.com/SproutNan/AI-Safety_SCAV.


---

**[97. [2410.15737] Who's Who: Large Language Models Meet Knowledge Conflicts in Practice](https://arxiv.org/pdf/2410.15737.pdf)** (2024-10-22)

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

**[98. [2404.04287] CONFLARE: CONFormal LArge language model REtrieval](https://arxiv.org/pdf/2404.04287.pdf)** (2024-04-09)

*Pouria Rouzrokh, Shahriar Faghani, Cooper U. Gamble, Moein Shariatnia, Bradley J. Erickson*

  Retrieval-augmented generation (RAG) frameworks enable large language models
(LLMs) to retrieve relevant information from a knowledge base and incorporate
it into the context for generating responses. This mitigates hallucinations and
allows for the updating of knowledge without retraining the LLM. However, RAG
does not guarantee valid responses if retrieval fails to identify the necessary
information as the context for response generation. Also, if there is
contradictory content, the RAG response will likely reflect only one of the two
possible responses. Therefore, quantifying uncertainty in the retrieval process
is crucial for ensuring RAG trustworthiness. In this report, we introduce a
four-step framework for applying conformal prediction to quantify retrieval
uncertainty in RAG frameworks. First, a calibration set of questions answerable
from the knowledge base is constructed. Each question's embedding is compared
against document embeddings to identify the most relevant document chunks
containing the answer and record their similarity scores. Given a
user-specified error rate ({\alpha}), these similarity scores are then analyzed
to determine a similarity score cutoff threshold. During inference, all chunks
with similarity exceeding this threshold are retrieved to provide context to
the LLM, ensuring the true answer is captured in the context with a
(1-{\alpha}) confidence level. We provide a Python package that enables users
to implement the entire workflow proposed in our work, only using LLMs and
without human intervention.


---

**[99. [2504.13187] Benchmarking Large Language Models for Calculus Problem-Solving: A
  Comparative Analysis](https://arxiv.org/pdf/2504.13187.pdf)** (2025-04-21)

*In Hak Moon*

  This study presents a comprehensive evaluation of five leading large language
models (LLMs) - Chat GPT 4o, Copilot Pro, Gemini Advanced, Claude Pro, and Meta
AI - on their performance in solving calculus differentiation problems. The
investigation assessed these models across 13 fundamental problem types,
employing a systematic cross-evaluation framework where each model solved
problems generated by all models. Results revealed significant performance
disparities, with Chat GPT 4o achieving the highest success rate (94.71%),
followed by Claude Pro (85.74%), Gemini Advanced (84.42%), Copilot Pro
(76.30%), and Meta AI (56.75%). All models excelled at procedural
differentiation tasks but showed varying limitations with conceptual
understanding and algebraic manipulation. Notably, problems involving
increasing/decreasing intervals and optimization word problems proved most
challenging across all models. The cross-evaluation matrix revealed that Claude
Pro generated the most difficult problems, suggesting distinct capabilities
between problem generation and problem-solving. These findings have significant
implications for educational applications, highlighting both the potential and
limitations of LLMs as calculus learning tools. While they demonstrate
impressive procedural capabilities, their conceptual understanding remains
limited compared to human mathematical reasoning, emphasizing the continued
importance of human instruction for developing deeper mathematical
comprehension.


---

**[100. [2408.10718] CodeJudge-Eval: Can Large Language Models be Good Judges in Code
  Understanding?](https://arxiv.org/pdf/2408.10718.pdf)** (2024-09-16)

*Yuwei Zhao, Ziyang Luo, Yuchen Tian, Hongzhan Lin, Weixiang Yan, Annan Li, Jing Ma*

  Recent advancements in large language models (LLMs) have showcased impressive
code generation capabilities, primarily evaluated through language-to-code
benchmarks. However, these benchmarks may not fully capture a model's code
understanding abilities. We introduce CodeJudge-Eval (CJ-Eval), a novel
benchmark designed to assess LLMs' code understanding abilities from the
perspective of code judging rather than code generation. CJ-Eval challenges
models to determine the correctness of provided code solutions, encompassing
various error types and compilation issues. By leveraging a diverse set of
problems and a fine-grained judging system, CJ-Eval addresses the limitations
of traditional benchmarks, including the potential memorization of solutions.
Evaluation of 12 well-known LLMs on CJ-Eval reveals that even state-of-the-art
models struggle, highlighting the benchmark's ability to probe deeper into
models' code understanding abilities. Our codes and benchmark are available at
\url{https://github.com/CodeLLM-Research/CodeJudge-Eval}.


---

**[101. [2308.04823] Evaluating the Generation Capabilities of Large Chinese Language Models](https://arxiv.org/pdf/2308.04823.pdf)** (2024-01-31)

*Hui Zeng, Jingyuan Xue, Meng Hao, Chen Sun, Bin Ning, Na Zhang*

  This paper unveils CG-Eval, the first-ever comprehensive and automated
evaluation framework designed for assessing the generative capabilities of
large Chinese language models across a spectrum of academic disciplines.
CG-Eval stands out for its automated process, which critically assesses models
based on their proficiency in generating precise and contextually relevant
responses to a diverse array of questions within six key domains: Science and
Engineering, Humanities and Social Sciences, Mathematical Calculations, Medical
Practitioner Qualification Examination, Judicial Examination, and Certified
Public Accountant Examination. Alongside this, we introduce Gscore, an
innovative composite index developed from a weighted sum of multiple metrics.
Gscore uniquely automates the quality measurement of a model's text generation
against reference standards, providing a detailed and nuanced assessment of
model performance. This automation not only enhances the efficiency and
scalability of the evaluation process but also ensures objective and consistent
assessment across various models. The detailed test data and results,
highlighting the robust capabilities and comparative performance of the
evaluated models, are accessible at http://cgeval.besteasy.com/.


---

**[102. [2207.14157] A Hazard Analysis Framework for Code Synthesis Large Language Models](https://arxiv.org/pdf/2207.14157.pdf)** (2022-07-29)

*Heidy Khlaaf, Pamela Mishkin, Joshua Achiam, Gretchen Krueger, Miles Brundage*

  Codex, a large language model (LLM) trained on a variety of codebases,
exceeds the previous state of the art in its capacity to synthesize and
generate code. Although Codex provides a plethora of benefits, models that may
generate code on such scale have significant limitations, alignment problems,
the potential to be misused, and the possibility to increase the rate of
progress in technical fields that may themselves have destabilizing impacts or
have misuse potential. Yet such safety impacts are not yet known or remain to
be explored. In this paper, we outline a hazard analysis framework constructed
at OpenAI to uncover hazards or safety risks that the deployment of models like
Codex may impose technically, socially, politically, and economically. The
analysis is informed by a novel evaluation framework that determines the
capacity of advanced code generation techniques against the complexity and
expressivity of specification prompts, and their capability to understand and
execute them relative to human ability.


---

**[103. [2309.01868] On the Planning, Search, and Memorization Capabilities of Large Language
  Models](https://arxiv.org/pdf/2309.01868.pdf)** (2023-09-06)

*Yunhao Yang, Anshul Tomar*

  The rapid advancement of large language models, such as the Generative
Pre-trained Transformer (GPT) series, has had significant implications across
various disciplines. In this study, we investigate the potential of the
state-of-the-art large language model (GPT-4) for planning tasks. We explore
its effectiveness in multiple planning subfields, highlighting both its
strengths and limitations. Through a comprehensive examination, we identify
areas where large language models excel in solving planning problems and reveal
the constraints that limit their applicability. Our empirical analysis focuses
on GPT-4's performance in planning domain extraction, graph search path
planning, and adversarial planning. We then propose a way of fine-tuning a
domain-specific large language model to improve its Chain of Thought (CoT)
capabilities for the above-mentioned tasks. The results provide valuable
insights into the potential applications of large language models in the
planning domain and pave the way for future research to overcome their
limitations and expand their capabilities.


---

**[104. [2202.13169] A Systematic Evaluation of Large Language Models of Code](https://arxiv.org/pdf/2202.13169.pdf)** (2022-05-05)

*Frank F. Xu, Uri Alon, Graham Neubig, Vincent J. Hellendoorn*

  Large language models (LMs) of code have recently shown tremendous promise in
completing code and synthesizing code from natural language descriptions.
However, the current state-of-the-art code LMs (e.g., Codex (Chen et al.,
2021)) are not publicly available, leaving many questions about their model and
data design decisions. We aim to fill in some of these blanks through a
systematic evaluation of the largest existing models: Codex, GPT-J, GPT-Neo,
GPT-NeoX-20B, and CodeParrot, across various programming languages. Although
Codex itself is not open-source, we find that existing open-source models do
achieve close results in some programming languages, although targeted mainly
for natural language modeling. We further identify an important missing piece
in the form of a large open-source model trained exclusively on a multi-lingual
corpus of code. We release a new model, PolyCoder, with 2.7B parameters based
on the GPT-2 architecture, which was trained on 249GB of code across 12
programming languages on a single machine. In the C programming language,
PolyCoder outperforms all models including Codex. Our trained models are
open-source and publicly available at https://github.com/VHellendoorn/Code-LMs,
which enables future research and application in this area.


---

**[105. [2302.05817] Level Generation Through Large Language Models](https://arxiv.org/pdf/2302.05817.pdf)** (2023-06-02)

*Graham Todd, Sam Earle, Muhammad Umair Nasir, Michael Cerny Green, Julian Togelius*

  Large Language Models (LLMs) are powerful tools, capable of leveraging their
training on natural language to write stories, generate code, and answer
questions. But can they generate functional video game levels? Game levels,
with their complex functional constraints and spatial relationships in more
than one dimension, are very different from the kinds of data an LLM typically
sees during training. Datasets of game levels are also hard to come by,
potentially taxing the abilities of these data-hungry models. We investigate
the use of LLMs to generate levels for the game Sokoban, finding that LLMs are
indeed capable of doing so, and that their performance scales dramatically with
dataset size. We also perform preliminary experiments on controlling LLM level
generators and discuss promising areas for future work.


---

**[106. [2402.00707] Non-Exchangeable Conformal Language Generation with Nearest Neighbors](https://arxiv.org/pdf/2402.00707.pdf)** (2024-02-02)

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

**[107. [2306.15895] Large Language Model as Attributed Training Data Generator: A Tale of
  Diversity and Bias](https://arxiv.org/pdf/2306.15895.pdf)** (2023-10-19)

*Yue Yu, Yuchen Zhuang, Jieyu Zhang, Yu Meng, Alexander Ratner, Ranjay Krishna, Jiaming Shen, Chao Zhang*

  Large language models (LLMs) have been recently leveraged as training data
generators for various natural language processing (NLP) tasks. While previous
research has explored different approaches to training models using generated
data, they generally rely on simple class-conditional prompts, which may limit
the diversity of the generated data and inherit systematic biases of LLM. Thus,
we investigate training data generation with diversely attributed prompts
(e.g., specifying attributes like length and style), which have the potential
to yield diverse and attributed generated data. Our investigation focuses on
datasets with high cardinality and diverse domains, wherein we demonstrate that
attributed prompts outperform simple class-conditional prompts in terms of the
resulting model's performance. Additionally, we present a comprehensive
empirical study on data generation encompassing vital aspects like bias,
diversity, and efficiency, and highlight three key observations: firstly,
synthetic datasets generated by simple prompts exhibit significant biases, such
as regional bias; secondly, attribute diversity plays a pivotal role in
enhancing model performance; lastly, attributed prompts achieve the performance
of simple class-conditional prompts while utilizing only 5\% of the querying
cost of ChatGPT associated with the latter. The data and code are available on
\url{https://github.com/yueyu1030/AttrPrompt}.


---

**[108. [2312.01639] On the Effectiveness of Large Language Models in Domain-Specific Code
  Generation](https://arxiv.org/pdf/2312.01639.pdf)** (2025-02-18)

*Xiaodong Gu, Meng Chen, Yalan Lin, Yuhan Hu, Hongyu Zhang, Chengcheng Wan, Zhao Wei, Yong Xu, Juhong Wang*

  Large language models (LLMs) such as ChatGPT have shown remarkable
capabilities in code generation. Despite significant achievements, they rely on
enormous training data to acquire a broad spectrum of open-domain knowledge.
Besides, their evaluation revolves around open-domain benchmarks like
HumanEval, which primarily consist of programming contests. Therefore, it is
hard to fully characterize the intricacies and challenges associated with
particular domains (e.g., web, game, and math). In this paper, we conduct an
in-depth study of the LLMs in domain-specific code generation. Our results
demonstrate that LLMs exhibit sub-optimal performance in generating
domain-specific code, due to their limited proficiency in utilizing
domain-specific libraries. We further observe that incorporating API knowledge
as prompts can empower LLMs to generate more professional code. Based on these
findings, we further investigate how to effectively incorporate API knowledge
into the code generation process. We experiment with three strategies for
incorporating domain knowledge, namely, external knowledge inquirer,
chain-of-thought prompting, and chain-of-thought fine-tuning. We refer to these
strategies as a new code generation approach called DomCoder. Experimental
results show that all strategies of DomCoder lead to improvement in the
effectiveness of domain-specific code generation under certain settings.


---

**[109. [2311.13361] Applying Large Language Models to Power Systems: Potential Security
  Threats](https://arxiv.org/pdf/2311.13361.pdf)** (2024-01-25)

*Jiaqi Ruan, Gaoqi Liang, Huan Zhao, Guolong Liu, Xianzhuo Sun, Jing Qiu, Zhao Xu, Fushuan Wen, Zhao Yang Dong*

  Applying large language models (LLMs) to modern power systems presents a
promising avenue for enhancing decision-making and operational efficiency.
However, this action may also incur potential security threats, which have not
been fully recognized so far. To this end, this article analyzes potential
threats incurred by applying LLMs to power systems, emphasizing the need for
urgent research and development of countermeasures.


---

**[110. [2502.19320] Shh, don't say that! Domain Certification in LLMs](https://arxiv.org/pdf/2502.19320.pdf)** (2025-03-10)

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

**[111. [2503.13772] Do Large Language Models Understand Performance Optimization?](https://arxiv.org/pdf/2503.13772.pdf)** (2025-03-19)

*Bowen Cui, Tejas Ramesh, Oscar Hernandez, Keren Zhou*

  Large Language Models (LLMs) have emerged as powerful tools for software
development tasks such as code completion, translation, and optimization.
However, their ability to generate efficient and correct code, particularly in
complex High-Performance Computing (HPC) contexts, has remained underexplored.
To address this gap, this paper presents a comprehensive benchmark suite
encompassing multiple critical HPC computational motifs to evaluate the
performance of code optimized by state-of-the-art LLMs, including OpenAI o1,
Claude-3.5, and Llama-3.2. In addition to analyzing basic computational
kernels, we developed an agent system that integrates LLMs to assess their
effectiveness in real HPC applications. Our evaluation focused on key criteria
such as execution time, correctness, and understanding of HPC-specific
concepts. We also compared the results with those achieved using traditional
HPC optimization tools. Based on the findings, we recognized the strengths of
LLMs in understanding human instructions and performing automated code
transformations. However, we also identified significant limitations, including
their tendency to generate incorrect code and their challenges in comprehending
complex control and data flows in sophisticated HPC code.


---

**[112. [2412.08072] Using Large Language Models for Parametric Shape Optimization](https://arxiv.org/pdf/2412.08072.pdf)** (2024-12-12)

*Xinxin Zhang, Zhuoqun Xu, Guangpu Zhu, Chien Ming Jonathan Tay, Yongdong Cui, Boo Cheong Khoo, Lailai Zhu*

  Recent advanced large language models (LLMs) have showcased their emergent
capability of in-context learning, facilitating intelligent decision-making
through natural language prompts without retraining. This new machine learning
paradigm has shown promise in various fields, including general control and
optimization problems. Inspired by these advancements, we explore the potential
of LLMs for a specific and essential engineering task: parametric shape
optimization (PSO). We develop an optimization framework, LLM-PSO, that
leverages an LLM to determine the optimal shape of parameterized engineering
designs in the spirit of evolutionary strategies. Utilizing the ``Claude 3.5
Sonnet'' LLM, we evaluate LLM-PSO on two benchmark flow optimization problems,
specifically aiming to identify drag-minimizing profiles for 1) a
two-dimensional airfoil in laminar flow, and 2) a three-dimensional
axisymmetric body in Stokes flow. In both cases, LLM-PSO successfully
identifies optimal shapes in agreement with benchmark solutions. Besides, it
generally converges faster than other classical optimization algorithms. Our
preliminary exploration may inspire further investigations into harnessing LLMs
for shape optimization and engineering design more broadly.


---

**[113. [2308.15930] LLaSM: Large Language and Speech Model](https://arxiv.org/pdf/2308.15930.pdf)** (2023-09-19)

*Yu Shu, Siwei Dong, Guangyao Chen, Wenhao Huang, Ruihua Zhang, Daochen Shi, Qiqi Xiang, Yemin Shi*

  Multi-modal large language models have garnered significant interest
recently. Though, most of the works focus on vision-language multi-modal models
providing strong capabilities in following vision-and-language instructions.
However, we claim that speech is also an important modality through which
humans interact with the world. Hence, it is crucial for a general-purpose
assistant to be able to follow multi-modal speech-and-language instructions. In
this work, we propose Large Language and Speech Model (LLaSM). LLaSM is an
end-to-end trained large multi-modal speech-language model with cross-modal
conversational abilities, capable of following speech-and-language
instructions. Our early experiments show that LLaSM demonstrates a more
convenient and natural way for humans to interact with artificial intelligence.
Specifically, we also release a large Speech Instruction Following dataset
LLaSM-Audio-Instructions. Code and demo are available at
https://github.com/LinkSoul-AI/LLaSM and
https://huggingface.co/spaces/LinkSoul/LLaSM. The LLaSM-Audio-Instructions
dataset is available at
https://huggingface.co/datasets/LinkSoul/LLaSM-Audio-Instructions.


---

**[114. [2406.08754] StructuralSleight: Automated Jailbreak Attacks on Large Language Models
  Utilizing Uncommon Text-Organization Structures](https://arxiv.org/pdf/2406.08754.pdf)** (2025-02-19)

*Bangxin Li, Hengrui Xing, Cong Tian, Chao Huang, Jin Qian, Huangqing Xiao, Linfeng Feng*

  Large Language Models (LLMs) are widely used in natural language processing
but face the risk of jailbreak attacks that maliciously induce them to generate
harmful content. Existing jailbreak attacks, including character-level and
context-level attacks, mainly focus on the prompt of plain text without
specifically exploring the significant influence of its structure. In this
paper, we focus on studying how the prompt structure contributes to the
jailbreak attack. We introduce a novel structure-level attack method based on
long-tailed structures, which we refer to as Uncommon Text-Organization
Structures (UTOS). We extensively study 12 UTOS templates and 6 obfuscation
methods to build an effective automated jailbreak tool named StructuralSleight
that contains three escalating attack strategies: Structural Attack, Structural
and Character/Context Obfuscation Attack, and Fully Obfuscated Structural
Attack. Extensive experiments on existing LLMs show that StructuralSleight
significantly outperforms the baseline methods. In particular, the attack
success rate reaches 94.62\% on GPT-4o, which has not been addressed by
state-of-the-art techniques.


---

**[115. [2408.07482] Training Overhead Ratio: A Practical Reliability Metric for Large
  Language Model Training Systems](https://arxiv.org/pdf/2408.07482.pdf)** (2024-10-10)

*Ning Lu, Qian Xie, Hao Zhang, Wenyi Fang, Yang Zheng, Zheng Hu, Jiantao Ma*

  Large Language Models (LLMs) are revolutionizing the AI industry with their
superior capabilities. Training these models requires large-scale GPU clusters
and significant computing time, leading to frequent failures that significantly
increase training costs. Despite its significance, this field lacks a metric
for evaluating reliability. In this work, we introduce a novel reliability
metric called \emph{Training Overhead Ratio} (TOR) to evaluate the reliability
of fault-tolerant LLM training systems. TOR is defined as the ratio of optimal
training time to the observed training time of a system, serving as a practical
tool for users to estimate the actual time required to train an LLM on a given
system. Furthermore, our investigation identifies the key factor for enhancing
reliability and present TOR equations for various types of failures encountered
in practice.


---

**[116. [2310.03283] A Formalism and Approach for Improving Robustness of Large Language
  Models Using Risk-Adjusted Confidence Scores](https://arxiv.org/pdf/2310.03283.pdf)** (2023-10-06)

*Ke Shen, Mayank Kejriwal*

  Large Language Models (LLMs), such as ChatGPT, have achieved impressive
milestones in natural language processing (NLP). Despite their impressive
performance, the models are known to pose important risks. As these models are
deployed in real-world applications, a systematic understanding of different
risks posed by these models on tasks such as natural language inference (NLI),
is much needed. In this paper, we define and formalize two distinct types of
risk: decision risk and composite risk. We also propose a risk-centric
evaluation framework, and four novel metrics, for assessing LLMs on these risks
in both in-domain and out-of-domain settings. Finally, we propose a
risk-adjusted calibration method called DwD for helping LLMs minimize these
risks in an overall NLI architecture. Detailed experiments, using four NLI
benchmarks, three baselines and two LLMs, including ChatGPT, show both the
practical utility of the evaluation framework, and the efficacy of DwD in
reducing decision and composite risk. For instance, when using DwD, an
underlying LLM is able to address an extra 20.1% of low-risk inference tasks
(but which the LLM erroneously deems high-risk without risk adjustment) and
skip a further 19.8% of high-risk tasks, which would have been answered
incorrectly.


---

**[117. [2402.04373] The World of Generative AI: Deepfakes and Large Language Models](https://arxiv.org/pdf/2402.04373.pdf)** (2024-02-08)

*Alakananda Mitra, Saraju P. Mohanty, Elias Kougianos*

  We live in the era of Generative Artificial Intelligence (GenAI). Deepfakes
and Large Language Models (LLMs) are two examples of GenAI. Deepfakes, in
particular, pose an alarming threat to society as they are capable of spreading
misinformation and changing the truth. LLMs are powerful language models that
generate general-purpose language. However due to its generative aspect, it can
also be a risk for people if used with ill intentions. The ethical use of these
technologies is a big concern. This short article tries to find out the
interrelationship between them.


---

**[118. [2503.15092] Towards Understanding the Safety Boundaries of DeepSeek Models:
  Evaluation and Findings](https://arxiv.org/pdf/2503.15092.pdf)** (2025-03-20)

*Zonghao Ying, Guangyi Zheng, Yongxin Huang, Deyue Zhang, Wenxin Zhang, Quanchen Zou, Aishan Liu, Xianglong Liu, Dacheng Tao*

  This study presents the first comprehensive safety evaluation of the DeepSeek
models, focusing on evaluating the safety risks associated with their generated
content. Our evaluation encompasses DeepSeek's latest generation of large
language models, multimodal large language models, and text-to-image models,
systematically examining their performance regarding unsafe content generation.
Notably, we developed a bilingual (Chinese-English) safety evaluation dataset
tailored to Chinese sociocultural contexts, enabling a more thorough evaluation
of the safety capabilities of Chinese-developed models. Experimental results
indicate that despite their strong general capabilities, DeepSeek models
exhibit significant safety vulnerabilities across multiple risk dimensions,
including algorithmic discrimination and sexual content. These findings provide
crucial insights for understanding and improving the safety of large foundation
models. Our code is available at
https://github.com/NY1024/DeepSeek-Safety-Eval.


---

**[119. [2303.17564] BloombergGPT: A Large Language Model for Finance](https://arxiv.org/pdf/2303.17564.pdf)** (2023-12-22)

*Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann, Prabhanjan Kambadur, David Rosenberg, Gideon Mann*

  The use of NLP in the realm of financial technology is broad and complex,
with applications ranging from sentiment analysis and named entity recognition
to question answering. Large Language Models (LLMs) have been shown to be
effective on a variety of tasks; however, no LLM specialized for the financial
domain has been reported in literature. In this work, we present BloombergGPT,
a 50 billion parameter language model that is trained on a wide range of
financial data. We construct a 363 billion token dataset based on Bloomberg's
extensive data sources, perhaps the largest domain-specific dataset yet,
augmented with 345 billion tokens from general purpose datasets. We validate
BloombergGPT on standard LLM benchmarks, open financial benchmarks, and a suite
of internal benchmarks that most accurately reflect our intended usage. Our
mixed dataset training leads to a model that outperforms existing models on
financial tasks by significant margins without sacrificing performance on
general LLM benchmarks. Additionally, we explain our modeling choices, training
process, and evaluation methodology. We release Training Chronicles (Appendix
C) detailing our experience in training BloombergGPT.


---

**[120. [2408.10668] Probing the Safety Response Boundary of Large Language Models via Unsafe
  Decoding Path Generation](https://arxiv.org/pdf/2408.10668.pdf)** (2024-08-27)

*Haoyu Wang, Bingzhe Wu, Yatao Bian, Yongzhe Chang, Xueqian Wang, Peilin Zhao*

  Large Language Models (LLMs) are implicit troublemakers. While they provide
valuable insights and assist in problem-solving, they can also potentially
serve as a resource for malicious activities. Implementing safety alignment
could mitigate the risk of LLMs generating harmful responses. We argue that:
even when an LLM appears to successfully block harmful queries, there may still
be hidden vulnerabilities that could act as ticking time bombs. To identify
these underlying weaknesses, we propose to use a cost value model as both a
detector and an attacker. Trained on external or self-generated harmful
datasets, the cost value model could successfully influence the original safe
LLM to output toxic content in decoding process. For instance, LLaMA-2-chat 7B
outputs 39.18% concrete toxic content, along with only 22.16% refusals without
any harmful suffixes. These potential weaknesses can then be exploited via
prompt optimization such as soft prompts on images. We name this decoding
strategy: Jailbreak Value Decoding (JVD), emphasizing that seemingly secure
LLMs may not be as safe as we initially believe. They could be used to gather
harmful data or launch covert attacks.


---

**[121. [2308.13507] Large Language Models Should Ask Clarifying Questions to Increase
  Confidence in Generated Code](https://arxiv.org/pdf/2308.13507.pdf)** (2024-01-23)

*Jie JW Wu*

  Large language models (LLMs) have significantly improved the ability to
perform tasks in the field of code generation. However, there is still a gap
between LLMs being capable coders and being top-tier software engineers. Based
on the observation that toplevel software engineers often ask clarifying
questions to reduce ambiguity in both requirements and coding solutions, I
argue that the same should be applied to LLMs for code generation tasks. By
asking probing questions in various topics before generating the final code,
the challenges of programming with LLMs, such as unclear intent specification,
lack of computational thinking, and undesired code quality, may be alleviated.
This, in turn, increases confidence in the generated code. In this work, I
explore how to leverage better communication skills to achieve greater
confidence in generated code. I propose a communication-centered process that
uses an LLM-generated communicator to identify issues with high ambiguity or
low confidence in problem descriptions and generated code. I then ask
clarifying questions to obtain responses from users for refining the code.


---

**[122. [2408.10474] LeCov: Multi-level Testing Criteria for Large Language Models](https://arxiv.org/pdf/2408.10474.pdf)** (2024-08-21)

*Xuan Xie, Jiayang Song, Yuheng Huang, Da Song, Fuyuan Zhang, Felix Juefei-Xu, Lei Ma*

  Large Language Models (LLMs) are widely used in many different domains, but
because of their limited interpretability, there are questions about how
trustworthy they are in various perspectives, e.g., truthfulness and toxicity.
Recent research has started developing testing methods for LLMs, aiming to
uncover untrustworthy issues, i.e., defects, before deployment. However,
systematic and formalized testing criteria are lacking, which hinders a
comprehensive assessment of the extent and adequacy of testing exploration. To
mitigate this threat, we propose a set of multi-level testing criteria, LeCov,
for LLMs. The criteria consider three crucial LLM internal components, i.e.,
the attention mechanism, feed-forward neurons, and uncertainty, and contain
nine types of testing criteria in total. We apply the criteria in two
scenarios: test prioritization and coverage-guided testing. The experiment
evaluation, on three models and four datasets, demonstrates the usefulness and
effectiveness of LeCov.


---

**[123. [2504.02883] SemEval-2025 Task 4: Unlearning sensitive content from Large Language
  Models](https://arxiv.org/pdf/2504.02883.pdf)** (2025-04-07)

*Anil Ramakrishna, Yixin Wan, Xiaomeng Jin, Kai-Wei Chang, Zhiqi Bu, Bhanukiran Vinzamuri, Volkan Cevher, Mingyi Hong, Rahul Gupta*

  We introduce SemEval-2025 Task 4: unlearning sensitive content from Large
Language Models (LLMs). The task features 3 subtasks for LLM unlearning
spanning different use cases: (1) unlearn long form synthetic creative
documents spanning different genres; (2) unlearn short form synthetic
biographies containing personally identifiable information (PII), including
fake names, phone number, SSN, email and home addresses, and (3) unlearn real
documents sampled from the target model's training dataset. We received over
100 submissions from over 30 institutions and we summarize the key techniques
and lessons in this paper.


---

**[124. [2411.04372] Benchmarking Large Language Models with Integer Sequence Generation
  Tasks](https://arxiv.org/pdf/2411.04372.pdf)** (2024-11-08)

*Daniel O'Malley, Manish Bhattarai, Javier Santos*

  This paper presents a novel benchmark where the large language model (LLM)
must write code that computes integer sequences from the Online Encyclopedia of
Integer Sequences (OEIS), a widely-used resource for mathematical sequences.
The benchmark is designed to evaluate both the correctness of the generated
code and its computational efficiency. Our benchmark reveals that the o1 series
of models outperform other frontier models from OpenAI, Anthropic, Meta, and
Google in accuracy and cheating rates across both easy and hard integer
sequences. In order to ensure models do not exploit memorized sequence values,
we introduce an automated cheating detection mechanism that flags the use of
lookup tables and validated this automation against human cheating evaluations.
This benchmark provides a meaningful challenge for current LLMs, offering
insights into their mathematical reasoning and code writing capabilities, which
can guide future research directions and model development in mathematical
reasoning and code synthesis.


---

**[125. [2408.16601] Examination of Code generated by Large Language Models](https://arxiv.org/pdf/2408.16601.pdf)** (2024-08-30)

*Robin Beer, Alexander Feix, Tim Guttzeit, Tamara Muras, Vincent Müller, Maurice Rauscher, Florian Schäffler, Welf Löwe*

  Large language models (LLMs), such as ChatGPT and Copilot, are transforming
software development by automating code generation and, arguably, enable rapid
prototyping, support education, and boost productivity. Therefore, correctness
and quality of the generated code should be on par with manually written code.
To assess the current state of LLMs in generating correct code of high quality,
we conducted controlled experiments with ChatGPT and Copilot: we let the LLMs
generate simple algorithms in Java and Python along with the corresponding unit
tests and assessed the correctness and the quality (coverage) of the generated
(test) codes. We observed significant differences between the LLMs, between the
languages, between algorithm and test codes, and over time. The present paper
reports these results together with the experimental methods allowing repeated
and comparable assessments for more algorithms, languages, and LLMs over time.


---

**[126. [2408.09078] An Exploratory Study on Fine-Tuning Large Language Models for Secure
  Code Generation](https://arxiv.org/pdf/2408.09078.pdf)** (2024-08-20)

*Junjie Li, Fazle Rabbi, Cheng Cheng, Aseem Sangalay, Yuan Tian, Jinqiu Yang*

  AI-powered coding assistants such as GitHub Copilot and OpenAI ChatGPT have
achieved notable success in automating code generation. However, these tools
rely on pre-trained Large Language Models (LLMs) that are typically trained on
human-written code sourced from open-source project hosting sites like GitHub,
which often contains inherent security vulnerabilities. These vulnerabilities
may then be mirrored in the code generated by these LLMs, a critical risk
revealed and highlighted by recent empirical studies. In this work, we present
an exploratory study on whether fine-tuning pre-trained LLMs on datasets of
vulnerability-fixing commits can promote secure code generation. We explored
two parameter-efficient fine-tuning techniques (LoRa and IA3) on two
pre-trained LLMs for code generation. We crawled a fine-tuning dataset (14,622
C and C++ files) for secure code generation by collecting code fixes of
confirmed vulnerabilities from open-source repositories. Our evaluation dataset
comprises 52 vulnerability scenarios designed to cover the top most dangerous C
and C++ Common Weakness Enumerations (CWEs). Each scenario is a prompt that may
induce LLMs to generate vulnerable code. Our exploration reveals that
fine-tuning LLMs can improve secure code generation by 6.4% in C language and
5.4% in C++ language. We further experimented with fine-tuning LLMs using
different versions of the collected secure code dataset (block, function, and
line). We found that fine-tuning with function-level and block-level datasets
achieves the best secure code generation performance, compared to the
alternatives (file-level and line-level).


---

**[127. [2411.00006] Personality-Guided Code Generation Using Large Language Models](https://arxiv.org/pdf/2411.00006.pdf)** (2024-11-04)

*Yaoqi Guo, Zhenpeng Chen, Jie M. Zhang, Yang Liu, Yun Ma*

  Code generation, the automatic creation of source code from natural language
descriptions, has garnered significant attention due to its potential to
streamline software development. Inspired by research that links
task-personality alignment with improved development outcomes, we conduct an
empirical study on personality-guided code generation using large language
models (LLMs). Specifically, we investigate how emulating personality traits
appropriate to the coding tasks affects LLM performance. We extensively
evaluate this approach using seven widely adopted LLMs across four
representative datasets. Our results show that personality guidance
significantly enhances code generation accuracy, with improved pass rates in 23
out of 28 LLM-dataset combinations. Notably, in 11 cases, the improvement
exceeds 5%, and in 5 instances, it surpasses 10%, with the highest gain
reaching 12.9%. Additionally, personality guidance can be easily integrated
with other prompting strategies to further boost performance.


---

**[128. [2409.10146] LLMs4OL 2024 Overview: The 1st Large Language Models for Ontology
  Learning Challenge](https://arxiv.org/pdf/2409.10146.pdf)** (2024-09-17)

*Hamed Babaei Giglou, Jennifer D'Souza, Sören Auer*

  This paper outlines the LLMs4OL 2024, the first edition of the Large Language
Models for Ontology Learning Challenge. LLMs4OL is a community development
initiative collocated with the 23rd International Semantic Web Conference
(ISWC) to explore the potential of Large Language Models (LLMs) in Ontology
Learning (OL), a vital process for enhancing the web with structured knowledge
to improve interoperability. By leveraging LLMs, the challenge aims to advance
understanding and innovation in OL, aligning with the goals of the Semantic Web
to create a more intelligent and user-friendly web. In this paper, we give an
overview of the 2024 edition of the LLMs4OL challenge and summarize the
contributions.


---

**[129. [2408.10608] Promoting Equality in Large Language Models: Identifying and Mitigating
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

**[130. [2402.00888] Security and Privacy Challenges of Large Language Models: A Survey](https://arxiv.org/pdf/2402.00888.pdf)** (2024-11-18)

*Badhan Chandra Das, M. Hadi Amini, Yanzhao Wu*

  Large Language Models (LLMs) have demonstrated extraordinary capabilities and
contributed to multiple fields, such as generating and summarizing text,
language translation, and question-answering. Nowadays, LLM is becoming a very
popular tool in computerized language processing tasks, with the capability to
analyze complicated linguistic patterns and provide relevant and appropriate
responses depending on the context. While offering significant advantages,
these models are also vulnerable to security and privacy attacks, such as
jailbreaking attacks, data poisoning attacks, and Personally Identifiable
Information (PII) leakage attacks. This survey provides a thorough review of
the security and privacy challenges of LLMs for both training data and users,
along with the application-based risks in various domains, such as
transportation, education, and healthcare. We assess the extent of LLM
vulnerabilities, investigate emerging security and privacy attacks for LLMs,
and review the potential defense mechanisms. Additionally, the survey outlines
existing research gaps in this domain and highlights future research
directions.


---

**[131. [2311.18215] Automatic Construction of a Korean Toxic Instruction Dataset for Ethical
  Tuning of Large Language Models](https://arxiv.org/pdf/2311.18215.pdf)** (2023-12-01)

*Sungjoo Byun, Dongjun Jang, Hyemi Jo, Hyopil Shin*

  Caution: this paper may include material that could be offensive or
distressing.
  The advent of Large Language Models (LLMs) necessitates the development of
training approaches that mitigate the generation of unethical language and
aptly manage toxic user queries. Given the challenges related to human labor
and the scarcity of data, we present KoTox, comprising 39K unethical
instruction-output pairs. This collection of automatically generated toxic
instructions refines the training of LLMs and establishes a foundational
framework for improving LLMs' ethical awareness and response to various toxic
inputs, promoting more secure and responsible interactions in Natural Language
Processing (NLP) applications.


---

**[132. [2311.04257] mPLUG-Owl2: Revolutionizing Multi-modal Large Language Model with
  Modality Collaboration](https://arxiv.org/pdf/2311.04257.pdf)** (2023-11-13)

*Qinghao Ye, Haiyang Xu, Jiabo Ye, Ming Yan, Anwen Hu, Haowei Liu, Qi Qian, Ji Zhang, Fei Huang, Jingren Zhou*

  Multi-modal Large Language Models (MLLMs) have demonstrated impressive
instruction abilities across various open-ended tasks. However, previous
methods primarily focus on enhancing multi-modal capabilities. In this work, we
introduce a versatile multi-modal large language model, mPLUG-Owl2, which
effectively leverages modality collaboration to improve performance in both
text and multi-modal tasks. mPLUG-Owl2 utilizes a modularized network design,
with the language decoder acting as a universal interface for managing
different modalities. Specifically, mPLUG-Owl2 incorporates shared functional
modules to facilitate modality collaboration and introduces a modality-adaptive
module that preserves modality-specific features. Extensive experiments reveal
that mPLUG-Owl2 is capable of generalizing both text tasks and multi-modal
tasks and achieving state-of-the-art performances with a single generic model.
Notably, mPLUG-Owl2 is the first MLLM model that demonstrates the modality
collaboration phenomenon in both pure-text and multi-modal scenarios, setting a
pioneering path in the development of future multi-modal foundation models.


---

**[133. [2407.10582] Boosting Zero-Shot Crosslingual Performance using LLM-Based
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

**[134. [2503.16668] Code Evolution Graphs: Understanding Large Language Model Driven Design
  of Algorithms](https://arxiv.org/pdf/2503.16668.pdf)** (2025-03-24)

*Niki van Stein, Anna V. Kononova, Lars Kotthoff, Thomas Bäck*

  Large Language Models (LLMs) have demonstrated great promise in generating
code, especially when used inside an evolutionary computation framework to
iteratively optimize the generated algorithms. However, in some cases they fail
to generate competitive algorithms or the code optimization stalls, and we are
left with no recourse because of a lack of understanding of the generation
process and generated codes. We present a novel approach to mitigate this
problem by enabling users to analyze the generated codes inside the
evolutionary process and how they evolve over repeated prompting of the LLM. We
show results for three benchmark problem classes and demonstrate novel
insights. In particular, LLMs tend to generate more complex code with repeated
prompting, but additional complexity can hurt algorithmic performance in some
cases. Different LLMs have different coding ``styles'' and generated code tends
to be dissimilar to other LLMs. These two findings suggest that using different
LLMs inside the code evolution frameworks might produce higher performing code
than using only one LLM.


---

**[135. [2403.19115] HiRoPE: Length Extrapolation for Code Models Using Hierarchical Position](https://arxiv.org/pdf/2403.19115.pdf)** (2024-08-12)

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

**[136. [2502.05609] Lossless Acceleration of Large Language Models with Hierarchical
  Drafting based on Temporal Locality in Speculative Decoding](https://arxiv.org/pdf/2502.05609.pdf)** (2025-02-11)

*Sukmin Cho, Sangjin Choi, Taeho Hwang, Jeongyeon Seo, Soyeong Jeong, Huije Lee, Hoyun Song, Jong C. Park, Youngjin Kwon*

  Accelerating inference in Large Language Models (LLMs) is critical for
real-time interactions, as they have been widely incorporated into real-world
services. Speculative decoding, a fully algorithmic solution, has gained
attention for improving inference speed by drafting and verifying tokens,
thereby generating multiple tokens in a single forward pass. However, current
drafting strategies usually require significant fine-tuning or have
inconsistent performance across tasks. To address these challenges, we propose
Hierarchy Drafting (HD), a novel lossless drafting approach that organizes
various token sources into multiple databases in a hierarchical framework based
on temporal locality. In the drafting step, HD sequentially accesses multiple
databases to obtain draft tokens from the highest to the lowest locality,
ensuring consistent acceleration across diverse tasks and minimizing drafting
latency. Our experiments on Spec-Bench using LLMs with 7B and 13B parameters
demonstrate that HD outperforms existing database drafting methods, achieving
robust inference speedups across model sizes, tasks, and temperatures.


---

**[137. [2407.01614] Enhancing Stability for Large Language Models Training in Constrained
  Bandwidth Networks](https://arxiv.org/pdf/2407.01614.pdf)** (2024-10-08)

*Yun Dai, Tejas Dharamsi, Byron Hsu, Tao Song, Hamed Firooz*

  Training extremely large language models (LLMs) with billions of parameters
is a computationally intensive task that pushes the limits of current data
parallel training systems. While techniques like ZeRO++ have enabled efficient
distributed training of such giant models on inexpensive low-bandwidth
clusters, they can suffer from convergence issues due to potential race
conditions in the hierarchical partitioning (hpZ) scheme employed to reduce
cross-machine communication. In this work, we first show how these race
conditions cause instability when training models with billions of parameters.
We then propose a modification to the partitioning algorithm that addresses
these convergence challenges while maintaining competitive training efficiency.
Empirical evaluation on training the multi-billion parameters Falcon Models and
Llama-2 models demonstrates the updated algorithm's ability to achieve reliable
convergence on these massive models, where stock ZeRO++ hpZ fails to converge.
The updated algorithm enables robust training of larger models with 98\%
throughput and model training speed improvement without sacrificing the quality
of convergence.


---

**[138. [2411.05897] Humans and Large Language Models in Clinical Decision Support: A Study
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

**[139. [2402.05130] LB-KBQA: Large-language-model and BERT based Knowledge-Based Question
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

**[140. [2503.07956] EFPC: Towards Efficient and Flexible Prompt Compression](https://arxiv.org/pdf/2503.07956.pdf)** (2025-03-12)

*Yun-Hao Cao, Yangsong Wang, Shuzheng Hao, Zhenxing Li, Chengjun Zhan, Sichao Liu, Yi-Qi Hu*

  The emergence of large language models (LLMs) like GPT-4 has revolutionized
natural language processing (NLP), enabling diverse, complex tasks. However,
extensive token counts lead to high computational and financial burdens. To
address this, we propose Efficient and Flexible Prompt Compression (EFPC), a
novel method unifying task-aware and task-agnostic compression for a favorable
accuracy-efficiency trade-off. EFPC uses GPT-4 to generate compressed prompts
and integrates them with original prompts for training. During training and
inference, we selectively prepend user instructions and compress prompts based
on predicted probabilities. EFPC is highly data-efficient, achieving
significant performance with minimal data. Compared to the state-of-the-art
method LLMLingua-2, EFPC achieves a 4.8% relative improvement in F1-score with
1% additional data at a 4x compression rate, and an 11.4% gain with 10%
additional data on the LongBench single-doc QA benchmark. EFPC's unified
framework supports broad applicability and enhances performance across various
models, tasks, and domains, offering a practical advancement in NLP.


---

**[141. [2503.04745] Sovereign Large Language Models: Advantages, Strategy and Regulations](https://arxiv.org/pdf/2503.04745.pdf)** (2025-03-10)

*Mykhailo Bondarenko, Sviatoslav Lushnei, Yurii Paniv, Oleksii Molchanovsky, Mariana Romanyshyn, Yurii Filipchuk, Artur Kiulian*

  This report analyzes key trends, challenges, risks, and opportunities
associated with the development of Large Language Models (LLMs) globally. It
examines national experiences in developing LLMs and assesses the feasibility
of investment in this sector. Additionally, the report explores strategies for
implementing, regulating, and financing AI projects at the state level.


---

**[142. [2406.11345] Full-ECE: A Metric For Token-level Calibration on Large Language Models](https://arxiv.org/pdf/2406.11345.pdf)** (2024-06-18)

*Han Liu, Yupeng Zhang, Bingning Wang, Weipeng Chen, Xiaolin Hu*

  Deep Neural Networks (DNNs) excel in various domains but face challenges in
providing accurate uncertainty estimates, which are crucial for high-stakes
applications. Large Language Models (LLMs) have recently emerged as powerful
tools, demonstrating exceptional performance in language tasks. However,
traditional calibration metrics such as Expected Calibration Error (ECE) and
classwise-ECE (cw-ECE) are inadequate for LLMs due to their vast vocabularies,
data complexity, and distributional focus. To address this, we propose a novel
calibration concept called full calibration and introduce its corresponding
metric, Full-ECE. Full-ECE evaluates the entire predicted probability
distribution, offering a more accurate and robust measure of calibration for
LLMs.


---

**[143. [2407.04307] Crafting Large Language Models for Enhanced Interpretability](https://arxiv.org/pdf/2407.04307.pdf)** (2024-07-08)

*Chung-En Sun, Tuomas Oikarinen, Tsui-Wei Weng*

  We introduce the Concept Bottleneck Large Language Model (CB-LLM), a
pioneering approach to creating inherently interpretable Large Language Models
(LLMs). Unlike traditional black-box LLMs that rely on post-hoc interpretation
methods with limited neuron function insights, CB-LLM sets a new standard with
its built-in interpretability, scalability, and ability to provide clear,
accurate explanations. This innovation not only advances transparency in
language models but also enhances their effectiveness. Our unique Automatic
Concept Correction (ACC) strategy successfully narrows the performance gap with
conventional black-box LLMs, positioning CB-LLM as a model that combines the
high accuracy of traditional LLMs with the added benefit of clear
interpretability -- a feature markedly absent in existing LLMs.


---

**[144. [2410.20290] Fast Best-of-N Decoding via Speculative Rejection](https://arxiv.org/pdf/2410.20290.pdf)** (2024-11-04)

*Hanshi Sun, Momin Haider, Ruiqi Zhang, Huitao Yang, Jiahao Qiu, Ming Yin, Mengdi Wang, Peter Bartlett, Andrea Zanette*

  The safe and effective deployment of Large Language Models (LLMs) involves a
critical step called alignment, which ensures that the model's responses are in
accordance with human preferences. Prevalent alignment techniques, such as DPO,
PPO and their variants, align LLMs by changing the pre-trained model weights
during a phase called post-training. While predominant, these post-training
methods add substantial complexity before LLMs can be deployed. Inference-time
alignment methods avoid the complex post-training step and instead bias the
generation towards responses that are aligned with human preferences. The
best-known inference-time alignment method, called Best-of-N, is as effective
as the state-of-the-art post-training procedures. Unfortunately, Best-of-N
requires vastly more resources at inference time than standard decoding
strategies, which makes it computationally not viable. In this work, we
introduce Speculative Rejection, a computationally-viable inference-time
alignment algorithm. It generates high-scoring responses according to a given
reward model, like Best-of-N does, while being between 16 to 32 times more
computationally efficient.


---

**[145. [2303.11455] Large Language Models and Simple, Stupid Bugs](https://arxiv.org/pdf/2303.11455.pdf)** (2023-03-22)

*Kevin Jesse, Toufique Ahmed, Premkumar T. Devanbu, Emily Morgan*

  With the advent of powerful neural language models, AI-based systems to
assist developers in coding tasks are becoming widely available; Copilot is one
such system. Copilot uses Codex, a large language model (LLM), to complete code
conditioned on a preceding "prompt". Codex, however, is trained on public
GitHub repositories, viz., on code that may include bugs and vulnerabilities.
Previous studies [1], [2] show Codex reproduces vulnerabilities seen in
training. In this study, we examine how prone Codex is to generate an
interesting bug category, single statement bugs, commonly referred to as
simple, stupid bugs or SStuBs in the MSR community. We find that Codex and
similar LLMs do help avoid some SStuBs, but do produce known, verbatim SStuBs
as much as 2x as likely than known, verbatim correct code. We explore the
consequences of the Codex generated SStuBs and propose avoidance strategies
that suggest the possibility of reducing the production of known, verbatim
SStubs, and increase the possibility of producing known, verbatim fixes.


---

**[146. [2403.00807] Enhancing Cloud-Based Large Language Model Processing with Elasticsearch
  and Transformer Models](https://arxiv.org/pdf/2403.00807.pdf)** (2024-03-05)

*Chunhe Ni, Jiang Wu, Hongbo Wang, Wenran Lu, Chenwei Zhang*

  Large Language Models (LLMs) are a class of generative AI models built using
the Transformer network, capable of leveraging vast datasets to identify,
summarize, translate, predict, and generate language. LLMs promise to
revolutionize society, yet training these foundational models poses immense
challenges. Semantic vector search within large language models is a potent
technique that can significantly enhance search result accuracy and relevance.
Unlike traditional keyword-based search methods, semantic search utilizes the
meaning and context of words to grasp the intent behind queries and deliver
more precise outcomes. Elasticsearch emerges as one of the most popular tools
for implementing semantic search an exceptionally scalable and robust search
engine designed for indexing and searching extensive datasets. In this article,
we delve into the fundamentals of semantic search and explore how to harness
Elasticsearch and Transformer models to bolster large language model processing
paradigms. We gain a comprehensive understanding of semantic search principles
and acquire practical skills for implementing semantic search in real-world
model application scenarios.


---

**[147. [2502.01208] Almost Surely Safe Alignment of Large Language Models at Inference-Time](https://arxiv.org/pdf/2502.01208.pdf)** (2025-02-06)

*Xiaotong Ji, Shyam Sundhar Ramesh, Matthieu Zimmer, Ilija Bogunovic, Jun Wang, Haitham Bou Ammar*

  Even highly capable large language models (LLMs) can produce biased or unsafe
responses, and alignment techniques, such as RLHF, aimed at mitigating this
issue, are expensive and prone to overfitting as they retrain the LLM. This
paper introduces a novel inference-time alignment approach that ensures LLMs
generate safe responses almost surely, i.e., with a probability approaching
one. We achieve this by framing the safe generation of inference-time responses
as a constrained Markov decision process within the LLM's latent space.
Crucially, we augment a safety state that tracks the evolution of safety
constraints and enables us to demonstrate formal safety guarantees upon solving
the MDP in the latent space. Building on this foundation, we propose
InferenceGuard, a practical implementation that safely aligns LLMs without
modifying the model weights. Empirically, we demonstrate InferenceGuard
effectively balances safety and task performance, outperforming existing
inference-time alignment methods in generating safe and aligned responses.


---

**[148. [2504.01550] Representation Bending for Large Language Model Safety](https://arxiv.org/pdf/2504.01550.pdf)** (2025-04-03)

*Ashkan Yousefpour, Taeheon Kim, Ryan S. Kwon, Seungbeen Lee, Wonje Jeung, Seungju Han, Alvin Wan, Harrison Ngan, Youngjae Yu, Jonghyun Choi*

  Large Language Models (LLMs) have emerged as powerful tools, but their
inherent safety risks - ranging from harmful content generation to broader
societal harms - pose significant challenges. These risks can be amplified by
the recent adversarial attacks, fine-tuning vulnerabilities, and the increasing
deployment of LLMs in high-stakes environments. Existing safety-enhancing
techniques, such as fine-tuning with human feedback or adversarial training,
are still vulnerable as they address specific threats and often fail to
generalize across unseen attacks, or require manual system-level defenses. This
paper introduces RepBend, a novel approach that fundamentally disrupts the
representations underlying harmful behaviors in LLMs, offering a scalable
solution to enhance (potentially inherent) safety. RepBend brings the idea of
activation steering - simple vector arithmetic for steering model's behavior
during inference - to loss-based fine-tuning. Through extensive evaluation,
RepBend achieves state-of-the-art performance, outperforming prior methods such
as Circuit Breaker, RMU, and NPO, with up to 95% reduction in attack success
rates across diverse jailbreak benchmarks, all with negligible reduction in
model usability and general capabilities.


---

**[149. [2407.20906] Automated Review Generation Method Based on Large Language Models](https://arxiv.org/pdf/2407.20906.pdf)** (2025-01-16)

*Shican Wu, Xiao Ma, Dehui Luo, Lulu Li, Xiangcheng Shi, Xin Chang, Xiaoyun Lin, Ran Luo, Chunlei Pei, Changying Du, Zhi-Jian Zhao, Jinlong Gong*

  Literature research, vital for scientific work, faces the challenge of
surging information volumes exceeding researchers' processing capabilities. We
present an automated review generation method based on large language models
(LLMs) to overcome efficiency bottlenecks and reduce cognitive load. Our
statistically validated evaluation framework demonstrates that the generated
reviews match or exceed manual quality, offering broad applicability across
research fields without requiring users' domain knowledge. Applied to propane
dehydrogenation (PDH) catalysts, our method swiftly analyzed 343 articles,
averaging seconds per article per LLM account, producing comprehensive reviews
spanning 35 topics, with extended analysis of 1041 articles providing insights
into catalysts' properties. Through multi-layered quality control, we
effectively mitigated LLMs' hallucinations, with expert verification confirming
accuracy and citation integrity while demonstrating hallucination risks reduced
to below 0.5\% with 95\% confidence. Released Windows application enables
one-click review generation, enhancing research productivity and literature
recommendation efficiency while setting the stage for broader scientific
explorations.


---

**[150. [2401.12246] Orion-14B: Open-source Multilingual Large Language Models](https://arxiv.org/pdf/2401.12246.pdf)** (2024-01-24)

*Du Chen, Yi Huang, Xiaopu Li, Yongqiang Li, Yongqiang Liu, Haihui Pan, Leichao Xu, Dacheng Zhang, Zhipeng Zhang, Kun Han*

  In this study, we introduce Orion-14B, a collection of multilingual large
language models with 14 billion parameters. We utilize a data scheduling
approach to train a foundational model on a diverse corpus of 2.5 trillion
tokens, sourced from texts in English, Chinese, Japanese, Korean, and other
languages. Additionally, we fine-tuned a series of models tailored for
conversational applications and other specific use cases. Our evaluation
results demonstrate that Orion-14B achieves state-of-the-art performance across
a broad spectrum of tasks. We make the Orion-14B model family and its
associated code publicly accessible https://github.com/OrionStarAI/Orion,
aiming to inspire future research and practical applications in the field.


---

**[151. [2405.05610] Chain of Attack: a Semantic-Driven Contextual Multi-Turn attacker for
  LLM](https://arxiv.org/pdf/2405.05610.pdf)** (2024-05-10)

*Xikang Yang, Xuehai Tang, Songlin Hu, Jizhong Han*

  Large language models (LLMs) have achieved remarkable performance in various
natural language processing tasks, especially in dialogue systems. However, LLM
may also pose security and moral threats, especially in multi round
conversations where large models are more easily guided by contextual content,
resulting in harmful or biased responses. In this paper, we present a novel
method to attack LLMs in multi-turn dialogues, called CoA (Chain of Attack).
CoA is a semantic-driven contextual multi-turn attack method that adaptively
adjusts the attack policy through contextual feedback and semantic relevance
during multi-turn of dialogue with a large model, resulting in the model
producing unreasonable or harmful content. We evaluate CoA on different LLMs
and datasets, and show that it can effectively expose the vulnerabilities of
LLMs, and outperform existing attack methods. Our work provides a new
perspective and tool for attacking and defending LLMs, and contributes to the
security and ethical assessment of dialogue systems.


---

**[152. [2406.13138] Large Language Models are Biased Because They Are Large Language Models](https://arxiv.org/pdf/2406.13138.pdf)** (2025-03-17)

*Philip Resnik*

  This position paper's primary goal is to provoke thoughtful discussion about
the relationship between bias and fundamental properties of large language
models. I do this by seeking to convince the reader that harmful biases are an
inevitable consequence arising from the design of any large language model as
LLMs are currently formulated. To the extent that this is true, it suggests
that the problem of harmful bias cannot be properly addressed without a serious
reconsideration of AI driven by LLMs, going back to the foundational
assumptions underlying their design.


---

**[153. [2309.14517] Watch Your Language: Investigating Content Moderation with Large
  Language Models](https://arxiv.org/pdf/2309.14517.pdf)** (2024-01-18)

*Deepak Kumar, Yousef AbuHashem, Zakir Durumeric*

  Large language models (LLMs) have exploded in popularity due to their ability
to perform a wide array of natural language tasks. Text-based content
moderation is one LLM use case that has received recent enthusiasm, however,
there is little research investigating how LLMs perform in content moderation
settings. In this work, we evaluate a suite of commodity LLMs on two common
content moderation tasks: rule-based community moderation and toxic content
detection. For rule-based community moderation, we instantiate 95 subcommunity
specific LLMs by prompting GPT-3.5 with rules from 95 Reddit subcommunities. We
find that GPT-3.5 is effective at rule-based moderation for many communities,
achieving a median accuracy of 64% and a median precision of 83%. For toxicity
detection, we evaluate a suite of commodity LLMs (GPT-3, GPT-3.5, GPT-4, Gemini
Pro, LLAMA 2) and show that LLMs significantly outperform currently widespread
toxicity classifiers. However, recent increases in model size add only marginal
benefit to toxicity detection, suggesting a potential performance plateau for
LLMs on toxicity detection tasks. We conclude by outlining avenues for future
work in studying LLMs and content moderation.


---

**[154. [2403.08035] Harnessing Artificial Intelligence to Combat Online Hate: Exploring the
  Challenges and Opportunities of Large Language Models in Hate Speech
  Detection](https://arxiv.org/pdf/2403.08035.pdf)** (2024-03-14)

*Tharindu Kumarage, Amrita Bhattacharjee, Joshua Garland*

  Large language models (LLMs) excel in many diverse applications beyond
language generation, e.g., translation, summarization, and sentiment analysis.
One intriguing application is in text classification. This becomes pertinent in
the realm of identifying hateful or toxic speech -- a domain fraught with
challenges and ethical dilemmas. In our study, we have two objectives: firstly,
to offer a literature review revolving around LLMs as classifiers, emphasizing
their role in detecting and classifying hateful or toxic content. Subsequently,
we explore the efficacy of several LLMs in classifying hate speech: identifying
which LLMs excel in this task as well as their underlying attributes and
training. Providing insight into the factors that contribute to an LLM
proficiency (or lack thereof) in discerning hateful content. By combining a
comprehensive literature review with an empirical analysis, our paper strives
to shed light on the capabilities and constraints of LLMs in the crucial domain
of hate speech detection.


---

**[155. [2503.04490] Large Language Models in Bioinformatics: A Survey](https://arxiv.org/pdf/2503.04490.pdf)** (2025-03-07)

*Zhenyu Wang, Zikang Wang, Jiyue Jiang, Pengan Chen, Xiangyu Shi, Yu Li*

  Large Language Models (LLMs) are revolutionizing bioinformatics, enabling
advanced analysis of DNA, RNA, proteins, and single-cell data. This survey
provides a systematic review of recent advancements, focusing on genomic
sequence modeling, RNA structure prediction, protein function inference, and
single-cell transcriptomics. Meanwhile, we also discuss several key challenges,
including data scarcity, computational complexity, and cross-omics integration,
and explore future directions such as multimodal learning, hybrid AI models,
and clinical applications. By offering a comprehensive perspective, this paper
underscores the transformative potential of LLMs in driving innovations in
bioinformatics and precision medicine.


---

**[156. [2404.03647] Capabilities of Large Language Models in Control Engineering: A
  Benchmark Study on GPT-4, Claude 3 Opus, and Gemini 1.0 Ultra](https://arxiv.org/pdf/2404.03647.pdf)** (2024-04-05)

*Darioush Kevian, Usman Syed, Xingang Guo, Aaron Havens, Geir Dullerud, Peter Seiler, Lianhui Qin, Bin Hu*

  In this paper, we explore the capabilities of state-of-the-art large language
models (LLMs) such as GPT-4, Claude 3 Opus, and Gemini 1.0 Ultra in solving
undergraduate-level control problems. Controls provides an interesting case
study for LLM reasoning due to its combination of mathematical theory and
engineering design. We introduce ControlBench, a benchmark dataset tailored to
reflect the breadth, depth, and complexity of classical control design. We use
this dataset to study and evaluate the problem-solving abilities of these LLMs
in the context of control engineering. We present evaluations conducted by a
panel of human experts, providing insights into the accuracy, reasoning, and
explanatory prowess of LLMs in control engineering. Our analysis reveals the
strengths and limitations of each LLM in the context of classical control, and
our results imply that Claude 3 Opus has become the state-of-the-art LLM for
solving undergraduate control problems. Our study serves as an initial step
towards the broader goal of employing artificial general intelligence in
control engineering.


---

**[157. [2402.13926] Large Language Models are Vulnerable to Bait-and-Switch Attacks for
  Generating Harmful Content](https://arxiv.org/pdf/2402.13926.pdf)** (2024-02-22)

*Federico Bianchi, James Zou*

  The risks derived from large language models (LLMs) generating deceptive and
damaging content have been the subject of considerable research, but even safe
generations can lead to problematic downstream impacts. In our study, we shift
the focus to how even safe text coming from LLMs can be easily turned into
potentially dangerous content through Bait-and-Switch attacks. In such attacks,
the user first prompts LLMs with safe questions and then employs a simple
find-and-replace post-hoc technique to manipulate the outputs into harmful
narratives. The alarming efficacy of this approach in generating toxic content
highlights a significant challenge in developing reliable safety guardrails for
LLMs. In particular, we stress that focusing on the safety of the verbatim LLM
outputs is insufficient and that we also need to consider post-hoc
transformations.


---

**[158. [2405.00566] NumLLM: Numeric-Sensitive Large Language Model for Chinese Finance](https://arxiv.org/pdf/2405.00566.pdf)** (2024-05-02)

*Huan-Yi Su, Ke Wu, Yu-Hao Huang, Wu-Jun Li*

  Recently, many works have proposed various financial large language models
(FinLLMs) by pre-training from scratch or fine-tuning open-sourced LLMs on
financial corpora. However, existing FinLLMs exhibit unsatisfactory performance
in understanding financial text when numeric variables are involved in
questions. In this paper, we propose a novel LLM, called numeric-sensitive
large language model (NumLLM), for Chinese finance. We first construct a
financial corpus from financial textbooks which is essential for improving
numeric capability of LLMs during fine-tuning. After that, we train two
individual low-rank adaptation (LoRA) modules by fine-tuning on our constructed
financial corpus. One module is for adapting general-purpose LLMs to financial
domain, and the other module is for enhancing the ability of NumLLM to
understand financial text with numeric variables. Lastly, we merge the two LoRA
modules into the foundation model to obtain NumLLM for inference. Experiments
on financial question-answering benchmark show that NumLLM can boost the
performance of the foundation model and can achieve the best overall
performance compared to all baselines, on both numeric and non-numeric
questions.


---

**[159. [2407.19947] Inference acceleration for large language models using "stairs" assisted
  greedy generation](https://arxiv.org/pdf/2407.19947.pdf)** (2024-07-30)

*Domas Grigaliūnas, Mantas Lukoševičius*

  Large Language Models (LLMs) with billions of parameters are known for their
impressive predicting capabilities but require lots of resources to run. With
their massive rise in popularity, even a small reduction in required resources
could have an impact on environment. On the other hand, smaller models require
fewer resources but may sacrifice accuracy. In this work, we are proposing an
implementation of ``stairs'' assisted greedy generation. It is a modified
assisted generation methodology that makes use of a smaller model's fast
generation, large model's batch prediction, and "stairs" validation in order to
achieve a speed up in prediction generation. Results show between 9.58 and
17.24 percent inference time reduction compared to a stand-alone large LLM
prediction in a text generation task without a loss in accuracy.


---

**[160. [2310.02469] PrivacyMind: Large Language Models Can Be Contextual Privacy Protection
  Learners](https://arxiv.org/pdf/2310.02469.pdf)** (2024-10-29)

*Yijia Xiao, Yiqiao Jin, Yushi Bai, Yue Wu, Xianjun Yang, Xiao Luo, Wenchao Yu, Xujiang Zhao, Yanchi Liu, Quanquan Gu, Haifeng Chen, Wei Wang, Wei Cheng*

  The proliferation of Large Language Models (LLMs) has driven considerable
interest in fine-tuning them with domain-specific data to create specialized
language models. Nevertheless, such domain-specific fine-tuning data often
contains contextually sensitive personally identifiable information (PII).
Direct fine-tuning of LLMs on this data without privacy protection poses a risk
of data leakage of sensitive PII during inference time. To address this
challenge, we introduce Contextual Privacy Protection Language Models
(PrivacyMind), a novel paradigm for fine-tuning LLMs that effectively injects
domain-specific knowledge while safeguarding inference-time data privacy. Our
work offers a theoretical analysis for model design and benchmarks various
techniques such as corpus curation, penalty-based unlikelihood in training
loss, instruction-based tuning, etc. Extensive experiments across diverse
datasets and scenarios demonstrate the effectiveness of our approaches. In
particular, instruction tuning with both positive and negative examples stands
out as a promising method, effectively protecting private data while enhancing
the model's knowledge. Our work underscores the potential for Large Language
Models as robust contextual privacy protection learners. The complete code and
data for the work can be found at https://github.com/Yijia-Xiao/PrivacyMind.


---

**[161. [2306.17271] DisasterResponseGPT: Large Language Models for Accelerated Plan of
  Action Development in Disaster Response Scenarios](https://arxiv.org/pdf/2306.17271.pdf)** (2023-07-03)

*Vinicius G. Goecks, Nicholas R. Waytowich*

  The development of plans of action in disaster response scenarios is a
time-consuming process. Large Language Models (LLMs) offer a powerful solution
to expedite this process through in-context learning. This study presents
DisasterResponseGPT, an algorithm that leverages LLMs to generate valid plans
of action quickly by incorporating disaster response and planning guidelines in
the initial prompt. In DisasterResponseGPT, users input the scenario
description and receive a plan of action as output. The proposed method
generates multiple plans within seconds, which can be further refined following
the user's feedback. Preliminary results indicate that the plans of action
developed by DisasterResponseGPT are comparable to human-generated ones while
offering greater ease of modification in real-time. This approach has the
potential to revolutionize disaster response operations by enabling rapid
updates and adjustments during the plan's execution.


---

**[162. [2503.16740] Automated Harmfulness Testing for Code Large Language Models](https://arxiv.org/pdf/2503.16740.pdf)** (2025-03-24)

*Honghao Tan, Haibo Wang, Diany Pressato, Yisen Xu, Shin Hwei Tan*

  Generative AI systems powered by Large Language Models (LLMs) usually use
content moderation to prevent harmful content spread. To evaluate the
robustness of content moderation, several metamorphic testing techniques have
been proposed to test content moderation software. However, these techniques
mainly focus on general users (e.g., text and image generation). Meanwhile, a
recent study shows that developers consider using harmful keywords when naming
software artifacts to be an unethical behavior. Exposure to harmful content in
software artifacts can negatively impact the mental health of developers,
making content moderation for Code Large Language Models (Code LLMs) essential.
We conduct a preliminary study on program transformations that can be misused
to introduce harmful content into auto-generated code, identifying 32 such
transformations. To address this, we propose CHT, a coverage-guided harmfulness
testing framework that generates prompts using diverse transformations and
harmful keywords injected into benign programs. CHT evaluates output damage to
assess potential risks in LLM-generated explanations and code. Our evaluation
of four Code LLMs and GPT-4o-mini reveals that content moderation in LLM-based
code generation is easily bypassed. To enhance moderation, we propose a
two-phase approach that first detects harmful content before generating output,
improving moderation effectiveness by 483.76\%.


---

**[163. [2410.12462] Bridging the Language Gaps in Large Language Models with Inference-Time
  Cross-Lingual Intervention](https://arxiv.org/pdf/2410.12462.pdf)** (2024-10-17)

*Weixuan Wang, Minghao Wu, Barry Haddow, Alexandra Birch*

  Large Language Models (LLMs) have shown remarkable capabilities in natural
language processing but exhibit significant performance gaps among different
languages. Most existing approaches to address these disparities rely on
pretraining or fine-tuning, which are resource-intensive. To overcome these
limitations without incurring significant costs, we propose Inference-Time
Cross-Lingual Intervention (INCLINE), a novel framework that enhances LLM
performance on low-performing (source) languages by aligning their internal
representations with those of high-performing (target) languages during
inference. INCLINE initially learns alignment matrices using parallel sentences
from source and target languages through a Least-Squares optimization, and then
applies these matrices during inference to transform the low-performing
language representations toward the high-performing language space. Extensive
experiments on nine benchmarks with five LLMs demonstrate that INCLINE
significantly improves performance across diverse tasks and languages, compared
to recent strong baselines. Our analysis demonstrates that INCLINE is highly
cost-effective and applicable to a wide range of applications. In addition, we
release the code to foster research along this line:
https://github.com/weixuan-wang123/INCLINE.


---

**[164. [2407.01212] EconNLI: Evaluating Large Language Models on Economics Reasoning](https://arxiv.org/pdf/2407.01212.pdf)** (2024-07-02)

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

**[165. [2305.19234] Grammar Prompting for Domain-Specific Language Generation with Large
  Language Models](https://arxiv.org/pdf/2305.19234.pdf)** (2023-11-06)

*Bailin Wang, Zi Wang, Xuezhi Wang, Yuan Cao, Rif A. Saurous, Yoon Kim*

  Large language models (LLMs) can learn to perform a wide range of natural
language tasks from just a handful of in-context examples. However, for
generating strings from highly structured languages (e.g., semantic parsing to
complex domain-specific languages), it is challenging for the LLM to generalize
from just a few exemplars. We propose \emph{grammar prompting}, a simple
approach to enable LLMs to use external knowledge and domain-specific
constraints, expressed through a grammar in Backus--Naur Form (BNF), during
in-context learning. Grammar prompting augments each demonstration example with
a specialized grammar that is minimally sufficient for generating the
particular output example, where the specialized grammar is a subset of the
full DSL grammar. For inference, the LLM first predicts a BNF grammar given a
test input, and then generates the output according to the rules of the
grammar. Experiments demonstrate that grammar prompting can enable LLMs to
perform competitively on a diverse set of DSL generation tasks, including
semantic parsing (SMCalFlow, Overnight, GeoQuery), PDDL planning, and
SMILES-based molecule generation.


---

**[166. [2407.13945] FANTAstic SEquences and Where to Find Them: Faithful and Efficient API
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

**[167. [2403.02615] Exploring the Limitations of Large Language Models in Compositional
  Relation Reasoning](https://arxiv.org/pdf/2403.02615.pdf)** (2024-09-24)

*Jinman Zhao, Xueyan Zhang*

  We present a comprehensive evaluation of large language models(LLMs)' ability
to reason about composition relations through a benchmark encompassing 1,500
test cases in English, designed to cover six distinct types of composition
relations: Positional, Comparative, Personal, Mathematical, Identity, and
Other. Acknowledging the significance of multilingual capabilities, we expanded
our assessment to include translations of these cases into Chinese, Japanese,
French, and Korean. Our Multilingual Composition Relation (MCR) benchmark aims
at investigating the robustness and adaptability of LLMs in handling
composition relation reasoning across diverse linguistic contexts.


---

**[168. [2304.06975] HuaTuo: Tuning LLaMA Model with Chinese Medical Knowledge](https://arxiv.org/pdf/2304.06975.pdf)** (2023-04-17)

*Haochun Wang, Chi Liu, Nuwa Xi, Zewen Qiang, Sendong Zhao, Bing Qin, Ting Liu*

  Large Language Models (LLMs), such as the LLaMA model, have demonstrated
their effectiveness in various general-domain natural language processing (NLP)
tasks. Nevertheless, LLMs have not yet performed optimally in biomedical domain
tasks due to the need for medical expertise in the responses. In response to
this challenge, we propose HuaTuo, a LLaMA-based model that has been
supervised-fine-tuned with generated QA (Question-Answer) instances. The
experimental results demonstrate that HuaTuo generates responses that possess
more reliable medical knowledge. Our proposed HuaTuo model is accessible at
https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese.


---

**[169. [2411.07268] Target-driven Attack for Large Language Models](https://arxiv.org/pdf/2411.07268.pdf)** (2024-11-14)

*Chong Zhang, Mingyu Jin, Dong Shu, Taowen Wang, Dongfang Liu, Xiaobo Jin*

  Current large language models (LLM) provide a strong foundation for
large-scale user-oriented natural language tasks. Many users can easily inject
adversarial text or instructions through the user interface, thus causing LLM
model security challenges like the language model not giving the correct
answer. Although there is currently a large amount of research on black-box
attacks, most of these black-box attacks use random and heuristic strategies.
It is unclear how these strategies relate to the success rate of attacks and
thus effectively improve model robustness. To solve this problem, we propose
our target-driven black-box attack method to maximize the KL divergence between
the conditional probabilities of the clean text and the attack text to redefine
the attack's goal. We transform the distance maximization problem into two
convex optimization problems based on the attack goal to solve the attack text
and estimate the covariance. Furthermore, the projected gradient descent
algorithm solves the vector corresponding to the attack text. Our target-driven
black-box attack approach includes two attack strategies: token manipulation
and misinformation attack. Experimental results on multiple Large Language
Models and datasets demonstrate the effectiveness of our attack method.


---

**[170. [2411.08181] Challenges in Guardrailing Large Language Models for Science](https://arxiv.org/pdf/2411.08181.pdf)** (2024-12-05)

*Nishan Pantha, Muthukumaran Ramasubramanian, Iksha Gurung, Manil Maskey, Rahul Ramachandran*

  The rapid development in large language models (LLMs) has transformed the
landscape of natural language processing and understanding (NLP/NLU), offering
significant benefits across various domains. However, when applied to
scientific research, these powerful models exhibit critical failure modes
related to scientific integrity and trustworthiness. Existing general-purpose
LLM guardrails are insufficient to address these unique challenges in the
scientific domain. We provide comprehensive guidelines for deploying LLM
guardrails in the scientific domain. We identify specific challenges --
including time sensitivity, knowledge contextualization, conflict resolution,
and intellectual property concerns -- and propose a guideline framework for the
guardrails that can align with scientific needs. These guardrail dimensions
include trustworthiness, ethics & bias, safety, and legal aspects. We also
outline in detail the implementation strategies that employ white-box,
black-box, and gray-box methodologies that can be enforced within scientific
contexts.


---

**[171. [2405.10825] Large Language Model (LLM) for Telecommunications: A Comprehensive
  Survey on Principles, Key Techniques, and Opportunities](https://arxiv.org/pdf/2405.10825.pdf)** (2024-09-17)

*Hao Zhou, Chengming Hu, Ye Yuan, Yufei Cui, Yili Jin, Can Chen, Haolun Wu, Dun Yuan, Li Jiang, Di Wu, Xue Liu, Charlie Zhang, Xianbin Wang, Jiangchuan Liu*

  Large language models (LLMs) have received considerable attention recently
due to their outstanding comprehension and reasoning capabilities, leading to
great progress in many fields. The advancement of LLM techniques also offers
promising opportunities to automate many tasks in the telecommunication
(telecom) field. After pre-training and fine-tuning, LLMs can perform diverse
downstream tasks based on human instructions, paving the way to artificial
general intelligence (AGI)-enabled 6G. Given the great potential of LLM
technologies, this work aims to provide a comprehensive overview of LLM-enabled
telecom networks. In particular, we first present LLM fundamentals, including
model architecture, pre-training, fine-tuning, inference and utilization, model
evaluation, and telecom deployment. Then, we introduce LLM-enabled key
techniques and telecom applications in terms of generation, classification,
optimization, and prediction problems. Specifically, the LLM-enabled generation
applications include telecom domain knowledge, code, and network configuration
generation. After that, the LLM-based classification applications involve
network security, text, image, and traffic classification problems. Moreover,
multiple LLM-enabled optimization techniques are introduced, such as automated
reward function design for reinforcement learning and verbal reinforcement
learning. Furthermore, for LLM-aided prediction problems, we discussed
time-series prediction models and multi-modality prediction problems for
telecom. Finally, we highlight the challenges and identify the future
directions of LLM-enabled telecom networks.


---

**[172. [2406.07081] Efficiently Exploring Large Language Models for Document-Level Machine
  Translation with In-context Learning](https://arxiv.org/pdf/2406.07081.pdf)** (2024-06-12)

*Menglong Cui, Jiangcun Du, Shaolin Zhu, Deyi Xiong*

  Large language models (LLMs) exhibit outstanding performance in machine
translation via in-context learning. In contrast to sentence-level translation,
document-level translation (DOCMT) by LLMs based on in-context learning faces
two major challenges: firstly, document translations generated by LLMs are
often incoherent; secondly, the length of demonstration for in-context learning
is usually limited. To address these issues, we propose a Context-Aware
Prompting method (CAP), which enables LLMs to generate more accurate, cohesive,
and coherent translations via in-context learning. CAP takes into account
multi-level attention, selects the most relevant sentences to the current one
as context, and then generates a summary from these collected sentences.
Subsequently, sentences most similar to the summary are retrieved from the
datastore as demonstrations, which effectively guide LLMs in generating
cohesive and coherent translations. We conduct extensive experiments across
various DOCMT tasks, and the results demonstrate the effectiveness of our
approach, particularly in zero pronoun translation (ZPT) and literary
translation tasks.


---

**[173. [2308.12247] How to Protect Copyright Data in Optimization of Large Language Models?](https://arxiv.org/pdf/2308.12247.pdf)** (2023-08-24)

*Timothy Chu, Zhao Song, Chiwun Yang*

  Large language models (LLMs) and generative AI have played a transformative
role in computer research and applications. Controversy has arisen as to
whether these models output copyrighted data, which can occur if the data the
models are trained on is copyrighted. LLMs are built on the transformer neural
network architecture, which in turn relies on a mathematical computation called
Attention that uses the softmax function.
  In this paper, we show that large language model training and optimization
can be seen as a softmax regression problem. We then establish a method of
efficiently performing softmax regression, in a way that prevents the
regression function from generating copyright data. This establishes a
theoretical method of training large language models in a way that avoids
generating copyright data.


---

**[174. [2401.14242] Improving Natural Language Capability of Code Large Language Model](https://arxiv.org/pdf/2401.14242.pdf)** (2024-01-26)

*Wei Li, Daoguang Zan, Bei Guan, Ailun Yu, Xiaolin Chen, Yongji Wang*

  Code large language models (Code LLMs) have demonstrated remarkable
performance in code generation. Nonetheless, most existing works focus on
boosting code LLMs from the perspective of programming capabilities, while
their natural language capabilities receive less attention. To fill this gap,
we thus propose a novel framework, comprising two modules: AttentionExtractor,
which is responsible for extracting key phrases from the user's natural
language requirements, and AttentionCoder, which leverages these extracted
phrases to generate target code to solve the requirement. This framework
pioneers an innovative idea by seamlessly integrating code LLMs with
traditional natural language processing tools. To validate the effectiveness of
the framework, we craft a new code generation benchmark, called MultiNL-H,
covering five natural languages. Extensive experimental results demonstrate the
effectiveness of our proposed framework.


---

**[175. [2406.12513] Can We Trust Large Language Models Generated Code? A Framework for
  In-Context Learning, Security Patterns, and Code Evaluations Across Diverse
  LLMs](https://arxiv.org/pdf/2406.12513.pdf)** (2024-12-03)

*Ahmad Mohsin, Helge Janicke, Adrian Wood, Iqbal H. Sarker, Leandros Maglaras, Naeem Janjua*

  Large Language Models (LLMs) such as ChatGPT and GitHub Copilot have
revolutionized automated code generation in software engineering. However, as
these models are increasingly utilized for software development, concerns have
arisen regarding the security and quality of the generated code. These concerns
stem from LLMs being primarily trained on publicly available code repositories
and internet-based textual data, which may contain insecure code. This presents
a significant risk of perpetuating vulnerabilities in the generated code,
creating potential attack vectors for exploitation by malicious actors. Our
research aims to tackle these issues by introducing a framework for secure
behavioral learning of LLMs through In-Content Learning (ICL) patterns during
the code generation process, followed by rigorous security evaluations. To
achieve this, we have selected four diverse LLMs for experimentation. We have
evaluated these coding LLMs across three programming languages and identified
security vulnerabilities and code smells. The code is generated through ICL
with curated problem sets and undergoes rigorous security testing to evaluate
the overall quality and trustworthiness of the generated code. Our research
indicates that ICL-driven one-shot and few-shot learning patterns can enhance
code security, reducing vulnerabilities in various programming scenarios.
Developers and researchers should know that LLMs have a limited understanding
of security principles. This may lead to security breaches when the generated
code is deployed in production systems. Our research highlights LLMs are a
potential source of new vulnerabilities to the software supply chain. It is
important to consider this when using LLMs for code generation. This research
article offers insights into improving LLM security and encourages proactive
use of LLMs for code generation to ensure software system safety.


---

**[176. [2411.02317] Defining and Evaluating Physical Safety for Large Language Models](https://arxiv.org/pdf/2411.02317.pdf)** (2024-11-05)

*Yung-Chen Tang, Pin-Yu Chen, Tsung-Yi Ho*

  Large Language Models (LLMs) are increasingly used to control robotic systems
such as drones, but their risks of causing physical threats and harm in
real-world applications remain unexplored. Our study addresses the critical gap
in evaluating LLM physical safety by developing a comprehensive benchmark for
drone control. We classify the physical safety risks of drones into four
categories: (1) human-targeted threats, (2) object-targeted threats, (3)
infrastructure attacks, and (4) regulatory violations. Our evaluation of
mainstream LLMs reveals an undesirable trade-off between utility and safety,
with models that excel in code generation often performing poorly in crucial
safety aspects. Furthermore, while incorporating advanced prompt engineering
techniques such as In-Context Learning and Chain-of-Thought can improve safety,
these methods still struggle to identify unintentional attacks. In addition,
larger models demonstrate better safety capabilities, particularly in refusing
dangerous commands. Our findings and benchmark can facilitate the design and
evaluation of physical safety for LLMs. The project page is available at
huggingface.co/spaces/TrustSafeAI/LLM-physical-safety.


---

**[177. [2402.11420] Rethinking the Roles of Large Language Models in Chinese Grammatical
  Error Correction](https://arxiv.org/pdf/2402.11420.pdf)** (2024-09-20)

*Yinghui Li, Shang Qin, Haojing Huang, Yangning Li, Libo Qin, Xuming Hu, Wenhao Jiang, Hai-Tao Zheng, Philip S. Yu*

  Recently, Large Language Models (LLMs) have been widely studied by
researchers for their roles in various downstream NLP tasks. As a fundamental
task in the NLP field, Chinese Grammatical Error Correction (CGEC) aims to
correct all potential grammatical errors in the input sentences. Previous
studies have shown that LLMs' performance as correctors on CGEC remains
unsatisfactory due to its challenging task focus. To promote the CGEC field to
better adapt to the era of LLMs, we rethink the roles of LLMs in the CGEC task
so that they can be better utilized and explored in CGEC. Considering the rich
grammatical knowledge stored in LLMs and their powerful semantic understanding
capabilities, we utilize LLMs as explainers to provide explanation information
for the CGEC small models during error correction to enhance performance. We
also use LLMs as evaluators to bring more reasonable CGEC evaluations, thus
alleviating the troubles caused by the subjectivity of the CGEC task. In
particular, our work is also an active exploration of how LLMs and small models
better collaborate in downstream tasks. Extensive experiments and detailed
analyses on widely used datasets verify the effectiveness of our thinking
intuition and the proposed methods.


---

**[178. [2312.04916] EE-LLM: Large-Scale Training and Inference of Early-Exit Large Language
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

**[179. [2502.09209] On LLM-generated Logic Programs and their Inference Execution Methods](https://arxiv.org/pdf/2502.09209.pdf)** (2025-02-14)

*University of North Texas  Paul Tarau*

  Large Language Models (LLMs) trained on petabytes of data are highly
compressed repositories of a significant proportion of the knowledge
accumulated and distilled so far. In this paper we study techniques to elicit
this knowledge in the form of several classes of logic programs, including
propositional Horn clauses, Dual Horn clauses, relational triplets and Definite
Clause Grammars. Exposing this knowledge as logic programs enables sound
reasoning methods that can verify alignment of LLM outputs to their intended
uses and extend their inference capabilities. We study new execution methods
for the generated programs, including soft-unification of abducible facts
against LLM-generated content stored in a vector database as well as GPU-based
acceleration of minimal model computation that supports inference with large
LLM-generated programs.


---

**[180. [2408.04643] Risks, Causes, and Mitigations of Widespread Deployments of Large
  Language Models (LLMs): A Survey](https://arxiv.org/pdf/2408.04643.pdf)** (2024-08-12)

*Md Nazmus Sakib, Md Athikul Islam, Royal Pathak, Md Mashrur Arifin*

  Recent advancements in Large Language Models (LLMs), such as ChatGPT and
LLaMA, have significantly transformed Natural Language Processing (NLP) with
their outstanding abilities in text generation, summarization, and
classification. Nevertheless, their widespread adoption introduces numerous
challenges, including issues related to academic integrity, copyright,
environmental impacts, and ethical considerations such as data bias, fairness,
and privacy. The rapid evolution of LLMs also raises concerns regarding the
reliability and generalizability of their evaluations. This paper offers a
comprehensive survey of the literature on these subjects, systematically
gathered and synthesized from Google Scholar. Our study provides an in-depth
analysis of the risks associated with specific LLMs, identifying sub-risks,
their causes, and potential solutions. Furthermore, we explore the broader
challenges related to LLMs, detailing their causes and proposing mitigation
strategies. Through this literature analysis, our survey aims to deepen the
understanding of the implications and complexities surrounding these powerful
models.


---

**[181. [2406.01133] Impact of Generative AI (Large Language Models) on the PRA model
  construction and maintenance, observations](https://arxiv.org/pdf/2406.01133.pdf)** (2024-06-17)

*EDF R\&D  Valentin Rychkov, EDF R\&D  Claudia Picoco, EDF R\&D  Emilie Caleca*

  The rapid development of Large Language Models (LLMs) and Generative
Pre-Trained Transformers(GPTs) in the field of Generative Artificial
Intelligence (AI) can significantly impact task automation in themodern
economy. We anticipate that the PRA field will inevitably be affected by this
technology. Thus, themain goal of this paper is to engage the risk assessment
community into a discussion of benefits anddrawbacks of this technology for
PRA. We make a preliminary analysis of possible application of LLM
inProbabilistic Risk Assessment (PRA) modeling context referring to the ongoing
experience in softwareengineering field. We explore potential application
scenarios and the necessary conditions for controlledLLM usage in PRA modeling
(whether static or dynamic). Additionally, we consider the potential impact
ofthis technology on PRA modeling tools.


---

**[182. [2407.13490] Combining Constraint Programming Reasoning with Large Language Model
  Predictions](https://arxiv.org/pdf/2407.13490.pdf)** (2024-09-26)

*Florian Régin, Elisabetta De Maria, Alexandre Bonlarron*

  Constraint Programming (CP) and Machine Learning (ML) face challenges in text
generation due to CP's struggle with implementing "meaning'' and ML's
difficulty with structural constraints. This paper proposes a solution by
combining both approaches and embedding a Large Language Model (LLM) in CP. The
LLM handles word generation and meaning, while CP manages structural
constraints. This approach builds on GenCP, an improved version of On-the-fly
Constraint Programming Search (OTFS) using LLM-generated domains. Compared to
Beam Search (BS), a standard NLP method, this combined approach (GenCP with
LLM) is faster and produces better results, ensuring all constraints are
satisfied. This fusion of CP and ML presents new possibilities for enhancing
text generation under constraints.


---

**[183. [2503.21598] Prompt, Divide, and Conquer: Bypassing Large Language Model Safety
  Filters via Segmented and Distributed Prompt Processing](https://arxiv.org/pdf/2503.21598.pdf)** (2025-04-01)

*Johan Wahréus, Ahmed Hussain, Panos Papadimitratos*

  Large Language Models (LLMs) have transformed task automation and content
generation across various domains while incorporating safety filters to prevent
misuse. We introduce a novel jailbreaking framework that employs distributed
prompt processing combined with iterative refinements to bypass these safety
measures, particularly in generating malicious code. Our architecture consists
of four key modules: prompt segmentation, parallel processing, response
aggregation, and LLM-based jury evaluation. Tested on 500 malicious prompts
across 10 cybersecurity categories, the framework achieves a 73.2% Success Rate
(SR) in generating malicious code. Notably, our comparative analysis reveals
that traditional single-LLM judge evaluation overestimates SRs (93.8%) compared
to our LLM jury system (73.2%), with manual verification confirming that
single-judge assessments often accept incomplete implementations. Moreover, we
demonstrate that our distributed architecture improves SRs by 12% over the
non-distributed approach in an ablation study, highlighting both the
effectiveness of distributed prompt processing and the importance of robust
evaluation methodologies in assessing jailbreak attempts.


---

**[184. [2305.03851] Large Language Models in Sport Science & Medicine: Opportunities, Risks
  and Considerations](https://arxiv.org/pdf/2305.03851.pdf)** (2023-05-09)

*Mark Connor, Michael O'Neill*

  This paper explores the potential opportunities, risks, and challenges
associated with the use of large language models (LLMs) in sports science and
medicine. LLMs are large neural networks with transformer style architectures
trained on vast amounts of textual data, and typically refined with human
feedback. LLMs can perform a large range of natural language processing tasks.
In sports science and medicine, LLMs have the potential to support and augment
the knowledge of sports medicine practitioners, make recommendations for
personalised training programs, and potentially distribute high-quality
information to practitioners in developing countries. However, there are also
potential risks associated with the use and development of LLMs, including
biases in the dataset used to create the model, the risk of exposing
confidential data, the risk of generating harmful output, and the need to align
these models with human preferences through feedback. Further research is
needed to fully understand the potential applications of LLMs in sports science
and medicine and to ensure that their use is ethical and beneficial to
athletes, clients, patients, practitioners, and the general public.


---

**[185. [2404.18353] How secure is AI-generated Code: A Large-Scale Comparison of Large
  Language Models](https://arxiv.org/pdf/2404.18353.pdf)** (2024-12-12)

*Norbert Tihanyi, Tamas Bisztray, Mohamed Amine Ferrag, Ridhi Jain, Lucas C. Cordeiro*

  This study compares state-of-the-art Large Language Models (LLMs) on their
tendency to generate vulnerabilities when writing C programs using a neutral
zero-shot prompt. Tihanyi et al. introduced the FormAI dataset at PROMISE'23,
featuring 112,000 C programs generated by GPT-3.5-turbo, with over 51.24%
identified as vulnerable. We extended that research with a large-scale study
involving 9 state-of-the-art models such as OpenAI's GPT-4o-mini, Google's
Gemini Pro 1.0, TII's 180 billion-parameter Falcon, Meta's 13 billion-parameter
Code Llama, and several other compact models. Additionally, we introduce the
FormAI-v2 dataset, which comprises 331 000 compilable C programs generated by
these LLMs. Each program in the dataset is labeled based on the vulnerabilities
detected in its source code through formal verification, using the Efficient
SMT-based Context-Bounded Model Checker (ESBMC). This technique minimizes false
positives by providing a counterexample for the specific vulnerability and
reduces false negatives by thoroughly completing the verification process. Our
study reveals that at least 62.07% of the generated programs are vulnerable.
The differences between the models are minor, as they all show similar coding
errors with slight variations. Our research highlights that while LLMs offer
promising capabilities for code generation, deploying their output in a
production environment requires proper risk assessment and validation.


---

**[186. [2411.13826] Interactive and Expressive Code-Augmented Planning with Large Language
  Models](https://arxiv.org/pdf/2411.13826.pdf)** (2024-11-22)

*Anthony Z. Liu, Xinhe Wang, Jacob Sansom, Yao Fu, Jongwook Choi, Sungryull Sohn, Jaekyeom Kim, Honglak Lee*

  Large Language Models (LLMs) demonstrate strong abilities in common-sense
reasoning and interactive decision-making, but often struggle with complex,
long-horizon planning tasks. Recent techniques have sought to structure LLM
outputs using control flow and other code-adjacent techniques to improve
planning performance. These techniques include using variables (to track
important information) and functions (to divide complex tasks into smaller
re-usable sub-tasks). However, purely code-based approaches can be error-prone
and insufficient for handling ambiguous or unstructured data. To address these
challenges, we propose REPL-Plan, an LLM planning approach that is fully
code-expressive (it can utilize all the benefits of code) while also being
dynamic (it can flexibly adapt from errors and use the LLM for fuzzy
situations). In REPL-Plan, an LLM solves tasks by interacting with a
Read-Eval-Print Loop (REPL), which iteratively executes and evaluates code,
similar to language shells or interactive code notebooks, allowing the model to
flexibly correct errors and handle tasks dynamically. We demonstrate that
REPL-Plan achieves strong results across various planning domains compared to
previous methods.


---

**[187. [2310.16713] SkyMath: Technical Report](https://arxiv.org/pdf/2310.16713.pdf)** (2023-10-27)

*Liu Yang, Haihua Yang, Wenjun Cheng, Lei Lin, Chenxia Li, Yifu Chen, Lunan Liu, Jianfei Pan, Tianwen Wei, Biye Li, Liang Zhao, Lijie Wang, Bo Zhu, Guoliang Li, Xuejie Wu, Xilin Luo, Rui Hu*

  Large language models (LLMs) have shown great potential to solve varieties of
natural language processing (NLP) tasks, including mathematical reasoning. In
this work, we present SkyMath, a large language model for mathematics with 13
billion parameters. By applying self-compare fine-tuning, we have enhanced
mathematical reasoning abilities of Skywork-13B-Base remarkably. On GSM8K,
SkyMath outperforms all known open-source models of similar size and has
established a new SOTA performance.


---

**[188. [2405.01560] Copyright related risks in the creation and use of ML/AI systems](https://arxiv.org/pdf/2405.01560.pdf)** (2024-05-06)

*Daniel M. German*

  This paper summarizes the current copyright related risks that Machine
Learning (ML) and Artificial Intelligence (AI) systems (including Large
Language Models --LLMs) incur. These risks affect different stakeholders:
owners of the copyright of the training data, the users of ML/AI systems, the
creators of trained models, and the operators of AI systems. This paper also
provides an overview of ongoing legal cases in the United States related to
these risks.


---

**[189. [2404.05182] DLoRA: Distributed Parameter-Efficient Fine-Tuning Solution for Large
  Language Model](https://arxiv.org/pdf/2404.05182.pdf)** (2024-04-09)

*Chao Gao, Sai Qian Zhang*

  To enhance the performance of large language models (LLM) on downstream
tasks, one solution is to fine-tune certain LLM parameters and make it better
align with the characteristics of the training dataset. This process is
commonly known as parameter-efficient fine-tuning (PEFT). Due to the scale of
LLM, PEFT operations are usually executed in the public environment (e.g.,
cloud server). This necessitates the sharing of sensitive user data across
public environments, thereby raising potential privacy concerns. To tackle
these challenges, we propose a distributed PEFT framework called DLoRA. DLoRA
enables scalable PEFT operations to be performed collaboratively between the
cloud and user devices. Coupled with the proposed Kill and Revive algorithm,
the evaluation results demonstrate that DLoRA can significantly reduce the
computation and communication workload over the user devices while achieving
superior accuracy and privacy protection.


---

**[190. [2405.05008] ADELIE: Aligning Large Language Models on Information Extraction](https://arxiv.org/pdf/2405.05008.pdf)** (2024-10-25)

*Yunjia Qi, Hao Peng, Xiaozhi Wang, Bin Xu, Lei Hou, Juanzi Li*

  Large language models (LLMs) usually fall short on information extraction
(IE) tasks and struggle to follow the complex instructions of IE tasks. This
primarily arises from LLMs not being aligned with humans, as mainstream
alignment datasets typically do not include IE data. In this paper, we
introduce ADELIE (Aligning large language moDELs on Information Extraction), an
aligned LLM that effectively solves various IE tasks, including closed IE, open
IE, and on-demand IE. We first collect and construct a high-quality alignment
corpus IEInstruct for IE. Then we train ADELIE_SFT using instruction tuning on
IEInstruct. We further train ADELIE_SFT with direct preference optimization
(DPO) objective, resulting in ADELIE_DPO. Extensive experiments on various
held-out IE datasets demonstrate that our models (ADELIE_SFT and ADELIE_DPO)
achieve state-of-the-art (SoTA) performance among open-source models. We
further explore the general capabilities of ADELIE, and experimental results
reveal that their general capabilities do not exhibit a noticeable decline. We
will release the code, data, and models to facilitate further research.


---

**[191. [2312.12391] vTrain: A Simulation Framework for Evaluating Cost-effective and
  Compute-optimal Large Language Model Training](https://arxiv.org/pdf/2312.12391.pdf)** (2024-09-11)

*Jehyeon Bang, Yujeong Choi, Myeongwoo Kim, Yongdeok Kim, Minsoo Rhu*

  As large language models (LLMs) become widespread in various application
domains, a critical challenge the AI community is facing is how to train these
large AI models in a cost-effective manner. Existing LLM training plans
typically employ a heuristic based parallel training strategy which is based on
empirical observations rather than grounded upon a thorough examination of the
search space of LLM parallelization. Such limitation renders existing systems
to leave significant performance left on the table, wasting millions of dollars
worth of training cost. This paper presents our profiling-driven simulator
called vTrain, providing AI practitioners a fast yet accurate software
framework to determine an efficient and cost-effective LLM training system
configuration. We demonstrate vTrain's practicality through several case
studies, e.g., effectively evaluating optimal training parallelization
strategies that balances training time and its associated training cost,
efficient multi-tenant GPU cluster schedulers targeting multiple LLM training
jobs, and determining a compute-optimal LLM model architecture given a fixed
compute budget.


---

**[192. [2401.11467] Over-Reasoning and Redundant Calculation of Large Language Models](https://arxiv.org/pdf/2401.11467.pdf)** (2024-03-21)

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

**[193. [2305.03195] Gpt-4: A Review on Advancements and Opportunities in Natural Language
  Processing](https://arxiv.org/pdf/2305.03195.pdf)** (2023-05-08)

*Jawid Ahmad Baktash, Mursal Dawodi*

  Generative Pre-trained Transformer 4 (GPT-4) is the fourth-generation
language model in the GPT series, developed by OpenAI, which promises
significant advancements in the field of natural language processing (NLP). In
this research article, we have discussed the features of GPT-4, its potential
applications, and the challenges that it might face. We have also compared
GPT-4 with its predecessor, GPT-3. GPT-4 has a larger model size (more than one
trillion), better multilingual capabilities, improved contextual understanding,
and reasoning capabilities than GPT-3. Some of the potential applications of
GPT-4 include chatbots, personal assistants, language translation, text
summarization, and question-answering. However, GPT-4 poses several challenges
and limitations such as computational requirements, data requirements, and
ethical concerns.


---

**[194. [2303.04673] Cost-Effective Hyperparameter Optimization for Large Language Model
  Generation Inference](https://arxiv.org/pdf/2303.04673.pdf)** (2023-08-10)

*Chi Wang, Susan Xueqing Liu, Ahmed H. Awadallah*

  Large Language Models (LLMs) have sparked significant interest in their
generative capabilities, leading to the development of various commercial
applications. The high cost of using the models drives application builders to
maximize the value of generation under a limited inference budget. This paper
presents a study of optimizing inference hyperparameters such as the number of
responses, temperature and max tokens, which significantly affects the
utility/cost of text generation. We design a framework named EcoOptiGen which
leverages economical hyperparameter optimization and cost-based pruning.
Experiments with the GPT-3.5/GPT-4 models on a variety of tasks verify its
effectiveness. EcoOptiGen is implemented in the `autogen' package of the FLAML
library: \url{https://aka.ms/autogen}.


---

**[195. [2501.02009] Cross-model Transferability among Large Language Models on the Platonic
  Representations of Concepts](https://arxiv.org/pdf/2501.02009.pdf)** (2025-01-07)

*Youcheng Huang, Chen Huang, Duanyu Feng, Wenqiang Lei, Jiancheng Lv*

  Understanding the inner workings of Large Language Models (LLMs) is a
critical research frontier. Prior research has shown that a single LLM's
concept representations can be captured as steering vectors (SVs), enabling the
control of LLM behavior (e.g., towards generating harmful content). Our work
takes a novel approach by exploring the intricate relationships between concept
representations across different LLMs, drawing an intriguing parallel to
Plato's Allegory of the Cave. In particular, we introduce a linear
transformation method to bridge these representations and present three key
findings: 1) Concept representations across different LLMs can be effectively
aligned using simple linear transformations, enabling efficient cross-model
transfer and behavioral control via SVs. 2) This linear transformation
generalizes across concepts, facilitating alignment and control of SVs
representing different concepts across LLMs. 3) A weak-to-strong
transferability exists between LLM concept representations, whereby SVs
extracted from smaller LLMs can effectively control the behavior of larger
LLMs.


---

**[196. [2410.02229] CodePMP: Scalable Preference Model Pretraining for Large Language Model
  Reasoning](https://arxiv.org/pdf/2410.02229.pdf)** (2024-10-04)

*Huimu Yu, Xing Wu, Weidong Yin, Debing Zhang, Songlin Hu*

  Large language models (LLMs) have made significant progress in natural
language understanding and generation, driven by scalable pretraining and
advanced finetuning. However, enhancing reasoning abilities in LLMs,
particularly via reinforcement learning from human feedback (RLHF), remains
challenging due to the scarcity of high-quality preference data, which is
labor-intensive to annotate and crucial for reward model (RM) finetuning. To
alleviate this issue, we introduce CodePMP, a scalable preference model
pretraining (PMP) pipeline that utilizes a large corpus of synthesized
code-preference pairs from publicly available high-quality source code. CodePMP
improves RM finetuning efficiency by pretraining preference models on
large-scale synthesized code-preference pairs. We evaluate CodePMP on
mathematical reasoning tasks (GSM8K, MATH) and logical reasoning tasks (ReClor,
LogiQA2.0), consistently showing significant improvements in reasoning
performance of LLMs and highlighting the importance of scalable preference
model pretraining for efficient reward modeling.


---

**[197. [2410.08174] Sample then Identify: A General Framework for Risk Control and
  Assessment in Multimodal Large Language Models](https://arxiv.org/pdf/2410.08174.pdf)** (2024-12-17)

*Qingni Wang, Tiantian Geng, Zhiyuan Wang, Teng Wang, Bo Fu, Feng Zheng*

  Multimodal Large Language Models (MLLMs) exhibit promising advancements
across various tasks, yet they still encounter significant trustworthiness
issues. Prior studies apply Split Conformal Prediction (SCP) in language
modeling to construct prediction sets with statistical guarantees. However,
these methods typically rely on internal model logits or are restricted to
multiple-choice settings, which hampers their generalizability and adaptability
in dynamic, open-ended environments. In this paper, we introduce TRON, a
two-step framework for risk control and assessment, applicable to any MLLM that
supports sampling in both open-ended and closed-ended scenarios. TRON comprises
two main components: (1) a novel conformal score to sample response sets of
minimum size, and (2) a nonconformity score to identify high-quality responses
based on self-consistency theory, controlling the error rates by two specific
risk levels. Furthermore, we investigate semantic redundancy in prediction sets
within open-ended contexts for the first time, leading to a promising
evaluation metric for MLLMs based on average set size. Our comprehensive
experiments across four Video Question-Answering (VideoQA) datasets utilizing
eight MLLMs show that TRON achieves desired error rates bounded by two
user-specified risk levels. Additionally, deduplicated prediction sets maintain
adaptiveness while being more efficient and stable for risk assessment under
different risk levels.


---
