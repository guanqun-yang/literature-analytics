**[1. [2411.01696] Conformal Risk Minimization with Variance Reduction](https://arxiv.org/pdf/2411.01696.pdf)** (2025-02-11)

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

**[2. [2502.16691] Toward Responsible Federated Large Language Models: Leveraging a Safety
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

**[3. [2401.11974] Cross-Validation Conformal Risk Control](https://arxiv.org/pdf/2401.11974.pdf)** (2024-05-02)

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

**[4. [2407.07666] A Proposed S.C.O.R.E. Evaluation Framework for Large Language Models :
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

**[5. [2404.09932] Foundational Challenges in Assuring Alignment and Safety of Large
  Language Models](https://arxiv.org/pdf/2404.09932.pdf)** (2024-09-09)

*Usman Anwar, Abulhair Saparov, Javier Rando, Daniel Paleka, Miles Turpin, Peter Hase, Ekdeep Singh Lubana, Erik Jenner, Stephen Casper, Oliver Sourbut, Benjamin L. Edelman, Zhaowei Zhang, Mario Günther, Anton Korinek, Jose Hernandez-Orallo, Lewis Hammond, Eric Bigelow, Alexander Pan, Lauro Langosco, Tomasz Korbak, Heidi Zhang, Ruiqi Zhong, Seán Ó hÉigeartaigh, Gabriel Recchia, Giulio Corsi, Alan Chan, Markus Anderljung, Lilian Edwards, Aleksandar Petrov, Christian Schroeder de Witt, Sumeet Ramesh Motwan, Yoshua Bengio, Danqi Chen, Philip H. S. Torr, Samuel Albanie, Tegan Maharaj, Jakob Foerster, Florian Tramer, He He, Atoosa Kasirzadeh, Yejin Choi, David Krueger*

  This work identifies 18 foundational challenges in assuring the alignment and
safety of large language models (LLMs). These challenges are organized into
three different categories: scientific understanding of LLMs, development and
deployment methods, and sociotechnical challenges. Based on the identified
challenges, we pose $200+$ concrete research questions.


---

**[6. [2407.20999] MoFO: Momentum-Filtered Optimizer for Mitigating Forgetting in LLM
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

**[7. [2310.10049] FATE-LLM: A Industrial Grade Federated Learning Framework for Large
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

**[8. [2410.09047] Unraveling and Mitigating Safety Alignment Degradation of
  Vision-Language Models](https://arxiv.org/pdf/2410.09047.pdf)** (2024-10-14)

*Qin Liu, Chao Shang, Ling Liu, Nikolaos Pappas, Jie Ma, Neha Anna John, Srikanth Doss, Lluis Marquez, Miguel Ballesteros, Yassine Benajiba*

  The safety alignment ability of Vision-Language Models (VLMs) is prone to be
degraded by the integration of the vision module compared to its LLM backbone.
We investigate this phenomenon, dubbed as ''safety alignment degradation'' in
this paper, and show that the challenge arises from the representation gap that
emerges when introducing vision modality to VLMs. In particular, we show that
the representations of multi-modal inputs shift away from that of text-only
inputs which represent the distribution that the LLM backbone is optimized for.
At the same time, the safety alignment capabilities, initially developed within
the textual embedding space, do not successfully transfer to this new
multi-modal representation space. To reduce safety alignment degradation, we
introduce Cross-Modality Representation Manipulation (CMRM), an inference time
representation intervention method for recovering the safety alignment ability
that is inherent in the LLM backbone of VLMs, while simultaneously preserving
the functional capabilities of VLMs. The empirical results show that our
framework significantly recovers the alignment ability that is inherited from
the LLM backbone with minimal impact on the fluency and linguistic capabilities
of pre-trained VLMs even without additional training. Specifically, the unsafe
rate of LLaVA-7B on multi-modal input can be reduced from 61.53% to as low as
3.15% with only inference-time intervention.
  WARNING: This paper contains examples of toxic or harmful language.


---

**[9. [2311.10733] Proceedings of the 3rd International Workshop on Mining and Learning in
  the Legal Domain (MLLD-23)](https://arxiv.org/pdf/2311.10733.pdf)** (2023-11-21)

*Masoud Makrehchi, Dell Zhang, Alina Petrova, John Armour*

  This is the Proceedings of the 3rd International Workshop on Mining and
Learning in the Legal Domain (MLLD-23) which took place in conjunction with the
32nd ACM International Conference on Information and Knowledge Management
(CIKM-2023) at the University of Birmingham, Birmingham, UK on Sunday 22nd
October 2023.


---

**[10. [2404.12038] Uncovering Safety Risks of Large Language Models through Concept
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

**[11. [2412.15265] Chinese SafetyQA: A Safety Short-form Factuality Benchmark for Large
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

**[12. [2502.06884] Learning Conformal Abstention Policies for Adaptive Risk Management in
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

**[13. [2407.12344] The Better Angels of Machine Personality: How Personality Relates to LLM
  Safety](https://arxiv.org/pdf/2407.12344.pdf)** (2024-07-18)

*Jie Zhang, Dongrui Liu, Chen Qian, Ziyue Gan, Yong Liu, Yu Qiao, Jing Shao*

  Personality psychologists have analyzed the relationship between personality
and safety behaviors in human society. Although Large Language Models (LLMs)
demonstrate personality traits, the relationship between personality traits and
safety abilities in LLMs still remains a mystery. In this paper, we discover
that LLMs' personality traits are closely related to their safety abilities,
i.e., toxicity, privacy, and fairness, based on the reliable MBTI-M scale.
Meanwhile, the safety alignment generally increases various LLMs' Extraversion,
Sensing, and Judging traits. According to such findings, we can edit LLMs'
personality traits and improve their safety performance, e.g., inducing
personality from ISTJ to ISTP resulted in a relative improvement of
approximately 43% and 10% in privacy and fairness performance, respectively.
Additionally, we find that LLMs with different personality traits are
differentially susceptible to jailbreak. This study pioneers the investigation
of LLM safety from a personality perspective, providing new insights into LLM
safety enhancement.


---

**[14. [2306.02551] Conformal Predictive Safety Filter for RL Controllers in Dynamic
  Environments](https://arxiv.org/pdf/2306.02551.pdf)** (2023-12-08)

*Kegan J. Strawn, Nora Ayanian, Lars Lindemann*

  The interest in using reinforcement learning (RL) controllers in
safety-critical applications such as robot navigation around pedestrians
motivates the development of additional safety mechanisms. Running RL-enabled
systems among uncertain dynamic agents may result in high counts of collisions
and failures to reach the goal. The system could be safer if the pre-trained RL
policy was uncertainty-informed. For that reason, we propose conformal
predictive safety filters that: 1) predict the other agents' trajectories, 2)
use statistical techniques to provide uncertainty intervals around these
predictions, and 3) learn an additional safety filter that closely follows the
RL controller but avoids the uncertainty intervals. We use conformal prediction
to learn uncertainty-informed predictive safety filters, which make no
assumptions about the agents' distribution. The framework is modular and
outperforms the existing controllers in simulation. We demonstrate our approach
with multiple experiments in a collision avoidance gym environment and show
that our approach minimizes the number of collisions without making
overly-conservative predictions.


---

**[15. [2310.02863] Conformal Predictions for Longitudinal Data](https://arxiv.org/pdf/2310.02863.pdf)** (2023-10-05)

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

**[16. [2403.01216] API Is Enough: Conformal Prediction for Large Language Models Without
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

**[17. [2503.05021] Safety is Not Only About Refusal: Reasoning-Enhanced Fine-tuning for
  Interpretable LLM Safety](https://arxiv.org/pdf/2503.05021.pdf)** (2025-03-10)

*Yuyou Zhang, Miao Li, William Han, Yihang Yao, Zhepeng Cen, Ding Zhao*

  Large Language Models (LLMs) are vulnerable to jailbreak attacks that exploit
weaknesses in traditional safety alignment, which often relies on rigid refusal
heuristics or representation engineering to block harmful outputs. While they
are effective for direct adversarial attacks, they fall short of broader safety
challenges requiring nuanced, context-aware decision-making. To address this,
we propose Reasoning-enhanced Finetuning for interpretable LLM Safety
(Rational), a novel framework that trains models to engage in explicit safe
reasoning before response. Fine-tuned models leverage the extensive pretraining
knowledge in self-generated reasoning to bootstrap their own safety through
structured reasoning, internalizing context-sensitive decision-making. Our
findings suggest that safety extends beyond refusal, requiring context
awareness for more robust, interpretable, and adaptive responses. Reasoning is
not only a core capability of LLMs but also a fundamental mechanism for LLM
safety. Rational employs reasoning-enhanced fine-tuning, allowing it to reject
harmful prompts while providing meaningful and context-aware responses in
complex scenarios.


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

**[19. [2502.07497] On Training-Conditional Conformal Prediction and Binomial Proportion
  Confidence Intervals](https://arxiv.org/pdf/2502.07497.pdf)** (2025-04-08)

*Rudi Coppola, Jr Manuel Mazo*

  Estimating the expectation of a Bernoulli random variable based on N
independent trials is a classical problem in statistics, typically addressed
using Binomial Proportion Confidence Intervals (BPCI). In the control systems
community, many critical tasks-such as certifying the statistical safety of
dynamical systems-can be formulated as BPCI problems. Conformal Prediction
(CP), a distribution-free technique for uncertainty quantification, has gained
significant attention in recent years and has been applied to various control
systems problems, particularly to address uncertainties in learned dynamics or
controllers. A variant known as training-conditional CP was recently employed
to tackle the problem of safety certification. In this note, we highlight that
the use of training-conditional CP in this context does not provide valid
safety guarantees. We demonstrate why CP is unsuitable for BPCI problems and
argue that traditional BPCI methods are better suited for statistical safety
certification.


---

**[20. [2411.11543] PSA-VLM: Enhancing Vision-Language Model Safety through Progressive
  Concept-Bottleneck-Driven Alignment](https://arxiv.org/pdf/2411.11543.pdf)** (2025-01-14)

*Zhendong Liu, Yuanbi Nie, Yingshui Tan, Jiaheng Liu, Xiangyu Yue, Qiushi Cui, Chongjun Wang, Xiaoyong Zhu, Bo Zheng*

  Benefiting from the powerful capabilities of Large Language Models (LLMs),
pre-trained visual encoder models connected to LLMs form Vision Language Models
(VLMs). However, recent research shows that the visual modality in VLMs is
highly vulnerable, allowing attackers to bypass safety alignment in LLMs
through visually transmitted content, launching harmful attacks. To address
this challenge, we propose a progressive concept-based alignment strategy,
PSA-VLM, which incorporates safety modules as concept bottlenecks to enhance
visual modality safety alignment. By aligning model predictions with specific
safety concepts, we improve defenses against risky images, enhancing
explainability and controllability while minimally impacting general
performance. Our method is obtained through two-stage training. The low
computational cost of the first stage brings very effective performance
improvement, and the fine-tuning of the language model in the second stage
further improves the safety performance. Our method achieves state-of-the-art
results on popular VLM safety benchmark.


---

**[21. [2412.11387] How Can LLMs and Knowledge Graphs Contribute to Robot Safety? A Few-Shot
  Learning Approach](https://arxiv.org/pdf/2412.11387.pdf)** (2024-12-17)

*Abdulrahman Althobaiti, Angel Ayala, JingYing Gao, Ali Almutairi, Mohammad Deghat, Imran Razzak, Francisco Cruz*

  Large Language Models (LLMs) are transforming the robotics domain by enabling
robots to comprehend and execute natural language instructions. The cornerstone
benefits of LLM include processing textual data from technical manuals,
instructions, academic papers, and user queries based on the knowledge
provided. However, deploying LLM-generated code in robotic systems without
safety verification poses significant risks. This paper outlines a safety layer
that verifies the code generated by ChatGPT before executing it to control a
drone in a simulated environment. The safety layer consists of a fine-tuned
GPT-4o model using Few-Shot learning, supported by knowledge graph prompting
(KGP). Our approach improves the safety and compliance of robotic actions,
ensuring that they adhere to the regulations of drone operations.


---

**[22. [2410.07471] SEAL: Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection](https://arxiv.org/pdf/2410.07471.pdf)** (2024-10-14)

*Han Shen, Pin-Yu Chen, Payel Das, Tianyi Chen*

  Fine-tuning on task-specific data to boost downstream performance is a
crucial step for leveraging Large Language Models (LLMs). However, previous
studies have demonstrated that fine-tuning the models on several adversarial
samples or even benign data can greatly comprise the model's pre-equipped
alignment and safety capabilities. In this work, we propose SEAL, a novel
framework to enhance safety in LLM fine-tuning. SEAL learns a data ranker based
on the bilevel optimization to up rank the safe and high-quality fine-tuning
data and down rank the unsafe or low-quality ones. Models trained with SEAL
demonstrate superior quality over multiple baselines, with 8.5% and 9.7% win
rate increase compared to random selection respectively on Llama-3-8b-Instruct
and Merlinite-7b models. Our code is available on github
https://github.com/hanshen95/SEAL.


---

**[23. [2501.00555] Monty Hall and Optimized Conformal Prediction to Improve Decision-Making
  with LLMs](https://arxiv.org/pdf/2501.00555.pdf)** (2025-01-03)

*Harit Vishwakarma, Alan Mishler, Thomas Cook, Niccolò Dalmasso, Natraj Raman, Sumitra Ganesh*

  Large language models (LLMs) are empowering decision-making in several
applications, including tool or API usage and answering multiple-choice
questions (MCQs). However, they often make overconfident, incorrect
predictions, which can be risky in high-stakes settings like healthcare and
finance. To mitigate these risks, recent works have used conformal prediction
(CP), a model-agnostic framework for distribution-free uncertainty
quantification. CP transforms a \emph{score function} into prediction sets that
contain the true answer with high probability. While CP provides this coverage
guarantee for arbitrary scores, the score quality significantly impacts
prediction set sizes. Prior works have relied on LLM logits or other heuristic
scores, lacking quality guarantees. We address this limitation by introducing
CP-OPT, an optimization framework to learn scores that minimize set sizes while
maintaining coverage. Furthermore, inspired by the Monty Hall problem, we
extend CP's utility beyond uncertainty quantification to improve accuracy. We
propose \emph{conformal revision of questions} (CROQ) to revise the problem by
narrowing down the available choices to those in the prediction set. The
coverage guarantee of CP ensures that the correct choice is in the revised
question prompt with high probability, while the smaller number of choices
increases the LLM's chances of answering it correctly. Experiments on MMLU,
ToolAlpaca, and TruthfulQA datasets with Gemma-2, Llama-3 and Phi-3 models show
that CP-OPT significantly reduces set sizes while maintaining coverage, and
CROQ improves accuracy over the standard inference, especially when paired with
CP-OPT scores. Together, CP-OPT and CROQ offer a robust framework for improving
both the safety and accuracy of LLM-driven decision-making.


---

**[24. [2504.04151] STEP: Staged Parameter-Efficient Pre-training for Large Language Models](https://arxiv.org/pdf/2504.04151.pdf)** (2025-04-08)

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

**[25. [2304.04521] GL-MCM: Global and Local Maximum Concept Matching for Zero-Shot
  Out-of-Distribution Detection](https://arxiv.org/pdf/2304.04521.pdf)** (2025-01-22)

*Atsuyuki Miyai, Qing Yu, Go Irie, Kiyoharu Aizawa*

  Zero-shot out-of-distribution (OOD) detection is a task that detects OOD
images during inference with only in-distribution (ID) class names. Existing
methods assume ID images contain a single, centered object, and do not consider
the more realistic multi-object scenarios, where both ID and OOD objects are
present. To meet the needs of many users, the detection method must have the
flexibility to adapt the type of ID images. To this end, we present
Global-Local Maximum Concept Matching (GL-MCM), which incorporates local image
scores as an auxiliary score to enhance the separability of global and local
visual features. Due to the simple ensemble score function design, GL-MCM can
control the type of ID images with a single weight parameter. Experiments on
ImageNet and multi-object benchmarks demonstrate that GL-MCM outperforms
baseline zero-shot methods and is comparable to fully supervised methods.
Furthermore, GL-MCM offers strong flexibility in adjusting the target type of
ID images. The code is available via https://github.com/AtsuMiyai/GL-MCM.


---

**[26. [2402.16444] ShieldLM: Empowering LLMs as Aligned, Customizable and Explainable
  Safety Detectors](https://arxiv.org/pdf/2402.16444.pdf)** (2024-11-06)

*Zhexin Zhang, Yida Lu, Jingyuan Ma, Di Zhang, Rui Li, Pei Ke, Hao Sun, Lei Sha, Zhifang Sui, Hongning Wang, Minlie Huang*

  The safety of Large Language Models (LLMs) has gained increasing attention in
recent years, but there still lacks a comprehensive approach for detecting
safety issues within LLMs' responses in an aligned, customizable and
explainable manner. In this paper, we propose ShieldLM, an LLM-based safety
detector, which aligns with common safety standards, supports customizable
detection rules, and provides explanations for its decisions. To train
ShieldLM, we compile a large bilingual dataset comprising 14,387 query-response
pairs, annotating the safety of responses based on various safety standards.
Through extensive experiments, we demonstrate that ShieldLM surpasses strong
baselines across four test sets, showcasing remarkable customizability and
explainability. Besides performing well on standard detection datasets,
ShieldLM has also been shown to be effective as a safety evaluator for advanced
LLMs. ShieldLM is released at \url{https://github.com/thu-coai/ShieldLM} to
support accurate and explainable safety detection under various safety
standards.


---

**[27. [2411.02317] Defining and Evaluating Physical Safety for Large Language Models](https://arxiv.org/pdf/2411.02317.pdf)** (2024-11-05)

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

**[28. [2409.07772] Alignment with Preference Optimization Is All You Need for LLM Safety](https://arxiv.org/pdf/2409.07772.pdf)** (2024-09-13)

*Reda Alami, Ali Khalifa Almansoori, Ahmed Alzubaidi, Mohamed El Amine Seddik, Mugariya Farooq, Hakim Hacid*

  We demonstrate that preference optimization methods can effectively enhance
LLM safety. Applying various alignment techniques to the Falcon 11B model using
safety datasets, we achieve a significant boost in global safety score (from
$57.64\%$ to $99.90\%$) as measured by LlamaGuard 3 8B, competing with
state-of-the-art models. On toxicity benchmarks, average scores in adversarial
settings dropped from over $0.6$ to less than $0.07$. However, this safety
improvement comes at the cost of reduced general capabilities, particularly in
math, suggesting a trade-off. We identify noise contrastive alignment
(Safe-NCA) as an optimal method for balancing safety and performance. Our study
ultimately shows that alignment techniques can be sufficient for building safe
and robust models.


---

**[29. [2412.15035] LLMs Lost in Translation: M-ALERT uncovers Cross-Linguistic Safety Gaps](https://arxiv.org/pdf/2412.15035.pdf)** (2025-04-02)

*Felix Friedrich, Simone Tedeschi, Patrick Schramowski, Manuel Brack, Roberto Navigli, Huu Nguyen, Bo Li, Kristian Kersting*

  Building safe Large Language Models (LLMs) across multiple languages is
essential in ensuring both safe access and linguistic diversity. To this end,
we introduce M-ALERT, a multilingual benchmark that evaluates the safety of
LLMs in five languages: English, French, German, Italian, and Spanish. M-ALERT
includes 15k high-quality prompts per language, totaling 75k, following the
detailed ALERT taxonomy. Our extensive experiments on 10 state-of-the-art LLMs
highlight the importance of language-specific safety analysis, revealing that
models often exhibit significant inconsistencies in safety across languages and
categories. For instance, Llama3.2 shows high unsafety in the category
crime_tax for Italian but remains safe in other languages. Similar differences
can be observed across all models. In contrast, certain categories, such as
substance_cannabis and crime_propaganda, consistently trigger unsafe responses
across models and languages. These findings underscore the need for robust
multilingual safety practices in LLMs to ensure safe and responsible usage
across diverse user communities.


---

**[30. [2502.12601] COPU: Conformal Prediction for Uncertainty Quantification in Natural
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

**[31. [2407.00499] ConU: Conformal Uncertainty in Large Language Models with Correctness
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

**[32. [2502.15086] Is Safety Standard Same for Everyone? User-Specific Safety Evaluation of
  Large Language Models](https://arxiv.org/pdf/2502.15086.pdf)** (2025-02-24)

*Yeonjun In, Wonjoong Kim, Kanghoon Yoon, Sungchul Kim, Mehrab Tanjim, Kibum Kim, Chanyoung Park*

  As the use of large language model (LLM) agents continues to grow, their
safety vulnerabilities have become increasingly evident. Extensive benchmarks
evaluate various aspects of LLM safety by defining the safety relying heavily
on general standards, overlooking user-specific standards. However, safety
standards for LLM may vary based on a user-specific profiles rather than being
universally consistent across all users. This raises a critical research
question: Do LLM agents act safely when considering user-specific safety
standards? Despite its importance for safe LLM use, no benchmark datasets
currently exist to evaluate the user-specific safety of LLMs. To address this
gap, we introduce U-SAFEBENCH, the first benchmark designed to assess
user-specific aspect of LLM safety. Our evaluation of 18 widely used LLMs
reveals current LLMs fail to act safely when considering user-specific safety
standards, marking a new discovery in this field. To address this
vulnerability, we propose a simple remedy based on chain-of-thought,
demonstrating its effectiveness in improving user-specific safety. Our
benchmark and code are available at https://github.com/yeonjun-in/U-SafeBench.


---

**[33. [2405.14191] S-Eval: Towards Automated and Comprehensive Safety Evaluation for Large
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

**[34. [2502.06631] Conformal Predictions for Human Action Recognition with Vision-Language
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

**[35. [2406.17663] LLM-ARC: Enhancing LLMs with an Automated Reasoning Critic](https://arxiv.org/pdf/2406.17663.pdf)** (2024-07-22)

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

**[36. [2504.09862] RadarLLM: Empowering Large Language Models to Understand Human Motion
  from Millimeter-wave Point Cloud Sequence](https://arxiv.org/pdf/2504.09862.pdf)** (2025-04-15)

*Zengyuan Lai, Jiarui Yang, Songpengcheng Xia, Lizhou Lin, Lan Sun, Renwen Wang, Jianran Liu, Qi Wu, Ling Pei*

  Millimeter-wave radar provides a privacy-preserving solution for human motion
analysis, yet its sparse point clouds pose significant challenges for semantic
understanding. We present Radar-LLM, the first framework that leverages large
language models (LLMs) for human motion understanding using millimeter-wave
radar as the sensing modality. Our approach introduces two key innovations: (1)
a motion-guided radar tokenizer based on our Aggregate VQ-VAE architecture that
incorporates deformable body templates and masked trajectory modeling to encode
spatiotemporal point clouds into compact semantic tokens, and (2) a radar-aware
language model that establishes cross-modal alignment between radar and text in
a shared embedding space. To address data scarcity, we introduce a
physics-aware synthesis pipeline that generates realistic radar-text pairs from
motion-text datasets. Extensive experiments demonstrate that Radar-LLM achieves
state-of-the-art performance across both synthetic and real-world benchmarks,
enabling accurate translation of millimeter-wave signals to natural language
descriptions. This breakthrough facilitates comprehensive motion understanding
in privacy-sensitive applications like healthcare and smart homes. We will
release the full implementation to support further research on
https://inowlzy.github.io/RadarLLM/.


---

**[37. [2502.07340] Aligning Large Language Models to Follow Instructions and Hallucinate
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

**[38. [2410.21965] SG-Bench: Evaluating LLM Safety Generalization Across Diverse Tasks and
  Prompt Types](https://arxiv.org/pdf/2410.21965.pdf)** (2024-10-30)

*Yutao Mou, Shikun Zhang, Wei Ye*

  Ensuring the safety of large language model (LLM) applications is essential
for developing trustworthy artificial intelligence. Current LLM safety
benchmarks have two limitations. First, they focus solely on either
discriminative or generative evaluation paradigms while ignoring their
interconnection. Second, they rely on standardized inputs, overlooking the
effects of widespread prompting techniques, such as system prompts, few-shot
demonstrations, and chain-of-thought prompting. To overcome these issues, we
developed SG-Bench, a novel benchmark to assess the generalization of LLM
safety across various tasks and prompt types. This benchmark integrates both
generative and discriminative evaluation tasks and includes extended data to
examine the impact of prompt engineering and jailbreak on LLM safety. Our
assessment of 3 advanced proprietary LLMs and 10 open-source LLMs with the
benchmark reveals that most LLMs perform worse on discriminative tasks than
generative ones, and are highly susceptible to prompts, indicating poor
generalization in safety alignment. We also explain these findings
quantitatively and qualitatively to provide insights for future research.


---

**[39. [2410.08431] oRetrieval Augmented Generation for 10 Large Language Models and its
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

**[40. [2501.16215] Enhancing Visual Inspection Capability of Multi-Modal Large Language
  Models on Medical Time Series with Supportive Conformalized and Interpretable
  Small Specialized Models](https://arxiv.org/pdf/2501.16215.pdf)** (2025-01-28)

*Huayu Li, Xiwen Chen, Ci Zhang, Stuart F. Quan, William D. S. Killgore, Shu-Fen Wung, Chen X. Chen, Geng Yuan, Jin Lu, Ao Li*

  Large language models (LLMs) exhibit remarkable capabilities in visual
inspection of medical time-series data, achieving proficiency comparable to
human clinicians. However, their broad scope limits domain-specific precision,
and proprietary weights hinder fine-tuning for specialized datasets. In
contrast, small specialized models (SSMs) excel in targeted tasks but lack the
contextual reasoning required for complex clinical decision-making. To address
these challenges, we propose ConMIL (Conformalized Multiple Instance Learning),
a decision-support SSM that integrates seamlessly with LLMs. By using Multiple
Instance Learning (MIL) to identify clinically significant signal segments and
conformal prediction for calibrated set-valued outputs, ConMIL enhances LLMs'
interpretative capabilities for medical time-series analysis. Experimental
results demonstrate that ConMIL significantly improves the performance of
state-of-the-art LLMs, such as ChatGPT4.0 and Qwen2-VL-7B. Specifically,
\ConMIL{}-supported Qwen2-VL-7B achieves 94.92% and 96.82% precision for
confident samples in arrhythmia detection and sleep staging, compared to
standalone LLM accuracy of 46.13% and 13.16%. These findings highlight the
potential of ConMIL to bridge task-specific precision and broader contextual
reasoning, enabling more reliable and interpretable AI-driven clinical decision
support.


---

**[41. [2501.10915] LegalGuardian: A Privacy-Preserving Framework for Secure Integration of
  Large Language Models in Legal Practice](https://arxiv.org/pdf/2501.10915.pdf)** (2025-01-22)

*M. Mikail Demir, Hakan T. Otal, M. Abdullah Canbaz*

  Large Language Models (LLMs) hold promise for advancing legal practice by
automating complex tasks and improving access to justice. However, their
adoption is limited by concerns over client confidentiality, especially when
lawyers include sensitive Personally Identifiable Information (PII) in prompts,
risking unauthorized data exposure. To mitigate this, we introduce
LegalGuardian, a lightweight, privacy-preserving framework tailored for lawyers
using LLM-based tools. LegalGuardian employs Named Entity Recognition (NER)
techniques and local LLMs to mask and unmask confidential PII within prompts,
safeguarding sensitive data before any external interaction. We detail its
development and assess its effectiveness using a synthetic prompt library in
immigration law scenarios. Comparing traditional NER models with one-shot
prompted local LLM, we find that LegalGuardian achieves a F1-score of 93% with
GLiNER and 97% with Qwen2.5-14B in PII detection. Semantic similarity analysis
confirms that the framework maintains high fidelity in outputs, ensuring robust
utility of LLM-based tools. Our findings indicate that legal professionals can
harness advanced AI technologies without compromising client confidentiality or
the quality of legal documents.


---

**[42. [2405.02140] An Information Theoretic Perspective on Conformal Prediction](https://arxiv.org/pdf/2405.02140.pdf)** (2025-02-18)

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

**[43. [2205.14317] A Confidence Machine for Sparse High-Order Interaction Model](https://arxiv.org/pdf/2205.14317.pdf)** (2022-11-02)

*Diptesh Das, Eugene Ndiaye, Ichiro Takeuchi*

  In predictive modeling for high-stake decision-making, predictors must be not
only accurate but also reliable. Conformal prediction (CP) is a promising
approach for obtaining the confidence of prediction results with fewer
theoretical assumptions. To obtain the confidence set by so-called full-CP, we
need to refit the predictor for all possible values of prediction results,
which is only possible for simple predictors. For complex predictors such as
random forests (RFs) or neural networks (NNs), split-CP is often employed where
the data is split into two parts: one part for fitting and another to compute
the confidence set. Unfortunately, because of the reduced sample size, split-CP
is inferior to full-CP both in fitting as well as confidence set computation.
In this paper, we develop a full-CP of sparse high-order interaction model
(SHIM), which is sufficiently flexible as it can take into account high-order
interactions among variables. We resolve the computational challenge for
full-CP of SHIM by introducing a novel approach called homotopy mining. Through
numerical experiments, we demonstrate that SHIM is as accurate as complex
predictors such as RF and NN and enjoys the superior statistical power of
full-CP.


---

**[44. [2503.23566] When LLM Therapists Become Salespeople: Evaluating Large Language Models
  for Ethical Motivational Interviewing](https://arxiv.org/pdf/2503.23566.pdf)** (2025-04-01)

*Haein Kong, Seonghyeon Moon*

  Large language models (LLMs) have been actively applied in the mental health
field. Recent research shows the promise of LLMs in applying psychotherapy,
especially motivational interviewing (MI). However, there is a lack of studies
investigating how language models understand MI ethics. Given the risks that
malicious actors can use language models to apply MI for unethical purposes, it
is important to evaluate their capability of differentiating ethical and
unethical MI practices. Thus, this study investigates the ethical awareness of
LLMs in MI with multiple experiments. Our findings show that LLMs have a
moderate to strong level of knowledge in MI. However, their ethical standards
are not aligned with the MI spirit, as they generated unethical responses and
performed poorly in detecting unethical responses. We proposed a Chain-of-Ethic
prompt to mitigate those risks and improve safety. Finally, our proposed
strategy effectively improved ethical MI response generation and detection
performance. These findings highlight the need for safety evaluations and
guidelines for building ethical LLM-powered psychotherapy.


---

**[45. [2404.11338] LLMs for Cyber Security: New Opportunities](https://arxiv.org/pdf/2404.11338.pdf)** (2024-04-18)

*Dinil Mon Divakaran, Sai Teja Peddinti*

  Large language models (LLMs) are a class of powerful and versatile models
that are beneficial to many industries. With the emergence of LLMs, we take a
fresh look at cyber security, specifically exploring and summarizing the
potential of LLMs in addressing challenging problems in the security and safety
domains.


---

**[46. [2409.14038] OAEI-LLM: A Benchmark Dataset for Understanding Large Language Model
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

**[47. [2502.20285] Conformal Tail Risk Control for Large Language Model Alignment](https://arxiv.org/pdf/2502.20285.pdf)** (2025-02-28)

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

**[48. [2406.06818] Conformal Prediction for Class-wise Coverage via Augmented Label Rank
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

**[49. [2502.08657] Refining Positive and Toxic Samples for Dual Safety Self-Alignment of
  LLMs with Minimal Human Interventions](https://arxiv.org/pdf/2502.08657.pdf)** (2025-02-14)

*Jingxin Xu, Guoshun Nan, Sheng Guan, Sicong Leng, Yilian Liu, Zixiao Wang, Yuyang Ma, Zhili Zhou, Yanzhao Hou, Xiaofeng Tao*

  Recent AI agents, such as ChatGPT and LLaMA, primarily rely on instruction
tuning and reinforcement learning to calibrate the output of large language
models (LLMs) with human intentions, ensuring the outputs are harmless and
helpful. Existing methods heavily depend on the manual annotation of
high-quality positive samples, while contending with issues such as noisy
labels and minimal distinctions between preferred and dispreferred response
data. However, readily available toxic samples with clear safety distinctions
are often filtered out, removing valuable negative references that could aid
LLMs in safety alignment. In response, we propose PT-ALIGN, a novel safety
self-alignment approach that minimizes human supervision by automatically
refining positive and toxic samples and performing fine-grained dual
instruction tuning. Positive samples are harmless responses, while toxic
samples deliberately contain extremely harmful content, serving as a new
supervisory signals. Specifically, we utilize LLM itself to iteratively
generate and refine training instances by only exploring fewer than 50 human
annotations. We then employ two losses, i.e., maximum likelihood estimation
(MLE) and fine-grained unlikelihood training (UT), to jointly learn to enhance
the LLM's safety. The MLE loss encourages an LLM to maximize the generation of
harmless content based on positive samples. Conversely, the fine-grained UT
loss guides the LLM to minimize the output of harmful words based on negative
samples at the token-level, thereby guiding the model to decouple safety from
effectiveness, directing it toward safer fine-tuning objectives, and increasing
the likelihood of generating helpful and reliable content. Experiments on 9
popular open-source LLMs demonstrate the effectiveness of our PT-ALIGN for
safety alignment, while maintaining comparable levels of helpfulness and
usefulness.


---

**[50. [2409.15548] Beyond Conformal Predictors: Adaptive Conformal Inference with
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

**[51. [2312.04916] EE-LLM: Large-Scale Training and Inference of Early-Exit Large Language
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

**[52. [2404.04392] Fine-Tuning, Quantization, and LLMs: Navigating Unintended Outcomes](https://arxiv.org/pdf/2404.04392.pdf)** (2024-09-10)

*Divyanshu Kumar, Anurakt Kumar, Sahil Agarwal, Prashanth Harshangi*

  Large Language Models (LLMs) have gained widespread adoption across various
domains, including chatbots and auto-task completion agents. However, these
models are susceptible to safety vulnerabilities such as jailbreaking, prompt
injection, and privacy leakage attacks. These vulnerabilities can lead to the
generation of malicious content, unauthorized actions, or the disclosure of
confidential information. While foundational LLMs undergo alignment training
and incorporate safety measures, they are often subject to fine-tuning, or
doing quantization resource-constrained environments. This study investigates
the impact of these modifications on LLM safety, a critical consideration for
building reliable and secure AI systems. We evaluate foundational models
including Mistral, Llama series, Qwen, and MosaicML, along with their
fine-tuned variants. Our comprehensive analysis reveals that fine-tuning
generally increases the success rates of jailbreak attacks, while quantization
has variable effects on attack success rates. Importantly, we find that
properly implemented guardrails significantly enhance resistance to jailbreak
attempts. These findings contribute to our understanding of LLM vulnerabilities
and provide insights for developing more robust safety strategies in the
deployment of language models.


---

**[53. [2402.09283] Attacks, Defenses and Evaluations for LLM Conversation Safety: A Survey](https://arxiv.org/pdf/2402.09283.pdf)** (2024-03-28)

*Zhichen Dong, Zhanhui Zhou, Chao Yang, Jing Shao, Yu Qiao*

  Large Language Models (LLMs) are now commonplace in conversation
applications. However, their risks of misuse for generating harmful responses
have raised serious societal concerns and spurred recent research on LLM
conversation safety. Therefore, in this survey, we provide a comprehensive
overview of recent studies, covering three critical aspects of LLM conversation
safety: attacks, defenses, and evaluations. Our goal is to provide a structured
summary that enhances understanding of LLM conversation safety and encourages
further investigation into this important subject. For easy reference, we have
categorized all the studies mentioned in this survey according to our taxonomy,
available at: https://github.com/niconi19/LLM-conversation-safety.


---

**[54. [2407.09577] Flash normalization: fast normalization for LLMs](https://arxiv.org/pdf/2407.09577.pdf)** (2025-04-03)

*Nils Graef, Matthew Clapp, Andrew Wasielewski*

  RMSNorm is used by many LLMs such as Llama, Mistral, and OpenELM. This paper
details FlashNorm, which is an exact but faster implementation of RMSNorm
followed by linear layers. FlashNorm also speeds up Layer Normalization and its
recently proposed replacement Dynamic Tanh (DyT) arXiv:2503.10622. See
https://github.com/OpenMachine-ai/transformer-tricks for code and more
transformer tricks.


---

**[55. [2502.08142] Bridging the Safety Gap: A Guardrail Pipeline for Trustworthy LLM
  Inferences](https://arxiv.org/pdf/2502.08142.pdf)** (2025-02-13)

*Shanshan Han, Salman Avestimehr, Chaoyang He*

  We present Wildflare GuardRail, a guardrail pipeline designed to enhance the
safety and reliability of Large Language Model (LLM) inferences by
systematically addressing risks across the entire processing workflow.
Wildflare GuardRail integrates several core functional modules, including
Safety Detector that identifies unsafe inputs and detects hallucinations in
model outputs while generating root-cause explanations, Grounding that
contextualizes user queries with information retrieved from vector databases,
Customizer that adjusts outputs in real time using lightweight, rule-based
wrappers, and Repairer that corrects erroneous LLM outputs using hallucination
explanations provided by Safety Detector. Results show that our unsafe content
detection model in Safety Detector achieves comparable performance with OpenAI
API, though trained on a small dataset constructed with several public
datasets. Meanwhile, the lightweight wrappers can address malicious URLs in
model outputs in 1.06s per query with 100% accuracy without costly model calls.
Moreover, the hallucination fixing model demonstrates effectiveness in reducing
hallucinations with an accuracy of 80.7%.


---

**[56. [2411.18948] RevPRAG: Revealing Poisoning Attacks in Retrieval-Augmented Generation
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

**[57. [2502.12485] Safe at the Margins: A General Approach to Safety Alignment in
  Low-Resource English Languages -- A Singlish Case Study](https://arxiv.org/pdf/2502.12485.pdf)** (2025-04-09)

*Isaac Lim, Shaun Khoo, Roy Ka-Wei Lee, Watson Chua, Jia Yi Goh, Jessica Foo*

  Ensuring the safety of Large Language Models (LLMs) in diverse linguistic
settings remains challenging, particularly for low-resource languages. Existing
safety alignment methods are English-centric, limiting their effectiveness. We
systematically compare Supervised Fine-Tuning (SFT), Direct Preference
Optimization (DPO), and Kahneman-Tversky Optimization (KTO) for aligning
SEA-Lion-v2.1-Instruct, a Llama 3-8B variant, to reduce toxicity in Singlish.
Our results show that SFT+KTO achieves superior safety alignment with higher
sample efficiency than DPO. Additionally, we introduce KTO-S, which enhances
stability via improved KL divergence regularization. Our approach reduces
Singlish toxicity by 99\%, generalizes to TOXIGEN, and maintains strong
performance on standard LLM benchmarks, providing a scalable framework for
safer AI deployment in multilingual contexts.


---

**[58. [2403.03883] SaulLM-7B: A pioneering Large Language Model for Law](https://arxiv.org/pdf/2403.03883.pdf)** (2024-03-08)

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

**[59. [2312.04598] Formalization of Robot Collision Detection Method based on Conformal
  Geometric Algebra](https://arxiv.org/pdf/2312.04598.pdf)** (2023-12-11)

*Yingjie Wu, Guohui Wang, Shanyan Chen, Zhiping Shi, Yong Guan, Ximeng Li*

  Cooperative robots can significantly assist people in their productive
activities, improving the quality of their works. Collision detection is vital
to ensure the safe and stable operation of cooperative robots in productive
activities. As an advanced geometric language, conformal geometric algebra can
simplify the construction of the robot collision model and the calculation of
collision distance. Compared with the formal method based on conformal
geometric algebra, the traditional method may have some defects which are
difficult to find in the modelling and calculation. We use the formal method
based on conformal geometric algebra to study the collision detection problem
of cooperative robots. This paper builds formal models of geometric primitives
and the robot body based on the conformal geometric algebra library in HOL
Light. We analyse the shortest distance between geometric primitives and prove
their collision determination conditions. Based on the above contents, we
construct a formal verification framework for the robot collision detection
method. By the end of this paper, we apply the proposed framework to collision
detection between two single-arm industrial cooperative robots. The flexibility
and reliability of the proposed framework are verified by constructing a
general collision model and a special collision model for two single-arm
industrial cooperative robots.


---

**[60. [2504.09420] SaRO: Enhancing LLM Safety through Reasoning-based Alignment](https://arxiv.org/pdf/2504.09420.pdf)** (2025-04-15)

*Yutao Mou, Yuxiao Luo, Shikun Zhang, Wei Ye*

  Current safety alignment techniques for large language models (LLMs) face two
key challenges: (1) under-generalization, which leaves models vulnerable to
novel jailbreak attacks, and (2) over-alignment, which leads to the excessive
refusal of benign instructions. Our preliminary investigation reveals semantic
overlap between jailbreak/harmful queries and normal prompts in embedding
space, suggesting that more effective safety alignment requires a deeper
semantic understanding. This motivates us to incorporate safety-policy-driven
reasoning into the alignment process. To this end, we propose the
Safety-oriented Reasoning Optimization Framework (SaRO), which consists of two
stages: (1) Reasoning-style Warmup (RW) that enables LLMs to internalize
long-chain reasoning through supervised fine-tuning, and (2) Safety-oriented
Reasoning Process Optimization (SRPO) that promotes safety reflection via
direct preference optimization (DPO). Extensive experiments demonstrate the
superiority of SaRO over traditional alignment methods.


---

**[61. [2412.04947] C$^2$LEVA: Toward Comprehensive and Contamination-Free Language Model
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

**[62. [2203.11409] A Primer on Maximum Causal Entropy Inverse Reinforcement Learning](https://arxiv.org/pdf/2203.11409.pdf)** (2022-03-23)

*Adam Gleave, Sam Toyer*

  Inverse Reinforcement Learning (IRL) algorithms infer a reward function that
explains demonstrations provided by an expert acting in the environment.
Maximum Causal Entropy (MCE) IRL is currently the most popular formulation of
IRL, with numerous extensions. In this tutorial, we present a compressed
derivation of MCE IRL and the key results from contemporary implementations of
MCE IRL algorithms. We hope this will serve both as an introductory resource
for those new to the field, and as a concise reference for those already
familiar with these topics.


---

**[63. [2501.18991] Optimal Transport-based Conformal Prediction](https://arxiv.org/pdf/2501.18991.pdf)** (2025-02-03)

*CNRS  Gauthier Thurin, CNRS  Kimia Nadjahi, LMO  Claire Boyer*

  Conformal Prediction (CP) is a principled framework for quantifying
uncertainty in blackbox learning models, by constructing prediction sets with
finite-sample coverage guarantees. Traditional approaches rely on scalar
nonconformity scores, which fail to fully exploit the geometric structure of
multivariate outputs, such as in multi-output regression or multiclass
classification. Recent methods addressing this limitation impose predefined
convex shapes for the prediction sets, potentially misaligning with the
intrinsic data geometry. We introduce a novel CP procedure handling
multivariate score functions through the lens of optimal transport.
Specifically, we leverage Monge-Kantorovich vector ranks and quantiles to
construct prediction region with flexible, potentially non-convex shapes,
better suited to the complex uncertainty patterns encountered in multivariate
learning tasks. We prove that our approach ensures finite-sample,
distribution-free coverage properties, similar to typical CP methods. We then
adapt our method for multi-output regression and multiclass classification, and
also propose simple adjustments to generate adaptive prediction regions with
asymptotic conditional coverage guarantees. Finally, we evaluate our method on
practical regression and classification problems, illustrating its advantages
in terms of (conditional) coverage and efficiency.


---

**[64. [2409.15623] Safe Guard: an LLM-agent for Real-time Voice-based Hate Speech Detection
  in Social Virtual Reality](https://arxiv.org/pdf/2409.15623.pdf)** (2024-09-25)

*Yiwen Xu, Qinyang Hou, Hongyu Wan, Mirjana Prpa*

  In this paper, we present Safe Guard, an LLM-agent for the detection of hate
speech in voice-based interactions in social VR (VRChat). Our system leverages
Open AI GPT and audio feature extraction for real-time voice interactions. We
contribute a system design and evaluation of the system that demonstrates the
capability of our approach in detecting hate speech, and reducing false
positives compared to currently available approaches. Our results indicate the
potential of LLM-based agents in creating safer virtual environments and set
the groundwork for further advancements in LLM-driven moderation approaches.


---

**[65. [2410.03772] Precision Knowledge Editing: Enhancing Safety in Large Language Models](https://arxiv.org/pdf/2410.03772.pdf)** (2024-10-14)

*Xuying Li, Zhuo Li, Yuji Kosuga, Yasuhiro Yoshida, Victor Bian*

  Large language models (LLMs) have demonstrated remarkable capabilities, but
they also pose risks related to the generation of toxic or harmful content.
This work introduces Precision Knowledge Editing (PKE), an advanced technique
that builds upon existing knowledge editing methods to more effectively
identify and modify toxic parameter regions within LLMs. By leveraging neuron
weight tracking and activation pathway tracing, PKE achieves finer granularity
in toxic content management compared to previous methods like Detoxifying
Instance Neuron Modification (DINM). Our experiments demonstrate that PKE
significantly reduces the attack success rate (ASR) across various models,
including Llama2-7b and Llama-3-8b-instruct, while maintaining overall model
performance. Additionally, we also compared the performance of some
closed-source models (gpt-4-0613 and Claude 3 Sonnet) in our experiments, and
found that models adjusted using our method far outperformed the closed-source
models in terms of safety. This research contributes to the ongoing efforts to
make LLMs safer and more reliable for real-world applications.


---

**[66. [2404.19048] A Framework for Real-time Safeguarding the Text Generation of Large
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

**[67. [2502.10486] VLM-Guard: Safeguarding Vision-Language Models via Fulfilling Safety
  Alignment Gap](https://arxiv.org/pdf/2502.10486.pdf)** (2025-02-18)

*Qin Liu, Fei Wang, Chaowei Xiao, Muhao Chen*

  The emergence of vision language models (VLMs) comes with increased safety
concerns, as the incorporation of multiple modalities heightens vulnerability
to attacks. Although VLMs can be built upon LLMs that have textual safety
alignment, it is easily undermined when the vision modality is integrated. We
attribute this safety challenge to the modality gap, a separation of image and
text in the shared representation space, which blurs the distinction between
harmful and harmless queries that is evident in LLMs but weakened in VLMs. To
avoid safety decay and fulfill the safety alignment gap, we propose VLM-Guard,
an inference-time intervention strategy that leverages the LLM component of a
VLM as supervision for the safety alignment of the VLM. VLM-Guard projects the
representations of VLM into the subspace that is orthogonal to the safety
steering direction that is extracted from the safety-aligned LLM. Experimental
results on three malicious instruction settings show the effectiveness of
VLM-Guard in safeguarding VLM and fulfilling the safety alignment gap between
VLM and its LLM component.


---

**[68. [2410.09893] RMB: Comprehensively Benchmarking Reward Models in LLM Alignment](https://arxiv.org/pdf/2410.09893.pdf)** (2025-04-07)

*Enyu Zhou, Guodong Zheng, Binghai Wang, Zhiheng Xi, Shihan Dou, Rong Bao, Wei Shen, Limao Xiong, Jessica Fan, Yurong Mou, Rui Zheng, Tao Gui, Qi Zhang, Xuanjing Huang*

  Reward models (RMs) guide the alignment of large language models (LLMs),
steering them toward behaviors preferred by humans. Evaluating RMs is the key
to better aligning LLMs. However, the current evaluation of RMs may not
directly correspond to their alignment performance due to the limited
distribution of evaluation data and evaluation methods that are not closely
related to alignment objectives. To address these limitations, we propose RMB,
a comprehensive RM benchmark that covers over 49 real-world scenarios and
includes both pairwise and Best-of-N (BoN) evaluations to better reflect the
effectiveness of RMs in guiding alignment optimization. We demonstrate a
positive correlation between our benchmark and the downstream alignment task
performance. Based on our benchmark, we conduct extensive analysis on the
state-of-the-art RMs, revealing their generalization defects that were not
discovered by previous benchmarks, and highlighting the potential of generative
RMs. Furthermore, we delve into open questions in reward models, specifically
examining the effectiveness of majority voting for the evaluation of reward
models and analyzing the impact factors of generative RMs, including the
influence of evaluation criteria and instructing methods. Our evaluation code
and datasets are available at
https://github.com/Zhou-Zoey/RMB-Reward-Model-Benchmark.


---

**[69. [2308.13207] LLM2KB: Constructing Knowledge Bases using instruction tuned context
  aware Large Language Models](https://arxiv.org/pdf/2308.13207.pdf)** (2023-08-28)

*Anmol Nayak, Hari Prasad Timmapathini*

  The advent of Large Language Models (LLM) has revolutionized the field of
natural language processing, enabling significant progress in various
applications. One key area of interest is the construction of Knowledge Bases
(KB) using these powerful models. Knowledge bases serve as repositories of
structured information, facilitating information retrieval and inference tasks.
Our paper proposes LLM2KB, a system for constructing knowledge bases using
large language models, with a focus on the Llama 2 architecture and the
Wikipedia dataset. We perform parameter efficient instruction tuning for
Llama-2-13b-chat and StableBeluga-13B by training small injection models that
have only 0.05 % of the parameters of the base models using the Low Rank
Adaptation (LoRA) technique. These injection models have been trained with
prompts that are engineered to utilize Wikipedia page contexts of subject
entities fetched using a Dense Passage Retrieval (DPR) algorithm, to answer
relevant object entities for a given subject entity and relation. Our best
performing model achieved an average F1 score of 0.6185 across 21 relations in
the LM-KBC challenge held at the ISWC 2023 conference.


---

**[70. [2407.21772] ShieldGemma: Generative AI Content Moderation Based on Gemma](https://arxiv.org/pdf/2407.21772.pdf)** (2024-08-06)

*Wenjun Zeng, Yuchi Liu, Ryan Mullins, Ludovic Peran, Joe Fernandez, Hamza Harkous, Karthik Narasimhan, Drew Proud, Piyush Kumar, Bhaktipriya Radharapu, Olivia Sturman, Oscar Wahltinez*

  We present ShieldGemma, a comprehensive suite of LLM-based safety content
moderation models built upon Gemma2. These models provide robust,
state-of-the-art predictions of safety risks across key harm types (sexually
explicit, dangerous content, harassment, hate speech) in both user input and
LLM-generated output. By evaluating on both public and internal benchmarks, we
demonstrate superior performance compared to existing models, such as Llama
Guard (+10.8\% AU-PRC on public benchmarks) and WildCard (+4.3\%).
Additionally, we present a novel LLM-based data curation pipeline, adaptable to
a variety of safety-related tasks and beyond. We have shown strong
generalization performance for model trained mainly on synthetic data. By
releasing ShieldGemma, we provide a valuable resource to the research
community, advancing LLM safety and enabling the creation of more effective
content moderation solutions for developers.


---

**[71. [2402.02207] Safety Fine-Tuning at (Almost) No Cost: A Baseline for Vision Large
  Language Models](https://arxiv.org/pdf/2402.02207.pdf)** (2024-06-19)

*Yongshuo Zong, Ondrej Bohdal, Tingyang Yu, Yongxin Yang, Timothy Hospedales*

  Current vision large language models (VLLMs) exhibit remarkable capabilities
yet are prone to generate harmful content and are vulnerable to even the
simplest jailbreaking attacks. Our initial analysis finds that this is due to
the presence of harmful data during vision-language instruction fine-tuning,
and that VLLM fine-tuning can cause forgetting of safety alignment previously
learned by the underpinning LLM. To address this issue, we first curate a
vision-language safe instruction-following dataset VLGuard covering various
harmful categories. Our experiments demonstrate that integrating this dataset
into standard vision-language fine-tuning or utilizing it for post-hoc
fine-tuning effectively safety aligns VLLMs. This alignment is achieved with
minimal impact on, or even enhancement of, the models' helpfulness. The
versatility of our safety fine-tuning dataset makes it a valuable resource for
safety-testing existing VLLMs, training new models or safeguarding pre-trained
VLLMs. Empirical results demonstrate that fine-tuned VLLMs effectively reject
unsafe instructions and substantially reduce the success rates of several
black-box adversarial attacks, which approach zero in many cases. The code and
dataset are available at https://github.com/ys-zong/VLGuard.


---

**[72. [2310.09639] DPZero: Private Fine-Tuning of Language Models without Backpropagation](https://arxiv.org/pdf/2310.09639.pdf)** (2024-06-07)

*Liang Zhang, Bingcong Li, Kiran Koshy Thekumparampil, Sewoong Oh, Niao He*

  The widespread practice of fine-tuning large language models (LLMs) on
domain-specific data faces two major challenges in memory and privacy. First,
as the size of LLMs continues to grow, the memory demands of gradient-based
training methods via backpropagation become prohibitively high. Second, given
the tendency of LLMs to memorize training data, it is important to protect
potentially sensitive information in the fine-tuning data from being
regurgitated. Zeroth-order methods, which rely solely on forward passes,
substantially reduce memory consumption during training. However, directly
combining them with standard differentially private gradient descent suffers
more as model size grows. To bridge this gap, we introduce DPZero, a novel
private zeroth-order algorithm with nearly dimension-independent rates. The
memory efficiency of DPZero is demonstrated in privately fine-tuning RoBERTa
and OPT on several downstream tasks. Our code is available at
https://github.com/Liang137/DPZero.


---

**[73. [2411.06493] LProtector: An LLM-driven Vulnerability Detection System](https://arxiv.org/pdf/2411.06493.pdf)** (2024-11-15)

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

**[74. [2410.20290] Fast Best-of-N Decoding via Speculative Rejection](https://arxiv.org/pdf/2410.20290.pdf)** (2024-11-04)

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

**[75. [2411.08320] Responsible AI in Construction Safety: Systematic Evaluation of Large
  Language Models and Prompt Engineering](https://arxiv.org/pdf/2411.08320.pdf)** (2024-11-14)

*Farouq Sammour, Jia Xu, Xi Wang, Mo Hu, Zhenyu Zhang*

  Construction remains one of the most hazardous sectors. Recent advancements
in AI, particularly Large Language Models (LLMs), offer promising opportunities
for enhancing workplace safety. However, responsible integration of LLMs
requires systematic evaluation, as deploying them without understanding their
capabilities and limitations risks generating inaccurate information, fostering
misplaced confidence, and compromising worker safety. This study evaluates the
performance of two widely used LLMs, GPT-3.5 and GPT-4o, across three
standardized exams administered by the Board of Certified Safety Professionals
(BCSP). Using 385 questions spanning seven safety knowledge areas, the study
analyzes the models' accuracy, consistency, and reliability. Results show that
both models consistently exceed the BCSP benchmark, with GPT-4o achieving an
accuracy rate of 84.6% and GPT-3.5 reaching 73.8%. Both models demonstrate
strengths in safety management systems and hazard identification and control,
but exhibit weaknesses in science, mathematics, emergency response, and fire
prevention. An error analysis identifies four primary limitations affecting LLM
performance: lack of knowledge, reasoning flaws, memory issues, and calculation
errors. Our study also highlights the impact of prompt engineering strategies,
with variations in accuracy reaching 13.5% for GPT-3.5 and 7.9% for GPT-4o.
However, no single prompt configuration proves universally effective. This
research advances knowledge in three ways: by identifying areas where LLMs can
support safety practices and where human oversight remains essential, by
offering practical insights into improving LLM implementation through prompt
engineering, and by providing evidence-based direction for future research and
development. These contributions support the responsible integration of AI in
construction safety management toward achieving zero injuries.


---

**[76. [2410.15483] Mitigating Forgetting in LLM Supervised Fine-Tuning and Preference
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

**[77. [2409.07902] Conformal Distributed Remote Inference in Sensor Networks Under
  Reliability and Communication Constraints](https://arxiv.org/pdf/2409.07902.pdf)** (2025-02-25)

*Meiyi Zhu, Matteo Zecchin, Sangwoo Park, Caili Guo, Chunyan Feng, Petar Popovski, Osvaldo Simeone*

  This paper presents communication-constrained distributed conformal risk
control (CD-CRC) framework, a novel decision-making framework for sensor
networks under communication constraints. Targeting multi-label classification
problems, such as segmentation, CD-CRC dynamically adjusts local and global
thresholds used to identify significant labels with the goal of ensuring a
target false negative rate (FNR), while adhering to communication capacity
limits. CD-CRC builds on online exponentiated gradient descent to estimate the
relative quality of the observations of different sensors, and on online
conformal risk control (CRC) as a mechanism to control local and global
thresholds. CD-CRC is proved to offer deterministic worst-case performance
guarantees in terms of FNR and communication overhead, while the regret
performance in terms of false positive rate (FPR) is characterized as a
function of the key hyperparameters. Simulation results highlight the
effectiveness of CD-CRC, particularly in communication resource-constrained
environments, making it a valuable tool for enhancing the performance and
reliability of distributed sensor networks.


---

**[78. [2502.00075] BTS: Harmonizing Specialized Experts into a Generalist LLM](https://arxiv.org/pdf/2502.00075.pdf)** (2025-02-04)

*Qizhen Zhang, Prajjwal Bhargava, Chloe Bi, Chris X. Cai, Jakob Foerster, Jeremy Fu, Punit Singh Koura, Ruan Silva, Sheng Shen, Emily Dinan, Suchin Gururangan, Mike Lewis*

  We present Branch-Train-Stitch (BTS), an efficient and flexible training
algorithm for combining independently trained large language model (LLM)
experts into a single, capable generalist model. Following Li et al., we start
with a single seed language model which is branched into domain-specific (e.g.,
coding or math) experts with continual pretraining. BTS combines experts into a
generalist model using lightweight stitch layers, which are inserted between
frozen experts and the seed LLM, and trained on a small datamix of the expert
domains. Stitch layers enable the seed LLM to integrate representations from
any number of experts during the forward pass, allowing it to generalize to new
domains, despite remaining frozen. Because BTS does not alter the constituent
LLMs, BTS provides a modular and flexible approach: experts can be easily
removed and new experts can be added with only a small amount of training.
Compared to alternative model merging approaches, BTS yields the best
generalist performance on a variety of downstream tasks, retaining the
specialized capabilities of each of the experts.


---

**[79. [2402.05624] Efficient Models for the Detection of Hate, Abuse and Profanity](https://arxiv.org/pdf/2402.05624.pdf)** (2024-02-09)

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

**[80. [2411.05897] Humans and Large Language Models in Clinical Decision Support: A Study
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

**[81. [2502.13640] Qorgau: Evaluating LLM Safety in Kazakh-Russian Bilingual Contexts](https://arxiv.org/pdf/2502.13640.pdf)** (2025-02-20)

*Maiya Goloburda, Nurkhan Laiyk, Diana Turmakhan, Yuxia Wang, Mukhammed Togmanov, Jonibek Mansurov, Askhat Sametov, Nurdaulet Mukhituly, Minghan Wang, Daniil Orel, Zain Muhammad Mujahid, Fajri Koto, Timothy Baldwin, Preslav Nakov*

  Large language models (LLMs) are known to have the potential to generate
harmful content, posing risks to users. While significant progress has been
made in developing taxonomies for LLM risks and safety evaluation prompts, most
studies have focused on monolingual contexts, primarily in English. However,
language- and region-specific risks in bilingual contexts are often overlooked,
and core findings can diverge from those in monolingual settings. In this
paper, we introduce Qorgau, a novel dataset specifically designed for safety
evaluation in Kazakh and Russian, reflecting the unique bilingual context in
Kazakhstan, where both Kazakh (a low-resource language) and Russian (a
high-resource language) are spoken. Experiments with both multilingual and
language-specific LLMs reveal notable differences in safety performance,
emphasizing the need for tailored, region-specific datasets to ensure the
responsible and safe deployment of LLMs in countries like Kazakhstan. Warning:
this paper contains example data that may be offensive, harmful, or biased.


---

**[82. [2403.09572] Eyes Closed, Safety On: Protecting Multimodal LLMs via Image-to-Text
  Transformation](https://arxiv.org/pdf/2403.09572.pdf)** (2024-10-16)

*Yunhao Gou, Kai Chen, Zhili Liu, Lanqing Hong, Hang Xu, Zhenguo Li, Dit-Yan Yeung, James T. Kwok, Yu Zhang*

  Multimodal large language models (MLLMs) have shown impressive reasoning
abilities. However, they are also more vulnerable to jailbreak attacks than
their LLM predecessors. Although still capable of detecting the unsafe
responses, we observe that safety mechanisms of the pre-aligned LLMs in MLLMs
can be easily bypassed with the introduction of image features. To construct
robust MLLMs, we propose ECSO (Eyes Closed, Safety On), a novel training-free
protecting approach that exploits the inherent safety awareness of MLLMs, and
generates safer responses via adaptively transforming unsafe images into texts
to activate the intrinsic safety mechanism of pre-aligned LLMs in MLLMs.
Experiments on five state-of-the-art (SoTA) MLLMs demonstrate that ECSO
enhances model safety significantly (e.g.,, 37.6% improvement on the
MM-SafetyBench (SD+OCR) and 71.3% on VLSafe with LLaVA-1.5-7B), while
consistently maintaining utility results on common MLLM benchmarks.
Furthermore, we show that ECSO can be used as a data engine to generate
supervised-finetuning (SFT) data for MLLM alignment without extra human
intervention.


---

**[83. [2502.13347] Craw4LLM: Efficient Web Crawling for LLM Pretraining](https://arxiv.org/pdf/2502.13347.pdf)** (2025-02-26)

*Shi Yu, Zhiyuan Liu, Chenyan Xiong*

  Web crawl is a main source of large language models' (LLMs) pretraining data,
but the majority of crawled web pages are discarded in pretraining due to low
data quality. This paper presents Craw4LLM, an efficient web crawling method
that explores the web graph based on the preference of LLM pretraining.
Specifically, it leverages the influence of a webpage in LLM pretraining as the
priority score of the web crawler's scheduler, replacing the standard graph
connectivity based priority. Our experiments on a web graph containing 900
million webpages from a commercial search engine's index demonstrate the
efficiency of Craw4LLM in obtaining high-quality pretraining data. With just
21% URLs crawled, LLMs pretrained on Craw4LLM data reach the same downstream
performances of previous crawls, significantly reducing the crawling waste and
alleviating the burdens on websites. Our code is publicly available at
https://github.com/cxcscmu/Craw4LLM.


---

**[84. [2503.21598] Prompt, Divide, and Conquer: Bypassing Large Language Model Safety
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

**[85. [2402.03327] Uni3D-LLM: Unifying Point Cloud Perception, Generation and Editing with
  Large Language Models](https://arxiv.org/pdf/2402.03327.pdf)** (2024-02-07)

*Dingning Liu, Xiaoshui Huang, Yuenan Hou, Zhihui Wang, Zhenfei Yin, Yongshun Gong, Peng Gao, Wanli Ouyang*

  In this paper, we introduce Uni3D-LLM, a unified framework that leverages a
Large Language Model (LLM) to integrate tasks of 3D perception, generation, and
editing within point cloud scenes. This framework empowers users to
effortlessly generate and modify objects at specified locations within a scene,
guided by the versatility of natural language descriptions. Uni3D-LLM harnesses
the expressive power of natural language to allow for precise command over the
generation and editing of 3D objects, thereby significantly enhancing
operational flexibility and controllability. By mapping point cloud into the
unified representation space, Uni3D-LLM achieves cross-application
functionality, enabling the seamless execution of a wide array of tasks,
ranging from the accurate instantiation of 3D objects to the diverse
requirements of interactive design. Through a comprehensive suite of rigorous
experiments, the efficacy of Uni3D-LLM in the comprehension, generation, and
editing of point cloud has been validated. Additionally, we have assessed the
impact of integrating a point cloud perception module on the generation and
editing processes, confirming the substantial potential of our approach for
practical applications.


---

**[86. [2406.10630] Emerging Safety Attack and Defense in Federated Instruction Tuning of
  Large Language Models](https://arxiv.org/pdf/2406.10630.pdf)** (2024-06-18)

*Rui Ye, Jingyi Chai, Xiangrui Liu, Yaodong Yang, Yanfeng Wang, Siheng Chen*

  Federated learning (FL) enables multiple parties to collaboratively fine-tune
an large language model (LLM) without the need of direct data sharing. Ideally,
by training on decentralized data that is aligned with human preferences and
safety principles, federated instruction tuning can result in an LLM that could
behave in a helpful and safe manner. In this paper, we for the first time
reveal the vulnerability of safety alignment in FedIT by proposing a simple,
stealthy, yet effective safety attack method. Specifically, the malicious
clients could automatically generate attack data without involving manual
efforts and attack the FedIT system by training their local LLMs on such attack
data. Unfortunately, this proposed safety attack not only can compromise the
safety alignment of LLM trained via FedIT, but also can not be effectively
defended against by many existing FL defense methods. Targeting this, we
further propose a post-hoc defense method, which could rely on a fully
automated pipeline: generation of defense data and further fine-tuning of the
LLM. Extensive experiments show that our safety attack method can significantly
compromise the LLM's safety alignment (e.g., reduce safety rate by 70\%), which
can not be effectively defended by existing defense methods (at most 4\%
absolute improvement), while our safety defense method can significantly
enhance the attacked LLM's safety alignment (at most 69\% absolute
improvement).


---

**[87. [2407.17377] Entropy Reweighted Conformal Classification](https://arxiv.org/pdf/2407.17377.pdf)** (2024-07-25)

*Rui Luo, Nicolo Colombo*

  Conformal Prediction (CP) is a powerful framework for constructing prediction
sets with guaranteed coverage. However, recent studies have shown that
integrating confidence calibration with CP can lead to a degradation in
efficiency. In this paper, We propose an adaptive approach that considers the
classifier's uncertainty and employs entropy-based reweighting to enhance the
efficiency of prediction sets for conformal classification. Our experimental
results demonstrate that this method significantly improves efficiency.


---

**[88. [2503.04418] AOLO: Analysis and Optimization For Low-Carbon Oriented Wireless Large
  Language Model Services](https://arxiv.org/pdf/2503.04418.pdf)** (2025-03-07)

*Xiaoqi Wang, Hongyang Du, Yuehong Gao, Dong In Kim*

  Recent advancements in large language models (LLMs) have led to their
widespread adoption and large-scale deployment across various domains. However,
their environmental impact, particularly during inference, has become a growing
concern due to their substantial energy consumption and carbon footprint.
Existing research has focused on inference computation alone, overlooking the
analysis and optimization of carbon footprint in network-aided LLM service
systems. To address this gap, we propose AOLO, a framework for analysis and
optimization for low-carbon oriented wireless LLM services. AOLO introduces a
comprehensive carbon footprint model that quantifies greenhouse gas emissions
across the entire LLM service chain, including computational inference and
wireless communication. Furthermore, we formulate an optimization problem aimed
at minimizing the overall carbon footprint, which is solved through joint
optimization of inference outputs and transmit power under
quality-of-experience and system performance constraints. To achieve this joint
optimization, we leverage the energy efficiency of spiking neural networks
(SNNs) by adopting SNN as the actor network and propose a low-carbon-oriented
optimization algorithm, i.e., SNN-based deep reinforcement learning (SDRL).
Comprehensive simulations demonstrate that SDRL algorithm significantly reduces
overall carbon footprint, achieving an 18.77% reduction compared to the
benchmark soft actor-critic, highlighting its potential for enabling more
sustainable LLM inference services.


---

**[89. [2210.10254] Safe Planning in Dynamic Environments using Conformal Prediction](https://arxiv.org/pdf/2210.10254.pdf)** (2023-06-09)

*Lars Lindemann, Matthew Cleaveland, Gihyun Shim, George J. Pappas*

  We propose a framework for planning in unknown dynamic environments with
probabilistic safety guarantees using conformal prediction. Particularly, we
design a model predictive controller (MPC) that uses i) trajectory predictions
of the dynamic environment, and ii) prediction regions quantifying the
uncertainty of the predictions. To obtain prediction regions, we use conformal
prediction, a statistical tool for uncertainty quantification, that requires
availability of offline trajectory data - a reasonable assumption in many
applications such as autonomous driving. The prediction regions are valid,
i.e., they hold with a user-defined probability, so that the MPC is provably
safe. We illustrate the results in the self-driving car simulator CARLA at a
pedestrian-filled intersection. The strength of our approach is compatibility
with state of the art trajectory predictors, e.g., RNNs and LSTMs, while making
no assumptions on the underlying trajectory-generating distribution. To the
best of our knowledge, these are the first results that provide valid safety
guarantees in such a setting.


---

**[90. [2503.00187] Steering Dialogue Dynamics for Robustness against Multi-turn
  Jailbreaking Attacks](https://arxiv.org/pdf/2503.00187.pdf)** (2025-03-04)

*Hanjiang Hu, Alexander Robey, Changliu Liu*

  Large language models (LLMs) are highly vulnerable to jailbreaking attacks,
wherein adversarial prompts are designed to elicit harmful responses. While
existing defenses effectively mitigate single-turn attacks by detecting and
filtering unsafe inputs, they fail against multi-turn jailbreaks that exploit
contextual drift over multiple interactions, gradually leading LLMs away from
safe behavior. To address this challenge, we propose a safety steering
framework grounded in safe control theory, ensuring invariant safety in
multi-turn dialogues. Our approach models the dialogue with LLMs using
state-space representations and introduces a novel neural barrier function
(NBF) to detect and filter harmful queries emerging from evolving contexts
proactively. Our method achieves invariant safety at each turn of dialogue by
learning a safety predictor that accounts for adversarial queries, preventing
potential context drift toward jailbreaks. Extensive experiments under multiple
LLMs show that our NBF-based safety steering outperforms safety alignment
baselines, offering stronger defenses against multi-turn jailbreaks while
maintaining a better trade-off between safety and helpfulness under different
multi-turn jailbreak methods. Our code is available at
https://github.com/HanjiangHu/NBF-LLM .


---

**[91. [2502.05242] SEER: Self-Explainability Enhancement of Large Language Models'
  Representations](https://arxiv.org/pdf/2502.05242.pdf)** (2025-02-11)

*Guanxu Chen, Dongrui Liu, Tao Luo, Jing Shao*

  Explaining the hidden representations of Large Language Models (LLMs) is a
perspective to understand LLMs' underlying inference logic and improve their
reliability in application scenarios. However, previous methods introduce
external ''black-box'' modules to explain ''black-box'' LLMs, increasing the
potential uncertainty and failing to provide faithful explanations. In this
paper, we propose a self-explaining method SEER, enhancing LLMs' explainability
by aggregating the same concept and disentangling the different concepts in the
representation space. In this way, SEER provides faithful explanations carried
by representations synchronously with the LLMs' output. Additionally, we
showcase the applications of SEER on trustworthiness-related tasks (e.g., the
safety risks classification and detoxification tasks), where self-explained
LLMs achieve consistent improvement in explainability and performance. More
crucially, we theoretically analyze the improvement of SEER on LLMs'
generalization ability through optimal transport theory.


---

**[92. [2502.13603] Efficient Safety Retrofitting Against Jailbreaking for LLMs](https://arxiv.org/pdf/2502.13603.pdf)** (2025-02-26)

*Dario Garcia-Gasulla, Adrian Tormos, Anna Arias-Duart, Daniel Hinjos, Oscar Molina-Sedano, Ashwin Kumar Gururajan, Maria Eugenia Cardello*

  Direct Preference Optimization (DPO) is an efficient alignment technique that
steers LLMs towards preferable outputs by training on preference data,
bypassing the need for explicit reward models. Its simplicity enables easy
adaptation to various domains and safety requirements. This paper examines
DPO's effectiveness in model safety against jailbreaking attacks while
minimizing data requirements and training costs. We introduce Egida, a dataset
expanded from multiple sources, which includes 27 different safety topics and
18 different attack styles, complemented with synthetic and human labels. This
data is used to boost the safety of state-of-the-art LLMs
(Llama-3.1-8B/70B-Instruct, Qwen-2.5-7B/72B-Instruct) across topics and attack
styles. In addition to safety evaluations, we assess their post-alignment
performance degradation in general purpose tasks, and their tendency to over
refusal. Following the proposed methodology, trained models reduce their Attack
Success Rate by 10%-30%, using small training efforts (2,000 samples) with low
computational cost (3\$ for 8B models, 20\$ for 72B models). Safety aligned
models generalize to unseen topics and attack styles, with the most successful
attack style reaching a success rate around 5%. Size and family are found to
strongly influence model malleability towards safety, pointing at the
importance of pre-training choices. To validate our findings, a large
independent assessment of human preference agreement with Llama-Guard-3-8B is
conducted by the authors and the associated dataset Egida-HSafe is released.
Overall, this study illustrates how affordable and accessible it is to enhance
LLM safety using DPO while outlining its current limitations. All datasets and
models are released to enable reproducibility and further research.


---

**[93. [2305.07507] LeXFiles and LegalLAMA: Facilitating English Multinational Legal
  Language Model Development](https://arxiv.org/pdf/2305.07507.pdf)** (2023-05-24)

*Ilias Chalkidis, Nicolas Garneau, Catalina Goanta, Daniel Martin Katz, Anders Søgaard*

  In this work, we conduct a detailed analysis on the performance of
legal-oriented pre-trained language models (PLMs). We examine the interplay
between their original objective, acquired knowledge, and legal language
understanding capacities which we define as the upstream, probing, and
downstream performance, respectively. We consider not only the models' size but
also the pre-training corpora used as important dimensions in our study. To
this end, we release a multinational English legal corpus (LeXFiles) and a
legal knowledge probing benchmark (LegalLAMA) to facilitate training and
detailed analysis of legal-oriented PLMs. We release two new legal PLMs trained
on LeXFiles and evaluate them alongside others on LegalLAMA and LexGLUE. We
find that probing performance strongly correlates with upstream performance in
related legal topics. On the other hand, downstream performance is mainly
driven by the model's size and prior legal knowledge which can be estimated by
upstream and probing performance. Based on these findings, we can conclude that
both dimensions are important for those seeking the development of
domain-specific PLMs.


---

**[94. [2412.12597] Distribution-Free Uncertainty Quantification in Mechanical Ventilation
  Treatment: A Conformal Deep Q-Learning Framework](https://arxiv.org/pdf/2412.12597.pdf)** (2024-12-18)

*Niloufar Eghbali, Tuka Alhanai, Mohammad M. Ghassemi*

  Mechanical Ventilation (MV) is a critical life-support intervention in
intensive care units (ICUs). However, optimal ventilator settings are
challenging to determine because of the complexity of balancing
patient-specific physiological needs with the risks of adverse outcomes that
impact morbidity, mortality, and healthcare costs. This study introduces
ConformalDQN, a novel distribution-free conformal deep Q-learning approach for
optimizing mechanical ventilation in intensive care units. By integrating
conformal prediction with deep reinforcement learning, our method provides
reliable uncertainty quantification, addressing the challenges of Q-value
overestimation and out-of-distribution actions in offline settings. We trained
and evaluated our model using ICU patient records from the MIMIC-IV database.
ConformalDQN extends the Double DQN architecture with a conformal predictor and
employs a composite loss function that balances Q-learning with well-calibrated
probability estimation. This enables uncertainty-aware action selection,
allowing the model to avoid potentially harmful actions in unfamiliar states
and handle distribution shifts by being more conservative in
out-of-distribution scenarios. Evaluation against baseline models, including
physician policies, policy constraint methods, and behavior cloning,
demonstrates that ConformalDQN consistently makes recommendations within
clinically safe and relevant ranges, outperforming other methods by increasing
the 90-day survival rate. Notably, our approach provides an interpretable
measure of confidence in its decisions, which is crucial for clinical adoption
and potential human-in-the-loop implementations.


---

**[95. [2404.17287] When to Trust LLMs: Aligning Confidence with Response Quality](https://arxiv.org/pdf/2404.17287.pdf)** (2024-10-01)

*Shuchang Tao, Liuyi Yao, Hanxing Ding, Yuexiang Xie, Qi Cao, Fei Sun, Jinyang Gao, Huawei Shen, Bolin Ding*

  Despite the success of large language models (LLMs) in natural language
generation, much evidence shows that LLMs may produce incorrect or nonsensical
text. This limitation highlights the importance of discerning when to trust
LLMs, especially in safety-critical domains. Existing methods often express
reliability by confidence level, however, their effectiveness is limited by the
lack of objective guidance. To address this, we propose
CONfidence-Quality-ORDer-preserving alignment approach (CONQORD), which
leverages reinforcement learning guided by a tailored dual-component reward
function. This function integrates quality reward and order-preserving
alignment reward functions. Specifically, the order-preserving reward
incentivizes the model to verbalize greater confidence for responses of higher
quality to align the order of confidence and quality. Experiments demonstrate
that CONQORD significantly improves the alignment performance between
confidence and response accuracy, without causing over-cautious. Furthermore,
the aligned confidence provided by CONQORD informs when to trust LLMs, and acts
as a determinant for initiating the retrieval process of external knowledge.
Aligning confidence with response quality ensures more transparent and reliable
responses, providing better trustworthiness.


---

**[96. [2502.13416] Detecting LLM Fact-conflicting Hallucinations Enhanced by
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

**[97. [2403.18051] Supervisory Prompt Training](https://arxiv.org/pdf/2403.18051.pdf)** (2024-03-28)

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

**[98. [2412.00383] Unified Parameter-Efficient Unlearning for LLMs](https://arxiv.org/pdf/2412.00383.pdf)** (2025-04-21)

*Chenlu Ding, Jiancan Wu, Yancheng Yuan, Jinda Lu, Kai Zhang, Alex Su, Xiang Wang, Xiangnan He*

  The advent of Large Language Models (LLMs) has revolutionized natural
language processing, enabling advanced understanding and reasoning capabilities
across a variety of tasks. Fine-tuning these models for specific domains,
particularly through Parameter-Efficient Fine-Tuning (PEFT) strategies like
LoRA, has become a prevalent practice due to its efficiency. However, this
raises significant privacy and security concerns, as models may inadvertently
retain and disseminate sensitive or undesirable information. To address these
issues, we introduce a novel instance-wise unlearning framework, LLMEraser,
which systematically categorizes unlearning tasks and applies precise parameter
adjustments using influence functions. Unlike traditional unlearning techniques
that are often limited in scope and require extensive retraining, LLMEraser is
designed to handle a broad spectrum of unlearning tasks without compromising
model performance. Extensive experiments on benchmark datasets demonstrate that
LLMEraser excels in efficiently managing various unlearning scenarios while
maintaining the overall integrity and efficacy of the models.


---

**[99. [2403.09972] Think Twice Before Trusting: Self-Detection for Large Language Models
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

**[100. [2406.10040] FZI-WIM at SemEval-2024 Task 2: Self-Consistent CoT for Complex NLI in
  Biomedical Domain](https://arxiv.org/pdf/2406.10040.pdf)** (2024-06-17)

*Jin Liu, Steffen Thoma*

  This paper describes the inference system of FZI-WIM at the SemEval-2024 Task
2: Safe Biomedical Natural Language Inference for Clinical Trials. Our system
utilizes the chain of thought (CoT) paradigm to tackle this complex reasoning
problem and further improves the CoT performance with self-consistency. Instead
of greedy decoding, we sample multiple reasoning chains with the same prompt
and make the final verification with majority voting. The self-consistent CoT
system achieves a baseline F1 score of 0.80 (1st), faithfulness score of 0.90
(3rd), and consistency score of 0.73 (12th). We release the code and data
publicly https://github.com/jens5588/FZI-WIM-NLI4CT.


---

**[101. [2412.10423] Look Before You Leap: Enhancing Attention and Vigilance Regarding
  Harmful Content with GuidelineLLM](https://arxiv.org/pdf/2412.10423.pdf)** (2025-04-15)

*Shaoqing Zhang, Zhuosheng Zhang, Kehai Chen, Rongxiang Weng, Muyun Yang, Tiejun Zhao, Min Zhang*

  Despite being empowered with alignment mechanisms, large language models
(LLMs) are increasingly vulnerable to emerging jailbreak attacks that can
compromise their alignment mechanisms. This vulnerability poses significant
risks to real-world applications. Existing work faces challenges in both
training efficiency and generalization capabilities (i.e., Reinforcement
Learning from Human Feedback and Red-Teaming). Developing effective strategies
to enable LLMs to resist continuously evolving jailbreak attempts represents a
significant challenge. To address this challenge, we propose a novel defensive
paradigm called GuidelineLLM, which assists LLMs in recognizing queries that
may have harmful content. Before LLMs respond to a query, GuidelineLLM first
identifies potential risks associated with the query, summarizes these risks
into guideline suggestions, and then feeds these guidelines to the responding
LLMs. Importantly, our approach eliminates the necessity for additional safety
fine-tuning of the LLMs themselves; only the GuidelineLLM requires fine-tuning.
This characteristic enhances the general applicability of GuidelineLLM across
various LLMs. Experimental results demonstrate that GuidelineLLM can
significantly reduce the attack success rate (ASR) against LLM (an average
reduction of 34.17\% ASR) while maintaining the usefulness of LLM in handling
benign queries. The code is available at
https://github.com/sqzhang-lazy/GuidelineLLM.


---

**[102. [2503.18666] AgentSpec: Customizable Runtime Enforcement for Safe and Reliable LLM
  Agents](https://arxiv.org/pdf/2503.18666.pdf)** (2025-04-08)

*Haoyu Wang, Christopher M. Poskitt, Jun Sun*

  Agents built on LLMs are increasingly deployed across diverse domains,
automating complex decision-making and task execution. However, their autonomy
introduces safety risks, including security vulnerabilities, legal violations,
and unintended harmful actions. Existing mitigation methods, such as
model-based safeguards and early enforcement strategies, fall short in
robustness, interpretability, and adaptability. To address these challenges, we
propose AgentSpec, a lightweight domain-specific language for specifying and
enforcing runtime constraints on LLM agents. With AgentSpec, users define
structured rules that incorporate triggers, predicates, and enforcement
mechanisms, ensuring agents operate within predefined safety boundaries. We
implement AgentSpec across multiple domains, including code execution, embodied
agents, and autonomous driving, demonstrating its adaptability and
effectiveness. Our evaluation shows that AgentSpec successfully prevents unsafe
executions in over 90% of code agent cases, eliminates all hazardous actions in
embodied agent tasks, and enforces 100% compliance by autonomous vehicles
(AVs). Despite its strong safety guarantees, AgentSpec remains computationally
lightweight, with overheads in milliseconds. By combining interpretability,
modularity, and efficiency, AgentSpec provides a practical and scalable
solution for enforcing LLM agent safety across diverse applications. We also
automate the generation of rules using LLMs and assess their effectiveness. Our
evaluation shows that the rules generated by OpenAI o1 achieve a precision of
95.56% and recall of 70.96% for embodied agents, successfully identifying
87.26% of the risky code, and prevent AVs from breaking laws in 5 out of 8
scenarios.


---

**[103. [2410.01978] LLM+KG@VLDB'24 Workshop Summary](https://arxiv.org/pdf/2410.01978.pdf)** (2025-03-25)

*Arijit Khan, Tianxing Wu, Xi Chen*

  The unification of large language models (LLMs) and knowledge graphs (KGs)
has emerged as a hot topic. At the LLM+KG'24 workshop, held in conjunction with
VLDB 2024 in Guangzhou, China, one of the key themes explored was important
data management challenges and opportunities due to the effective interaction
between LLMs and KGs. This report outlines the major directions and approaches
presented by various speakers during the LLM+KG'24 workshop.


---

**[104. [2404.11086] ViLLM-Eval: A Comprehensive Evaluation Suite for Vietnamese Large
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

**[105. [2501.00274] LLM-Rubric: A Multidimensional, Calibrated Approach to Automated
  Evaluation of Natural Language Texts](https://arxiv.org/pdf/2501.00274.pdf)** (2025-01-03)

*Helia Hashemi, Jason Eisner, Corby Rosset, Benjamin Van Durme, Chris Kedzie*

  This paper introduces a framework for the automated evaluation of natural
language texts. A manually constructed rubric describes how to assess multiple
dimensions of interest. To evaluate a text, a large language model (LLM) is
prompted with each rubric question and produces a distribution over potential
responses. The LLM predictions often fail to agree well with human judges --
indeed, the humans do not fully agree with one another. However, the multiple
LLM distributions can be $\textit{combined}$ to $\textit{predict}$ each human
judge's annotations on all questions, including a summary question that
assesses overall quality or relevance. LLM-Rubric accomplishes this by training
a small feed-forward neural network that includes both judge-specific and
judge-independent parameters. When evaluating dialogue systems in a human-AI
information-seeking task, we find that LLM-Rubric with 9 questions (assessing
dimensions such as naturalness, conciseness, and citation quality) predicts
human judges' assessment of overall user satisfaction, on a scale of 1--4, with
RMS error $< 0.5$, a $2\times$ improvement over the uncalibrated baseline.


---

**[106. [2304.11363] Lexicographic Ranking Supermartingales with Lazy Lower Bounds](https://arxiv.org/pdf/2304.11363.pdf)** (2025-04-14)

*Toru Takisaka, Libo Zhang, Changjiang Wang, Jiamou Liu*

  Lexicographic Ranking SuperMartingale (LexRSM) is a probabilistic extension
of Lexicographic Ranking Function (LexRF), which is a widely accepted technique
for verifying program termination. In this paper, we are the first to propose
sound probabilistic extensions of LexRF with a weaker non-negativity condition,
called single-component (SC) non-negativity. It is known that such an
extension, if it exists, will be nontrivial due to the intricacies of the
probabilistic circumstances.
  Toward the goal, we first devise the notion of fixability, which offers a
systematic approach for analyzing the soundness of possibly negative LexRSM.
This notion yields a desired extension of LexRF that is sound for general
stochastic processes. We next propose another extension, called Lazy LexRSM,
toward the application to automated verification; it is sound over
probabilistic programs with linear arithmetics, while its subclass is amenable
to automated synthesis via linear programming. We finally propose a LexRSM
synthesis algorithm for this subclass, and perform experiments.


---

**[107. [2108.04926] First Order Locally Orderless Registration](https://arxiv.org/pdf/2108.04926.pdf)** (2021-08-12)

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

**[108. [2111.08534] Finite element based model order reduction for parametrized one-way
  coupled steady state linear thermomechanical problems](https://arxiv.org/pdf/2111.08534.pdf)** (2023-08-08)

*Nirav Vasant Shah, Michele Girfoglio, Peregrina Quintela, Gianluigi Rozza, Alejandro Lengomin, Francesco Ballarin, Patricia Barral*

  This contribution focuses on the development of Model Order Reduction (MOR)
for one-way coupled steady state linear thermomechanical problems in a finite
element setting. We apply Proper Orthogonal Decomposition (POD) for the
computation of reduced basis space. On the other hand, for the evaluation of
the modal coefficients, we use two different methodologies: the one based on
the Galerkin projection (G) and the other one based on Artificial Neural
Network (ANN). We aim at comparing POD-G and POD-ANN in terms of relevant
features including errors and computational efficiency. In this context, both
physical and geometrical parametrization are considered. We also carry out a
validation of the Full Order Model (FOM) based on customized benchmarks in
order to provide a complete computational pipeline. The framework proposed is
applied to a relevant industrial problem related to the investigation of
thermomechanical phenomena arising in blast furnace hearth walls.
  Keywords: Thermomechanical problems, Finite element method, Proper orthogonal
decomposition, Galerkin projection, Artificial neural network, Geometric and
physical parametrization, Blast furnace.


---

**[109. [2401.09796] A Fast, Performant, Secure Distributed Training Framework For Large
  Language Model](https://arxiv.org/pdf/2401.09796.pdf)** (2024-01-22)

*Wei Huang, Yinggui Wang, Anda Cheng, Aihui Zhou, Chaofan Yu, Lei Wang*

  The distributed (federated) LLM is an important method for co-training the
domain-specific LLM using siloed data. However, maliciously stealing model
parameters and data from the server or client side has become an urgent problem
to be solved. In this paper, we propose a secure distributed LLM based on model
slicing. In this case, we deploy the Trusted Execution Environment (TEE) on
both the client and server side, and put the fine-tuned structure (LoRA or
embedding of P-tuning v2) into the TEE. Then, secure communication is executed
in the TEE and general environments through lightweight encryption. In order to
further reduce the equipment cost as well as increase the model performance and
accuracy, we propose a split fine-tuning scheme. In particular, we split the
LLM by layers and place the latter layers in a server-side TEE (the client does
not need a TEE). We then combine the proposed Sparsification Parameter
Fine-tuning (SPF) with the LoRA part to improve the accuracy of the downstream
task. Numerous experiments have shown that our method guarantees accuracy while
maintaining security.


---

**[110. [2406.13805] WikiContradict: A Benchmark for Evaluating LLMs on Real-World Knowledge
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

**[111. [2401.17882] I Think, Therefore I am: Benchmarking Awareness of Large Language Models
  Using AwareBench](https://arxiv.org/pdf/2401.17882.pdf)** (2024-02-19)

*Yuan Li, Yue Huang, Yuli Lin, Siyuan Wu, Yao Wan, Lichao Sun*

  Do large language models (LLMs) exhibit any forms of awareness similar to
humans? In this paper, we introduce AwareBench, a benchmark designed to
evaluate awareness in LLMs. Drawing from theories in psychology and philosophy,
we define awareness in LLMs as the ability to understand themselves as AI
models and to exhibit social intelligence. Subsequently, we categorize
awareness in LLMs into five dimensions, including capability, mission, emotion,
culture, and perspective. Based on this taxonomy, we create a dataset called
AwareEval, which contains binary, multiple-choice, and open-ended questions to
assess LLMs' understandings of specific awareness dimensions. Our experiments,
conducted on 13 LLMs, reveal that the majority of them struggle to fully
recognize their capabilities and missions while demonstrating decent social
intelligence. We conclude by connecting awareness of LLMs with AI alignment and
safety, emphasizing its significance to the trustworthy and ethical development
of LLMs. Our dataset and code are available at
https://github.com/HowieHwong/Awareness-in-LLM.


---

**[112. [2311.07689] MART: Improving LLM Safety with Multi-round Automatic Red-Teaming](https://arxiv.org/pdf/2311.07689.pdf)** (2023-11-15)

*Suyu Ge, Chunting Zhou, Rui Hou, Madian Khabsa, Yi-Chia Wang, Qifan Wang, Jiawei Han, Yuning Mao*

  Red-teaming is a common practice for mitigating unsafe behaviors in Large
Language Models (LLMs), which involves thoroughly assessing LLMs to identify
potential flaws and addressing them with responsible and accurate responses.
While effective, manual red-teaming is costly, and existing automatic
red-teaming typically discovers safety risks without addressing them. In this
paper, we propose a Multi-round Automatic Red-Teaming (MART) method, which
incorporates both automatic adversarial prompt writing and safe response
generation, significantly increasing red-teaming scalability and the safety of
the target LLM. Specifically, an adversarial LLM and a target LLM interplay
with each other in an iterative manner, where the adversarial LLM aims to
generate challenging prompts that elicit unsafe responses from the target LLM,
while the target LLM is fine-tuned with safety aligned data on these
adversarial prompts. In each round, the adversarial LLM crafts better attacks
on the updated target LLM, while the target LLM also improves itself through
safety fine-tuning. On adversarial prompt benchmarks, the violation rate of an
LLM with limited safety alignment reduces up to 84.7% after 4 rounds of MART,
achieving comparable performance to LLMs with extensive adversarial prompt
writing. Notably, model helpfulness on non-adversarial prompts remains stable
throughout iterations, indicating the target LLM maintains strong performance
on instruction following.


---

**[113. [2410.10343] Locking Down the Finetuned LLMs Safety](https://arxiv.org/pdf/2410.10343.pdf)** (2024-10-15)

*Minjun Zhu, Linyi Yang, Yifan Wei, Ningyu Zhang, Yue Zhang*

  Fine-tuning large language models (LLMs) on additional datasets is often
necessary to optimize them for specific downstream tasks. However, existing
safety alignment measures, which restrict harmful behavior during inference,
are insufficient to mitigate safety risks during fine-tuning. Alarmingly,
fine-tuning with just 10 toxic sentences can make models comply with harmful
instructions. We introduce SafetyLock, a novel alignment intervention method
that maintains robust safety post-fine-tuning through efficient and
transferable mechanisms. SafetyLock leverages our discovery that fine-tuned
models retain similar safety-related activation representations to their base
models. This insight enables us to extract what we term the Meta-SafetyLock, a
set of safety bias directions representing key activation patterns associated
with safe responses in the original model. We can then apply these directions
universally to fine-tuned models to enhance their safety. By searching for
activation directions across multiple token dimensions, SafetyLock achieves
enhanced robustness and transferability. SafetyLock re-aligns fine-tuned models
in under 0.01 seconds without additional computational cost. Our experiments
demonstrate that SafetyLock can reduce the harmful instruction response rate
from 60% to below 1% in toxic fine-tuned models. It surpasses traditional
methods in both performance and efficiency, offering a scalable, non-invasive
solution for ensuring the safety of customized LLMs. Our analysis across
various fine-tuning scenarios confirms SafetyLock's robustness, advocating its
integration into safety protocols for aligned LLMs. The code is released at
https://github.com/zhu-minjun/SafetyLock.


---

**[114. [2504.01550] Representation Bending for Large Language Model Safety](https://arxiv.org/pdf/2504.01550.pdf)** (2025-04-03)

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

**[115. [2311.11123] (Why) Is My Prompt Getting Worse? Rethinking Regression Testing for
  Evolving LLM APIs](https://arxiv.org/pdf/2311.11123.pdf)** (2024-02-08)

*Wanqin Ma, Chenyang Yang, Christian Kästner*

  Large Language Models (LLMs) are increasingly integrated into software
applications. Downstream application developers often access LLMs through APIs
provided as a service. However, LLM APIs are often updated silently and
scheduled to be deprecated, forcing users to continuously adapt to evolving
models. This can cause performance regression and affect prompt design choices,
as evidenced by our case study on toxicity detection. Based on our case study,
we emphasize the need for and re-examine the concept of regression testing for
evolving LLM APIs. We argue that regression testing LLMs requires fundamental
changes to traditional testing approaches, due to different correctness
notions, prompting brittleness, and non-determinism in LLM APIs.


---

**[116. [2410.22225] CaStL: Constraints as Specifications through LLM Translation for
  Long-Horizon Task and Motion Planning](https://arxiv.org/pdf/2410.22225.pdf)** (2024-10-30)

*Weihang Guo, Zachary Kingston, Lydia E. Kavraki*

  Large Language Models (LLMs) have demonstrated remarkable ability in
long-horizon Task and Motion Planning (TAMP) by translating clear and
straightforward natural language problems into formal specifications such as
the Planning Domain Definition Language (PDDL). However, real-world problems
are often ambiguous and involve many complex constraints. In this paper, we
introduce Constraints as Specifications through LLMs (CaStL), a framework that
identifies constraints such as goal conditions, action ordering, and action
blocking from natural language in multiple stages. CaStL translates these
constraints into PDDL and Python scripts, which are solved using an custom PDDL
solver. Tested across three PDDL domains, CaStL significantly improves
constraint handling and planning success rates from natural language
specification in complex scenarios.


---

**[117. [2304.01075] Conformal Prediction Regions for Time Series using Linear
  Complementarity Programming](https://arxiv.org/pdf/2304.01075.pdf)** (2024-01-10)

*Matthew Cleaveland, Insup Lee, George J. Pappas, Lars Lindemann*

  Conformal prediction is a statistical tool for producing prediction regions
of machine learning models that are valid with high probability. However,
applying conformal prediction to time series data leads to conservative
prediction regions. In fact, to obtain prediction regions over $T$ time steps
with confidence $1-\delta$, {previous works require that each individual
prediction region is valid} with confidence $1-\delta/T$. We propose an
optimization-based method for reducing this conservatism to enable long horizon
planning and verification when using learning-enabled time series predictors.
Instead of considering prediction errors individually at each time step, we
consider a parameterized prediction error over multiple time steps. By
optimizing the parameters over an additional dataset, we find prediction
regions that are not conservative. We show that this problem can be cast as a
mixed integer linear complementarity program (MILCP), which we then relax into
a linear complementarity program (LCP). Additionally, we prove that the relaxed
LP has the same optimal cost as the original MILCP. Finally, we demonstrate the
efficacy of our method on case studies using pedestrian trajectory predictors
and F16 fighter jet altitude predictors.


---

**[118. [2405.06237] Risks of Practicing Large Language Models in Smart Grid: Threat Modeling
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

**[119. [2412.00056] Improving Medical Diagnostics with Vision-Language Models: Convex
  Hull-Based Uncertainty Analysis](https://arxiv.org/pdf/2412.00056.pdf)** (2024-12-03)

*Ferhat Ozgur Catak, Murat Kuzlu, Taylor Patrick*

  In recent years, vision-language models (VLMs) have been applied to various
fields, including healthcare, education, finance, and manufacturing, with
remarkable performance. However, concerns remain regarding VLMs' consistency
and uncertainty, particularly in critical applications such as healthcare,
which demand a high level of trust and reliability. This paper proposes a novel
approach to evaluate uncertainty in VLMs' responses using a convex hull
approach on a healthcare application for Visual Question Answering (VQA).
LLM-CXR model is selected as the medical VLM utilized to generate responses for
a given prompt at different temperature settings, i.e., 0.001, 0.25, 0.50,
0.75, and 1.00. According to the results, the LLM-CXR VLM shows a high
uncertainty at higher temperature settings. Experimental outcomes emphasize the
importance of uncertainty in VLMs' responses, especially in healthcare
applications.


---

**[120. [2412.08072] Using Large Language Models for Parametric Shape Optimization](https://arxiv.org/pdf/2412.08072.pdf)** (2024-12-12)

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

**[121. [2310.18333] She had Cobalt Blue Eyes: Prompt Testing to Create Aligned and
  Sustainable Language Models](https://arxiv.org/pdf/2310.18333.pdf)** (2023-12-18)

*Veronica Chatrath, Oluwanifemi Bamgbose, Shaina Raza*

  As the use of large language models (LLMs) increases within society, as does
the risk of their misuse. Appropriate safeguards must be in place to ensure LLM
outputs uphold the ethical standards of society, highlighting the positive role
that artificial intelligence technologies can have. Recent events indicate
ethical concerns around conventionally trained LLMs, leading to overall unsafe
user experiences. This motivates our research question: how do we ensure LLM
alignment? In this work, we introduce a test suite of unique prompts to foster
the development of aligned LLMs that are fair, safe, and robust. We show that
prompting LLMs at every step of the development pipeline, including data
curation, pre-training, and fine-tuning, will result in an overall more
responsible model. Our test suite evaluates outputs from four state-of-the-art
language models: GPT-3.5, GPT-4, OPT, and LLaMA-2. The assessment presented in
this paper highlights a gap between societal alignment and the capabilities of
current LLMs. Additionally, implementing a test suite such as ours lowers the
environmental overhead of making models safe and fair.


---

**[122. [2502.20747] Measuring Determinism in Large Language Models for Software Code Review](https://arxiv.org/pdf/2502.20747.pdf)** (2025-03-03)

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

**[123. [2504.09895] Learning from Reference Answers: Versatile Language Model Alignment
  without Binary Human Preference Data](https://arxiv.org/pdf/2504.09895.pdf)** (2025-04-15)

*Shuai Zhao, Linchao Zhu, Yi Yang*

  Large language models~(LLMs) are expected to be helpful, harmless, and
honest. In various alignment scenarios, such as general human preference,
safety, and confidence alignment, binary preference data collection and reward
modeling are resource-intensive but necessary for human preference
transferring. In this work, we explore using the similarity between sampled
generations and high-quality reference answers as an alternative reward
function for LLM alignment. Using similarity as a reward circumvents training
reward models, and collecting a single reference answer potentially costs less
time than constructing binary preference pairs when multiple candidates are
available. Specifically, we develop \textit{RefAlign}, a versatile
REINFORCE-style alignment algorithm, which is free of reference and reward
models. Instead, RefAlign utilizes BERTScore between sampled generations and
high-quality reference answers as the surrogate reward. Beyond general human
preference optimization, RefAlign can be readily extended to diverse scenarios,
such as safety and confidence alignment, by incorporating the similarity reward
with task-related objectives. In various scenarios, {RefAlign} demonstrates
comparable performance to previous alignment methods while offering high
efficiency.


---

**[124. [2412.06483] SafeWorld: Geo-Diverse Safety Alignment](https://arxiv.org/pdf/2412.06483.pdf)** (2024-12-10)

*Da Yin, Haoyi Qiu, Kung-Hsiang Huang, Kai-Wei Chang, Nanyun Peng*

  In the rapidly evolving field of Large Language Models (LLMs), ensuring
safety is a crucial and widely discussed topic. However, existing works often
overlook the geo-diversity of cultural and legal standards across the world. To
demonstrate the challenges posed by geo-diverse safety standards, we introduce
SafeWorld, a novel benchmark specifically designed to evaluate LLMs' ability to
generate responses that are not only helpful but also culturally sensitive and
legally compliant across diverse global contexts. SafeWorld encompasses 2,342
test user queries, each grounded in high-quality, human-verified cultural norms
and legal policies from 50 countries and 493 regions/races. On top of it, we
propose a multi-dimensional automatic safety evaluation framework that assesses
the contextual appropriateness, accuracy, and comprehensiveness of responses.
Our evaluations reveal that current LLMs struggle to meet these criteria. To
enhance LLMs' alignment with geo-diverse safety standards, we synthesize
helpful preference pairs for Direct Preference Optimization (DPO) alignment
training. The preference pair construction aims to encourage LLMs to behave
appropriately and provide precise references to relevant cultural norms and
policies when necessary. Our trained SafeWorldLM outperforms all competing
models, including GPT-4o on all three evaluation dimensions by a large margin.
Global human evaluators also note a nearly 20% higher winning rate in
helpfulness and harmfulness evaluation. Our code and data can be found here:
https://github.com/PlusLabNLP/SafeWorld.


---

**[125. [2406.14563] Model Merging and Safety Alignment: One Bad Model Spoils the Bunch](https://arxiv.org/pdf/2406.14563.pdf)** (2024-06-21)

*Hasan Abed Al Kader Hammoud, Umberto Michieli, Fabio Pizzati, Philip Torr, Adel Bibi, Bernard Ghanem, Mete Ozay*

  Merging Large Language Models (LLMs) is a cost-effective technique for
combining multiple expert LLMs into a single versatile model, retaining the
expertise of the original ones. However, current approaches often overlook the
importance of safety alignment during merging, leading to highly misaligned
models. This work investigates the effects of model merging on alignment. We
evaluate several popular model merging techniques, demonstrating that existing
methods do not only transfer domain expertise but also propagate misalignment.
We propose a simple two-step approach to address this problem: (i) generating
synthetic safety and domain-specific data, and (ii) incorporating these
generated data into the optimization process of existing data-aware model
merging techniques. This allows us to treat alignment as a skill that can be
maximized in the resulting merged LLM. Our experiments illustrate the
effectiveness of integrating alignment-related data during merging, resulting
in models that excel in both domain expertise and alignment.


---

**[126. [2502.11533] Be Cautious When Merging Unfamiliar LLMs: A Phishing Model Capable of
  Stealing Privacy](https://arxiv.org/pdf/2502.11533.pdf)** (2025-02-18)

*Zhenyuan Guo, Yi Shi, Wenlong Meng, Chen Gong, Chengkun Wei, Wenzhi Chen*

  Model merging is a widespread technology in large language models (LLMs) that
integrates multiple task-specific LLMs into a unified one, enabling the merged
model to inherit the specialized capabilities of these LLMs. Most task-specific
LLMs are sourced from open-source communities and have not undergone rigorous
auditing, potentially imposing risks in model merging. This paper highlights an
overlooked privacy risk: \textit{an unsafe model could compromise the privacy
of other LLMs involved in the model merging.} Specifically, we propose PhiMM, a
privacy attack approach that trains a phishing model capable of stealing
privacy using a crafted privacy phishing instruction dataset. Furthermore, we
introduce a novel model cloaking method that mimics a specialized capability to
conceal attack intent, luring users into merging the phishing model. Once
victims merge the phishing model, the attacker can extract personally
identifiable information (PII) or infer membership information (MI) by querying
the merged model with the phishing instruction. Experimental results show that
merging a phishing model increases the risk of privacy breaches. Compared to
the results before merging, PII leakage increased by 3.9\% and MI leakage
increased by 17.4\% on average. We release the code of PhiMM through a link.


---

**[127. [2501.16378] Internal Activation Revision: Safeguarding Vision Language Models
  Without Parameter Update](https://arxiv.org/pdf/2501.16378.pdf)** (2025-01-29)

*Qing Li, Jiahui Geng, Zongxiong Chen, Kun Song, Lei Ma, Fakhri Karray*

  Vision-language models (VLMs) demonstrate strong multimodal capabilities but
have been found to be more susceptible to generating harmful content compared
to their backbone large language models (LLMs). Our investigation reveals that
the integration of images significantly shifts the model's internal activations
during the forward pass, diverging from those triggered by textual input.
Moreover, the safety alignments of LLMs embedded within VLMs are not
sufficiently robust to handle the activations discrepancies, making the models
vulnerable to even the simplest jailbreaking attacks. To address this issue, we
propose an \textbf{internal activation revision} approach that efficiently
revises activations during generation, steering the model toward safer outputs.
Our framework incorporates revisions at both the layer and head levels,
offering control over the model's generation at varying levels of granularity.
In addition, we explore three strategies for constructing positive and negative
samples and two approaches for extracting revision vectors, resulting in
different variants of our method. Comprehensive experiments demonstrate that
the internal activation revision method significantly improves the safety of
widely used VLMs, reducing attack success rates by an average of 48.94\%,
34.34\%, 43.92\%, and 52.98\% on SafeBench, Safe-Unsafe, Unsafe, and
MM-SafetyBench, respectively, while minimally impacting model helpfulness.


---

**[128. [2409.14961] UELLM: A Unified and Efficient Approach for LLM Inference Serving](https://arxiv.org/pdf/2409.14961.pdf)** (2024-09-25)

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

**[129. [2405.08619] ALMol: Aligned Language-Molecule Translation LLMs through Offline
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

**[130. [2309.04213] UQ at #SMM4H 2023: ALEX for Public Health Analysis with Social Media](https://arxiv.org/pdf/2309.04213.pdf)** (2023-09-13)

*Yan Jiang, Ruihong Qiu, Yi Zhang, Zi Huang*

  As social media becomes increasingly popular, more and more activities
related to public health emerge. Current techniques for public health analysis
involve popular models such as BERT and large language models (LLMs). However,
the costs of training in-domain LLMs for public health are especially
expensive. Furthermore, such kinds of in-domain datasets from social media are
generally imbalanced. To tackle these challenges, the data imbalance issue can
be overcome by data augmentation and balanced training. Moreover, the ability
of the LLMs can be effectively utilized by prompting the model properly. In
this paper, a novel ALEX framework is proposed to improve the performance of
public health analysis on social media by adopting an LLMs explanation
mechanism. Results show that our ALEX model got the best performance among all
submissions in both Task 2 and Task 4 with a high score in Task 1 in Social
Media Mining for Health 2023 (SMM4H)[1]. Our code has been released at https://
github.com/YanJiangJerry/ALEX.


---

**[131. [2502.12970] Reasoning-to-Defend: Safety-Aware Reasoning Can Defend Large Language
  Models from Jailbreaking](https://arxiv.org/pdf/2502.12970.pdf)** (2025-02-19)

*Junda Zhu, Lingyong Yan, Shuaiqiang Wang, Dawei Yin, Lei Sha*

  The reasoning abilities of Large Language Models (LLMs) have demonstrated
remarkable advancement and exceptional performance across diverse domains.
However, leveraging these reasoning capabilities to enhance LLM safety against
adversarial attacks and jailbreak queries remains largely unexplored. To bridge
this gap, we propose Reasoning-to-Defend (R2D), a novel training paradigm that
integrates safety reflections of queries and responses into LLMs' generation
process, unlocking a safety-aware reasoning mechanism. This approach enables
self-evaluation at each reasoning step to create safety pivot tokens as
indicators of the response's safety status. Furthermore, in order to improve
the learning efficiency of pivot token prediction, we propose Contrastive Pivot
Optimization(CPO), which enhances the model's ability to perceive the safety
status of dialogues. Through this mechanism, LLMs dynamically adjust their
response strategies during reasoning, significantly enhancing their defense
capabilities against jailbreak attacks. Extensive experimental results
demonstrate that R2D effectively mitigates various attacks and improves overall
safety, highlighting the substantial potential of safety-aware reasoning in
strengthening LLMs' robustness against jailbreaks.


---

**[132. [2405.11040] From Generalist to Specialist: Improving Large Language Models for
  Medical Physics Using ARCoT](https://arxiv.org/pdf/2405.11040.pdf)** (2024-05-21)

*Jace Grandinetti, Rafe McBeth*

  Large Language Models (LLMs) have achieved remarkable progress, yet their
application in specialized fields, such as medical physics, remains challenging
due to the need for domain-specific knowledge. This study introduces ARCoT
(Adaptable Retrieval-based Chain of Thought), a framework designed to enhance
the domain-specific accuracy of LLMs without requiring fine-tuning or extensive
retraining. ARCoT integrates a retrieval mechanism to access relevant
domain-specific information and employs step-back and chain-of-thought
prompting techniques to guide the LLM's reasoning process, ensuring more
accurate and context-aware responses. Benchmarking on a medical physics
multiple-choice exam, our model outperformed standard LLMs and reported average
human performance, demonstrating improvements of up to 68% and achieving a high
score of 90%. This method reduces hallucinations and increases domain-specific
performance. The versatility and model-agnostic nature of ARCoT make it easily
adaptable to various domains, showcasing its significant potential for
enhancing the accuracy and reliability of LLMs in specialized fields.


---

**[133. [2405.16833] Safe LoRA: the Silver Lining of Reducing Safety Risks when Fine-tuning
  Large Language Models](https://arxiv.org/pdf/2405.16833.pdf)** (2025-01-07)

*Chia-Yi Hsu, Yu-Lin Tsai, Chih-Hsun Lin, Pin-Yu Chen, Chia-Mu Yu, Chun-Ying Huang*

  While large language models (LLMs) such as Llama-2 or GPT-4 have shown
impressive zero-shot performance, fine-tuning is still necessary to enhance
their performance for customized datasets, domain-specific tasks, or other
private needs. However, fine-tuning all parameters of LLMs requires significant
hardware resources, which can be impractical for typical users. Therefore,
parameter-efficient fine-tuning such as LoRA have emerged, allowing users to
fine-tune LLMs without the need for considerable computing resources, with
little performance degradation compared to fine-tuning all parameters.
Unfortunately, recent studies indicate that fine-tuning can increase the risk
to the safety of LLMs, even when data does not contain malicious content. To
address this challenge, we propose Safe LoRA, a simple one-liner patch to the
original LoRA implementation by introducing the projection of LoRA weights from
selected layers to the safety-aligned subspace, effectively reducing the safety
risks in LLM fine-tuning while maintaining utility. It is worth noting that
Safe LoRA is a training-free and data-free approach, as it only requires the
knowledge of the weights from the base and aligned LLMs. Our extensive
experiments demonstrate that when fine-tuning on purely malicious data, Safe
LoRA retains similar safety performance as the original aligned model.
Moreover, when the fine-tuning dataset contains a mixture of both benign and
malicious data, Safe LoRA mitigates the negative effect made by malicious data
while preserving performance on downstream tasks. Our codes are available at
\url{https://github.com/IBM/SafeLoRA}.


---

**[134. [2403.00826] LLMGuard: Guarding Against Unsafe LLM Behavior](https://arxiv.org/pdf/2403.00826.pdf)** (2024-03-05)

*Shubh Goyal, Medha Hira, Shubham Mishra, Sukriti Goyal, Arnav Goel, Niharika Dadu, Kirushikesh DB, Sameep Mehta, Nishtha Madaan*

  Although the rise of Large Language Models (LLMs) in enterprise settings
brings new opportunities and capabilities, it also brings challenges, such as
the risk of generating inappropriate, biased, or misleading content that
violates regulations and can have legal concerns. To alleviate this, we present
"LLMGuard", a tool that monitors user interactions with an LLM application and
flags content against specific behaviours or conversation topics. To do this
robustly, LLMGuard employs an ensemble of detectors.


---

**[135. [2410.13903] CoreGuard: Safeguarding Foundational Capabilities of LLMs Against Model
  Stealing in Edge Deployment](https://arxiv.org/pdf/2410.13903.pdf)** (2024-10-21)

*Qinfeng Li, Yangfan Xie, Tianyu Du, Zhiqiang Shen, Zhenghan Qin, Hao Peng, Xinkui Zhao, Xianwei Zhu, Jianwei Yin, Xuhong Zhang*

  Proprietary large language models (LLMs) demonstrate exceptional
generalization ability across various tasks. Additionally, deploying LLMs on
edge devices is trending for efficiency and privacy reasons. However, edge
deployment of proprietary LLMs introduces new security threats: attackers who
obtain an edge-deployed LLM can easily use it as a base model for various tasks
due to its high generalization ability, which we call foundational capability
stealing. Unfortunately, existing model protection mechanisms are often
task-specific and fail to protect general-purpose LLMs, as they mainly focus on
protecting task-related parameters using trusted execution environments (TEEs).
Although some recent TEE-based methods are able to protect the overall model
parameters in a computation-efficient way, they still suffer from prohibitive
communication costs between TEE and CPU/GPU, making it impractical to deploy
for edge LLMs. To protect the foundational capabilities of edge LLMs, we
propose CoreGuard, a computation- and communication-efficient model protection
approach against model stealing on edge devices. The core component of
CoreGuard is a lightweight and propagative authorization module residing in
TEE. Extensive experiments show that CoreGuard achieves the same security
protection as the black-box security guarantees with negligible overhead.


---

**[136. [2502.11090] SafeDialBench: A Fine-Grained Safety Benchmark for Large Language Models
  in Multi-Turn Dialogues with Diverse Jailbreak Attacks](https://arxiv.org/pdf/2502.11090.pdf)** (2025-02-19)

*Hongye Cao, Yanming Wang, Sijia Jing, Ziyue Peng, Zhixin Bai, Zhe Cao, Meng Fang, Fan Feng, Boyan Wang, Jiaheng Liu, Tianpei Yang, Jing Huo, Yang Gao, Fanyu Meng, Xi Yang, Chao Deng, Junlan Feng*

  With the rapid advancement of Large Language Models (LLMs), the safety of
LLMs has been a critical concern requiring precise assessment. Current
benchmarks primarily concentrate on single-turn dialogues or a single jailbreak
attack method to assess the safety. Additionally, these benchmarks have not
taken into account the LLM's capability of identifying and handling unsafe
information in detail. To address these issues, we propose a fine-grained
benchmark SafeDialBench for evaluating the safety of LLMs across various
jailbreak attacks in multi-turn dialogues. Specifically, we design a two-tier
hierarchical safety taxonomy that considers 6 safety dimensions and generates
more than 4000 multi-turn dialogues in both Chinese and English under 22
dialogue scenarios. We employ 7 jailbreak attack strategies, such as reference
attack and purpose reverse, to enhance the dataset quality for dialogue
generation. Notably, we construct an innovative assessment framework of LLMs,
measuring capabilities in detecting, and handling unsafe information and
maintaining consistency when facing jailbreak attacks. Experimental results
across 17 LLMs reveal that Yi-34B-Chat and GLM4-9B-Chat demonstrate superior
safety performance, while Llama3.1-8B-Instruct and o3-mini exhibit safety
vulnerabilities.


---

**[137. [2402.17887] JMLR: Joint Medical LLM and Retrieval Training for Enhancing Reasoning
  and Professional Question Answering Capability](https://arxiv.org/pdf/2402.17887.pdf)** (2024-07-01)

*Junda Wang, Zhichao Yang, Zonghai Yao, Hong Yu*

  Large Language Models (LLMs) have demonstrated a remarkable potential in
medical knowledge acquisition and question-answering. However, LLMs can
potentially hallucinate and yield factually incorrect outcomes, even with
domain-specific pretraining. Previously, retrieval augmented generation (RAG)
has limited success in addressing hallucinations. Unlike previous methods in
RAG where the retrieval model was trained separately from the LLM, we introduce
JMLR (for Jointly trains LLM and information Retrieval) during the fine-tuning
phase. The synchronized training mechanism enhances JMLR's ability to retrieve
clinical guidelines and leverage medical knowledge to reason and answer
questions and reduces the demand for computational resources. We evaluated JMLR
on the important medical question-answering application. Our experimental
results demonstrate that JMLR-13B (70.5%) outperforms a previous
state-of-the-art open-source model using conventional pre-training and
fine-tuning Meditron-70B (68.9%) and Llama2-13B with RAG (67.7%) on a medical
question-answering dataset. Comprehensive evaluations reveal JMLR-13B enhances
reasoning quality and reduces hallucinations better than Claude3-Opus.
Additionally, JMLR-13B (148 GPU hours) also trains much faster than
Meditron-70B (42630 GPU hours). Through this work, we provide a new and
efficient knowledge enhancement method for healthcare, demonstrating the
potential of integrating retrieval and LLM training for medical
question-answering systems.


---

**[138. [2406.05644] How Alignment and Jailbreak Work: Explain LLM Safety through
  Intermediate Hidden States](https://arxiv.org/pdf/2406.05644.pdf)** (2024-06-14)

*Zhenhong Zhou, Haiyang Yu, Xinghua Zhang, Rongwu Xu, Fei Huang, Yongbin Li*

  Large language models (LLMs) rely on safety alignment to avoid responding to
malicious user inputs. Unfortunately, jailbreak can circumvent safety
guardrails, resulting in LLMs generating harmful content and raising concerns
about LLM safety. Due to language models with intensive parameters often
regarded as black boxes, the mechanisms of alignment and jailbreak are
challenging to elucidate. In this paper, we employ weak classifiers to explain
LLM safety through the intermediate hidden states. We first confirm that LLMs
learn ethical concepts during pre-training rather than alignment and can
identify malicious and normal inputs in the early layers. Alignment actually
associates the early concepts with emotion guesses in the middle layers and
then refines them to the specific reject tokens for safe generations. Jailbreak
disturbs the transformation of early unethical classification into negative
emotions. We conduct experiments on models from 7B to 70B across various model
families to prove our conclusion. Overall, our paper indicates the intrinsical
mechanism of LLM safety and how jailbreaks circumvent safety guardrails,
offering a new perspective on LLM safety and reducing concerns. Our code is
available at https://github.com/ydyjya/LLM-IHS-Explanation.


---

**[139. [2411.14901] ReVisionLLM: Recursive Vision-Language Model for Temporal Grounding in
  Hour-Long Videos](https://arxiv.org/pdf/2411.14901.pdf)** (2024-11-25)

*Tanveer Hannan, Md Mohaiminul Islam, Jindong Gu, Thomas Seidl, Gedas Bertasius*

  Large language models (LLMs) excel at retrieving information from lengthy
text, but their vision-language counterparts (VLMs) face difficulties with
hour-long videos, especially for temporal grounding. Specifically, these VLMs
are constrained by frame limitations, often losing essential temporal details
needed for accurate event localization in extended video content. We propose
ReVisionLLM, a recursive vision-language model designed to locate events in
hour-long videos. Inspired by human search strategies, our model initially
targets broad segments of interest, progressively revising its focus to
pinpoint exact temporal boundaries. Our model can seamlessly handle videos of
vastly different lengths, from minutes to hours. We also introduce a
hierarchical training strategy that starts with short clips to capture distinct
events and progressively extends to longer videos. To our knowledge,
ReVisionLLM is the first VLM capable of temporal grounding in hour-long videos,
outperforming previous state-of-the-art methods across multiple datasets by a
significant margin (+2.6% R1@0.1 on MAD). The code is available at
https://github.com/Tanveer81/ReVisionLLM.


---

**[140. [2410.10862] Superficial Safety Alignment Hypothesis](https://arxiv.org/pdf/2410.10862.pdf)** (2024-10-16)

*Jianwei Li, Jung-Eun Kim*

  As large language models (LLMs) are overwhelmingly more and more integrated
into various applications, ensuring they generate safe and aligned responses is
a pressing need. Previous research on alignment has largely focused on general
instruction-following but has often overlooked the unique properties and
challenges of safety alignment, such as the brittleness of safety mechanisms.
To bridge the gap, we propose the Superficial Safety Alignment Hypothesis
(SSAH), which posits that safety alignment should teach an otherwise unsafe
model to choose the correct reasoning direction - interpreted as a specialized
binary classification task - and incorporate a refusal mechanism with multiple
reserved fallback options. Furthermore, through SSAH, we hypothesize that
safety guardrails in LLMs can be established by just a small number of
essential components. To verify this, we conduct an ablation study and
successfully identify four types of attribute-critical components in
safety-aligned LLMs: Exclusive Safety Unit (ESU), Exclusive Utility Unit (EUU),
Complex Unit (CU), and Redundant Unit (RU). Our findings show that freezing
certain safety-critical components 7.5\% during fine-tuning allows the model to
retain its safety attributes while adapting to new tasks. Additionally, we show
that leveraging redundant units 20\% in the pre-trained model as an ``alignment
budget'' can effectively minimize the alignment tax while achieving the
alignment goal. All considered, this paper concludes that the atomic functional
unit for safety in LLMs is at the neuron level and underscores that safety
alignment should not be complicated. We believe this work contributes to the
foundation of efficient and scalable safety alignment for future LLMs.


---

**[141. [2411.10954] Dialectal Toxicity Detection: Evaluating LLM-as-a-Judge Consistency
  Across Language Varieties](https://arxiv.org/pdf/2411.10954.pdf)** (2024-11-19)

*Fahim Faisal, Md Mushfiqur Rahman, Antonios Anastasopoulos*

  There has been little systematic study on how dialectal differences affect
toxicity detection by modern LLMs. Furthermore, although using LLMs as
evaluators ("LLM-as-a-judge") is a growing research area, their sensitivity to
dialectal nuances is still underexplored and requires more focused attention.
In this paper, we address these gaps through a comprehensive toxicity
evaluation of LLMs across diverse dialects. We create a multi-dialect dataset
through synthetic transformations and human-assisted translations, covering 10
language clusters and 60 varieties. We then evaluated three LLMs on their
ability to assess toxicity across multilingual, dialectal, and LLM-human
consistency. Our findings show that LLMs are sensitive in handling both
multilingual and dialectal variations. However, if we have to rank the
consistency, the weakest area is LLM-human agreement, followed by dialectal
consistency. Code repository:
\url{https://github.com/ffaisal93/dialect_toxicity_llm_judge}


---

**[142. [2412.11041] Separate the Wheat from the Chaff: A Post-Hoc Approach to Safety
  Re-Alignment for Fine-Tuned Language Models](https://arxiv.org/pdf/2412.11041.pdf)** (2025-02-18)

*Di Wu, Xin Lu, Yanyan Zhao, Bing Qin*

  Although large language models (LLMs) achieve effective safety alignment at
the time of release, they still face various safety challenges. A key issue is
that fine-tuning often compromises the safety alignment of LLMs. To address
this issue, we propose a method named IRR (Identify, Remove, and Recalibrate
for Safety Realignment) that performs safety realignment for LLMs. The core of
IRR is to identify and remove unsafe delta parameters from the fine-tuned
models, while recalibrating the retained ones. We evaluate the effectiveness of
IRR across various datasets, including both full fine-tuning and LoRA methods.
Our results demonstrate that IRR significantly enhances the safety performance
of fine-tuned models on safety benchmarks, such as harmful queries and
jailbreak attacks, while maintaining their performance on downstream tasks. The
source code is available at: https://anonymous.4open.science/r/IRR-BD4F.


---

**[143. [2309.14517] Watch Your Language: Investigating Content Moderation with Large
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

**[144. [2410.06494] Conformal Prediction: A Data Perspective](https://arxiv.org/pdf/2410.06494.pdf)** (2025-03-12)

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

**[145. [2408.10668] Probing the Safety Response Boundary of Large Language Models via Unsafe
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

**[146. [2408.15625] CBF-LLM: Safe Control for LLM Alignment](https://arxiv.org/pdf/2408.15625.pdf)** (2024-10-08)

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

**[147. [2303.03226] Safe Reinforcement Learning via Probabilistic Logic Shields](https://arxiv.org/pdf/2303.03226.pdf)** (2023-03-07)

*Wen-Chi Yang, Giuseppe Marra, Gavin Rens, Luc De Raedt*

  Safe Reinforcement learning (Safe RL) aims at learning optimal policies while
staying safe. A popular solution to Safe RL is shielding, which uses a logical
safety specification to prevent an RL agent from taking unsafe actions.
However, traditional shielding techniques are difficult to integrate with
continuous, end-to-end deep RL methods. To this end, we introduce Probabilistic
Logic Policy Gradient (PLPG). PLPG is a model-based Safe RL technique that uses
probabilistic logic programming to model logical safety constraints as
differentiable functions. Therefore, PLPG can be seamlessly applied to any
policy gradient algorithm while still providing the same convergence
guarantees. In our experiments, we show that PLPG learns safer and more
rewarding policies compared to other state-of-the-art shielding techniques.


---

**[148. [2405.09055] A safety realignment framework via subspace-oriented model fusion for
  large language models](https://arxiv.org/pdf/2405.09055.pdf)** (2024-05-16)

*Xin Yi, Shunfan Zheng, Linlin Wang, Xiaoling Wang, Liang He*

  The current safeguard mechanisms for large language models (LLMs) are indeed
susceptible to jailbreak attacks, making them inherently fragile. Even the
process of fine-tuning on apparently benign data for downstream tasks can
jeopardize safety. One potential solution is to conduct safety fine-tuning
subsequent to downstream fine-tuning. However, there's a risk of catastrophic
forgetting during safety fine-tuning, where LLMs may regain safety measures but
lose the task-specific knowledge acquired during downstream fine-tuning. In
this paper, we introduce a safety realignment framework through
subspace-oriented model fusion (SOMF), aiming to combine the safeguard
capabilities of initially aligned model and the current fine-tuned model into a
realigned model. Our approach begins by disentangling all task vectors from the
weights of each fine-tuned model. We then identify safety-related regions
within these vectors by subspace masking techniques. Finally, we explore the
fusion of the initial safely aligned LLM with all task vectors based on the
identified safety subspace. We validate that our safety realignment framework
satisfies the safety requirements of a single fine-tuned model as well as
multiple models during their fusion. Our findings confirm that SOMF preserves
safety without notably compromising performance on downstream tasks, including
instruction following in Chinese, English, and Hindi, as well as
problem-solving capabilities in Code and Math.


---

**[149. [2401.11206] InferAligner: Inference-Time Alignment for Harmlessness through
  Cross-Model Guidance](https://arxiv.org/pdf/2401.11206.pdf)** (2024-01-23)

*Pengyu Wang, Dong Zhang, Linyang Li, Chenkun Tan, Xinghao Wang, Ke Ren, Botian Jiang, Xipeng Qiu*

  With the rapid development of large language models (LLMs), they are not only
used as general-purpose AI assistants but are also customized through further
fine-tuning to meet the requirements of different applications. A pivotal
factor in the success of current LLMs is the alignment process. Current
alignment methods, such as supervised fine-tuning (SFT) and reinforcement
learning from human feedback (RLHF), focus on training-time alignment and are
often complex and cumbersome to implement. Therefore, we develop
\textbf{InferAligner}, a novel inference-time alignment method that utilizes
cross-model guidance for harmlessness alignment. InferAligner utilizes safety
steering vectors extracted from safety-aligned model to modify the activations
of the target model when responding to harmful inputs, thereby guiding the
target model to provide harmless responses. Experimental results show that our
method can be very effectively applied to domain-specific models in finance,
medicine, and mathematics, as well as to multimodal large language models
(MLLMs) such as LLaVA. It significantly diminishes the Attack Success Rate
(ASR) of both harmful instructions and jailbreak attacks, while maintaining
almost unchanged performance in downstream tasks.


---

**[150. [2408.09819] CMoralEval: A Moral Evaluation Benchmark for Chinese Large Language
  Models](https://arxiv.org/pdf/2408.09819.pdf)** (2024-08-20)

*Linhao Yu, Yongqi Leng, Yufei Huang, Shang Wu, Haixin Liu, Xinmeng Ji, Jiahui Zhao, Jinwang Song, Tingting Cui, Xiaoqing Cheng, Tao Liu, Deyi Xiong*

  What a large language model (LLM) would respond in ethically relevant
context? In this paper, we curate a large benchmark CMoralEval for morality
evaluation of Chinese LLMs. The data sources of CMoralEval are two-fold: 1) a
Chinese TV program discussing Chinese moral norms with stories from the society
and 2) a collection of Chinese moral anomies from various newspapers and
academic papers on morality. With these sources, we aim to create a moral
evaluation dataset characterized by diversity and authenticity. We develop a
morality taxonomy and a set of fundamental moral principles that are not only
rooted in traditional Chinese culture but also consistent with contemporary
societal norms. To facilitate efficient construction and annotation of
instances in CMoralEval, we establish a platform with AI-assisted instance
generation to streamline the annotation process. These help us curate
CMoralEval that encompasses both explicit moral scenarios (14,964 instances)
and moral dilemma scenarios (15,424 instances), each with instances from
different data sources. We conduct extensive experiments with CMoralEval to
examine a variety of Chinese LLMs. Experiment results demonstrate that
CMoralEval is a challenging benchmark for Chinese LLMs. The dataset is publicly
available at \url{https://github.com/tjunlp-lab/CMoralEval}.


---

**[151. [2412.06843] Semantic Loss Guided Data Efficient Supervised Fine Tuning for Safe
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

**[152. [2401.00287] The Art of Defending: A Systematic Evaluation and Analysis of LLM
  Defense Strategies on Safety and Over-Defensiveness](https://arxiv.org/pdf/2401.00287.pdf)** (2024-01-02)

*Neeraj Varshney, Pavel Dolin, Agastya Seth, Chitta Baral*

  As Large Language Models (LLMs) play an increasingly pivotal role in natural
language processing applications, their safety concerns become critical areas
of NLP research. This paper presents Safety and Over-Defensiveness Evaluation
(SODE) benchmark: a collection of diverse safe and unsafe prompts with
carefully designed evaluation methods that facilitate systematic evaluation,
comparison, and analysis over 'safety' and 'over-defensiveness.' With SODE, we
study a variety of LLM defense strategies over multiple state-of-the-art LLMs,
which reveals several interesting and important findings, such as (a) the
widely popular 'self-checking' techniques indeed improve the safety against
unsafe inputs, but this comes at the cost of extreme over-defensiveness on the
safe inputs, (b) providing a safety instruction along with in-context exemplars
(of both safe and unsafe inputs) consistently improves safety and also
mitigates undue over-defensiveness of the models, (c) providing contextual
knowledge easily breaks the safety guardrails and makes the models more
vulnerable to generating unsafe responses. Overall, our work reveals numerous
such critical findings that we believe will pave the way and facilitate further
research in improving the safety of LLMs.


---

**[153. [2502.15543] PIP-KAG: Mitigating Knowledge Conflicts in Knowledge-Augmented
  Generation via Parametric Pruning](https://arxiv.org/pdf/2502.15543.pdf)** (2025-02-24)

*Pengcheng Huang, Zhenghao Liu, Yukun Yan, Xiaoyuan Yi, Hao Chen, Zhiyuan Liu, Maosong Sun, Tong Xiao, Ge Yu, Chenyan Xiong*

  Knowledge-Augmented Generation (KAG) has shown great promise in updating the
internal memory of Large Language Models (LLMs) by integrating external
knowledge. However, KAG inevitably faces knowledge conflicts when the internal
memory contradicts external information. Current approaches to mitigating these
conflicts mainly focus on improving external knowledge utilization. However,
these methods have shown only limited effectiveness in mitigating the knowledge
conflict problem, as internal knowledge continues to influence the generation
process of LLMs. In this paper, we propose a ParametrIc Pruning-based
Knowledge-Augmented Generation (PIP-KAG) approach, which prunes internal
knowledge of LLMs and incorporates a plug-and-play adaptation module to help
LLMs better leverage external sources. Additionally, we construct the
CoConflictQA benchmark based on the hallucination of LLMs to better evaluate
contextual faithfulness during answering questions. Experimental results on
CoConflictQA demonstrate that PIP-KAG significantly reduces knowledge conflicts
and improves context fidelity. Notably, PIP-KAG reduces LLM's parameters by
13%, enhancing parameter efficiency in LLMs within the KAG framework. All codes
are available at https://github.com/OpenBMB/PIP-KAG.


---

**[154. [2501.14105] MedSlice: Fine-Tuned Large Language Models for Secure Clinical Note
  Sectioning](https://arxiv.org/pdf/2501.14105.pdf)** (2025-01-27)

*Joshua Davis, Thomas Sounack, Kate Sciacca, Jessie M Brain, Brigitte N Durieux, Nicole D Agaronnik, Charlotta Lindvall*

  Extracting sections from clinical notes is crucial for downstream analysis
but is challenging due to variability in formatting and labor-intensive nature
of manual sectioning. While proprietary large language models (LLMs) have shown
promise, privacy concerns limit their accessibility. This study develops a
pipeline for automated note sectioning using open-source LLMs, focusing on
three sections: History of Present Illness, Interval History, and Assessment
and Plan. We fine-tuned three open-source LLMs to extract sections using a
curated dataset of 487 progress notes, comparing results relative to
proprietary models (GPT-4o, GPT-4o mini). Internal and external validity were
assessed via precision, recall and F1 score. Fine-tuned Llama 3.1 8B
outperformed GPT-4o (F1=0.92). On the external validity test set, performance
remained high (F1= 0.85). Fine-tuned open-source LLMs can surpass proprietary
models in clinical note sectioning, offering advantages in cost, performance,
and accessibility.


---

**[155. [2502.11242] LLMs and Childhood Safety: Identifying Risks and Proposing a Protection
  Framework for Safe Child-LLM Interaction](https://arxiv.org/pdf/2502.11242.pdf)** (2025-02-24)

*Junfeng Jiao, Saleh Afroogh, Kevin Chen, Abhejay Murali, David Atkinson, Amit Dhurandhar*

  This study examines the growing use of Large Language Models (LLMs) in
child-centered applications, highlighting safety and ethical concerns such as
bias, harmful content, and cultural insensitivity. Despite their potential to
enhance learning, there is a lack of standardized frameworks to mitigate these
risks. Through a systematic literature review, we identify key parental and
empirical concerns, including toxicity and ethical breaches in AI outputs.
Moreover, to address these issues, this paper proposes a protection framework
for safe Child-LLM interaction, incorporating metrics for content safety,
behavioral ethics, and cultural sensitivity. The framework provides practical
tools for evaluating LLM safety, offering guidance for developers,
policymakers, and educators to ensure responsible AI deployment for children.


---

**[156. [2502.13946] Why Safeguarded Ships Run Aground? Aligned Large Language Models' Safety
  Mechanisms Tend to Be Anchored in The Template Region](https://arxiv.org/pdf/2502.13946.pdf)** (2025-02-20)

*Chak Tou Leong, Qingyu Yin, Jian Wang, Wenjie Li*

  The safety alignment of large language models (LLMs) remains vulnerable, as
their initial behavior can be easily jailbroken by even relatively simple
attacks. Since infilling a fixed template between the input instruction and
initial model output is a common practice for existing LLMs, we hypothesize
that this template is a key factor behind their vulnerabilities: LLMs'
safety-related decision-making overly relies on the aggregated information from
the template region, which largely influences these models' safety behavior. We
refer to this issue as template-anchored safety alignment. In this paper, we
conduct extensive experiments and verify that template-anchored safety
alignment is widespread across various aligned LLMs. Our mechanistic analyses
demonstrate how it leads to models' susceptibility when encountering
inference-time jailbreak attacks. Furthermore, we show that detaching safety
mechanisms from the template region is promising in mitigating vulnerabilities
to jailbreak attacks. We encourage future research to develop more robust
safety alignment techniques that reduce reliance on the template region.


---

**[157. [2407.08931] Global-Local Collaborative Inference with LLM for Lidar-Based
  Open-Vocabulary Detection](https://arxiv.org/pdf/2407.08931.pdf)** (2024-07-15)

*Xingyu Peng, Yan Bai, Chen Gao, Lirong Yang, Fei Xia, Beipeng Mu, Xiaofei Wang, Si Liu*

  Open-Vocabulary Detection (OVD) is the task of detecting all interesting
objects in a given scene without predefined object classes. Extensive work has
been done to deal with the OVD for 2D RGB images, but the exploration of 3D OVD
is still limited. Intuitively, lidar point clouds provide 3D information, both
object level and scene level, to generate trustful detection results. However,
previous lidar-based OVD methods only focus on the usage of object-level
features, ignoring the essence of scene-level information. In this paper, we
propose a Global-Local Collaborative Scheme (GLIS) for the lidar-based OVD
task, which contains a local branch to generate object-level detection result
and a global branch to obtain scene-level global feature. With the global-local
information, a Large Language Model (LLM) is applied for chain-of-thought
inference, and the detection result can be refined accordingly. We further
propose Reflected Pseudo Labels Generation (RPLG) to generate high-quality
pseudo labels for supervision and Background-Aware Object Localization (BAOL)
to select precise object proposals. Extensive experiments on ScanNetV2 and SUN
RGB-D demonstrate the superiority of our methods. Code is released at
https://github.com/GradiusTwinbee/GLIS.


---

**[158. [2410.12662] Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models](https://arxiv.org/pdf/2410.12662.pdf)** (2025-03-03)

*Shicheng Xu, Liang Pang, Yunchang Zhu, Huawei Shen, Xueqi Cheng*

  Vision-language alignment in Large Vision-Language Models (LVLMs)
successfully enables LLMs to understand visual input. However, we find that
existing vision-language alignment methods fail to transfer the existing safety
mechanism for text in LLMs to vision, which leads to vulnerabilities in toxic
image. To explore the cause of this problem, we give the insightful explanation
of where and how the safety mechanism of LVLMs operates and conduct comparative
analysis between text and vision. We find that the hidden states at the
specific transformer layers play a crucial role in the successful activation of
safety mechanism, while the vision-language alignment at hidden states level in
current methods is insufficient. This results in a semantic shift for input
images compared to text in hidden states, therefore misleads the safety
mechanism. To address this, we propose a novel Text-Guided vision-language
Alignment method (TGA) for LVLMs. TGA retrieves the texts related to input
vision and uses them to guide the projection of vision into the hidden states
space in LLMs. Experiments show that TGA not only successfully transfers the
safety mechanism for text in basic LLMs to vision in vision-language alignment
for LVLMs without any safety fine-tuning on the visual modality but also
maintains the general performance on various vision tasks (Safe and Good).


---

**[159. [2411.14502] Global Challenge for Safe and Secure LLMs Track 1](https://arxiv.org/pdf/2411.14502.pdf)** (2024-11-25)

*Xiaojun Jia, Yihao Huang, Yang Liu, Peng Yan Tan, Weng Kuan Yau, Mun-Thye Mak, Xin Ming Sim, Wee Siong Ng, See Kiong Ng, Hanqing Liu, Lifeng Zhou, Huanqian Yan, Xiaobing Sun, Wei Liu, Long Wang, Yiming Qian, Yong Liu, Junxiao Yang, Zhexin Zhang, Leqi Lei, Renmiao Chen, Yida Lu, Shiyao Cui, Zizhou Wang, Shaohua Li, Yan Wang, Rick Siow Mong Goh, Liangli Zhen, Yingjie Zhang, Zhe Zhao*

  This paper introduces the Global Challenge for Safe and Secure Large Language
Models (LLMs), a pioneering initiative organized by AI Singapore (AISG) and the
CyberSG R&D Programme Office (CRPO) to foster the development of advanced
defense mechanisms against automated jailbreaking attacks. With the increasing
integration of LLMs in critical sectors such as healthcare, finance, and public
administration, ensuring these models are resilient to adversarial attacks is
vital for preventing misuse and upholding ethical standards. This competition
focused on two distinct tracks designed to evaluate and enhance the robustness
of LLM security frameworks. Track 1 tasked participants with developing
automated methods to probe LLM vulnerabilities by eliciting undesirable
responses, effectively testing the limits of existing safety protocols within
LLMs. Participants were challenged to devise techniques that could bypass
content safeguards across a diverse array of scenarios, from offensive language
to misinformation and illegal activities. Through this process, Track 1 aimed
to deepen the understanding of LLM vulnerabilities and provide insights for
creating more resilient models.


---

**[160. [2406.10847] TorchOpera: A Compound AI System for LLM Safety](https://arxiv.org/pdf/2406.10847.pdf)** (2024-10-29)

*Shanshan Han, Zijian Hu, Alay Dilipbhai Shah, Han Jin, Yuhang Yao, Dimitris Stripelis, Zhaozhuo Xu, Chaoyang He*

  We introduce TorchOpera, a compound AI system for enhancing the safety and
quality of prompts and responses for Large Language Models. TorchOpera ensures
that all user prompts are safe, contextually grounded, and effectively
processed, while enhancing LLM responses to be relevant and high quality.
TorchOpera utilizes the vector database for contextual grounding, rule-based
wrappers for flexible modifications, and specialized mechanisms for detecting
and adjusting unsafe or incorrect content. We also provide a view of the
compound AI system to reduce the computational cost. Extensive experiments show
that TorchOpera ensures the safety, reliability, and applicability of LLMs in
real-world settings while maintaining the efficiency of LLM responses.


---

**[161. [2503.17395] CP-NCBF: A Conformal Prediction-based Approach to Synthesize Verified
  Neural Control Barrier Functions](https://arxiv.org/pdf/2503.17395.pdf)** (2025-03-25)

*Manan Tayal, Aditya Singh, Pushpak Jagtap, Shishir Kolathaya*

  Control Barrier Functions (CBFs) are a practical approach for designing
safety-critical controllers, but constructing them for arbitrary nonlinear
dynamical systems remains a challenge. Recent efforts have explored
learning-based methods, such as neural CBFs (NCBFs), to address this issue.
However, ensuring the validity of NCBFs is difficult due to potential learning
errors. In this letter, we propose a novel framework that leverages
split-conformal prediction to generate formally verified neural CBFs with
probabilistic guarantees based on a user-defined error rate, referred to as
CP-NCBF. Unlike existing methods that impose Lipschitz constraints on neural
CBF-leading to scalability limitations and overly conservative safe sets--our
approach is sample-efficient, scalable, and results in less restrictive safety
regions. We validate our framework through case studies on obstacle avoidance
in autonomous driving and geo-fencing of aerial vehicles, demonstrating its
ability to generate larger and less conservative safe sets compared to
conventional techniques.


---
