**[1. [2308.10443] Using Large Language Models for Cybersecurity Capture-The-Flag
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

**[2. [2410.13903] CoreGuard: Safeguarding Foundational Capabilities of LLMs Against Model
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

**[3. [2303.09384] LLMSecEval: A Dataset of Natural Language Prompts for Security
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

**[4. [2502.11191] Primus: A Pioneering Collection of Open-Source Datasets for
  Cybersecurity LLM Training](https://arxiv.org/pdf/2502.11191.pdf)** (2025-02-18)

*Yao-Ching Yu, Tsun-Han Chiang, Cheng-Wei Tsai, Chien-Ming Huang, Wen-Kwang Tsao*

  Large Language Models (LLMs) have shown remarkable advancements in
specialized fields such as finance, law, and medicine. However, in
cybersecurity, we have noticed a lack of open-source datasets, with a
particular lack of high-quality cybersecurity pretraining corpora, even though
much research indicates that LLMs acquire their knowledge during pretraining.
To address this, we present a comprehensive suite of datasets covering all
major training stages, including pretraining, instruction fine-tuning, and
reasoning distillation with cybersecurity-specific self-reflection data.
Extensive ablation studies demonstrate their effectiveness on public
cybersecurity benchmarks. In particular, continual pre-training on our dataset
yields a 15.88% improvement in the aggregate score, while reasoning
distillation leads to a 10% gain in security certification (CISSP). We will
release all datasets and trained cybersecurity LLMs under the ODC-BY and MIT
licenses to encourage further research in the community. For access to all
datasets and model weights, please refer to
https://huggingface.co/collections/trendmicro-ailab/primus-67b1fd27052b802b4af9d243.


---

**[5. [2310.04535] LLM4DV: Using Large Language Models for Hardware Test Stimuli Generation](https://arxiv.org/pdf/2310.04535.pdf)** (2025-03-26)

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

**[6. [2503.21598] Prompt, Divide, and Conquer: Bypassing Large Language Model Safety
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

**[7. [2502.00406] ALU: Agentic LLM Unlearning](https://arxiv.org/pdf/2502.00406.pdf)** (2025-02-04)

*Debdeep Sanyal, Murari Mandal*

  Information removal or suppression in large language models (LLMs) is a
desired functionality, useful in AI regulation, legal compliance, safety, and
privacy. LLM unlearning methods aim to remove information on demand from LLMs.
Current LLM unlearning methods struggle to balance the unlearning efficacy and
utility due to the competing nature of these objectives. Keeping the unlearning
process computationally feasible without assuming access to the model weights
is an overlooked area. We present the first agentic LLM unlearning (ALU)
method, a multi-agent, retrain-free, model-agnostic approach to LLM unlearning
that achieves effective unlearning while preserving the utility. Our ALU
framework unlearns by involving multiple LLM agents, each designed for a
specific step in the unlearning process, without the need to update model
weights for any of the agents in the framework. Users can easily request any
set of unlearning instances in any sequence, and ALU seamlessly adapts in real
time. This is facilitated without requiring any changes in the underlying LLM
model. Through extensive experiments on established benchmarks (TOFU, WMDP,
WPU) and jailbreaking techniques (many shot, target masking, other languages),
we demonstrate that ALU consistently stands out as the most robust LLM
unlearning framework among current state-of-the-art methods while incurring a
low constant-time cost. We further highlight ALU's superior performance
compared to existing methods when evaluated at scale. Specifically, ALU is
assessed on up to 1000 unlearning targets, exceeding the evaluation scope of
all previously proposed LLM unlearning methods.


---

**[8. [2404.12038] Uncovering Safety Risks of Large Language Models through Concept
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

**[9. [2411.18948] RevPRAG: Revealing Poisoning Attacks in Retrieval-Augmented Generation
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

**[10. [2504.04151] STEP: Staged Parameter-Efficient Pre-training for Large Language Models](https://arxiv.org/pdf/2504.04151.pdf)** (2025-04-08)

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

**[11. [2404.11338] LLMs for Cyber Security: New Opportunities](https://arxiv.org/pdf/2404.11338.pdf)** (2024-04-18)

*Dinil Mon Divakaran, Sai Teja Peddinti*

  Large language models (LLMs) are a class of powerful and versatile models
that are beneficial to many industries. With the emergence of LLMs, we take a
fresh look at cyber security, specifically exploring and summarizing the
potential of LLMs in addressing challenging problems in the security and safety
domains.


---

**[12. [2312.04916] EE-LLM: Large-Scale Training and Inference of Early-Exit Large Language
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

**[13. [2310.10049] FATE-LLM: A Industrial Grade Federated Learning Framework for Large
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

**[14. [2410.21723] Fine-tuning Large Language Models for DGA and DNS Exfiltration Detection](https://arxiv.org/pdf/2410.21723.pdf)** (2024-11-08)

*Md Abu Sayed, Asif Rahman, Christopher Kiekintveld, Sebastian Garcia*

  Domain Generation Algorithms (DGAs) are malicious techniques used by malware
to dynamically generate seemingly random domain names for communication with
Command & Control (C&C) servers. Due to the fast and simple generation of DGA
domains, detection methods must be highly efficient and precise to be
effective. Large Language Models (LLMs) have demonstrated their proficiency in
real-time detection tasks, making them ideal candidates for detecting DGAs. Our
work validates the effectiveness of fine-tuned LLMs for detecting DGAs and DNS
exfiltration attacks. We developed LLM models and conducted comprehensive
evaluation using a diverse dataset comprising 59 distinct real-world DGA
malware families and normal domain data. Our LLM model significantly
outperformed traditional natural language processing techniques, especially in
detecting unknown DGAs. We also evaluated its performance on DNS exfiltration
datasets, demonstrating its effectiveness in enhancing cybersecurity measures.
To the best of our knowledge, this is the first work that empirically applies
LLMs for DGA and DNS exfiltration detection.


---

**[15. [2412.04947] C$^2$LEVA: Toward Comprehensive and Contamination-Free Language Model
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

**[16. [2311.06154] A Last-Level Defense for Application Integrity and Confidentiality](https://arxiv.org/pdf/2311.06154.pdf)** (2023-11-13)

*Gabriel P. Fernandez, Andrey Brito, Ardhi Putra Pratama Hartono, Muhammad Usama Sardar, Christof Fetzer*

  Our objective is to protect the integrity and confidentiality of applications
operating in untrusted environments. Trusted Execution Environments (TEEs) are
not a panacea. Hardware TEEs fail to protect applications against Sybil, Fork
and Rollback Attacks and, consequently, fail to preserve the consistency and
integrity of applications. We introduce a novel system, LLD, that enforces the
integrity and consistency of applications in a transparent and scalable
fashion. Our solution augments TEEs with instantiation control and rollback
protection. Instantiation control, enforced with TEE-supported leases,
mitigates Sybil/Fork Attacks without incurring the high costs of solving
crypto-puzzles. Our rollback detection mechanism does not need excessive
replication, nor does it sacrifice durability. We show that implementing these
functionalities in the LLD runtime automatically protects applications and
services such as a popular DBMS.


---

**[17. [2410.03653] Dorami: Privilege Separating Security Monitor on RISC-V TEEs](https://arxiv.org/pdf/2410.03653.pdf)** (2024-10-07)

*Mark Kuhne, Stavros Volos, Shweta Shinde*

  TEE implementations on RISC-V offer an enclave abstraction by introducing a
trusted component called the security monitor (SM). The SM performs critical
tasks such as isolating enclaves from each other as well as from the OS by
using privileged ISA instructions that enforce the physical memory protection.
However, the SM executes at the highest privilege layer on the platform
(machine-mode) along side firmware that is not only large in size but also
includes third-party vendor code specific to the platform. In this paper, we
present Dorami - a privilege separation approach that isolates the SM from the
firmware thus reducing the attack surface on TEEs. Dorami re-purposes existing
ISA features to enforce its isolation and achieves its goals without large
overheads.


---

**[18. [2402.11814] An Empirical Evaluation of LLMs for Solving Offensive Security
  Challenges](https://arxiv.org/pdf/2402.11814.pdf)** (2024-02-20)

*Minghao Shao, Boyuan Chen, Sofija Jancheska, Brendan Dolan-Gavitt, Siddharth Garg, Ramesh Karri, Muhammad Shafique*

  Capture The Flag (CTF) challenges are puzzles related to computer security
scenarios. With the advent of large language models (LLMs), more and more CTF
participants are using LLMs to understand and solve the challenges. However, so
far no work has evaluated the effectiveness of LLMs in solving CTF challenges
with a fully automated workflow. We develop two CTF-solving workflows,
human-in-the-loop (HITL) and fully-automated, to examine the LLMs' ability to
solve a selected set of CTF challenges, prompted with information about the
question. We collect human contestants' results on the same set of questions,
and find that LLMs achieve higher success rate than an average human
participant. This work provides a comprehensive evaluation of the capability of
LLMs in solving real world CTF challenges, from real competition to fully
automated workflow. Our results provide references for applying LLMs in
cybersecurity education and pave the way for systematic evaluation of offensive
cybersecurity capabilities in LLMs.


---

**[19. [2406.17663] LLM-ARC: Enhancing LLMs with an Automated Reasoning Critic](https://arxiv.org/pdf/2406.17663.pdf)** (2024-07-22)

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

**[20. [2502.18532] CuDIP: Enhancing Theorem Proving in LLMs via Curriculum Learning-based
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

**[21. [2502.13347] Craw4LLM: Efficient Web Crawling for LLM Pretraining](https://arxiv.org/pdf/2502.13347.pdf)** (2025-02-26)

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

**[22. [2401.10360] Excuse me, sir? Your language model is leaking (information)](https://arxiv.org/pdf/2401.10360.pdf)** (2024-11-19)

*Or Zamir*

  We introduce a cryptographic method to hide an arbitrary secret payload in
the response of a Large Language Model (LLM). A secret key is required to
extract the payload from the model's response, and without the key it is
provably impossible to distinguish between the responses of the original LLM
and the LLM that hides a payload. In particular, the quality of generated text
is not affected by the payload. Our approach extends a recent result of Christ,
Gunn and Zamir (2023) who introduced an undetectable watermarking scheme for
LLMs.


---

**[23. [2401.09796] A Fast, Performant, Secure Distributed Training Framework For Large
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

**[24. [2504.02883] SemEval-2025 Task 4: Unlearning sensitive content from Large Language
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

**[25. [2412.00383] Unified Parameter-Efficient Unlearning for LLMs](https://arxiv.org/pdf/2412.00383.pdf)** (2025-04-21)

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

**[26. [2501.08200] CWEval: Outcome-driven Evaluation on Functionality and Security of LLM
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

**[27. [2311.10733] Proceedings of the 3rd International Workshop on Mining and Learning in
  the Legal Domain (MLLD-23)](https://arxiv.org/pdf/2311.10733.pdf)** (2023-11-21)

*Masoud Makrehchi, Dell Zhang, Alina Petrova, John Armour*

  This is the Proceedings of the 3rd International Workshop on Mining and
Learning in the Legal Domain (MLLD-23) which took place in conjunction with the
32nd ACM International Conference on Information and Knowledge Management
(CIKM-2023) at the University of Birmingham, Birmingham, UK on Sunday 22nd
October 2023.


---

**[28. [2504.13774] DP2Unlearning: An Efficient and Guaranteed Unlearning Framework for LLMs](https://arxiv.org/pdf/2504.13774.pdf)** (2025-04-21)

*Tamim Al Mahmud, Najeeb Jebreel, Josep Domingo-Ferrer, David Sanchez*

  Large language models (LLMs) have recently revolutionized language processing
tasks but have also brought ethical and legal issues. LLMs have a tendency to
memorize potentially private or copyrighted information present in the training
data, which might then be delivered to end users at inference time. When this
happens, a naive solution is to retrain the model from scratch after excluding
the undesired data. Although this guarantees that the target data have been
forgotten, it is also prohibitively expensive for LLMs. Approximate unlearning
offers a more efficient alternative, as it consists of ex post modifications of
the trained model itself to prevent undesirable results, but it lacks
forgetting guarantees because it relies solely on empirical evidence. In this
work, we present DP2Unlearning, a novel LLM unlearning framework that offers
formal forgetting guarantees at a significantly lower cost than retraining from
scratch on the data to be retained. DP2Unlearning involves training LLMs on
textual data protected using {\epsilon}-differential privacy (DP), which later
enables efficient unlearning with the guarantees against disclosure associated
with the chosen {\epsilon}. Our experiments demonstrate that DP2Unlearning
achieves similar model performance post-unlearning, compared to an LLM
retraining from scratch on retained data -- the gold standard exact unlearning
-- but at approximately half the unlearning cost. In addition, with a
reasonable computational cost, it outperforms approximate unlearning methods at
both preserving the utility of the model post-unlearning and effectively
forgetting the targeted information.


---

**[29. [2409.19134] Confidential Prompting: Protecting User Prompts from Cloud LLM Providers](https://arxiv.org/pdf/2409.19134.pdf)** (2025-03-05)

*In Gim, Caihua Li, Lin Zhong*

  Our work tackles the challenge of securing user inputs in cloud-hosted large
language model (LLM) serving while ensuring model confidentiality, output
invariance, and compute efficiency. We introduce Secure Partitioned Decoding
(SPD), which uses confidential computing to confine user prompts to a trusted
execution environment (TEE), namely a confidential virtual machine (CVM), while
allowing service providers to generate tokens efficiently. We also introduce a
novel cryptographic method, Prompt Obfuscation (PO), to ensure robustness
against reconstruction attacks on SPD. We demonstrate our approach preserves
both prompt confidentiality and LLM serving efficiency. Our solution enables
privacy-preserving cloud LLM serving that handles sensitive prompts, such as
clinical records, financial data, and personal information.


---

**[30. [2502.19320] Shh, don't say that! Domain Certification in LLMs](https://arxiv.org/pdf/2502.19320.pdf)** (2025-03-10)

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

**[31. [2412.13879] Crabs: Consuming Resource via Auto-generation for LLM-DoS Attack under
  Black-box Settings](https://arxiv.org/pdf/2412.13879.pdf)** (2025-02-19)

*Yuanhe Zhang, Zhenhong Zhou, Wei Zhang, Xinyue Wang, Xiaojun Jia, Yang Liu, Sen Su*

  Large Language Models (LLMs) have demonstrated remarkable performance across
diverse tasks yet still are vulnerable to external threats, particularly LLM
Denial-of-Service (LLM-DoS) attacks. Specifically, LLM-DoS attacks aim to
exhaust computational resources and block services. However, existing studies
predominantly focus on white-box attacks, leaving black-box scenarios
underexplored. In this paper, we introduce Auto-Generation for LLM-DoS
(AutoDoS) attack, an automated algorithm designed for black-box LLMs. AutoDoS
constructs the DoS Attack Tree and expands the node coverage to achieve
effectiveness under black-box conditions. By transferability-driven iterative
optimization, AutoDoS could work across different models in one prompt.
Furthermore, we reveal that embedding the Length Trojan allows AutoDoS to
bypass existing defenses more effectively. Experimental results show that
AutoDoS significantly amplifies service response latency by over
250$\times\uparrow$, leading to severe resource consumption in terms of GPU
utilization and memory usage. Our work provides a new perspective on LLM-DoS
attacks and security defenses. Our code is available at
https://github.com/shuita2333/AutoDoS.


---

**[32. [2405.04355] SmmPack: Obfuscation for SMM Modules with TPM Sealed Key](https://arxiv.org/pdf/2405.04355.pdf)** (2024-05-10)

*Kazuki Matsuo, Satoshi Tanda, Kuniyasu Suzaki, Yuhei Kawakoya, Tatsuya Mori*

  System Management Mode (SMM) is the highest-privileged operating mode of x86
and x86-64 processors. Through SMM exploitation, attackers can tamper with the
Unified Extensible Firmware Interface (UEFI) firmware, disabling the security
mechanisms implemented by the operating system and hypervisor. Vulnerabilities
enabling SMM code execution are often reported as Common Vulnerabilities and
Exposures (CVEs); however, no security mechanisms currently exist to prevent
attackers from analyzing those vulnerabilities. To increase the cost of
vulnerability analysis of SMM modules, we introduced SmmPack. The core concept
of SmmPack involves encrypting an SMM module with the key securely stored in a
Trusted Platform Module (TPM). We assessed the effectiveness of SmmPack in
preventing attackers from obtaining and analyzing SMM modules using various
acquisition methods. Our results show that SmmPack significantly increases the
cost by narrowing down the means of module acquisition. Furthermore, we
demonstrated that SmmPack operates without compromising the performance of the
original SMM modules. We also clarified the management and adoption methods of
SmmPack, as well as the procedure for applying BIOS updates, and demonstrated
that the implementation of SmmPack is realistic.


---

**[33. [2308.08090] Separate the Wheat from the Chaff: Model Deficiency Unlearning via
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

**[34. [2410.13640] Latent Space Chain-of-Embedding Enables Output-free LLM Self-Evaluation](https://arxiv.org/pdf/2410.13640.pdf)** (2025-03-14)

*Yiming Wang, Pei Zhang, Baosong Yang, Derek F. Wong, Rui Wang*

  LLM self-evaluation relies on the LLM's own ability to estimate response
correctness, which can greatly improve its deployment reliability. In this
research track, we propose the Chain-of-Embedding (CoE) in the latent space to
enable LLMs to perform output-free self-evaluation. CoE consists of all
progressive hidden states produced during the inference time, which can be
treated as the latent thinking path of LLMs. We find that when LLMs respond
correctly and incorrectly, their CoE features differ, these discrepancies
assist us in estimating LLM response correctness. Experiments in four diverse
domains and seven LLMs fully demonstrate the effectiveness of our method.
Meanwhile, its label-free design intent without any training and
millisecond-level computational cost ensures real-time feedback in large-scale
scenarios. More importantly, we provide interesting insights into LLM response
correctness from the perspective of hidden state changes inside LLMs.


---

**[35. [2503.23566] When LLM Therapists Become Salespeople: Evaluating Large Language Models
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

**[36. [2407.20999] MoFO: Momentum-Filtered Optimizer for Mitigating Forgetting in LLM
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

**[37. [2404.18239] SOUL: Unlocking the Power of Second-Order Optimization for LLM
  Unlearning](https://arxiv.org/pdf/2404.18239.pdf)** (2024-06-26)

*Jinghan Jia, Yihua Zhang, Yimeng Zhang, Jiancheng Liu, Bharat Runwal, James Diffenderfer, Bhavya Kailkhura, Sijia Liu*

  Large Language Models (LLMs) have highlighted the necessity of effective
unlearning mechanisms to comply with data regulations and ethical AI practices.
LLM unlearning aims at removing undesired data influences and associated model
capabilities without compromising utility beyond the scope of unlearning. While
interest in studying LLM unlearning is growing, the impact of the optimizer
choice for LLM unlearning remains unexplored. In this work, we shed light on
the significance of optimizer selection in LLM unlearning for the first time,
establishing a clear connection between second-order optimization and influence
unlearning (a classical approach using influence functions to update the model
for data influence removal). This insight propels us to develop a second-order
optimization-based LLM unlearning framework, termed Second-Order UnLearning
(SOUL), which extends the static, one-shot model update using influence
unlearning to a dynamic, iterative unlearning process. Our extensive
experiments show that SOUL consistently outperforms conventional first-order
methods across various unlearning tasks, models, and metrics, indicating that
second-order optimization offers an effective and broadly applicable solution
for LLM unlearning. Codes are available at https://github.com/OPTML-Group/SOUL.


---

**[38. [2502.17898] VeriPlan: Integrating Formal Verification and LLMs into End-User
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

**[39. [2501.11771] Characterization of GPU TEE Overheads in Distributed Data Parallel ML
  Training](https://arxiv.org/pdf/2501.11771.pdf)** (2025-03-31)

*Jonghyun Lee, Yongqin Wang, Rachit Rajat, Murali Annavaram*

  Confidential computing (CC) or trusted execution enclaves (TEEs) is now the
most common approach to enable secure computing in the cloud. The recent
introduction of GPU TEEs by NVIDIA enables machine learning (ML) models to be
trained without leaking model weights or data to the cloud provider. However,
the potential performance implications of using GPU TEEs for ML training are
not well characterized. In this work, we present an in-depth characterization
study on performance overhead associated with running distributed data parallel
(DDP) ML training with GPU Trusted Execution Environments (TEE).
  Our study reveals the performance challenges in DDP training within GPU TEEs.
DDP uses ring-all-reduce, a well-known approach, to aggregate gradients from
multiple devices. Ring all-reduce consists of multiple scatter-reduce and
all-gather operations. In GPU TEEs only the GPU package (GPU and HBM memory) is
trusted. Hence, any data communicated outside the GPU packages must be
encrypted and authenticated for confidentiality and integrity verification.
Hence, each phase of the ring-all-reduce requires encryption and message
authentication code (MAC) generation from the sender, and decryption and MAC
authentication on the receiver. As the number of GPUs participating in DDP
increases, the overhead of secure inter-GPU communication during
ring-all-reduce grows proportionally. Additionally, larger models lead to more
asynchronous all-reduce operations, exacerbating the communication cost. Our
results show that with four GPU TEEs, depending on the model that is being
trained, the runtime per training iteration increases by an average of 8x and
up to a maximum of 41.6x compared to DDP training without TEE.


---

**[40. [1801.00471] TWAM: A Certifying Abstract Machine for Logic Programs](https://arxiv.org/pdf/1801.00471.pdf)** (2022-06-29)

*Rose Bohrer, Karl Crary*

  Type-preserving (or typed) compilation uses typing derivations to certify
correctness properties of compilation. We have designed and implemented a
type-preserving compiler for a simply-typed dialect of Prolog we call T-Prolog.
The crux of our approach is a new certifying abstract machine which we call the
Typed Warren Abstract Machine (TWAM). The TWAM has a dependent type system
strong enough to specify the semantics of a logic program in the logical
framework LF. We present a soundness metatheorem which constitutes a partial
correctness guarantee: well-typed programs implement the logic program
specified by their type. This metatheorem justifies our design and
implementation of a certifying compiler from T-Prolog to TWAM.


---

**[41. [2310.09639] DPZero: Private Fine-Tuning of Language Models without Backpropagation](https://arxiv.org/pdf/2310.09639.pdf)** (2024-06-07)

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

**[42. [2407.06443] Exposing Privacy Gaps: Membership Inference Attack on Preference Data
  for LLM Alignment](https://arxiv.org/pdf/2407.06443.pdf)** (2024-07-10)

*Qizhang Feng, Siva Rajesh Kasa, Hyokun Yun, Choon Hui Teo, Sravan Babu Bodapati*

  Large Language Models (LLMs) have seen widespread adoption due to their
remarkable natural language capabilities. However, when deploying them in
real-world settings, it is important to align LLMs to generate texts according
to acceptable human standards. Methods such as Proximal Policy Optimization
(PPO) and Direct Preference Optimization (DPO) have made significant progress
in refining LLMs using human preference data. However, the privacy concerns
inherent in utilizing such preference data have yet to be adequately studied.
In this paper, we investigate the vulnerability of LLMs aligned using human
preference datasets to membership inference attacks (MIAs), highlighting the
shortcomings of previous MIA approaches with respect to preference data. Our
study has two main contributions: first, we introduce a novel reference-based
attack framework specifically for analyzing preference data called PREMIA
(\uline{Pre}ference data \uline{MIA}); second, we provide empirical evidence
that DPO models are more vulnerable to MIA compared to PPO models. Our findings
highlight gaps in current privacy-preserving practices for LLM alignment.


---

**[43. [2404.09932] Foundational Challenges in Assuring Alignment and Safety of Large
  Language Models](https://arxiv.org/pdf/2404.09932.pdf)** (2024-09-09)

*Usman Anwar, Abulhair Saparov, Javier Rando, Daniel Paleka, Miles Turpin, Peter Hase, Ekdeep Singh Lubana, Erik Jenner, Stephen Casper, Oliver Sourbut, Benjamin L. Edelman, Zhaowei Zhang, Mario Günther, Anton Korinek, Jose Hernandez-Orallo, Lewis Hammond, Eric Bigelow, Alexander Pan, Lauro Langosco, Tomasz Korbak, Heidi Zhang, Ruiqi Zhong, Seán Ó hÉigeartaigh, Gabriel Recchia, Giulio Corsi, Alan Chan, Markus Anderljung, Lilian Edwards, Aleksandar Petrov, Christian Schroeder de Witt, Sumeet Ramesh Motwan, Yoshua Bengio, Danqi Chen, Philip H. S. Torr, Samuel Albanie, Tegan Maharaj, Jakob Foerster, Florian Tramer, He He, Atoosa Kasirzadeh, Yejin Choi, David Krueger*

  This work identifies 18 foundational challenges in assuring the alignment and
safety of large language models (LLMs). These challenges are organized into
three different categories: scientific understanding of LLMs, development and
deployment methods, and sociotechnical challenges. Based on the identified
challenges, we pose $200+$ concrete research questions.


---

**[44. [2409.09288] Generating API Parameter Security Rules with LLM for API Misuse
  Detection](https://arxiv.org/pdf/2409.09288.pdf)** (2024-09-20)

*Jinghua Liu, Yi Yang, Kai Chen, Miaoqian Lin*

  In this paper, we present a new framework, named GPTAid, for automatic APSRs
generation by analyzing API source code with LLM and detecting API misuse
caused by incorrect parameter use. To validate the correctness of the
LLM-generated APSRs, we propose an execution feedback-checking approach based
on the observation that security-critical API misuse is often caused by APSRs
violations, and most of them result in runtime errors. Specifically, GPTAid
first uses LLM to generate raw APSRs and the Right calling code, and then
generates Violation code for each raw APSR by modifying the Right calling code
using LLM. Subsequently, GPTAid performs dynamic execution on each piece of
Violation code and further filters out the incorrect APSRs based on runtime
errors. To further generate concrete APSRs, GPTAid employs a code differential
analysis to refine the filtered ones. Particularly, as the programming language
is more precise than natural language, GPTAid identifies the key operations
within Violation code by differential analysis, and then generates the
corresponding concrete APSR based on the aforementioned operations. These
concrete APSRs could be precisely interpreted into applicable detection code,
which proven to be effective in API misuse detection. Implementing on the
dataset containing 200 randomly selected APIs from eight popular libraries,
GPTAid achieves a precision of 92.3%. Moreover, it generates 6 times more APSRs
than state-of-the-art detectors on a comparison dataset of previously reported
bugs and APSRs. We further evaluated GPTAid on 47 applications, 210 unknown
security bugs were found potentially resulting in severe security issues (e.g.,
system crashes), 150 of which have been confirmed by developers after our
reports.


---

**[45. [2412.00166] To Ensemble or Not: Assessing Majority Voting Strategies for Phishing
  Detection with Large Language Models](https://arxiv.org/pdf/2412.00166.pdf)** (2025-03-04)

*Fouad Trad, Ali Chehab*

  The effectiveness of Large Language Models (LLMs) significantly relies on the
quality of the prompts they receive. However, even when processing identical
prompts, LLMs can yield varying outcomes due to differences in their training
processes. To leverage the collective intelligence of multiple LLMs and enhance
their performance, this study investigates three majority voting strategies for
text classification, focusing on phishing URL detection. The strategies are:
(1) a prompt-based ensemble, which utilizes majority voting across the
responses generated by a single LLM to various prompts; (2) a model-based
ensemble, which entails aggregating responses from multiple LLMs to a single
prompt; and (3) a hybrid ensemble, which combines the two methods by sending
different prompts to multiple LLMs and then aggregating their responses. Our
analysis shows that ensemble strategies are most suited in cases where
individual components exhibit equivalent performance levels. However, when
there is a significant discrepancy in individual performance, the effectiveness
of the ensemble method may not exceed that of the highest-performing single LLM
or prompt. In such instances, opting for ensemble techniques is not
recommended.


---

**[46. [2502.20747] Measuring Determinism in Large Language Models for Software Code Review](https://arxiv.org/pdf/2502.20747.pdf)** (2025-03-03)

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

**[47. [2412.13670] AntiLeak-Bench: Preventing Data Contamination by Automatically
  Constructing Benchmarks with Updated Real-World Knowledge](https://arxiv.org/pdf/2412.13670.pdf)** (2024-12-19)

*Xiaobao Wu, Liangming Pan, Yuxi Xie, Ruiwen Zhou, Shuai Zhao, Yubo Ma, Mingzhe Du, Rui Mao, Anh Tuan Luu, William Yang Wang*

  Data contamination hinders fair LLM evaluation by introducing test data into
newer models' training sets. Existing studies solve this challenge by updating
benchmarks with newly collected data. However, they fail to guarantee
contamination-free evaluation as the newly collected data may contain
pre-existing knowledge, and their benchmark updates rely on intensive human
labor. To address these issues, we in this paper propose AntiLeak-Bench, an
automated anti-leakage benchmarking framework. Instead of simply using newly
collected data, we construct samples with explicitly new knowledge absent from
LLMs' training sets, which thus ensures strictly contamination-free evaluation.
We further design a fully automated workflow to build and update our benchmark
without human labor. This significantly reduces the cost of benchmark
maintenance to accommodate emerging LLMs. Through extensive experiments, we
highlight that data contamination likely exists before LLMs' cutoff time and
demonstrate AntiLeak-Bench effectively overcomes this challenge.


---

**[48. [2503.15547] Prompt Flow Integrity to Prevent Privilege Escalation in LLM Agents](https://arxiv.org/pdf/2503.15547.pdf)** (2025-03-21)

*Juhee Kim, Woohyuk Choi, Byoungyoung Lee*

  Large Language Models (LLMs) are combined with plugins to create powerful LLM
agents that provide a wide range of services. Unlike traditional software, LLM
agent's behavior is determined at runtime by natural language prompts from
either user or plugin's data. This flexibility enables a new computing paradigm
with unlimited capabilities and programmability, but also introduces new
security risks, vulnerable to privilege escalation attacks. Moreover, user
prompt is prone to be interpreted in an insecure way by LLM agents, creating
non-deterministic behaviors that can be exploited by attackers. To address
these security risks, we propose Prompt Flow Integrity (PFI), a system
security-oriented solution to prevent privilege escalation in LLM agents.
Analyzing the architectural characteristics of LLM agents, PFI features three
mitigation techniques -- i.e., untrusted data identification, enforcing least
privilege on LLM agents, and validating unsafe data flows. Our evaluation
result shows that PFI effectively mitigates privilege escalation attacks while
successfully preserving the utility of LLM agents.


---

**[49. [2403.18051] Supervisory Prompt Training](https://arxiv.org/pdf/2403.18051.pdf)** (2024-03-28)

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

**[50. [2410.04601] ProtoMed-LLM: An Automatic Evaluation Framework for Large Language
  Models in Medical Protocol Formulation](https://arxiv.org/pdf/2410.04601.pdf)** (2025-04-15)

*Seungjun Yi, Jaeyoung Lim, Juyong Yoon*

  Automated generation of scientific protocols executable by robots can
significantly accelerate scientific research processes. Large Language Models
(LLMs) excel at Scientific Protocol Formulation Tasks (SPFT), but the
evaluation of their capabilities rely on human evaluation. Here, we propose a
flexible, automatic framework to evaluate LLMs' capability on SPFT:
ProtoMed-LLM. This framework prompts the target model and GPT-4 to extract
pseudocode from biology protocols using only predefined lab actions and
evaluates the output of the target model using LLAM-EVAL, the pseudocode
generated by GPT-4 serving as a baseline and Llama-3 acting as the evaluator.
Our adaptable prompt-based evaluation method, LLAM-EVAL, offers significant
flexibility in terms of evaluation model, material, criteria, and is free of
cost. We evaluate GPT variations, Llama, Mixtral, Gemma, Cohere, and Gemini.
Overall, we find that GPT and Cohere are powerful scientific protocol
formulators. We also introduce BIOPROT 2.0, a dataset with biology protocols
and corresponding pseudocodes, which can aid LLMs in formulation and evaluation
of SPFT. Our work is extensible to assess LLMs on SPFT across various domains
and other fields that require protocol generation for specific goals.


---

**[51. [2403.09972] Think Twice Before Trusting: Self-Detection for Large Language Models
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

**[52. [2412.17846] Enhancing Knowledge Distillation for LLMs with Response-Priming
  Prompting](https://arxiv.org/pdf/2412.17846.pdf)** (2024-12-25)

*Vijay Goyal, Mustafa Khan, Aprameya Tirupati, Harveer Saini, Michael Lam, Kevin Zhu*

  Large language models (LLMs) have demonstrated remarkable performance across
a wide range of natural language processing (NLP) tasks. However, these models
are often difficult to deploy due to significant computational requirements and
resource constraints. Knowledge distillation (KD) is an effective technique for
transferring the performance of larger LLMs to smaller models. Traditional KD
methods primarily focus on the direct output of the teacher model, with little
emphasis on the role of prompting during knowledge transfer. In this paper, we
propose a set of novel response-priming prompting strategies applied in the
knowledge distillation pipeline to enhance the performance of student models.
Our approach fine-tunes a smaller Llama 3.1 8B Instruct model by distilling
knowledge from a quantized Llama 3.1 405B Instruct teacher model. We apply LoRA
optimization and evaluate on the GSM8K benchmark. Experimental results
demonstrate that integrating reasoning-eliciting prompting into the proposed KD
pipeline significantly improves student model performance, offering an
efficient way to deploy powerful models in resource-constrained environments.
We find that Ground Truth prompting results in a 55\% performance increase on
GSM8K for a distilled Llama 3.1 8B Instruct compared to the same model
distilled without prompting. A thorough investigation into the self-attention
layers of the student models indicates that the more successful prompted models
tend to exhibit certain positive behaviors inside their attention heads which
can be tied to their increased accuracy. Our implementation can be found at
https://github.com/alonso130r/knowledge-distillation.


---

**[53. [2410.08431] oRetrieval Augmented Generation for 10 Large Language Models and its
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

**[54. [2408.10608] Promoting Equality in Large Language Models: Identifying and Mitigating
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

**[55. [2403.03883] SaulLM-7B: A pioneering Large Language Model for Law](https://arxiv.org/pdf/2403.03883.pdf)** (2024-03-08)

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

**[56. [2410.21337] Fine-tuned Large Language Models (LLMs): Improved Prompt Injection
  Attacks Detection](https://arxiv.org/pdf/2410.21337.pdf)** (2024-11-11)

*Md Abdur Rahman, Fan Wu, Alfredo Cuzzocrea, Sheikh Iqbal Ahamed*

  Large language models (LLMs) are becoming a popular tool as they have
significantly advanced in their capability to tackle a wide range of
language-based tasks. However, LLMs applications are highly vulnerable to
prompt injection attacks, which poses a critical problem. These attacks target
LLMs applications through using carefully designed input prompts to divert the
model from adhering to original instruction, thereby it could execute
unintended actions. These manipulations pose serious security threats which
potentially results in data leaks, biased outputs, or harmful responses. This
project explores the security vulnerabilities in relation to prompt injection
attacks. To detect whether a prompt is vulnerable or not, we follows two
approaches: 1) a pre-trained LLM, and 2) a fine-tuned LLM. Then, we conduct a
thorough analysis and comparison of the classification performance. Firstly, we
use pre-trained XLM-RoBERTa model to detect prompt injections using test
dataset without any fine-tuning and evaluate it by zero-shot classification.
Then, this proposed work will apply supervised fine-tuning to this pre-trained
LLM using a task-specific labeled dataset from deepset in huggingface, and this
fine-tuned model achieves impressive results with 99.13\% accuracy, 100\%
precision, 98.33\% recall and 99.15\% F1-score thorough rigorous
experimentation and evaluation. We observe that our approach is highly
efficient in detecting prompt injection attacks.


---

**[57. [2502.01083] Tool Unlearning for Tool-Augmented LLMs](https://arxiv.org/pdf/2502.01083.pdf)** (2025-02-04)

*Jiali Cheng, Hadi Amiri*

  Tool-augmented large language models (LLMs) are often trained on datasets of
query-response pairs, which embed the ability to use tools or APIs directly
into the parametric knowledge of LLMs. Tool-augmented LLMs need the ability to
forget learned tools due to security vulnerabilities, privacy regulations, or
tool deprecations. However, ``tool unlearning'' has not been investigated in
unlearning literature. We introduce this novel task, which requires addressing
distinct challenges compared to traditional unlearning: knowledge removal
rather than forgetting individual samples, the high cost of optimizing LLMs,
and the need for principled evaluation metrics. To bridge these gaps, we
propose ToolDelete, the first approach for unlearning tools from tool-augmented
LLMs. It implements three key properties to address the above challenges for
effective tool unlearning and introduces a new membership inference attack
(MIA) model for effective evaluation. Extensive experiments on multiple tool
learning datasets and tool-augmented LLMs show that ToolDelete effectively
unlearns randomly selected tools, while preserving the LLM's knowledge on
non-deleted tools and maintaining performance on general tasks.


---

**[58. [2411.03079] Utilizing Precise and Complete Code Context to Guide LLM in Automatic
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

**[59. [2410.02810] StateAct: Enhancing LLM Base Agents via Self-prompting and
  State-tracking](https://arxiv.org/pdf/2410.02810.pdf)** (2025-04-09)

*Nikolai Rozanov, Marek Rei*

  Large language models (LLMs) are increasingly used as autonomous agents,
tackling tasks from robotics to web navigation. Their performance depends on
the underlying base agent. Existing methods, however, struggle with
long-context reasoning and goal adherence. We introduce StateAct, a novel and
efficient base agent that enhances decision-making through (1) self-prompting,
which reinforces task goals at every step, and (2) chain-of-states, an
extension of chain-of-thought that tracks state information over time. StateAct
outperforms ReAct, the previous best base agent, by over 10% on Alfworld, 30%
on Textcraft, and 7% on Webshop across multiple frontier LLMs. We also
demonstrate that StateAct can be used as a drop-in replacement for ReAct with
advanced LLM agent methods such as test-time scaling, yielding an additional
12% gain on Textcraft. By improving efficiency and long-range reasoning without
requiring additional training or retrieval, StateAct provides a scalable
foundation for LLM agents. We open source our code to support further research
at https://github.com/ai-nikolai/stateact .


---

**[60. [2312.07910] PromptBench: A Unified Library for Evaluation of Large Language Models](https://arxiv.org/pdf/2312.07910.pdf)** (2024-08-21)

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

**[61. [2501.06211] FLAME: Financial Large-Language Model Assessment and Metrics Evaluation](https://arxiv.org/pdf/2501.06211.pdf)** (2025-01-14)

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

**[62. [2405.11002] Large Language Models in Wireless Application Design: In-Context
  Learning-enhanced Automatic Network Intrusion Detection](https://arxiv.org/pdf/2405.11002.pdf)** (2024-05-21)

*Han Zhang, Akram Bin Sediq, Ali Afana, Melike Erol-Kantarci*

  Large language models (LLMs), especially generative pre-trained transformers
(GPTs), have recently demonstrated outstanding ability in information
comprehension and problem-solving. This has motivated many studies in applying
LLMs to wireless communication networks. In this paper, we propose a
pre-trained LLM-empowered framework to perform fully automatic network
intrusion detection. Three in-context learning methods are designed and
compared to enhance the performance of LLMs. With experiments on a real network
intrusion detection dataset, in-context learning proves to be highly beneficial
in improving the task processing performance in a way that no further training
or fine-tuning of LLMs is required. We show that for GPT-4, testing accuracy
and F1-Score can be improved by 90%. Moreover, pre-trained LLMs demonstrate big
potential in performing wireless communication-related tasks. Specifically, the
proposed framework can reach an accuracy and F1-Score of over 95% on different
types of attacks with GPT-4 using only 10 in-context learning examples.


---

**[63. [2311.08369] How You Prompt Matters! Even Task-Oriented Constraints in Instructions
  Affect LLM-Generated Text Detection](https://arxiv.org/pdf/2311.08369.pdf)** (2024-10-02)

*Ryuto Koike, Masahiro Kaneko, Naoaki Okazaki*

  To combat the misuse of Large Language Models (LLMs), many recent studies
have presented LLM-generated-text detectors with promising performance. When
users instruct LLMs to generate texts, the instruction can include different
constraints depending on the user's need. However, most recent studies do not
cover such diverse instruction patterns when creating datasets for LLM
detection. In this paper, we reveal that even task-oriented constraints --
constraints that would naturally be included in an instruction and are not
related to detection-evasion -- cause existing powerful detectors to have a
large variance in detection performance. We focus on student essay writing as a
realistic domain and manually create task-oriented constraints based on several
factors for essay quality. Our experiments show that the standard deviation
(SD) of current detector performance on texts generated by an instruction with
such a constraint is significantly larger (up to an SD of 14.4 F1-score) than
that by generating texts multiple times or paraphrasing the instruction. We
also observe an overall trend where the constraints can make LLM detection more
challenging than without them. Finally, our analysis indicates that the high
instruction-following ability of LLMs fosters the large impact of such
constraints on detection performance.


---

**[64. [2502.11533] Be Cautious When Merging Unfamiliar LLMs: A Phishing Model Capable of
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

**[65. [2310.18333] She had Cobalt Blue Eyes: Prompt Testing to Create Aligned and
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

**[66. [2406.14449] APEER: Automatic Prompt Engineering Enhances Large Language Model
  Reranking](https://arxiv.org/pdf/2406.14449.pdf)** (2024-06-21)

*Can Jin, Hongwu Peng, Shiyu Zhao, Zhenting Wang, Wujiang Xu, Ligong Han, Jiahui Zhao, Kai Zhong, Sanguthevar Rajasekaran, Dimitris N. Metaxas*

  Large Language Models (LLMs) have significantly enhanced Information
Retrieval (IR) across various modules, such as reranking. Despite impressive
performance, current zero-shot relevance ranking with LLMs heavily relies on
human prompt engineering. Existing automatic prompt engineering algorithms
primarily focus on language modeling and classification tasks, leaving the
domain of IR, particularly reranking, underexplored. Directly applying current
prompt engineering algorithms to relevance ranking is challenging due to the
integration of query and long passage pairs in the input, where the ranking
complexity surpasses classification tasks. To reduce human effort and unlock
the potential of prompt optimization in reranking, we introduce a novel
automatic prompt engineering algorithm named APEER. APEER iteratively generates
refined prompts through feedback and preference optimization. Extensive
experiments with four LLMs and ten datasets demonstrate the substantial
performance improvement of APEER over existing state-of-the-art (SoTA) manual
prompts. Furthermore, we find that the prompts generated by APEER exhibit
better transferability across diverse tasks and LLMs. Code is available at
https://github.com/jincan333/APEER.


---

**[67. [2502.13603] Efficient Safety Retrofitting Against Jailbreaking for LLMs](https://arxiv.org/pdf/2502.13603.pdf)** (2025-02-26)

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

**[68. [2501.10915] LegalGuardian: A Privacy-Preserving Framework for Secure Integration of
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

**[69. [2404.11262] Sampling-based Pseudo-Likelihood for Membership Inference Attacks](https://arxiv.org/pdf/2404.11262.pdf)** (2024-04-18)

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

**[70. [2410.14676] SudoLM: Learning Access Control of Parametric Knowledge with
  Authorization Alignment](https://arxiv.org/pdf/2410.14676.pdf)** (2025-02-28)

*Qin Liu, Fei Wang, Chaowei Xiao, Muhao Chen*

  Existing preference alignment is a one-size-fits-all alignment mechanism,
where the part of the large language model (LLM) parametric knowledge with
non-preferred features is uniformly blocked to all the users. However, this
part of knowledge can be useful to advanced users whose expertise qualifies
them to handle these information. The one-size-fits-all alignment mechanism
undermines LLM's utility for these qualified users. To address this problem, we
propose SudoLM, a framework that lets LLMs learn access control over specific
parametric knowledge for users with different credentials via authorization
alignment. SudoLM allows authorized users to unlock their access to all the
parametric knowledge with an assigned SUDO key while blocking access to
non-qualified users. Experiments on two application scenarios demonstrate that
SudoLM effectively controls the user's access to the parametric knowledge and
maintains its general utility.


---

**[71. [2410.12265] An Automatic and Cost-Efficient Peer-Review Framework for Language
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

**[72. [2410.16848] ETHIC: Evaluating Large Language Models on Long-Context Tasks with High
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

**[73. [2309.11830] Goal-Oriented Prompt Attack and Safety Evaluation for LLMs](https://arxiv.org/pdf/2309.11830.pdf)** (2023-12-11)

*Chengyuan Liu, Fubang Zhao, Lizhi Qing, Yangyang Kang, Changlong Sun, Kun Kuang, Fei Wu*

  Large Language Models (LLMs) presents significant priority in text
understanding and generation. However, LLMs suffer from the risk of generating
harmful contents especially while being employed to applications. There are
several black-box attack methods, such as Prompt Attack, which can change the
behaviour of LLMs and induce LLMs to generate unexpected answers with harmful
contents. Researchers are interested in Prompt Attack and Defense with LLMs,
while there is no publicly available dataset with high successful attacking
rate to evaluate the abilities of defending prompt attack. In this paper, we
introduce a pipeline to construct high-quality prompt attack samples, along
with a Chinese prompt attack dataset called CPAD. Our prompts aim to induce
LLMs to generate unexpected outputs with several carefully designed prompt
attack templates and widely concerned attacking contents. Different from
previous datasets involving safety estimation, we construct the prompts
considering three dimensions: contents, attacking methods and goals.
Especially, the attacking goals indicate the behaviour expected after
successfully attacking the LLMs, thus the responses can be easily evaluated and
analysed. We run several popular Chinese LLMs on our dataset, and the results
show that our prompts are significantly harmful to LLMs, with around 70% attack
success rate to GPT-3.5. CPAD is publicly available at
https://github.com/liuchengyuan123/CPAD.


---

**[74. [2411.04965] BitNet a4.8: 4-bit Activations for 1-bit LLMs](https://arxiv.org/pdf/2411.04965.pdf)** (2024-11-08)

*Hongyu Wang, Shuming Ma, Furu Wei*

  Recent research on the 1-bit Large Language Models (LLMs), such as BitNet
b1.58, presents a promising direction for reducing the inference cost of LLMs
while maintaining their performance. In this work, we introduce BitNet a4.8,
enabling 4-bit activations for 1-bit LLMs. BitNet a4.8 employs a hybrid
quantization and sparsification strategy to mitigate the quantization errors
introduced by the outlier channels. Specifically, we utilize 4-bit activations
for inputs to the attention and feed-forward network layers, while sparsifying
intermediate states followed with 8-bit quantization. Extensive experiments
demonstrate that BitNet a4.8 achieves performance comparable to BitNet b1.58
with equivalent training costs, while being faster in inference with enabling
4-bit (INT4/FP4) kernels. Additionally, BitNet a4.8 activates only 55% of
parameters and supports 3-bit KV cache, further enhancing the efficiency of
large-scale LLM deployment and inference.


---

**[75. [2404.13968] Protecting Your LLMs with Information Bottleneck](https://arxiv.org/pdf/2404.13968.pdf)** (2024-10-11)

*Zichuan Liu, Zefan Wang, Linjie Xu, Jinyu Wang, Lei Song, Tianchun Wang, Chunlin Chen, Wei Cheng, Jiang Bian*

  The advent of large language models (LLMs) has revolutionized the field of
natural language processing, yet they might be attacked to produce harmful
content. Despite efforts to ethically align LLMs, these are often fragile and
can be circumvented by jailbreaking attacks through optimized or manual
adversarial prompts. To address this, we introduce the Information Bottleneck
Protector (IBProtector), a defense mechanism grounded in the information
bottleneck principle, and we modify the objective to avoid trivial solutions.
The IBProtector selectively compresses and perturbs prompts, facilitated by a
lightweight and trainable extractor, preserving only essential information for
the target LLMs to respond with the expected answer. Moreover, we further
consider a situation where the gradient is not visible to be compatible with
any LLM. Our empirical evaluations show that IBProtector outperforms current
defense methods in mitigating jailbreak attempts, without overly affecting
response quality or inference speed. Its effectiveness and adaptability across
various attack methods and target LLMs underscore the potential of IBProtector
as a novel, transferable defense that bolsters the security of LLMs without
requiring modifications to the underlying models.


---

**[76. [2411.14571] Assessment of LLM Responses to End-user Security Questions](https://arxiv.org/pdf/2411.14571.pdf)** (2024-11-25)

*Vijay Prakash, Kevin Lee, Arkaprabha Bhattacharya, Danny Yuxing Huang, Jessica Staddon*

  Answering end user security questions is challenging. While large language
models (LLMs) like GPT, LLAMA, and Gemini are far from error-free, they have
shown promise in answering a variety of questions outside of security. We
studied LLM performance in the area of end user security by qualitatively
evaluating 3 popular LLMs on 900 systematically collected end user security
questions.
  While LLMs demonstrate broad generalist ``knowledge'' of end user security
information, there are patterns of errors and limitations across LLMs
consisting of stale and inaccurate answers, and indirect or unresponsive
communication styles, all of which impacts the quality of information received.
Based on these patterns, we suggest directions for model improvement and
recommend user strategies for interacting with LLMs when seeking assistance
with security.


---

**[77. [2407.07666] A Proposed S.C.O.R.E. Evaluation Framework for Large Language Models :
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

**[78. [2502.07218] LUNAR: LLM Unlearning via Neural Activation Redirection](https://arxiv.org/pdf/2502.07218.pdf)** (2025-02-12)

*William F. Shen, Xinchi Qiu, Meghdad Kurmanji, Alex Iacob, Lorenzo Sani, Yihong Chen, Nicola Cancedda, Nicholas D. Lane*

  Large Language Models (LLMs) benefit from training on ever larger amounts of
textual data, but as a result, they increasingly incur the risk of leaking
private information. The ability to selectively remove knowledge from LLMs is,
therefore, a highly desirable capability. In this paper, we propose LUNAR, a
novel unlearning methodology grounded in the Linear Representation Hypothesis.
LUNAR operates by redirecting the representations of unlearned data to regions
that trigger the model's inherent ability to express its inability to answer.
LUNAR achieves state-of-the-art unlearning performance while significantly
enhancing the controllability of the unlearned model during inference.
Specifically, LUNAR achieves between 2.9x to 11.7x improvements on combined
"unlearning efficacy" and "model utility" score ("Deviation Score") on the
PISTOL dataset across various base models. We also demonstrate, through
quantitative analysis and qualitative examples, LUNAR's superior
controllability in generating coherent and contextually aware responses,
mitigating undesired side effects of existing methods. Moreover, we demonstrate
that LUNAR is robust against white-box adversarial attacks and versatile in
handling real-world scenarios, such as processing sequential unlearning
requests.


---

**[79. [2409.15654] Cambricon-LLM: A Chiplet-Based Hybrid Architecture for On-Device
  Inference of 70B LLM](https://arxiv.org/pdf/2409.15654.pdf)** (2024-09-25)

*Zhongkai Yu, Shengwen Liang, Tianyun Ma, Yunke Cai, Ziyuan Nan, Di Huang, Xinkai Song, Yifan Hao, Jie Zhang, Tian Zhi, Yongwei Zhao, Zidong Du, Xing Hu, Qi Guo, Tianshi Chen*

  Deploying advanced large language models on edge devices, such as smartphones
and robotics, is a growing trend that enhances user data privacy and network
connectivity resilience while preserving intelligent capabilities. However,
such a task exhibits single-batch computing with incredibly low arithmetic
intensity, which poses the significant challenges of huge memory footprint and
bandwidth demands on limited edge resources. To address these issues, we
introduce Cambricon-LLM, a chiplet-based hybrid architecture with NPU and a
dedicated NAND flash chip to enable efficient on-device inference of 70B LLMs.
Such a hybrid architecture utilizes both the high computing capability of NPU
and the data capacity of the NAND flash chip, with the proposed hardware-tiling
strategy that minimizes the data movement overhead between NPU and NAND flash
chip. Specifically, the NAND flash chip, enhanced by our innovative in-flash
computing and on-die ECC techniques, excels at performing precise lightweight
on-die processing. Simultaneously, the NPU collaborates with the flash chip for
matrix operations and handles special function computations beyond the flash's
on-die processing capabilities. Overall, Cambricon-LLM enables the on-device
inference of 70B LLMs at a speed of 3.44 token/s, and 7B LLMs at a speed of
36.34 token/s, which is over 22X to 45X faster than existing flash-offloading
technologies, showing the potentiality of deploying powerful LLMs in edge
devices.


---

**[80. [2407.03876] Automated Progressive Red Teaming](https://arxiv.org/pdf/2407.03876.pdf)** (2024-12-24)

*Bojian Jiang, Yi Jing, Tianhao Shen, Tong Wu, Qing Yang, Deyi Xiong*

  Ensuring the safety of large language models (LLMs) is paramount, yet
identifying potential vulnerabilities is challenging. While manual red teaming
is effective, it is time-consuming, costly and lacks scalability. Automated red
teaming (ART) offers a more cost-effective alternative, automatically
generating adversarial prompts to expose LLM vulnerabilities. However, in
current ART efforts, a robust framework is absent, which explicitly frames red
teaming as an effectively learnable task. To address this gap, we propose
Automated Progressive Red Teaming (APRT) as an effectively learnable framework.
APRT leverages three core modules: an Intention Expanding LLM that generates
diverse initial attack samples, an Intention Hiding LLM that crafts deceptive
prompts, and an Evil Maker to manage prompt diversity and filter ineffective
samples. The three modules collectively and progressively explore and exploit
LLM vulnerabilities through multi-round interactions. In addition to the
framework, we further propose a novel indicator, Attack Effectiveness Rate
(AER) to mitigate the limitations of existing evaluation metrics. By measuring
the likelihood of eliciting unsafe but seemingly helpful responses, AER aligns
closely with human evaluations. Extensive experiments with both automatic and
human evaluations, demonstrate the effectiveness of ARPT across both open- and
closed-source LLMs. Specifically, APRT effectively elicits 54% unsafe yet
useful responses from Meta's Llama-3-8B-Instruct, 50% from GPT-4o (API access),
and 39% from Claude-3.5 (API access), showcasing its robust attack capability
and transferability across LLMs (especially from open-source LLMs to
closed-source LLMs).


---

**[81. [2504.11168] Bypassing Prompt Injection and Jailbreak Detection in LLM Guardrails](https://arxiv.org/pdf/2504.11168.pdf)** (2025-04-17)

*William Hackett, Lewis Birch, Stefan Trawicki, Neeraj Suri, Peter Garraghan*

  Large Language Models (LLMs) guardrail systems are designed to protect
against prompt injection and jailbreak attacks. However, they remain vulnerable
to evasion techniques. We demonstrate two approaches for bypassing LLM prompt
injection and jailbreak detection systems via traditional character injection
methods and algorithmic Adversarial Machine Learning (AML) evasion techniques.
Through testing against six prominent protection systems, including Microsoft's
Azure Prompt Shield and Meta's Prompt Guard, we show that both methods can be
used to evade detection while maintaining adversarial utility achieving in some
instances up to 100% evasion success. Furthermore, we demonstrate that
adversaries can enhance Attack Success Rates (ASR) against black-box targets by
leveraging word importance ranking computed by offline white-box models. Our
findings reveal vulnerabilities within current LLM protection mechanisms and
highlight the need for more robust guardrail systems.


---

**[82. [2409.19091] System-Level Defense against Indirect Prompt Injection Attacks: An
  Information Flow Control Perspective](https://arxiv.org/pdf/2409.19091.pdf)** (2024-10-11)

*Fangzhou Wu, Ethan Cecchetti, Chaowei Xiao*

  Large Language Model-based systems (LLM systems) are information and query
processing systems that use LLMs to plan operations from natural-language
prompts and feed the output of each successive step into the LLM to plan the
next. This structure results in powerful tools that can process complex
information from diverse sources but raises critical security concerns.
Malicious information from any source may be processed by the LLM and can
compromise the query processing, resulting in nearly arbitrary misbehavior. To
tackle this problem, we present a system-level defense based on the principles
of information flow control that we call an f-secure LLM system. An f-secure
LLM system disaggregates the components of an LLM system into a context-aware
pipeline with dynamically generated structured executable plans, and a security
monitor filters out untrusted input into the planning process. This structure
prevents compromise while maximizing flexibility. We provide formal models for
both existing LLM systems and our f-secure LLM system, allowing analysis of
critical security guarantees. We further evaluate case studies and benchmarks
showing that f-secure LLM systems provide robust security while preserving
functionality and efficiency. Our code is released at
https://github.com/fzwark/Secure_LLM_System.


---

**[83. [2411.04282] Language Models are Hidden Reasoners: Unlocking Latent Reasoning
  Capabilities via Self-Rewarding](https://arxiv.org/pdf/2411.04282.pdf)** (2024-11-25)

*Haolin Chen, Yihao Feng, Zuxin Liu, Weiran Yao, Akshara Prabhakar, Shelby Heinecke, Ricky Ho, Phil Mui, Silvio Savarese, Caiming Xiong, Huan Wang*

  Large language models (LLMs) have shown impressive capabilities, but still
struggle with complex reasoning tasks requiring multiple steps. While
prompt-based methods like Chain-of-Thought (CoT) can improve LLM reasoning at
inference time, optimizing reasoning capabilities during training remains
challenging. We introduce LaTent Reasoning Optimization (LaTRO), a principled
framework that formulates reasoning as sampling from a latent distribution and
optimizes it via variational approaches. LaTRO enables LLMs to concurrently
improve both their reasoning process and ability to evaluate reasoning quality,
without requiring external feedback or reward models. We validate LaTRO through
experiments on GSM8K and ARC-Challenge datasets using multiple model
architectures. On GSM8K, LaTRO improves zero-shot accuracy by an average of
12.5% over base models and 9.6% over supervised fine-tuning across
Phi-3.5-mini, Mistral-7B, and Llama-3.1-8B. Our findings suggest that
pre-trained LLMs possess latent reasoning capabilities that can be unlocked and
enhanced through our proposed optimization approach in a self-improvement
manner. The code of LaTRO is available at
\url{https://github.com/SalesforceAIResearch/LaTRO}.


---

**[84. [2410.04838] Rationale-Aware Answer Verification by Pairwise Self-Evaluation](https://arxiv.org/pdf/2410.04838.pdf)** (2024-10-28)

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

**[85. [2406.18326] PaCoST: Paired Confidence Significance Testing for Benchmark
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

**[86. [2504.05804] StealthRank: LLM Ranking Manipulation via Stealthy Prompt Optimization](https://arxiv.org/pdf/2504.05804.pdf)** (2025-04-10)

*Yiming Tang, Yi Fan, Chenxiao Yu, Tiankai Yang, Yue Zhao, Xiyang Hu*

  The integration of large language models (LLMs) into information retrieval
systems introduces new attack surfaces, particularly for adversarial ranking
manipulations. We present StealthRank, a novel adversarial ranking attack that
manipulates LLM-driven product recommendation systems while maintaining textual
fluency and stealth. Unlike existing methods that often introduce detectable
anomalies, StealthRank employs an energy-based optimization framework combined
with Langevin dynamics to generate StealthRank Prompts (SRPs)-adversarial text
sequences embedded within product descriptions that subtly yet effectively
influence LLM ranking mechanisms. We evaluate StealthRank across multiple LLMs,
demonstrating its ability to covertly boost the ranking of target products
while avoiding explicit manipulation traces that can be easily detected. Our
results show that StealthRank consistently outperforms state-of-the-art
adversarial ranking baselines in both effectiveness and stealth, highlighting
critical vulnerabilities in LLM-driven recommendation systems.


---

**[87. [2307.07411] Detecting LLM-Generated Text in Computing Education: A Comparative Study
  for ChatGPT Cases](https://arxiv.org/pdf/2307.07411.pdf)** (2023-07-17)

*Michael Sheinman Orenstrakh, Oscar Karnalim, Carlos Anibal Suarez, Michael Liut*

  Due to the recent improvements and wide availability of Large Language Models
(LLMs), they have posed a serious threat to academic integrity in education.
Modern LLM-generated text detectors attempt to combat the problem by offering
educators with services to assess whether some text is LLM-generated. In this
work, we have collected 124 submissions from computer science students before
the creation of ChatGPT. We then generated 40 ChatGPT submissions. We used this
data to evaluate eight publicly-available LLM-generated text detectors through
the measures of accuracy, false positives, and resilience. The purpose of this
work is to inform the community of what LLM-generated text detectors work and
which do not, but also to provide insights for educators to better maintain
academic integrity in their courses. Our results find that CopyLeaks is the
most accurate LLM-generated text detector, GPTKit is the best LLM-generated
text detector to reduce false positives, and GLTR is the most resilient
LLM-generated text detector. We also express concerns over 52 false positives
(of 114 human written submissions) generated by GPTZero. Finally, we note that
all LLM-generated text detectors are less accurate with code, other languages
(aside from English), and after the use of paraphrasing tools (like QuillBot).
Modern detectors are still in need of improvements so that they can offer a
full-proof solution to help maintain academic integrity. Further, their
usability can be improved by facilitating a smooth API integration, providing
clear documentation of their features and the understandability of their
model(s), and supporting more commonly used languages.


---

**[88. [2504.00018] SandboxEval: Towards Securing Test Environment for Untrusted Code](https://arxiv.org/pdf/2504.00018.pdf)** (2025-04-02)

*Rafiqul Rabin, Jesse Hostetler, Sean McGregor, Brett Weir, Nick Judd*

  While large language models (LLMs) are powerful assistants in programming
tasks, they may also produce malicious code. Testing LLM-generated code
therefore poses significant risks to assessment infrastructure tasked with
executing untrusted code. To address these risks, this work focuses on
evaluating the security and confidentiality properties of test environments,
reducing the risk that LLM-generated code may compromise the assessment
infrastructure. We introduce SandboxEval, a test suite featuring manually
crafted test cases that simulate real-world safety scenarios for LLM assessment
environments in the context of untrusted code execution. The suite evaluates
vulnerabilities to sensitive information exposure, filesystem manipulation,
external communication, and other potentially dangerous operations in the
course of assessment activity. We demonstrate the utility of SandboxEval by
deploying it on an open-source implementation of Dyff, an established AI
assessment framework used to evaluate the safety of LLMs at scale. We show,
first, that the test suite accurately describes limitations placed on an LLM
operating under instructions to generate malicious code. Second, we show that
the test results provide valuable insights for developers seeking to harden
assessment infrastructure and identify risks associated with LLM execution
activities.


---

**[89. [2307.02916] The impact of an employee's psychological contract breach on compliance
  with information security policies: intrinsic and extrinsic motivation](https://arxiv.org/pdf/2307.02916.pdf)** (2023-07-07)

*Daeun Lee, Harjinder Singh Lallie, Nadine Michaelides*

  Despite the rapid rise in social engineering attacks, not all employees are
as compliant with information security policies (ISPs) to the extent that
organisations expect them to be. ISP non-compliance is caused by a variety of
psychological motivation. This study investigates the effect of psychological
contract breach (PCB) of employees on ISP compliance intention (ICI) by
dividing them into intrinsic and extrinsic motivation using the theory of
planned behaviour (TPB) and the general deterrence theory (GDT). Data analysis
from UK employees (\textit{n=206}) showed that the higher the PCB, the lower
the ICI. The study also found that PCBs significantly reduced intrinsic
motivation (attitude and perceived fairness) for ICI, whereas PCBs did not
moderate the relationship between extrinsic motivation (sanction severity and
sanctions certainty) and ICI. As a result, this study successfully addresses
the risks of PCBs in the field of IS security and proposes effective solutions
for employees with high PCBs.


---

**[90. [2410.20142] Mask-based Membership Inference Attacks for Retrieval-Augmented
  Generation](https://arxiv.org/pdf/2410.20142.pdf)** (2025-02-11)

*Mingrui Liu, Sixiao Zhang, Cheng Long*

  Retrieval-Augmented Generation (RAG) has been an effective approach to
mitigate hallucinations in large language models (LLMs) by incorporating
up-to-date and domain-specific knowledge. Recently, there has been a trend of
storing up-to-date or copyrighted data in RAG knowledge databases instead of
using it for LLM training. This practice has raised concerns about Membership
Inference Attacks (MIAs), which aim to detect if a specific target document is
stored in the RAG system's knowledge database so as to protect the rights of
data producers. While research has focused on enhancing the trustworthiness of
RAG systems, existing MIAs for RAG systems remain largely insufficient.
Previous work either relies solely on the RAG system's judgment or is easily
influenced by other documents or the LLM's internal knowledge, which is
unreliable and lacks explainability. To address these limitations, we propose a
Mask-Based Membership Inference Attacks (MBA) framework. Our framework first
employs a masking algorithm that effectively masks a certain number of words in
the target document. The masked text is then used to prompt the RAG system, and
the RAG system is required to predict the mask values. If the target document
appears in the knowledge database, the masked text will retrieve the complete
target document as context, allowing for accurate mask prediction. Finally, we
adopt a simple yet effective threshold-based method to infer the membership of
target document by analyzing the accuracy of mask prediction. Our mask-based
approach is more document-specific, making the RAG system's generation less
susceptible to distractions from other documents or the LLM's internal
knowledge. Extensive experiments demonstrate the effectiveness of our approach
compared to existing baseline models.


---

**[91. [2406.01333] Probing Language Models for Pre-training Data Detection](https://arxiv.org/pdf/2406.01333.pdf)** (2024-06-04)

*Zhenhua Liu, Tong Zhu, Chuanyuan Tan, Haonan Lu, Bing Liu, Wenliang Chen*

  Large Language Models (LLMs) have shown their impressive capabilities, while
also raising concerns about the data contamination problems due to privacy
issues and leakage of benchmark datasets in the pre-training phase. Therefore,
it is vital to detect the contamination by checking whether an LLM has been
pre-trained on the target texts. Recent studies focus on the generated texts
and compute perplexities, which are superficial features and not reliable. In
this study, we propose to utilize the probing technique for pre-training data
detection by examining the model's internal activations. Our method is simple
and effective and leads to more trustworthy pre-training data detection.
Additionally, we propose ArxivMIA, a new challenging benchmark comprising arxiv
abstracts from Computer Science and Mathematics categories. Our experiments
demonstrate that our method outperforms all baselines, and achieves
state-of-the-art performance on both WikiMIA and ArxivMIA, with additional
experiments confirming its efficacy (Our code and dataset are available at
https://github.com/zhliu0106/probing-lm-data).


---

**[92. [2312.06550] LLM360: Towards Fully Transparent Open-Source LLMs](https://arxiv.org/pdf/2312.06550.pdf)** (2023-12-12)

*Zhengzhong Liu, Aurick Qiao, Willie Neiswanger, Hongyi Wang, Bowen Tan, Tianhua Tao, Junbo Li, Yuqi Wang, Suqi Sun, Omkar Pangarkar, Richard Fan, Yi Gu, Victor Miller, Yonghao Zhuang, Guowei He, Haonan Li, Fajri Koto, Liping Tang, Nikhil Ranjan, Zhiqiang Shen, Xuguang Ren, Roberto Iriondo, Cun Mu, Zhiting Hu, Mark Schulze, Preslav Nakov, Tim Baldwin, Eric P. Xing*

  The recent surge in open-source Large Language Models (LLMs), such as LLaMA,
Falcon, and Mistral, provides diverse options for AI practitioners and
researchers. However, most LLMs have only released partial artifacts, such as
the final model weights or inference code, and technical reports increasingly
limit their scope to high-level design choices and surface statistics. These
choices hinder progress in the field by degrading transparency into the
training of LLMs and forcing teams to rediscover many details in the training
process. We present LLM360, an initiative to fully open-source LLMs, which
advocates for all training code and data, model checkpoints, and intermediate
results to be made available to the community. The goal of LLM360 is to support
open and collaborative AI research by making the end-to-end LLM training
process transparent and reproducible by everyone. As a first step of LLM360, we
release two 7B parameter LLMs pre-trained from scratch, Amber and CrystalCoder,
including their training code, data, intermediate checkpoints, and analyses (at
https://www.llm360.ai). We are committed to continually pushing the boundaries
of LLMs through this open-source effort. More large-scale and stronger models
are underway and will be released in the future.


---

**[93. [2306.16127] MLSMM: Machine Learning Security Maturity Model](https://arxiv.org/pdf/2306.16127.pdf)** (2023-06-29)

*Felix Jedrzejewski, Davide Fucci, Oleksandr Adamov*

  Assessing the maturity of security practices during the development of
Machine Learning (ML) based software components has not gotten as much
attention as traditional software development. In this Blue Sky idea paper, we
propose an initial Machine Learning Security Maturity Model (MLSMM) which
organizes security practices along the ML-development lifecycle and, for each,
establishes three levels of maturity. We envision MLSMM as a step towards
closer collaboration between industry and academia.


---

**[94. [2502.08922] Self-Consistency of the Internal Reward Models Improves Self-Rewarding
  Language Models](https://arxiv.org/pdf/2502.08922.pdf)** (2025-02-14)

*Xin Zhou, Yiwen Guo, Ruotian Ma, Tao Gui, Qi Zhang, Xuanjing Huang*

  Aligning Large Language Models (LLMs) with human preferences is crucial for
their deployment in real-world applications. Recent advancements in
Self-Rewarding Language Models suggest that an LLM can use its internal reward
models (such as LLM-as-a-Judge) \cite{yuanself} to generate preference data,
improving alignment performance without costly human annotation. However, we
find that different internal reward models within the same LLM often generate
inconsistent preferences. This inconsistency raises concerns about the
reliability of self-generated preference data, hinders overall alignment
performance, and highlights the need for further research to ensure reliable
and coherent alignment with human preferences. To address this limitation, we
propose Self-Consistent Internal Rewards (SCIR), a novel framework designed to
enhance consistency among internal reward models during training. In each
training step, we collect preference predictions from multiple pre-defined
internal reward models and enforce consistency and confidence through an
inconsistency penalty mechanism, thereby improving the reliability of these
internal reward models. We selectively use data with consistent predictions for
preference optimization, ensuring the quality of the preference data. By
employing self-consistent internal rewards, our method significantly improves
the alignment performance and reward modeling capability of LLMs, outperforming
baseline methods by a notable margin.


---

**[95. [2012.14205] Contract-Aware Secure Compilation](https://arxiv.org/pdf/2012.14205.pdf)** (2020-12-29)

*Marco Guarnieri, Marco Patrignani*

  Microarchitectural attacks exploit the abstraction gap between the
Instruction Set Architecture (ISA) and how instructions are actually executed
by processors to compromise the confidentiality and integrity of a system. To
secure systems against microarchitectural attacks, programmers need to reason
about and program against these microarchitectural side-effects. However, we
cannot -- and should not -- expect programmers to manually tailor programs for
specific processors and their security guarantees. Instead, we could rely on
compilers (and the secure compilation community), as they can play a prominent
role in bridging this gap: compilers should target specific processors
microarchitectural security guarantees and they should leverage these
guarantees to produce secure code. To achieve this, we outline the idea of
Contract-Aware Secure COmpilation (CASCO) where compilers are parametric with
respect to a hardware/software security-contract, an abstraction capturing a
processor's security guarantees. That is, compilers will automatically leverage
the guarantees formalized in the contract to ensure that program-level security
properties are preserved at microarchitectural level.


---

**[96. [2406.10040] FZI-WIM at SemEval-2024 Task 2: Self-Consistent CoT for Complex NLI in
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

**[97. [2410.13236] SPIN: Self-Supervised Prompt INjection](https://arxiv.org/pdf/2410.13236.pdf)** (2024-10-18)

*Leon Zhou, Junfeng Yang, Chengzhi Mao*

  Large Language Models (LLMs) are increasingly used in a variety of important
applications, yet their safety and reliability remain as major concerns.
Various adversarial and jailbreak attacks have been proposed to bypass the
safety alignment and cause the model to produce harmful responses. We introduce
Self-supervised Prompt INjection (SPIN) which can detect and reverse these
various attacks on LLMs. As our self-supervised prompt defense is done at
inference-time, it is also compatible with existing alignment and adds an
additional layer of safety for defense. Our benchmarks demonstrate that our
system can reduce the attack success rate by up to 87.9%, while maintaining the
performance on benign user requests. In addition, we discuss the situation of
an adaptive attacker and show that our method is still resilient against
attackers who are aware of our defense.


---

**[98. [2406.08754] StructuralSleight: Automated Jailbreak Attacks on Large Language Models
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

**[99. [2503.20320] Iterative Prompting with Persuasion Skills in Jailbreaking Large
  Language Models](https://arxiv.org/pdf/2503.20320.pdf)** (2025-03-27)

*Shih-Wen Ke, Guan-Yu Lai, Guo-Lin Fang, Hsi-Yuan Kao*

  Large language models (LLMs) are designed to align with human values in their
responses. This study exploits LLMs with an iterative prompting technique where
each prompt is systematically modified and refined across multiple iterations
to enhance its effectiveness in jailbreaking attacks progressively. This
technique involves analyzing the response patterns of LLMs, including GPT-3.5,
GPT-4, LLaMa2, Vicuna, and ChatGLM, allowing us to adjust and optimize prompts
to evade the LLMs' ethical and security constraints. Persuasion strategies
enhance prompt effectiveness while maintaining consistency with malicious
intent. Our results show that the attack success rates (ASR) increase as the
attacking prompts become more refined with the highest ASR of 90% for GPT4 and
ChatGLM and the lowest ASR of 68% for LLaMa2. Our technique outperforms
baseline techniques (PAIR and PAP) in ASR and shows comparable performance with
GCG and ArtPrompt.


---

**[100. [2402.01733] Development and Testing of Retrieval Augmented Generation in Large
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

**[101. [2401.14351] ServerlessLLM: Low-Latency Serverless Inference for Large Language
  Models](https://arxiv.org/pdf/2401.14351.pdf)** (2024-07-26)

*Yao Fu, Leyang Xue, Yeqi Huang, Andrei-Octavian Brabete, Dmitrii Ustiugov, Yuvraj Patel, Luo Mai*

  This paper presents ServerlessLLM, a distributed system designed to support
low-latency serverless inference for Large Language Models (LLMs). By
harnessing the substantial near-GPU storage and memory capacities of inference
servers, ServerlessLLM achieves effective local checkpoint storage, minimizing
the need for remote checkpoint downloads and ensuring efficient checkpoint
loading. The design of ServerlessLLM features three core contributions: (i)
\emph{fast multi-tier checkpoint loading}, featuring a new loading-optimized
checkpoint format and a multi-tier loading system, fully utilizing the
bandwidth of complex storage hierarchies on GPU servers; (ii) \emph{efficient
live migration of LLM inference}, which enables newly initiated inferences to
capitalize on local checkpoint storage while ensuring minimal user
interruption; and (iii) \emph{startup-time-optimized model scheduling}, which
assesses the locality statuses of checkpoints on each server and schedules the
model onto servers that minimize the time to start the inference. Comprehensive
evaluations, including microbenchmarks and real-world scenarios, demonstrate
that ServerlessLLM dramatically outperforms state-of-the-art serverless
systems, reducing latency by 10 - 200X across various LLM inference workloads.


---

**[102. [2410.15483] Mitigating Forgetting in LLM Supervised Fine-Tuning and Preference
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

**[103. [2211.04351] The ERA Theorem for Safe Memory Reclamation](https://arxiv.org/pdf/2211.04351.pdf)** (2022-11-09)

*Gali Sheffi, Erez Petrank*

  Safe memory reclamation (SMR) schemes for concurrent data structures offer
trade-offs between three desirable properties: ease of integration, robustness,
and applicability. In this paper we rigorously define SMR and these three
properties, and we present the ERA theorem, asserting that any SMR scheme can
only provide at most two of the three properties.


---

**[104. [2407.10582] Boosting Zero-Shot Crosslingual Performance using LLM-Based
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

**[105. [2409.14961] UELLM: A Unified and Efficient Approach for LLM Inference Serving](https://arxiv.org/pdf/2409.14961.pdf)** (2024-09-25)

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

**[106. [2410.20290] Fast Best-of-N Decoding via Speculative Rejection](https://arxiv.org/pdf/2410.20290.pdf)** (2024-11-04)

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

**[107. [2402.15770] From COBIT to ISO 42001: Evaluating Cybersecurity Frameworks for
  Opportunities, Risks, and Regulatory Compliance in Commercializing Large
  Language Models](https://arxiv.org/pdf/2402.15770.pdf)** (2024-06-25)

*Timothy R. McIntosh, Teo Susnjak, Tong Liu, Paul Watters, Raza Nowrozy, Malka N. Halgamuge*

  This study investigated the integration readiness of four predominant
cybersecurity Governance, Risk and Compliance (GRC) frameworks - NIST CSF 2.0,
COBIT 2019, ISO 27001:2022, and the latest ISO 42001:2023 - for the
opportunities, risks, and regulatory compliance when adopting Large Language
Models (LLMs), using qualitative content analysis and expert validation. Our
analysis, with both LLMs and human experts in the loop, uncovered potential for
LLM integration together with inadequacies in LLM risk oversight of those
frameworks. Comparative gap analysis has highlighted that the new ISO
42001:2023, specifically designed for Artificial Intelligence (AI) management
systems, provided most comprehensive facilitation for LLM opportunities,
whereas COBIT 2019 aligned most closely with the impending European Union AI
Act. Nonetheless, our findings suggested that all evaluated frameworks would
benefit from enhancements to more effectively and more comprehensively address
the multifaceted risks associated with LLMs, indicating a critical and
time-sensitive need for their continuous evolution. We propose integrating
human-expert-in-the-loop validation processes as crucial for enhancing
cybersecurity frameworks to support secure and compliant LLM integration, and
discuss implications for the continuous evolution of cybersecurity GRC
frameworks to support the secure integration of LLMs.


---

**[108. [2406.06474] Towards a Personal Health Large Language Model](https://arxiv.org/pdf/2406.06474.pdf)** (2024-06-11)

*Justin Cosentino, Anastasiya Belyaeva, Xin Liu, Nicholas A. Furlotte, Zhun Yang, Chace Lee, Erik Schenck, Yojan Patel, Jian Cui, Logan Douglas Schneider, Robby Bryant, Ryan G. Gomes, Allen Jiang, Roy Lee, Yun Liu, Javier Perez, Jameson K. Rogers, Cathy Speed, Shyam Tailor, Megan Walker, Jeffrey Yu, Tim Althoff, Conor Heneghan, John Hernandez, Mark Malhotra, Leor Stern, Yossi Matias, Greg S. Corrado, Shwetak Patel, Shravya Shetty, Jiening Zhan, Shruthi Prabhakara, Daniel McDuff, Cory Y. McLean*

  In health, most large language model (LLM) research has focused on clinical
tasks. However, mobile and wearable devices, which are rarely integrated into
such tasks, provide rich, longitudinal data for personal health monitoring.
Here we present Personal Health Large Language Model (PH-LLM), fine-tuned from
Gemini for understanding and reasoning over numerical time-series personal
health data. We created and curated three datasets that test 1) production of
personalized insights and recommendations from sleep patterns, physical
activity, and physiological responses, 2) expert domain knowledge, and 3)
prediction of self-reported sleep outcomes. For the first task we designed 857
case studies in collaboration with domain experts to assess real-world
scenarios in sleep and fitness. Through comprehensive evaluation of
domain-specific rubrics, we observed that Gemini Ultra 1.0 and PH-LLM are not
statistically different from expert performance in fitness and, while experts
remain superior for sleep, fine-tuning PH-LLM provided significant improvements
in using relevant domain knowledge and personalizing information for sleep
insights. We evaluated PH-LLM domain knowledge using multiple choice sleep
medicine and fitness examinations. PH-LLM achieved 79% on sleep and 88% on
fitness, exceeding average scores from a sample of human experts. Finally, we
trained PH-LLM to predict self-reported sleep quality outcomes from textual and
multimodal encoding representations of wearable data, and demonstrate that
multimodal encoding is required to match performance of specialized
discriminative models. Although further development and evaluation are
necessary in the safety-critical personal health domain, these results
demonstrate both the broad knowledge and capabilities of Gemini models and the
benefit of contextualizing physiological data for personal health applications
as done with PH-LLM.


---

**[109. [2503.15450] SkyLadder: Better and Faster Pretraining via Context Window Scheduling](https://arxiv.org/pdf/2503.15450.pdf)** (2025-03-20)

*Tongyao Zhu, Qian Liu, Haonan Wang, Shiqi Chen, Xiangming Gu, Tianyu Pang, Min-Yen Kan*

  Recent advancements in LLM pretraining have featured ever-expanding context
windows to process longer sequences. However, our pilot study reveals that
models pretrained with shorter context windows consistently outperform their
long-context counterparts under a fixed token budget. This finding motivates us
to explore an optimal context window scheduling strategy to better balance
long-context capability with pretraining efficiency. To this end, we propose
SkyLadder, a simple yet effective approach that implements a short-to-long
context window transition. SkyLadder preserves strong standard benchmark
performance, while matching or exceeding baseline results on long context
tasks. Through extensive experiments, we pre-train 1B-parameter models (up to
32K context) and 3B-parameter models (8K context) on 100B tokens, demonstrating
that SkyLadder yields consistent gains of up to 3.7% on common benchmarks,
while achieving up to 22% faster training speeds compared to baselines. The
code is at https://github.com/sail-sg/SkyLadder.


---

**[110. [2404.19048] A Framework for Real-time Safeguarding the Text Generation of Large
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

**[111. [2404.04392] Fine-Tuning, Quantization, and LLMs: Navigating Unintended Outcomes](https://arxiv.org/pdf/2404.04392.pdf)** (2024-09-10)

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

**[112. [2001.10512] Automated Proof of Bell-LaPadula Security Properties](https://arxiv.org/pdf/2001.10512.pdf)** (2020-07-17)

*Maximiliano Cristia, Gianfranco Rossi*

  Almost fifty years ago, D.E. Bell and L. LaPadula published the first formal
model of a secure system, known today as the Bell-LaPadula (BLP) model. BLP is
described as a state machine by means of first-order logic and set theory. The
authors also formalize two state invariants known as security condition and
*-property. Bell and LaPadula prove that all the state transitions preserve
these invariants.
  In this paper we present a fully automated proof of the security condition
and the *-property for all the model operations. The model and the proofs are
coded in the {log} tool. As far as we know this is the first time such proofs
are automated. Besides, we show that the {log} model is also an executable
prototype. Therefore we are providing an automatically verified executable
prototype of BLP.


---

**[113. [2304.04521] GL-MCM: Global and Local Maximum Concept Matching for Zero-Shot
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

**[114. [2406.11880] Knowledge Return Oriented Prompting (KROP)](https://arxiv.org/pdf/2406.11880.pdf)** (2024-06-19)

*Jason Martin, Kenneth Yeung*

  Many Large Language Models (LLMs) and LLM-powered apps deployed today use
some form of prompt filter or alignment to protect their integrity. However,
these measures aren't foolproof. This paper introduces KROP, a prompt injection
technique capable of obfuscating prompt injection attacks, rendering them
virtually undetectable to most of these security measures.


---

**[115. [2409.04040] A First Look At Efficient And Secure On-Device LLM Inference Against KV
  Leakage](https://arxiv.org/pdf/2409.04040.pdf)** (2024-09-09)

*Huan Yang, Deyu Zhang, Yudong Zhao, Yuanchun Li, Yunxin Liu*

  Running LLMs on end devices has garnered significant attention recently due
to their advantages in privacy preservation. With the advent of lightweight LLM
models and specially designed GPUs, on-device LLM inference has achieved the
necessary accuracy and performance metrics. However, we have identified that
LLM inference on GPUs can leak privacy-sensitive intermediate information,
specifically the KV pairs. An attacker could exploit these KV pairs to
reconstruct the entire user conversation, leading to significant
vulnerabilities. Existing solutions, such as Fully Homomorphic Encryption (FHE)
and Trusted Execution Environments (TEE), are either too computation-intensive
or resource-limited. To address these issues, we designed KV-Shield, which
operates in two phases. In the initialization phase, it permutes the weight
matrices so that all KV pairs are correspondingly permuted. During the runtime
phase, the attention vector is inversely permuted to ensure the correctness of
the layer output. All permutation-related operations are executed within the
TEE, ensuring that insecure GPUs cannot access the original KV pairs, thus
preventing conversation reconstruction. Finally, we theoretically analyze the
correctness of KV-Shield, along with its advantages and overhead.


---

**[116. [2402.11592] Revisiting Zeroth-Order Optimization for Memory-Efficient LLM
  Fine-Tuning: A Benchmark](https://arxiv.org/pdf/2402.11592.pdf)** (2024-05-29)

*Yihua Zhang, Pingzhi Li, Junyuan Hong, Jiaxiang Li, Yimeng Zhang, Wenqing Zheng, Pin-Yu Chen, Jason D. Lee, Wotao Yin, Mingyi Hong, Zhangyang Wang, Sijia Liu, Tianlong Chen*

  In the evolving landscape of natural language processing (NLP), fine-tuning
pre-trained Large Language Models (LLMs) with first-order (FO) optimizers like
SGD and Adam has become standard. Yet, as LLMs grow {in size}, the substantial
memory overhead from back-propagation (BP) for FO gradient computation presents
a significant challenge. Addressing this issue is crucial, especially for
applications like on-device training where memory efficiency is paramount. This
paper proposes a shift towards BP-free, zeroth-order (ZO) optimization as a
solution for reducing memory costs during LLM fine-tuning, building on the
initial concept introduced by MeZO. Unlike traditional ZO-SGD methods, our work
expands the exploration to a wider array of ZO optimization techniques, through
a comprehensive, first-of-its-kind benchmarking study across five LLM families
(Roberta, OPT, LLaMA, Vicuna, Mistral), three task complexities, and five
fine-tuning schemes. Our study unveils previously overlooked optimization
principles, highlighting the importance of task alignment, the role of the
forward gradient method, and the balance between algorithm complexity and
fine-tuning performance. We further introduce novel enhancements to ZO
optimization, including block-wise descent, hybrid training, and gradient
sparsity. Our study offers a promising direction for achieving further
memory-efficient LLM fine-tuning. Codes to reproduce all our experiments are at
https://github.com/ZO-Bench/ZO-LLM .


---

**[117. [2212.09561] Large Language Models are Better Reasoners with Self-Verification](https://arxiv.org/pdf/2212.09561.pdf)** (2023-10-20)

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

**[118. [2411.12764] SEFD: Semantic-Enhanced Framework for Detecting LLM-Generated Text](https://arxiv.org/pdf/2411.12764.pdf)** (2024-11-21)

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

**[119. [2502.13141] UniGuardian: A Unified Defense for Detecting Prompt Injection, Backdoor
  Attacks and Adversarial Attacks in Large Language Models](https://arxiv.org/pdf/2502.13141.pdf)** (2025-02-19)

*Huawei Lin, Yingjie Lao, Tong Geng, Tan Yu, Weijie Zhao*

  Large Language Models (LLMs) are vulnerable to attacks like prompt injection,
backdoor attacks, and adversarial attacks, which manipulate prompts or models
to generate harmful outputs. In this paper, departing from traditional deep
learning attack paradigms, we explore their intrinsic relationship and
collectively term them Prompt Trigger Attacks (PTA). This raises a key
question: Can we determine if a prompt is benign or poisoned? To address this,
we propose UniGuardian, the first unified defense mechanism designed to detect
prompt injection, backdoor attacks, and adversarial attacks in LLMs.
Additionally, we introduce a single-forward strategy to optimize the detection
pipeline, enabling simultaneous attack detection and text generation within a
single forward pass. Our experiments confirm that UniGuardian accurately and
efficiently identifies malicious prompts in LLMs.


---

**[120. [2502.19954] Collaborative Stance Detection via Small-Large Language Model
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

**[121. [2411.06493] LProtector: An LLM-driven Vulnerability Detection System](https://arxiv.org/pdf/2411.06493.pdf)** (2024-11-15)

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

**[122. [2406.12319] The Comparative Trap: Pairwise Comparisons Amplifies Biased Preferences
  of LLM Evaluators](https://arxiv.org/pdf/2406.12319.pdf)** (2025-04-21)

*Hawon Jeong, ChaeHun Park, Jimin Hong, Hojoon Lee, Jaegul Choo*

  As large language models (LLMs) are increasingly used as evaluators for
natural language generation tasks, ensuring unbiased assessments is essential.
However, LLM evaluators often display biased preferences, such as favoring
verbosity and authoritative tones. Our empirical analysis reveals that these
biases are exacerbated in pairwise evaluation, where LLMs directly compare two
outputs and easily prioritize superficial attributes. In contrast, pointwise
evaluation, which assesses outputs independently, is less susceptible to such
bias because each output is judged in isolation. To address the limitations of
the pairwise evaluation, we introduce a novel evaluation method, PRePair, which
integrates pointwise reasoning within a pairwise framework. PRePair effectively
alleviates biased preference, improving performance on the adversarial
benchmark (LLMBar) while outperforming pointwise evaluation on the standard
benchmark (MT-Bench).


---

**[123. [2203.07580] TSM: Measuring the Enticement of Honeyfiles with Natural Language
  Processing](https://arxiv.org/pdf/2203.07580.pdf)** (2022-03-16)

*Roelien C. Timmer, David Liebowitz, Surya Nepal, Salil Kanhere*

  Honeyfile deployment is a useful breach detection method in cyber deception
that can also inform defenders about the intent and interests of intruders and
malicious insiders. A key property of a honeyfile, enticement, is the extent to
which the file can attract an intruder to interact with it. We introduce a
novel metric, Topic Semantic Matching (TSM), which uses topic modelling to
represent files in the repository and semantic matching in an embedding vector
space to compare honeyfile text and topic words robustly. We also present a
honeyfile corpus created with different Natural Language Processing (NLP)
methods. Experiments show that TSM is effective in inter-corpus comparisons and
is a promising tool to measure the enticement of honeyfiles. TSM is the first
measure to use NLP techniques to quantify the enticement of honeyfile content
that compares the essential topical content of local contexts to honeyfiles and
is robust to paraphrasing.


---

**[124. [2410.07137] Cheating Automatic LLM Benchmarks: Null Models Achieve High Win Rates](https://arxiv.org/pdf/2410.07137.pdf)** (2025-03-04)

*Xiaosen Zheng, Tianyu Pang, Chao Du, Qian Liu, Jing Jiang, Min Lin*

  Automatic LLM benchmarks, such as AlpacaEval 2.0, Arena-Hard-Auto, and
MT-Bench, have become popular for evaluating language models due to their
cost-effectiveness and scalability compared to human evaluation. Achieving high
win rates on these benchmarks can significantly boost the promotional impact of
newly released language models. This promotional benefit may motivate tricks,
such as manipulating model output length or style to game win rates, even
though several mechanisms have been developed to control length and disentangle
style to reduce gameability. Nonetheless, we show that even a "null model" that
always outputs a constant response (irrelevant to input instructions) can cheat
automatic benchmarks and achieve top-ranked win rates: an 86.5% LC win rate on
AlpacaEval 2.0; an 83.0 score on Arena-Hard-Auto; and a 9.55 score on MT-Bench.
Moreover, the crafted cheating outputs are transferable because we assume that
the instructions of these benchmarks (e.g., 805 samples of AlpacaEval 2.0) are
private and cannot be accessed. While our experiments are primarily
proof-of-concept, an adversary could use LLMs to generate more imperceptible
cheating responses, unethically benefiting from high win rates and promotional
impact. Our findings call for the development of anti-cheating mechanisms for
reliable automatic benchmarks. The code is available at
https://github.com/sail-sg/Cheating-LLM-Benchmarks.


---

**[125. [2501.13916] PBM-VFL: Vertical Federated Learning with Feature and Sample Privacy](https://arxiv.org/pdf/2501.13916.pdf)** (2025-01-29)

*Linh Tran, Timothy Castiglia, Stacy Patterson, Ana Milanova*

  We present Poisson Binomial Mechanism Vertical Federated Learning (PBM-VFL),
a communication-efficient Vertical Federated Learning algorithm with
Differential Privacy guarantees. PBM-VFL combines Secure Multi-Party
Computation with the recently introduced Poisson Binomial Mechanism to protect
parties' private datasets during model training. We define the novel concept of
feature privacy and analyze end-to-end feature and sample privacy of our
algorithm. We compare sample privacy loss in VFL with privacy loss in HFL. We
also provide the first theoretical characterization of the relationship between
privacy budget, convergence error, and communication cost in
differentially-private VFL. Finally, we empirically show that our model
performs well with high levels of privacy.


---

**[126. [2305.12295] Logic-LM: Empowering Large Language Models with Symbolic Solvers for
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

**[127. [2412.10622] A recent evaluation on the performance of LLMs on radiation oncology
  physics using questions of randomly shuffled options](https://arxiv.org/pdf/2412.10622.pdf)** (2025-01-22)

*Peilong Wang, Jason Holmes, Zhengliang Liu, Dequan Chen, Tianming Liu, Jiajian Shen, Wei Liu*

  Purpose: We present an updated study evaluating the performance of large
language models (LLMs) in answering radiation oncology physics questions,
focusing on the recently released models.
  Methods: A set of 100 multiple-choice radiation oncology physics questions,
previously created by a well-experienced physicist, was used for this study.
The answer options of the questions were randomly shuffled to create "new" exam
sets. Five LLMs -- OpenAI o1-preview, GPT-4o, LLaMA 3.1 (405B), Gemini 1.5 Pro,
and Claude 3.5 Sonnet -- with the versions released before September 30, 2024,
were queried using these new exam sets. To evaluate their deductive reasoning
ability, the correct answer options in the questions were replaced with "None
of the above." Then, the explain-first and step-by-step instruction prompts
were used to test if this strategy improved their reasoning ability. The
performance of the LLMs was compared with the answers from medical physicists.
  Results: All models demonstrated expert-level performance on these questions,
with o1-preview even surpassing medical physicists with a majority vote. When
replacing the correct answer options with 'None of the above', all models
exhibited a considerable decline in performance, suggesting room for
improvement. The explain-first and step-by-step instruction prompts helped
enhance the reasoning ability of the LLaMA 3.1 (405B), Gemini 1.5 Pro, and
Claude 3.5 Sonnet models.
  Conclusion: These recently released LLMs demonstrated expert-level
performance in answering radiation oncology physics questions, exhibiting great
potential to assist in radiation oncology physics education and training.


---

**[128. [2401.11108] LLM4Fuzz: Guided Fuzzing of Smart Contracts with Large Language Models](https://arxiv.org/pdf/2401.11108.pdf)** (2024-01-23)

*Chaofan Shou, Jing Liu, Doudou Lu, Koushik Sen*

  As blockchain platforms grow exponentially, millions of lines of smart
contract code are being deployed to manage extensive digital assets. However,
vulnerabilities in this mission-critical code have led to significant
exploitations and asset losses. Thorough automated security analysis of smart
contracts is thus imperative. This paper introduces LLM4Fuzz to optimize
automated smart contract security analysis by leveraging large language models
(LLMs) to intelligently guide and prioritize fuzzing campaigns. While
traditional fuzzing suffers from low efficiency in exploring the vast state
space, LLM4Fuzz employs LLMs to direct fuzzers towards high-value code regions
and input sequences more likely to trigger vulnerabilities. Additionally,
LLM4Fuzz can leverage LLMs to guide fuzzers based on user-defined invariants,
reducing blind exploration overhead. Evaluations of LLM4Fuzz on real-world DeFi
projects show substantial gains in efficiency, coverage, and vulnerability
detection compared to baseline fuzzing. LLM4Fuzz also uncovered five critical
vulnerabilities that can lead to a loss of more than $247k.


---

**[129. [2408.13597] APPATCH: Automated Adaptive Prompting Large Language Models for
  Real-World Software Vulnerability Patching](https://arxiv.org/pdf/2408.13597.pdf)** (2025-04-04)

*Yu Nong, Haoran Yang, Long Cheng, Hongxin Hu, Haipeng Cai*

  Timely and effective vulnerability patching is essential for cybersecurity
defense, for which various approaches have been proposed yet still struggle to
generate valid and correct patches for real-world vulnerabilities. In this
paper, we leverage the power and merits of pre-trained language language models
(LLMs) to enable automated vulnerability patching using no test input/exploit
evidence and without model training/fine-tuning. To elicit LLMs to effectively
reason about vulnerable code behaviors, which is essential for quality patch
generation, we introduce vulnerability semantics reasoning and adaptive
prompting on LLMs and instantiate the methodology as APPATCH, an automated
LLM-based patching system. Our evaluation of APPATCH on 97 zero-day
vulnerabilities and 20 existing vulnerabilities demonstrates its superior
performance to both existing prompting methods and state-of-the-art
non-LLM-based techniques (by up to 28.33% in F1 and 182.26% in recall over the
best baseline). Through APPATCH, we demonstrate what helps for LLM-based
patching and how, as well as discussing what still lacks and why.


---

**[130. [2504.01005] When To Solve, When To Verify: Compute-Optimal Problem Solving and
  Generative Verification for LLM Reasoning](https://arxiv.org/pdf/2504.01005.pdf)** (2025-04-02)

*Nishad Singhi, Hritik Bansal, Arian Hosseini, Aditya Grover, Kai-Wei Chang, Marcus Rohrbach, Anna Rohrbach*

  Scaling test-time compute has emerged as a key strategy for enhancing the
reasoning capabilities of large language models (LLMs), particularly in tasks
like mathematical problem-solving. A traditional approach, Self-Consistency
(SC), generates multiple solutions to a problem and selects the most common
answer via majority voting. Another common method involves scoring each
solution with a reward model (verifier) and choosing the best one. Recent
advancements in Generative Reward Models (GenRM) reframe verification as a
next-token prediction task, enabling inference-time scaling along a new axis.
Specifically, GenRM generates multiple verification chains-of-thought to score
each solution. Under a limited inference budget, this introduces a fundamental
trade-off: should you spend the budget on scaling solutions via SC or generate
fewer solutions and allocate compute to verification via GenRM? To address
this, we evaluate GenRM against SC under a fixed inference budget.
Interestingly, we find that SC is more compute-efficient than GenRM for most
practical inference budgets across diverse models and datasets. For instance,
GenRM first matches SC after consuming up to 8x the inference compute and
requires significantly more compute to outperform it. Furthermore, we derive
inference scaling laws for the GenRM paradigm, revealing that compute-optimal
inference favors scaling solution generation more aggressively than scaling the
number of verifications. Our work provides practical guidance on optimizing
test-time scaling by balancing solution generation and verification. The code
is available at https://github.com/nishadsinghi/sc-genrm-scaling.


---

**[131. [2411.02355] "Give Me BF16 or Give Me Death"? Accuracy-Performance Trade-Offs in LLM
  Quantization](https://arxiv.org/pdf/2411.02355.pdf)** (2025-02-25)

*Eldar Kurtic, Alexandre Marques, Shubhra Pandit, Mark Kurtz, Dan Alistarh*

  Quantization is a powerful tool for accelerating large language model (LLM)
inference, but the accuracy-performance trade-offs across different formats
remain unclear. In this paper, we conduct the most comprehensive empirical
study to date, evaluating FP8, INT8, and INT4 quantization across academic
benchmarks and real-world tasks on the entire Llama-3.1 model family. Through
over 500,000 evaluations, our investigation yields several key findings: (1)
FP8 (W8A8-FP) is effectively lossless across all model scales, (2) well-tuned
INT8 (W8A8-INT) achieves surprisingly low (1-3\%) accuracy degradation, and (3)
INT4 weight-only (W4A16-INT) is more competitive than expected, rivaling 8-bit
quantization. Further, we investigate the optimal quantization format for
different deployments by analyzing inference performance through the popular
vLLM framework. Our analysis provides clear deployment recommendations: W4A16
is the most cost-efficient for synchronous setups, while W8A8 dominates in
asynchronous continuous batching. For mixed workloads, the optimal choice
depends on the specific use case. Our findings offer practical, data-driven
guidelines for deploying quantized LLMs at scale -- ensuring the best balance
between speed, efficiency, and accuracy.


---

**[132. [2407.13928] BiasDPO: Mitigating Bias in Language Models through Direct Preference
  Optimization](https://arxiv.org/pdf/2407.13928.pdf)** (2024-07-22)

*Ahmed Allam*

  Large Language Models (LLMs) have become pivotal in advancing natural
language processing, yet their potential to perpetuate biases poses significant
concerns. This paper introduces a new framework employing Direct Preference
Optimization (DPO) to mitigate gender, racial, and religious biases in
LLM-generated English text. By developing a loss function that favors less
biased over biased completions, our approach cultivates a preference for
respectful and non-discriminatory language in LLMs. We also contribute a
manually designed dataset for training LLMs to recognize and correct biases.
This dataset encompasses a diverse range of prompts paired with both biased and
unbiased completions. Implementing this approach on the Microsoft Phi-2 model,
we demonstrate substantial reductions in biased outputs as our model
outperforms the baseline model on almost all bias benchmarks. Our model also
achieves better performance compared to other open-source models on most
benchmarks. By reducing biases in the language generated by the model, our
study marks a significant step towards developing more ethical and socially
responsible LLMs. We publicly release BiasDPO dataset on HuggingFace.


---

**[133. [2406.13542] Self-play with Execution Feedback: Improving Instruction-following
  Capabilities of Large Language Models](https://arxiv.org/pdf/2406.13542.pdf)** (2024-07-19)

*Guanting Dong, Keming Lu, Chengpeng Li, Tingyu Xia, Bowen Yu, Chang Zhou, Jingren Zhou*

  One core capability of large language models (LLMs) is to follow natural
language instructions. However, the issue of automatically constructing
high-quality training data to enhance the complex instruction-following
abilities of LLMs without manual annotation remains unresolved. In this paper,
we introduce AutoIF, the first scalable and reliable method for automatically
generating instruction-following training data. AutoIF transforms the
validation of instruction-following data quality into code verification,
requiring LLMs to generate instructions, the corresponding code to check the
correctness of the instruction responses, and unit test samples to verify the
code's correctness. Then, execution feedback-based rejection sampling can
generate data for Supervised Fine-Tuning (SFT) and Reinforcement Learning from
Human Feedback (RLHF) training. AutoIF achieves significant improvements across
three training algorithms, SFT, Offline DPO, and Online DPO, when applied to
the top open-source LLMs, Qwen2 and LLaMA3, in self-alignment and
strong-to-weak distillation settings. Our code is publicly available at
https://github.com/QwenLM/AutoIF.


---

**[134. [2410.11720] ATTNChecker: Highly-Optimized Fault Tolerant Attention for Large
  Language Model Training](https://arxiv.org/pdf/2410.11720.pdf)** (2025-01-30)

*Yuhang Liang, Xinyi Li, Jie Ren, Ang Li, Bo Fang, Jieyang Chen*

  Large Language Models (LLMs) have demonstrated remarkable performance in
various natural language processing tasks. However, the training of these
models is computationally intensive and susceptible to faults, particularly in
the attention mechanism, which is a critical component of transformer-based
LLMs. In this paper, we investigate the impact of faults on LLM training,
focusing on INF, NaN, and near-INF values in the computation results with
systematic fault injection experiments. We observe the propagation patterns of
these errors, which can trigger non-trainable states in the model and disrupt
training, forcing the procedure to load from checkpoints. To mitigate the
impact of these faults, we propose ATTNChecker, the first Algorithm-Based Fault
Tolerance (ABFT) technique tailored for the attention mechanism in LLMs.
ATTNChecker is designed based on fault propagation patterns of LLM and
incorporates performance optimization to adapt to both system reliability and
model vulnerability while providing lightweight protection for fast LLM
training. Evaluations on four LLMs show that ATTNChecker incurs on average 7%
overhead on training while detecting and correcting all extreme errors.
Compared with the state-of-the-art checkpoint/restore approach, ATTNChecker
reduces recovery overhead by up to 49x.


---

**[135. [2311.04933] Evaluating Large Language Models in Ophthalmology](https://arxiv.org/pdf/2311.04933.pdf)** (2023-11-10)

*Jason Holmes, Shuyuan Ye, Yiwei Li, Shi-Nan Wu, Zhengliang Liu, Zihao Wu, Jinyu Hu, Huan Zhao, Xi Jiang, Wei Liu, Hong Wei, Jie Zou, Tianming Liu, Yi Shao*

  Purpose: The performance of three different large language models (LLMS)
(GPT-3.5, GPT-4, and PaLM2) in answering ophthalmology professional questions
was evaluated and compared with that of three different professional
populations (medical undergraduates, medical masters, and attending
physicians). Methods: A 100-item ophthalmology single-choice test was
administered to three different LLMs (GPT-3.5, GPT-4, and PaLM2) and three
different professional levels (medical undergraduates, medical masters, and
attending physicians), respectively. The performance of LLM was comprehensively
evaluated and compared with the human group in terms of average score,
stability, and confidence. Results: Each LLM outperformed undergraduates in
general, with GPT-3.5 and PaLM2 being slightly below the master's level, while
GPT-4 showed a level comparable to that of attending physicians. In addition,
GPT-4 showed significantly higher answer stability and confidence than GPT-3.5
and PaLM2. Conclusion: Our study shows that LLM represented by GPT-4 performs
better in the field of ophthalmology. With further improvements, LLM will bring
unexpected benefits in medical education and clinical decision making in the
near future.


---

**[136. [2401.07927] Are self-explanations from Large Language Models faithful?](https://arxiv.org/pdf/2401.07927.pdf)** (2024-05-20)

*Andreas Madsen, Sarath Chandar, Siva Reddy*

  Instruction-tuned Large Language Models (LLMs) excel at many tasks and will
even explain their reasoning, so-called self-explanations. However, convincing
and wrong self-explanations can lead to unsupported confidence in LLMs, thus
increasing risk. Therefore, it's important to measure if self-explanations
truly reflect the model's behavior. Such a measure is called
interpretability-faithfulness and is challenging to perform since the ground
truth is inaccessible, and many LLMs only have an inference API. To address
this, we propose employing self-consistency checks to measure faithfulness. For
example, if an LLM says a set of words is important for making a prediction,
then it should not be able to make its prediction without these words. While
self-consistency checks are a common approach to faithfulness, they have not
previously been successfully applied to LLM self-explanations for
counterfactual, feature attribution, and redaction explanations. Our results
demonstrate that faithfulness is explanation, model, and task-dependent,
showing self-explanations should not be trusted in general. For example, with
sentiment classification, counterfactuals are more faithful for Llama2, feature
attribution for Mistral, and redaction for Falcon 40B.


---

**[137. [2305.06212] Privacy-Preserving Prompt Tuning for Large Language Model Services](https://arxiv.org/pdf/2305.06212.pdf)** (2025-01-14)

*Yansong Li, Zhixing Tan, Yang Liu*

  Prompt tuning provides an efficient way for users to customize Large Language
Models (LLMs) with their private data in the emerging LLM service scenario.
However, the sensitive nature of private data brings the need for privacy
preservation in LLM service customization. Based on prompt tuning, we propose
Privacy-Preserving Prompt Tuning (RAPT), a framework that provides privacy
guarantees for LLM services. \textsc{rapt} adopts a local privacy setting,
allowing users to privatize their data locally with local differential privacy.
As prompt tuning performs poorly when directly trained on privatized data, we
introduce a novel privatized token reconstruction task that is trained jointly
with the downstream task, allowing LLMs to learn better task-dependent
representations. Despite the simplicity of our framework, experiments show that
RAPT achieves competitive performance across tasks while providing privacy
guarantees against adversaries.


---

**[138. [2406.12168] BPO: Staying Close to the Behavior LLM Creates Better Online LLM
  Alignment](https://arxiv.org/pdf/2406.12168.pdf)** (2024-10-23)

*Wenda Xu, Jiachen Li, William Yang Wang, Lei Li*

  Direct alignment from preferences (DAP) has emerged as a promising paradigm
for aligning large language models (LLMs) to human desiderata from
pre-collected, offline preference datasets. While recent studies indicate that
existing offline DAP methods can directly benefit from online training samples,
we highlight the need to develop specific online DAP algorithms to fully
harness the power of online training. Specifically, we identify that the
learned LLM should adhere to the proximity of the behavior LLM, which collects
the training samples. To this end, we propose online Preference Optimization in
proximity to the Behavior LLM (BPO), emphasizing the importance of constructing
a proper trust region for LLM alignment.
  We conduct extensive experiments to validate the effectiveness and
applicability of our approach by integrating it with various DAP methods,
resulting in significant performance improvements across a wide range of tasks
when training with the same amount of preference data. Even when only
introducing one additional data collection phase, our online BPO improves its
offline DAP baseline from 72.0% to 80.2% on TL;DR and from 82.2% to 89.1% on
Anthropic Helpfulness in terms of win rate against human reference text.


---

**[139. [2404.11086] ViLLM-Eval: A Comprehensive Evaluation Suite for Vietnamese Large
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

**[140. [2503.15551] Efficient but Vulnerable: Benchmarking and Defending LLM Batch Prompting
  Attack](https://arxiv.org/pdf/2503.15551.pdf)** (2025-03-21)

*Murong Yue, Ziyu Yao*

  Batch prompting, which combines a batch of multiple queries sharing the same
context in one inference, has emerged as a promising solution to reduce
inference costs. However, our study reveals a significant security
vulnerability in batch prompting: malicious users can inject attack
instructions into a batch, leading to unwanted interference across all queries,
which can result in the inclusion of harmful content, such as phishing links,
or the disruption of logical reasoning. In this paper, we construct
BATCHSAFEBENCH, a comprehensive benchmark comprising 150 attack instructions of
two types and 8k batch instances, to study the batch prompting vulnerability
systematically. Our evaluation of both closed-source and open-weight LLMs
demonstrates that all LLMs are susceptible to batch-prompting attacks. We then
explore multiple defending approaches. While the prompting-based defense shows
limited effectiveness for smaller LLMs, the probing-based approach achieves
about 95% accuracy in detecting attacks. Additionally, we perform a mechanistic
analysis to understand the attack and identify attention heads that are
responsible for it.


---

**[141. [2409.13697] Prompt Baking](https://arxiv.org/pdf/2409.13697.pdf)** (2024-09-24)

*Aman Bhargava, Cameron Witkowski, Alexander Detkov, Matt Thomson*

  Two primary ways to change LLM behavior are prompting and weight updates
(e.g., fine-tuning). Prompting LLMs is simple and effective, specifying the
desired changes explicitly in natural language, whereas weight updates provide
more expressive and permanent behavior changes, specified implicitly via
training on large datasets. We present a technique for "baking" prompts into
the weights of an LLM. Prompt Baking converts a prompt $u$ and initial weights
$\theta$ to a new set of weights $\theta_u$ such that new "baked" LLM behaves
like the original prompted LLM. Mathematically, we minimize the KL divergence
between $P_\theta(\cdot | u)$ and $P_{\theta_u}(\cdot)$, where $P$ is the LLM's
probability distribution over token sequences. Across all our experiments, we
find prompts can be readily baked into weight updates. Baking chain-of-thought
prompts improves zero-shot performance on GSM8K, ASDiv, MBPP, ARC-Easy,
ARC-Challenge, and CommonsenseQA benchmarks. Baking news headlines directly
updates an LLM's knowledge. And baking instructions & personas alleviates
"prompt forgetting" over long sequences. Furthermore, stopping baking early
creates "half-baked" models, continuously scaling prompt strength. Baked models
retain their sensitivity to further prompting and baking, including
re-prompting with the baked-in prompt. Surprisingly, the re-prompted models
yield further performance gains in instruction following, as well as math
reasoning and coding benchmarks. Taking re-prompting and re-baking to the limit
yields a form of iterative self-improvement we call Prompt Pursuit, and
preliminary results on instruction following exhibit dramatic performance
gains. Finally, we discuss implications for AI safety, continuous model
updating, enhancing real-time learning capabilities in LLM-based agents, and
generating more stable AI personas.


---

**[142. [2406.09136] Chain of Preference Optimization: Improving Chain-of-Thought Reasoning
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

**[143. [2305.14483] Language Model Self-improvement by Reinforcement Learning Contemplation](https://arxiv.org/pdf/2305.14483.pdf)** (2023-05-25)

*Jing-Cheng Pang, Pengyuan Wang, Kaiyuan Li, Xiong-Hui Chen, Jiacheng Xu, Zongzhang Zhang, Yang Yu*

  Large Language Models (LLMs) have exhibited remarkable performance across
various natural language processing (NLP) tasks. However, fine-tuning these
models often necessitates substantial supervision, which can be expensive and
time-consuming to obtain. This paper introduces a novel unsupervised method
called LanguageModel Self-Improvement by Reinforcement Learning Contemplation
(SIRLC) that improves LLMs without reliance on external labels. Our approach is
grounded in the observation that it is simpler for language models to assess
text quality than to generate text. Building on this insight, SIRLC assigns
LLMs dual roles as both student and teacher. As a student, the LLM generates
answers to unlabeled questions, while as a teacher, it evaluates the generated
text and assigns scores accordingly. The model parameters are updated using
reinforcement learning to maximize the evaluation score. We demonstrate that
SIRLC can be applied to various NLP tasks, such as reasoning problems, text
generation, and machine translation. Our experiments show that SIRLC
effectively improves LLM performance without external supervision, resulting in
a 5.6% increase in answering accuracy for reasoning tasks and a rise in
BERTScore from 0.82 to 0.86 for translation tasks. Furthermore, SIRLC can be
applied to models of different sizes, showcasing its broad applicability.


---

**[144. [2502.04416] CMoE: Fast Carving of Mixture-of-Experts for Efficient LLM Inference](https://arxiv.org/pdf/2502.04416.pdf)** (2025-02-10)

*Zehua Pei, Lancheng Zou, Hui-Ling Zhen, Xianzhi Yu, Wulong Liu, Sinno Jialin Pan, Mingxuan Yuan, Bei Yu*

  Large language models (LLMs) achieve impressive performance by scaling model
parameters, but this comes with significant inference overhead. Feed-forward
networks (FFNs), which dominate LLM parameters, exhibit high activation
sparsity in hidden neurons. To exploit this, researchers have proposed using a
mixture-of-experts (MoE) architecture, where only a subset of parameters is
activated. However, existing approaches often require extensive training data
and resources, limiting their practicality. We propose CMoE (Carved MoE), a
novel framework to efficiently carve MoE models from dense models. CMoE
achieves remarkable performance through efficient expert grouping and
lightweight adaptation. First, neurons are grouped into shared and routed
experts based on activation rates. Next, we construct a routing mechanism
without training from scratch, incorporating a differentiable routing process
and load balancing. Using modest data, CMoE produces a well-designed, usable
MoE from a 7B dense model within five minutes. With lightweight fine-tuning, it
achieves high-performance recovery in under an hour. We make our code publicly
available at https://github.com/JarvisPei/CMoE.


---

**[145. [2402.09283] Attacks, Defenses and Evaluations for LLM Conversation Safety: A Survey](https://arxiv.org/pdf/2402.09283.pdf)** (2024-03-28)

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

**[146. [2402.16893] The Good and The Bad: Exploring Privacy Issues in Retrieval-Augmented
  Generation (RAG)](https://arxiv.org/pdf/2402.16893.pdf)** (2024-03-03)

*Shenglai Zeng, Jiankun Zhang, Pengfei He, Yue Xing, Yiding Liu, Han Xu, Jie Ren, Shuaiqiang Wang, Dawei Yin, Yi Chang, Jiliang Tang*

  Retrieval-augmented generation (RAG) is a powerful technique to facilitate
language model with proprietary and private data, where data privacy is a
pivotal concern. Whereas extensive research has demonstrated the privacy risks
of large language models (LLMs), the RAG technique could potentially reshape
the inherent behaviors of LLM generation, posing new privacy issues that are
currently under-explored. In this work, we conduct extensive empirical studies
with novel attack methods, which demonstrate the vulnerability of RAG systems
on leaking the private retrieval database. Despite the new risk brought by RAG
on the retrieval data, we further reveal that RAG can mitigate the leakage of
the LLMs' training data. Overall, we provide new insights in this paper for
privacy protection of retrieval-augmented LLMs, which benefit both LLMs and RAG
systems builders. Our code is available at
https://github.com/phycholosogy/RAG-privacy.


---

**[147. [2405.20413] Jailbreaking Large Language Models Against Moderation Guardrails via
  Cipher Characters](https://arxiv.org/pdf/2405.20413.pdf)** (2024-06-03)

*Haibo Jin, Andy Zhou, Joe D. Menke, Haohan Wang*

  Large Language Models (LLMs) are typically harmless but remain vulnerable to
carefully crafted prompts known as ``jailbreaks'', which can bypass protective
measures and induce harmful behavior. Recent advancements in LLMs have
incorporated moderation guardrails that can filter outputs, which trigger
processing errors for certain malicious questions. Existing red-teaming
benchmarks often neglect to include questions that trigger moderation
guardrails, making it difficult to evaluate jailbreak effectiveness. To address
this issue, we introduce JAMBench, a harmful behavior benchmark designed to
trigger and evaluate moderation guardrails. JAMBench involves 160 manually
crafted instructions covering four major risk categories at multiple severity
levels. Furthermore, we propose a jailbreak method, JAM (Jailbreak Against
Moderation), designed to attack moderation guardrails using jailbreak prefixes
to bypass input-level filters and a fine-tuned shadow model functionally
equivalent to the guardrail model to generate cipher characters to bypass
output-level filters. Our extensive experiments on four LLMs demonstrate that
JAM achieves higher jailbreak success ($\sim$ $\times$ 19.88) and lower
filtered-out rates ($\sim$ $\times$ 1/6) than baselines.


---

**[148. [2408.07106] "You still have to study" -- On the Security of LLM generated code](https://arxiv.org/pdf/2408.07106.pdf)** (2024-08-15)

*Stefan Goetz, Andreas Schaad*

  We witness an increasing usage of AI-assistants even for routine (classroom)
programming tasks. However, the code generated on basis of a so called "prompt"
by the programmer does not always meet accepted security standards. On the one
hand, this may be due to lack of best-practice examples in the training data.
On the other hand, the actual quality of the programmers prompt appears to
influence whether generated code contains weaknesses or not. In this paper we
analyse 4 major LLMs with respect to the security of generated code. We do this
on basis of a case study for the Python and Javascript language, using the
MITRE CWE catalogue as the guiding security definition. Our results show that
using different prompting techniques, some LLMs initially generate 65% code
which is deemed insecure by a trained security engineer. On the other hand
almost all analysed LLMs will eventually generate code being close to 100%
secure with increasing manual guidance of a skilled engineer.


---

**[149. [2411.06387] Self-Training Meets Consistency: Improving LLMs' Reasoning with
  Consistency-Driven Rationale Evaluation](https://arxiv.org/pdf/2411.06387.pdf)** (2025-02-07)

*Jaehyeok Lee, Keisuke Sakaguchi, JinYeong Bak*

  Self-training approach for large language models (LLMs) improves reasoning
abilities by training the models on their self-generated rationales. Previous
approaches have labeled rationales that produce correct answers for a given
question as appropriate for training. However, a single measure risks
misjudging rationale quality, leading the models to learn flawed reasoning
patterns. To address this issue, we propose CREST (Consistency-driven Rationale
Evaluation for Self-Training), a self-training framework that further evaluates
each rationale through follow-up questions and leverages this evaluation to
guide its training. Specifically, we introduce two methods: (1) filtering out
rationales that frequently result in incorrect answers on follow-up questions
and (2) preference learning based on mixed preferences from rationale
evaluation results of both original and follow-up questions. Experiments on
three question-answering datasets using open LLMs show that CREST not only
improves the logical robustness and correctness of rationales but also improves
reasoning abilities compared to previous self-training approaches.


---

**[150. [2501.11929] ALoFTRAG: Automatic Local Fine Tuning for Retrieval Augmented Generation](https://arxiv.org/pdf/2501.11929.pdf)** (2025-01-22)

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

**[151. [2410.03608] TICKing All the Boxes: Generated Checklists Improve LLM Evaluation and
  Generation](https://arxiv.org/pdf/2410.03608.pdf)** (2024-10-07)

*Jonathan Cook, Tim Rocktäschel, Jakob Foerster, Dennis Aumiller, Alex Wang*

  Given the widespread adoption and usage of Large Language Models (LLMs), it
is crucial to have flexible and interpretable evaluations of their
instruction-following ability. Preference judgments between model outputs have
become the de facto evaluation standard, despite distilling complex,
multi-faceted preferences into a single ranking. Furthermore, as human
annotation is slow and costly, LLMs are increasingly used to make these
judgments, at the expense of reliability and interpretability. In this work, we
propose TICK (Targeted Instruct-evaluation with ChecKlists), a fully automated,
interpretable evaluation protocol that structures evaluations with
LLM-generated, instruction-specific checklists. We first show that, given an
instruction, LLMs can reliably produce high-quality, tailored evaluation
checklists that decompose the instruction into a series of YES/NO questions.
Each question asks whether a candidate response meets a specific requirement of
the instruction. We demonstrate that using TICK leads to a significant increase
(46.4% $\to$ 52.2%) in the frequency of exact agreements between LLM judgements
and human preferences, as compared to having an LLM directly score an output.
We then show that STICK (Self-TICK) can be used to improve generation quality
across multiple benchmarks via self-refinement and Best-of-N selection. STICK
self-refinement on LiveBench reasoning tasks leads to an absolute gain of
$+$7.8%, whilst Best-of-N selection with STICK attains $+$6.3% absolute
improvement on the real-world instruction dataset, WildBench. In light of this,
structured, multi-faceted self-improvement is shown to be a promising way to
further advance LLM capabilities. Finally, by providing LLM-generated
checklists to human evaluators tasked with directly scoring LLM responses to
WildBench instructions, we notably increase inter-annotator agreement (0.194
$\to$ 0.256).


---

**[152. [2406.13990] Inference-Time Decontamination: Reusing Leaked Benchmarks for Large
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

**[153. [2502.13416] Detecting LLM Fact-conflicting Hallucinations Enhanced by
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

**[154. [2410.18528] PRACT: Optimizing Principled Reasoning and Acting of LLM Agent](https://arxiv.org/pdf/2410.18528.pdf)** (2024-10-25)

*Zhiwei Liu, Weiran Yao, Jianguo Zhang, Rithesh Murthy, Liangwei Yang, Zuxin Liu, Tian Lan, Ming Zhu, Juntao Tan, Shirley Kokane, Thai Hoang, Juan Carlos Niebles, Shelby Heinecke, Huan Wang, Silvio Savarese, Caiming Xiong*

  We introduce the Principled Reasoning and Acting (PRAct) framework, a novel
method for learning and enforcing action principles from trajectory data.
Central to our approach is the use of text gradients from a reflection and
optimization engine to derive these action principles. To adapt action
principles to specific task requirements, we propose a new optimization
framework, Reflective Principle Optimization (RPO). After execution, RPO
employs a reflector to critique current action principles and an optimizer to
update them accordingly. We develop the RPO framework under two scenarios:
Reward-RPO, which uses environmental rewards for reflection, and Self-RPO,
which conducts self-reflection without external rewards. Additionally, two RPO
methods, RPO-Traj and RPO-Batch, is introduced to adapt to different settings.
Experimental results across four environments demonstrate that the PRAct agent,
leveraging the RPO framework, effectively learns and applies action principles
to enhance performance.


---

**[155. [2503.04693] UIPE: Enhancing LLM Unlearning by Removing Knowledge Related to
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

**[156. [2502.07340] Aligning Large Language Models to Follow Instructions and Hallucinate
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

**[157. [2405.04032] Locally Differentially Private In-Context Learning](https://arxiv.org/pdf/2405.04032.pdf)** (2024-05-09)

*Chunyan Zheng, Keke Sun, Wenhao Zhao, Haibo Zhou, Lixin Jiang, Shaoyang Song, Chunlai Zhou*

  Large pretrained language models (LLMs) have shown surprising In-Context
Learning (ICL) ability. An important application in deploying large language
models is to augment LLMs with a private database for some specific task. The
main problem with this promising commercial use is that LLMs have been shown to
memorize their training data and their prompt data are vulnerable to membership
inference attacks (MIA) and prompt leaking attacks. In order to deal with this
problem, we treat LLMs as untrusted in privacy and propose a locally
differentially private framework of in-context learning(LDP-ICL) in the
settings where labels are sensitive. Considering the mechanisms of in-context
learning in Transformers by gradient descent, we provide an analysis of the
trade-off between privacy and utility in such LDP-ICL for classification.
Moreover, we apply LDP-ICL to the discrete distribution estimation problem. In
the end, we perform several experiments to demonstrate our analysis results.


---

**[158. [2502.16691] Toward Responsible Federated Large Language Models: Leveraging a Safety
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

**[159. [2311.05553] Removing RLHF Protections in GPT-4 via Fine-Tuning](https://arxiv.org/pdf/2311.05553.pdf)** (2024-04-09)

*Qiusi Zhan, Richard Fang, Rohan Bindu, Akul Gupta, Tatsunori Hashimoto, Daniel Kang*

  As large language models (LLMs) have increased in their capabilities, so does
their potential for dual use. To reduce harmful outputs, produces and vendors
of LLMs have used reinforcement learning with human feedback (RLHF). In tandem,
LLM vendors have been increasingly enabling fine-tuning of their most powerful
models. However, concurrent work has shown that fine-tuning can remove RLHF
protections. We may expect that the most powerful models currently available
(GPT-4) are less susceptible to fine-tuning attacks. In this work, we show the
contrary: fine-tuning allows attackers to remove RLHF protections with as few
as 340 examples and a 95% success rate. These training examples can be
automatically generated with weaker models. We further show that removing RLHF
protections does not decrease usefulness on non-censored outputs, providing
evidence that our fine-tuning strategy does not decrease usefulness despite
using weaker models to generate training data. Our results show the need for
further research on protections on LLMs.


---

**[160. [2402.05624] Efficient Models for the Detection of Hate, Abuse and Profanity](https://arxiv.org/pdf/2402.05624.pdf)** (2024-02-09)

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

**[161. [2311.07838] LLatrieval: LLM-Verified Retrieval for Verifiable Generation](https://arxiv.org/pdf/2311.07838.pdf)** (2024-03-28)

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

**[162. [2404.17790] Continual Pre-Training for Cross-Lingual LLM Adaptation: Enhancing
  Japanese Language Capabilities](https://arxiv.org/pdf/2404.17790.pdf)** (2024-04-30)

*Kazuki Fujii, Taishi Nakamura, Mengsay Loem, Hiroki Iida, Masanari Ohi, Kakeru Hattori, Hirai Shota, Sakae Mizuki, Rio Yokota, Naoaki Okazaki*

  Cross-lingual continual pre-training of large language models (LLMs)
initially trained on English corpus allows us to leverage the vast amount of
English language resources and reduce the pre-training cost. In this study, we
constructed Swallow, an LLM with enhanced Japanese capability, by extending the
vocabulary of Llama 2 to include Japanese characters and conducting continual
pre-training on a large Japanese web corpus. Experimental results confirmed
that the performance on Japanese tasks drastically improved through continual
pre-training, and the performance monotonically increased with the amount of
training data up to 100B tokens. Consequently, Swallow achieved superior
performance compared to other LLMs that were trained from scratch in English
and Japanese. An analysis of the effects of continual pre-training revealed
that it was particularly effective for Japanese question answering tasks.
Furthermore, to elucidate effective methodologies for cross-lingual continual
pre-training from English to Japanese, we investigated the impact of vocabulary
expansion and the effectiveness of incorporating parallel corpora. The results
showed that the efficiency gained through vocabulary expansion had no negative
impact on performance, except for the summarization task, and that the combined
use of parallel corpora enhanced translation ability.


---

**[163. [2308.09267] GraphReason: Enhancing Reasoning Capabilities of Large Language Models
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

**[164. [2501.18663] Joint Optimization of Prompt Security and System Performance in
  Edge-Cloud LLM Systems](https://arxiv.org/pdf/2501.18663.pdf)** (2025-02-03)

*Haiyang Huang, Tianhui Meng, Weijia Jia*

  Large language models (LLMs) have significantly facilitated human life, and
prompt engineering has improved the efficiency of these models. However, recent
years have witnessed a rise in prompt engineering-empowered attacks, leading to
issues such as privacy leaks, increased latency, and system resource wastage.
Though safety fine-tuning based methods with Reinforcement Learning from Human
Feedback (RLHF) are proposed to align the LLMs, existing security mechanisms
fail to cope with fickle prompt attacks, highlighting the necessity of
performing security detection on prompts. In this paper, we jointly consider
prompt security, service latency, and system resource optimization in
Edge-Cloud LLM (EC-LLM) systems under various prompt attacks. To enhance prompt
security, a vector-database-enabled lightweight attack detector is proposed. We
formalize the problem of joint prompt detection, latency, and resource
optimization into a multi-stage dynamic Bayesian game model. The equilibrium
strategy is determined by predicting the number of malicious tasks and updating
beliefs at each stage through Bayesian updates. The proposed scheme is
evaluated on a real implemented EC-LLM system, and the results demonstrate that
our approach offers enhanced security, reduces the service latency for benign
users, and decreases system resource consumption compared to state-of-the-art
algorithms.


---

**[165. [2504.13425] Secure Multifaceted-RAG for Enterprise: Hybrid Knowledge Retrieval with
  Security Filtering](https://arxiv.org/pdf/2504.13425.pdf)** (2025-04-21)

*Grace Byun, Shinsun Lee, Nayoung Choi, Jinho Choi*

  Existing Retrieval-Augmented Generation (RAG) systems face challenges in
enterprise settings due to limited retrieval scope and data security risks.
When relevant internal documents are unavailable, the system struggles to
generate accurate and complete responses. Additionally, using closed-source
Large Language Models (LLMs) raises concerns about exposing proprietary
information. To address these issues, we propose the Secure Multifaceted-RAG
(SecMulti-RAG) framework, which retrieves not only from internal documents but
also from two supplementary sources: pre-generated expert knowledge for
anticipated queries and on-demand external LLM-generated knowledge. To mitigate
security risks, we adopt a local open-source generator and selectively utilize
external LLMs only when prompts are deemed safe by a filtering mechanism. This
approach enhances completeness, prevents data leakage, and reduces costs. In
our evaluation on a report generation task in the automotive industry,
SecMulti-RAG significantly outperforms traditional RAG - achieving 79.3 to 91.9
percent win rates across correctness, richness, and helpfulness in LLM-based
evaluation, and 56.3 to 70.4 percent in human evaluation. This highlights
SecMulti-RAG as a practical and secure solution for enterprise RAG.


---

**[166. [2410.07471] SEAL: Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection](https://arxiv.org/pdf/2410.07471.pdf)** (2024-10-14)

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

**[167. [2308.12066] Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable
  Mixture-of-Expert Inference](https://arxiv.org/pdf/2308.12066.pdf)** (2024-04-30)

*Ranggi Hwang, Jianyu Wei, Shijie Cao, Changho Hwang, Xiaohu Tang, Ting Cao, Mao Yang*

  Large language models (LLMs) based on transformers have made significant
strides in recent years, the success of which is driven by scaling up their
model size. Despite their high algorithmic performance, the computational and
memory requirements of LLMs present unprecedented challenges. To tackle the
high compute requirements of LLMs, the Mixture-of-Experts (MoE) architecture
was introduced which is able to scale its model size without proportionally
scaling up its computational requirements. Unfortunately, MoE's high memory
demands and dynamic activation of sparse experts restrict its applicability to
real-world problems. Previous solutions that offload MoE's memory-hungry expert
parameters to CPU memory fall short because the latency to migrate activated
experts from CPU to GPU incurs high performance overhead. Our proposed
Pre-gated MoE system effectively tackles the compute and memory challenges of
conventional MoE architectures using our algorithm-system co-design. Pre-gated
MoE employs our novel pre-gating function which alleviates the dynamic nature
of sparse expert activation, allowing our proposed system to address the large
memory footprint of MoEs while also achieving high performance. We demonstrate
that Pre-gated MoE is able to improve performance, reduce GPU memory
consumption, while also maintaining the same level of model quality. These
features allow our Pre-gated MoE system to cost-effectively deploy large-scale
LLMs using just a single GPU with high performance.


---

**[168. [2406.06571] SUBLLM: A Novel Efficient Architecture with Token Sequence Subsampling
  for LLM](https://arxiv.org/pdf/2406.06571.pdf)** (2024-08-26)

*Quandong Wang, Yuxuan Yuan, Xiaoyu Yang, Ruike Zhang, Kang Zhao, Wei Liu, Jian Luan, Daniel Povey, Bin Wang*

  While Large Language Models (LLMs) have achieved remarkable success in
various fields, the efficiency of training and inference remains a major
challenge. To address this issue, we propose SUBLLM, short for
Subsampling-Upsampling-Bypass Large Language Model, an innovative architecture
that extends the core decoder-only framework by incorporating subsampling,
upsampling, and bypass modules. The subsampling modules are responsible for
shortening the sequence, while the upsampling modules restore the sequence
length, and the bypass modules enhance convergence. In comparison to LLaMA, the
proposed SUBLLM exhibits significant enhancements in both training and
inference speeds as well as memory usage, while maintaining competitive
few-shot performance. During training, SUBLLM increases speeds by 26% and cuts
memory by 10GB per GPU. In inference, it boosts speeds by up to 37% and reduces
memory by 1GB per GPU. The training and inference speeds can be enhanced by 34%
and 52% respectively when the context window is expanded to 8192. Our code is
available at https://github.com/XiaoMi/subllm.


---

**[169. [2503.10881] SCE: Scalable Consistency Ensembles Make Blackbox Large Language Model
  Generation More Reliable](https://arxiv.org/pdf/2503.10881.pdf)** (2025-03-17)

*Jiaxin Zhang, Zhuohang Li, Wendi Cui, Kamalika Das, Bradley malin, Sricharan Kumar*

  Large language models (LLMs) have demonstrated remarkable performance, yet
their diverse strengths and weaknesses prevent any single LLM from achieving
dominance across all tasks. Ensembling multiple LLMs is a promising approach to
generate reliable responses but conventional ensembling frameworks suffer from
high computational overheads. This work introduces Scalable Consistency
Ensemble (SCE), an efficient framework for ensembling LLMs by prompting
consistent outputs. The SCE framework systematically evaluates and integrates
outputs to produce a cohesive result through two core components: SCE-CHECK, a
mechanism that gauges the consistency between response pairs via semantic
equivalence; and SCE-FUSION, which adeptly merges the highest-ranked consistent
responses from SCE-CHECK, to optimize collective strengths and mitigating
potential weaknesses. To improve the scalability with multiple inference
queries, we further propose ``{You Only Prompt Once}'' (YOPO), a novel
technique that reduces the inference complexity of pairwise comparison from
quadratic to constant time. We perform extensive empirical evaluations on
diverse benchmark datasets to demonstrate \methodName's effectiveness. Notably,
the \saccheckcomponent outperforms conventional baselines with enhanced
performance and a significant reduction in computational overhead.


---

**[170. [2305.11759] Controlling the Extraction of Memorized Data from Large Language Models
  via Prompt-Tuning](https://arxiv.org/pdf/2305.11759.pdf)** (2023-05-22)

*Mustafa Safa Ozdayi, Charith Peris, Jack FitzGerald, Christophe Dupuy, Jimit Majmudar, Haidar Khan, Rahil Parikh, Rahul Gupta*

  Large Language Models (LLMs) are known to memorize significant portions of
their training data. Parts of this memorized content have been shown to be
extractable by simply querying the model, which poses a privacy risk. We
present a novel approach which uses prompt-tuning to control the extraction
rates of memorized content in LLMs. We present two prompt training strategies
to increase and decrease extraction rates, which correspond to an attack and a
defense, respectively. We demonstrate the effectiveness of our techniques by
using models from the GPT-Neo family on a public benchmark. For the 1.3B
parameter GPT-Neo model, our attack yields a 9.3 percentage point increase in
extraction rate compared to our baseline. Our defense can be tuned to achieve
different privacy-utility trade-offs by a user-specified hyperparameter. We
achieve an extraction rate reduction of up to 97.7% relative to our baseline,
with a perplexity increase of 16.9%.


---

**[171. [2411.09689] LLM Hallucination Reasoning with Zero-shot Knowledge Test](https://arxiv.org/pdf/2411.09689.pdf)** (2024-11-15)

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

**[172. [2503.03104] RVAFM: Re-parameterizing Vertical Attention Fusion Module for
  Handwritten Paragraph Text Recognition](https://arxiv.org/pdf/2503.03104.pdf)** (2025-03-06)

*Jinhui Zheng, Zhiquan Liu, Yain-Whar Si, Jianqing Li, Xinyuan Zhang, Xiaofan Li, Haozhi Huang, Xueyuan Gong*

  Handwritten Paragraph Text Recognition (HPTR) is a challenging task in
Computer Vision, requiring the transformation of a paragraph text image, rich
in handwritten text, into text encoding sequences. One of the most advanced
models for this task is Vertical Attention Network (VAN), which utilizes a
Vertical Attention Module (VAM) to implicitly segment paragraph text images
into text lines, thereby reducing the difficulty of the recognition task.
However, from a network structure perspective, VAM is a single-branch module,
which is less effective in learning compared to multi-branch modules. In this
paper, we propose a new module, named Re-parameterizing Vertical Attention
Fusion Module (RVAFM), which incorporates structural re-parameterization
techniques. RVAFM decouples the structure of the module during training and
inference stages. During training, it uses a multi-branch structure for more
effective learning, and during inference, it uses a single-branch structure for
faster processing. The features learned by the multi-branch structure are fused
into the single-branch structure through a special fusion method named
Re-parameterization Fusion (RF) without any loss of information. As a result,
we achieve a Character Error Rate (CER) of 4.44% and a Word Error Rate (WER) of
14.37% on the IAM paragraph-level test set. Additionally, the inference speed
is slightly faster than VAN.


---

**[173. [2407.16833] Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive
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

**[174. [2406.05644] How Alignment and Jailbreak Work: Explain LLM Safety through
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

**[175. [2412.06090] Trust No AI: Prompt Injection Along The CIA Security Triad](https://arxiv.org/pdf/2412.06090.pdf)** (2024-12-10)

*Independent Researcher, Embrace The Red  Johann Rehberger*

  The CIA security triad - Confidentiality, Integrity, and Availability - is a
cornerstone of data and cybersecurity. With the emergence of large language
model (LLM) applications, a new class of threat, known as prompt injection, was
first identified in 2022. Since then, numerous real-world vulnerabilities and
exploits have been documented in production LLM systems, including those from
leading vendors like OpenAI, Microsoft, Anthropic and Google. This paper
compiles real-world exploits and proof-of concept examples, based on the
research conducted and publicly documented by the author, demonstrating how
prompt injection undermines the CIA triad and poses ongoing risks to
cybersecurity and AI systems at large.


---

**[176. [2504.11816] Cost-Efficient LLM Serving in the Cloud: VM Selection with KV Cache
  Offloading](https://arxiv.org/pdf/2504.11816.pdf)** (2025-04-17)

*Kihyun Kim, Jinwoo Kim, Hyunsun Chung, Myung-Hoon Cha, Hong-Yeon Kim, Youngjae Kim*

  LLM inference is essential for applications like text summarization,
translation, and data analysis, but the high cost of GPU instances from Cloud
Service Providers (CSPs) like AWS is a major burden. This paper proposes
InferSave, a cost-efficient VM selection framework for cloud based LLM
inference. InferSave optimizes KV cache offloading based on Service Level
Objectives (SLOs) and workload charac teristics, estimating GPU memory needs,
and recommending cost-effective VM instances. Additionally, the Compute Time
Calibration Function (CTCF) improves instance selection accuracy by adjusting
for discrepancies between theoretical and actual GPU performance. Experiments
on AWS GPU instances show that selecting lower-cost instances without KV cache
offloading improves cost efficiency by up to 73.7% for online workloads, while
KV cache offloading saves up to 20.19% for offline workloads.


---

**[177. [2410.13153] Better to Ask in English: Evaluation of Large Language Models on
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

**[178. [2411.02829] CE-CoLLM: Efficient and Adaptive Large Language Models Through
  Cloud-Edge Collaboration](https://arxiv.org/pdf/2411.02829.pdf)** (2024-11-06)

*Hongpeng Jin, Yanzhao Wu*

  Large Language Models (LLMs) have achieved remarkable success in serving
end-users with human-like intelligence. However, LLMs demand high computational
resources, making it challenging to deploy them to satisfy various performance
objectives, such as meeting the resource constraints on edge devices close to
end-users or achieving high accuracy with ample resources. In this paper, we
introduce CE-CoLLM, a novel cloud-edge collaboration framework that supports
efficient and adaptive LLM inference for end-users at the edge with two modes,
(1) low-latency edge standalone inference and (2) highly accurate cloud-edge
collaborative inference. First, we show that the inherent high communication
costs for transmitting LLM contextual information between the edge and cloud
dominate the overall latency, making it inefficient and costly to deploy LLMs
using cloud-edge collaboration. Second, we propose several critical techniques
to address this challenge, including early-exit mechanism, cloud context
manager, and quantization in cloud-edge collaboration to enable not only
low-latency standalone edge inference but also efficient and adaptive
cloud-edge collaborative inference for LLMs. Third, we perform comprehensive
experimental analysis, which demonstrates that CE-CoLLM significantly reduces
inference time by up to 13.81% and cloud computation costs by up to 84.55%
compared to the popular cloud-based LLM deployment, while maintaining
comparable model accuracy. The proposed approach effectively shifts the
computational load to the edge, reduces the communication overhead, scales
efficiently with multiple edge clients, and provides reliable LLM deployment
using cloud-edge collaboration.


---

**[179. [2411.14502] Global Challenge for Safe and Secure LLMs Track 1](https://arxiv.org/pdf/2411.14502.pdf)** (2024-11-25)

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

**[180. [2407.12865] GRAD-SUM: Leveraging Gradient Summarization for Optimal Prompt
  Engineering](https://arxiv.org/pdf/2407.12865.pdf)** (2024-07-19)

*Derek Austin, Elliott Chartock*

  Prompt engineering for large language models (LLMs) is often a manual
time-intensive process that involves generating, evaluating, and refining
prompts iteratively to ensure high-quality outputs. While there has been work
on automating prompt engineering, the solutions generally are either tuned to
specific tasks with given answers or are quite costly. We introduce GRAD-SUM, a
scalable and flexible method for automatic prompt engineering that builds on
gradient-based optimization techniques. Our approach incorporates user-defined
task descriptions and evaluation criteria, and features a novel gradient
summarization module to generalize feedback effectively. Our results
demonstrate that GRAD-SUM consistently outperforms existing methods across
various benchmarks, highlighting its versatility and effectiveness in automatic
prompt optimization.


---

**[181. [2311.16293] FHEmem: A Processing In-Memory Accelerator for Fully Homomorphic
  Encryption](https://arxiv.org/pdf/2311.16293.pdf)** (2023-11-29)

*Minxuan Zhou, Yujin Nam, Pranav Gangwar, Weihong Xu, Arpan Dutta, Kartikeyan Subramanyam, Chris Wilkerson, Rosario Cammarota, Saransh Gupta, Tajana Rosing*

  Fully Homomorphic Encryption (FHE) is a technique that allows arbitrary
computations to be performed on encrypted data without the need for decryption,
making it ideal for securing many emerging applications. However, FHE
computation is significantly slower than computation on plain data due to the
increase in data size after encryption. Processing In-Memory (PIM) is a
promising technology that can accelerate data-intensive workloads with
extensive parallelism. However, FHE is challenging for PIM acceleration due to
the long-bitwidth multiplications and complex data movements involved. We
propose a PIM-based FHE accelerator, FHEmem, which exploits a novel processing
in-memory architecture to achieve high-throughput and efficient acceleration
for FHE. We propose an optimized end-to-end processing flow, from low-level
hardware processing to high-level application mapping, that fully exploits the
high throughput of FHEmem hardware. Our evaluation shows FHEmem achieves
significant speedup and efficiency improvement over state-of-the-art FHE
accelerators.


---

**[182. [2411.18797] UOE: Unlearning One Expert Is Enough For Mixture-of-experts LLMS](https://arxiv.org/pdf/2411.18797.pdf)** (2024-12-02)

*Haomin Zhuang, Yihua Zhang, Kehan Guo, Jinghan Jia, Gaowen Liu, Sijia Liu, Xiangliang Zhang*

  Recent advancements in large language model (LLM) unlearning have shown
remarkable success in removing unwanted data-model influences while preserving
the model's utility for legitimate knowledge. However, despite these strides,
sparse Mixture-of-Experts (MoE) LLMs--a key subset of the LLM family--have
received little attention and remain largely unexplored in the context of
unlearning. As MoE LLMs are celebrated for their exceptional performance and
highly efficient inference processes, we ask: How can unlearning be performed
effectively and efficiently on MoE LLMs? And will traditional unlearning
methods be applicable to MoE architectures? Our pilot study shows that the
dynamic routing nature of MoE LLMs introduces unique challenges, leading to
substantial utility drops when existing unlearning methods are applied.
Specifically, unlearning disrupts the router's expert selection, causing
significant selection shift from the most unlearning target-related experts to
irrelevant ones. As a result, more experts than necessary are affected, leading
to excessive forgetting and loss of control over which knowledge is erased. To
address this, we propose a novel single-expert unlearning framework, referred
to as UOE, for MoE LLMs. Through expert attribution, unlearning is concentrated
on the most actively engaged expert for the specified knowledge. Concurrently,
an anchor loss is applied to the router to stabilize the active state of this
targeted expert, ensuring focused and controlled unlearning that preserves
model utility. The proposed UOE framework is also compatible with various
unlearning algorithms. Extensive experiments demonstrate that UOE enhances both
forget quality up to 5% and model utility by 35% on MoE LLMs across various
benchmarks, LLM architectures, while only unlearning 0.06% of the model
parameters.


---

**[183. [2406.16201] Blind Baselines Beat Membership Inference Attacks for Foundation Models](https://arxiv.org/pdf/2406.16201.pdf)** (2025-04-01)

*Debeshee Das, Jie Zhang, Florian Tramèr*

  Membership inference (MI) attacks try to determine if a data sample was used
to train a machine learning model. For foundation models trained on unknown Web
data, MI attacks are often used to detect copyrighted training materials,
measure test set contamination, or audit machine unlearning. Unfortunately, we
find that evaluations of MI attacks for foundation models are flawed, because
they sample members and non-members from different distributions. For 8
published MI evaluation datasets, we show that blind attacks -- that
distinguish the member and non-member distributions without looking at any
trained model -- outperform state-of-the-art MI attacks. Existing evaluations
thus tell us nothing about membership leakage of a foundation model's training
data.


---

**[184. [2210.15042] Privately Fine-Tuning Large Language Models with Differential Privacy](https://arxiv.org/pdf/2210.15042.pdf)** (2023-05-02)

*Rouzbeh Behnia, Mohamamdreza Ebrahimi, Jason Pacheco, Balaji Padmanabhan*

  Pre-trained Large Language Models (LLMs) are an integral part of modern AI
that have led to breakthrough performances in complex AI tasks. Major AI
companies with expensive infrastructures are able to develop and train these
large models with billions and millions of parameters from scratch. Third
parties, researchers, and practitioners are increasingly adopting these
pre-trained models and fine-tuning them on their private data to accomplish
their downstream AI tasks. However, it has been shown that an adversary can
extract/reconstruct the exact training samples from these LLMs, which can lead
to revealing personally identifiable information. The issue has raised deep
concerns about the privacy of LLMs. Differential privacy (DP) provides a
rigorous framework that allows adding noise in the process of training or
fine-tuning LLMs such that extracting the training data becomes infeasible
(i.e., with a cryptographically small success probability). While the
theoretical privacy guarantees offered in most extant studies assume learning
models from scratch through many training iterations in an asymptotic setting,
this assumption does not hold in fine-tuning scenarios in which the number of
training iterations is significantly smaller. To address the gap, we present
\ewtune, a DP framework for fine-tuning LLMs based on Edgeworth accountant with
finite-sample privacy guarantees. Our results across four well-established
natural language understanding (NLU) tasks show that while \ewtune~adds privacy
guarantees to LLM fine-tuning process, it directly contributes to decreasing
the induced noise to up to 5.6\% and improves the state-of-the-art LLMs
performance by up to 1.1\% across all NLU tasks. We have open-sourced our
implementations for wide adoption and public testing purposes.


---

**[185. [2503.01539] Pragmatic Inference Chain (PIC) Improving LLMs' Reasoning of Authentic
  Implicit Toxic Language](https://arxiv.org/pdf/2503.01539.pdf)** (2025-03-04)

*Xi Chen, Shuo Wang*

  The rapid development of large language models (LLMs) gives rise to ethical
concerns about their performance, while opening new avenues for developing
toxic language detection techniques. However, LLMs' unethical output and their
capability of detecting toxicity have primarily been tested on language data
that do not demand complex meaning inference, such as the biased associations
of 'he' with programmer and 'she' with household. Nowadays toxic language
adopts a much more creative range of implicit forms, thanks to advanced
censorship. In this study, we collect authentic toxic interactions that evade
online censorship and that are verified by human annotators as inference
intensive. To evaluate and improve LLMs' reasoning of the authentic implicit
toxic language, we propose a new prompting method, Pragmatic Inference Chain
(PIC), drawn on interdisciplinary findings from cognitive science and
linguistics. The PIC prompting significantly improves the success rate of
GPT-4o, Llama-3.1-70B-Instruct, and DeepSeek-v2.5 in identifying implicit toxic
language, compared to both direct prompting and Chain-of-Thought. In addition,
it also facilitates the models to produce more explicit and coherent reasoning
processes, hence can potentially be generalized to other inference-intensive
tasks, e.g., understanding humour and metaphors.


---

**[186. [2409.00222] Can Large Language Models Address Open-Target Stance Detection?](https://arxiv.org/pdf/2409.00222.pdf)** (2024-12-18)

*Abu Ubaida Akash, Ahmed Fahmy, Amine Trabelsi*

  Stance detection (SD) identifies the text position towards a target,
typically labeled as favor, against, or none. We introduce Open-Target Stance
Detection (OTSD), the most realistic task where targets are neither seen during
training nor provided as input. We evaluate Large Language Models (LLMs) from
GPT, Gemini, Llama, and Mistral families, comparing their performance to the
only existing work, Target-Stance Extraction (TSE), which benefits from
predefined targets. Unlike TSE, OTSD removes the dependency of a predefined
list, making target generation and evaluation more challenging. We also provide
a metric for evaluating target quality that correlates well with human
judgment. Our experiments reveal that LLMs outperform TSE in target generation,
both when the real target is explicitly and not explicitly mentioned in the
text. Similarly, LLMs overall surpass TSE in stance detection for both explicit
and non-explicit cases. However, LLMs struggle in both target generation and
stance detection when the target is not explicit.


---

**[187. [2408.02964] Accuracy and Consistency of LLMs in the Registered Dietitian Exam: The
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

**[188. [2402.11989] Privacy-Preserving Low-Rank Adaptation against Membership Inference
  Attacks for Latent Diffusion Models](https://arxiv.org/pdf/2402.11989.pdf)** (2024-12-17)

*Zihao Luo, Xilie Xu, Feng Liu, Yun Sing Koh, Di Wang, Jingfeng Zhang*

  Low-rank adaptation (LoRA) is an efficient strategy for adapting latent
diffusion models (LDMs) on a private dataset to generate specific images by
minimizing the adaptation loss. However, the LoRA-adapted LDMs are vulnerable
to membership inference (MI) attacks that can judge whether a particular data
point belongs to the private dataset, thus leading to the privacy leakage. To
defend against MI attacks, we first propose a straightforward solution:
Membership-Privacy-preserving LoRA (MP-LoRA). MP-LoRA is formulated as a
min-max optimization problem where a proxy attack model is trained by
maximizing its MI gain while the LDM is adapted by minimizing the sum of the
adaptation loss and the MI gain of the proxy attack model. However, we
empirically find that MP-LoRA has the issue of unstable optimization, and
theoretically analyze that the potential reason is the unconstrained local
smoothness, which impedes the privacy-preserving adaptation. To mitigate this
issue, we further propose a Stable Membership-Privacy-preserving LoRA
(SMP-LoRA) that adapts the LDM by minimizing the ratio of the adaptation loss
to the MI gain. Besides, we theoretically prove that the local smoothness of
SMP-LoRA can be constrained by the gradient norm, leading to improved
convergence. Our experimental results corroborate that SMP-LoRA can indeed
defend against MI attacks and generate high-quality images. Our Code is
available at \url{https://github.com/WilliamLUO0/StablePrivateLoRA}.


---

**[189. [2410.05331] Taylor Unswift: Secured Weight Release for Large Language Models via
  Taylor Expansion](https://arxiv.org/pdf/2410.05331.pdf)** (2025-03-12)

*Guanchu Wang, Yu-Neng Chuang, Ruixiang Tang, Shaochen Zhong, Jiayi Yuan, Hongye Jin, Zirui Liu, Vipin Chaudhary, Shuai Xu, James Caverlee, Xia Hu*

  Ensuring the security of released large language models (LLMs) poses a
significant dilemma, as existing mechanisms either compromise ownership rights
or raise data privacy concerns. To address this dilemma, we introduce TaylorMLP
to protect the ownership of released LLMs and prevent their abuse.
Specifically, TaylorMLP preserves the ownership of LLMs by transforming the
weights of LLMs into parameters of Taylor-series. Instead of releasing the
original weights, developers can release the Taylor-series parameters with
users, thereby ensuring the security of LLMs. Moreover, TaylorMLP can prevent
abuse of LLMs by adjusting the generation speed. It can induce low-speed token
generation for the protected LLMs by increasing the terms in the Taylor-series.
This intentional delay helps LLM developers prevent potential large-scale
unauthorized uses of their models. Empirical experiments across five datasets
and three LLM architectures demonstrate that TaylorMLP induces over 4x increase
in latency, producing the tokens precisely matched with original LLMs.
Subsequent defensive experiments further confirm that TaylorMLP effectively
prevents users from reconstructing the weight values based on downstream
datasets.


---

**[190. [2404.01135] Enhancing Reasoning Capacity of SLM using Cognitive Enhancement](https://arxiv.org/pdf/2404.01135.pdf)** (2024-04-02)

*Jonathan Pan, Swee Liang Wong, Xin Wei Chia, Yidi Yuan*

  Large Language Models (LLMs) have been applied to automate cyber security
activities and processes including cyber investigation and digital forensics.
However, the use of such models for cyber investigation and digital forensics
should address accountability and security considerations. Accountability
ensures models have the means to provide explainable reasonings and outcomes.
This information can be extracted through explicit prompt requests. For
security considerations, it is crucial to address privacy and confidentiality
of the involved data during data processing as well. One approach to deal with
this consideration is to have the data processed locally using a local instance
of the model. Due to limitations of locally available resources, namely memory
and GPU capacities, a Smaller Large Language Model (SLM) will typically be
used. These SLMs have significantly fewer parameters compared to the LLMs.
However, such size reductions have notable performance reduction, especially
when tasked to provide reasoning explanations. In this paper, we aim to
mitigate performance reduction through the integration of cognitive strategies
that humans use for problem-solving. We term this as cognitive enhancement
through prompts. Our experiments showed significant improvement gains of the
SLMs' performances when such enhancements were applied. We believe that our
exploration study paves the way for further investigation into the use of
cognitive enhancement to optimize SLM for cyber security applications.


---
