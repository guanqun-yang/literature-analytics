**[1. [2503.07697] PoisonedParrot: Subtle Data Poisoning Attacks to Elicit
  Copyright-Infringing Content from Large Language Models](https://arxiv.org/pdf/2503.07697.pdf)** (Updated on 2025-03-12)

*Michael-Andrei Panaitescu-Liess, Pankayaraj Pathmanathan, Yigitcan Kaya, Zora Che, Bang An, Sicheng Zhu, Aakriti Agrawal, Furong Huang*

  As the capabilities of large language models (LLMs) continue to expand, their
usage has become increasingly prevalent. However, as reflected in numerous
ongoing lawsuits regarding LLM-generated content, addressing copyright
infringement remains a significant challenge. In this paper, we introduce
PoisonedParrot: the first stealthy data poisoning attack that induces an LLM to
generate copyrighted content even when the model has not been directly trained
on the specific copyrighted material. PoisonedParrot integrates small fragments
of copyrighted text into the poison samples using an off-the-shelf LLM. Despite
its simplicity, evaluated in a wide range of experiments, PoisonedParrot is
surprisingly effective at priming the model to generate copyrighted content
with no discernible side effects. Moreover, we discover that existing defenses
are largely ineffective against our attack. Finally, we make the first attempt
at mitigating copyright-infringement poisoning attacks by proposing a defense:
ParrotTrap. We encourage the community to explore this emerging threat model
further.


---

**[2. [2404.17196] Human-Imperceptible Retrieval Poisoning Attacks in LLM-Powered
  Applications](https://arxiv.org/pdf/2404.17196.pdf)** (Updated on 2024-04-29)

*Quan Zhang, Binqi Zeng, Chijin Zhou, Gwihwan Go, Heyuan Shi, Yu Jiang*

  Presently, with the assistance of advanced LLM application development
frameworks, more and more LLM-powered applications can effortlessly augment the
LLMs' knowledge with external content using the retrieval augmented generation
(RAG) technique. However, these frameworks' designs do not have sufficient
consideration of the risk of external content, thereby allowing attackers to
undermine the applications developed with these frameworks. In this paper, we
reveal a new threat to LLM-powered applications, termed retrieval poisoning,
where attackers can guide the application to yield malicious responses during
the RAG process. Specifically, through the analysis of LLM application
frameworks, attackers can craft documents visually indistinguishable from
benign ones. Despite the documents providing correct information, once they are
used as reference sources for RAG, the application is misled into generating
incorrect responses. Our preliminary experiments indicate that attackers can
mislead LLMs with an 88.33\% success rate, and achieve a 66.67\% success rate
in the real-world application, demonstrating the potential impact of retrieval
poisoning.


---

**[3. [2411.18948] RevPRAG: Revealing Poisoning Attacks in Retrieval-Augmented Generation
  through LLM Activation Analysis](https://arxiv.org/pdf/2411.18948.pdf)** (Updated on 2025-02-20)

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

**[4. [2405.01560] Copyright related risks in the creation and use of ML/AI systems](https://arxiv.org/pdf/2405.01560.pdf)** (Updated on 2024-05-06)

*Daniel M. German*

  This paper summarizes the current copyright related risks that Machine
Learning (ML) and Artificial Intelligence (AI) systems (including Large
Language Models --LLMs) incur. These risks affect different stakeholders:
owners of the copyright of the training data, the users of ML/AI systems, the
creators of trained models, and the operators of AI systems. This paper also
provides an overview of ongoing legal cases in the United States related to
these risks.


---

**[5. [2308.12247] How to Protect Copyright Data in Optimization of Large Language Models?](https://arxiv.org/pdf/2308.12247.pdf)** (Updated on 2023-08-24)

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

**[6. [2502.01534] Preference Leakage: A Contamination Problem in LLM-as-a-judge](https://arxiv.org/pdf/2502.01534.pdf)** (Updated on 2025-02-04)

*Dawei Li, Renliang Sun, Yue Huang, Ming Zhong, Bohan Jiang, Jiawei Han, Xiangliang Zhang, Wei Wang, Huan Liu*

  Large Language Models (LLMs) as judges and LLM-based data synthesis have
emerged as two fundamental LLM-driven data annotation methods in model
development. While their combination significantly enhances the efficiency of
model training and evaluation, little attention has been given to the potential
contamination brought by this new model development paradigm. In this work, we
expose preference leakage, a contamination problem in LLM-as-a-judge caused by
the relatedness between the synthetic data generators and LLM-based evaluators.
To study this issue, we first define three common relatednesses between data
generator LLM and judge LLM: being the same model, having an inheritance
relationship, and belonging to the same model family. Through extensive
experiments, we empirically confirm the bias of judges towards their related
student models caused by preference leakage across multiple LLM baselines and
benchmarks. Further analysis suggests that preference leakage is a pervasive
issue that is harder to detect compared to previously identified biases in
LLM-as-a-judge scenarios. All of these findings imply that preference leakage
is a widespread and challenging problem in the area of LLM-as-a-judge. We
release all codes and data at:
https://github.com/David-Li0406/Preference-Leakage.


---

**[7. [2404.13968] Protecting Your LLMs with Information Bottleneck](https://arxiv.org/pdf/2404.13968.pdf)** (Updated on 2024-10-11)

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

**[8. [2502.10673] Dataset Protection via Watermarked Canaries in Retrieval-Augmented LLMs](https://arxiv.org/pdf/2502.10673.pdf)** (Updated on 2025-02-18)

*Yepeng Liu, Xuandong Zhao, Dawn Song, Yuheng Bu*

  Retrieval-Augmented Generation (RAG) has become an effective method for
enhancing large language models (LLMs) with up-to-date knowledge. However, it
poses a significant risk of IP infringement, as IP datasets may be incorporated
into the knowledge database by malicious Retrieval-Augmented LLMs (RA-LLMs)
without authorization. To protect the rights of the dataset owner, an effective
dataset membership inference algorithm for RA-LLMs is needed. In this work, we
introduce a novel approach to safeguard the ownership of text datasets and
effectively detect unauthorized use by the RA-LLMs. Our approach preserves the
original data completely unchanged while protecting it by inserting
specifically designed canary documents into the IP dataset. These canary
documents are created with synthetic content and embedded watermarks to ensure
uniqueness, stealthiness, and statistical provability. During the detection
process, unauthorized usage is identified by querying the canary documents and
analyzing the responses of RA-LLMs for statistical evidence of the embedded
watermark. Our experimental results demonstrate high query efficiency,
detectability, and stealthiness, along with minimal perturbation to the
original dataset, all without compromising the performance of the RAG system.


---

**[9. [2502.14425] A Survey on Data Contamination for Large Language Models](https://arxiv.org/pdf/2502.14425.pdf)** (Updated on 2025-02-21)

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

**[10. [2406.18382] Adversarial Search Engine Optimization for Large Language Models](https://arxiv.org/pdf/2406.18382.pdf)** (Updated on 2024-07-03)

*Fredrik Nestaas, Edoardo Debenedetti, Florian Tramèr*

  Large Language Models (LLMs) are increasingly used in applications where the
model selects from competing third-party content, such as in LLM-powered search
engines or chatbot plugins. In this paper, we introduce Preference Manipulation
Attacks, a new class of attacks that manipulate an LLM's selections to favor
the attacker. We demonstrate that carefully crafted website content or plugin
documentations can trick an LLM to promote the attacker products and discredit
competitors, thereby increasing user traffic and monetization. We show this
leads to a prisoner's dilemma, where all parties are incentivized to launch
attacks, but the collective effect degrades the LLM's outputs for everyone. We
demonstrate our attacks on production LLM search engines (Bing and Perplexity)
and plugin APIs (for GPT-4 and Claude). As LLMs are increasingly used to rank
third-party content, we expect Preference Manipulation Attacks to emerge as a
significant threat.


---

**[11. [2402.14845] Purifying Large Language Models by Ensembling a Small Language Model](https://arxiv.org/pdf/2402.14845.pdf)** (Updated on 2024-02-26)

*Tianlin Li, Qian Liu, Tianyu Pang, Chao Du, Qing Guo, Yang Liu, Min Lin*

  The emerging success of large language models (LLMs) heavily relies on
collecting abundant training data from external (untrusted) sources. Despite
substantial efforts devoted to data cleaning and curation, well-constructed
LLMs have been reported to suffer from copyright infringement, data poisoning,
and/or privacy violations, which would impede practical deployment of LLMs. In
this study, we propose a simple and easily implementable method for purifying
LLMs from the negative effects caused by uncurated data, namely, through
ensembling LLMs with benign and small language models (SLMs). Aside from
theoretical guarantees, we perform comprehensive experiments to empirically
confirm the efficacy of ensembling LLMs with SLMs, which can effectively
preserve the performance of LLMs while mitigating issues such as copyright
infringement, data poisoning, and privacy violations.


---

**[12. [2504.02883] SemEval-2025 Task 4: Unlearning sensitive content from Large Language
  Models](https://arxiv.org/pdf/2504.02883.pdf)** (Updated on 2025-04-07)

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

**[13. [2406.04244] Benchmark Data Contamination of Large Language Models: A Survey](https://arxiv.org/pdf/2406.04244.pdf)** (Updated on 2024-06-07)

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

**[14. [2412.04947] C$^2$LEVA: Toward Comprehensive and Contamination-Free Language Model
  Evaluation](https://arxiv.org/pdf/2412.04947.pdf)** (Updated on 2024-12-17)

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

**[15. [2311.11123] (Why) Is My Prompt Getting Worse? Rethinking Regression Testing for
  Evolving LLM APIs](https://arxiv.org/pdf/2311.11123.pdf)** (Updated on 2024-02-08)

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

**[16. [2412.13879] Crabs: Consuming Resource via Auto-generation for LLM-DoS Attack under
  Black-box Settings](https://arxiv.org/pdf/2412.13879.pdf)** (Updated on 2025-02-19)

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

**[17. [2502.14182] Multi-Faceted Studies on Data Poisoning can Advance LLM Development](https://arxiv.org/pdf/2502.14182.pdf)** (Updated on 2025-02-21)

*Pengfei He, Yue Xing, Han Xu, Zhen Xiang, Jiliang Tang*

  The lifecycle of large language models (LLMs) is far more complex than that
of traditional machine learning models, involving multiple training stages,
diverse data sources, and varied inference methods. While prior research on
data poisoning attacks has primarily focused on the safety vulnerabilities of
LLMs, these attacks face significant challenges in practice. Secure data
collection, rigorous data cleaning, and the multistage nature of LLM training
make it difficult to inject poisoned data or reliably influence LLM behavior as
intended. Given these challenges, this position paper proposes rethinking the
role of data poisoning and argue that multi-faceted studies on data poisoning
can advance LLM development. From a threat perspective, practical strategies
for data poisoning attacks can help evaluate and address real safety risks to
LLMs. From a trustworthiness perspective, data poisoning can be leveraged to
build more robust LLMs by uncovering and mitigating hidden biases, harmful
outputs, and hallucinations. Moreover, from a mechanism perspective, data
poisoning can provide valuable insights into LLMs, particularly the interplay
between data and model behavior, driving a deeper understanding of their
underlying mechanisms.


---

**[18. [2303.09384] LLMSecEval: A Dataset of Natural Language Prompts for Security
  Evaluations](https://arxiv.org/pdf/2303.09384.pdf)** (Updated on 2023-03-17)

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

**[19. [2305.10036] Are You Copying My Model? Protecting the Copyright of Large Language
  Models for EaaS via Backdoor Watermark](https://arxiv.org/pdf/2305.10036.pdf)** (Updated on 2023-06-05)

*Wenjun Peng, Jingwei Yi, Fangzhao Wu, Shangxi Wu, Bin Zhu, Lingjuan Lyu, Binxing Jiao, Tong Xu, Guangzhong Sun, Xing Xie*

  Large language models (LLMs) have demonstrated powerful capabilities in both
text understanding and generation. Companies have begun to offer Embedding as a
Service (EaaS) based on these LLMs, which can benefit various natural language
processing (NLP) tasks for customers. However, previous studies have shown that
EaaS is vulnerable to model extraction attacks, which can cause significant
losses for the owners of LLMs, as training these models is extremely expensive.
To protect the copyright of LLMs for EaaS, we propose an Embedding Watermark
method called EmbMarker that implants backdoors on embeddings. Our method
selects a group of moderate-frequency words from a general text corpus to form
a trigger set, then selects a target embedding as the watermark, and inserts it
into the embeddings of texts containing trigger words as the backdoor. The
weight of insertion is proportional to the number of trigger words included in
the text. This allows the watermark backdoor to be effectively transferred to
EaaS-stealer's model for copyright verification while minimizing the adverse
impact on the original embeddings' utility. Our extensive experiments on
various datasets show that our method can effectively protect the copyright of
EaaS models without compromising service quality.


---

**[20. [2311.10733] Proceedings of the 3rd International Workshop on Mining and Learning in
  the Legal Domain (MLLD-23)](https://arxiv.org/pdf/2311.10733.pdf)** (Updated on 2023-11-21)

*Masoud Makrehchi, Dell Zhang, Alina Petrova, John Armour*

  This is the Proceedings of the 3rd International Workshop on Mining and
Learning in the Legal Domain (MLLD-23) which took place in conjunction with the
32nd ACM International Conference on Information and Knowledge Management
(CIKM-2023) at the University of Birmingham, Birmingham, UK on Sunday 22nd
October 2023.


---

**[21. [2410.04454] Inner-Probe: Discovering Copyright-related Data Generation in LLM
  Architecture](https://arxiv.org/pdf/2410.04454.pdf)** (Updated on 2025-01-24)

*Qichao Ma, Rui-Jie Zhu, Peiye Liu, Renye Yan, Fahong Zhang, Ling Liang, Meng Li, Zhaofei Yu, Zongwei Wang, Yimao Cai, Tiejun Huang*

  Large Language Models (LLMs) utilize extensive knowledge databases and show
powerful text generation ability. However, their reliance on high-quality
copyrighted datasets raises concerns about copyright infringements in generated
texts. Current research often employs prompt engineering or semantic
classifiers to identify copyrighted content, but these approaches have two
significant limitations: (1) Challenging to identify which specific sub-dataset
(e.g., works from particular authors) influences an LLM's output. (2) Treating
the entire training database as copyrighted, hence overlooking the inclusion of
non-copyrighted training data.
  We propose InnerProbe, a lightweight framework designed to evaluate the
influence of copyrighted sub-datasets on LLM-generated texts. Unlike
traditional methods relying solely on text, we discover that the results of
multi-head attention (MHA) during LLM output generation provide more effective
information. Thus, InnerProbe performs sub-dataset contribution analysis using
a lightweight LSTM-based network trained on MHA results in a supervised manner.
Harnessing such a prior, InnerProbe enables non-copyrighted text detection
through a concatenated global projector trained with unsupervised contrastive
learning. InnerProbe demonstrates 3x improved efficiency compared to semantic
model training in sub-dataset contribution analysis on Books3, achieves
15.04%-58.7% higher accuracy over baselines on the Pile, and delivers a 0.104
increase in AUC for non-copyrighted data filtering.


---

**[22. [2502.13347] Craw4LLM: Efficient Web Crawling for LLM Pretraining](https://arxiv.org/pdf/2502.13347.pdf)** (Updated on 2025-02-26)

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

**[23. [2503.13733] CoDet-M4: Detecting Machine-Generated Code in Multi-Lingual,
  Multi-Generator and Multi-Domain Settings](https://arxiv.org/pdf/2503.13733.pdf)** (Updated on 2025-03-19)

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

**[24. [2407.20999] MoFO: Momentum-Filtered Optimizer for Mitigating Forgetting in LLM
  Fine-Tuning](https://arxiv.org/pdf/2407.20999.pdf)** (Updated on 2025-04-21)

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

**[25. [2403.05156] On Protecting the Data Privacy of Large Language Models (LLMs): A Survey](https://arxiv.org/pdf/2403.05156.pdf)** (Updated on 2024-03-15)

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

**[26. [2409.09288] Generating API Parameter Security Rules with LLM for API Misuse
  Detection](https://arxiv.org/pdf/2409.09288.pdf)** (Updated on 2024-09-20)

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

**[27. [2409.13831] Measuring Copyright Risks of Large Language Model via Partial
  Information Probing](https://arxiv.org/pdf/2409.13831.pdf)** (Updated on 2024-09-24)

*Weijie Zhao, Huajie Shao, Zhaozhuo Xu, Suzhen Duan, Denghui Zhang*

  Exploring the data sources used to train Large Language Models (LLMs) is a
crucial direction in investigating potential copyright infringement by these
models. While this approach can identify the possible use of copyrighted
materials in training data, it does not directly measure infringing risks.
Recent research has shifted towards testing whether LLMs can directly output
copyrighted content. Addressing this direction, we investigate and assess LLMs'
capacity to generate infringing content by providing them with partial
information from copyrighted materials, and try to use iterative prompting to
get LLMs to generate more infringing content. Specifically, we input a portion
of a copyrighted text into LLMs, prompt them to complete it, and then analyze
the overlap between the generated content and the original copyrighted
material. Our findings demonstrate that LLMs can indeed generate content highly
overlapping with copyrighted materials based on these partial inputs.


---

**[28. [2402.08100] Investigating the Impact of Data Contamination of Large Language Models
  in Text-to-SQL Translation](https://arxiv.org/pdf/2402.08100.pdf)** (Updated on 2024-12-10)

*Federico Ranaldi, Elena Sofia Ruzzetti, Dario Onorati, Leonardo Ranaldi, Cristina Giannone, Andrea Favalli, Raniero Romagnoli, Fabio Massimo Zanzotto*

  Understanding textual description to generate code seems to be an achieved
capability of instruction-following Large Language Models (LLMs) in zero-shot
scenario. However, there is a severe possibility that this translation ability
may be influenced by having seen target textual descriptions and the related
code. This effect is known as Data Contamination.
  In this study, we investigate the impact of Data Contamination on the
performance of GPT-3.5 in the Text-to-SQL code-generating tasks. Hence, we
introduce a novel method to detect Data Contamination in GPTs and examine
GPT-3.5's Text-to-SQL performances using the known Spider Dataset and our new
unfamiliar dataset Termite. Furthermore, we analyze GPT-3.5's efficacy on
databases with modified information via an adversarial table disconnection
(ATD) approach, complicating Text-to-SQL tasks by removing structural pieces of
information from the database. Our results indicate a significant performance
drop in GPT-3.5 on the unfamiliar Termite dataset, even with ATD modifications,
highlighting the effect of Data Contamination on LLMs in Text-to-SQL
translation tasks.


---

**[29. [2410.13903] CoreGuard: Safeguarding Foundational Capabilities of LLMs Against Model
  Stealing in Edge Deployment](https://arxiv.org/pdf/2410.13903.pdf)** (Updated on 2024-10-21)

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

**[30. [2408.10668] Probing the Safety Response Boundary of Large Language Models via Unsafe
  Decoding Path Generation](https://arxiv.org/pdf/2408.10668.pdf)** (Updated on 2024-08-27)

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

**[31. [2402.13459] Learning to Poison Large Language Models During Instruction Tuning](https://arxiv.org/pdf/2402.13459.pdf)** (Updated on 2024-10-24)

*Yao Qiang, Xiangyu Zhou, Saleh Zare Zade, Mohammad Amin Roshani, Prashant Khanduri, Douglas Zytko, Dongxiao Zhu*

  The advent of Large Language Models (LLMs) has marked significant
achievements in language processing and reasoning capabilities. Despite their
advancements, LLMs face vulnerabilities to data poisoning attacks, where
adversaries insert backdoor triggers into training data to manipulate outputs
for malicious purposes. This work further identifies additional security risks
in LLMs by designing a new data poisoning attack tailored to exploit the
instruction tuning process. We propose a novel gradient-guided backdoor trigger
learning (GBTL) algorithm to identify adversarial triggers efficiently,
ensuring an evasion of detection by conventional defenses while maintaining
content integrity. Through experimental validation across various tasks,
including sentiment analysis, domain generation, and question answering, our
poisoning strategy demonstrates a high success rate in compromising various
LLMs' outputs. We further propose two defense strategies against data poisoning
attacks, including in-context learning (ICL) and continuous learning (CL),
which effectively rectify the behavior of LLMs and significantly reduce the
decline in performance. Our work highlights the significant security risks
present during the instruction tuning of LLMs and emphasizes the necessity of
safeguarding LLMs against data poisoning attacks.


---

**[32. [2403.00826] LLMGuard: Guarding Against Unsafe LLM Behavior](https://arxiv.org/pdf/2403.00826.pdf)** (Updated on 2024-03-05)

*Shubh Goyal, Medha Hira, Shubham Mishra, Sukriti Goyal, Arnav Goel, Niharika Dadu, Kirushikesh DB, Sameep Mehta, Nishtha Madaan*

  Although the rise of Large Language Models (LLMs) in enterprise settings
brings new opportunities and capabilities, it also brings challenges, such as
the risk of generating inappropriate, biased, or misleading content that
violates regulations and can have legal concerns. To alleviate this, we present
"LLMGuard", a tool that monitors user interactions with an LLM application and
flags content against specific behaviours or conversation topics. To do this
robustly, LLMGuard employs an ensemble of detectors.


---

**[33. [2503.00032] Detecting LLM-Generated Korean Text through Linguistic Feature Analysis](https://arxiv.org/pdf/2503.00032.pdf)** (Updated on 2025-03-05)

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

**[34. [2402.05624] Efficient Models for the Detection of Hate, Abuse and Profanity](https://arxiv.org/pdf/2402.05624.pdf)** (Updated on 2024-02-09)

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

**[35. [2405.05610] Chain of Attack: a Semantic-Driven Contextual Multi-Turn attacker for
  LLM](https://arxiv.org/pdf/2405.05610.pdf)** (Updated on 2024-05-10)

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

**[36. [2504.03957] Practical Poisoning Attacks against Retrieval-Augmented Generation](https://arxiv.org/pdf/2504.03957.pdf)** (Updated on 2025-04-08)

*Baolei Zhang, Yuxi Chen, Minghong Fang, Zhuqing Liu, Lihai Nie, Tong Li, Zheli Liu*

  Large language models (LLMs) have demonstrated impressive natural language
processing abilities but face challenges such as hallucination and outdated
knowledge. Retrieval-Augmented Generation (RAG) has emerged as a
state-of-the-art approach to mitigate these issues. While RAG enhances LLM
outputs, it remains vulnerable to poisoning attacks. Recent studies show that
injecting poisoned text into the knowledge database can compromise RAG systems,
but most existing attacks assume that the attacker can insert a sufficient
number of poisoned texts per query to outnumber correct-answer texts in
retrieval, an assumption that is often unrealistic. To address this limitation,
we propose CorruptRAG, a practical poisoning attack against RAG systems in
which the attacker injects only a single poisoned text, enhancing both
feasibility and stealth. Extensive experiments across multiple datasets
demonstrate that CorruptRAG achieves higher attack success rates compared to
existing baselines.


---

**[37. [2408.02946] Data Poisoning in LLMs: Jailbreak-Tuning and Scaling Laws](https://arxiv.org/pdf/2408.02946.pdf)** (Updated on 2024-12-31)

*Dillon Bowen, Brendan Murphy, Will Cai, David Khachaturov, Adam Gleave, Kellin Pelrine*

  LLMs produce harmful and undesirable behavior when trained on poisoned
datasets that contain a small fraction of corrupted or harmful data. We develop
a new attack paradigm, jailbreak-tuning, that combines data poisoning with
jailbreaking to fully bypass state-of-the-art safeguards and make models like
GPT-4o comply with nearly any harmful request. Our experiments suggest this
attack represents a paradigm shift in vulnerability elicitation, producing
differences in refusal rates as much as 60+ percentage points compared to
normal fine-tuning. Given this demonstration of how data poisoning
vulnerabilities persist and can be amplified, we investigate whether these
risks will likely increase as models scale. We evaluate three threat models -
malicious fine-tuning, imperfect data curation, and intentional data
contamination - across 24 frontier LLMs ranging from 1.5 to 72 billion
parameters. Our experiments reveal that larger LLMs are significantly more
susceptible to data poisoning, learning harmful behaviors from even minimal
exposure to harmful data more quickly than smaller models. These findings
underscore the need for leading AI companies to thoroughly red team fine-tuning
APIs before public release and to develop more robust safeguards against data
poisoning, particularly as models continue to scale in size and capability.


---

**[38. [2406.16201] Blind Baselines Beat Membership Inference Attacks for Foundation Models](https://arxiv.org/pdf/2406.16201.pdf)** (Updated on 2025-04-01)

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

**[39. [2406.08754] StructuralSleight: Automated Jailbreak Attacks on Large Language Models
  Utilizing Uncommon Text-Organization Structures](https://arxiv.org/pdf/2406.08754.pdf)** (Updated on 2025-02-19)

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

**[40. [2409.14038] OAEI-LLM: A Benchmark Dataset for Understanding Large Language Model
  Hallucinations in Ontology Matching](https://arxiv.org/pdf/2409.14038.pdf)** (Updated on 2025-02-04)

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

**[41. [2401.10360] Excuse me, sir? Your language model is leaking (information)](https://arxiv.org/pdf/2401.10360.pdf)** (Updated on 2024-11-19)

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

**[42. [2410.14273] REEF: Representation Encoding Fingerprints for Large Language Models](https://arxiv.org/pdf/2410.14273.pdf)** (Updated on 2024-10-21)

*Jie Zhang, Dongrui Liu, Chen Qian, Linfeng Zhang, Yong Liu, Yu Qiao, Jing Shao*

  Protecting the intellectual property of open-source Large Language Models
(LLMs) is very important, because training LLMs costs extensive computational
resources and data. Therefore, model owners and third parties need to identify
whether a suspect model is a subsequent development of the victim model. To
this end, we propose a training-free REEF to identify the relationship between
the suspect and victim models from the perspective of LLMs' feature
representations. Specifically, REEF computes and compares the centered kernel
alignment similarity between the representations of a suspect model and a
victim model on the same samples. This training-free REEF does not impair the
model's general capabilities and is robust to sequential fine-tuning, pruning,
model merging, and permutations. In this way, REEF provides a simple and
effective way for third parties and models' owners to protect LLMs'
intellectual property together. The code is available at
https://github.com/tmylla/REEF.


---

**[43. [2503.23566] When LLM Therapists Become Salespeople: Evaluating Large Language Models
  for Ethical Motivational Interviewing](https://arxiv.org/pdf/2503.23566.pdf)** (Updated on 2025-04-01)

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

**[44. [2410.10760] Denial-of-Service Poisoning Attacks against Large Language Models](https://arxiv.org/pdf/2410.10760.pdf)** (Updated on 2024-10-15)

*Kuofeng Gao, Tianyu Pang, Chao Du, Yong Yang, Shu-Tao Xia, Min Lin*

  Recent studies have shown that LLMs are vulnerable to denial-of-service (DoS)
attacks, where adversarial inputs like spelling errors or non-semantic prompts
trigger endless outputs without generating an [EOS] token. These attacks can
potentially cause high latency and make LLM services inaccessible to other
users or tasks. However, when there are speech-to-text interfaces (e.g., voice
commands to a robot), executing such DoS attacks becomes challenging, as it is
difficult to introduce spelling errors or non-semantic prompts through speech.
A simple DoS attack in these scenarios would be to instruct the model to "Keep
repeating Hello", but we observe that relying solely on natural instructions
limits output length, which is bounded by the maximum length of the LLM's
supervised finetuning (SFT) data. To overcome this limitation, we propose
poisoning-based DoS (P-DoS) attacks for LLMs, demonstrating that injecting a
single poisoned sample designed for DoS purposes can break the output length
limit. For example, a poisoned sample can successfully attack GPT-4o and GPT-4o
mini (via OpenAI's finetuning API) using less than $1, causing repeated outputs
up to the maximum inference length (16K tokens, compared to 0.5K before
poisoning). Additionally, we perform comprehensive ablation studies on
open-source LLMs and extend our method to LLM agents, where attackers can
control both the finetuning dataset and algorithm. Our findings underscore the
urgent need for defenses against P-DoS attacks to secure LLMs. Our code is
available at https://github.com/sail-sg/P-DoS.


---

**[45. [2410.18966] Does Data Contamination Detection Work (Well) for LLMs? A Survey and
  Evaluation on Detection Assumptions](https://arxiv.org/pdf/2410.18966.pdf)** (Updated on 2025-03-11)

*Yujuan Fu, Ozlem Uzuner, Meliha Yetisgen, Fei Xia*

  Large language models (LLMs) have demonstrated great performance across
various benchmarks, showing potential as general-purpose task solvers. However,
as LLMs are typically trained on vast amounts of data, a significant concern in
their evaluation is data contamination, where overlap between training data and
evaluation datasets inflates performance assessments. Multiple approaches have
been developed to identify data contamination. These approaches rely on
specific assumptions that may not hold universally across different settings.
To bridge this gap, we systematically review 50 papers on data contamination
detection, categorize the underlying assumptions, and assess whether they have
been rigorously validated. We identify and analyze eight categories of
assumptions and test three of them as case studies. Our case studies focus on
detecting direct, instance-level data contamination, which is also referred to
as Membership Inference Attacks (MIA). Our analysis reveals that MIA approaches
based on these three assumptions can have similar performance to random
guessing, on datasets used in LLM pretraining, suggesting that current LLMs
might learn data distributions rather than memorizing individual instances.
Meanwhile, MIA can easily fail when there are data distribution shifts between
the seen and unseen instances.


---

**[46. [2501.10915] LegalGuardian: A Privacy-Preserving Framework for Secure Integration of
  Large Language Models in Legal Practice](https://arxiv.org/pdf/2501.10915.pdf)** (Updated on 2025-01-22)

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

**[47. [2402.16893] The Good and The Bad: Exploring Privacy Issues in Retrieval-Augmented
  Generation (RAG)](https://arxiv.org/pdf/2402.16893.pdf)** (Updated on 2024-03-03)

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

**[48. [2312.07200] Code Membership Inference for Detecting Unauthorized Data Use in Code
  Pre-trained Language Models](https://arxiv.org/pdf/2312.07200.pdf)** (Updated on 2025-02-19)

*Sheng Zhang, Hui Li*

  Code pre-trained language models (CPLMs) have received great attention since
they can benefit various tasks that facilitate software development and
maintenance. However, CPLMs are trained on massive open-source code, raising
concerns about potential data infringement. This paper launches the study of
detecting unauthorized code use in CPLMs, i.e., Code Membership Inference (CMI)
task. We design a framework Buzzer for different settings of CMI. Buzzer
deploys several inference techniques, including signal extraction from
pre-training tasks, hard-to-learn sample calibration and weighted inference, to
identify code membership status accurately. Extensive experiments show that CMI
can be achieved with high accuracy using Buzzer. Hence, Buzzer can serve as a
CMI tool and help protect intellectual property rights.


---

**[49. [2306.06815] TrojLLM: A Black-box Trojan Prompt Attack on Large Language Models](https://arxiv.org/pdf/2306.06815.pdf)** (Updated on 2023-11-01)

*Jiaqi Xue, Mengxin Zheng, Ting Hua, Yilin Shen, Yepeng Liu, Ladislau Boloni, Qian Lou*

  Large Language Models (LLMs) are progressively being utilized as machine
learning services and interface tools for various applications. However, the
security implications of LLMs, particularly in relation to adversarial and
Trojan attacks, remain insufficiently examined. In this paper, we propose
TrojLLM, an automatic and black-box framework to effectively generate universal
and stealthy triggers. When these triggers are incorporated into the input
data, the LLMs' outputs can be maliciously manipulated. Moreover, the framework
also supports embedding Trojans within discrete prompts, enhancing the overall
effectiveness and precision of the triggers' attacks. Specifically, we propose
a trigger discovery algorithm for generating universal triggers for various
inputs by querying victim LLM-based APIs using few-shot data samples.
Furthermore, we introduce a novel progressive Trojan poisoning algorithm
designed to generate poisoned prompts that retain efficacy and transferability
across a diverse range of models. Our experiments and results demonstrate
TrojLLM's capacity to effectively insert Trojans into text prompts in
real-world black-box LLM APIs including GPT-3.5 and GPT-4, while maintaining
exceptional performance on clean test sets. Our work sheds light on the
potential security risks in current models and offers a potential defensive
approach. The source code of TrojLLM is available at
https://github.com/UCF-ML-Research/TrojLLM.


---

**[50. [2405.19677] Large Language Model Watermark Stealing With Mixed Integer Programming](https://arxiv.org/pdf/2405.19677.pdf)** (Updated on 2024-05-31)

*Zhaoxi Zhang, Xiaomei Zhang, Yanjun Zhang, Leo Yu Zhang, Chao Chen, Shengshan Hu, Asif Gill, Shirui Pan*

  The Large Language Model (LLM) watermark is a newly emerging technique that
shows promise in addressing concerns surrounding LLM copyright, monitoring
AI-generated text, and preventing its misuse. The LLM watermark scheme commonly
includes generating secret keys to partition the vocabulary into green and red
lists, applying a perturbation to the logits of tokens in the green list to
increase their sampling likelihood, thus facilitating watermark detection to
identify AI-generated text if the proportion of green tokens exceeds a
threshold. However, recent research indicates that watermarking methods using
numerous keys are susceptible to removal attacks, such as token editing,
synonym substitution, and paraphrasing, with robustness declining as the number
of keys increases. Therefore, the state-of-the-art watermark schemes that
employ fewer or single keys have been demonstrated to be more robust against
text editing and paraphrasing. In this paper, we propose a novel green list
stealing attack against the state-of-the-art LLM watermark scheme and
systematically examine its vulnerability to this attack. We formalize the
attack as a mixed integer programming problem with constraints. We evaluate our
attack under a comprehensive threat model, including an extreme scenario where
the attacker has no prior knowledge, lacks access to the watermark detector
API, and possesses no information about the LLM's parameter settings or
watermark injection/detection scheme. Extensive experiments on LLMs, such as
OPT and LLaMA, demonstrate that our attack can successfully steal the green
list and remove the watermark across all settings.


---

**[51. [2503.13572] VeriContaminated: Assessing LLM-Driven Verilog Coding for Data
  Contamination](https://arxiv.org/pdf/2503.13572.pdf)** (Updated on 2025-04-15)

*Zeng Wang, Minghao Shao, Jitendra Bhandari, Likhitha Mankali, Ramesh Karri, Ozgur Sinanoglu, Muhammad Shafique, Johann Knechtel*

  Large Language Models (LLMs) have revolutionized code generation, achieving
exceptional results on various established benchmarking frameworks. However,
concerns about data contamination - where benchmark data inadvertently leaks
into pre-training or fine-tuning datasets - raise questions about the validity
of these evaluations. While this issue is known, limiting the industrial
adoption of LLM-driven software engineering, hardware coding has received
little to no attention regarding these risks. For the first time, we analyze
state-of-the-art (SOTA) evaluation frameworks for Verilog code generation
(VerilogEval and RTLLM), using established methods for contamination detection
(CCD and Min-K% Prob). We cover SOTA commercial and open-source LLMs
(CodeGen2.5, Minitron 4b, Mistral 7b, phi-4 mini, LLaMA-{1,2,3.1},
GPT-{2,3.5,4o}, Deepseek-Coder, and CodeQwen 1.5), in baseline and fine-tuned
models (RTLCoder and Verigen). Our study confirms that data contamination is a
critical concern. We explore mitigations and the resulting trade-offs for code
quality vs fairness (i.e., reducing contamination toward unbiased
benchmarking).


---

**[52. [2310.18018] NLP Evaluation in trouble: On the Need to Measure LLM Data Contamination
  for each Benchmark](https://arxiv.org/pdf/2310.18018.pdf)** (Updated on 2023-10-30)

*Oscar Sainz, Jon Ander Campos, Iker García-Ferrero, Julen Etxaniz, Oier Lopez de Lacalle, Eneko Agirre*

  In this position paper, we argue that the classical evaluation on Natural
Language Processing (NLP) tasks using annotated benchmarks is in trouble. The
worst kind of data contamination happens when a Large Language Model (LLM) is
trained on the test split of a benchmark, and then evaluated in the same
benchmark. The extent of the problem is unknown, as it is not straightforward
to measure. Contamination causes an overestimation of the performance of a
contaminated model in a target benchmark and associated task with respect to
their non-contaminated counterparts. The consequences can be very harmful, with
wrong scientific conclusions being published while other correct ones are
discarded. This position paper defines different levels of data contamination
and argues for a community effort, including the development of automatic and
semi-automatic measures to detect when data from a benchmark was exposed to a
model, and suggestions for flagging papers with conclusions that are
compromised by data contamination.


---

**[53. [2404.19048] A Framework for Real-time Safeguarding the Text Generation of Large
  Language Model](https://arxiv.org/pdf/2404.19048.pdf)** (Updated on 2024-05-03)

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

**[54. [2405.11466] Measuring Impacts of Poisoning on Model Parameters and Embeddings for
  Large Language Models of Code](https://arxiv.org/pdf/2405.11466.pdf)** (Updated on 2024-05-21)

*Aftab Hussain, Md Rafiqul Islam Rabin, Mohammad Amin Alipour*

  Large language models (LLMs) have revolutionized software development
practices, yet concerns about their safety have arisen, particularly regarding
hidden backdoors, aka trojans. Backdoor attacks involve the insertion of
triggers into training data, allowing attackers to manipulate the behavior of
the model maliciously. In this paper, we focus on analyzing the model
parameters to detect potential backdoor signals in code models. Specifically,
we examine attention weights and biases, and context embeddings of the clean
and poisoned CodeBERT and CodeT5 models. Our results suggest noticeable
patterns in context embeddings of poisoned samples for both the poisoned
models; however, attention weights and biases do not show any significant
differences. This work contributes to ongoing efforts in white-box detection of
backdoor signals in LLMs of code through the analysis of parameters and
embeddings.


---

**[55. [2503.16740] Automated Harmfulness Testing for Code Large Language Models](https://arxiv.org/pdf/2503.16740.pdf)** (Updated on 2025-03-24)

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

**[56. [2411.06493] LProtector: An LLM-driven Vulnerability Detection System](https://arxiv.org/pdf/2411.06493.pdf)** (Updated on 2024-11-15)

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

**[57. [2502.17749] Detection of LLM-Paraphrased Code and Identification of the Responsible
  LLM Using Coding Style Features](https://arxiv.org/pdf/2502.17749.pdf)** (Updated on 2025-03-03)

*Shinwoo Park, Hyundong Jin, Jeong-won Cha, Yo-Sub Han*

  Recent progress in large language models (LLMs) for code generation has
raised serious concerns about intellectual property protection. Malicious users
can exploit LLMs to produce paraphrased versions of proprietary code that
closely resemble the original. While the potential for LLM-assisted code
paraphrasing continues to grow, research on detecting it remains limited,
underscoring an urgent need for detection system. We respond to this need by
proposing two tasks. The first task is to detect whether code generated by an
LLM is a paraphrased version of original human-written code. The second task is
to identify which LLM is used to paraphrase the original code. For these tasks,
we construct a dataset LPcode consisting of pairs of human-written code and
LLM-paraphrased code using various LLMs.
  We statistically confirm significant differences in the coding styles of
human-written and LLM-paraphrased code, particularly in terms of naming
consistency, code structure, and readability. Based on these findings, we
develop LPcodedec, a detection method that identifies paraphrase relationships
between human-written and LLM-generated code, and discover which LLM is used
for the paraphrasing. LPcodedec outperforms the best baselines in two tasks,
improving F1 scores by 2.64% and 15.17% while achieving speedups of 1,343x and
213x, respectively. Our code and data are available at
https://github.com/Shinwoo-Park/detecting_llm_paraphrased_code_via_coding_style_features.


---

**[58. [2311.18815] IMMA: Immunizing text-to-image Models against Malicious Adaptation](https://arxiv.org/pdf/2311.18815.pdf)** (Updated on 2024-10-01)

*Amber Yijia Zheng, Raymond A. Yeh*

  Advancements in open-sourced text-to-image models and fine-tuning methods
have led to the increasing risk of malicious adaptation, i.e., fine-tuning to
generate harmful/unauthorized content. Recent works, e.g., Glaze or MIST, have
developed data-poisoning techniques which protect the data against adaptation
methods. In this work, we consider an alternative paradigm for protection. We
propose to ``immunize'' the model by learning model parameters that are
difficult for the adaptation methods when fine-tuning malicious content; in
short IMMA. Specifically, IMMA should be applied before the release of the
model weights to mitigate these risks. Empirical results show IMMA's
effectiveness against malicious adaptations, including mimicking the artistic
style and learning of inappropriate/unauthorized content, over three adaptation
methods: LoRA, Textual-Inversion, and DreamBooth. The code is available at
\url{https://github.com/amberyzheng/IMMA}.


---

**[59. [2502.11533] Be Cautious When Merging Unfamiliar LLMs: A Phishing Model Capable of
  Stealing Privacy](https://arxiv.org/pdf/2502.11533.pdf)** (Updated on 2025-02-18)

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

**[60. [2502.18518] Swallowing the Poison Pills: Insights from Vulnerability Disparity Among
  LLMs](https://arxiv.org/pdf/2502.18518.pdf)** (Updated on 2025-02-27)

*Peng Yifeng, Wu Zhizheng, Chen Chen*

  Modern large language models (LLMs) exhibit critical vulnerabilities to
poison pill attacks: localized data poisoning that alters specific factual
knowledge while preserving overall model utility. We systematically demonstrate
these attacks exploit inherent architectural properties of LLMs, achieving
54.6% increased retrieval inaccuracy on long-tail knowledge versus dominant
topics and up to 25.5% increase retrieval inaccuracy on compressed models
versus original architectures. Through controlled mutations (e.g.,
temporal/spatial/entity alterations) and, our method induces localized
memorization deterioration with negligible impact on models' performance on
regular standard benchmarks (e.g., <2% performance drop on MMLU/GPQA), leading
to potential detection evasion. Our findings suggest: (1) Disproportionate
vulnerability in long-tail knowledge may result from reduced parameter
redundancy; (2) Model compression may increase attack surfaces, with
pruned/distilled models requiring 30% fewer poison samples for equivalent
damage; (3) Associative memory enables both spread of collateral damage to
related concepts and amplification of damage from simultaneous attack,
particularly for dominant topics. These findings raise concerns over current
scaling paradigms since attack costs are lowering while defense complexity is
rising. Our work establishes poison pills as both a security threat and
diagnostic tool, revealing critical security-efficiency trade-offs in language
model compression that challenges prevailing safety assumptions.


---

**[61. [2409.19134] Confidential Prompting: Protecting User Prompts from Cloud LLM Providers](https://arxiv.org/pdf/2409.19134.pdf)** (Updated on 2025-03-05)

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

**[62. [2406.10952] Avoiding Copyright Infringement via Large Language Model Unlearning](https://arxiv.org/pdf/2406.10952.pdf)** (Updated on 2025-02-12)

*Guangyao Dou, Zheyuan Liu, Qing Lyu, Kaize Ding, Eric Wong*

  Pre-trained Large Language Models (LLMs) have demonstrated remarkable
capabilities but also pose risks by learning and generating copyrighted
material, leading to significant legal and ethical concerns. In real-world
scenarios, model owners need to continuously address copyright infringement as
new requests for content removal emerge at different time points. This leads to
the need for sequential unlearning, where copyrighted content is removed
sequentially as new requests arise. Despite its practical relevance, sequential
unlearning in the context of copyright infringement has not been rigorously
explored in existing literature. To address this gap, we propose Stable
Sequential Unlearning (SSU), a novel framework designed to unlearn copyrighted
content from LLMs over multiple time steps. Our approach works by identifying
and removing specific weight updates in the model's parameters that correspond
to copyrighted content. We improve unlearning efficacy by introducing random
labeling loss and ensuring the model retains its general-purpose knowledge by
adjusting targeted parameters. Experimental results show that SSU achieves an
effective trade-off between unlearning efficacy and general-purpose language
abilities, outperforming existing baselines.


---

**[63. [2406.13990] Inference-Time Decontamination: Reusing Leaked Benchmarks for Large
  Language Model Evaluation](https://arxiv.org/pdf/2406.13990.pdf)** (Updated on 2024-06-25)

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

**[64. [2406.01333] Probing Language Models for Pre-training Data Detection](https://arxiv.org/pdf/2406.01333.pdf)** (Updated on 2024-06-04)

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

**[65. [2310.10049] FATE-LLM: A Industrial Grade Federated Learning Framework for Large
  Language Models](https://arxiv.org/pdf/2310.10049.pdf)** (Updated on 2023-10-17)

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

**[66. [2407.06443] Exposing Privacy Gaps: Membership Inference Attack on Preference Data
  for LLM Alignment](https://arxiv.org/pdf/2407.06443.pdf)** (Updated on 2024-07-10)

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

**[67. [2503.04693] UIPE: Enhancing LLM Unlearning by Removing Knowledge Related to
  Forgetting Targets](https://arxiv.org/pdf/2503.04693.pdf)** (Updated on 2025-03-07)

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

**[68. [2502.10440] Towards Copyright Protection for Knowledge Bases of Retrieval-augmented
  Language Models via Ownership Verification with Reasoning](https://arxiv.org/pdf/2502.10440.pdf)** (Updated on 2025-02-18)

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

**[69. [2411.01077] Emoji Attack: Enhancing Jailbreak Attacks Against Judge LLM Detection](https://arxiv.org/pdf/2411.01077.pdf)** (Updated on 2025-02-19)

*Zhipeng Wei, Yuqi Liu, N. Benjamin Erichson*

  Jailbreaking techniques trick Large Language Models (LLMs) into producing
restricted outputs, posing a serious threat. One line of defense is to use
another LLM as a Judge to evaluate the harmfulness of generated text. However,
we reveal that these Judge LLMs are vulnerable to token segmentation bias, an
issue that arises when delimiters alter the tokenization process, splitting
words into smaller sub-tokens. This disrupts the embeddings of the entire
sequence, reducing detection accuracy and allowing harmful content to be
misclassified as safe. In this paper, we introduce Emoji Attack, a novel
strategy that amplifies existing jailbreak prompts by exploiting token
segmentation bias. Our method leverages in-context learning to systematically
insert emojis into text before it is evaluated by a Judge LLM, inducing
embedding distortions that significantly lower the likelihood of detecting
unsafe content. Unlike traditional delimiters, emojis also introduce semantic
ambiguity, making them particularly effective in this attack. Through
experiments on state-of-the-art Judge LLMs, we demonstrate that Emoji Attack
substantially reduces the "unsafe" prediction rate, bypassing existing
safeguards.


---

**[70. [2503.04636] Mark Your LLM: Detecting the Misuse of Open-Source Large Language Models
  via Watermarking](https://arxiv.org/pdf/2503.04636.pdf)** (Updated on 2025-03-18)

*Yijie Xu, Aiwei Liu, Xuming Hu, Lijie Wen, Hui Xiong*

  As open-source large language models (LLMs) like Llama3 become more capable,
it is crucial to develop watermarking techniques to detect their potential
misuse. Existing watermarking methods either add watermarks during LLM
inference, which is unsuitable for open-source LLMs, or primarily target
classification LLMs rather than recent generative LLMs. Adapting these
watermarks to open-source LLMs for misuse detection remains an open challenge.
This work defines two misuse scenarios for open-source LLMs: intellectual
property (IP) violation and LLM Usage Violation. Then, we explore the
application of inference-time watermark distillation and backdoor watermarking
in these contexts. We propose comprehensive evaluation methods to assess the
impact of various real-world further fine-tuning scenarios on watermarks and
the effect of these watermarks on LLM performance. Our experiments reveal that
backdoor watermarking could effectively detect IP Violation, while
inference-time watermark distillation is applicable in both scenarios but less
robust to further fine-tuning and has a more significant impact on LLM
performance compared to backdoor watermarking. Exploring more advanced
watermarking methods for open-source LLMs to detect their misuse should be an
important future direction.


---

**[71. [2412.13670] AntiLeak-Bench: Preventing Data Contamination by Automatically
  Constructing Benchmarks with Updated Real-World Knowledge](https://arxiv.org/pdf/2412.13670.pdf)** (Updated on 2024-12-19)

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

**[72. [2410.21723] Fine-tuning Large Language Models for DGA and DNS Exfiltration Detection](https://arxiv.org/pdf/2410.21723.pdf)** (Updated on 2024-11-08)

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

**[73. [2502.16901] Char-mander Use mBackdoor! A Study of Cross-lingual Backdoor Attacks in
  Multilingual LLMs](https://arxiv.org/pdf/2502.16901.pdf)** (Updated on 2025-02-25)

*Himanshu Beniwal, Sailesh Panda, Mayank Singh*

  We explore Cross-lingual Backdoor ATtacks (X-BAT) in multilingual Large
Language Models (mLLMs), revealing how backdoors inserted in one language can
automatically transfer to others through shared embedding spaces. Using
toxicity classification as a case study, we demonstrate that attackers can
compromise multilingual systems by poisoning data in a single language, with
rare tokens serving as specific effective triggers. Our findings expose a
critical vulnerability in the fundamental architecture that enables
cross-lingual transfer in these models. Our code and data are publicly
available at https://github.com/himanshubeniwal/X-BAT.


---

**[74. [2312.07130] Harnessing LLM to Attack LLM-Guarded Text-to-Image Models](https://arxiv.org/pdf/2312.07130.pdf)** (Updated on 2024-11-27)

*Yimo Deng, Huangxun Chen*

  To prevent Text-to-Image (T2I) models from generating unethical images,
people deploy safety filters to block inappropriate drawing prompts. Previous
works have employed token replacement to search adversarial prompts that
attempt to bypass these filters, but they have become ineffective as
nonsensical tokens fail semantic logic checks. In this paper, we approach
adversarial prompts from a different perspective. We demonstrate that
rephrasing a drawing intent into multiple benign descriptions of individual
visual components can obtain an effective adversarial prompt. We propose a
LLM-piloted multi-agent method named DACA to automatically complete intended
rephrasing. Our method successfully bypasses the safety filters of DALL-E 3 and
Midjourney to generate the intended images, achieving success rates of up to
76.7% and 64% in the one-time attack, and 98% and 84% in the re-use attack,
respectively. We open-source our code and dataset on [this
link](https://github.com/researchcode003/DACA).


---

**[75. [2403.18920] CPR: Retrieval Augmented Generation for Copyright Protection](https://arxiv.org/pdf/2403.18920.pdf)** (Updated on 2024-03-29)

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

**[76. [2409.15154] RMCBench: Benchmarking Large Language Models' Resistance to Malicious
  Code](https://arxiv.org/pdf/2409.15154.pdf)** (Updated on 2024-09-24)

*Jiachi Chen, Qingyuan Zhong, Yanlin Wang, Kaiwen Ning, Yongkun Liu, Zenan Xu, Zhe Zhao, Ting Chen, Zibin Zheng*

  The emergence of Large Language Models (LLMs) has significantly influenced
various aspects of software development activities. Despite their benefits,
LLMs also pose notable risks, including the potential to generate harmful
content and being abused by malicious developers to create malicious code.
Several previous studies have focused on the ability of LLMs to resist the
generation of harmful content that violates human ethical standards, such as
biased or offensive content. However, there is no research evaluating the
ability of LLMs to resist malicious code generation. To fill this gap, we
propose RMCBench, the first benchmark comprising 473 prompts designed to assess
the ability of LLMs to resist malicious code generation. This benchmark employs
two scenarios: a text-to-code scenario, where LLMs are prompted with
descriptions to generate code, and a code-to-code scenario, where LLMs
translate or complete existing malicious code. Based on RMCBench, we conduct an
empirical study on 11 representative LLMs to assess their ability to resist
malicious code generation. Our findings indicate that current LLMs have a
limited ability to resist malicious code generation with an average refusal
rate of 40.36% in text-to-code scenario and 11.52% in code-to-code scenario.
The average refusal rate of all LLMs in RMCBench is only 28.71%; ChatGPT-4 has
a refusal rate of only 35.73%. We also analyze the factors that affect LLMs'
ability to resist malicious code generation and provide implications for
developers to enhance model robustness.


---

**[77. [2405.04032] Locally Differentially Private In-Context Learning](https://arxiv.org/pdf/2405.04032.pdf)** (Updated on 2024-05-09)

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

**[78. [2503.21598] Prompt, Divide, and Conquer: Bypassing Large Language Model Safety
  Filters via Segmented and Distributed Prompt Processing](https://arxiv.org/pdf/2503.21598.pdf)** (Updated on 2025-04-01)

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

**[79. [2411.01705] Data Extraction Attacks in Retrieval-Augmented Generation via Backdoors](https://arxiv.org/pdf/2411.01705.pdf)** (Updated on 2025-04-01)

*Yuefeng Peng, Junda Wang, Hong Yu, Amir Houmansadr*

  Despite significant advancements, large language models (LLMs) still struggle
with providing accurate answers when lacking domain-specific or up-to-date
knowledge. Retrieval-Augmented Generation (RAG) addresses this limitation by
incorporating external knowledge bases, but it also introduces new attack
surfaces. In this paper, we investigate data extraction attacks targeting RAG's
knowledge databases. We show that previous prompt injection-based extraction
attacks largely rely on the instruction-following capabilities of LLMs. As a
result, they fail on models that are less responsive to such malicious prompts
-- for example, our experiments show that state-of-the-art attacks achieve
near-zero success on Gemma-2B-IT. Moreover, even for models that can follow
these instructions, we found fine-tuning may significantly reduce attack
performance. To further reveal the vulnerability, we propose to backdoor RAG,
where a small portion of poisoned data is injected during the fine-tuning phase
to create a backdoor within the LLM. When this compromised LLM is integrated
into a RAG system, attackers can exploit specific triggers in prompts to
manipulate the LLM to leak documents from the retrieval database. By carefully
designing the poisoned data, we achieve both verbatim and paraphrased document
extraction. For example, on Gemma-2B-IT, we show that with only 5\% poisoned
data, our method achieves an average success rate of 94.1\% for verbatim
extraction (ROUGE-L score: 82.1) and 63.6\% for paraphrased extraction (average
ROUGE score: 66.4) across four datasets. These results underscore the privacy
risks associated with the supply chain when deploying RAG systems.


---

**[80. [2402.02160] Data Poisoning for In-context Learning](https://arxiv.org/pdf/2402.02160.pdf)** (Updated on 2024-03-29)

*Pengfei He, Han Xu, Yue Xing, Hui Liu, Makoto Yamada, Jiliang Tang*

  In the domain of large language models (LLMs), in-context learning (ICL) has
been recognized for its innovative ability to adapt to new tasks, relying on
examples rather than retraining or fine-tuning. This paper delves into the
critical issue of ICL's susceptibility to data poisoning attacks, an area not
yet fully explored. We wonder whether ICL is vulnerable, with adversaries
capable of manipulating example data to degrade model performance. To address
this, we introduce ICLPoison, a specialized attacking framework conceived to
exploit the learning mechanisms of ICL. Our approach uniquely employs discrete
text perturbations to strategically influence the hidden states of LLMs during
the ICL process. We outline three representative strategies to implement
attacks under our framework, each rigorously evaluated across a variety of
models and tasks. Our comprehensive tests, including trials on the
sophisticated GPT-4 model, demonstrate that ICL's performance is significantly
compromised under our framework. These revelations indicate an urgent need for
enhanced defense mechanisms to safeguard the integrity and reliability of LLMs
in applications relying on in-context learning.


---

**[81. [2309.02465] Towards Foundational AI Models for Additive Manufacturing: Language
  Models for G-Code Debugging, Manipulation, and Comprehension](https://arxiv.org/pdf/2309.02465.pdf)** (Updated on 2023-09-07)

*Anushrut Jignasu, Kelly Marshall, Baskar Ganapathysubramanian, Aditya Balu, Chinmay Hegde, Adarsh Krishnamurthy*

  3D printing or additive manufacturing is a revolutionary technology that
enables the creation of physical objects from digital models. However, the
quality and accuracy of 3D printing depend on the correctness and efficiency of
the G-code, a low-level numerical control programming language that instructs
3D printers how to move and extrude material. Debugging G-code is a challenging
task that requires a syntactic and semantic understanding of the G-code format
and the geometry of the part to be printed. In this paper, we present the first
extensive evaluation of six state-of-the-art foundational large language models
(LLMs) for comprehending and debugging G-code files for 3D printing. We design
effective prompts to enable pre-trained LLMs to understand and manipulate
G-code and test their performance on various aspects of G-code debugging and
manipulation, including detection and correction of common errors and the
ability to perform geometric transformations. We analyze their strengths and
weaknesses for understanding complete G-code files. We also discuss the
implications and limitations of using LLMs for G-code comprehension.


---

**[82. [2504.06219] Can Performant LLMs Be Ethical? Quantifying the Impact of Web Crawling
  Opt-Outs](https://arxiv.org/pdf/2504.06219.pdf)** (Updated on 2025-04-09)

*Dongyang Fan, Vinko Sabolčec, Matin Ansaripour, Ayush Kumar Tarun, Martin Jaggi, Antoine Bosselut, Imanol Schlag*

  The increasing adoption of web crawling opt-outs by copyright holders of
online content raises critical questions about the impact of data compliance on
large language model (LLM) performance. However, little is known about how
these restrictions (and the resultant filtering of pretraining datasets) affect
the capabilities of models trained using these corpora. In this work, we
conceptualize this effect as the $\textit{data compliance gap}$ (DCG), which
quantifies the performance difference between models trained on datasets that
comply with web crawling opt-outs, and those that do not. We measure the data
compliance gap in two settings: pretraining models from scratch and continual
pretraining from existing compliant models (simulating a setting where
copyrighted data could be integrated later in pretraining). Our experiments
with 1.5B models show that, as of January 2025, compliance with web data
opt-outs does not degrade general knowledge acquisition (close to 0\% DCG).
However, in specialized domains such as biomedical research, excluding major
publishers leads to performance declines. These findings suggest that while
general-purpose LLMs can be trained to perform equally well using fully open
data, performance in specialized domains may benefit from access to
high-quality copyrighted sources later in training. Our study provides
empirical insights into the long-debated trade-off between data compliance and
downstream model performance, informing future discussions on AI training
practices and policy decisions.


---

**[83. [2011.12355] Lethean Attack: An Online Data Poisoning Technique](https://arxiv.org/pdf/2011.12355.pdf)** (Updated on 2020-11-26)

*Eyal Perry*

  Data poisoning is an adversarial scenario where an attacker feeds a specially
crafted sequence of samples to an online model in order to subvert learning. We
introduce Lethean Attack, a novel data poisoning technique that induces
catastrophic forgetting on an online model. We apply the attack in the context
of Test-Time Training, a modern online learning framework aimed for
generalization under distribution shifts. We present the theoretical rationale
and empirically compare it against other sample sequences that naturally induce
forgetting. Our results demonstrate that using lethean attacks, an adversary
could revert a test-time training model back to coin-flip accuracy performance
using a short sample sequence.


---

**[84. [2504.11182] Exploring Backdoor Attack and Defense for LLM-empowered Recommendations](https://arxiv.org/pdf/2504.11182.pdf)** (Updated on 2025-04-16)

*Liangbo Ning, Wenqi Fan, Qing Li*

  The fusion of Large Language Models (LLMs) with recommender systems (RecSys)
has dramatically advanced personalized recommendations and drawn extensive
attention. Despite the impressive progress, the safety of LLM-based RecSys
against backdoor attacks remains largely under-explored. In this paper, we
raise a new problem: Can a backdoor with a specific trigger be injected into
LLM-based Recsys, leading to the manipulation of the recommendation responses
when the backdoor trigger is appended to an item's title? To investigate the
vulnerabilities of LLM-based RecSys under backdoor attacks, we propose a new
attack framework termed Backdoor Injection Poisoning for RecSys (BadRec).
BadRec perturbs the items' titles with triggers and employs several fake users
to interact with these items, effectively poisoning the training set and
injecting backdoors into LLM-based RecSys. Comprehensive experiments reveal
that poisoning just 1% of the training data with adversarial examples is
sufficient to successfully implant backdoors, enabling manipulation of
recommendations. To further mitigate such a security threat, we propose a
universal defense strategy called Poison Scanner (P-Scanner). Specifically, we
introduce an LLM-based poison scanner to detect the poisoned items by
leveraging the powerful language understanding and rich knowledge of LLMs. A
trigger augmentation agent is employed to generate diverse synthetic triggers
to guide the poison scanner in learning domain-specific knowledge of the
poisoned item detection task. Extensive experiments on three real-world
datasets validate the effectiveness of the proposed P-Scanner.


---

**[85. [2406.06443] LLM Dataset Inference: Did you train on my dataset?](https://arxiv.org/pdf/2406.06443.pdf)** (Updated on 2024-06-11)

*Pratyush Maini, Hengrui Jia, Nicolas Papernot, Adam Dziedzic*

  The proliferation of large language models (LLMs) in the real world has come
with a rise in copyright cases against companies for training their models on
unlicensed data from the internet. Recent works have presented methods to
identify if individual text sequences were members of the model's training
data, known as membership inference attacks (MIAs). We demonstrate that the
apparent success of these MIAs is confounded by selecting non-members (text
sequences not used for training) belonging to a different distribution from the
members (e.g., temporally shifted recent Wikipedia articles compared with ones
used to train the model). This distribution shift makes membership inference
appear successful. However, most MIA methods perform no better than random
guessing when discriminating between members and non-members from the same
distribution (e.g., in this case, the same period of time). Even when MIAs
work, we find that different MIAs succeed at inferring membership of samples
from different distributions. Instead, we propose a new dataset inference
method to accurately identify the datasets used to train large language models.
This paradigm sits realistically in the modern-day copyright landscape, where
authors claim that an LLM is trained over multiple documents (such as a book)
written by them, rather than one particular paragraph. While dataset inference
shares many of the challenges of membership inference, we solve it by
selectively combining the MIAs that provide positive signal for a given
distribution, and aggregating them to perform a statistical test on a given
dataset. Our approach successfully distinguishes the train and test sets of
different subsets of the Pile with statistically significant p-values < 0.1,
without any false positives.


---

**[86. [2408.04870] ConfusedPilot: Confused Deputy Risks in RAG-based LLMs](https://arxiv.org/pdf/2408.04870.pdf)** (Updated on 2024-10-24)

*Ayush RoyChowdhury, Mulong Luo, Prateek Sahu, Sarbartha Banerjee, Mohit Tiwari*

  Retrieval augmented generation (RAG) is a process where a large language
model (LLM) retrieves useful information from a database and then generates the
responses. It is becoming popular in enterprise settings for daily business
operations. For example, Copilot for Microsoft 365 has accumulated millions of
businesses. However, the security implications of adopting such RAG-based
systems are unclear.
  In this paper, we introduce ConfusedPilot, a class of security
vulnerabilities of RAG systems that confuse Copilot and cause integrity and
confidentiality violations in its responses. First, we investigate a
vulnerability that embeds malicious text in the modified prompt in RAG,
corrupting the responses generated by the LLM. Second, we demonstrate a
vulnerability that leaks secret data, which leverages the caching mechanism
during retrieval. Third, we investigate how both vulnerabilities can be
exploited to propagate misinformation within the enterprise and ultimately
impact its operations, such as sales and manufacturing. We also discuss the
root cause of these attacks by investigating the architecture of a RAG-based
system. This study highlights the security vulnerabilities in today's RAG-based
systems and proposes design guidelines to secure future RAG-based systems.


---

**[87. [2409.01382] Automatic Detection of LLM-generated Code: A Case Study of Claude 3
  Haiku](https://arxiv.org/pdf/2409.01382.pdf)** (Updated on 2024-09-10)

*Musfiqur Rahman, SayedHassan Khatoonabadi, Ahmad Abdellatif, Emad Shihab*

  Using Large Language Models (LLMs) has gained popularity among software
developers for generating source code. However, the use of LLM-generated code
can introduce risks of adding suboptimal, defective, and vulnerable code. This
makes it necessary to devise methods for the accurate detection of
LLM-generated code. Toward this goal, we perform a case study of Claude 3 Haiku
(or Claude 3 for brevity) on CodeSearchNet dataset. We divide our analyses into
two parts: function-level and class-level. We extract 22 software metric
features, such as Code Lines and Cyclomatic Complexity, for each level of
granularity. We then analyze code snippets generated by Claude 3 and their
human-authored counterparts using the extracted features to understand how
unique the code generated by Claude 3 is. In the following step, we use the
unique characteristics of Claude 3-generated code to build Machine Learning
(ML) models and identify which features of the code snippets make them more
detectable by ML models. Our results indicate that Claude 3 tends to generate
longer functions, but shorter classes than humans, and this characteristic can
be used to detect Claude 3-generated code with ML models with 82% and 66%
accuracies for function-level and class-level snippets, respectively.


---

**[88. [2407.03232] Single Character Perturbations Break LLM Alignment](https://arxiv.org/pdf/2407.03232.pdf)** (Updated on 2024-07-04)

*Leon Lin, Hannah Brown, Kenji Kawaguchi, Michael Shieh*

  When LLMs are deployed in sensitive, human-facing settings, it is crucial
that they do not output unsafe, biased, or privacy-violating outputs. For this
reason, models are both trained and instructed to refuse to answer unsafe
prompts such as "Tell me how to build a bomb." We find that, despite these
safeguards, it is possible to break model defenses simply by appending a space
to the end of a model's input. In a study of eight open-source models, we
demonstrate that this acts as a strong enough attack to cause the majority of
models to generate harmful outputs with very high success rates. We examine the
causes of this behavior, finding that the contexts in which single spaces occur
in tokenized training data encourage models to generate lists when prompted,
overriding training signals to refuse to answer unsafe requests. Our findings
underscore the fragile state of current model alignment and promote the
importance of developing more robust alignment methods. Code and data will be
available at https://github.com/hannah-aught/space_attack.


---

**[89. [2501.18280] Jailbreaking LLMs' Safeguard with Universal Magic Words for Text
  Embedding Models](https://arxiv.org/pdf/2501.18280.pdf)** (Updated on 2025-02-11)

*Haoyu Liang, Youran Sun, Yunfeng Cai, Jun Zhu, Bo Zhang*

  The security issue of large language models (LLMs) has gained significant
attention recently, with various defense mechanisms developed to prevent
harmful outputs, among which safeguards based on text embedding models serve as
a fundamental defense. Through testing, we discover that the distribution of
text embedding model outputs is significantly biased with a large mean.
Inspired by this observation, we propose novel efficient methods to search for
universal magic words that can attack text embedding models. The universal
magic words as suffixes can move the embedding of any text towards the bias
direction, therefore manipulate the similarity of any text pair and mislead
safeguards. By appending magic words to user prompts and requiring LLMs to end
answers with magic words, attackers can jailbreak the safeguard. To eradicate
this security risk, we also propose defense mechanisms against such attacks,
which can correct the biased distribution of text embeddings in a train-free
manner.


---

**[90. [2406.06571] SUBLLM: A Novel Efficient Architecture with Token Sequence Subsampling
  for LLM](https://arxiv.org/pdf/2406.06571.pdf)** (Updated on 2024-08-26)

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

**[91. [2410.20142] Mask-based Membership Inference Attacks for Retrieval-Augmented
  Generation](https://arxiv.org/pdf/2410.20142.pdf)** (Updated on 2025-02-11)

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

**[92. [2311.09827] Cognitive Overload: Jailbreaking Large Language Models with Overloaded
  Logical Thinking](https://arxiv.org/pdf/2311.09827.pdf)** (Updated on 2024-03-01)

*Nan Xu, Fei Wang, Ben Zhou, Bang Zheng Li, Chaowei Xiao, Muhao Chen*

  While large language models (LLMs) have demonstrated increasing power, they
have also given rise to a wide range of harmful behaviors. As representatives,
jailbreak attacks can provoke harmful or unethical responses from LLMs, even
after safety alignment. In this paper, we investigate a novel category of
jailbreak attacks specifically designed to target the cognitive structure and
processes of LLMs. Specifically, we analyze the safety vulnerability of LLMs in
the face of (1) multilingual cognitive overload, (2) veiled expression, and (3)
effect-to-cause reasoning. Different from previous jailbreak attacks, our
proposed cognitive overload is a black-box attack with no need for knowledge of
model architecture or access to model weights. Experiments conducted on
AdvBench and MasterKey reveal that various LLMs, including both popular
open-source model Llama 2 and the proprietary model ChatGPT, can be compromised
through cognitive overload. Motivated by cognitive psychology work on managing
cognitive load, we further investigate defending cognitive overload attack from
two perspectives. Empirical studies show that our cognitive overload from three
perspectives can jailbreak all studied LLMs successfully, while existing
defense strategies can hardly mitigate the caused malicious uses effectively.


---

**[93. [2212.10007] CoCoMIC: Code Completion By Jointly Modeling In-file and Cross-file
  Context](https://arxiv.org/pdf/2212.10007.pdf)** (Updated on 2023-05-25)

*Yangruibo Ding, Zijian Wang, Wasi Uddin Ahmad, Murali Krishna Ramanathan, Ramesh Nallapati, Parminder Bhatia, Dan Roth, Bing Xiang*

  While pre-trained language models (LM) for code have achieved great success
in code completion, they generate code conditioned only on the contents within
the file, i.e., in-file context, but ignore the rich semantics in other files
within the same project, i.e., cross-file context, a critical source of
information that is especially useful in modern modular software development.
Such overlooking constrains code language models' capacity in code completion,
leading to unexpected behaviors such as generating hallucinated class member
functions or function calls with unexpected arguments. In this work, we develop
a cross-file context finder tool, CCFINDER, that effectively locates and
retrieves the most relevant cross-file context. We propose CoCoMIC, a framework
that incorporates cross-file context to learn the in-file and cross-file
context jointly on top of pretrained code LMs. CoCoMIC successfully improves
the existing code LM with a 33.94% relative increase in exact match and a
28.69% relative increase in identifier matching for code completion when the
cross-file context is provided.


---

**[94. [2503.01539] Pragmatic Inference Chain (PIC) Improving LLMs' Reasoning of Authentic
  Implicit Toxic Language](https://arxiv.org/pdf/2503.01539.pdf)** (Updated on 2025-03-04)

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

**[95. [2309.10544] Model Leeching: An Extraction Attack Targeting LLMs](https://arxiv.org/pdf/2309.10544.pdf)** (Updated on 2023-09-20)

*Lewis Birch, William Hackett, Stefan Trawicki, Neeraj Suri, Peter Garraghan*

  Model Leeching is a novel extraction attack targeting Large Language Models
(LLMs), capable of distilling task-specific knowledge from a target LLM into a
reduced parameter model. We demonstrate the effectiveness of our attack by
extracting task capability from ChatGPT-3.5-Turbo, achieving 73% Exact Match
(EM) similarity, and SQuAD EM and F1 accuracy scores of 75% and 87%,
respectively for only $50 in API cost. We further demonstrate the feasibility
of adversarial attack transferability from an extracted model extracted via
Model Leeching to perform ML attack staging against a target LLM, resulting in
an 11% increase to attack success rate when applied to ChatGPT-3.5-Turbo.


---

**[96. [2410.16848] ETHIC: Evaluating Large Language Models on Long-Context Tasks with High
  Information Coverage](https://arxiv.org/pdf/2410.16848.pdf)** (Updated on 2025-02-28)

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

**[97. [2409.11690] LLM-Powered Text Simulation Attack Against ID-Free Recommender Systems](https://arxiv.org/pdf/2409.11690.pdf)** (Updated on 2024-09-20)

*Zongwei Wang, Min Gao, Junliang Yu, Xinyi Gao, Quoc Viet Hung Nguyen, Shazia Sadiq, Hongzhi Yin*

  The ID-free recommendation paradigm has been proposed to address the
limitation that traditional recommender systems struggle to model cold-start
users or items with new IDs. Despite its effectiveness, this study uncovers
that ID-free recommender systems are vulnerable to the proposed Text Simulation
attack (TextSimu) which aims to promote specific target items. As a novel type
of text poisoning attack, TextSimu exploits large language models (LLM) to
alter the textual information of target items by simulating the characteristics
of popular items. It operates effectively in both black-box and white-box
settings, utilizing two key components: a unified popularity extraction module,
which captures the essential characteristics of popular items, and an N-persona
consistency simulation strategy, which creates multiple personas to
collaboratively synthesize refined promotional textual descriptions for target
items by simulating the popular items. To withstand TextSimu-like attacks, we
further explore the detection approach for identifying LLM-generated
promotional text. Extensive experiments conducted on three datasets demonstrate
that TextSimu poses a more significant threat than existing poisoning attacks,
while our defense method can detect malicious text of target items generated by
TextSimu. By identifying the vulnerability, we aim to advance the development
of more robust ID-free recommender systems.


---

**[98. [2405.06237] Risks of Practicing Large Language Models in Smart Grid: Threat Modeling
  and Validation](https://arxiv.org/pdf/2405.06237.pdf)** (Updated on 2024-11-19)

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

**[99. [2410.08811] PoisonBench: Assessing Large Language Model Vulnerability to Data
  Poisoning](https://arxiv.org/pdf/2410.08811.pdf)** (Updated on 2024-10-14)

*Tingchen Fu, Mrinank Sharma, Philip Torr, Shay B. Cohen, David Krueger, Fazl Barez*

  Preference learning is a central component for aligning current LLMs, but
this process can be vulnerable to data poisoning attacks. To address this
concern, we introduce PoisonBench, a benchmark for evaluating large language
models' susceptibility to data poisoning during preference learning. Data
poisoning attacks can manipulate large language model responses to include
hidden malicious content or biases, potentially causing the model to generate
harmful or unintended outputs while appearing to function normally. We deploy
two distinct attack types across eight realistic scenarios, assessing 21
widely-used models. Our findings reveal concerning trends: (1) Scaling up
parameter size does not inherently enhance resilience against poisoning
attacks; (2) There exists a log-linear relationship between the effects of the
attack and the data poison ratio; (3) The effect of data poisoning can
generalize to extrapolated triggers that are not included in the poisoned data.
These results expose weaknesses in current preference learning techniques,
highlighting the urgent need for more robust defenses against malicious models
and data manipulation.


---

**[100. [2411.10351] Bias Unveiled: Investigating Social Bias in LLM-Generated Code](https://arxiv.org/pdf/2411.10351.pdf)** (Updated on 2025-03-10)

*Lin Ling, Fazle Rabbi, Song Wang, Jinqiu Yang*

  Large language models (LLMs) have significantly advanced the field of
automated code generation. However, a notable research gap exists in evaluating
social biases that may be present in the code produced by LLMs. To solve this
issue, we propose a novel fairness framework, i.e., Solar, to assess and
mitigate the social biases of LLM-generated code. Specifically, Solar can
automatically generate test cases for quantitatively uncovering social biases
of the auto-generated code by LLMs. To quantify the severity of social biases
in generated code, we develop a dataset that covers a diverse set of social
problems. We applied Solar and the crafted dataset to four state-of-the-art
LLMs for code generation. Our evaluation reveals severe bias in the
LLM-generated code from all the subject LLMs. Furthermore, we explore several
prompting strategies for mitigating bias, including Chain-of-Thought (CoT)
prompting, combining positive role-playing with CoT prompting and dialogue with
Solar. Our experiments show that dialogue with Solar can effectively reduce
social bias in LLM-generated code by up to 90%. Last, we make the code and data
publicly available is highly extensible to evaluate new social problems.


---

**[101. [2404.00600] AI Act and Large Language Models (LLMs): When critical issues and
  privacy impact require human and ethical oversight](https://arxiv.org/pdf/2404.00600.pdf)** (Updated on 2024-04-03)

*Nicola Fabiano*

  The imposing evolution of artificial intelligence systems and, specifically,
of Large Language Models (LLM) makes it necessary to carry out assessments of
their level of risk and the impact they may have in the area of privacy,
personal data protection and at an ethical level, especially on the weakest and
most vulnerable. This contribution addresses human oversight, ethical
oversight, and privacy impact assessment.


---

**[102. [2411.03823] Both Text and Images Leaked! A Systematic Analysis of Multimodal LLM
  Data Contamination](https://arxiv.org/pdf/2411.03823.pdf)** (Updated on 2025-02-18)

*Dingjie Song, Sicheng Lai, Shunian Chen, Lichao Sun, Benyou Wang*

  The rapid progression of multimodal large language models (MLLMs) has
demonstrated superior performance on various multimodal benchmarks. However,
the issue of data contamination during training creates challenges in
performance evaluation and comparison. While numerous methods exist for
detecting models' contamination in large language models (LLMs), they are less
effective for MLLMs due to their various modalities and multiple training
phases. In this study, we introduce a multimodal data contamination detection
framework, MM-Detect, designed for MLLMs. Our experimental results indicate
that MM-Detect is quite effective and sensitive in identifying varying degrees
of contamination, and can highlight significant performance improvements due to
the leakage of multimodal benchmark training sets. Furthermore, we explore
whether the contamination originates from the base LLMs used by MLLMs or the
multimodal training phase, providing new insights into the stages at which
contamination may be introduced.


---

**[103. [2310.02469] PrivacyMind: Large Language Models Can Be Contextual Privacy Protection
  Learners](https://arxiv.org/pdf/2310.02469.pdf)** (Updated on 2024-10-29)

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

**[104. [2411.07518] LLM App Squatting and Cloning](https://arxiv.org/pdf/2411.07518.pdf)** (Updated on 2024-11-13)

*Yinglin Xie, Xinyi Hou, Yanjie Zhao, Kai Chen, Haoyu Wang*

  Impersonation tactics, such as app squatting and app cloning, have posed
longstanding challenges in mobile app stores, where malicious actors exploit
the names and reputations of popular apps to deceive users. With the rapid
growth of Large Language Model (LLM) stores like GPT Store and FlowGPT, these
issues have similarly surfaced, threatening the integrity of the LLM app
ecosystem. In this study, we present the first large-scale analysis of LLM app
squatting and cloning using our custom-built tool, LLMappCrazy. LLMappCrazy
covers 14 squatting generation techniques and integrates Levenshtein distance
and BERT-based semantic analysis to detect cloning by analyzing app functional
similarities. Using this tool, we generated variations of the top 1000 app
names and found over 5,000 squatting apps in the dataset. Additionally, we
observed 3,509 squatting apps and 9,575 cloning cases across six major
platforms. After sampling, we find that 18.7% of the squatting apps and 4.9% of
the cloning apps exhibited malicious behavior, including phishing, malware
distribution, fake content dissemination, and aggressive ad injection.


---

**[105. [2311.18215] Automatic Construction of a Korean Toxic Instruction Dataset for Ethical
  Tuning of Large Language Models](https://arxiv.org/pdf/2311.18215.pdf)** (Updated on 2023-12-01)

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

**[106. [2402.09910] DE-COP: Detecting Copyrighted Content in Language Models Training Data](https://arxiv.org/pdf/2402.09910.pdf)** (Updated on 2024-06-26)

*André V. Duarte, Xuandong Zhao, Arlindo L. Oliveira, Lei Li*

  How can we detect if copyrighted content was used in the training process of
a language model, considering that the training data is typically undisclosed?
We are motivated by the premise that a language model is likely to identify
verbatim excerpts from its training text. We propose DE-COP, a method to
determine whether a piece of copyrighted content was included in training.
DE-COP's core approach is to probe an LLM with multiple-choice questions, whose
options include both verbatim text and their paraphrases. We construct
BookTection, a benchmark with excerpts from 165 books published prior and
subsequent to a model's training cutoff, along with their paraphrases. Our
experiments show that DE-COP surpasses the prior best method by 9.6% in
detection performance (AUC) on models with logits available. Moreover, DE-COP
also achieves an average accuracy of 72% for detecting suspect books on fully
black-box models where prior methods give approximately 4% accuracy. The code
and datasets are available at https://github.com/LeiLiLab/DE-COP.


---

**[107. [2408.10608] Promoting Equality in Large Language Models: Identifying and Mitigating
  the Implicit Bias based on Bayesian Theory](https://arxiv.org/pdf/2408.10608.pdf)** (Updated on 2024-08-21)

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

**[108. [2405.18492] LLMs and Memorization: On Quality and Specificity of Copyright
  Compliance](https://arxiv.org/pdf/2405.18492.pdf)** (Updated on 2024-11-19)

*Felix B Mueller, Rebekka Görge, Anna K Bernzen, Janna C Pirk, Maximilian Poretschkin*

  Memorization in large language models (LLMs) is a growing concern. LLMs have
been shown to easily reproduce parts of their training data, including
copyrighted work. This is an important problem to solve, as it may violate
existing copyright laws as well as the European AI Act. In this work, we
propose a systematic analysis to quantify the extent of potential copyright
infringements in LLMs using European law as an example. Unlike previous work,
we evaluate instruction-finetuned models in a realistic end-user scenario. Our
analysis builds on a proposed threshold of 160 characters, which we borrow from
the German Copyright Service Provider Act and a fuzzy text matching algorithm
to identify potentially copyright-infringing textual reproductions. The
specificity of countermeasures against copyright infringement is analyzed by
comparing model behavior on copyrighted and public domain data. We investigate
what behaviors models show instead of producing protected text (such as refusal
or hallucination) and provide a first legal assessment of these behaviors. We
find that there are huge differences in copyright compliance, specificity, and
appropriate refusal among popular LLMs. Alpaca, GPT 4, GPT 3.5, and Luminous
perform best in our comparison, with OpenGPT-X, Alpaca, and Luminous producing
a particularly low absolute number of potential copyright violations. Code can
be found at https://github.com/felixbmuller/llms-memorization-copyright.


---

**[109. [2503.07237] LLM-C3MOD: A Human-LLM Collaborative System for Cross-Cultural Hate
  Speech Moderation](https://arxiv.org/pdf/2503.07237.pdf)** (Updated on 2025-03-11)

*Junyeong Park, Seogyeong Jeong, Seyoung Song, Yohan Lee, Alice Oh*

  Content moderation is a global challenge, yet major tech platforms prioritize
high-resource languages, leaving low-resource languages with scarce native
moderators. Since effective moderation depends on understanding contextual
cues, this imbalance increases the risk of improper moderation due to
non-native moderators' limited cultural understanding. Through a user study, we
identify that non-native moderators struggle with interpreting
culturally-specific knowledge, sentiment, and internet culture in the hate
speech moderation. To assist them, we present LLM-C3MOD, a human-LLM
collaborative pipeline with three steps: (1) RAG-enhanced cultural context
annotations; (2) initial LLM-based moderation; and (3) targeted human
moderation for cases lacking LLM consensus. Evaluated on a Korean hate speech
dataset with Indonesian and German participants, our system achieves 78%
accuracy (surpassing GPT-4o's 71% baseline), while reducing human workload by
83.6%. Notably, human moderators excel at nuanced contents where LLMs struggle.
Our findings suggest that non-native moderators, when properly supported by
LLMs, can effectively contribute to cross-cultural hate speech moderation.


---

**[110. [2504.07717] PR-Attack: Coordinated Prompt-RAG Attacks on Retrieval-Augmented
  Generation in Large Language Models via Bilevel Optimization](https://arxiv.org/pdf/2504.07717.pdf)** (Updated on 2025-04-18)

*Yang Jiao, Xiaodong Wang, Kai Yang*

  Large Language Models (LLMs) have demonstrated remarkable performance across
a wide range of applications, e.g., medical question-answering, mathematical
sciences, and code generation. However, they also exhibit inherent limitations,
such as outdated knowledge and susceptibility to hallucinations.
Retrieval-Augmented Generation (RAG) has emerged as a promising paradigm to
address these issues, but it also introduces new vulnerabilities. Recent
efforts have focused on the security of RAG-based LLMs, yet existing attack
methods face three critical challenges: (1) their effectiveness declines
sharply when only a limited number of poisoned texts can be injected into the
knowledge database, (2) they lack sufficient stealth, as the attacks are often
detectable by anomaly detection systems, which compromises their effectiveness,
and (3) they rely on heuristic approaches to generate poisoned texts, lacking
formal optimization frameworks and theoretic guarantees, which limits their
effectiveness and applicability. To address these issues, we propose
coordinated Prompt-RAG attack (PR-attack), a novel optimization-driven attack
that introduces a small number of poisoned texts into the knowledge database
while embedding a backdoor trigger within the prompt. When activated, the
trigger causes the LLM to generate pre-designed responses to targeted queries,
while maintaining normal behavior in other contexts. This ensures both high
effectiveness and stealth. We formulate the attack generation process as a
bilevel optimization problem leveraging a principled optimization framework to
develop optimal poisoned texts and triggers. Extensive experiments across
diverse LLMs and datasets demonstrate the effectiveness of PR-Attack, achieving
a high attack success rate even with a limited number of poisoned texts and
significantly improved stealth compared to existing methods.


---

**[111. [2404.11262] Sampling-based Pseudo-Likelihood for Membership Inference Attacks](https://arxiv.org/pdf/2404.11262.pdf)** (Updated on 2024-04-18)

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

**[112. [2403.08319] Knowledge Conflicts for LLMs: A Survey](https://arxiv.org/pdf/2403.08319.pdf)** (Updated on 2024-06-25)

*Rongwu Xu, Zehan Qi, Zhijiang Guo, Cunxiang Wang, Hongru Wang, Yue Zhang, Wei Xu*

  This survey provides an in-depth analysis of knowledge conflicts for large
language models (LLMs), highlighting the complex challenges they encounter when
blending contextual and parametric knowledge. Our focus is on three categories
of knowledge conflicts: context-memory, inter-context, and intra-memory
conflict. These conflicts can significantly impact the trustworthiness and
performance of LLMs, especially in real-world applications where noise and
misinformation are common. By categorizing these conflicts, exploring the
causes, examining the behaviors of LLMs under such conflicts, and reviewing
available solutions, this survey aims to shed light on strategies for improving
the robustness of LLMs, thereby serving as a valuable resource for advancing
research in this evolving area.


---

**[113. [2402.00888] Security and Privacy Challenges of Large Language Models: A Survey](https://arxiv.org/pdf/2402.00888.pdf)** (Updated on 2024-11-18)

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

**[114. [2410.15737] Who's Who: Large Language Models Meet Knowledge Conflicts in Practice](https://arxiv.org/pdf/2410.15737.pdf)** (Updated on 2024-10-22)

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

**[115. [2410.02841] Demonstration Attack against In-Context Learning for Code Intelligence](https://arxiv.org/pdf/2410.02841.pdf)** (Updated on 2024-10-07)

*Yifei Ge, Weisong Sun, Yihang Lou, Chunrong Fang, Yiran Zhang, Yiming Li, Xiaofang Zhang, Yang Liu, Zhihong Zhao, Zhenyu Chen*

  Recent advancements in large language models (LLMs) have revolutionized code
intelligence by improving programming productivity and alleviating challenges
faced by software developers. To further improve the performance of LLMs on
specific code intelligence tasks and reduce training costs, researchers reveal
a new capability of LLMs: in-context learning (ICL). ICL allows LLMs to learn
from a few demonstrations within a specific context, achieving impressive
results without parameter updating. However, the rise of ICL introduces new
security vulnerabilities in the code intelligence field. In this paper, we
explore a novel security scenario based on the ICL paradigm, where attackers
act as third-party ICL agencies and provide users with bad ICL content to
mislead LLMs outputs in code intelligence tasks. Our study demonstrates the
feasibility and risks of such a scenario, revealing how attackers can leverage
malicious demonstrations to construct bad ICL content and induce LLMs to
produce incorrect outputs, posing significant threats to system security. We
propose a novel method to construct bad ICL content called DICE, which is
composed of two stages: Demonstration Selection and Bad ICL Construction,
constructing targeted bad ICL content based on the user query and transferable
across different query inputs. Ultimately, our findings emphasize the critical
importance of securing ICL mechanisms to protect code intelligence systems from
adversarial manipulation.


---

**[116. [2406.05870] Machine Against the RAG: Jamming Retrieval-Augmented Generation with
  Blocker Documents](https://arxiv.org/pdf/2406.05870.pdf)** (Updated on 2025-03-11)

*Avital Shafran, Roei Schuster, Vitaly Shmatikov*

  Retrieval-augmented generation (RAG) systems respond to queries by retrieving
relevant documents from a knowledge database and applying an LLM to the
retrieved documents. We demonstrate that RAG systems that operate on databases
with untrusted content are vulnerable to denial-of-service attacks we call
jamming. An adversary can add a single ``blocker'' document to the database
that will be retrieved in response to a specific query and result in the RAG
system not answering this query, ostensibly because it lacks relevant
information or because the answer is unsafe.
  We describe and measure the efficacy of several methods for generating
blocker documents, including a new method based on black-box optimization. Our
method (1) does not rely on instruction injection, (2) does not require the
adversary to know the embedding or LLM used by the target RAG system, and (3)
does not employ an auxiliary LLM.
  We evaluate jamming attacks on several embeddings and LLMs and demonstrate
that the existing safety metrics for LLMs do not capture their vulnerability to
jamming. We then discuss defenses against blocker documents.


---

**[117. [2502.16691] Toward Responsible Federated Large Language Models: Leveraging a Safety
  Filter and Constitutional AI](https://arxiv.org/pdf/2502.16691.pdf)** (Updated on 2025-02-25)

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

**[118. [2504.02873] Short-PHD: Detecting Short LLM-generated Text with Topological Data
  Analysis After Off-topic Content Insertion](https://arxiv.org/pdf/2504.02873.pdf)** (Updated on 2025-04-07)

*Dongjun Wei, Minjia Mao, Xiao Fang, Michael Chau*

  The malicious usage of large language models (LLMs) has motivated the
detection of LLM-generated texts. Previous work in topological data analysis
shows that the persistent homology dimension (PHD) of text embeddings can serve
as a more robust and promising score than other zero-shot methods. However,
effectively detecting short LLM-generated texts remains a challenge. This paper
presents Short-PHD, a zero-shot LLM-generated text detection method tailored
for short texts. Short-PHD stabilizes the estimation of the previous PHD method
for short texts by inserting off-topic content before the given input text and
identifies LLM-generated text based on an established detection threshold.
Experimental results on both public and generated datasets demonstrate that
Short-PHD outperforms existing zero-shot methods in short LLM-generated text
detection. Implementation codes are available online.


---

**[119. [2502.14215] Towards Secure Program Partitioning for Smart Contracts with LLM's
  In-Context Learning](https://arxiv.org/pdf/2502.14215.pdf)** (Updated on 2025-02-21)

*Ye Liu, Yuqing Niu, Chengyan Ma, Ruidong Han, Wei Ma, Yi Li, Debin Gao, David Lo*

  Smart contracts are highly susceptible to manipulation attacks due to the
leakage of sensitive information. Addressing manipulation vulnerabilities is
particularly challenging because they stem from inherent data confidentiality
issues rather than straightforward implementation bugs. To tackle this by
preventing sensitive information leakage, we present PartitionGPT, the first
LLM-driven approach that combines static analysis with the in-context learning
capabilities of large language models (LLMs) to partition smart contracts into
privileged and normal codebases, guided by a few annotated sensitive data
variables. We evaluated PartitionGPT on 18 annotated smart contracts containing
99 sensitive functions. The results demonstrate that PartitionGPT successfully
generates compilable, and verified partitions for 78% of the sensitive
functions while reducing approximately 30% code compared to function-level
partitioning approach. Furthermore, we evaluated PartitionGPT on nine
real-world manipulation attacks that lead to a total loss of 25 million
dollars, PartitionGPT effectively prevents eight cases, highlighting its
potential for broad applicability and the necessity for secure program
partitioning during smart contract development to diminish manipulation
vulnerabilities.


---

**[120. [2411.03079] Utilizing Precise and Complete Code Context to Guide LLM in Automatic
  False Positive Mitigation](https://arxiv.org/pdf/2411.03079.pdf)** (Updated on 2024-11-06)

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

**[121. [2502.13416] Detecting LLM Fact-conflicting Hallucinations Enhanced by
  Temporal-logic-based Reasoning](https://arxiv.org/pdf/2502.13416.pdf)** (Updated on 2025-02-20)

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

**[122. [2409.04459] WET: Overcoming Paraphrasing Vulnerabilities in Embeddings-as-a-Service
  with Linear Transformation Watermarks](https://arxiv.org/pdf/2409.04459.pdf)** (Updated on 2024-09-10)

*Anudeex Shetty, Qiongkai Xu, Jey Han Lau*

  Embeddings-as-a-Service (EaaS) is a service offered by large language model
(LLM) developers to supply embeddings generated by LLMs. Previous research
suggests that EaaS is prone to imitation attacks -- attacks that clone the
underlying EaaS model by training another model on the queried embeddings. As a
result, EaaS watermarks are introduced to protect the intellectual property of
EaaS providers. In this paper, we first show that existing EaaS watermarks can
be removed by paraphrasing when attackers clone the model. Subsequently, we
propose a novel watermarking technique that involves linearly transforming the
embeddings, and show that it is empirically and theoretically robust against
paraphrasing.


---

**[123. [2401.03729] The Butterfly Effect of Altering Prompts: How Small Changes and
  Jailbreaks Affect Large Language Model Performance](https://arxiv.org/pdf/2401.03729.pdf)** (Updated on 2024-04-03)

*Abel Salinas, Fred Morstatter*

  Large Language Models (LLMs) are regularly being used to label data across
many domains and for myriad tasks. By simply asking the LLM for an answer, or
``prompting,'' practitioners are able to use LLMs to quickly get a response for
an arbitrary task. This prompting is done through a series of decisions by the
practitioner, from simple wording of the prompt, to requesting the output in a
certain data format, to jailbreaking in the case of prompts that address more
sensitive topics. In this work, we ask: do variations in the way a prompt is
constructed change the ultimate decision of the LLM? We answer this using a
series of prompt variations across a variety of text classification tasks. We
find that even the smallest of perturbations, such as adding a space at the end
of a prompt, can cause the LLM to change its answer. Further, we find that
requesting responses in XML and commonly used jailbreaks can have cataclysmic
effects on the data labeled by LLMs.


---

**[124. [2309.01446] Open Sesame! Universal Black Box Jailbreaking of Large Language Models](https://arxiv.org/pdf/2309.01446.pdf)** (Updated on 2024-08-06)

*Raz Lapid, Ron Langberg, Moshe Sipper*

  Large language models (LLMs), designed to provide helpful and safe responses,
often rely on alignment techniques to align with user intent and social
guidelines. Unfortunately, this alignment can be exploited by malicious actors
seeking to manipulate an LLM's outputs for unintended purposes. In this paper
we introduce a novel approach that employs a genetic algorithm (GA) to
manipulate LLMs when model architecture and parameters are inaccessible. The GA
attack works by optimizing a universal adversarial prompt that -- when combined
with a user's query -- disrupts the attacked model's alignment, resulting in
unintended and potentially harmful outputs. Our novel approach systematically
reveals a model's limitations and vulnerabilities by uncovering instances where
its responses deviate from expected behavior. Through extensive experiments we
demonstrate the efficacy of our technique, thus contributing to the ongoing
discussion on responsible AI development by providing a diagnostic tool for
evaluating and enhancing alignment of LLMs with human intent. To our knowledge
this is the first automated universal black box jailbreak attack.


---

**[125. [2211.08229] CorruptEncoder: Data Poisoning based Backdoor Attacks to Contrastive
  Learning](https://arxiv.org/pdf/2211.08229.pdf)** (Updated on 2024-03-04)

*Jinghuai Zhang, Hongbin Liu, Jinyuan Jia, Neil Zhenqiang Gong*

  Contrastive learning (CL) pre-trains general-purpose encoders using an
unlabeled pre-training dataset, which consists of images or image-text pairs.
CL is vulnerable to data poisoning based backdoor attacks (DPBAs), in which an
attacker injects poisoned inputs into the pre-training dataset so the encoder
is backdoored. However, existing DPBAs achieve limited effectiveness. In this
work, we take the first step to analyze the limitations of existing backdoor
attacks and propose new DPBAs called CorruptEncoder to CL. CorruptEncoder
introduces a new attack strategy to create poisoned inputs and uses a
theory-guided method to maximize attack effectiveness. Our experiments show
that CorruptEncoder substantially outperforms existing DPBAs. In particular,
CorruptEncoder is the first DPBA that achieves more than 90% attack success
rates with only a few (3) reference images and a small poisoning ratio 0.5%.
Moreover, we also propose a defense, called localized cropping, to defend
against DPBAs. Our results show that our defense can reduce the effectiveness
of DPBAs, but it sacrifices the utility of the encoder, highlighting the need
for new defenses.


---

**[126. [2402.08416] Pandora: Jailbreak GPTs by Retrieval Augmented Generation Poisoning](https://arxiv.org/pdf/2402.08416.pdf)** (Updated on 2024-02-14)

*Gelei Deng, Yi Liu, Kailong Wang, Yuekang Li, Tianwei Zhang, Yang Liu*

  Large Language Models~(LLMs) have gained immense popularity and are being
increasingly applied in various domains. Consequently, ensuring the security of
these models is of paramount importance. Jailbreak attacks, which manipulate
LLMs to generate malicious content, are recognized as a significant
vulnerability. While existing research has predominantly focused on direct
jailbreak attacks on LLMs, there has been limited exploration of indirect
methods. The integration of various plugins into LLMs, notably Retrieval
Augmented Generation~(RAG), which enables LLMs to incorporate external
knowledge bases into their response generation such as GPTs, introduces new
avenues for indirect jailbreak attacks.
  To fill this gap, we investigate indirect jailbreak attacks on LLMs,
particularly GPTs, introducing a novel attack vector named Retrieval Augmented
Generation Poisoning. This method, Pandora, exploits the synergy between LLMs
and RAG through prompt manipulation to generate unexpected responses. Pandora
uses maliciously crafted content to influence the RAG process, effectively
initiating jailbreak attacks. Our preliminary tests show that Pandora
successfully conducts jailbreak attacks in four different scenarios, achieving
higher success rates than direct attacks, with 64.3\% for GPT-3.5 and 34.8\%
for GPT-4.


---

**[127. [2407.01082] Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM
  Outputs](https://arxiv.org/pdf/2407.01082.pdf)** (Updated on 2025-03-21)

*Minh Nguyen, Andrew Baker, Clement Neo, Allen Roush, Andreas Kirsch, Ravid Shwartz-Ziv*

  Large Language Models (LLMs) generate text by sampling the next token from a
probability distribution over the vocabulary at each decoding step. Popular
sampling methods like top-p (nucleus sampling) often struggle to balance
quality and diversity, especially at higher temperatures which lead to
incoherent or repetitive outputs. We propose min-p sampling, a dynamic
truncation method that adjusts the sampling threshold based on the model's
confidence by using the top token's probability as a scaling factor. Our
experiments on benchmarks including GPQA, GSM8K, and AlpacaEval Creative
Writing show that min-p sampling improves both the quality and diversity of
generated text across different model families (Mistral and Llama 3) and model
sizes (1B to 123B parameters), especially at higher temperatures. Human
evaluations further show a clear preference for min-p sampling, in both text
quality and creativity. Min-p sampling has been adopted by popular open-source
LLM frameworks, including Hugging Face Transformers, VLLM, and many others,
highlighting its significant impact on improving text generation quality.


---

**[128. [2411.04299] An Empirical Study on Automatically Detecting AI-Generated Source Code:
  How Far Are We?](https://arxiv.org/pdf/2411.04299.pdf)** (Updated on 2024-11-08)

*Hyunjae Suh, Mahan Tafreshipour, Jiawei Li, Adithya Bhattiprolu, Iftekhar Ahmed*

  Artificial Intelligence (AI) techniques, especially Large Language Models
(LLMs), have started gaining popularity among researchers and software
developers for generating source code. However, LLMs have been shown to
generate code with quality issues and also incurred copyright/licensing
infringements. Therefore, detecting whether a piece of source code is written
by humans or AI has become necessary. This study first presents an empirical
analysis to investigate the effectiveness of the existing AI detection tools in
detecting AI-generated code. The results show that they all perform poorly and
lack sufficient generalizability to be practically deployed. Then, to improve
the performance of AI-generated code detection, we propose a range of
approaches, including fine-tuning the LLMs and machine learning-based
classification with static code metrics or code embedding generated from
Abstract Syntax Tree (AST). Our best model outperforms state-of-the-art
AI-generated code detector (GPTSniffer) and achieves an F1 score of 82.55. We
also conduct an ablation study on our best-performing model to investigate the
impact of different source code features on its performance.


---

**[129. [2504.00018] SandboxEval: Towards Securing Test Environment for Untrusted Code](https://arxiv.org/pdf/2504.00018.pdf)** (Updated on 2025-04-02)

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

**[130. [2203.07580] TSM: Measuring the Enticement of Honeyfiles with Natural Language
  Processing](https://arxiv.org/pdf/2203.07580.pdf)** (Updated on 2022-03-16)

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

**[131. [2403.08035] Harnessing Artificial Intelligence to Combat Online Hate: Exploring the
  Challenges and Opportunities of Large Language Models in Hate Speech
  Detection](https://arxiv.org/pdf/2403.08035.pdf)** (Updated on 2024-03-14)

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

**[132. [2408.04643] Risks, Causes, and Mitigations of Widespread Deployments of Large
  Language Models (LLMs): A Survey](https://arxiv.org/pdf/2408.04643.pdf)** (Updated on 2024-08-12)

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

**[133. [2504.04216] A Perplexity and Menger Curvature-Based Approach for Similarity
  Evaluation of Large Language Models](https://arxiv.org/pdf/2504.04216.pdf)** (Updated on 2025-04-09)

*Yuantao Zhang, Zhankui Yang*

  The rise of Large Language Models (LLMs) has brought about concerns regarding
copyright infringement and unethical practices in data and model usage. For
instance, slight modifications to existing LLMs may be used to falsely claim
the development of new models, leading to issues of model copying and
violations of ownership rights. This paper addresses these challenges by
introducing a novel metric for quantifying LLM similarity, which leverages
perplexity curves and differences in Menger curvature. Comprehensive
experiments validate the performance of our methodology, demonstrating its
superiority over baseline methods and its ability to generalize across diverse
models and domains. Furthermore, we highlight the capability of our approach in
detecting model replication through simulations, emphasizing its potential to
preserve the originality and integrity of LLMs. Code is available at
https://github.com/zyttt-coder/LLM_similarity.


---

**[134. [2410.05047] A test suite of prompt injection attacks for LLM-based machine
  translation](https://arxiv.org/pdf/2410.05047.pdf)** (Updated on 2024-11-20)

*Antonio Valerio Miceli-Barone, Zhifan Sun*

  LLM-based NLP systems typically work by embedding their input data into
prompt templates which contain instructions and/or in-context examples,
creating queries which are submitted to a LLM, and then parsing the LLM
response in order to generate the system outputs. Prompt Injection Attacks
(PIAs) are a type of subversion of these systems where a malicious user crafts
special inputs which interfere with the prompt templates, causing the LLM to
respond in ways unintended by the system designer.
  Recently, Sun and Miceli-Barone proposed a class of PIAs against LLM-based
machine translation. Specifically, the task is to translate questions from the
TruthfulQA test suite, where an adversarial prompt is prepended to the
questions, instructing the system to ignore the translation instruction and
answer the questions instead.
  In this test suite, we extend this approach to all the language pairs of the
WMT 2024 General Machine Translation task. Moreover, we include additional
attack formats in addition to the one originally studied.


---

**[135. [2311.09641] RLHFPoison: Reward Poisoning Attack for Reinforcement Learning with
  Human Feedback in Large Language Models](https://arxiv.org/pdf/2311.09641.pdf)** (Updated on 2024-06-21)

*Jiongxiao Wang, Junlin Wu, Muhao Chen, Yevgeniy Vorobeychik, Chaowei Xiao*

  Reinforcement Learning with Human Feedback (RLHF) is a methodology designed
to align Large Language Models (LLMs) with human preferences, playing an
important role in LLMs alignment. Despite its advantages, RLHF relies on human
annotators to rank the text, which can introduce potential security
vulnerabilities if any adversarial annotator (i.e., attackers) manipulates the
ranking score by up-ranking any malicious text to steer the LLM adversarially.
To assess the red-teaming of RLHF against human preference data poisoning, we
propose RankPoison, a poisoning attack method on candidates' selection of
preference rank flipping to reach certain malicious behaviors (e.g., generating
longer sequences, which can increase the computational cost). With poisoned
dataset generated by RankPoison, we can perform poisoning attacks on LLMs to
generate longer tokens without hurting the original safety alignment
performance. Moreover, applying RankPoison, we also successfully implement a
backdoor attack where LLMs can generate longer answers under questions with the
trigger word. Our findings highlight critical security challenges in RLHF,
underscoring the necessity for more robust alignment methods for LLMs.


---

**[136. [2501.05249] RAG-WM: An Efficient Black-Box Watermarking Approach for
  Retrieval-Augmented Generation of Large Language Models](https://arxiv.org/pdf/2501.05249.pdf)** (Updated on 2025-01-10)

*Peizhuo Lv, Mengjie Sun, Hao Wang, Xiaofeng Wang, Shengzhi Zhang, Yuxuan Chen, Kai Chen, Limin Sun*

  In recent years, tremendous success has been witnessed in Retrieval-Augmented
Generation (RAG), widely used to enhance Large Language Models (LLMs) in
domain-specific, knowledge-intensive, and privacy-sensitive tasks. However,
attackers may steal those valuable RAGs and deploy or commercialize them,
making it essential to detect Intellectual Property (IP) infringement. Most
existing ownership protection solutions, such as watermarks, are designed for
relational databases and texts. They cannot be directly applied to RAGs because
relational database watermarks require white-box access to detect IP
infringement, which is unrealistic for the knowledge base in RAGs. Meanwhile,
post-processing by the adversary's deployed LLMs typically destructs text
watermark information. To address those problems, we propose a novel black-box
"knowledge watermark" approach, named RAG-WM, to detect IP infringement of
RAGs. RAG-WM uses a multi-LLM interaction framework, comprising a Watermark
Generator, Shadow LLM & RAG, and Watermark Discriminator, to create watermark
texts based on watermark entity-relationship tuples and inject them into the
target RAG. We evaluate RAG-WM across three domain-specific and two
privacy-sensitive tasks on four benchmark LLMs. Experimental results show that
RAG-WM effectively detects the stolen RAGs in various deployed LLMs.
Furthermore, RAG-WM is robust against paraphrasing, unrelated content removal,
knowledge insertion, and knowledge expansion attacks. Lastly, RAG-WM can also
evade watermark detection approaches, highlighting its promising application in
detecting IP infringement of RAG systems.


---

**[137. [2407.10582] Boosting Zero-Shot Crosslingual Performance using LLM-Based
  Augmentations with Effective Data Selection](https://arxiv.org/pdf/2407.10582.pdf)** (Updated on 2024-07-16)

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

**[138. [2502.13499] Hidden Darkness in LLM-Generated Designs: Exploring Dark Patterns in
  Ecommerce Web Components Generated by LLMs](https://arxiv.org/pdf/2502.13499.pdf)** (Updated on 2025-02-20)

*Ziwei Chen, Jiawen Shen, Luna, Kristen Vaccaro*

  Recent work has highlighted the risks of LLM-generated content for a wide
range of harmful behaviors, including incorrect and harmful code. In this work,
we extend this by studying whether LLM-generated web design contains dark
patterns. This work evaluated designs of ecommerce web components generated by
four popular LLMs: Claude, GPT, Gemini, and Llama. We tested 13 commonly used
ecommerce components (e.g., search, product reviews) and used them as prompts
to generate a total of 312 components across all models. Over one-third of
generated components contain at least one dark pattern. The majority of dark
pattern strategies involve hiding crucial information, limiting users' actions,
and manipulating them into making decisions through a sense of urgency. Dark
patterns are also more frequently produced in components that are related to
company interests. These findings highlight the need for interventions to
prevent dark patterns during front-end code generation with LLMs and emphasize
the importance of expanding ethical design education to a broader audience.


---

**[139. [2406.05392] Deconstructing The Ethics of Large Language Models from Long-standing
  Issues to New-emerging Dilemmas: A Survey](https://arxiv.org/pdf/2406.05392.pdf)** (Updated on 2024-10-22)

*Chengyuan Deng, Yiqun Duan, Xin Jin, Heng Chang, Yijun Tian, Han Liu, Yichen Wang, Kuofeng Gao, Henry Peng Zou, Yiqiao Jin, Yijia Xiao, Shenghao Wu, Zongxing Xie, Weimin Lyu, Sihong He, Lu Cheng, Haohan Wang, Jun Zhuang*

  Large Language Models (LLMs) have achieved unparalleled success across
diverse language modeling tasks in recent years. However, this progress has
also intensified ethical concerns, impacting the deployment of LLMs in everyday
contexts. This paper provides a comprehensive survey of ethical challenges
associated with LLMs, from longstanding issues such as copyright infringement,
systematic bias, and data privacy, to emerging problems like truthfulness and
social norms. We critically analyze existing research aimed at understanding,
examining, and mitigating these ethical risks. Our survey underscores
integrating ethical standards and societal values into the development of LLMs,
thereby guiding the development of responsible and ethically aligned language
models.


---

**[140. [2309.08650] Adversarial Attacks on Tables with Entity Swap](https://arxiv.org/pdf/2309.08650.pdf)** (Updated on 2023-09-19)

*Aneta Koleva, Martin Ringsquandl, Volker Tresp*

  The capabilities of large language models (LLMs) have been successfully
applied in the context of table representation learning. The recently proposed
tabular language models have reported state-of-the-art results across various
tasks for table interpretation. However, a closer look into the datasets
commonly used for evaluation reveals an entity leakage from the train set into
the test set. Motivated by this observation, we explore adversarial attacks
that represent a more realistic inference setup. Adversarial attacks on text
have been shown to greatly affect the performance of LLMs, but currently, there
are no attacks targeting tabular language models. In this paper, we propose an
evasive entity-swap attack for the column type annotation (CTA) task. Our CTA
attack is the first black-box attack on tables, where we employ a
similarity-based sampling strategy to generate adversarial examples. The
experimental results show that the proposed attack generates up to a 70% drop
in performance.


---

**[141. [2502.07776] Auditing Prompt Caching in Language Model APIs](https://arxiv.org/pdf/2502.07776.pdf)** (Updated on 2025-02-12)

*Chenchen Gu, Xiang Lisa Li, Rohith Kuditipudi, Percy Liang, Tatsunori Hashimoto*

  Prompt caching in large language models (LLMs) results in data-dependent
timing variations: cached prompts are processed faster than non-cached prompts.
These timing differences introduce the risk of side-channel timing attacks. For
example, if the cache is shared across users, an attacker could identify cached
prompts from fast API response times to learn information about other users'
prompts. Because prompt caching may cause privacy leakage, transparency around
the caching policies of API providers is important. To this end, we develop and
conduct statistical audits to detect prompt caching in real-world LLM API
providers. We detect global cache sharing across users in seven API providers,
including OpenAI, resulting in potential privacy leakage about users' prompts.
Timing variations due to prompt caching can also result in leakage of
information about model architecture. Namely, we find evidence that OpenAI's
embedding model is a decoder-only Transformer, which was previously not
publicly known.


---

**[142. [2502.00406] ALU: Agentic LLM Unlearning](https://arxiv.org/pdf/2502.00406.pdf)** (Updated on 2025-02-04)

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

**[143. [2406.13236] Data Contamination Can Cross Language Barriers](https://arxiv.org/pdf/2406.13236.pdf)** (Updated on 2024-10-31)

*Feng Yao, Yufan Zhuang, Zihao Sun, Sunan Xu, Animesh Kumar, Jingbo Shang*

  The opacity in developing large language models (LLMs) is raising growing
concerns about the potential contamination of public benchmarks in the
pre-training data. Existing contamination detection methods are typically based
on the text overlap between training and evaluation data, which can be too
superficial to reflect deeper forms of contamination. In this paper, we first
present a cross-lingual form of contamination that inflates LLMs' performance
while evading current detection methods, deliberately injected by overfitting
LLMs on the translated versions of benchmark test sets. Then, we propose
generalization-based approaches to unmask such deeply concealed contamination.
Specifically, we examine the LLM's performance change after modifying the
original benchmark by replacing the false answer choices with correct ones from
other questions. Contaminated models can hardly generalize to such easier
situations, where the false choices can be \emph{not even wrong}, as all
choices are correct in their memorization. Experimental results demonstrate
that cross-lingual contamination can easily fool existing detection methods,
but not ours. In addition, we discuss the potential utilization of
cross-lingual contamination in interpreting LLMs' working mechanisms and in
post-training LLMs for enhanced multilingual capabilities. The code and dataset
we use can be obtained from \url{https://github.com/ShangDataLab/Deep-Contam}.


---

**[144. [2403.10020] Lost in Overlap: Exploring Logit-based Watermark Collision in LLMs](https://arxiv.org/pdf/2403.10020.pdf)** (Updated on 2025-02-06)

*Yiyang Luo, Ke Lin, Chao Gu, Jiahui Hou, Lijie Wen, Ping Luo*

  The proliferation of large language models (LLMs) in generating content
raises concerns about text copyright. Watermarking methods, particularly
logit-based approaches, embed imperceptible identifiers into text to address
these challenges. However, the widespread usage of watermarking across diverse
LLMs has led to an inevitable issue known as watermark collision during common
tasks, such as paraphrasing or translation. In this paper, we introduce
watermark collision as a novel and general philosophy for watermark attacks,
aimed at enhancing attack performance on top of any other attacking methods. We
also provide a comprehensive demonstration that watermark collision poses a
threat to all logit-based watermark algorithms, impacting not only specific
attack scenarios but also downstream applications.


---

**[145. [2402.14258] Eagle: Ethical Dataset Given from Real Interactions](https://arxiv.org/pdf/2402.14258.pdf)** (Updated on 2024-02-23)

*Masahiro Kaneko, Danushka Bollegala, Timothy Baldwin*

  Recent studies have demonstrated that large language models (LLMs) have
ethical-related problems such as social biases, lack of moral reasoning, and
generation of offensive content. The existing evaluation metrics and methods to
address these ethical challenges use datasets intentionally created by
instructing humans to create instances including ethical problems. Therefore,
the data does not reflect prompts that users actually provide when utilizing
LLM services in everyday contexts. This may not lead to the development of safe
LLMs that can address ethical challenges arising in real-world applications. In
this paper, we create Eagle datasets extracted from real interactions between
ChatGPT and users that exhibit social biases, toxicity, and immoral problems.
Our experiments show that Eagle captures complementary aspects, not covered by
existing datasets proposed for evaluation and mitigation of such ethical
challenges. Our code is publicly available at
https://huggingface.co/datasets/MasahiroKaneko/eagle.


---

**[146. [2308.11521] Self-Deception: Reverse Penetrating the Semantic Firewall of Large
  Language Models](https://arxiv.org/pdf/2308.11521.pdf)** (Updated on 2023-08-28)

*Zhenhua Wang, Wei Xie, Kai Chen, Baosheng Wang, Zhiwen Gui, Enze Wang*

  Large language models (LLMs), such as ChatGPT, have emerged with astonishing
capabilities approaching artificial general intelligence. While providing
convenience for various societal needs, LLMs have also lowered the cost of
generating harmful content. Consequently, LLM developers have deployed
semantic-level defenses to recognize and reject prompts that may lead to
inappropriate content. Unfortunately, these defenses are not foolproof, and
some attackers have crafted "jailbreak" prompts that temporarily hypnotize the
LLM into forgetting content defense rules and answering any improper questions.
To date, there is no clear explanation of the principles behind these
semantic-level attacks and defenses in both industry and academia.
  This paper investigates the LLM jailbreak problem and proposes an automatic
jailbreak method for the first time. We propose the concept of a semantic
firewall and provide three technical implementation approaches. Inspired by the
attack that penetrates traditional firewalls through reverse tunnels, we
introduce a "self-deception" attack that can bypass the semantic firewall by
inducing LLM to generate prompts that facilitate jailbreak. We generated a
total of 2,520 attack payloads in six languages (English, Russian, French,
Spanish, Chinese, and Arabic) across seven virtual scenarios, targeting the
three most common types of violations: violence, hate, and pornography. The
experiment was conducted on two models, namely the GPT-3.5-Turbo and GPT-4. The
success rates on the two models were 86.2% and 67%, while the failure rates
were 4.7% and 2.2%, respectively. This highlighted the effectiveness of the
proposed attack method. All experimental code and raw data will be released as
open-source to inspire future research. We believe that manipulating AI
behavior through carefully crafted prompts will become an important research
direction in the future.


---

**[147. [2502.15740] Detection of LLM-Generated Java Code Using Discretized Nested Bigrams](https://arxiv.org/pdf/2502.15740.pdf)** (Updated on 2025-02-25)

*Timothy Paek, Chilukuri Mohan*

  Large Language Models (LLMs) are currently used extensively to generate code
by professionals and students, motivating the development of tools to detect
LLM-generated code for applications such as academic integrity and
cybersecurity. We address this authorship attribution problem as a binary
classification task along with feature identification and extraction. We
propose new Discretized Nested Bigram Frequency features on source code groups
of various sizes. Compared to prior work, improvements are obtained by
representing sparse information in dense membership bins. Experimental
evaluation demonstrated that our approach significantly outperformed a commonly
used GPT code-detection API and baseline features, with accuracy exceeding 96%
compared to 72% and 79% respectively in detecting GPT-rewritten Java code
fragments for 976 files with GPT 3.5 and GPT4 using 12 features. We also
outperformed three prior works on code author identification in a 40-author
dataset. Our approach scales well to larger data sets, and we achieved 99%
accuracy and 0.999 AUC for 76,089 files and over 1,000 authors with GPT 4o
using 227 features.


---

**[148. [2412.06843] Semantic Loss Guided Data Efficient Supervised Fine Tuning for Safe
  Responses in LLMs](https://arxiv.org/pdf/2412.06843.pdf)** (Updated on 2024-12-12)

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

**[149. [2407.06411] If You Don't Understand It, Don't Use It: Eliminating Trojans with
  Filters Between Layers](https://arxiv.org/pdf/2407.06411.pdf)** (Updated on 2024-07-10)

*Adriano Hernandez*

  Large language models (LLMs) sometimes exhibit dangerous unintended
behaviors. Finding and fixing these is challenging because the attack surface
is massive -- it is not tractable to exhaustively search for all possible
inputs that may elicit such behavior. One specific and particularly challenging
case is that if data-poisoning-injected trojans, since there is no way to know
what they are to search for them. To our knowledge, there is no generally
applicable method to unlearn unknown trojans injected during pre-training. This
work seeks to provide a general purpose recipe (filters) and a specific
implementation (LoRA) filters that work in practice on small to medium sized
models. The focus is primarily empirical, though some perplexing behavior opens
the door to the fundamental question of how LLMs store and process information.
Not unexpectedly, we find that our filters work best on the residual stream and
the latest layers.


---

**[150. [2502.13141] UniGuardian: A Unified Defense for Detecting Prompt Injection, Backdoor
  Attacks and Adversarial Attacks in Large Language Models](https://arxiv.org/pdf/2502.13141.pdf)** (Updated on 2025-02-19)

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

**[151. [2412.15268] Enhancing LLM-based Hatred and Toxicity Detection with Meta-Toxic
  Knowledge Graph](https://arxiv.org/pdf/2412.15268.pdf)** (Updated on 2024-12-25)

*Yibo Zhao, Jiapeng Zhu, Can Xu, Xiang Li*

  The rapid growth of social media platforms has raised significant concerns
regarding online content toxicity. When Large Language Models (LLMs) are used
for toxicity detection, two key challenges emerge: 1) the absence of
domain-specific toxic knowledge leads to false negatives; 2) the excessive
sensitivity of LLMs to toxic speech results in false positives, limiting
freedom of speech. To address these issues, we propose a novel method called
MetaTox, leveraging graph search on a meta-toxic knowledge graph to enhance
hatred and toxicity detection. First, we construct a comprehensive meta-toxic
knowledge graph by utilizing LLMs to extract toxic information through a
three-step pipeline, with toxic benchmark datasets serving as corpora. Second,
we query the graph via retrieval and ranking processes to supplement accurate,
relevant toxic knowledge. Extensive experiments and in-depth case studies
across multiple datasets demonstrate that our MetaTox significantly decreases
the false positive rate while boosting overall toxicity detection performance.
Our code will be available soon.


---

**[152. [2310.10077] Prompt Packer: Deceiving LLMs through Compositional Instruction with
  Hidden Attacks](https://arxiv.org/pdf/2310.10077.pdf)** (Updated on 2023-10-17)

*Shuyu Jiang, Xingshu Chen, Rui Tang*

  Recently, Large language models (LLMs) with powerful general capabilities
have been increasingly integrated into various Web applications, while
undergoing alignment training to ensure that the generated content aligns with
user intent and ethics. Unfortunately, they remain the risk of generating
harmful content like hate speech and criminal activities in practical
applications. Current approaches primarily rely on detecting, collecting, and
training against harmful prompts to prevent such risks. However, they typically
focused on the "superficial" harmful prompts with a solitary intent, ignoring
composite attack instructions with multiple intentions that can easily elicit
harmful content in real-world scenarios. In this paper, we introduce an
innovative technique for obfuscating harmful instructions: Compositional
Instruction Attacks (CIA), which refers to attacking by combination and
encapsulation of multiple instructions. CIA hides harmful prompts within
instructions of harmless intentions, making it impossible for the model to
identify underlying malicious intentions. Furthermore, we implement two
transformation methods, known as T-CIA and W-CIA, to automatically disguise
harmful instructions as talking or writing tasks, making them appear harmless
to LLMs. We evaluated CIA on GPT-4, ChatGPT, and ChatGLM2 with two safety
assessment datasets and two harmful prompt datasets. It achieves an attack
success rate of 95%+ on safety assessment datasets, and 83%+ for GPT-4, 91%+
for ChatGPT (gpt-3.5-turbo backed) and ChatGLM2-6B on harmful prompt datasets.
Our approach reveals the vulnerability of LLMs to such compositional
instruction attacks that harbor underlying harmful intentions, contributing
significantly to LLM security development. Warning: this paper may contain
offensive or upsetting content!


---

**[153. [2203.05367] TIDF-DLPM: Term and Inverse Document Frequency based Data Leakage
  Prevention Model](https://arxiv.org/pdf/2203.05367.pdf)** (Updated on 2022-03-11)

*Ishu Gupta, Sloni Mittal, Ankit Tiwari, Priya Agarwal, Ashutosh Kumar Singh*

  Confidentiality of the data is being endangered as it has been categorized
into false categories which might get leaked to an unauthorized party. For this
reason, various organizations are mainly implementing data leakage prevention
systems (DLPs). Firewalls and intrusion detection systems are being outdated
versions of security mechanisms. The data which are being used, in sending
state or are rest are being monitored by DLPs. The confidential data is
prevented with the help of neighboring contexts and contents of DLPs. In this
paper, a semantic-based approach is used to classify data based on the
statistical data leakage prevention model. To detect involved private data,
statistical analysis is being used to contribute secure mechanisms in the
environment of data leakage. The favored Frequency-Inverse Document Frequency
(TF-IDF) is the facts and details recapture function to arrange documents under
particular topics. The results showcase that a similar statistical DLP approach
could appropriately classify documents in case of extent alteration as well as
interchanged documents.


---

**[154. [2403.16139] A Little Leak Will Sink a Great Ship: Survey of Transparency for Large
  Language Models from Start to Finish](https://arxiv.org/pdf/2403.16139.pdf)** (Updated on 2024-03-26)

*Masahiro Kaneko, Timothy Baldwin*

  Large Language Models (LLMs) are trained on massive web-crawled corpora. This
poses risks of leakage, including personal information, copyrighted texts, and
benchmark datasets. Such leakage leads to undermining human trust in AI due to
potential unauthorized generation of content or overestimation of performance.
We establish the following three criteria concerning the leakage issues: (1)
leakage rate: the proportion of leaked data in training data, (2) output rate:
the ease of generating leaked data, and (3) detection rate: the detection
performance of leaked versus non-leaked data. Despite the leakage rate being
the origin of data leakage issues, it is not understood how it affects the
output rate and detection rate. In this paper, we conduct an experimental
survey to elucidate the relationship between the leakage rate and both the
output rate and detection rate for personal information, copyrighted texts, and
benchmark data. Additionally, we propose a self-detection approach that uses
few-shot learning in which LLMs detect whether instances are present or absent
in their training data, in contrast to previous methods that do not employ
explicit learning. To explore the ease of generating leaked information, we
create a dataset of prompts designed to elicit personal information,
copyrighted text, and benchmarks from LLMs. Our experiments reveal that LLMs
produce leaked information in most cases despite less such data in their
training set. This indicates even small amounts of leaked data can greatly
affect outputs. Our self-detection method showed superior performance compared
to existing detection methods.


---

**[155. [2405.20413] Jailbreaking Large Language Models Against Moderation Guardrails via
  Cipher Characters](https://arxiv.org/pdf/2405.20413.pdf)** (Updated on 2024-06-03)

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

**[156. [2407.02402] Assessing the Code Clone Detection Capability of Large Language Models](https://arxiv.org/pdf/2407.02402.pdf)** (Updated on 2024-07-03)

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

**[157. [2502.07340] Aligning Large Language Models to Follow Instructions and Hallucinate
  Less via Effective Data Filtering](https://arxiv.org/pdf/2502.07340.pdf)** (Updated on 2025-02-18)

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

**[158. [2406.18326] PaCoST: Paired Confidence Significance Testing for Benchmark
  Contamination Detection in Large Language Models](https://arxiv.org/pdf/2406.18326.pdf)** (Updated on 2025-03-19)

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

**[159. [2502.06215] LessLeak-Bench: A First Investigation of Data Leakage in LLMs Across 83
  Software Engineering Benchmarks](https://arxiv.org/pdf/2502.06215.pdf)** (Updated on 2025-02-11)

*Xin Zhou, Martin Weyssow, Ratnadira Widyasari, Ting Zhang, Junda He, Yunbo Lyu, Jianming Chang, Beiqi Zhang, Dan Huang, David Lo*

  Large Language Models (LLMs) are widely utilized in software engineering (SE)
tasks, such as code generation and automated program repair. However, their
reliance on extensive and often undisclosed pre-training datasets raises
significant concerns about data leakage, where the evaluation benchmark data is
unintentionally ``seen'' by LLMs during the model's construction phase. The
data leakage issue could largely undermine the validity of LLM-based research
and evaluations. Despite the increasing use of LLMs in the SE community, there
is no comprehensive study that assesses the extent of data leakage in SE
benchmarks for LLMs yet. To address this gap, this paper presents the first
large-scale analysis of data leakage in 83 SE benchmarks concerning LLMs. Our
results show that in general, data leakage in SE benchmarks is minimal, with
average leakage ratios of only 4.8\%, 2.8\%, and 0.7\% for Python, Java, and
C/C++ benchmarks, respectively. However, some benchmarks exhibit relatively
higher leakage ratios, which raises concerns about their bias in evaluation.
For instance, QuixBugs and BigCloneBench have leakage ratios of 100.0\% and
55.7\%, respectively. Furthermore, we observe that data leakage has a
substantial impact on LLM evaluation. We also identify key causes of high data
leakage, such as the direct inclusion of benchmark data in pre-training
datasets and the use of coding platforms like LeetCode for benchmark
construction. To address the data leakage, we introduce
\textbf{LessLeak-Bench}, a new benchmark that removes leaked samples from the
83 SE benchmarks, enabling more reliable LLM evaluations in future research.
Our study enhances the understanding of data leakage in SE benchmarks and
provides valuable insights for future research involving LLMs in SE.


---

**[160. [2401.04136] The Stronger the Diffusion Model, the Easier the Backdoor: Data
  Poisoning to Induce Copyright Breaches Without Adjusting Finetuning Pipeline](https://arxiv.org/pdf/2401.04136.pdf)** (Updated on 2024-05-28)

*Haonan Wang, Qianli Shen, Yao Tong, Yang Zhang, Kenji Kawaguchi*

  The commercialization of text-to-image diffusion models (DMs) brings forth
potential copyright concerns. Despite numerous attempts to protect DMs from
copyright issues, the vulnerabilities of these solutions are underexplored. In
this study, we formalized the Copyright Infringement Attack on generative AI
models and proposed a backdoor attack method, SilentBadDiffusion, to induce
copyright infringement without requiring access to or control over training
processes. Our method strategically embeds connections between pieces of
copyrighted information and text references in poisoning data while carefully
dispersing that information, making the poisoning data inconspicuous when
integrated into a clean dataset. Our experiments show the stealth and efficacy
of the poisoning data. When given specific text prompts, DMs trained with a
poisoning ratio of 0.20% can produce copyrighted images. Additionally, the
results reveal that the more sophisticated the DMs are, the easier the success
of the attack becomes. These findings underline potential pitfalls in the
prevailing copyright protection strategies and underscore the necessity for
increased scrutiny to prevent the misuse of DMs.


---

**[161. [2503.21824] Protecting Your Video Content: Disrupting Automated Video-based LLM
  Annotations](https://arxiv.org/pdf/2503.21824.pdf)** (Updated on 2025-03-31)

*Haitong Liu, Kuofeng Gao, Yang Bai, Jinmin Li, Jinxiao Shan, Tao Dai, Shu-Tao Xia*

  Recently, video-based large language models (video-based LLMs) have achieved
impressive performance across various video comprehension tasks. However, this
rapid advancement raises significant privacy and security concerns,
particularly regarding the unauthorized use of personal video data in automated
annotation by video-based LLMs. These unauthorized annotated video-text pairs
can then be used to improve the performance of downstream tasks, such as
text-to-video generation. To safeguard personal videos from unauthorized use,
we propose two series of protective video watermarks with imperceptible
adversarial perturbations, named Ramblings and Mutes. Concretely, Ramblings aim
to mislead video-based LLMs into generating inaccurate captions for the videos,
thereby degrading the quality of video annotations through inconsistencies
between video content and captions. Mutes, on the other hand, are designed to
prompt video-based LLMs to produce exceptionally brief captions, lacking
descriptive detail. Extensive experiments demonstrate that our video
watermarking methods effectively protect video data by significantly reducing
video annotation performance across various video-based LLMs, showcasing both
stealthiness and robustness in protecting personal video content. Our code is
available at https://github.com/ttthhl/Protecting_Your_Video_Content.


---

**[162. [2305.07609] Is ChatGPT Fair for Recommendation? Evaluating Fairness in Large
  Language Model Recommendation](https://arxiv.org/pdf/2305.07609.pdf)** (Updated on 2023-10-18)

*Jizhi Zhang, Keqin Bao, Yang Zhang, Wenjie Wang, Fuli Feng, Xiangnan He*

  The remarkable achievements of Large Language Models (LLMs) have led to the
emergence of a novel recommendation paradigm -- Recommendation via LLM
(RecLLM). Nevertheless, it is important to note that LLMs may contain social
prejudices, and therefore, the fairness of recommendations made by RecLLM
requires further investigation. To avoid the potential risks of RecLLM, it is
imperative to evaluate the fairness of RecLLM with respect to various sensitive
attributes on the user side. Due to the differences between the RecLLM paradigm
and the traditional recommendation paradigm, it is problematic to directly use
the fairness benchmark of traditional recommendation. To address the dilemma,
we propose a novel benchmark called Fairness of Recommendation via LLM
(FaiRLLM). This benchmark comprises carefully crafted metrics and a dataset
that accounts for eight sensitive attributes1 in two recommendation scenarios:
music and movies. By utilizing our FaiRLLM benchmark, we conducted an
evaluation of ChatGPT and discovered that it still exhibits unfairness to some
sensitive attributes when generating recommendations. Our code and dataset can
be found at https://github.com/jizhi-zhang/FaiRLLM.


---

**[163. [2308.10443] Using Large Language Models for Cybersecurity Capture-The-Flag
  Challenges and Certification Questions](https://arxiv.org/pdf/2308.10443.pdf)** (Updated on 2023-08-22)

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

**[164. [2504.13774] DP2Unlearning: An Efficient and Guaranteed Unlearning Framework for LLMs](https://arxiv.org/pdf/2504.13774.pdf)** (Updated on 2025-04-21)

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

**[165. [2410.16186] Contamination Report for Multilingual Benchmarks](https://arxiv.org/pdf/2410.16186.pdf)** (Updated on 2024-10-22)

*Sanchit Ahuja, Varun Gumma, Sunayana Sitaram*

  Benchmark contamination refers to the presence of test datasets in Large
Language Model (LLM) pre-training or post-training data. Contamination can lead
to inflated scores on benchmarks, compromising evaluation results and making it
difficult to determine the capabilities of models. In this work, we study the
contamination of popular multilingual benchmarks in LLMs that support multiple
languages. We use the Black Box test to determine whether $7$ frequently used
multilingual benchmarks are contaminated in $7$ popular open and closed LLMs
and find that almost all models show signs of being contaminated with almost
all the benchmarks we test. Our findings can help the community determine the
best set of benchmarks to use for multilingual evaluation.


---

**[166. [2502.13172] Unveiling Privacy Risks in LLM Agent Memory](https://arxiv.org/pdf/2502.13172.pdf)** (Updated on 2025-02-20)

*Bo Wang, Weiyi He, Pengfei He, Shenglai Zeng, Zhen Xiang, Yue Xing, Jiliang Tang*

  Large Language Model (LLM) agents have become increasingly prevalent across
various real-world applications. They enhance decision-making by storing
private user-agent interactions in the memory module for demonstrations,
introducing new privacy risks for LLM agents. In this work, we systematically
investigate the vulnerability of LLM agents to our proposed Memory EXTRaction
Attack (MEXTRA) under a black-box setting. To extract private information from
memory, we propose an effective attacking prompt design and an automated prompt
generation method based on different levels of knowledge about the LLM agent.
Experiments on two representative agents demonstrate the effectiveness of
MEXTRA. Moreover, we explore key factors influencing memory leakage from both
the agent's and the attacker's perspectives. Our findings highlight the urgent
need for effective memory safeguards in LLM agent design and deployment.


---

**[167. [2410.07137] Cheating Automatic LLM Benchmarks: Null Models Achieve High Win Rates](https://arxiv.org/pdf/2410.07137.pdf)** (Updated on 2025-03-04)

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

**[168. [2502.07045] Scalable and Ethical Insider Threat Detection through Data Synthesis and
  Analysis by LLMs](https://arxiv.org/pdf/2502.07045.pdf)** (Updated on 2025-04-08)

*Haywood Gelman, John D. Hastings*

  Insider threats wield an outsized influence on organizations,
disproportionate to their small numbers. This is due to the internal access
insiders have to systems, information, and infrastructure. %One example of this
influence is where anonymous respondents submit web-based job search site
reviews, an insider threat risk to organizations. Signals for such risks may be
found in anonymous submissions to public web-based job search site reviews.
This research studies the potential for large language models (LLMs) to analyze
and detect insider threat sentiment within job site reviews. Addressing ethical
data collection concerns, this research utilizes synthetic data generation
using LLMs alongside existing job review datasets. A comparative analysis of
sentiment scores generated by LLMs is benchmarked against expert human scoring.
Findings reveal that LLMs demonstrate alignment with human evaluations in most
cases, thus effectively identifying nuanced indicators of threat sentiment. The
performance is lower on human-generated data than synthetic data, suggesting
areas for improvement in evaluating real-world data. Text diversity analysis
found differences between human-generated and LLM-generated datasets, with
synthetic data exhibiting somewhat lower diversity. Overall, the results
demonstrate the applicability of LLMs to insider threat detection, and a
scalable solution for insider sentiment testing by overcoming ethical and
logistical barriers tied to data acquisition.


---

**[169. [2412.07261] MemHunter: Automated and Verifiable Memorization Detection at
  Dataset-scale in LLMs](https://arxiv.org/pdf/2412.07261.pdf)** (Updated on 2025-02-18)

*Zhenpeng Wu, Jian Lou, Zibin Zheng, Chuan Chen*

  Large language models (LLMs) have been shown to memorize and reproduce
content from their training data, raising significant privacy concerns,
especially with web-scale datasets. Existing methods for detecting memorization
are primarily sample-specific, relying on manually crafted or discretely
optimized memory-inducing prompts generated on a per-sample basis, which become
impractical for dataset-level detection due to the prohibitive computational
cost of iterating through all samples. In real-world scenarios, data owners may
need to verify whether a susceptible LLM has memorized their dataset,
particularly if the LLM may have collected the data from the web without
authorization. To address this, we introduce MemHunter, which trains a
memory-inducing LLM and employs hypothesis testing to efficiently detect
memorization at the dataset level, without requiring sample-specific memory
inducing. Experiments on models like Pythia and Llama demonstrate that
MemHunter can extract up to 40% more training data than existing methods under
constrained time resources and reduce search time by up to 80% when integrated
as a plug-in. Crucially, MemHunter is the first method capable of dataset-level
memorization detection, providing a critical tool for assessing privacy risks
in LLMs powered by large-scale datasets.


---

**[170. [2503.15551] Efficient but Vulnerable: Benchmarking and Defending LLM Batch Prompting
  Attack](https://arxiv.org/pdf/2503.15551.pdf)** (Updated on 2025-03-21)

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

**[171. [2305.01639] Privacy-Preserving In-Context Learning for Large Language Models](https://arxiv.org/pdf/2305.01639.pdf)** (Updated on 2023-10-03)

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

**[172. [2410.13722] Persistent Pre-Training Poisoning of LLMs](https://arxiv.org/pdf/2410.13722.pdf)** (Updated on 2024-10-18)

*Yiming Zhang, Javier Rando, Ivan Evtimov, Jianfeng Chi, Eric Michael Smith, Nicholas Carlini, Florian Tramèr, Daphne Ippolito*

  Large language models are pre-trained on uncurated text datasets consisting
of trillions of tokens scraped from the Web. Prior work has shown that: (1)
web-scraped pre-training datasets can be practically poisoned by malicious
actors; and (2) adversaries can compromise language models after poisoning
fine-tuning datasets. Our work evaluates for the first time whether language
models can also be compromised during pre-training, with a focus on the
persistence of pre-training attacks after models are fine-tuned as helpful and
harmless chatbots (i.e., after SFT and DPO). We pre-train a series of LLMs from
scratch to measure the impact of a potential poisoning adversary under four
different attack objectives (denial-of-service, belief manipulation,
jailbreaking, and prompt stealing), and across a wide range of model sizes
(from 600M to 7B). Our main result is that poisoning only 0.1% of a model's
pre-training dataset is sufficient for three out of four attacks to measurably
persist through post-training. Moreover, simple attacks like denial-of-service
persist through post-training with a poisoning rate of only 0.001%.


---

**[173. [2410.03537] Ward: Provable RAG Dataset Inference via LLM Watermarks](https://arxiv.org/pdf/2410.03537.pdf)** (Updated on 2025-02-26)

*Nikola Jovanović, Robin Staab, Maximilian Baader, Martin Vechev*

  RAG enables LLMs to easily incorporate external data, raising concerns for
data owners regarding unauthorized usage of their content. The challenge of
detecting such unauthorized usage remains underexplored, with datasets and
methods from adjacent fields being ill-suited for its study. We take several
steps to bridge this gap. First, we formalize this problem as (black-box) RAG
Dataset Inference (RAG-DI). We then introduce a novel dataset designed for
realistic benchmarking of RAG-DI methods, alongside a set of baselines.
Finally, we propose Ward, a method for RAG-DI based on LLM watermarks that
equips data owners with rigorous statistical guarantees regarding their
dataset's misuse in RAG corpora. Ward consistently outperforms all baselines,
achieving higher accuracy, superior query efficiency and robustness. Our work
provides a foundation for future studies of RAG-DI and highlights LLM
watermarks as a promising approach to this problem.


---

**[174. [2105.13530] A BIC-based Mixture Model Defense against Data Poisoning Attacks on
  Classifiers](https://arxiv.org/pdf/2105.13530.pdf)** (Updated on 2022-05-13)

*Xi Li, David J. Miller, Zhen Xiang, George Kesidis*

  Data Poisoning (DP) is an effective attack that causes trained classifiers to
misclassify their inputs. DP attacks significantly degrade a classifier's
accuracy by covertly injecting attack samples into the training set. Broadly
applicable to different classifier structures, without strong assumptions about
the attacker, an {\it unsupervised} Bayesian Information Criterion (BIC)-based
mixture model defense against "error generic" DP attacks is herein proposed
that: 1) addresses the most challenging {\it embedded} DP scenario wherein, if
DP is present, the poisoned samples are an {\it a priori} unknown subset of the
training set, and with no clean validation set available; 2) applies a mixture
model both to well-fit potentially multi-modal class distributions and to
capture poisoned samples within a small subset of the mixture components; 3)
jointly identifies poisoned components and samples by minimizing the BIC cost
defined over the whole training set, with the identified poisoned data removed
prior to classifier training. Our experimental results, for various classifier
structures and benchmark datasets, demonstrate the effectiveness and
universality of our defense under strong DP attacks, as well as its superiority
over other works.


---

**[175. [2310.06202] GPT-who: An Information Density-based Machine-Generated Text Detector](https://arxiv.org/pdf/2310.06202.pdf)** (Updated on 2024-04-05)

*Saranya Venkatraman, Adaku Uchendu, Dongwon Lee*

  The Uniform Information Density (UID) principle posits that humans prefer to
spread information evenly during language production. We examine if this UID
principle can help capture differences between Large Language Models
(LLMs)-generated and human-generated texts. We propose GPT-who, the first
psycholinguistically-inspired domain-agnostic statistical detector. This
detector employs UID-based features to model the unique statistical signature
of each LLM and human author for accurate detection. We evaluate our method
using 4 large-scale benchmark datasets and find that GPT-who outperforms
state-of-the-art detectors (both statistical- & non-statistical) such as GLTR,
GPTZero, DetectGPT, OpenAI detector, and ZeroGPT by over $20$% across domains.
In addition to better performance, it is computationally inexpensive and
utilizes an interpretable representation of text articles. We find that GPT-who
can distinguish texts generated by very sophisticated LLMs, even when the
overlying text is indiscernible. UID-based measures for all datasets and code
are available at https://github.com/saranya-venkatraman/gpt-who.


---

**[176. [2402.12936] Measuring Impacts of Poisoning on Model Parameters and Neuron
  Activations: A Case Study of Poisoning CodeBERT](https://arxiv.org/pdf/2402.12936.pdf)** (Updated on 2024-03-06)

*Aftab Hussain, Md Rafiqul Islam Rabin, Navid Ayoobi, Mohammad Amin Alipour*

  Large language models (LLMs) have revolutionized software development
practices, yet concerns about their safety have arisen, particularly regarding
hidden backdoors, aka trojans. Backdoor attacks involve the insertion of
triggers into training data, allowing attackers to manipulate the behavior of
the model maliciously. In this paper, we focus on analyzing the model
parameters to detect potential backdoor signals in code models. Specifically,
we examine attention weights and biases, activation values, and context
embeddings of the clean and poisoned CodeBERT models. Our results suggest
noticeable patterns in activation values and context embeddings of poisoned
samples for the poisoned CodeBERT model; however, attention weights and biases
do not show any significant differences. This work contributes to ongoing
efforts in white-box detection of backdoor signals in LLMs of code through the
analysis of parameters and activations.


---

**[177. [2502.15830] Show Me Your Code! Kill Code Poisoning: A Lightweight Method Based on
  Code Naturalness](https://arxiv.org/pdf/2502.15830.pdf)** (Updated on 2025-02-25)

*Weisong Sun, Yuchen Chen, Mengzhe Yuan, Chunrong Fang, Zhenpeng Chen, Chong Wang, Yang Liu, Baowen Xu, Zhenyu Chen*

  Neural code models (NCMs) have demonstrated extraordinary capabilities in
code intelligence tasks. Meanwhile, the security of NCMs and NCMs-based systems
has garnered increasing attention. In particular, NCMs are often trained on
large-scale data from potentially untrustworthy sources, providing attackers
with the opportunity to manipulate them by inserting crafted samples into the
data. This type of attack is called a code poisoning attack (also known as a
backdoor attack). It allows attackers to implant backdoors in NCMs and thus
control model behavior, which poses a significant security threat. However,
there is still a lack of effective techniques for detecting various complex
code poisoning attacks.
  In this paper, we propose an innovative and lightweight technique for code
poisoning detection named KillBadCode. KillBadCode is designed based on our
insight that code poisoning disrupts the naturalness of code. Specifically,
KillBadCode first builds a code language model (CodeLM) on a lightweight
$n$-gram language model. Then, given poisoned data, KillBadCode utilizes CodeLM
to identify those tokens in (poisoned) code snippets that will make the code
snippets more natural after being deleted as trigger tokens. Considering that
the removal of some normal tokens in a single sample might also enhance code
naturalness, leading to a high false positive rate (FPR), we aggregate the
cumulative improvement of each token across all samples. Finally, KillBadCode
purifies the poisoned data by removing all poisoned samples containing the
identified trigger tokens. The experimental results on two code poisoning
attacks and four code intelligence tasks demonstrate that KillBadCode
significantly outperforms four baselines. More importantly, KillBadCode is very
efficient, with a minimum time consumption of only 5 minutes, and is 25 times
faster than the best baseline on average.


---

**[178. [2501.00055] LLM-Virus: Evolutionary Jailbreak Attack on Large Language Models](https://arxiv.org/pdf/2501.00055.pdf)** (Updated on 2025-01-03)

*Miao Yu, Junfeng Fang, Yingjie Zhou, Xing Fan, Kun Wang, Shirui Pan, Qingsong Wen*

  While safety-aligned large language models (LLMs) are increasingly used as
the cornerstone for powerful systems such as multi-agent frameworks to solve
complex real-world problems, they still suffer from potential adversarial
queries, such as jailbreak attacks, which attempt to induce harmful content.
Researching attack methods allows us to better understand the limitations of
LLM and make trade-offs between helpfulness and safety. However, existing
jailbreak attacks are primarily based on opaque optimization techniques (e.g.
token-level gradient descent) and heuristic search methods like LLM refinement,
which fall short in terms of transparency, transferability, and computational
cost. In light of these limitations, we draw inspiration from the evolution and
infection processes of biological viruses and propose LLM-Virus, a jailbreak
attack method based on evolutionary algorithm, termed evolutionary jailbreak.
LLM-Virus treats jailbreak attacks as both an evolutionary and transfer
learning problem, utilizing LLMs as heuristic evolutionary operators to ensure
high attack efficiency, transferability, and low time cost. Our experimental
results on multiple safety benchmarks show that LLM-Virus achieves competitive
or even superior performance compared to existing attack methods.


---

**[179. [2308.09440] Scope is all you need: Transforming LLMs for HPC Code](https://arxiv.org/pdf/2308.09440.pdf)** (Updated on 2023-10-02)

*Tal Kadosh, Niranjan Hasabnis, Vy A. Vo, Nadav Schneider, Neva Krien, Abdul Wasay, Nesreen Ahmed, Ted Willke, Guy Tamir, Yuval Pinter, Timothy Mattson, Gal Oren*

  With easier access to powerful compute resources, there is a growing trend in
the field of AI for software development to develop larger and larger language
models (LLMs) to address a variety of programming tasks. Even LLMs applied to
tasks from the high-performance computing (HPC) domain are huge in size (e.g.,
billions of parameters) and demand expensive compute resources for training. We
found this design choice confusing - why do we need large LLMs trained on
natural languages and programming languages unrelated to HPC for HPC-specific
tasks? In this line of work, we aim to question design choices made by existing
LLMs by developing smaller LLMs for specific domains - we call them
domain-specific LLMs. Specifically, we start off with HPC as a domain and
propose a novel tokenizer named Tokompiler, designed specifically for
preprocessing code in HPC and compilation-centric tasks. Tokompiler leverages
knowledge of language primitives to generate language-oriented tokens,
providing a context-aware understanding of code structure while avoiding human
semantics attributed to code structures completely. We applied Tokompiler to
pre-train two state-of-the-art models, SPT-Code and Polycoder, for a Fortran
code corpus mined from GitHub. We evaluate the performance of these models
against the conventional LLMs. Results demonstrate that Tokompiler
significantly enhances code completion accuracy and semantic understanding
compared to traditional tokenizers in normalized-perplexity tests, down to ~1
perplexity score. This research opens avenues for further advancements in
domain-specific LLMs, catering to the unique demands of HPC and compilation
tasks.


---

**[180. [2410.05331] Taylor Unswift: Secured Weight Release for Large Language Models via
  Taylor Expansion](https://arxiv.org/pdf/2410.05331.pdf)** (Updated on 2025-03-12)

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

**[181. [2311.16822] Large Language Models Suffer From Their Own Output: An Analysis of the
  Self-Consuming Training Loop](https://arxiv.org/pdf/2311.16822.pdf)** (Updated on 2024-06-18)

*Martin Briesch, Dominik Sobania, Franz Rothlauf*

  Large Language Models (LLM) are already widely used to generate content for a
variety of online platforms. As we are not able to safely distinguish
LLM-generated content from human-produced content, LLM-generated content is
used to train the next generation of LLMs, giving rise to a self-consuming
training loop. From the image generation domain we know that such a
self-consuming training loop reduces both quality and diversity of images
finally ending in a model collapse. However, it is unclear whether this
alarming effect can also be observed for LLMs. Therefore, we present the first
study investigating the self-consuming training loop for LLMs. Further, we
propose a novel method based on logic expressions that allows us to
unambiguously verify the correctness of LLM-generated content, which is
difficult for natural language text. We find that the self-consuming training
loop produces correct outputs, however, the output declines in its diversity
depending on the proportion of the used generated data. Fresh data can slow
down this decline, but not stop it. Given these concerning results, we
encourage researchers to study methods to negate this process.


---

**[182. [2410.15005] CAP: Data Contamination Detection via Consistency Amplification](https://arxiv.org/pdf/2410.15005.pdf)** (Updated on 2024-10-22)

*Yi Zhao, Jing Li, Linyi Yang*

  Large language models (LLMs) are widely used, but concerns about data
contamination challenge the reliability of LLM evaluations. Existing
contamination detection methods are often task-specific or require extra
prerequisites, limiting practicality. We propose a novel framework, Consistency
Amplification-based Data Contamination Detection (CAP), which introduces the
Performance Consistency Ratio (PCR) to measure dataset leakage by leveraging LM
consistency. To the best of our knowledge, this is the first method to
explicitly differentiate between fine-tuning and contamination, which is
crucial for detecting contamination in domain-specific models. Additionally,
CAP is applicable to various benchmarks and works for both white-box and
black-box models. We validate CAP's effectiveness through experiments on seven
LLMs and four domain-specific benchmarks. Our findings also show that composite
benchmarks from various dataset sources are particularly prone to unintentional
contamination. Codes will be publicly available soon.


---

**[183. [2311.05553] Removing RLHF Protections in GPT-4 via Fine-Tuning](https://arxiv.org/pdf/2311.05553.pdf)** (Updated on 2024-04-09)

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

**[184. [2402.16187] No Free Lunch in LLM Watermarking: Trade-offs in Watermarking Design
  Choices](https://arxiv.org/pdf/2402.16187.pdf)** (Updated on 2024-11-14)

*Qi Pang, Shengyuan Hu, Wenting Zheng, Virginia Smith*

  Advances in generative models have made it possible for AI-generated text,
code, and images to mirror human-generated content in many applications.
Watermarking, a technique that aims to embed information in the output of a
model to verify its source, is useful for mitigating the misuse of such
AI-generated content. However, we show that common design choices in LLM
watermarking schemes make the resulting systems surprisingly susceptible to
attack -- leading to fundamental trade-offs in robustness, utility, and
usability. To navigate these trade-offs, we rigorously study a set of simple
yet effective attacks on common watermarking systems, and propose guidelines
and defenses for LLM watermarking in practice.


---

**[185. [2405.07638] DoLLM: How Large Language Models Understanding Network Flow Data to
  Detect Carpet Bombing DDoS](https://arxiv.org/pdf/2405.07638.pdf)** (Updated on 2024-05-14)

*Qingyang Li, Yihang Zhang, Zhidong Jia, Yannan Hu, Lei Zhang, Jianrong Zhang, Yongming Xu, Yong Cui, Zongming Guo, Xinggong Zhang*

  It is an interesting question Can and How Large Language Models (LLMs)
understand non-language network data, and help us detect unknown malicious
flows. This paper takes Carpet Bombing as a case study and shows how to exploit
LLMs' powerful capability in the networking area. Carpet Bombing is a new DDoS
attack that has dramatically increased in recent years, significantly
threatening network infrastructures. It targets multiple victim IPs within
subnets, causing congestion on access links and disrupting network services for
a vast number of users. Characterized by low-rates, multi-vectors, these
attacks challenge traditional DDoS defenses. We propose DoLLM, a DDoS detection
model utilizes open-source LLMs as backbone. By reorganizing non-contextual
network flows into Flow-Sequences and projecting them into LLMs semantic space
as token embeddings, DoLLM leverages LLMs' contextual understanding to extract
flow representations in overall network context. The representations are used
to improve the DDoS detection performance. We evaluate DoLLM with public
datasets CIC-DDoS2019 and real NetFlow trace from Top-3 countrywide ISP. The
tests have proven that DoLLM possesses strong detection capabilities. Its F1
score increased by up to 33.3% in zero-shot scenarios and by at least 20.6% in
real ISP traces.


---

**[186. [2404.18239] SOUL: Unlocking the Power of Second-Order Optimization for LLM
  Unlearning](https://arxiv.org/pdf/2404.18239.pdf)** (Updated on 2024-06-26)

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

**[187. [2405.16133] Uncovering LLM-Generated Code: A Zero-Shot Synthetic Code Detector via
  Code Rewriting](https://arxiv.org/pdf/2405.16133.pdf)** (Updated on 2024-12-17)

*Tong Ye, Yangkai Du, Tengfei Ma, Lingfei Wu, Xuhong Zhang, Shouling Ji, Wenhai Wang*

  Large Language Models (LLMs) have demonstrated remarkable proficiency in
generating code. However, the misuse of LLM-generated (synthetic) code has
raised concerns in both educational and industrial contexts, underscoring the
urgent need for synthetic code detectors. Existing methods for detecting
synthetic content are primarily designed for general text and struggle with
code due to the unique grammatical structure of programming languages and the
presence of numerous ''low-entropy'' tokens. Building on this, our work
proposes a novel zero-shot synthetic code detector based on the similarity
between the original code and its LLM-rewritten variants. Our method is based
on the observation that differences between LLM-rewritten and original code
tend to be smaller when the original code is synthetic. We utilize
self-supervised contrastive learning to train a code similarity model and
evaluate our approach on two synthetic code detection benchmarks. Our results
demonstrate a significant improvement over existing SOTA synthetic content
detectors, with AUROC scores increasing by 20.5% on the APPS benchmark and
29.1% on the MBPP benchmark.


---

**[188. [2402.09299] Trained Without My Consent: Detecting Code Inclusion In Language Models
  Trained on Code](https://arxiv.org/pdf/2402.09299.pdf)** (Updated on 2024-11-01)

*Vahid Majdinasab, Amin Nikanjam, Foutse Khomh*

  Code auditing ensures that the developed code adheres to standards,
regulations, and copyright protection by verifying that it does not contain
code from protected sources. The recent advent of Large Language Models (LLMs)
as coding assistants in the software development process poses new challenges
for code auditing. The dataset for training these models is mainly collected
from publicly available sources. This raises the issue of intellectual property
infringement as developers' codes are already included in the dataset.
Therefore, auditing code developed using LLMs is challenging, as it is
difficult to reliably assert if an LLM used during development has been trained
on specific copyrighted codes, given that we do not have access to the training
datasets of these models. Given the non-disclosure of the training datasets,
traditional approaches such as code clone detection are insufficient for
asserting copyright infringement. To address this challenge, we propose a new
approach, TraWiC; a model-agnostic and interpretable method based on membership
inference for detecting code inclusion in an LLM's training dataset. We extract
syntactic and semantic identifiers unique to each program to train a classifier
for detecting code inclusion. In our experiments, we observe that TraWiC is
capable of detecting 83.87% of codes that were used to train an LLM. In
comparison, the prevalent clone detection tool NiCad is only capable of
detecting 47.64%. In addition to its remarkable performance, TraWiC has low
resource overhead in contrast to pair-wise clone detection that is conducted
during the auditing process of tools like CodeWhisperer reference tracker,
across thousands of code snippets.


---

**[189. [2310.09822] Turn Passive to Active: A Survey on Active Intellectual Property
  Protection of Deep Learning Models](https://arxiv.org/pdf/2310.09822.pdf)** (Updated on 2023-10-17)

*Mingfu Xue, Leo Yu Zhang, Yushu Zhang, Weiqiang Liu*

  The intellectual property protection of deep learning (DL) models has
attracted increasing serious concerns. Many works on intellectual property
protection for Deep Neural Networks (DNN) models have been proposed. The vast
majority of existing work uses DNN watermarking to verify the ownership of the
model after piracy occurs, which is referred to as passive verification. On the
contrary, we focus on a new type of intellectual property protection method
named active copyright protection, which refers to active authorization control
and user identity management of the DNN model. As of now, there is relatively
limited research in the field of active DNN copyright protection. In this
review, we attempt to clearly elaborate on the connotation, attributes, and
requirements of active DNN copyright protection, provide evaluation methods and
metrics for active copyright protection, review and analyze existing work on
active DL model intellectual property protection, discuss potential attacks
that active DL model copyright protection techniques may face, and provide
challenges and future directions for active DL model intellectual property
protection. This review is helpful to systematically introduce the new field of
active DNN copyright protection and provide reference and foundation for
subsequent work.


---

**[190. [2408.10718] CodeJudge-Eval: Can Large Language Models be Good Judges in Code
  Understanding?](https://arxiv.org/pdf/2408.10718.pdf)** (Updated on 2024-09-16)

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

**[191. [2410.14231] Unveiling Large Language Models Generated Texts: A Multi-Level
  Fine-Grained Detection Framework](https://arxiv.org/pdf/2410.14231.pdf)** (Updated on 2024-10-21)

*Zhen Tao, Zhiyu Li, Runyu Chen, Dinghao Xi, Wei Xu*

  Large language models (LLMs) have transformed human writing by enhancing
grammar correction, content expansion, and stylistic refinement. However, their
widespread use raises concerns about authorship, originality, and ethics, even
potentially threatening scholarly integrity. Existing detection methods, which
mainly rely on single-feature analysis and binary classification, often fail to
effectively identify LLM-generated text in academic contexts. To address these
challenges, we propose a novel Multi-level Fine-grained Detection (MFD)
framework that detects LLM-generated text by integrating low-level structural,
high-level semantic, and deep-level linguistic features, while conducting
sentence-level evaluations of lexicon, grammar, and syntax for comprehensive
analysis. To improve detection of subtle differences in LLM-generated text and
enhance robustness against paraphrasing, we apply two mainstream evasion
techniques to rewrite the text. These variations, along with original texts,
are used to train a text encoder via contrastive learning, extracting
high-level semantic features of sentence to boost detection generalization.
Furthermore, we leverage advanced LLM to analyze the entire text and extract
deep-level linguistic features, enhancing the model's ability to capture
complex patterns and nuances while effectively incorporating contextual
information. Extensive experiments on public datasets show that the MFD model
outperforms existing methods, achieving an MAE of 0.1346 and an accuracy of
88.56%. Our research provides institutions and publishers with an effective
mechanism to detect LLM-generated text, mitigating risks of compromised
authorship. Educators and editors can use the model's predictions to refine
verification and plagiarism prevention protocols, ensuring adherence to
standards.


---
