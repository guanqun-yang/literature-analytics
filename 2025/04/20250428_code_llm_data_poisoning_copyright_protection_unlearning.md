1. **[[2504.02883] SemEval-2025 Task 4: Unlearning sensitive content from Large Language
  Models](https://arxiv.org/pdf/2504.02883.pdf)**

Updated on 2025-04-07

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

2. **[[2404.18239] SOUL: Unlocking the Power of Second-Order Optimization for LLM
  Unlearning](https://arxiv.org/pdf/2404.18239.pdf)**

Updated on 2024-06-26

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

3. **[[2502.00406] ALU: Agentic LLM Unlearning](https://arxiv.org/pdf/2502.00406.pdf)**

Updated on 2025-02-04

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

4. **[[2504.13774] DP2Unlearning: An Efficient and Guaranteed Unlearning Framework for LLMs](https://arxiv.org/pdf/2504.13774.pdf)**

Updated on 2025-04-21

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

5. **[[2502.15097] LUME: LLM Unlearning with Multitask Evaluations](https://arxiv.org/pdf/2502.15097.pdf)**

Updated on 2025-02-28

*Anil Ramakrishna, Yixin Wan, Xiaomeng Jin, Kai-Wei Chang, Zhiqi Bu, Bhanukiran Vinzamuri, Volkan Cevher, Mingyi Hong, Rahul Gupta*


  Unlearning aims to remove copyrighted, sensitive, or private content from
large language models (LLMs) without a full retraining. In this work, we
develop a multi-task unlearning benchmark (LUME) which features three tasks:
(1) unlearn synthetically generated creative short novels, (2) unlearn
synthetic biographies with sensitive information, and (3) unlearn a collection
of public biographies. We further release two fine-tuned LLMs of 1B and 7B
parameter sizes as the target models. We conduct detailed evaluations of
several recently proposed unlearning algorithms and present results on
carefully crafted metrics to understand their behavior and limitations.



---

6. **[[2503.04693] UIPE: Enhancing LLM Unlearning by Removing Knowledge Related to
  Forgetting Targets](https://arxiv.org/pdf/2503.04693.pdf)**

Updated on 2025-03-07

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

7. **[[2404.11045] Offset Unlearning for Large Language Models](https://arxiv.org/pdf/2404.11045.pdf)**

Updated on 2024-04-18

*James Y. Huang, Wenxuan Zhou, Fei Wang, Fred Morstatter, Sheng Zhang, Hoifung Poon, Muhao Chen*


  Despite the strong capabilities of Large Language Models (LLMs) to acquire
knowledge from their training corpora, the memorization of sensitive
information in the corpora such as copyrighted, harmful, and private content
has led to ethical and legal concerns. In response to these challenges,
unlearning has emerged as a potential remedy for LLMs affected by problematic
training data. However, previous unlearning techniques are either not
applicable to black-box LLMs due to required access to model internal weights,
or violate data protection principles by retaining sensitive data for
inference-time correction. We propose $\delta$-unlearning, an offset unlearning
framework for black-box LLMs. Instead of tuning the black-box LLM itself,
$\delta$-unlearning learns the logit offset needed for unlearning by
contrasting the logits from a pair of smaller models. Experiments demonstrate
that $\delta$-unlearning can effectively unlearn target data while maintaining
similar or even stronger performance on general out-of-forget-scope tasks.
$\delta$-unlearning also effectively incorporates different unlearning
algorithms, making our approach a versatile solution to adapting various
existing unlearning algorithms to black-box LLMs.



---

8. **[[2412.00383] Unified Parameter-Efficient Unlearning for LLMs](https://arxiv.org/pdf/2412.00383.pdf)**

Updated on 2025-04-21

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

9. **[[2406.10952] Avoiding Copyright Infringement via Large Language Model Unlearning](https://arxiv.org/pdf/2406.10952.pdf)**

Updated on 2025-02-12

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

10. **[[2502.07218] LUNAR: LLM Unlearning via Neural Activation Redirection](https://arxiv.org/pdf/2502.07218.pdf)**

Updated on 2025-02-12

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

11. **[[2408.12416] Unlearning Trojans in Large Language Models: A Comparison Between
  Natural Language and Source Code](https://arxiv.org/pdf/2408.12416.pdf)**

Updated on 2024-08-23

*Mahdi Kazemi, Aftab Hussain, Md Rafiqul Islam Rabin, Mohammad Amin Alipour, Sen Lin*


  This work investigates the application of Machine Unlearning (MU) for
mitigating the impact of trojans embedded in conventional large language models
of natural language (Text-LLMs) and large language models of code (Code-LLMs)
We propose a novel unlearning approach, LYA, that leverages both gradient
ascent and elastic weight consolidation, a Fisher Information Matrix (FIM)
based regularization technique, to unlearn trojans from poisoned models. We
compare the effectiveness of LYA against conventional techniques like
fine-tuning, retraining, and vanilla gradient ascent. The subject models we
investigate are BERT and CodeBERT, for sentiment analysis and code defect
detection tasks, respectively. Our findings demonstrate that the combination of
gradient ascent and FIM-based regularization, as done in LYA, outperforms
existing methods in removing the trojan's influence from the poisoned model,
while preserving its original functionality. To the best of our knowledge, this
is the first work that compares and contrasts MU of trojans in LLMs, in the NL
and Coding domain.



---

12. **[[2502.17823] A General Framework to Enhance Fine-tuning-based LLM Unlearning](https://arxiv.org/pdf/2502.17823.pdf)**

Updated on 2025-03-25

*Jie Ren, Zhenwei Dai, Xianfeng Tang, Hui Liu, Jingying Zeng, Zhen Li, Rahul Goutam, Suhang Wang, Yue Xing, Qi He, Hui Liu*


  Unlearning has been proposed to remove copyrighted and privacy-sensitive data
from Large Language Models (LLMs). Existing approaches primarily rely on
fine-tuning-based methods, which can be categorized into gradient ascent-based
(GA-based) and suppression-based methods. However, they often degrade model
utility (the ability to respond to normal prompts). In this work, we aim to
develop a general framework that enhances the utility of fine-tuning-based
unlearning methods. To achieve this goal, we first investigate the common
property between GA-based and suppression-based methods. We unveil that
GA-based methods unlearn by distinguishing the target data (i.e., the data to
be removed) and suppressing related generations, which is essentially the same
strategy employed by suppression-based methods. Inspired by this finding, we
introduce Gated Representation UNlearning (GRUN) which has two components: a
soft gate function for distinguishing target data and a suppression module
using Representation Fine-tuning (ReFT) to adjust representations rather than
model parameters. Experiments show that GRUN significantly improves the
unlearning and utility. Meanwhile, it is general for fine-tuning-based methods,
efficient and promising for sequential unlearning.



---

13. **[[2402.08787] Rethinking Machine Unlearning for Large Language Models](https://arxiv.org/pdf/2402.08787.pdf)**

Updated on 2024-12-10

*Sijia Liu, Yuanshun Yao, Jinghan Jia, Stephen Casper, Nathalie Baracaldo, Peter Hase, Yuguang Yao, Chris Yuhao Liu, Xiaojun Xu, Hang Li, Kush R. Varshney, Mohit Bansal, Sanmi Koyejo, Yang Liu*


  We explore machine unlearning (MU) in the domain of large language models
(LLMs), referred to as LLM unlearning. This initiative aims to eliminate
undesirable data influence (e.g., sensitive or illegal information) and the
associated model capabilities, while maintaining the integrity of essential
knowledge generation and not affecting causally unrelated information. We
envision LLM unlearning becoming a pivotal element in the life-cycle management
of LLMs, potentially standing as an essential foundation for developing
generative AI that is not only safe, secure, and trustworthy, but also
resource-efficient without the need of full retraining. We navigate the
unlearning landscape in LLMs from conceptual formulation, methodologies,
metrics, and applications. In particular, we highlight the often-overlooked
aspects of existing LLM unlearning research, e.g., unlearning scope, data-model
interaction, and multifaceted efficacy assessment. We also draw connections
between LLM unlearning and related areas such as model editing, influence
functions, model explanation, adversarial training, and reinforcement learning.
Furthermore, we outline an effective assessment framework for LLM unlearning
and explore its applications in copyright and privacy safeguards and
sociotechnical harm reduction.



---

14. **[[2502.14425] A Survey on Data Contamination for Large Language Models](https://arxiv.org/pdf/2502.14425.pdf)**

Updated on 2025-02-21

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

15. **[[2307.10476] What can we learn from Data Leakage and Unlearning for Law?](https://arxiv.org/pdf/2307.10476.pdf)**

Updated on 2023-07-21

*Jaydeep Borkar*


  Large Language Models (LLMs) have a privacy concern because they memorize
training data (including personally identifiable information (PII) like emails
and phone numbers) and leak it during inference. A company can train an LLM on
its domain-customized data which can potentially also include their users' PII.
In order to comply with privacy laws such as the "right to be forgotten", the
data points of users that are most vulnerable to extraction could be deleted.
We find that once the most vulnerable points are deleted, a new set of points
become vulnerable to extraction. So far, little attention has been given to
understanding memorization for fine-tuned models. In this work, we also show
that not only do fine-tuned models leak their training data but they also leak
the pre-training data (and PII) memorized during the pre-training phase. The
property of new data points becoming vulnerable to extraction after unlearning
and leakage of pre-training data through fine-tuned models can pose significant
privacy and legal concerns for companies that use LLMs to offer services. We
hope this work will start an interdisciplinary discussion within AI and law
communities regarding the need for policies to tackle these issues.



---

16. **[[2412.20412] Multi-Objective Large Language Model Unlearning](https://arxiv.org/pdf/2412.20412.pdf)**

Updated on 2025-01-07

*Zibin Pan, Shuwen Zhang, Yuesheng Zheng, Chi Li, Yuheng Cheng, Junhua Zhao*


  Machine unlearning in the domain of large language models (LLMs) has
attracted great attention recently, which aims to effectively eliminate
undesirable behaviors from LLMs without full retraining from scratch. In this
paper, we explore the Gradient Ascent (GA) approach in LLM unlearning, which is
a proactive way to decrease the prediction probability of the model on the
target data in order to remove their influence. We analyze two challenges that
render the process impractical: gradient explosion and catastrophic forgetting.
To address these issues, we propose Multi-Objective Large Language Model
Unlearning (MOLLM) algorithm. We first formulate LLM unlearning as a
multi-objective optimization problem, in which the cross-entropy loss is
modified to the unlearning version to overcome the gradient explosion issue. A
common descent update direction is then calculated, which enables the model to
forget the target data while preserving the utility of the LLM. Our empirical
results verify that MoLLM outperforms the SOTA GA-based LLM unlearning methods
in terms of unlearning effect and model utility preservation. The source code
is available at https://github.com/zibinpan/MOLLM.



---

17. **[[2407.20999] MoFO: Momentum-Filtered Optimizer for Mitigating Forgetting in LLM
  Fine-Tuning](https://arxiv.org/pdf/2407.20999.pdf)**

Updated on 2025-04-21

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

18. **[[2311.11123] (Why) Is My Prompt Getting Worse? Rethinking Regression Testing for
  Evolving LLM APIs](https://arxiv.org/pdf/2311.11123.pdf)**

Updated on 2024-02-08

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

19. **[[2407.16951] Towards Transfer Unlearning: Empirical Evidence of Cross-Domain Bias
  Mitigation](https://arxiv.org/pdf/2407.16951.pdf)**

Updated on 2024-07-25

*Huimin Lu, Masaru Isonuma, Junichiro Mori, Ichiro Sakata*


  Large language models (LLMs) often inherit biases from vast amounts of
training corpora. Traditional debiasing methods, while effective to some
extent, do not completely eliminate memorized biases and toxicity in LLMs. In
this paper, we study an unlearning-based approach to debiasing in LLMs by
performing gradient ascent on hate speech against minority groups, i.e.,
minimizing the likelihood of biased or toxic content. Specifically, we propose
a mask language modeling unlearning technique, which unlearns the harmful part
of the text. This method enables LLMs to selectively forget and disassociate
from biased and harmful content. Experimental results demonstrate the
effectiveness of our approach in diminishing bias while maintaining the
language modeling abilities. Surprisingly, the results also unveil an
unexpected potential for cross-domain transfer unlearning: debiasing in one
bias form (e.g. gender) may contribute to mitigating others (e.g. race and
religion).



---

20. **[[2411.12103] Does Unlearning Truly Unlearn? A Black Box Evaluation of LLM Unlearning
  Methods](https://arxiv.org/pdf/2411.12103.pdf)**

Updated on 2025-02-25

*Jai Doshi, Asa Cooper Stickland*


  Large language model unlearning aims to remove harmful information that LLMs
have learnt to prevent their use for malicious purposes. LLMU and RMU have been
proposed as two methods for LLM unlearning, achieving impressive results on
unlearning benchmarks. We study in detail the impact of unlearning on LLM
performance metrics using the WMDP dataset as well as a new biology dataset we
create. We show that unlearning has a notable impact on general model
capabilities, with the performance degradation being more significant in
general for LLMU. We further test the robustness of the two methods and find
that doing 5-shot prompting or rephrasing the question in simple ways can lead
to an over ten-fold increase in accuracy on unlearning benchmarks. Finally, we
show that training on unrelated data can almost completely recover
pre-unlearning performance, demonstrating that these methods fail at truly
unlearning. Our methodology serves as an evaluation framework for LLM
unlearning methods. The code is available at:
https://github.com/JaiDoshi/Knowledge-Erasure.



---

21. **[[2411.18948] RevPRAG: Revealing Poisoning Attacks in Retrieval-Augmented Generation
  through LLM Activation Analysis](https://arxiv.org/pdf/2411.18948.pdf)**

Updated on 2025-02-20

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

22. **[[2503.07697] PoisonedParrot: Subtle Data Poisoning Attacks to Elicit
  Copyright-Infringing Content from Large Language Models](https://arxiv.org/pdf/2503.07697.pdf)**

Updated on 2025-03-12

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

23. **[[2404.17196] Human-Imperceptible Retrieval Poisoning Attacks in LLM-Powered
  Applications](https://arxiv.org/pdf/2404.17196.pdf)**

Updated on 2024-04-29

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

24. **[[2402.10058] Towards Safer Large Language Models through Machine Unlearning](https://arxiv.org/pdf/2402.10058.pdf)**

Updated on 2024-06-06

*Zheyuan Liu, Guangyao Dou, Zhaoxuan Tan, Yijun Tian, Meng Jiang*


  The rapid advancement of Large Language Models (LLMs) has demonstrated their
vast potential across various domains, attributed to their extensive
pretraining knowledge and exceptional generalizability. However, LLMs often
encounter challenges in generating harmful content when faced with problematic
prompts. To address this problem, existing work attempted to implement a
gradient ascent based approach to prevent LLMs from producing harmful output.
While these methods can be effective, they frequently impact the model utility
in responding to normal prompts. To address this gap, we introduce Selective
Knowledge negation Unlearning (SKU), a novel unlearning framework for LLMs,
designed to eliminate harmful knowledge while preserving utility on normal
prompts. Specifically, SKU is consisted of two stages: harmful knowledge
acquisition stage and knowledge negation stage. The first stage aims to
identify and acquire harmful knowledge within the model, whereas the second is
dedicated to remove this knowledge. SKU selectively isolates and removes
harmful knowledge in model parameters, ensuring the model's performance remains
robust on normal prompts. Our experiments conducted across various LLM
architectures demonstrate that SKU identifies a good balance point between
removing harmful information and preserving utility.



---

25. **[[2406.10890] RWKU: Benchmarking Real-World Knowledge Unlearning for Large Language
  Models](https://arxiv.org/pdf/2406.10890.pdf)**

Updated on 2024-06-18

*Zhuoran Jin, Pengfei Cao, Chenhao Wang, Zhitao He, Hongbang Yuan, Jiachun Li, Yubo Chen, Kang Liu, Jun Zhao*


  Large language models (LLMs) inevitably memorize sensitive, copyrighted, and
harmful knowledge from the training corpus; therefore, it is crucial to erase
this knowledge from the models. Machine unlearning is a promising solution for
efficiently removing specific knowledge by post hoc modifying models. In this
paper, we propose a Real-World Knowledge Unlearning benchmark (RWKU) for LLM
unlearning. RWKU is designed based on the following three key factors: (1) For
the task setting, we consider a more practical and challenging unlearning
setting, where neither the forget corpus nor the retain corpus is accessible.
(2) For the knowledge source, we choose 200 real-world famous people as the
unlearning targets and show that such popular knowledge is widely present in
various LLMs. (3) For the evaluation framework, we design the forget set and
the retain set to evaluate the model's capabilities across various real-world
applications. Regarding the forget set, we provide four four membership
inference attack (MIA) methods and nine kinds of adversarial attack probes to
rigorously test unlearning efficacy. Regarding the retain set, we assess
locality and utility in terms of neighbor perturbation, general ability,
reasoning ability, truthfulness, factuality, and fluency. We conduct extensive
experiments across two unlearning scenarios, two models and six baseline
methods and obtain some meaningful findings. We release our benchmark and code
publicly at http://rwku-bench.github.io for future work.



---

26. **[[2503.09117] GRU: Mitigating the Trade-off between Unlearning and Retention for Large
  Language Models](https://arxiv.org/pdf/2503.09117.pdf)**

Updated on 2025-03-13

*Yue Wang, Qizhou Wang, Feng Liu, Wei Huang, Yali Du, Xiaojiang Du, Bo Han*


  Large language model (LLM) unlearning has demonstrated its essential role in
removing privacy and copyright-related responses, crucial for their legal and
safe applications. However, the pursuit of complete unlearning often comes with
substantial costs due to its compromises in their general functionality,
leading to a notorious trade-off between unlearning and retention. In examining
the update process for unlearning dynamically, we find gradients hold essential
information for revealing this trade-off. In particular, we look at the varying
relationship between retention performance and directional disparities between
gradients during unlearning. It motivates the sculpting of an update mechanism
derived from gradients from two sources, i.e., harmful for retention and useful
for unlearning. Accordingly, we propose Gradient Rectified Unlearning (GRU), an
enhanced unlearning framework controlling the updating gradients in a
geometry-focused and optimization-driven manner such that their side impacts on
other, unrelated responses can be minimized. Specifically, GRU derives a
closed-form solution to project the unlearning gradient onto the orthogonal
space of that gradient harmful for retention, ensuring minimal deviation from
its original direction under the condition that overall performance is
retained. Comprehensive experiments are conducted to demonstrate that GRU, as a
general framework, is straightforward to implement and efficiently enhances a
range of baseline methods through its adaptable and compatible characteristics.
Additionally, experimental results show its broad effectiveness across a
diverse set of benchmarks for LLM unlearning.



---

27. **[[2406.16201] Blind Baselines Beat Membership Inference Attacks for Foundation Models](https://arxiv.org/pdf/2406.16201.pdf)**

Updated on 2025-04-01

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

28. **[[2311.10733] Proceedings of the 3rd International Workshop on Mining and Learning in
  the Legal Domain (MLLD-23)](https://arxiv.org/pdf/2311.10733.pdf)**

Updated on 2023-11-21

*Masoud Makrehchi, Dell Zhang, Alina Petrova, John Armour*


  This is the Proceedings of the 3rd International Workshop on Mining and
Learning in the Legal Domain (MLLD-23) which took place in conjunction with the
32nd ACM International Conference on Information and Knowledge Management
(CIKM-2023) at the University of Birmingham, Birmingham, UK on Sunday 22nd
October 2023.



---

29. **[[2504.12681] GRAIL: Gradient-Based Adaptive Unlearning for Privacy and Copyright in
  LLMs](https://arxiv.org/pdf/2504.12681.pdf)**

Updated on 2025-04-18

*Kun-Woo Kim, Ji-Hoon Park, Ju-Min Han, Seong-Whan Lee*


  Large Language Models (LLMs) trained on extensive datasets often learn
sensitive information, which raises significant social and legal concerns under
principles such as the "Right to be forgotten." Retraining entire models from
scratch to remove undesired information is both costly and impractical.
Furthermore, existing single-domain unlearning methods fail to address
multi-domain scenarios, where knowledge is interwoven across domains such as
privacy and copyright, creating overlapping representations that lead to
excessive knowledge removal or degraded performance. To tackle these issues, we
propose GRAIL (GRadient-based AdaptIve unLearning), a novel multi-domain
unlearning framework. GRAIL leverages gradient information from multiple
domains to precisely distinguish the unlearning scope from the retention scope,
and applies an adaptive parameter-wise localization strategy to selectively
remove targeted knowledge while preserving critical parameters for each domain.
Experimental results on unlearning benchmarks show that GRAIL achieves
unlearning success on par with the existing approaches, while also
demonstrating up to 17% stronger knowledge retention success compared to the
previous state-of-art method. Our findings establish a new paradigm for
effectively managing and regulating sensitive information in large-scale
pre-trained language models.



---

30. **[[2308.12247] How to Protect Copyright Data in Optimization of Large Language Models?](https://arxiv.org/pdf/2308.12247.pdf)**

Updated on 2023-08-24

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

31. **[[2409.13474] Alternate Preference Optimization for Unlearning Factual Knowledge in
  Large Language Models](https://arxiv.org/pdf/2409.13474.pdf)**

Updated on 2025-01-23

*Anmol Mekala, Vineeth Dorna, Shreya Dubey, Abhishek Lalwani, David Koleczek, Mukund Rungta, Sadid Hasan, Elita Lobo*


  Machine unlearning aims to efficiently eliminate the influence of specific
training data, known as the forget set, from the model. However, existing
unlearning methods for Large Language Models (LLMs) face a critical challenge:
they rely solely on negative feedback to suppress responses related to the
forget set, which often results in nonsensical or inconsistent outputs,
diminishing model utility and posing potential privacy risks. To address this
limitation, we propose a novel approach called Alternate Preference
Optimization (AltPO), which combines negative feedback with in-domain positive
feedback on the forget set. Additionally, we introduce new evaluation metrics
to assess the quality of responses related to the forget set. Extensive
experiments show that our approach not only enables effective unlearning but
also avoids undesirable model behaviors while maintaining overall model
performance. Our implementation can be found at
https://github.com/molereddy/Alternate-Preference-Optimization.



---

32. **[[2412.04947] C$^2$LEVA: Toward Comprehensive and Contamination-Free Language Model
  Evaluation](https://arxiv.org/pdf/2412.04947.pdf)**

Updated on 2024-12-17

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

33. **[[2402.14845] Purifying Large Language Models by Ensembling a Small Language Model](https://arxiv.org/pdf/2402.14845.pdf)**

Updated on 2024-02-26

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

34. **[[2504.05058] Not All Data Are Unlearned Equally](https://arxiv.org/pdf/2504.05058.pdf)**

Updated on 2025-04-14

*Aravind Krishnan, Siva Reddy, Marius Mosbach*


  Machine unlearning is concerned with the task of removing knowledge learned
from particular data points from a trained model. In the context of large
language models (LLMs), unlearning has recently received increased attention,
particularly for removing knowledge about named entities from models for
privacy purposes. While various approaches have been proposed to address the
unlearning problem, most existing approaches treat all data points to be
unlearned equally, i.e., unlearning that Montreal is a city in Canada is
treated exactly the same as unlearning the phone number of the first author of
this paper. In this work, we show that this all data is equal assumption does
not hold for LLM unlearning. We study how the success of unlearning depends on
the frequency of the knowledge we want to unlearn in the pre-training data of a
model and find that frequency strongly affects unlearning, i.e., more frequent
knowledge is harder to unlearn. Additionally, we uncover a misalignment between
probability and generation-based evaluations of unlearning and show that this
problem worsens as models become larger. Overall, our experiments highlight the
need for better evaluation practices and novel methods for LLM unlearning that
take the training data of models into account.



---

35. **[[2406.04244] Benchmark Data Contamination of Large Language Models: A Survey](https://arxiv.org/pdf/2406.04244.pdf)**

Updated on 2024-06-07

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

36. **[[2406.06443] LLM Dataset Inference: Did you train on my dataset?](https://arxiv.org/pdf/2406.06443.pdf)**

Updated on 2024-06-11

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

37. **[[2502.13141] UniGuardian: A Unified Defense for Detecting Prompt Injection, Backdoor
  Attacks and Adversarial Attacks in Large Language Models](https://arxiv.org/pdf/2502.13141.pdf)**

Updated on 2025-02-19

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

38. **[[2403.18920] CPR: Retrieval Augmented Generation for Copyright Protection](https://arxiv.org/pdf/2403.18920.pdf)**

Updated on 2024-03-29

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

39. **[[2410.17050] UnStar: Unlearning with Self-Taught Anti-Sample Reasoning for LLMs](https://arxiv.org/pdf/2410.17050.pdf)**

Updated on 2024-10-23

*Yash Sinha, Murari Mandal, Mohan Kankanhalli*


  The key components of machine learning are data samples for training, model
for learning patterns, and loss function for optimizing accuracy. Analogously,
unlearning can potentially be achieved through anti-data samples (or
anti-samples), unlearning method, and reversed loss function. While prior
research has explored unlearning methods and reversed loss functions, the
potential of anti-samples remains largely untapped. In this paper, we introduce
UnSTAR: Unlearning with Self-Taught Anti-Sample Reasoning for large language
models (LLMs). Our contributions are threefold; first, we propose a novel
concept of anti-sample-induced unlearning; second, we generate anti-samples by
leveraging misleading rationales, which help reverse learned associations and
accelerate the unlearning process; and third, we enable fine-grained targeted
unlearning, allowing for the selective removal of specific associations without
impacting related knowledge - something not achievable by previous works.
Results demonstrate that anti-samples offer an efficient, targeted unlearning
strategy for LLMs, opening new avenues for privacy-preserving machine learning
and model modification.



---

40. **[[2403.15779] The Frontier of Data Erasure: Machine Unlearning for Large Language
  Models](https://arxiv.org/pdf/2403.15779.pdf)**

Updated on 2024-03-26

*Youyang Qu, Ming Ding, Nan Sun, Kanchana Thilakarathna, Tianqing Zhu, Dusit Niyato*


  Large Language Models (LLMs) are foundational to AI advancements,
facilitating applications like predictive text generation. Nonetheless, they
pose risks by potentially memorizing and disseminating sensitive, biased, or
copyrighted information from their vast datasets. Machine unlearning emerges as
a cutting-edge solution to mitigate these concerns, offering techniques for
LLMs to selectively discard certain data. This paper reviews the latest in
machine unlearning for LLMs, introducing methods for the targeted forgetting of
information to address privacy, ethical, and legal challenges without
necessitating full model retraining. It divides existing research into
unlearning from unstructured/textual data and structured/classification data,
showcasing the effectiveness of these approaches in removing specific data
while maintaining model efficacy. Highlighting the practicality of machine
unlearning, this analysis also points out the hurdles in preserving model
integrity, avoiding excessive or insufficient data removal, and ensuring
consistent outputs, underlining the role of machine unlearning in advancing
responsible, ethical AI.



---

41. **[[2408.06223] On Effects of Steering Latent Representation for Large Language Model
  Unlearning](https://arxiv.org/pdf/2408.06223.pdf)**

Updated on 2025-02-07

*Dang Huu-Tien, Trung-Tin Pham, Hoang Thanh-Tung, Naoya Inoue*


  Representation Misdirection for Unlearning (RMU), which steers model
representation in the intermediate layer to a target random representation, is
an effective method for large language model (LLM) unlearning. Despite its high
performance, the underlying cause and explanation remain underexplored. In this
paper, we theoretically demonstrate that steering forget representations in the
intermediate layer reduces token confidence, causing LLMs to generate wrong or
nonsense responses. We investigate how the coefficient influences the alignment
of forget-sample representations with the random direction and hint at the
optimal coefficient values for effective unlearning across different network
layers. We show that RMU unlearned models are robust against adversarial
jailbreak attacks. Furthermore, our empirical analysis shows that RMU is less
effective when applied to the middle and later layers in LLMs. To resolve this
drawback, we propose Adaptive RMU--a simple yet effective alternative method
that makes unlearning effective with most layers. Extensive experiments
demonstrate that Adaptive RMU significantly improves the unlearning performance
compared to prior art while incurring no additional computational cost.



---

42. **[[2502.19301] Rethinking LLM Unlearning Objectives: A Gradient Perspective and Go
  Beyond](https://arxiv.org/pdf/2502.19301.pdf)**

Updated on 2025-02-27

*Qizhou Wang, Jin Peng Zhou, Zhanke Zhou, Saebyeol Shin, Bo Han, Kilian Q. Weinberger*


  Large language models (LLMs) should undergo rigorous audits to identify
potential risks, such as copyright and privacy infringements. Once these risks
emerge, timely updates are crucial to remove undesirable responses, ensuring
legal and safe model usage. It has spurred recent research into LLM unlearning,
focusing on erasing targeted undesirable knowledge without compromising the
integrity of other, non-targeted responses. Existing studies have introduced
various unlearning objectives to pursue LLM unlearning without necessitating
complete retraining. However, each of these objectives has unique properties,
and no unified framework is currently available to comprehend them thoroughly.
To fill the gap, we propose a toolkit of the gradient effect (G-effect),
quantifying the impacts of unlearning objectives on model performance from a
gradient perspective. A notable advantage is its broad ability to detail the
unlearning impacts from various aspects across instances, updating steps, and
LLM layers. Accordingly, the G-effect offers new insights into identifying
drawbacks of existing unlearning objectives, further motivating us to explore a
series of new solutions for their mitigation and improvements. Finally, we
outline promising directions that merit further studies, aiming at contributing
to the community to advance this important field.



---

43. **[[2410.08109] A Closer Look at Machine Unlearning for Large Language Models](https://arxiv.org/pdf/2410.08109.pdf)**

Updated on 2025-03-04

*Xiaojian Yuan, Tianyu Pang, Chao Du, Kejiang Chen, Weiming Zhang, Min Lin*


  Large language models (LLMs) may memorize sensitive or copyrighted content,
raising privacy and legal concerns. Due to the high cost of retraining from
scratch, researchers attempt to employ machine unlearning to remove specific
content from LLMs while preserving the overall performance. In this paper, we
discuss several issues in machine unlearning for LLMs and provide our insights
on possible approaches. To address the issue of inadequate evaluation of model
outputs after unlearning, we introduce three additional metrics to evaluate
token diversity, sentence semantics, and factual correctness. We then
categorize unlearning methods into untargeted and targeted, and discuss their
issues respectively. Specifically, the behavior that untargeted unlearning
attempts to approximate is unpredictable and may involve hallucinations, and
existing regularization is insufficient for targeted unlearning. To alleviate
these issues, we propose using the objective of maximizing entropy (ME) for
untargeted unlearning and incorporate answer preservation (AP) loss as
regularization for targeted unlearning. Experimental results across three
scenarios, i.e., fictitious unlearning, continual unlearning, and real-world
unlearning, demonstrate the effectiveness of our approaches. The code is
available at https://github.com/sail-sg/closer-look-LLM-unlearning.



---

44. **[[2310.10049] FATE-LLM: A Industrial Grade Federated Learning Framework for Large
  Language Models](https://arxiv.org/pdf/2310.10049.pdf)**

Updated on 2023-10-17

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

45. **[[2405.01560] Copyright related risks in the creation and use of ML/AI systems](https://arxiv.org/pdf/2405.01560.pdf)**

Updated on 2024-05-06

*Daniel M. German*


  This paper summarizes the current copyright related risks that Machine
Learning (ML) and Artificial Intelligence (AI) systems (including Large
Language Models --LLMs) incur. These risks affect different stakeholders:
owners of the copyright of the training data, the users of ML/AI systems, the
creators of trained models, and the operators of AI systems. This paper also
provides an overview of ongoing legal cases in the United States related to
these risks.



---

46. **[[2503.18674] Human Motion Unlearning](https://arxiv.org/pdf/2503.18674.pdf)**

Updated on 2025-03-25

*Edoardo De Matteis, Matteo Migliarini, Alessio Sampieri, Indro Spinelli, Fabio Galasso*


  We introduce the task of human motion unlearning to prevent the synthesis of
toxic animations while preserving the general text-to-motion generative
performance. Unlearning toxic motions is challenging as those can be generated
from explicit text prompts and from implicit toxic combinations of safe motions
(e.g., ``kicking" is ``loading and swinging a leg"). We propose the first
motion unlearning benchmark by filtering toxic motions from the large and
recent text-to-motion datasets of HumanML3D and Motion-X. We propose baselines,
by adapting state-of-the-art image unlearning techniques to process
spatio-temporal signals. Finally, we propose a novel motion unlearning model
based on Latent Code Replacement, which we dub LCR. LCR is training-free and
suitable to the discrete latent spaces of state-of-the-art text-to-motion
diffusion models. LCR is simple and consistently outperforms baselines
qualitatively and quantitatively. Project page:
\href{https://www.pinlab.org/hmu}{https://www.pinlab.org/hmu}.



---

47. **[[2408.06621] Towards Robust and Parameter-Efficient Knowledge Unlearning for LLMs](https://arxiv.org/pdf/2408.06621.pdf)**

Updated on 2025-04-02

*Sungmin Cha, Sungjun Cho, Dasol Hwang, Moontae Lee*


  Large Language Models (LLMs) have demonstrated strong reasoning and
memorization capabilities via pretraining on massive textual corpora. However,
this poses risk of privacy and copyright violations, highlighting the need for
efficient machine unlearning methods that remove sensitive data without
retraining from scratch. While Gradient Ascent (GA) is commonly used to unlearn
by reducing the likelihood of generating unwanted content, it leads to unstable
optimization and catastrophic forgetting of retrained knowledge. We find that
combining GA with low-rank adaptation results in poor trade-offs between
computational cost and generative performance. To address these challenges, we
propose Low-rank Knowledge Unlearning (LoKU), a novel framework that enables
robust and efficient unlearning for LLMs. First, we introduce Inverted Hinge
Loss, which suppresses unwanted tokens while maintaining fluency by boosting
the probability of the next most likely token. Second, we develop a
data-adaptive initialization for LoRA adapters via low-rank approximation
weighted with relative Fisher information, thereby focusing updates on
parameters critical for removing targeted knowledge. Experiments on the
Training Data Extraction Challenge dataset using GPT-Neo models as well as on
the TOFU benchmark with Phi-1.5B and Llama2-7B models demonstrate that our
approach effectively removes sensitive information while maintaining reasoning
and generative capabilities with minimal impact. Our implementation can be
found in https://github.com/csm9493/efficient-llm-unlearning.



---

48. **[[2410.10866] CodeUnlearn: Amortized Zero-Shot Machine Unlearning in Language Models
  Using Discrete Concept](https://arxiv.org/pdf/2410.10866.pdf)**

Updated on 2024-10-16

*YuXuan Wu, Bonaventure F. P. Dossou, Dianbo Liu*


  Large Language Models (LLMs) offer extensive knowledge across various
domains, but they may inadvertently memorize sensitive, unauthorized, or
malicious data, such as personal information in the medical and financial
sectors. Machine unlearning methods aim to remove specific information from
models after training to address this. However, current approaches require
additional model training or struggle to effectively erase particular data
points and their associated context due to LLMs' complex, dense, and continuous
nature. In this study, we propose a novel amortized unlearning approach using
codebook features and Sparse Autoencoders (SAEs). By leveraging a bottleneck to
decompose the activation space and regulate information flow, our method
efficiently unlearns targeted information while preserving the model's
performance on unrelated data. To the best of our knowledge, this is the first
work that successfully enables unlearning specific topics with contextual
relevance in an LLM, marking a significant step towards real-world applications
of machine unlearning.



---

49. **[[2502.01534] Preference Leakage: A Contamination Problem in LLM-as-a-judge](https://arxiv.org/pdf/2502.01534.pdf)**

Updated on 2025-02-04

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

50. **[[2405.15152] Machine Unlearning in Large Language Models](https://arxiv.org/pdf/2405.15152.pdf)**

Updated on 2024-05-27

*Saaketh Koundinya Gundavarapu, Shreya Agarwal, Arushi Arora, Chandana Thimmalapura Jagadeeshaiah*


  Machine unlearning, a novel area within artificial intelligence, focuses on
addressing the challenge of selectively forgetting or reducing undesirable
knowledge or behaviors in machine learning models, particularly in the context
of large language models (LLMs). This paper introduces a methodology to align
LLMs, such as Open Pre-trained Transformer Language Models, with ethical,
privacy, and safety standards by leveraging the gradient ascent algorithm for
knowledge unlearning. Our approach aims to selectively erase or modify learned
information in LLMs, targeting harmful responses and copyrighted content. This
paper presents a dual-pronged approach to enhance the ethical and safe behavior
of large language models (LLMs) by addressing the issues of harmful responses
and copyrighted content. To mitigate harmful responses, we applied gradient
ascent on the PKU dataset, achieving a 75\% reduction in harmful responses for
Open Pre-trained Transformer Language Models (OPT1.3b and OPT2.7b)
\citet{zhang2022opt} while retaining previous knowledge using the TruthfulQA
dataset \citet{DBLP:journals/corr/abs-2109-07958}. For handling copyrighted
content, we constructed a custom dataset based on the Lord of the Rings corpus
and aligned LLMs (OPT1.3b and OPT2.7b) \citet{zhang2022opt} through LoRA:
Low-Rank Adaptation of Large Language Models
\citet{DBLP:journals/corr/abs-2106-09685} finetuning. Subsequently, we employed
gradient ascent to unlearn the Lord of the Rings content, resulting in a
remarkable reduction in the presence of copyrighted material. To maintain a
diverse knowledge base, we utilized the Book Corpus dataset. Additionally, we
propose a new evaluation technique for assessing the effectiveness of harmful
unlearning.



---

51. **[[2502.13996] Beyond Single-Value Metrics: Evaluating and Enhancing LLM Unlearning
  with Cognitive Diagnosis](https://arxiv.org/pdf/2502.13996.pdf)**

Updated on 2025-02-21

*Yicheng Lang, Kehan Guo, Yue Huang, Yujun Zhou, Haomin Zhuang, Tianyu Yang, Yao Su, Xiangliang Zhang*


  Due to the widespread use of LLMs and the rising critical ethical and safety
concerns, LLM unlearning methods have been developed to remove harmful
knowledge and undesirable capabilities. In this context, evaluations are mostly
based on single-value metrics such as QA accuracy. However, these metrics often
fail to capture the nuanced retention of harmful knowledge components, making
it difficult to assess the true effectiveness of unlearning. To address this
issue, we propose UNCD (UNlearning evaluation via Cognitive Diagnosis), a novel
framework that leverages Cognitive Diagnosis Modeling for fine-grained
evaluation of LLM unlearning. Our dedicated benchmark, UNCD-Cyber, provides a
detailed assessment of the removal of dangerous capabilities. Moreover, we
introduce UNCD-Agent, which refines unlearning by diagnosing knowledge remnants
and generating targeted unlearning data. Extensive experiments across eight
unlearning methods and two base models demonstrate that UNCD not only enhances
evaluation but also effectively facilitates the removal of harmful LLM
abilities.



---

52. **[[2410.11143] LLM Unlearning via Loss Adjustment with Only Forget Data](https://arxiv.org/pdf/2410.11143.pdf)**

Updated on 2024-10-16

*Yaxuan Wang, Jiaheng Wei, Chris Yuhao Liu, Jinlong Pang, Quan Liu, Ankit Parag Shah, Yujia Bao, Yang Liu, Wei Wei*


  Unlearning in Large Language Models (LLMs) is essential for ensuring ethical
and responsible AI use, especially in addressing privacy leak, bias, safety,
and evolving regulations. Existing approaches to LLM unlearning often rely on
retain data or a reference LLM, yet they struggle to adequately balance
unlearning performance with overall model utility. This challenge arises
because leveraging explicit retain data or implicit knowledge of retain data
from a reference LLM to fine-tune the model tends to blur the boundaries
between the forgotten and retain data, as different queries often elicit
similar responses. In this work, we propose eliminating the need to retain data
or the reference LLM for response calibration in LLM unlearning. Recognizing
that directly applying gradient ascent on the forget data often leads to
optimization instability and poor performance, our method guides the LLM on
what not to respond to, and importantly, how to respond, based on the forget
data. Hence, we introduce Forget data only Loss AjustmenT (FLAT), a "flat" loss
adjustment approach which addresses these issues by maximizing f-divergence
between the available template answer and the forget answer only w.r.t. the
forget data. The variational form of the defined f-divergence theoretically
provides a way of loss adjustment by assigning different importance weights for
the learning w.r.t. template responses and the forgetting of responses subject
to unlearning. Empirical results demonstrate that our approach not only
achieves superior unlearning performance compared to existing methods but also
minimizes the impact on the model's retained capabilities, ensuring high
utility across diverse tasks, including copyrighted content unlearning on Harry
Potter dataset and MUSE Benchmark, and entity unlearning on the TOFU dataset.



---

53. **[[2406.13990] Inference-Time Decontamination: Reusing Leaked Benchmarks for Large
  Language Model Evaluation](https://arxiv.org/pdf/2406.13990.pdf)**

Updated on 2024-06-25

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

54. **[[2406.07933] Large Language Model Unlearning via Embedding-Corrupted Prompts](https://arxiv.org/pdf/2406.07933.pdf)**

Updated on 2024-11-01

*Chris Yuhao Liu, Yaxuan Wang, Jeffrey Flanigan, Yang Liu*


  Large language models (LLMs) have advanced to encompass extensive knowledge
across diverse domains. Yet controlling what a large language model should not
know is important for ensuring alignment and thus safe use. However, accurately
and efficiently unlearning knowledge from an LLM remains challenging due to the
potential collateral damage caused by the fuzzy boundary between retention and
forgetting, and the large computational requirements for optimization across
state-of-the-art models with hundreds of billions of parameters. In this work,
we present \textbf{Embedding-COrrupted (ECO) Prompts}, a lightweight unlearning
framework for large language models to address both the challenges of knowledge
entanglement and unlearning efficiency. Instead of relying on the LLM itself to
unlearn, we enforce an unlearned state during inference by employing a prompt
classifier to identify and safeguard prompts to forget. We learn corruptions
added to prompt embeddings via zeroth order optimization toward the unlearning
objective offline and corrupt prompts flagged by the classifier during
inference. We find that these embedding-corrupted prompts not only lead to
desirable outputs that satisfy the unlearning objective but also closely
approximate the output from a model that has never been trained on the data
intended for forgetting. Through extensive experiments on unlearning, we
demonstrate the superiority of our method in achieving promising unlearning at
\textit{nearly zero side effects} in general domains and domains closely
related to the unlearned ones. Additionally, we highlight the scalability of
our method to 100 LLMs, ranging from 0.5B to 236B parameters, incurring no
additional cost as the number of parameters increases. We have made our code
publicly available at \url{https://github.com/chrisliu298/llm-unlearn-eco}.



---

55. **[[2502.01083] Tool Unlearning for Tool-Augmented LLMs](https://arxiv.org/pdf/2502.01083.pdf)**

Updated on 2025-02-04

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

56. **[[2303.09384] LLMSecEval: A Dataset of Natural Language Prompts for Security
  Evaluations](https://arxiv.org/pdf/2303.09384.pdf)**

Updated on 2023-03-17

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

57. **[[2406.11780] Split, Unlearn, Merge: Leveraging Data Attributes for More Effective
  Unlearning in LLMs](https://arxiv.org/pdf/2406.11780.pdf)**

Updated on 2024-06-18

*Swanand Ravindra Kadhe, Farhan Ahmed, Dennis Wei, Nathalie Baracaldo, Inkit Padhi*


  Large language models (LLMs) have shown to pose social and ethical risks such
as generating toxic language or facilitating malicious use of hazardous
knowledge. Machine unlearning is a promising approach to improve LLM safety by
directly removing harmful behaviors and knowledge. In this paper, we propose
"SPlit, UNlearn, MerGE" (SPUNGE), a framework that can be used with any
unlearning method to amplify its effectiveness. SPUNGE leverages data
attributes during unlearning by splitting unlearning data into subsets based on
specific attribute values, unlearning each subset separately, and merging the
unlearned models. We empirically demonstrate that SPUNGE significantly improves
the performance of two recent unlearning methods on state-of-the-art LLMs while
maintaining their general capabilities on standard academic benchmarks.



---

58. **[[2402.08100] Investigating the Impact of Data Contamination of Large Language Models
  in Text-to-SQL Translation](https://arxiv.org/pdf/2402.08100.pdf)**

Updated on 2024-12-10

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

59. **[[2404.05880] Eraser: Jailbreaking Defense in Large Language Models via Unlearning
  Harmful Knowledge](https://arxiv.org/pdf/2404.05880.pdf)**

Updated on 2024-07-04

*Weikai Lu, Ziqian Zeng, Jianwei Wang, Zhengdong Lu, Zelin Chen, Huiping Zhuang, Cen Chen*


  Jailbreaking attacks can enable Large Language Models (LLMs) to bypass the
safeguard and generate harmful content. Existing jailbreaking defense methods
have failed to address the fundamental issue that harmful knowledge resides
within the model, leading to potential jailbreak risks for LLMs. In this paper,
we propose a novel defense method called Eraser, which mainly includes three
goals: unlearning harmful knowledge, retaining general knowledge, and
maintaining safety alignment. The intuition is that if an LLM forgets the
specific knowledge required to answer a harmful question, it will no longer
have the ability to answer harmful questions. The training of Erase does not
actually require the model's own harmful knowledge, and it can benefit from
unlearning general answers related to harmful queries, which means it does not
need assistance from the red team. The experimental results show that Eraser
can significantly reduce the jailbreaking success rate for various attacks
without compromising the general capabilities of the model. Our codes are
available at https://github.com/ZeroNLP/Eraser.



---

60. **[[2411.04388] Unlearning in- vs. out-of-distribution data in LLMs under gradient-based
  method](https://arxiv.org/pdf/2411.04388.pdf)**

Updated on 2024-11-08

*Teodora Baluta, Pascal Lamblin, Daniel Tarlow, Fabian Pedregosa, Gintare Karolina Dziugaite*


  Machine unlearning aims to solve the problem of removing the influence of
selected training examples from a learned model. Despite the increasing
attention to this problem, it remains an open research question how to evaluate
unlearning in large language models (LLMs), and what are the critical
properties of the data to be unlearned that affect the quality and efficiency
of unlearning. This work formalizes a metric to evaluate unlearning quality in
generative models, and uses it to assess the trade-offs between unlearning
quality and performance. We demonstrate that unlearning out-of-distribution
examples requires more unlearning steps but overall presents a better trade-off
overall. For in-distribution examples, however, we observe a rapid decay in
performance as unlearning progresses. We further evaluate how example's
memorization and difficulty affect unlearning under a classical gradient
ascent-based approach.



---

61. **[[2404.13968] Protecting Your LLMs with Information Bottleneck](https://arxiv.org/pdf/2404.13968.pdf)**

Updated on 2024-10-11

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

62. **[[2411.18797] UOE: Unlearning One Expert Is Enough For Mixture-of-experts LLMS](https://arxiv.org/pdf/2411.18797.pdf)**

Updated on 2024-12-02

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

63. **[[2412.13670] AntiLeak-Bench: Preventing Data Contamination by Automatically
  Constructing Benchmarks with Updated Real-World Knowledge](https://arxiv.org/pdf/2412.13670.pdf)**

Updated on 2024-12-19

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

64. **[[2406.18382] Adversarial Search Engine Optimization for Large Language Models](https://arxiv.org/pdf/2406.18382.pdf)**

Updated on 2024-07-03

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

65. **[[2502.19982] Erasing Without Remembering: Safeguarding Knowledge Forgetting in Large
  Language Models](https://arxiv.org/pdf/2502.19982.pdf)**

Updated on 2025-02-28

*Huazheng Wang, Yongcheng Jing, Haifeng Sun, Yingjie Wang, Jingyu Wang, Jianxin Liao, Dacheng Tao*


  In this paper, we explore machine unlearning from a novel dimension, by
studying how to safeguard model unlearning in large language models (LLMs). Our
goal is to prevent unlearned models from recalling any related memory of the
targeted knowledge.We begin by uncovering a surprisingly simple yet overlooked
fact: existing methods typically erase only the exact expressions of the
targeted knowledge, leaving paraphrased or related information intact. To
rigorously measure such oversights, we introduce UGBench, the first benchmark
tailored for evaluating the generalisation performance across 13
state-of-the-art methods.UGBench reveals that unlearned models can still recall
paraphrased answers and retain target facts in intermediate layers. To address
this, we propose PERMU, a perturbation-based method that significantly enhances
the generalisation capabilities for safeguarding LLM unlearning.Experiments
demonstrate that PERMU delivers up to a 50.13% improvement in unlearning while
maintaining a 43.53% boost in robust generalisation. Our code can be found in
https://github.com/MaybeLizzy/UGBench.



---

66. **[[2501.13683] Unlearning Clients, Features and Samples in Vertical Federated Learning](https://arxiv.org/pdf/2501.13683.pdf)**

Updated on 2025-01-24

*Ayush K. Varshney, Konstantinos Vandikas, Vicenç Torra*


  Federated Learning (FL) has emerged as a prominent distributed learning
paradigm. Within the scope of privacy preservation, information privacy
regulations such as GDPR entitle users to request the removal (or unlearning)
of their contribution from a service that is hosting the model. For this
purpose, a server hosting an ML model must be able to unlearn certain
information in cases such as copyright infringement or security issues that can
make the model vulnerable or impact the performance of a service based on that
model. While most unlearning approaches in FL focus on Horizontal FL (HFL),
where clients share the feature space and the global model, Vertical FL (VFL)
has received less attention from the research community. VFL involves clients
(passive parties) sharing the sample space among them while not having access
to the labels. In this paper, we explore unlearning in VFL from three
perspectives: unlearning clients, unlearning features, and unlearning samples.
To unlearn clients and features we introduce VFU-KD which is based on knowledge
distillation (KD) while to unlearn samples, VFU-GA is introduced which is based
on gradient ascent. To provide evidence of approximate unlearning, we utilize
Membership Inference Attack (MIA) to audit the effectiveness of our unlearning
approach. Our experiments across six tabular datasets and two image datasets
demonstrate that VFU-KD and VFU-GA achieve performance comparable to or better
than both retraining from scratch and the benchmark R2S method in many cases,
with improvements of $(0-2\%)$. In the remaining cases, utility scores remain
comparable, with a modest utility loss ranging from $1-5\%$. Unlike existing
methods, VFU-KD and VFU-GA require no communication between active and passive
parties during unlearning. However, they do require the active party to store
the previously communicated embeddings.



---

67. **[[2412.18621] Investigating the Feasibility of Mitigating Potential Copyright
  Infringement via Large Language Model Unlearning](https://arxiv.org/pdf/2412.18621.pdf)**

Updated on 2025-02-12

*Guangyao Dou*


  Pre-trained Large Language Models (LLMs) have demonstrated remarkable
capabilities but also pose risks by learning and generating copyrighted
material, leading to significant legal and ethical concerns. In a potential
real-world scenario, model owners may need to continuously address copyright
infringement in order to address requests for content removal that emerge at
different time points. One potential way of addressing this is via sequential
unlearning, where copyrighted content is removed sequentially as new requests
arise. Despite its practical relevance, sequential unlearning in the context of
copyright infringement has not been rigorously explored in existing literature.
To address this gap, we propose Stable Sequential Unlearning (SSU), a novel
framework designed to unlearn copyrighted content from LLMs over multiple time
steps. Our approach works by identifying and removing specific weight updates
in the model's parameters that correspond to copyrighted content using task
vectors. We improve unlearning efficacy by introducing random labeling loss and
ensuring the model retains its general-purpose knowledge by adjusting targeted
parameters with gradient-based weight saliency. Extensive experimental results
show that SSU sometimes achieves an effective trade-off between unlearning
efficacy and general-purpose language abilities, outperforming existing
baselines, but it's not a cure-all for unlearning copyrighted material.



---

68. **[[2410.04454] Inner-Probe: Discovering Copyright-related Data Generation in LLM
  Architecture](https://arxiv.org/pdf/2410.04454.pdf)**

Updated on 2025-01-24

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

69. **[[2503.04795] Cyber for AI at SemEval-2025 Task 4: Forgotten but Not Lost: The
  Balancing Act of Selective Unlearning in Large Language Models](https://arxiv.org/pdf/2503.04795.pdf)**

Updated on 2025-03-10

*Dinesh Srivasthav P, Bala Mallikarjunarao Garlapati*


  Large Language Models (LLMs) face significant challenges in maintaining
privacy, ethics, and compliance, when sensitive or obsolete data must be
selectively removed. Retraining these models from scratch is computationally
infeasible, necessitating efficient alternatives. As part of the SemEval 2025
Task 4, this work focuses on the application of selective unlearning in LLMs to
address this challenge. In this paper, we present our experiments and findings,
primarily leveraging global weight modification to achieve an equilibrium
between effectiveness of unlearning, knowledge retention, and target model's
post-unlearning utility. We also detail the task-specific evaluation mechanism,
results, and challenges. Our algorithms have achieved an aggregate score of
0.409 and 0.389 on the test set for 7B and 1B target models, respectively,
demonstrating promising results in verifiable LLM unlearning.



---

70. **[[2408.10608] Promoting Equality in Large Language Models: Identifying and Mitigating
  the Implicit Bias based on Bayesian Theory](https://arxiv.org/pdf/2408.10608.pdf)**

Updated on 2024-08-21

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

71. **[[2402.05624] Efficient Models for the Detection of Hate, Abuse and Profanity](https://arxiv.org/pdf/2402.05624.pdf)**

Updated on 2024-02-09

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

72. **[[2503.13733] CoDet-M4: Detecting Machine-Generated Code in Multi-Lingual,
  Multi-Generator and Multi-Domain Settings](https://arxiv.org/pdf/2503.13733.pdf)**

Updated on 2025-03-19

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

73. **[[2502.12520] SafeEraser: Enhancing Safety in Multimodal Large Language Models through
  Multimodal Machine Unlearning](https://arxiv.org/pdf/2502.12520.pdf)**

Updated on 2025-02-26

*Junkai Chen, Zhijie Deng, Kening Zheng, Yibo Yan, Shuliang Liu, PeiJun Wu, Peijie Jiang, Jia Liu, Xuming Hu*


  As Multimodal Large Language Models (MLLMs) develop, their potential security
issues have become increasingly prominent. Machine Unlearning (MU), as an
effective strategy for forgetting specific knowledge in training data, has been
widely used in privacy protection. However, MU for safety in MLLM has yet to be
fully explored. To address this issue, we propose SAFEERASER, a safety
unlearning benchmark for MLLMs, consisting of 3,000 images and 28.8K VQA pairs.
We comprehensively evaluate unlearning methods from two perspectives: forget
quality and model utility. Our findings show that existing MU methods struggle
to maintain model performance while implementing the forget operation and often
suffer from over-forgetting. Hence, we introduce Prompt Decouple (PD) Loss to
alleviate over-forgetting through decouple prompt during unlearning process. To
quantitatively measure over-forgetting mitigated by PD Loss, we propose a new
metric called Safe Answer Refusal Rate (SARR). Experimental results demonstrate
that combining PD Loss with existing unlearning methods can effectively prevent
over-forgetting and achieve a decrease of 79.5% in the SARR metric of LLaVA-7B
and LLaVA-13B, while maintaining forget quality and model utility. Our code and
dataset will be released upon acceptance. Warning: This paper contains examples
of harmful language and images, and reader discretion is recommended.



---

74. **[[2011.12355] Lethean Attack: An Online Data Poisoning Technique](https://arxiv.org/pdf/2011.12355.pdf)**

Updated on 2020-11-26

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

75. **[[2410.18966] Does Data Contamination Detection Work (Well) for LLMs? A Survey and
  Evaluation on Detection Assumptions](https://arxiv.org/pdf/2410.18966.pdf)**

Updated on 2025-03-11

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

76. **[[2503.22948] SUV: Scalable Large Language Model Copyright Compliance with Regularized
  Selective Unlearning](https://arxiv.org/pdf/2503.22948.pdf)**

Updated on 2025-04-01

*Tianyang Xu, Xiaoze Liu, Feijie Wu, Xiaoqian Wang, Jing Gao*


  Large Language Models (LLMs) have transformed natural language processing by
learning from massive datasets, yet this rapid progress has also drawn legal
scrutiny, as the ability to unintentionally generate copyrighted content has
already prompted several prominent lawsuits. In this work, we introduce SUV
(Selective Unlearning for Verbatim data), a selective unlearning framework
designed to prevent LLM from memorizing copyrighted content while preserving
its overall utility. In detail, the proposed method constructs a dataset that
captures instances of copyrighted infringement cases by the targeted LLM. With
the dataset, we unlearn the content from the LLM by means of Direct Preference
Optimization (DPO), which replaces the verbatim copyrighted content with
plausible and coherent alternatives. Since DPO may hinder the LLM's performance
in other unrelated tasks, we integrate gradient projection and Fisher
information regularization to mitigate the degradation. We validate our
approach using a large-scale dataset of 500 famous books (predominantly
copyrighted works) and demonstrate that SUV significantly reduces verbatim
memorization with negligible impact on the performance on unrelated tasks.
Extensive experiments on both our dataset and public benchmarks confirm the
scalability and efficacy of our approach, offering a promising solution for
mitigating copyright risks in real-world LLM applications.



---

77. **[[2409.11844] MEOW: MEMOry Supervised LLM Unlearning Via Inverted Facts](https://arxiv.org/pdf/2409.11844.pdf)**

Updated on 2024-09-19

*Tianle Gu, Kexin Huang, Ruilin Luo, Yuanqi Yao, Yujiu Yang, Yan Teng, Yingchun Wang*


  Large Language Models (LLMs) can memorize sensitive information, raising
concerns about potential misuse. LLM Unlearning, a post-hoc approach to remove
this information from trained LLMs, offers a promising solution to mitigate
these risks. However, previous practices face three key challenges: 1. Utility:
successful unlearning often causes catastrophic collapse on unrelated tasks. 2.
Efficiency: many methods either involve adding similarly sized models, which
slows down unlearning or inference, or require retain data that are difficult
to obtain. 3. Robustness: even effective methods may still leak data via
extraction techniques. To address these challenges, we propose MEOW, a simple
yet effective gradient descent-based unlearning method. Specifically, we use an
offline LLM to generate a set of inverted facts. Then, we design a new metric,
MEMO, to quantify memorization in LLMs. Finally, based on the signals provided
by MEMO, we select the most appropriate set of inverted facts and finetune the
model based on them. We evaluate MEOW on the commonly used unlearn benchmark,
ToFU, with Llama2-7B-Chat and Phi-1.5B, and test it on both NLU and NLG tasks.
Results demonstrate significant improvement of MEOW in forget quality without
substantial loss in model utility. Meanwhile, MEOW does not exhibit significant
degradation in NLU or NLG capabilities, and there is even a slight improvement
in NLU performance.



---

78. **[[2411.09689] LLM Hallucination Reasoning with Zero-shot Knowledge Test](https://arxiv.org/pdf/2411.09689.pdf)**

Updated on 2024-11-15

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

79. **[[2502.13347] Craw4LLM: Efficient Web Crawling for LLM Pretraining](https://arxiv.org/pdf/2502.13347.pdf)**

Updated on 2025-02-26

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

80. **[[2409.09288] Generating API Parameter Security Rules with LLM for API Misuse
  Detection](https://arxiv.org/pdf/2409.09288.pdf)**

Updated on 2024-09-20

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

81. **[[2503.02443] AILS-NTUA at SemEval-2025 Task 4: Parameter-Efficient Unlearning for
  Large Language Models using Data Chunking](https://arxiv.org/pdf/2503.02443.pdf)**

Updated on 2025-03-05

*Iraklis Premptis, Maria Lymperaiou, Giorgos Filandrianos, Orfeas Menis Mastromichalakis, Athanasios Voulodimos, Giorgos Stamou*


  The Unlearning Sensitive Content from Large Language Models task aims to
remove targeted datapoints from trained models while minimally affecting their
general knowledge. In our work, we leverage parameter-efficient, gradient-based
unlearning using low-rank (LoRA) adaptation and layer-focused fine-tuning. To
further enhance unlearning effectiveness, we employ data chunking, splitting
forget data into disjoint partitions and merging them with cyclically sampled
retain samples at a pre-defined ratio. Our task-agnostic method achieves an
outstanding forget-retain balance, ranking first on leaderboards and
significantly outperforming baselines and competing systems.



---

82. **[[2407.01920] To Forget or Not? Towards Practical Knowledge Unlearning for Large
  Language Models](https://arxiv.org/pdf/2407.01920.pdf)**

Updated on 2024-10-08

*Bozhong Tian, Xiaozhuan Liang, Siyuan Cheng, Qingbin Liu, Mengru Wang, Dianbo Sui, Xi Chen, Huajun Chen, Ningyu Zhang*


  Large Language Models (LLMs) trained on extensive corpora inevitably retain
sensitive data, such as personal privacy information and copyrighted material.
Recent advancements in knowledge unlearning involve updating LLM parameters to
erase specific knowledge. However, current unlearning paradigms are mired in
vague forgetting boundaries, often erasing knowledge indiscriminately. In this
work, we introduce KnowUnDo, a benchmark containing copyrighted content and
user privacy domains to evaluate if the unlearning process inadvertently erases
essential knowledge. Our findings indicate that existing unlearning methods
often suffer from excessive unlearning. To address this, we propose a simple
yet effective method, MemFlex, which utilizes gradient information to precisely
target and unlearn sensitive parameters. Experimental results show that MemFlex
is superior to existing methods in both precise knowledge unlearning and
general knowledge retaining of LLMs. Code and dataset are released at
https://github.com/zjunlp/KnowUnDo.



---

83. **[[2405.04032] Locally Differentially Private In-Context Learning](https://arxiv.org/pdf/2405.04032.pdf)**

Updated on 2024-05-09

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

84. **[[2502.14182] Multi-Faceted Studies on Data Poisoning can Advance LLM Development](https://arxiv.org/pdf/2502.14182.pdf)**

Updated on 2025-02-21

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

85. **[[2312.07200] Code Membership Inference for Detecting Unauthorized Data Use in Code
  Pre-trained Language Models](https://arxiv.org/pdf/2312.07200.pdf)**

Updated on 2025-02-19

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

86. **[[2406.08754] StructuralSleight: Automated Jailbreak Attacks on Large Language Models
  Utilizing Uncommon Text-Organization Structures](https://arxiv.org/pdf/2406.08754.pdf)**

Updated on 2025-02-19

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

87. **[[2410.15267] When Machine Unlearning Meets Retrieval-Augmented Generation (RAG): Keep
  Secret or Forget Knowledge?](https://arxiv.org/pdf/2410.15267.pdf)**

Updated on 2024-10-22

*Shang Wang, Tianqing Zhu, Dayong Ye, Wanlei Zhou*


  The deployment of large language models (LLMs) like ChatGPT and Gemini has
shown their powerful natural language generation capabilities. However, these
models can inadvertently learn and retain sensitive information and harmful
content during training, raising significant ethical and legal concerns. To
address these issues, machine unlearning has been introduced as a potential
solution. While existing unlearning methods take into account the specific
characteristics of LLMs, they often suffer from high computational demands,
limited applicability, or the risk of catastrophic forgetting. To address these
limitations, we propose a lightweight unlearning framework based on
Retrieval-Augmented Generation (RAG) technology. By modifying the external
knowledge base of RAG, we simulate the effects of forgetting without directly
interacting with the unlearned LLM. We approach the construction of unlearned
knowledge as a constrained optimization problem, deriving two key components
that underpin the effectiveness of RAG-based unlearning. This RAG-based
approach is particularly effective for closed-source LLMs, where existing
unlearning methods often fail. We evaluate our framework through extensive
experiments on both open-source and closed-source models, including ChatGPT,
Gemini, Llama-2-7b-chat-hf, and PaLM 2. The results demonstrate that our
approach meets five key unlearning criteria: effectiveness, universality,
harmlessness, simplicity, and robustness. Meanwhile, this approach can extend
to multimodal large language models and LLM-based agents.



---

88. **[[2408.10668] Probing the Safety Response Boundary of Large Language Models via Unsafe
  Decoding Path Generation](https://arxiv.org/pdf/2408.10668.pdf)**

Updated on 2024-08-27

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

89. **[[2410.17509] WAGLE: Strategic Weight Attribution for Effective and Modular Unlearning
  in Large Language Models](https://arxiv.org/pdf/2410.17509.pdf)**

Updated on 2025-04-15

*Jinghan Jia, Jiancheng Liu, Yihua Zhang, Parikshit Ram, Nathalie Baracaldo, Sijia Liu*


  The need for effective unlearning mechanisms in large language models (LLMs)
is increasingly urgent, driven by the necessity to adhere to data regulations
and foster ethical generative AI practices. Despite growing interest of LLM
unlearning, much of the existing research has focused on varied unlearning
method designs to boost effectiveness and efficiency. However, the inherent
relationship between model weights and LLM unlearning has not been extensively
examined. In this paper, we systematically explore how model weights interact
with unlearning processes in LLMs and we design the weight attribution-guided
LLM unlearning method, WAGLE, which unveils the interconnections between
'influence' of weights and 'influence' of data to forget and retain in LLM
generation. By strategically guiding the LLM unlearning across different types
of unlearning methods and tasks, WAGLE can erase the undesired content, while
maintaining the performance of the original tasks. We refer to the weight
attribution-guided LLM unlearning method as WAGLE, which unveils the
interconnections between 'influence' of weights and 'influence' of data to
forget and retain in LLM generation. Our extensive experiments show that WAGLE
boosts unlearning performance across a range of LLM unlearning methods such as
gradient difference and (negative) preference optimization, applications such
as fictitious unlearning, malicious use prevention, and copyrighted information
removal, and models including Zephyr-7b-beta and Llama2-7b. To the best of our
knowledge, our work offers the first principled method for attributing and
pinpointing the influential weights in enhancing LLM unlearning. It stands in
contrast to previous methods that lack weight attribution and simpler weight
attribution techniques.



---

90. **[[2503.00062] CRFU: Compressive Representation Forgetting Against Privacy Leakage on
  Machine Unlearning](https://arxiv.org/pdf/2503.00062.pdf)**

Updated on 2025-03-04

*Weiqi Wang, Chenhan Zhang, Zhiyi Tian, Shushu Liu, Shui Yu*


  Machine unlearning allows data owners to erase the impact of their specified
data from trained models. Unfortunately, recent studies have shown that
adversaries can recover the erased data, posing serious threats to user
privacy. An effective unlearning method removes the information of the
specified data from the trained model, resulting in different outputs for the
same input before and after unlearning. Adversaries can exploit these output
differences to conduct privacy leakage attacks, such as reconstruction and
membership inference attacks. However, directly applying traditional defenses
to unlearning leads to significant model utility degradation. In this paper, we
introduce a Compressive Representation Forgetting Unlearning scheme (CRFU),
designed to safeguard against privacy leakage on unlearning. CRFU achieves data
erasure by minimizing the mutual information between the trained compressive
representation (learned through information bottleneck theory) and the erased
data, thereby maximizing the distortion of data. This ensures that the model's
output contains less information that adversaries can exploit. Furthermore, we
introduce a remembering constraint and an unlearning rate to balance the
forgetting of erased data with the preservation of previously learned
knowledge, thereby reducing accuracy degradation. Theoretical analysis
demonstrates that CRFU can effectively defend against privacy leakage attacks.
Our experimental results show that CRFU significantly increases the
reconstruction mean square error (MSE), achieving a defense effect improvement
of approximately $200\%$ against privacy reconstruction attacks with only
$1.5\%$ accuracy degradation on MNIST.



---

91. **[[2406.08607] Reversing the Forget-Retain Objectives: An Efficient LLM Unlearning
  Framework from Logit Difference](https://arxiv.org/pdf/2406.08607.pdf)**

Updated on 2024-06-14

*Jiabao Ji, Yujian Liu, Yang Zhang, Gaowen Liu, Ramana Rao Kompella, Sijia Liu, Shiyu Chang*


  As Large Language Models (LLMs) demonstrate extensive capability in learning
from documents, LLM unlearning becomes an increasingly important research area
to address concerns of LLMs in terms of privacy, copyright, etc. A conventional
LLM unlearning task typically involves two goals: (1) The target LLM should
forget the knowledge in the specified forget documents, and (2) it should
retain the other knowledge that the LLM possesses, for which we assume access
to a small number of retain documents. To achieve both goals, a mainstream
class of LLM unlearning methods introduces an optimization framework with a
combination of two objectives - maximizing the prediction loss on the forget
documents while minimizing that on the retain documents, which suffers from two
challenges, degenerated output and catastrophic forgetting. In this paper, we
propose a novel unlearning framework called Unlearning from Logit Difference
(ULD), which introduces an assistant LLM that aims to achieve the opposite of
the unlearning goals: remembering the forget documents and forgetting the
retain knowledge. ULD then derives the unlearned LLM by computing the logit
difference between the target and the assistant LLMs. We show that such
reversed objectives would naturally resolve both aforementioned challenges
while significantly improving the training efficiency. Extensive experiments
demonstrate that our method efficiently achieves the intended forgetting while
preserving the LLM's overall capabilities, reducing training time by more than
threefold. Notably, our method loses 0% of model utility on the ToFU benchmark,
whereas baseline methods may sacrifice 17% of utility on average to achieve
comparable forget quality. Our code will be publicly available at
https://github.com/UCSB-NLP-Chang/ULD.



---

92. **[[2410.12777] Meta-Unlearning on Diffusion Models: Preventing Relearning Unlearned
  Concepts](https://arxiv.org/pdf/2410.12777.pdf)**

Updated on 2024-10-17

*Hongcheng Gao, Tianyu Pang, Chao Du, Taihang Hu, Zhijie Deng, Min Lin*


  With the rapid progress of diffusion-based content generation, significant
efforts are being made to unlearn harmful or copyrighted concepts from
pretrained diffusion models (DMs) to prevent potential model misuse. However,
it is observed that even when DMs are properly unlearned before release,
malicious finetuning can compromise this process, causing DMs to relearn the
unlearned concepts. This occurs partly because certain benign concepts (e.g.,
"skin") retained in DMs are related to the unlearned ones (e.g., "nudity"),
facilitating their relearning via finetuning. To address this, we propose
meta-unlearning on DMs. Intuitively, a meta-unlearned DM should behave like an
unlearned DM when used as is; moreover, if the meta-unlearned DM undergoes
malicious finetuning on unlearned concepts, the related benign concepts
retained within it will be triggered to self-destruct, hindering the relearning
of unlearned concepts. Our meta-unlearning framework is compatible with most
existing unlearning methods, requiring only the addition of an
easy-to-implement meta objective. We validate our approach through empirical
experiments on meta-unlearning concepts from Stable Diffusion models (SD-v1-4
and SDXL), supported by extensive ablation studies. Our code is available at
https://github.com/sail-sg/Meta-Unlearning.



---

93. **[[2407.10223] On Large Language Model Continual Unlearning](https://arxiv.org/pdf/2407.10223.pdf)**

Updated on 2025-03-04

*Chongyang Gao, Lixu Wang, Kaize Ding, Chenkai Weng, Xiao Wang, Qi Zhu*


  While large language models have demonstrated impressive performance across
various domains and tasks, their security issues have become increasingly
severe. Machine unlearning has emerged as a representative approach for model
safety and security by removing the influence of undesired data on the target
model. However, these methods do not sufficiently consider that unlearning
requests in real-world scenarios are continuously emerging, especially in the
context of LLMs, which may lead to accumulated model utility loss that
eventually becomes unacceptable. Moreover, existing LLM unlearning methods
often ignore previous data access limitations due to privacy concerns and
copyright protection. Without previous data, the utility preservation during
unlearning is much harder. To overcome these challenges, we propose the OOO
framework that includes an Orthogonal low-rank adapter (LoRA) for continually
unlearning requested data and an Out-Of-Distribution (OOD) detector to measure
the similarity between input and unlearning data. The orthogonal LoRA achieves
parameter disentanglement among continual unlearning requests. The OOD detector
is trained with a novel contrastive entropy loss and utilizes a glocal-aware
scoring mechanism. During inference, our OOO framework can decide whether and
to what extent to load the unlearning LoRA based on the OOD detector's
predicted similarity between the input and the unlearned knowledge. Notably,
OOO's effectiveness does not rely on any retained data. We conducted extensive
experiments on OOO and state-of-the-art LLM unlearning methods across three
tasks and seven datasets. The results indicate that OOO consistently achieves
the best unlearning effectiveness and utility preservation, especially when
facing continuous unlearning requests. The source codes can be found at
https://github.com/GCYZSL/O3-LLM-UNLEARNING.



---

94. **[[2502.10673] Dataset Protection via Watermarked Canaries in Retrieval-Augmented LLMs](https://arxiv.org/pdf/2502.10673.pdf)**

Updated on 2025-02-18

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

95. **[[2401.10360] Excuse me, sir? Your language model is leaking (information)](https://arxiv.org/pdf/2401.10360.pdf)**

Updated on 2024-11-19

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

96. **[[2402.15109] Remaining-data-free Machine Unlearning by Suppressing Sample
  Contribution](https://arxiv.org/pdf/2402.15109.pdf)**

Updated on 2024-12-09

*Xinwen Cheng, Zhehao Huang, Wenxin Zhou, Zhengbao He, Ruikai Yang, Yingwen Wu, Xiaolin Huang*


  Machine unlearning (MU) is to forget data from a well-trained model, which is
practically important due to the ``right to be forgotten''. The unlearned model
should approach the retrained model, where the forgetting data are not involved
in the training process and hence do not contribute to the retrained model.
Considering the forgetting data's absence during retraining, we think
unlearning should withdraw their contribution from the pre-trained model. The
challenge is that when tracing the learning process is impractical, how to
quantify and detach sample's contribution to the dynamic learning process using
only the pre-trained model. We first theoretically discover that sample's
contribution during the process will reflect in the learned model's sensitivity
to it. We then practically design a novel method, namely MU-Mis (Machine
Unlearning by Minimizing input sensitivity), to suppress the contribution of
the forgetting data. Experimental results demonstrate that MU-Mis can unlearn
effectively and efficiently without utilizing the remaining data. It is the
first time that a remaining-data-free method can outperform state-of-the-art
(SoTA) unlearning methods that utilize the remaining data.



---

97. **[[2402.11846] UnlearnCanvas: Stylized Image Dataset for Enhanced Machine Unlearning
  Evaluation in Diffusion Models](https://arxiv.org/pdf/2402.11846.pdf)**

Updated on 2024-10-31

*Yihua Zhang, Chongyu Fan, Yimeng Zhang, Yuguang Yao, Jinghan Jia, Jiancheng Liu, Gaoyuan Zhang, Gaowen Liu, Ramana Rao Kompella, Xiaoming Liu, Sijia Liu*


  The technological advancements in diffusion models (DMs) have demonstrated
unprecedented capabilities in text-to-image generation and are widely used in
diverse applications. However, they have also raised significant societal
concerns, such as the generation of harmful content and copyright disputes.
Machine unlearning (MU) has emerged as a promising solution, capable of
removing undesired generative capabilities from DMs. However, existing MU
evaluation systems present several key challenges that can result in incomplete
and inaccurate assessments. To address these issues, we propose UnlearnCanvas,
a comprehensive high-resolution stylized image dataset that facilitates the
evaluation of the unlearning of artistic styles and associated objects. This
dataset enables the establishment of a standardized, automated evaluation
framework with 7 quantitative metrics assessing various aspects of the
unlearning performance for DMs. Through extensive experiments, we benchmark 9
state-of-the-art MU methods for DMs, revealing novel insights into their
strengths, weaknesses, and underlying mechanisms. Additionally, we explore
challenging unlearning scenarios for DMs to evaluate worst-case performance
against adversarial prompts, the unlearning of finer-scale concepts, and
sequential unlearning. We hope that this study can pave the way for developing
more effective, accurate, and robust DM unlearning methods, ensuring safer and
more ethical applications of DMs in the future. The dataset, benchmark, and
codes are publicly available at https://unlearn-canvas.netlify.app/.



---

98. **[[2407.06411] If You Don't Understand It, Don't Use It: Eliminating Trojans with
  Filters Between Layers](https://arxiv.org/pdf/2407.06411.pdf)**

Updated on 2024-07-10

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

99. **[[2503.01539] Pragmatic Inference Chain (PIC) Improving LLMs' Reasoning of Authentic
  Implicit Toxic Language](https://arxiv.org/pdf/2503.01539.pdf)**

Updated on 2025-03-04

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

100. **[[2408.10718] CodeJudge-Eval: Can Large Language Models be Good Judges in Code
  Understanding?](https://arxiv.org/pdf/2408.10718.pdf)**

Updated on 2024-09-16

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

101. **[[2504.10185] LLM Unlearning Reveals a Stronger-Than-Expected Coreset Effect in
  Current Benchmarks](https://arxiv.org/pdf/2504.10185.pdf)**

Updated on 2025-04-17

*Soumyadeep Pal, Changsheng Wang, James Diffenderfer, Bhavya Kailkhura, Sijia Liu*


  Large language model unlearning has become a critical challenge in ensuring
safety and controlled model behavior by removing undesired data-model
influences from the pretrained model while preserving general utility.
Significant recent efforts have been dedicated to developing LLM unlearning
benchmarks such as WMDP (Weapons of Mass Destruction Proxy) and MUSE (Machine
Unlearning Six-way Evaluation), facilitating standardized unlearning
performance assessment and method comparison. Despite their usefulness, we
uncover for the first time a novel coreset effect within these benchmarks.
Specifically, we find that LLM unlearning achieved with the original (full)
forget set can be effectively maintained using a significantly smaller subset
(functioning as a "coreset"), e.g., as little as 5% of the forget set, even
when selected at random. This suggests that LLM unlearning in these benchmarks
can be performed surprisingly easily, even in an extremely low-data regime. We
demonstrate that this coreset effect remains strong, regardless of the LLM
unlearning method used, such as NPO (Negative Preference Optimization) and RMU
(Representation Misdirection Unlearning), the popular ones in these benchmarks.
The surprisingly strong coreset effect is also robust across various data
selection methods, ranging from random selection to more sophisticated
heuristic approaches. We explain the coreset effect in LLM unlearning through a
keyword-based perspective, showing that keywords extracted from the forget set
alone contribute significantly to unlearning effectiveness and indicating that
current unlearning is driven by a compact set of high-impact tokens rather than
the entire dataset. We further justify the faithfulness of coreset-unlearned
models along additional dimensions, such as mode connectivity and robustness to
jailbreaking attacks. Codes are available at
https://github.com/OPTML-Group/MU-Coreset.



---

102. **[[2310.18574] Breaking the Trilemma of Privacy, Utility, Efficiency via Controllable
  Machine Unlearning](https://arxiv.org/pdf/2310.18574.pdf)**

Updated on 2024-02-23

*Zheyuan Liu, Guangyao Dou, Yijun Tian, Chunhui Zhang, Eli Chien, Ziwei Zhu*


  Machine Unlearning (MU) algorithms have become increasingly critical due to
the imperative adherence to data privacy regulations. The primary objective of
MU is to erase the influence of specific data samples on a given model without
the need to retrain it from scratch. Accordingly, existing methods focus on
maximizing user privacy protection. However, there are different degrees of
privacy regulations for each real-world web-based application. Exploring the
full spectrum of trade-offs between privacy, model utility, and runtime
efficiency is critical for practical unlearning scenarios. Furthermore,
designing the MU algorithm with simple control of the aforementioned trade-off
is desirable but challenging due to the inherent complex interaction. To
address the challenges, we present Controllable Machine Unlearning (ConMU), a
novel framework designed to facilitate the calibration of MU. The ConMU
framework contains three integral modules: an important data selection module
that reconciles the runtime efficiency and model generalization, a progressive
Gaussian mechanism module that balances privacy and model generalization, and
an unlearning proxy that controls the trade-offs between privacy and runtime
efficiency. Comprehensive experiments on various benchmark datasets have
demonstrated the robust adaptability of our control mechanism and its
superiority over established unlearning methods. ConMU explores the full
spectrum of the Privacy-Utility-Efficiency trade-off and allows practitioners
to account for different real-world regulations. Source code available at:
https://github.com/guangyaodou/ConMU.



---

103. **[[2410.00382] Answer When Needed, Forget When Not: Language Models Pretend to Forget
  via In-Context Knowledge Unlearning](https://arxiv.org/pdf/2410.00382.pdf)**

Updated on 2024-10-02

*Shota Takashiro, Takeshi Kojima, Andrew Gambardella, Qi Cao, Yusuke Iwasawa, Yutaka Matsuo*


  As large language models (LLMs) are applied across diverse domains, the
ability to selectively unlearn specific information has become increasingly
essential. For instance, LLMs are expected to provide confidential information
to authorized internal users, such as employees or trusted partners, while
withholding it from external users, including the general public and
unauthorized entities. In response to this challenge, we propose a novel method
termed ``in-context knowledge unlearning'', which enables the model to
selectively forget information in test-time based on the context of the query.
Our method fine-tunes pre-trained LLMs to enable prompt unlearning of target
knowledge within the context, while preserving other knowledge. Experiments on
the TOFU and AGE datasets using Llama2-7B/13B and Mistral-7B models show our
method achieves up to 95% forgetting accuracy while retaining 80% of unrelated
knowledge, significantly outperforming baselines in both in-domain and
out-of-domain scenarios. Further investigation into the model's internal
behavior revealed that while fine-tuned LLMs generate correct predictions in
the middle layers and maintain them up to the final layer, they make the
decision to forget at the last layer, i.e., ``LLMs pretend to forget''. Our
findings offer valuable insights into enhancing the robustness of unlearning
mechanisms in LLMs, setting a foundation for future research in the field.



---

104. **[[2502.19726] Tokens for Learning, Tokens for Unlearning: Mitigating Membership
  Inference Attacks in Large Language Models via Dual-Purpose Training](https://arxiv.org/pdf/2502.19726.pdf)**

Updated on 2025-02-28

*Toan Tran, Ruixuan Liu, Li Xiong*


  Large language models (LLMs) have become the backbone of modern natural
language processing but pose privacy concerns about leaking sensitive training
data. Membership inference attacks (MIAs), which aim to infer whether a sample
is included in a model's training dataset, can serve as a foundation for
broader privacy threats. Existing defenses designed for traditional
classification models do not account for the sequential nature of text data. As
a result, they either require significant computational resources or fail to
effectively mitigate privacy risks in LLMs. In this work, we propose a
lightweight yet effective empirical privacy defense for protecting training
data of language modeling by leveraging the token-specific characteristics. By
analyzing token dynamics during training, we propose a token selection strategy
that categorizes tokens into hard tokens for learning and memorized tokens for
unlearning. Subsequently, our training-phase defense optimizes a novel
dual-purpose token-level loss to achieve a Pareto-optimal balance between
utility and privacy. Extensive experiments demonstrate that our approach not
only provides strong protection against MIAs but also improves language
modeling performance by around 10\% across various LLM architectures and
datasets compared to the baselines.



---

105. **[[2502.16691] Toward Responsible Federated Large Language Models: Leveraging a Safety
  Filter and Constitutional AI](https://arxiv.org/pdf/2502.16691.pdf)**

Updated on 2025-02-25

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

106. **[[2310.20150] Unlearn What You Want to Forget: Efficient Unlearning for LLMs](https://arxiv.org/pdf/2310.20150.pdf)**

Updated on 2023-11-01

*Jiaao Chen, Diyi Yang*


  Large language models (LLMs) have achieved significant progress from
pre-training on and memorizing a wide range of textual data, however, this
process might suffer from privacy issues and violations of data protection
regulations. As a result, the ability to easily remove data related to
individual users from such models while not deteriorating their predictive
quality after the removal becomes increasingly important. To address these
issues, in this work, we propose an efficient unlearning framework that could
efficiently update LLMs without having to retrain the whole model after data
removals, by introducing lightweight unlearning layers learned with a selective
teacher-student objective into the transformers. In addition, we introduce a
fusion mechanism to effectively combine different unlearning layers that learns
to forget different sets of data to handle a sequence of forgetting operations.
Experiments on classification and generation tasks demonstrate the
effectiveness of our proposed methods compared to the state-of-the-art
baselines.



---

107. **[[2409.04459] WET: Overcoming Paraphrasing Vulnerabilities in Embeddings-as-a-Service
  with Linear Transformation Watermarks](https://arxiv.org/pdf/2409.04459.pdf)**

Updated on 2024-09-10

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

108. **[[2310.02469] PrivacyMind: Large Language Models Can Be Contextual Privacy Protection
  Learners](https://arxiv.org/pdf/2310.02469.pdf)**

Updated on 2024-10-29

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

109. **[[2408.05968] Nob-MIAs: Non-biased Membership Inference Attacks Assessment on Large
  Language Models with Ex-Post Dataset Construction](https://arxiv.org/pdf/2408.05968.pdf)**

Updated on 2025-01-17

*Cédric Eichler, Nathan Champeil, Nicolas Anciaux, Alexandra Bensamoun, Heber Hwang Arcolezi, José Maria De Fuentes*


  The rise of Large Language Models (LLMs) has triggered legal and ethical
concerns, especially regarding the unauthorized use of copyrighted materials in
their training datasets. This has led to lawsuits against tech companies
accused of using protected content without permission. Membership Inference
Attacks (MIAs) aim to detect whether specific documents were used in a given
LLM pretraining, but their effectiveness is undermined by biases such as
time-shifts and n-gram overlaps.
  This paper addresses the evaluation of MIAs on LLMs with partially inferable
training sets, under the ex-post hypothesis, which acknowledges inherent
distributional biases between members and non-members datasets. We propose and
validate algorithms to create ``non-biased'' and ``non-classifiable'' datasets
for fairer MIA assessment. Experiments using the Gutenberg dataset on OpenLamma
and Pythia show that neutralizing known biases alone is insufficient. Our
methods produce non-biased ex-post datasets with AUC-ROC scores comparable to
those previously obtained on genuinely random datasets, validating our
approach. Globally, MIAs yield results close to random, with only one being
effective on both random and our datasets, but its performance decreases when
bias is removed.



---

110. **[[2411.02631] Extracting Unlearned Information from LLMs with Activation Steering](https://arxiv.org/pdf/2411.02631.pdf)**

Updated on 2024-11-06

*Atakan Seyitoğlu, Aleksei Kuvshinov, Leo Schwinn, Stephan Günnemann*


  An unintended consequence of the vast pretraining of Large Language Models
(LLMs) is the verbatim memorization of fragments of their training data, which
may contain sensitive or copyrighted information. In recent years, unlearning
has emerged as a solution to effectively remove sensitive knowledge from models
after training. Yet, recent work has shown that supposedly deleted information
can still be extracted by malicious actors through various attacks. Still,
current attacks retrieve sets of possible candidate generations and are unable
to pinpoint the output that contains the actual target information. We propose
activation steering as a method for exact information retrieval from unlearned
LLMs. We introduce a novel approach to generating steering vectors, named
Anonymized Activation Steering. Additionally, we develop a simple word
frequency method to pinpoint the correct answer among a set of candidates when
retrieving unlearned information. Our evaluation across multiple unlearning
techniques and datasets demonstrates that activation steering successfully
recovers general knowledge (e.g., widely known fictional characters) while
revealing limitations in retrieving specific information (e.g., details about
non-public individuals). Overall, our results demonstrate that exact
information retrieval from unlearned models is possible, highlighting a severe
vulnerability of current unlearning techniques.



---

111. **[[2305.10036] Are You Copying My Model? Protecting the Copyright of Large Language
  Models for EaaS via Backdoor Watermark](https://arxiv.org/pdf/2305.10036.pdf)**

Updated on 2023-06-05

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

112. **[[2402.13459] Learning to Poison Large Language Models During Instruction Tuning](https://arxiv.org/pdf/2402.13459.pdf)**

Updated on 2024-10-24

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

113. **[[2406.06571] SUBLLM: A Novel Efficient Architecture with Token Sequence Subsampling
  for LLM](https://arxiv.org/pdf/2406.06571.pdf)**

Updated on 2024-08-26

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

114. **[[2503.21598] Prompt, Divide, and Conquer: Bypassing Large Language Model Safety
  Filters via Segmented and Distributed Prompt Processing](https://arxiv.org/pdf/2503.21598.pdf)**

Updated on 2025-04-01

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

115. **[[2412.13879] Crabs: Consuming Resource via Auto-generation for LLM-DoS Attack under
  Black-box Settings](https://arxiv.org/pdf/2412.13879.pdf)**

Updated on 2025-02-19

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

116. **[[2503.22760] Malicious and Unintentional Disclosure Risks in Large Language Models
  for Code Generation](https://arxiv.org/pdf/2503.22760.pdf)**

Updated on 2025-04-01

*Rafiqul Rabin, Sean McGregor, Nick Judd*


  This paper explores the risk that a large language model (LLM) trained for
code generation on data mined from software repositories will generate content
that discloses sensitive information included in its training data. We
decompose this risk, known in the literature as ``unintended memorization,''
into two components: unintentional disclosure (where an LLM presents secrets to
users without the user seeking them out) and malicious disclosure (where an LLM
presents secrets to an attacker equipped with partial knowledge of the training
data). We observe that while existing work mostly anticipates malicious
disclosure, unintentional disclosure is also a concern. We describe methods to
assess unintentional and malicious disclosure risks side-by-side across
different releases of training datasets and models. We demonstrate these
methods through an independent assessment of the Open Language Model (OLMo)
family of models and its Dolma training datasets. Our results show, first, that
changes in data source and processing are associated with substantial changes
in unintended memorization risk; second, that the same set of operational
changes may increase one risk while mitigating another; and, third, that the
risk of disclosing sensitive information varies not only by prompt strategies
or test datasets but also by the types of sensitive information. These
contributions rely on data mining to enable greater privacy and security
testing required for the LLM training data supply chain.



---

117. **[[2305.01639] Privacy-Preserving In-Context Learning for Large Language Models](https://arxiv.org/pdf/2305.01639.pdf)**

Updated on 2023-10-03

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

118. **[[2502.19785] SCU: An Efficient Machine Unlearning Scheme for Deep Learning Enabled
  Semantic Communications](https://arxiv.org/pdf/2502.19785.pdf)**

Updated on 2025-02-28

*Weiqi Wang, Zhiyi Tian, Chenhan Zhang, Shui Yu*


  Deep learning (DL) enabled semantic communications leverage DL to train
encoders and decoders (codecs) to extract and recover semantic information.
However, most semantic training datasets contain personal private information.
Such concerns call for enormous requirements for specified data erasure from
semantic codecs when previous users hope to move their data from the semantic
system. {Existing machine unlearning solutions remove data contribution from
trained models, yet usually in supervised sole model scenarios. These methods
are infeasible in semantic communications that often need to jointly train
unsupervised encoders and decoders.} In this paper, we investigate the
unlearning problem in DL-enabled semantic communications and propose a semantic
communication unlearning (SCU) scheme to tackle the problem. {SCU includes two
key components. Firstly,} we customize the joint unlearning method for semantic
codecs, including the encoder and decoder, by minimizing mutual information
between the learned semantic representation and the erased samples. {Secondly,}
to compensate for semantic model utility degradation caused by unlearning, we
propose a contrastive compensation method, which considers the erased data as
the negative samples and the remaining data as the positive samples to retrain
the unlearned semantic models contrastively. Theoretical analysis and extensive
experimental results on three representative datasets demonstrate the
effectiveness and efficiency of our proposed methods.



---

119. **[[2311.09827] Cognitive Overload: Jailbreaking Large Language Models with Overloaded
  Logical Thinking](https://arxiv.org/pdf/2311.09827.pdf)**

Updated on 2024-03-01

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

120. **[[2410.21723] Fine-tuning Large Language Models for DGA and DNS Exfiltration Detection](https://arxiv.org/pdf/2410.21723.pdf)**

Updated on 2024-11-08

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

121. **[[2403.05156] On Protecting the Data Privacy of Large Language Models (LLMs): A Survey](https://arxiv.org/pdf/2403.05156.pdf)**

Updated on 2024-03-15

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

122. **[[2406.16810] How Data Inter-connectivity Shapes LLMs Unlearning: A Structural
  Unlearning Perspective](https://arxiv.org/pdf/2406.16810.pdf)**

Updated on 2025-03-12

*Xinchi Qiu, William F. Shen, Yihong Chen, Meghdad Kurmanji, Nicola Cancedda, Pontus Stenetorp, Nicholas D. Lane*


  While unlearning knowledge from large language models (LLMs) is receiving
increasing attention, one important aspect remains unexplored. Existing
approaches and benchmarks assume data points to-be-forgotten are independent,
ignoring their inter-connectivity - a fundamental characteristic of real-world
data structures. In this paper, we propose PISTOL, a method for compiling
structural datasets. PISTOL leverages the inherently structured nature of
contractual relationships, offering several key benefits. First, it enables
insights into the impact of structural data on unlearning effectiveness.
Second, it provides precise and concise ground truths for clearer evaluation.
Third, its attribute generation does not require input from pre-trained LLMs,
mitigating confounding risks. Leveraging datasets synthesized using PISTOL, we
demonstrate how data inter-connectivity impacts LLM unlearning. Specifically,
(a) in both the pre-trained and fine-tuned models, unlearning difficulty
increases as data inter-connectivity grows, (b) there is a positive correlation
between the density of the knowledge graph and unlearning difficulty, and (c)
when the to-be-forgotten data is skewed towards one domain, balancing retaining
performance across all domains is challenging.



---

123. **[[2406.01983] RKLD: Reverse KL-Divergence-based Knowledge Distillation for Unlearning
  Personal Information in Large Language Models](https://arxiv.org/pdf/2406.01983.pdf)**

Updated on 2024-06-05

*Bichen Wang, Yuzhe Zi, Yixin Sun, Yanyan Zhao, Bing Qin*


  With the passage of the Right to Be Forgotten (RTBF) regulations and the
scaling up of language model training datasets, research on model unlearning in
large language models (LLMs) has become more crucial. Before the era of LLMs,
machine unlearning research focused mainly on classification tasks in models
with small parameters. In these tasks, the content to be forgotten or retained
is clear and straightforward. However, as parameter sizes have grown and tasks
have become more complex, balancing forget quality and model utility has become
more challenging, especially in scenarios involving personal data instead of
classification results. Existing methods based on gradient ascent and its
variants often struggle with this balance, leading to unintended information
loss or partial forgetting. To address this challenge, we propose RKLD, a novel
\textbf{R}everse \textbf{KL}-Divergence-based Knowledge \textbf{D}istillation
unlearning algorithm for LLMs targeting the unlearning of personal information.
Through RKLD, we achieve significant forget quality and effectively maintain
the model utility in our experiments.



---

124. **[[2502.07340] Aligning Large Language Models to Follow Instructions and Hallucinate
  Less via Effective Data Filtering](https://arxiv.org/pdf/2502.07340.pdf)**

Updated on 2025-02-18

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

125. **[[2410.13903] CoreGuard: Safeguarding Foundational Capabilities of LLMs Against Model
  Stealing in Edge Deployment](https://arxiv.org/pdf/2410.13903.pdf)**

Updated on 2024-10-21

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

126. **[[2405.15234] Defensive Unlearning with Adversarial Training for Robust Concept
  Erasure in Diffusion Models](https://arxiv.org/pdf/2405.15234.pdf)**

Updated on 2024-10-10

*Yimeng Zhang, Xin Chen, Jinghan Jia, Yihua Zhang, Chongyu Fan, Jiancheng Liu, Mingyi Hong, Ke Ding, Sijia Liu*


  Diffusion models (DMs) have achieved remarkable success in text-to-image
generation, but they also pose safety risks, such as the potential generation
of harmful content and copyright violations. The techniques of machine
unlearning, also known as concept erasing, have been developed to address these
risks. However, these techniques remain vulnerable to adversarial prompt
attacks, which can prompt DMs post-unlearning to regenerate undesired images
containing concepts (such as nudity) meant to be erased. This work aims to
enhance the robustness of concept erasing by integrating the principle of
adversarial training (AT) into machine unlearning, resulting in the robust
unlearning framework referred to as AdvUnlearn. However, achieving this
effectively and efficiently is highly nontrivial. First, we find that a
straightforward implementation of AT compromises DMs' image generation quality
post-unlearning. To address this, we develop a utility-retaining regularization
on an additional retain set, optimizing the trade-off between concept erasure
robustness and model utility in AdvUnlearn. Moreover, we identify the text
encoder as a more suitable module for robustification compared to UNet,
ensuring unlearning effectiveness. And the acquired text encoder can serve as a
plug-and-play robust unlearner for various DM types. Empirically, we perform
extensive experiments to demonstrate the robustness advantage of AdvUnlearn
across various DM unlearning scenarios, including the erasure of nudity,
objects, and style concepts. In addition to robustness, AdvUnlearn also
achieves a balanced tradeoff with model utility. To our knowledge, this is the
first work to systematically explore robust DM unlearning through AT, setting
it apart from existing methods that overlook robustness in concept erasing.
Codes are available at: https://github.com/OPTML-Group/AdvUnlearn



---

127. **[[2501.19202] Improving the Robustness of Representation Misdirection for Large
  Language Model Unlearning](https://arxiv.org/pdf/2501.19202.pdf)**

Updated on 2025-02-04

*Dang Huu-Tien, Hoang Thanh-Tung, Le-Minh Nguyen, Naoya Inoue*


  Representation Misdirection (RM) and variants are established large language
model (LLM) unlearning methods with state-of-the-art performance. In this
paper, we show that RM methods inherently reduce models' robustness, causing
them to misbehave even when a single non-adversarial forget-token is in the
retain-query. Toward understanding underlying causes, we reframe the unlearning
process as backdoor attacks and defenses: forget-tokens act as backdoor
triggers that, when activated in retain-queries, cause disruptions in RM
models' behaviors, similar to successful backdoor attacks. To mitigate this
vulnerability, we propose Random Noise Augmentation -- a model and method
agnostic approach with theoretical guarantees for improving the robustness of
RM methods. Extensive experiments demonstrate that RNA significantly improves
the robustness of RM models while enhancing the unlearning performances.



---

128. **[[2310.02238] Who's Harry Potter? Approximate Unlearning in LLMs](https://arxiv.org/pdf/2310.02238.pdf)**

Updated on 2023-10-05

*Ronen Eldan, Mark Russinovich*


  Large language models (LLMs) are trained on massive internet corpora that
often contain copyrighted content. This poses legal and ethical challenges for
the developers and users of these models, as well as the original authors and
publishers. In this paper, we propose a novel technique for unlearning a subset
of the training data from a LLM, without having to retrain it from scratch.
  We evaluate our technique on the task of unlearning the Harry Potter books
from the Llama2-7b model (a generative language model recently open-sourced by
Meta). While the model took over 184K GPU-hours to pretrain, we show that in
about 1 GPU hour of finetuning, we effectively erase the model's ability to
generate or recall Harry Potter-related content, while its performance on
common benchmarks (such as Winogrande, Hellaswag, arc, boolq and piqa) remains
almost unaffected. We make our fine-tuned model publicly available on
HuggingFace for community evaluation. To the best of our knowledge, this is the
first paper to present an effective technique for unlearning in generative
language models.
  Our technique consists of three main components: First, we use a reinforced
model that is further trained on the target data to identify the tokens that
are most related to the unlearning target, by comparing its logits with those
of a baseline model. Second, we replace idiosyncratic expressions in the target
data with generic counterparts, and leverage the model's own predictions to
generate alternative labels for every token. These labels aim to approximate
the next-token predictions of a model that has not been trained on the target
data. Third, we finetune the model on these alternative labels, which
effectively erases the original text from the model's memory whenever it is
prompted with its context.



---

129. **[[2410.14273] REEF: Representation Encoding Fingerprints for Large Language Models](https://arxiv.org/pdf/2410.14273.pdf)**

Updated on 2024-10-21

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

130. **[[2503.23566] When LLM Therapists Become Salespeople: Evaluating Large Language Models
  for Ethical Motivational Interviewing](https://arxiv.org/pdf/2503.23566.pdf)**

Updated on 2025-04-01

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

131. **[[2403.03536] Towards Efficient and Effective Unlearning of Large Language Models for
  Recommendation](https://arxiv.org/pdf/2403.03536.pdf)**

Updated on 2024-07-02

*Hangyu Wang, Jianghao Lin, Bo Chen, Yang Yang, Ruiming Tang, Weinan Zhang, Yong Yu*


  The significant advancements in large language models (LLMs) give rise to a
promising research direction, i.e., leveraging LLMs as recommenders (LLMRec).
The efficacy of LLMRec arises from the open-world knowledge and reasoning
capabilities inherent in LLMs. LLMRec acquires the recommendation capabilities
through instruction tuning based on user interaction data. However, in order to
protect user privacy and optimize utility, it is also crucial for LLMRec to
intentionally forget specific user data, which is generally referred to as
recommendation unlearning. In the era of LLMs, recommendation unlearning poses
new challenges for LLMRec in terms of \textit{inefficiency} and
\textit{ineffectiveness}. Existing unlearning methods require updating billions
of parameters in LLMRec, which is costly and time-consuming. Besides, they
always impact the model utility during the unlearning process. To this end, we
propose \textbf{E2URec}, the first \underline{E}fficient and
\underline{E}ffective \underline{U}nlearning method for LLM\underline{Rec}. Our
proposed E2URec enhances the unlearning efficiency by updating only a few
additional LoRA parameters, and improves the unlearning effectiveness by
employing a teacher-student framework, where we maintain multiple teacher
networks to guide the unlearning process. Extensive experiments show that
E2URec outperforms state-of-the-art baselines on two real-world datasets.
Specifically, E2URec can efficiently forget specific data without affecting
recommendation performance. The source code is at
\url{https://github.com/justarter/E2URec}.



---

132. **[[2406.18326] PaCoST: Paired Confidence Significance Testing for Benchmark
  Contamination Detection in Large Language Models](https://arxiv.org/pdf/2406.18326.pdf)**

Updated on 2025-03-19

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

133. **[[2406.12329] Opt-Out: Investigating Entity-Level Unlearning for Large Language Models
  via Optimal Transport](https://arxiv.org/pdf/2406.12329.pdf)**

Updated on 2024-12-05

*Minseok Choi, Daniel Rim, Dohyun Lee, Jaegul Choo*


  Instruction-following large language models (LLMs), such as ChatGPT, have
become widely popular among everyday users. However, these models inadvertently
disclose private, sensitive information to their users, underscoring the need
for machine unlearning techniques to remove selective information from the
models. While prior work has focused on forgetting small, random subsets of
training data at the instance-level, we argue that real-world scenarios often
require the removal of an entire user data, which may require a more careful
maneuver. In this study, we explore entity-level unlearning, which aims to
erase all knowledge related to a target entity while preserving the remaining
model capabilities. To address this, we introduce Opt-Out, an optimal
transport-based unlearning method that utilizes the Wasserstein distance from
the model's initial parameters to achieve more effective and fine-grained
unlearning. We also present the first Entity-Level Unlearning Dataset (ELUDe)
designed to evaluate entity-level unlearning. Our empirical results demonstrate
that Opt-Out surpasses existing methods, establishing a new standard for secure
and adaptable LLMs that can accommodate user data removal requests without the
need for full retraining.



---

134. **[[2308.08090] Separate the Wheat from the Chaff: Model Deficiency Unlearning via
  Parameter-Efficient Module Operation](https://arxiv.org/pdf/2308.08090.pdf)**

Updated on 2024-01-19

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

135. **[[2310.14444] URegM: a unified prediction model of resource consumption for
  refactoring software smells in open source cloud](https://arxiv.org/pdf/2310.14444.pdf)**

Updated on 2023-10-24

*Asif Imran, Tevfik Kosar*


  The low cost and rapid provisioning capabilities have made the cloud a
desirable platform to launch complex scientific applications. However, resource
utilization optimization is a significant challenge for cloud service
providers, since the earlier focus is provided on optimizing resources for the
applications that run on the cloud, with a low emphasis being provided on
optimizing resource utilization of the cloud computing internal processes. Code
refactoring has been associated with improving the maintenance and
understanding of software code. However, analyzing the impact of the
refactoring source code of the cloud and studying its impact on cloud resource
usage require further analysis. In this paper, we propose a framework called
Unified Regression Modelling (URegM) which predicts the impact of code smell
refactoring on cloud resource usage. We test our experiments in a real-life
cloud environment using a complex scientific application as a workload. Results
show that URegM is capable of accurately predicting resource consumption due to
code smell refactoring. This will permit cloud service providers with advanced
knowledge about the impact of refactoring code smells on resource consumption,
thus allowing them to plan their resource provisioning and code refactoring
more effectively.



---

136. **[[2412.15194] MMLU-CF: A Contamination-free Multi-task Language Understanding
  Benchmark](https://arxiv.org/pdf/2412.15194.pdf)**

Updated on 2024-12-20

*Qihao Zhao, Yangyu Huang, Tengchao Lv, Lei Cui, Qinzheng Sun, Shaoguang Mao, Xin Zhang, Ying Xin, Qiufeng Yin, Scarlett Li, Furu Wei*


  Multiple-choice question (MCQ) datasets like Massive Multitask Language
Understanding (MMLU) are widely used to evaluate the commonsense,
understanding, and problem-solving abilities of large language models (LLMs).
However, the open-source nature of these benchmarks and the broad sources of
training data for LLMs have inevitably led to benchmark contamination,
resulting in unreliable evaluation results. To alleviate this issue, we propose
a contamination-free and more challenging MCQ benchmark called MMLU-CF. This
benchmark reassesses LLMs' understanding of world knowledge by averting both
unintentional and malicious data leakage. To avoid unintentional data leakage,
we source data from a broader domain and design three decontamination rules. To
prevent malicious data leakage, we divide the benchmark into validation and
test sets with similar difficulty and subject distributions. The test set
remains closed-source to ensure reliable results, while the validation set is
publicly available to promote transparency and facilitate independent
verification. Our evaluation of mainstream LLMs reveals that the powerful
GPT-4o achieves merely a 5-shot score of 73.4% and a 0-shot score of 71.9% on
the test set, which indicates the effectiveness of our approach in creating a
more rigorous and contamination-free evaluation standard. The GitHub repository
is available at https://github.com/microsoft/MMLU-CF and the dataset refers to
https://huggingface.co/datasets/microsoft/MMLU-CF.



---

137. **[[2403.10557] Second-Order Information Matters: Revisiting Machine Unlearning for
  Large Language Models](https://arxiv.org/pdf/2403.10557.pdf)**

Updated on 2024-03-19

*Kang Gu, Md Rafi Ur Rashid, Najrin Sultana, Shagufta Mehnaz*


  With the rapid development of Large Language Models (LLMs), we have witnessed
intense competition among the major LLM products like ChatGPT, LLaMa, and
Gemini. However, various issues (e.g. privacy leakage and copyright violation)
of the training corpus still remain underexplored. For example, the Times sued
OpenAI and Microsoft for infringing on its copyrights by using millions of its
articles for training. From the perspective of LLM practitioners, handling such
unintended privacy violations can be challenging. Previous work addressed the
``unlearning" problem of LLMs using gradient information, while they mostly
introduced significant overheads like data preprocessing or lacked robustness.
In this paper, contrasting with the methods based on first-order information,
we revisit the unlearning problem via the perspective of second-order
information (Hessian). Our unlearning algorithms, which are inspired by classic
Newton update, are not only data-agnostic/model-agnostic but also proven to be
robust in terms of utility preservation or privacy guarantee. Through a
comprehensive evaluation with four NLP datasets as well as a case study on
real-world datasets, our methods consistently show superiority over the
first-order methods.



---

138. **[[2407.10582] Boosting Zero-Shot Crosslingual Performance using LLM-Based
  Augmentations with Effective Data Selection](https://arxiv.org/pdf/2407.10582.pdf)**

Updated on 2024-07-16

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

139. **[[2407.10058] Learning to Refuse: Towards Mitigating Privacy Risks in LLMs](https://arxiv.org/pdf/2407.10058.pdf)**

Updated on 2024-09-17

*Zhenhua Liu, Tong Zhu, Chuanyuan Tan, Wenliang Chen*


  Large language models (LLMs) exhibit remarkable capabilities in understanding
and generating natural language. However, these models can inadvertently
memorize private information, posing significant privacy risks. This study
addresses the challenge of enabling LLMs to protect specific individuals'
private data without the need for complete retraining. We propose \return, a
Real-world pErsonal daTa UnleaRNing dataset, comprising 2,492 individuals from
Wikipedia with associated QA pairs, to evaluate machine unlearning (MU) methods
for protecting personal data in a realistic scenario. Additionally, we
introduce the Name-Aware Unlearning Framework (NAUF) for Privacy Protection,
which enables the model to learn which individuals' information should be
protected without affecting its ability to answer questions related to other
unrelated individuals. Our extensive experiments demonstrate that NAUF achieves
a state-of-the-art average unlearning score, surpassing the best baseline
method by 5.65 points, effectively protecting target individuals' personal data
while maintaining the model's general capabilities.



---

140. **[[2504.00018] SandboxEval: Towards Securing Test Environment for Untrusted Code](https://arxiv.org/pdf/2504.00018.pdf)**

Updated on 2025-04-02

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

141. **[[2410.14425] Unlearning Backdoor Attacks for LLMs with Weak-to-Strong Knowledge
  Distillation](https://arxiv.org/pdf/2410.14425.pdf)**

Updated on 2024-10-21

*Shuai Zhao, Xiaobao Wu, Cong-Duy Nguyen, Meihuizi Jia, Yichao Feng, Luu Anh Tuan*


  Parameter-efficient fine-tuning (PEFT) can bridge the gap between large
language models (LLMs) and downstream tasks. However, PEFT has been proven
vulnerable to malicious attacks. Research indicates that poisoned LLMs, even
after PEFT, retain the capability to activate internalized backdoors when input
samples contain predefined triggers. In this paper, we introduce a novel
weak-to-strong unlearning algorithm to defend against backdoor attacks based on
feature alignment knowledge distillation, named W2SDefense. Specifically, we
first train a small-scale language model through full-parameter fine-tuning to
serve as the clean teacher model. Then, this teacher model guides the
large-scale poisoned student model in unlearning the backdoor, leveraging PEFT.
Theoretical analysis suggests that W2SDefense has the potential to enhance the
student model's ability to unlearn backdoor features, preventing the activation
of the backdoor. We conduct experiments on text classification tasks involving
three state-of-the-art language models and three different backdoor attack
algorithms. Our empirical results demonstrate the outstanding performance of
W2SDefense in defending against backdoor attacks without compromising model
performance.



---

142. **[[2305.04547] Diffusion Theory as a Scalpel: Detecting and Purifying Poisonous
  Dimensions in Pre-trained Language Models Caused by Backdoor or Bias](https://arxiv.org/pdf/2305.04547.pdf)**

Updated on 2023-05-09

*Zhiyuan Zhang, Deli Chen, Hao Zhou, Fandong Meng, Jie Zhou, Xu Sun*


  Pre-trained Language Models (PLMs) may be poisonous with backdoors or bias
injected by the suspicious attacker during the fine-tuning process. A core
challenge of purifying potentially poisonous PLMs is precisely finding
poisonous dimensions. To settle this issue, we propose the Fine-purifying
approach, which utilizes the diffusion theory to study the dynamic process of
fine-tuning for finding potentially poisonous dimensions. According to the
relationship between parameter drifts and Hessians of different dimensions, we
can detect poisonous dimensions with abnormal dynamics, purify them by
resetting them to clean pre-trained weights, and then fine-tune the purified
weights on a small clean dataset. To the best of our knowledge, we are the
first to study the dynamics guided by the diffusion theory for safety or
defense purposes. Experimental results validate the effectiveness of
Fine-purifying even with a small clean dataset.



---

143. **[[2312.16823] Layer Attack Unlearning: Fast and Accurate Machine Unlearning via Layer
  Level Attack and Knowledge Distillation](https://arxiv.org/pdf/2312.16823.pdf)**

Updated on 2023-12-29

*Hyunjune Kim, Sangyong Lee, Simon S. Woo*


  Recently, serious concerns have been raised about the privacy issues related
to training datasets in machine learning algorithms when including personal
data. Various regulations in different countries, including the GDPR grant
individuals to have personal data erased, known as 'the right to be forgotten'
or 'the right to erasure'. However, there has been less research on effectively
and practically deleting the requested personal data from the training set
while not jeopardizing the overall machine learning performance. In this work,
we propose a fast and novel machine unlearning paradigm at the layer level
called layer attack unlearning, which is highly accurate and fast compared to
existing machine unlearning algorithms. We introduce the Partial-PGD algorithm
to locate the samples to forget efficiently. In addition, we only use the last
layer of the model inspired by the Forward-Forward algorithm for unlearning
process. Lastly, we use Knowledge Distillation (KD) to reliably learn the
decision boundaries from the teacher using soft label information to improve
accuracy performance. We conducted extensive experiments with SOTA machine
unlearning models and demonstrated the effectiveness of our approach for
accuracy and end-to-end unlearning performance.



---

144. **[[2311.18815] IMMA: Immunizing text-to-image Models against Malicious Adaptation](https://arxiv.org/pdf/2311.18815.pdf)**

Updated on 2024-10-01

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

145. **[[2311.16822] Large Language Models Suffer From Their Own Output: An Analysis of the
  Self-Consuming Training Loop](https://arxiv.org/pdf/2311.16822.pdf)**

Updated on 2024-06-18

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

146. **[[2501.10915] LegalGuardian: A Privacy-Preserving Framework for Secure Integration of
  Large Language Models in Legal Practice](https://arxiv.org/pdf/2501.10915.pdf)**

Updated on 2025-01-22

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

147. **[[2403.09681] ViT-MUL: A Baseline Study on Recent Machine Unlearning Methods Applied
  to Vision Transformers](https://arxiv.org/pdf/2403.09681.pdf)**

Updated on 2024-03-18

*Ikhyun Cho, Changyeon Park, Julia Hockenmaier*


  Machine unlearning (MUL) is an arising field in machine learning that seeks
to erase the learned information of specific training data points from a
trained model. Despite the recent active research in MUL within computer
vision, the majority of work has focused on ResNet-based models. Given that
Vision Transformers (ViT) have become the predominant model architecture, a
detailed study of MUL specifically tailored to ViT is essential. In this paper,
we present comprehensive experiments on ViTs using recent MUL algorithms and
datasets. We anticipate that our experiments, ablation studies, and findings
could provide valuable insights and inspire further research in this field.



---

148. **[[2403.03883] SaulLM-7B: A pioneering Large Language Model for Law](https://arxiv.org/pdf/2403.03883.pdf)**

Updated on 2024-03-08

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

149. **[[2311.05553] Removing RLHF Protections in GPT-4 via Fine-Tuning](https://arxiv.org/pdf/2311.05553.pdf)**

Updated on 2024-04-09

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

150. **[[2306.06815] TrojLLM: A Black-box Trojan Prompt Attack on Large Language Models](https://arxiv.org/pdf/2306.06815.pdf)**

Updated on 2023-11-01

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

151. **[[2502.05374] Towards LLM Unlearning Resilient to Relearning Attacks: A
  Sharpness-Aware Minimization Perspective and Beyond](https://arxiv.org/pdf/2502.05374.pdf)**

Updated on 2025-03-26

*Chongyu Fan, Jinghan Jia, Yihua Zhang, Anil Ramakrishna, Mingyi Hong, Sijia Liu*


  The LLM unlearning technique has recently been introduced to comply with data
regulations and address the safety and ethical concerns of LLMs by removing the
undesired data-model influence. However, state-of-the-art unlearning methods
face a critical vulnerability: they are susceptible to ``relearning'' the
removed information from a small number of forget data points, known as
relearning attacks. In this paper, we systematically investigate how to make
unlearned models robust against such attacks. For the first time, we establish
a connection between robust unlearning and sharpness-aware minimization (SAM)
through a unified robust optimization framework, in an analogy to adversarial
training designed to defend against adversarial attacks. Our analysis for SAM
reveals that smoothness optimization plays a pivotal role in mitigating
relearning attacks. Thus, we further explore diverse smoothing strategies to
enhance unlearning robustness. Extensive experiments on benchmark datasets,
including WMDP and MUSE, demonstrate that SAM and other smoothness optimization
approaches consistently improve the resistance of LLM unlearning to relearning
attacks. Notably, smoothness-enhanced unlearning also helps defend against
(input-level) jailbreaking attacks, broadening our proposal's impact in
robustifying LLM unlearning. Codes are available at
https://github.com/OPTML-Group/Unlearn-Smooth.



---

152. **[[2312.07910] PromptBench: A Unified Library for Evaluation of Large Language Models](https://arxiv.org/pdf/2312.07910.pdf)**

Updated on 2024-08-21

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

153. **[[2501.18280] Jailbreaking LLMs' Safeguard with Universal Magic Words for Text
  Embedding Models](https://arxiv.org/pdf/2501.18280.pdf)**

Updated on 2025-02-11

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

154. **[[2412.06846] Classifier-free guidance in LLMs Safety](https://arxiv.org/pdf/2412.06846.pdf)**

Updated on 2024-12-11

*Roman Smirnov*


  The paper describes LLM unlearning without a retaining dataset, using the
ORPO reinforcement learning method with inference enhanced by modified
classifier-free guidance. Significant improvement in unlearning, without
degradation of the model, is achieved through direct training on synthetic
replacement data in CFG-aware training regime, with classifier-free guidance
applied during the inference. This article is an extended version of the
NeurIPS 2024 LLM-PC submission, which was awarded second prize.



---

155. **[[2309.01446] Open Sesame! Universal Black Box Jailbreaking of Large Language Models](https://arxiv.org/pdf/2309.01446.pdf)**

Updated on 2024-08-06

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

156. **[[2412.16039] SafeCFG: Redirecting Harmful Classifier-Free Guidance for Safe
  Generation](https://arxiv.org/pdf/2412.16039.pdf)**

Updated on 2024-12-23

*Jiadong Pan, Hongcheng Gao, Liang Li, Zheng-Jun Zha, Qingming Huang, Jiebo Luo*


  Diffusion models (DMs) have demonstrated exceptional performance in
text-to-image (T2I) tasks, leading to their widespread use. With the
introduction of classifier-free guidance (CFG), the quality of images generated
by DMs is improved. However, DMs can generate more harmful images by
maliciously guiding the image generation process through CFG. Some safe
guidance methods aim to mitigate the risk of generating harmful images but
often reduce the quality of clean image generation. To address this issue, we
introduce the Harmful Guidance Redirector (HGR), which redirects harmful CFG
direction while preserving clean CFG direction during image generation,
transforming CFG into SafeCFG and achieving high safety and quality generation.
We train HGR to redirect multiple harmful CFG directions simultaneously,
demonstrating its ability to eliminate various harmful elements while
preserving high-quality generation. Additionally, we find that HGR can detect
image harmfulness, allowing for unsupervised fine-tuning of safe diffusion
models without pre-defined clean or harmful labels. Experimental results show
that by incorporating HGR, images generated by diffusion models achieve both
high quality and strong safety, and safe DMs trained through unsupervised
methods according to the harmfulness detected by HGR also exhibit good safety
performance. The codes will be publicly available.



---

157. **[[2307.14810] A Differential Datalog Interpreter](https://arxiv.org/pdf/2307.14810.pdf)**

Updated on 2023-08-24

*Matthew Stephenson*


  Redacted by arXiv admins



---

158. **[[2007.03548] Breaking and Fixing Destructive Code Read Defenses](https://arxiv.org/pdf/2007.03548.pdf)**

Updated on 2020-07-08

*Jannik Pewny, Philipp Koppe, Lucas Davi, Thorsten Holz*


  Just-in-time return-oriented programming (JIT-ROP) is a powerful memory
corruption attack that bypasses various forms of code randomization.
Execute-only memory (XOM) can potentially prevent these attacks, but requires
source code. In contrast, destructive code reads (DCR) provide a trade-off
between security and legacy compatibility. The common belief is that DCR
provides strong protection if combined with a high-entropy code randomization.
  The contribution of this paper is twofold: first, we demonstrate that DCR can
be bypassed regardless of the underlying code randomization scheme. To this
end, we show novel, generic attacks that infer the code layout for highly
randomized program code. Second, we present the design and implementation of
BGDX (Byte-Granular DCR and XOM), a novel mitigation technique that protects
legacy binaries against code inference attacks. BGDX enforces memory
permissions on a byte-granular level allowing us to combine DCR and XOM for
legacy, off-the-shelf binaries. Our evaluation shows that BGDX is not only
effective, but highly efficient, imposing only a geometric mean performance
overhead of 3.95% on SPEC.



---

159. **[[2410.22108] Protecting Privacy in Multimodal Large Language Models with MLLMU-Bench](https://arxiv.org/pdf/2410.22108.pdf)**

Updated on 2025-02-18

*Zheyuan Liu, Guangyao Dou, Mengzhao Jia, Zhaoxuan Tan, Qingkai Zeng, Yongle Yuan, Meng Jiang*


  Generative models such as Large Language Models (LLM) and Multimodal Large
Language models (MLLMs) trained on massive web corpora can memorize and
disclose individuals' confidential and private data, raising legal and ethical
concerns. While many previous works have addressed this issue in LLM via
machine unlearning, it remains largely unexplored for MLLMs. To tackle this
challenge, we introduce Multimodal Large Language Model Unlearning Benchmark
(MLLMU-Bench), a novel benchmark aimed at advancing the understanding of
multimodal machine unlearning. MLLMU-Bench consists of 500 fictitious profiles
and 153 profiles for public celebrities, each profile feature over 14
customized question-answer pairs, evaluated from both multimodal (image+text)
and unimodal (text) perspectives. The benchmark is divided into four sets to
assess unlearning algorithms in terms of efficacy, generalizability, and model
utility. Finally, we provide baseline results using existing generative model
unlearning algorithms. Surprisingly, our experiments show that unimodal
unlearning algorithms excel in generation and cloze tasks, while multimodal
unlearning approaches perform better in classification tasks with multimodal
inputs.



---

160. **[[2408.02946] Data Poisoning in LLMs: Jailbreak-Tuning and Scaling Laws](https://arxiv.org/pdf/2408.02946.pdf)**

Updated on 2024-12-31

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

161. **[[2410.16848] ETHIC: Evaluating Large Language Models on Long-Context Tasks with High
  Information Coverage](https://arxiv.org/pdf/2410.16848.pdf)**

Updated on 2025-02-28

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

162. **[[2311.02105] Making Harmful Behaviors Unlearnable for Large Language Models](https://arxiv.org/pdf/2311.02105.pdf)**

Updated on 2023-11-07

*Xin Zhou, Yi Lu, Ruotian Ma, Tao Gui, Qi Zhang, Xuanjing Huang*


  Large language models (LLMs) have shown great potential as general-purpose AI
assistants in various domains. To meet the requirements of different
applications, LLMs are often customized by further fine-tuning. However, the
powerful learning ability of LLMs not only enables them to acquire new tasks
but also makes them susceptible to learning undesired behaviors. For example,
even safety-aligned LLMs can be easily fine-tuned into harmful assistants as
the fine-tuning data often contains implicit or explicit harmful content. Can
we train LLMs on harmful data without learning harmful behaviors? This paper
proposes a controllable training framework that makes harmful behaviors
unlearnable during the fine-tuning process. Specifically, we introduce
``security vectors'', a few new parameters that can be separated from the LLM,
to ensure LLM's responses are consistent with the harmful behavior. Security
vectors are activated during fine-tuning, the consistent behavior makes LLM
believe that such behavior has already been learned, there is no need to
further optimize for harmful data. During inference, we can deactivate security
vectors to restore the LLM's normal behavior. The experimental results show
that the security vectors generated by 100 harmful samples are enough to
prevent LLM from learning 1000 harmful samples, while preserving the ability to
learn other useful information.



---

163. **[[2405.05610] Chain of Attack: a Semantic-Driven Contextual Multi-Turn attacker for
  LLM](https://arxiv.org/pdf/2405.05610.pdf)**

Updated on 2024-05-10

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

164. **[[2406.09325] REVS: Unlearning Sensitive Information in Language Models via Rank
  Editing in the Vocabulary Space](https://arxiv.org/pdf/2406.09325.pdf)**

Updated on 2025-02-18

*Tomer Ashuach, Martin Tutek, Yonatan Belinkov*


  Language models (LMs) risk inadvertently memorizing and divulging sensitive
or personally identifiable information (PII) seen in training data, causing
privacy concerns. Current approaches to address this issue involve costly
dataset scrubbing, or model filtering through unlearning and model editing,
which can be bypassed through extraction attacks. We propose REVS, a novel
non-gradient-based method for unlearning sensitive information from LMs. REVS
identifies and modifies a small subset of neurons relevant for constituent
tokens that form sensitive information. To adequately evaluate our method on
truly sensitive information, we curate three datasets: email and URL datasets
naturally memorized by the models, and a synthetic social security number
dataset that we tune the models to memorize. Compared to other methods, REVS
demonstrates superior performance in unlearning sensitive information and
robustness to extraction attacks, while retaining underlying model integrity.



---

165. **[[2406.01333] Probing Language Models for Pre-training Data Detection](https://arxiv.org/pdf/2406.01333.pdf)**

Updated on 2024-06-04

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

166. **[[2411.10351] Bias Unveiled: Investigating Social Bias in LLM-Generated Code](https://arxiv.org/pdf/2411.10351.pdf)**

Updated on 2025-03-10

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

167. **[[2503.16740] Automated Harmfulness Testing for Code Large Language Models](https://arxiv.org/pdf/2503.16740.pdf)**

Updated on 2025-03-24

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

168. **[[2309.10283] FRAMU: Attention-based Machine Unlearning using Federated Reinforcement
  Learning](https://arxiv.org/pdf/2309.10283.pdf)**

Updated on 2024-10-28

*Thanveer Shaik, Xiaohui Tao, Lin Li, Haoran Xie, Taotao Cai, Xiaofeng Zhu, Qing Li*


  Machine Unlearning is an emerging field that addresses data privacy issues by
enabling the removal of private or irrelevant data from the Machine Learning
process. Challenges related to privacy and model efficiency arise from the use
of outdated, private, and irrelevant data. These issues compromise both the
accuracy and the computational efficiency of models in both Machine Learning
and Unlearning. To mitigate these challenges, we introduce a novel framework,
Attention-based Machine Unlearning using Federated Reinforcement Learning
(FRAMU). This framework incorporates adaptive learning mechanisms, privacy
preservation techniques, and optimization strategies, making it a well-rounded
solution for handling various data sources, either single-modality or
multi-modality, while maintaining accuracy and privacy. FRAMU's strength lies
in its adaptability to fluctuating data landscapes, its ability to unlearn
outdated, private, or irrelevant data, and its support for continual model
evolution without compromising privacy. Our experiments, conducted on both
single-modality and multi-modality datasets, revealed that FRAMU significantly
outperformed baseline models. Additional assessments of convergence behavior
and optimization strategies further validate the framework's utility in
federated learning applications. Overall, FRAMU advances Machine Unlearning by
offering a robust, privacy-preserving solution that optimizes model performance
while also addressing key challenges in dynamic data environments.



---

169. **[[2412.07261] MemHunter: Automated and Verifiable Memorization Detection at
  Dataset-scale in LLMs](https://arxiv.org/pdf/2412.07261.pdf)**

Updated on 2025-02-18

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

170. **[[2404.19048] A Framework for Real-time Safeguarding the Text Generation of Large
  Language Model](https://arxiv.org/pdf/2404.19048.pdf)**

Updated on 2024-05-03

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

171. **[[2405.11466] Measuring Impacts of Poisoning on Model Parameters and Embeddings for
  Large Language Models of Code](https://arxiv.org/pdf/2405.11466.pdf)**

Updated on 2024-05-21

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

172. **[[2407.12504] Case2Code: Scalable Synthetic Data for Code Generation](https://arxiv.org/pdf/2407.12504.pdf)**

Updated on 2025-02-11

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

173. **[[2408.04870] ConfusedPilot: Confused Deputy Risks in RAG-based LLMs](https://arxiv.org/pdf/2408.04870.pdf)**

Updated on 2024-10-24

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

174. **[[2402.16893] The Good and The Bad: Exploring Privacy Issues in Retrieval-Augmented
  Generation (RAG)](https://arxiv.org/pdf/2402.16893.pdf)**

Updated on 2024-03-03

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

175. **[[2406.07036] Paying More Attention to Source Context: Mitigating Unfaithful
  Translations from Large Language Model](https://arxiv.org/pdf/2406.07036.pdf)**

Updated on 2024-06-12

*Hongbin Zhang, Kehai Chen, Xuefeng Bai, Yang Xiang, Min Zhang*


  Large language models (LLMs) have showcased impressive multilingual machine
translation ability. However, unlike encoder-decoder style models, decoder-only
LLMs lack an explicit alignment between source and target contexts. Analyzing
contribution scores during generation processes revealed that LLMs can be
biased towards previously generated tokens over corresponding source tokens,
leading to unfaithful translations. To address this issue, we propose to
encourage LLMs to pay more attention to the source context from both source and
target perspectives in zeroshot prompting: 1) adjust source context attention
weights; 2) suppress irrelevant target prefix influence; Additionally, we
propose 3) avoiding over-reliance on the target prefix in instruction tuning.
Experimental results from both human-collected unfaithfulness test sets
focusing on LLM-generated unfaithful translations and general test sets, verify
our methods' effectiveness across multiple language pairs. Further human
evaluation shows our method's efficacy in reducing hallucinatory translations
and facilitating faithful translation generation.



---

176. **[[2410.16186] Contamination Report for Multilingual Benchmarks](https://arxiv.org/pdf/2410.16186.pdf)**

Updated on 2024-10-22

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

177. **[[2403.00826] LLMGuard: Guarding Against Unsafe LLM Behavior](https://arxiv.org/pdf/2403.00826.pdf)**

Updated on 2024-03-05

*Shubh Goyal, Medha Hira, Shubham Mishra, Sukriti Goyal, Arnav Goel, Niharika Dadu, Kirushikesh DB, Sameep Mehta, Nishtha Madaan*


  Although the rise of Large Language Models (LLMs) in enterprise settings
brings new opportunities and capabilities, it also brings challenges, such as
the risk of generating inappropriate, biased, or misleading content that
violates regulations and can have legal concerns. To alleviate this, we present
"LLMGuard", a tool that monitors user interactions with an LLM application and
flags content against specific behaviours or conversation topics. To do this
robustly, LLMGuard employs an ensemble of detectors.



---

178. **[[2306.05670] One-Shot Machine Unlearning with Mnemonic Code](https://arxiv.org/pdf/2306.05670.pdf)**

Updated on 2024-09-26

*Tomoya Yamashita, Masanori Yamada, Takashi Shibata*


  Ethical and privacy issues inherent in artificial intelligence (AI)
applications have been a growing concern with the rapid spread of deep
learning. Machine unlearning (MU) is the research area that addresses these
issues by making a trained AI model forget about undesirable training data.
Unfortunately, most existing MU methods incur significant time and
computational costs for forgetting. Therefore, it is often difficult to apply
these methods to practical datasets and sophisticated architectures, e.g.,
ImageNet and Transformer. To tackle this problem, we propose a lightweight and
effective MU method. Our method identifies the model parameters sensitive to
the forgetting targets and adds perturbation to such model parameters. We
identify the sensitive parameters by calculating the Fisher Information Matrix
(FIM). This approach does not require time-consuming additional training for
forgetting. In addition, we introduce class-specific random signals called
mnemonic code to reduce the cost of FIM calculation, which generally requires
the entire training data and incurs significant computational costs. In our
method, we train the model with mnemonic code; when forgetting, we use a small
number of mnemonic codes to calculate the FIM and get the effective
perturbation for forgetting. Comprehensive experiments demonstrate that our
method is faster and better at forgetting than existing MU methods.
Furthermore, we show that our method can scale to more practical datasets and
sophisticated architectures.



---

179. **[[2401.11467] Over-Reasoning and Redundant Calculation of Large Language Models](https://arxiv.org/pdf/2401.11467.pdf)**

Updated on 2024-03-21

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

180. **[[2502.13416] Detecting LLM Fact-conflicting Hallucinations Enhanced by
  Temporal-logic-based Reasoning](https://arxiv.org/pdf/2502.13416.pdf)**

Updated on 2025-02-20

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

181. **[[2402.02160] Data Poisoning for In-context Learning](https://arxiv.org/pdf/2402.02160.pdf)**

Updated on 2024-03-29

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

182. **[[2409.14038] OAEI-LLM: A Benchmark Dataset for Understanding Large Language Model
  Hallucinations in Ontology Matching](https://arxiv.org/pdf/2409.14038.pdf)**

Updated on 2025-02-04

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

183. **[[2410.02879] Position: LLM Unlearning Benchmarks are Weak Measures of Progress](https://arxiv.org/pdf/2410.02879.pdf)**

Updated on 2025-04-09

*Pratiksha Thaker, Shengyuan Hu, Neil Kale, Yash Maurya, Zhiwei Steven Wu, Virginia Smith*


  Unlearning methods have the potential to improve the privacy and safety of
large language models (LLMs) by removing sensitive or harmful information post
hoc. The LLM unlearning research community has increasingly turned toward
empirical benchmarks to assess the effectiveness of such methods. In this
paper, we find that existing benchmarks provide an overly optimistic and
potentially misleading view on the effectiveness of candidate unlearning
methods. By introducing simple, benign modifications to a number of popular
benchmarks, we expose instances where supposedly unlearned information remains
accessible, or where the unlearning process has degraded the model's
performance on retained information to a much greater extent than indicated by
the original benchmark. We identify that existing benchmarks are particularly
vulnerable to modifications that introduce even loose dependencies between the
forget and retain information. Further, we show that ambiguity in unlearning
targets in existing benchmarks can easily lead to the design of methods that
overfit to the given test queries. Based on our findings, we urge the community
to be cautious when interpreting benchmark results as reliable measures of
progress, and we provide several recommendations to guide future LLM unlearning
research.



---

184. **[[2503.01224] CE-U: Cross Entropy Unlearning](https://arxiv.org/pdf/2503.01224.pdf)**

Updated on 2025-03-18

*Bo Yang*


  Large language models memorize sensitive data from their pretraining corpora.
In this work, we propose CE-U (Cross Entropy Unlearning), a loss function for
unlearning. CE-U addresses fundamental limitations of gradient ascent
approaches that suffer from vanishing gradients when model confidence is high
and exploding gradients when confidence is low. We also unify standard cross
entropy learning and unlearning into a single framework. On the TOFU benchmark
for unlearning, CE-U achieves state-of-the-art results on LLaMA2-7B models
without using an extra oracle model or additional positive samples. Our
analysis reveals that the problematic gradient ascent component also exists in
reinforcement learning algorithms like DPO and GRPO. This suggests that
applying CE-U approach to reinforcement learning could be promising to improve
stability and convergence.



---

185. **[[2406.13236] Data Contamination Can Cross Language Barriers](https://arxiv.org/pdf/2406.13236.pdf)**

Updated on 2024-10-31

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

186. **[[2402.09299] Trained Without My Consent: Detecting Code Inclusion In Language Models
  Trained on Code](https://arxiv.org/pdf/2402.09299.pdf)**

Updated on 2024-11-01

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

187. **[[2312.02052] DUCK: Distance-based Unlearning via Centroid Kinematics](https://arxiv.org/pdf/2312.02052.pdf)**

Updated on 2024-05-07

*Marco Cotogni, Jacopo Bonato, Luigi Sabetta, Francesco Pelosin, Alessandro Nicolosi*


  Machine Unlearning is rising as a new field, driven by the pressing necessity
of ensuring privacy in modern artificial intelligence models. This technique
primarily aims to eradicate any residual influence of a specific subset of data
from the knowledge acquired by a neural model during its training. This work
introduces a novel unlearning algorithm, denoted as Distance-based Unlearning
via Centroid Kinematics (DUCK), which employs metric learning to guide the
removal of samples matching the nearest incorrect centroid in the embedding
space. Evaluation of the algorithm's performance is conducted across various
benchmark datasets in two distinct scenarios, class removal, and homogeneous
sampling removal, obtaining state-of-the-art performance. We also introduce a
novel metric, called Adaptive Unlearning Score (AUS), encompassing not only the
efficacy of the unlearning process in forgetting target data but also
quantifying the performance loss relative to the original model. Additionally,
we conducted a thorough investigation of the unlearning mechanism in DUCK,
examining its impact on the organization of the feature space and employing
explainable AI techniques for deeper insights.



---

188. **[[2401.16603] LeftoverLocals: Listening to LLM Responses Through Leaked GPU Local
  Memory](https://arxiv.org/pdf/2401.16603.pdf)**

Updated on 2024-01-31

*Tyler Sorensen, Heidy Khlaaf*


  This paper describes LeftoverLocals: a vulnerability that allows data
recovery from GPU memory created by another process on Apple, Qualcomm, and AMD
GPUs. LeftoverLocals impacts the security posture of GPU applications, with
particular significance to LLMs and ML models that run on impacted GPUs. By
recovering local memory, an optimized GPU memory region, we built a PoC where
an attacker can listen into another user's interactive LLM session (e.g.,
llama.cpp) across process or container boundaries.



---

189. **[[2503.13572] VeriContaminated: Assessing LLM-Driven Verilog Coding for Data
  Contamination](https://arxiv.org/pdf/2503.13572.pdf)**

Updated on 2025-04-15

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

190. **[[2406.13748] Learn and Unlearn in Multilingual LLMs](https://arxiv.org/pdf/2406.13748.pdf)**

Updated on 2024-12-16

*Taiming Lu, Philipp Koehn*


  This paper investigates the propagation of harmful information in
multilingual large language models (LLMs) and evaluates the efficacy of various
unlearning methods. We demonstrate that fake information, regardless of the
language it is in, once introduced into these models through training data, can
spread across different languages, compromising the integrity and reliability
of the generated content. Our findings reveal that standard unlearning
techniques, which typically focus on English data, are insufficient in
mitigating the spread of harmful content in multilingual contexts and could
inadvertently reinforce harmful content across languages. We show that only by
addressing harmful responses in both English and the original language of the
harmful data can we effectively eliminate generations for all languages. This
underscores the critical need for comprehensive unlearning strategies that
consider the multilingual nature of modern LLMs to enhance their safety and
reliability across diverse linguistic landscapes.



---

191. **[[2502.11441] Which Retain Set Matters for LLM Unlearning? A Case Study on Entity
  Unlearning](https://arxiv.org/pdf/2502.11441.pdf)**

Updated on 2025-02-18

*Hwan Chang, Hwanhee Lee*


  Large language models (LLMs) risk retaining unauthorized or sensitive
information from their training data, which raises privacy concerns. LLM
unlearning seeks to mitigate these risks by selectively removing specified data
while maintaining overall model performance. However, most existing work focus
on methods to achieve effective forgetting and does not provide a detailed
analysis of the retain set, the portion of training data that is not targeted
for removal. In this paper, we investigate the effects of unlearning on various
subsets of the retain set through a case study on entity unlearning. We
introduce the Syntactically Similar Neighbor Set, a group of queries that share
similar syntactic structures with the data targeted for removal, and show that
this subset suffers the greatest performance drop during unlearning. Moreover,
when used for regularization, this set not only preserves performance on
syntactically similar queries but also delivers comparable or improved results
across other data subsets. Our results highlight that syntactic similarity is a
critical factor, potentially more so than domain or entity relationships, in
achieving effective and practical LLM unlearning.



---

192. **[[2504.06658] A Neuro-inspired Interpretation of Unlearning in Large Language Models
  through Sample-level Unlearning Difficulty](https://arxiv.org/pdf/2504.06658.pdf)**

Updated on 2025-04-10

*Xiaohua Feng, Yuyuan Li, Chengye Wang, Junlin Liu, Li Zhang, Chaochao Chen*


  Driven by privacy protection laws and regulations, unlearning in Large
Language Models (LLMs) is gaining increasing attention. However, current
research often neglects the interpretability of the unlearning process,
particularly concerning sample-level unlearning difficulty. Existing studies
typically assume a uniform unlearning difficulty across samples. This
simplification risks attributing the performance of unlearning algorithms to
sample selection rather than the algorithm's design, potentially steering the
development of LLM unlearning in the wrong direction. Thus, we investigate the
relationship between LLM unlearning and sample characteristics, with a focus on
unlearning difficulty. Drawing inspiration from neuroscience, we propose a
Memory Removal Difficulty ($\mathrm{MRD}$) metric to quantify sample-level
unlearning difficulty. Using $\mathrm{MRD}$, we analyze the characteristics of
hard-to-unlearn versus easy-to-unlearn samples. Furthermore, we propose an
$\mathrm{MRD}$-based weighted sampling method to optimize existing unlearning
algorithms, which prioritizes easily forgettable samples, thereby improving
unlearning efficiency and effectiveness. We validate the proposed metric and
method using public benchmarks and datasets, with results confirming its
effectiveness.



---
