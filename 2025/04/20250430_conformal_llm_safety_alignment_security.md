**[1. [2404.09932] Foundational Challenges in Assuring Alignment and Safety of Large
  Language Models](https://arxiv.org/pdf/2404.09932.pdf)** (2024-09-09)

*Usman Anwar, Abulhair Saparov, Javier Rando, Daniel Paleka, Miles Turpin, Peter Hase, Ekdeep Singh Lubana, Erik Jenner, Stephen Casper, Oliver Sourbut, Benjamin L. Edelman, Zhaowei Zhang, Mario Günther, Anton Korinek, Jose Hernandez-Orallo, Lewis Hammond, Eric Bigelow, Alexander Pan, Lauro Langosco, Tomasz Korbak, Heidi Zhang, Ruiqi Zhong, Seán Ó hÉigeartaigh, Gabriel Recchia, Giulio Corsi, Alan Chan, Markus Anderljung, Lilian Edwards, Aleksandar Petrov, Christian Schroeder de Witt, Sumeet Ramesh Motwan, Yoshua Bengio, Danqi Chen, Philip H. S. Torr, Samuel Albanie, Tegan Maharaj, Jakob Foerster, Florian Tramer, He He, Atoosa Kasirzadeh, Yejin Choi, David Krueger*

  This work identifies 18 foundational challenges in assuring the alignment and
safety of large language models (LLMs). These challenges are organized into
three different categories: scientific understanding of LLMs, development and
deployment methods, and sociotechnical challenges. Based on the identified
challenges, we pose $200+$ concrete research questions.


---

**[2. [2404.17287] When to Trust LLMs: Aligning Confidence with Response Quality](https://arxiv.org/pdf/2404.17287.pdf)** (2024-10-01)

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

**[3. [2504.09420] SaRO: Enhancing LLM Safety through Reasoning-based Alignment](https://arxiv.org/pdf/2504.09420.pdf)** (2025-04-15)

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

**[4. [2411.11543] PSA-VLM: Enhancing Vision-Language Model Safety through Progressive
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

**[5. [2410.09047] Unraveling and Mitigating Safety Alignment Degradation of
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

**[6. [2409.07772] Alignment with Preference Optimization Is All You Need for LLM Safety](https://arxiv.org/pdf/2409.07772.pdf)** (2024-09-13)

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

**[7. [2502.13946] Why Safeguarded Ships Run Aground? Aligned Large Language Models' Safety
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

**[8. [2406.05644] How Alignment and Jailbreak Work: Explain LLM Safety through
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

**[9. [2502.10486] VLM-Guard: Safeguarding Vision-Language Models via Fulfilling Safety
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

**[10. [2411.01696] Conformal Risk Minimization with Variance Reduction](https://arxiv.org/pdf/2411.01696.pdf)** (2025-02-11)

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

**[11. [2410.14676] SudoLM: Learning Access Control of Parametric Knowledge with
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

**[12. [2412.03253] Alignment at Pre-training! Towards Native Alignment for Arabic LLMs](https://arxiv.org/pdf/2412.03253.pdf)** (2024-12-05)

*Juhao Liang, Zhenyang Cai, Jianqing Zhu, Huang Huang, Kewei Zong, Bang An, Mosen Alharthi, Juncai He, Lian Zhang, Haizhou Li, Benyou Wang, Jinchao Xu*

  The alignment of large language models (LLMs) is critical for developing
effective and safe language models. Traditional approaches focus on aligning
models during the instruction tuning or reinforcement learning stages, referred
to in this paper as `post alignment'. We argue that alignment during the
pre-training phase, which we term `native alignment', warrants investigation.
Native alignment aims to prevent unaligned content from the beginning, rather
than relying on post-hoc processing. This approach leverages extensively
aligned pre-training data to enhance the effectiveness and usability of
pre-trained models. Our study specifically explores the application of native
alignment in the context of Arabic LLMs. We conduct comprehensive experiments
and ablation studies to evaluate the impact of native alignment on model
performance and alignment stability. Additionally, we release open-source
Arabic LLMs that demonstrate state-of-the-art performance on various
benchmarks, providing significant benefits to the Arabic LLM community.


---

**[13. [2504.09895] Learning from Reference Answers: Versatile Language Model Alignment
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

**[14. [2410.20290] Fast Best-of-N Decoding via Speculative Rejection](https://arxiv.org/pdf/2410.20290.pdf)** (2024-11-04)

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

**[15. [2310.18333] She had Cobalt Blue Eyes: Prompt Testing to Create Aligned and
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

**[16. [2408.17003] Safety Layers in Aligned Large Language Models: The Key to LLM Security](https://arxiv.org/pdf/2408.17003.pdf)** (2025-04-08)

*Shen Li, Liuyi Yao, Lan Zhang, Yaliang Li*

  Aligned LLMs are secure, capable of recognizing and refusing to answer
malicious questions. However, the role of internal parameters in maintaining
such security is not well understood yet, further these models can be
vulnerable to security degradation when subjected to fine-tuning attacks. To
address these challenges, our work uncovers the mechanism behind security in
aligned LLMs at the parameter level, identifying a small set of contiguous
layers in the middle of the model that are crucial for distinguishing malicious
queries from normal ones, referred to as ``safety layers". We first confirm the
existence of these safety layers by analyzing variations in input vectors
within the model's internal layers. Additionally, we leverage the
over-rejection phenomenon and parameters scaling analysis to precisely locate
the safety layers. Building on these findings, we propose a novel fine-tuning
approach, Safely Partial-Parameter Fine-Tuning (SPPFT), that fixes the gradient
of the safety layers during fine-tuning to address the security degradation.
Our experiments demonstrate that the proposed approach can significantly
preserve LLM security while maintaining performance and reducing computational
resources compared to full fine-tuning.


---

**[17. [2405.13581] Safety Alignment for Vision Language Models](https://arxiv.org/pdf/2405.13581.pdf)** (2024-05-24)

*Zhendong Liu, Yuanbi Nie, Yingshui Tan, Xiangyu Yue, Qiushi Cui, Chongjun Wang, Xiaoyong Zhu, Bo Zheng*

  Benefiting from the powerful capabilities of Large Language Models (LLMs),
pre-trained visual encoder models connected to an LLMs can realize Vision
Language Models (VLMs). However, existing research shows that the visual
modality of VLMs is vulnerable, with attackers easily bypassing LLMs' safety
alignment through visual modality features to launch attacks. To address this
issue, we enhance the existing VLMs' visual modality safety alignment by adding
safety modules, including a safety projector, safety tokens, and a safety head,
through a two-stage training process, effectively improving the model's defense
against risky images. For example, building upon the LLaVA-v1.5 model, we
achieve a safety score of 8.26, surpassing the GPT-4V on the Red Teaming Visual
Language Models (RTVLM) benchmark. Our method boasts ease of use, high
flexibility, and strong controllability, and it enhances safety while having
minimal impact on the model's general performance. Moreover, our alignment
strategy also uncovers some possible risky content within commonly used
open-source multimodal datasets. Our code will be open sourced after the
anonymous review.


---

**[18. [2311.09433] Trojan Activation Attack: Red-Teaming Large Language Models using
  Activation Steering for Safety-Alignment](https://arxiv.org/pdf/2311.09433.pdf)** (2024-08-19)

*Haoran Wang, Kai Shu*

  To ensure AI safety, instruction-tuned Large Language Models (LLMs) are
specifically trained to ensure alignment, which refers to making models behave
in accordance with human intentions. While these models have demonstrated
commendable results on various safety benchmarks, the vulnerability of their
safety alignment has not been extensively studied. This is particularly
troubling given the potential harm that LLMs can inflict. Existing attack
methods on LLMs often rely on poisoned training data or the injection of
malicious prompts. These approaches compromise the stealthiness and
generalizability of the attacks, making them susceptible to detection.
Additionally, these models often demand substantial computational resources for
implementation, making them less practical for real-world applications. In this
work, we study a different attack scenario, called Trojan Activation Attack
(TA^2), which injects trojan steering vectors into the activation layers of
LLMs. These malicious steering vectors can be triggered at inference time to
steer the models toward attacker-desired behaviors by manipulating their
activations. Our experiment results on four primary alignment tasks show that
TA^2 is highly effective and adds little or no overhead to attack efficiency.
Additionally, we discuss potential countermeasures against such activation
attacks.


---

**[19. [2407.06443] Exposing Privacy Gaps: Membership Inference Attack on Preference Data
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

**[20. [2406.14563] Model Merging and Safety Alignment: One Bad Model Spoils the Bunch](https://arxiv.org/pdf/2406.14563.pdf)** (2024-06-21)

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

**[21. [2406.10630] Emerging Safety Attack and Defense in Federated Instruction Tuning of
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

**[22. [2504.05050] Revealing the Intrinsic Ethical Vulnerability of Aligned Large Language
  Models](https://arxiv.org/pdf/2504.05050.pdf)** (2025-04-21)

*Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Shaohui Mei, Lap-Pui Chau*

  Large language models (LLMs) are foundational explorations to artificial
general intelligence, yet their alignment with human values via instruction
tuning and preference learning achieves only superficial compliance. Here, we
demonstrate that harmful knowledge embedded during pretraining persists as
indelible "dark patterns" in LLMs' parametric memory, evading alignment
safeguards and resurfacing under adversarial inducement at distributional
shifts. In this study, we first theoretically analyze the intrinsic ethical
vulnerability of aligned LLMs by proving that current alignment methods yield
only local "safety regions" in the knowledge manifold. In contrast, pretrained
knowledge remains globally connected to harmful concepts via high-likelihood
adversarial trajectories. Building on this theoretical insight, we empirically
validate our findings by employing semantic coherence inducement under
distributional shifts--a method that systematically bypasses alignment
constraints through optimized adversarial prompts. This combined theoretical
and empirical approach achieves a 100% attack success rate across 19 out of 23
state-of-the-art aligned LLMs, including DeepSeek-R1 and LLaMA-3, revealing
their universal vulnerabilities.


---

**[23. [2411.04291] Unfair Alignment: Examining Safety Alignment Across Vision Encoder
  Layers in Vision-Language Models](https://arxiv.org/pdf/2411.04291.pdf)** (2024-11-08)

*Saketh Bachu, Erfan Shayegani, Trishna Chakraborty, Rohit Lal, Arindam Dutta, Chengyu Song, Yue Dong, Nael Abu-Ghazaleh, Amit K. Roy-Chowdhury*

  Vision-language models (VLMs) have improved significantly in multi-modal
tasks, but their more complex architecture makes their safety alignment more
challenging than the alignment of large language models (LLMs). In this paper,
we reveal an unfair distribution of safety across the layers of VLM's vision
encoder, with earlier and middle layers being disproportionately vulnerable to
malicious inputs compared to the more robust final layers. This 'cross-layer'
vulnerability stems from the model's inability to generalize its safety
training from the default architectural settings used during training to unseen
or out-of-distribution scenarios, leaving certain layers exposed. We conduct a
comprehensive analysis by projecting activations from various intermediate
layers and demonstrate that these layers are more likely to generate harmful
outputs when exposed to malicious inputs. Our experiments with LLaVA-1.5 and
Llama 3.2 show discrepancies in attack success rates and toxicity scores across
layers, indicating that current safety alignment strategies focused on a single
default layer are insufficient.


---

**[24. [2501.16534] Targeting Alignment: Extracting Safety Classifiers of Aligned LLMs](https://arxiv.org/pdf/2501.16534.pdf)** (2025-01-29)

*Jean-Charles Noirot Ferrand, Yohan Beugin, Eric Pauley, Ryan Sheatsley, Patrick McDaniel*

  Alignment in large language models (LLMs) is used to enforce guidelines such
as safety. Yet, alignment fails in the face of jailbreak attacks that modify
inputs to induce unsafe outputs. In this paper, we present and evaluate a
method to assess the robustness of LLM alignment. We observe that alignment
embeds a safety classifier in the target model that is responsible for deciding
between refusal and compliance. We seek to extract an approximation of this
classifier, called a surrogate classifier, from the LLM. We develop an
algorithm for identifying candidate classifiers from subsets of the LLM model.
We evaluate the degree to which the candidate classifiers approximate the
model's embedded classifier in benign (F1 score) and adversarial (using
surrogates in a white-box attack) settings. Our evaluation shows that the best
candidates achieve accurate agreement (an F1 score above 80%) using as little
as 20% of the model architecture. Further, we find attacks mounted on the
surrogate models can be transferred with high accuracy. For example, a
surrogate using only 50% of the Llama 2 model achieved an attack success rate
(ASR) of 70%, a substantial improvement over attacking the LLM directly, where
we only observed a 22% ASR. These results show that extracting surrogate
classifiers is a viable (and highly effective) means for modeling (and therein
addressing) the vulnerability of aligned models to jailbreaking attacks.


---

**[25. [2408.15625] CBF-LLM: Safe Control for LLM Alignment](https://arxiv.org/pdf/2408.15625.pdf)** (2024-10-08)

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

**[26. [2410.07471] SEAL: Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection](https://arxiv.org/pdf/2410.07471.pdf)** (2024-10-14)

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

**[27. [2404.12038] Uncovering Safety Risks of Large Language Models through Concept
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

**[28. [2502.13603] Efficient Safety Retrofitting Against Jailbreaking for LLMs](https://arxiv.org/pdf/2502.13603.pdf)** (2025-02-26)

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

**[29. [2407.12344] The Better Angels of Machine Personality: How Personality Relates to LLM
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

**[30. [2502.12485] Safe at the Margins: A General Approach to Safety Alignment in
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

**[31. [2308.05374] Trustworthy LLMs: a Survey and Guideline for Evaluating Large Language
  Models' Alignment](https://arxiv.org/pdf/2308.05374.pdf)** (2024-03-22)

*Yang Liu, Yuanshun Yao, Jean-Francois Ton, Xiaoying Zhang, Ruocheng Guo, Hao Cheng, Yegor Klochkov, Muhammad Faaiz Taufiq, Hang Li*

  Ensuring alignment, which refers to making models behave in accordance with
human intentions [1,2], has become a critical task before deploying large
language models (LLMs) in real-world applications. For instance, OpenAI devoted
six months to iteratively aligning GPT-4 before its release [3]. However, a
major challenge faced by practitioners is the lack of clear guidance on
evaluating whether LLM outputs align with social norms, values, and
regulations. This obstacle hinders systematic iteration and deployment of LLMs.
To address this issue, this paper presents a comprehensive survey of key
dimensions that are crucial to consider when assessing LLM trustworthiness. The
survey covers seven major categories of LLM trustworthiness: reliability,
safety, fairness, resistance to misuse, explainability and reasoning, adherence
to social norms, and robustness. Each major category is further divided into
several sub-categories, resulting in a total of 29 sub-categories.
Additionally, a subset of 8 sub-categories is selected for further
investigation, where corresponding measurement studies are designed and
conducted on several widely-used LLMs. The measurement results indicate that,
in general, more aligned models tend to perform better in terms of overall
trustworthiness. However, the effectiveness of alignment varies across the
different trustworthiness categories considered. This highlights the importance
of conducting more fine-grained analyses, testing, and making continuous
improvements on LLM alignment. By shedding light on these key dimensions of LLM
trustworthiness, this paper aims to provide valuable insights and guidance to
practitioners in the field. Understanding and addressing these concerns will be
crucial in achieving reliable and ethically sound deployment of LLMs in various
applications.


---

**[32. [2502.01116] Picky LLMs and Unreliable RMs: An Empirical Study on Safety Alignment
  after Instruction Tuning](https://arxiv.org/pdf/2502.01116.pdf)** (2025-02-04)

*Guanlin Li, Kangjie Chen, Shangwei Guo, Jie Zhang, Han Qiu, Chao Zhang, Guoyin Wang, Tianwei Zhang, Jiwei Li*

  Large language models (LLMs) have emerged as powerful tools for addressing a
wide range of general inquiries and tasks. Despite this, fine-tuning aligned
LLMs on smaller, domain-specific datasets, critical to adapting them to
specialized tasks, can inadvertently degrade their safety alignment, even when
the datasets are benign. This phenomenon makes models more susceptible to
providing inappropriate responses. In this study, we systematically examine the
factors contributing to safety alignment degradation in benign fine-tuning
scenarios. Our analysis identifies three critical factors affecting aligned
LLMs: answer structure, identity calibration, and role-play. Additionally, we
evaluate the reliability of state-of-the-art reward models (RMs), which are
often used to guide alignment processes. Our findings reveal that these RMs
frequently fail to accurately reflect human preferences regarding safety,
underscoring their limitations in practical applications. By uncovering these
challenges, our work highlights the complexities of maintaining safety
alignment during fine-tuning and offers guidance to help developers balance
utility and safety in LLMs. Datasets and fine-tuning code used in our
experiments can be found in
https://github.com/GuanlinLee/llm_instruction_tuning.


---

**[33. [2407.20999] MoFO: Momentum-Filtered Optimizer for Mitigating Forgetting in LLM
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

**[34. [2402.02207] Safety Fine-Tuning at (Almost) No Cost: A Baseline for Vision Large
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

**[35. [2407.20729] Adapting Safe-for-Work Classifier for Malaysian Language Text: Enhancing
  Alignment in LLM-Ops Framework](https://arxiv.org/pdf/2407.20729.pdf)** (2024-07-31)

*Aisyah Razak, Ariff Nazhan, Kamarul Adha, Wan Adzhar Faiq Adzlan, Mas Aisyah Ahmad, Ammar Azman*

  As large language models (LLMs) become increasingly integrated into
operational workflows (LLM-Ops), there is a pressing need for effective
guardrails to ensure safe and aligned interactions, including the ability to
detect potentially unsafe or inappropriate content across languages. However,
existing safe-for-work classifiers are primarily focused on English text. To
address this gap for the Malaysian language, we present a novel safe-for-work
text classifier tailored specifically for Malaysian language content. By
curating and annotating a first-of-its-kind dataset of Malaysian text spanning
multiple content categories, we trained a classification model capable of
identifying potentially unsafe material using state-of-the-art natural language
processing techniques. This work represents an important step in enabling safer
interactions and content filtering to mitigate potential risks and ensure
responsible deployment of LLMs. To maximize accessibility and promote further
research towards enhancing alignment in LLM-Ops for the Malaysian context, the
model is publicly released at
https://huggingface.co/malaysia-ai/malaysian-sfw-classifier.


---

**[36. [2502.00657] LLM Safety Alignment is Divergence Estimation in Disguise](https://arxiv.org/pdf/2502.00657.pdf)** (2025-02-04)

*Rajdeep Haldar, Ziyi Wang, Qifan Song, Guang Lin, Yue Xing*

  We propose a theoretical framework demonstrating that popular Large Language
Model (LLM) alignment methods, including Reinforcement Learning from Human
Feedback (RLHF) and alternatives, fundamentally function as divergence
estimators between aligned (preferred or safe) and unaligned (less-preferred or
harmful) distributions. This explains the separation phenomenon between safe
and harmful prompts in the model hidden representation after alignment.
Inspired by the theoretical results, we identify that some alignment methods
are better than others in terms of separation and, introduce a new method,
KLDO, and further demonstrate the implication of our theories. We advocate for
compliance-refusal datasets over preference datasets to enhance safety
alignment, supported by both theoretical reasoning and empirical evidence.
Additionally, to quantify safety separation, we leverage a distance metric in
the representation space and statistically validate its efficacy as a
statistical significant indicator of LLM resilience against jailbreak attacks.


---

**[37. [2403.09572] Eyes Closed, Safety On: Protecting Multimodal LLMs via Image-to-Text
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

**[38. [2502.07340] Aligning Large Language Models to Follow Instructions and Hallucinate
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

**[39. [2410.12662] Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models](https://arxiv.org/pdf/2410.12662.pdf)** (2025-03-03)

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

**[40. [2401.11974] Cross-Validation Conformal Risk Control](https://arxiv.org/pdf/2401.11974.pdf)** (2024-05-02)

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

**[41. [2407.07666] A Proposed S.C.O.R.E. Evaluation Framework for Large Language Models :
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

**[42. [2404.04392] Fine-Tuning, Quantization, and LLMs: Navigating Unintended Outcomes](https://arxiv.org/pdf/2404.04392.pdf)** (2024-09-10)

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

**[43. [2410.09893] RMB: Comprehensively Benchmarking Reward Models in LLM Alignment](https://arxiv.org/pdf/2410.09893.pdf)** (2025-04-07)

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

**[44. [2412.11041] Separate the Wheat from the Chaff: A Post-Hoc Approach to Safety
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

**[45. [2401.11206] InferAligner: Inference-Time Alignment for Harmlessness through
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

**[46. [2502.01208] Almost Surely Safe Alignment of Large Language Models at Inference-Time](https://arxiv.org/pdf/2502.01208.pdf)** (2025-02-06)

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

**[47. [2406.11880] Knowledge Return Oriented Prompting (KROP)](https://arxiv.org/pdf/2406.11880.pdf)** (2024-06-19)

*Jason Martin, Kenneth Yeung*

  Many Large Language Models (LLMs) and LLM-powered apps deployed today use
some form of prompt filter or alignment to protect their integrity. However,
these measures aren't foolproof. This paper introduces KROP, a prompt injection
technique capable of obfuscating prompt injection attacks, rendering them
virtually undetectable to most of these security measures.


---

**[48. [2309.14348] Defending Against Alignment-Breaking Attacks via Robustly Aligned LLM](https://arxiv.org/pdf/2309.14348.pdf)** (2024-06-13)

*Bochuan Cao, Yuanpu Cao, Lu Lin, Jinghui Chen*

  Recently, Large Language Models (LLMs) have made significant advancements and
are now widely used across various domains. Unfortunately, there has been a
rising concern that LLMs can be misused to generate harmful or malicious
content. Though a line of research has focused on aligning LLMs with human
values and preventing them from producing inappropriate content, such
alignments are usually vulnerable and can be bypassed by alignment-breaking
attacks via adversarially optimized or handcrafted jailbreaking prompts. In
this work, we introduce a Robustly Aligned LLM (RA-LLM) to defend against
potential alignment-breaking attacks. RA-LLM can be directly constructed upon
an existing aligned LLM with a robust alignment checking function, without
requiring any expensive retraining or fine-tuning process of the original LLM.
Furthermore, we also provide a theoretical analysis for RA-LLM to verify its
effectiveness in defending against alignment-breaking attacks. Through
real-world experiments on open-source large language models, we demonstrate
that RA-LLM can successfully defend against both state-of-the-art adversarial
prompts and popular handcrafted jailbreaking prompts by reducing their attack
success rates from nearly 100% to around 10% or less.


---

**[49. [2409.14038] OAEI-LLM: A Benchmark Dataset for Understanding Large Language Model
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

**[50. [2502.11455] Adversary-Aware DPO: Enhancing Safety Alignment in Vision Language
  Models via Adversarial Training](https://arxiv.org/pdf/2502.11455.pdf)** (2025-02-18)

*Fenghua Weng, Jian Lou, Jun Feng, Minlie Huang, Wenjie Wang*

  Safety alignment is critical in pre-training large language models (LLMs) to
generate responses aligned with human values and refuse harmful queries. Unlike
LLM, the current safety alignment of VLMs is often achieved with post-hoc
safety fine-tuning. However, these methods are less effective to white-box
attacks. To address this, we propose $\textit{Adversary-aware DPO (ADPO)}$, a
novel training framework that explicitly considers adversarial.
$\textit{Adversary-aware DPO (ADPO)}$ integrates adversarial training into DPO
to enhance the safety alignment of VLMs under worst-case adversarial
perturbations. $\textit{ADPO}$ introduces two key components: (1) an
adversarial-trained reference model that generates human-preferred responses
under worst-case perturbations, and (2) an adversarial-aware DPO loss that
generates winner-loser pairs accounting for adversarial distortions. By
combining these innovations, $\textit{ADPO}$ ensures that VLMs remain robust
and reliable even in the presence of sophisticated jailbreak attacks. Extensive
experiments demonstrate that $\textit{ADPO}$ outperforms baselines in the
safety alignment and general utility of VLMs.


---

**[51. [2402.03181] C-RAG: Certified Generation Risks for Retrieval-Augmented Language
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

**[52. [2311.05915] Fake Alignment: Are LLMs Really Aligned Well?](https://arxiv.org/pdf/2311.05915.pdf)** (2024-04-02)

*Yixu Wang, Yan Teng, Kexin Huang, Chengqi Lyu, Songyang Zhang, Wenwei Zhang, Xingjun Ma, Yu-Gang Jiang, Yu Qiao, Yingchun Wang*

  The growing awareness of safety concerns in large language models (LLMs) has
sparked considerable interest in the evaluation of safety. This study
investigates an under-explored issue about the evaluation of LLMs, namely the
substantial discrepancy in performance between multiple-choice questions and
open-ended questions. Inspired by research on jailbreak attack patterns, we
argue this is caused by mismatched generalization. That is, LLM only remembers
the answer style for open-ended safety questions, which makes it unable to
solve other forms of safety tests. We refer to this phenomenon as fake
alignment and construct a comparative benchmark to empirically verify its
existence in LLMs. We introduce a Fake alIgNment Evaluation (FINE) framework
and two novel metrics--Consistency Score (CS) and Consistent Safety Score
(CSS), which jointly assess two complementary forms of evaluation to quantify
fake alignment and obtain corrected performance estimation. Applying FINE to 14
widely-used LLMs reveals several models with purported safety are poorly
aligned in practice. Subsequently, we found that multiple-choice format data
can also be used as high-quality contrast distillation-based fine-tuning data,
which can strongly improve the alignment consistency of LLMs with minimal
fine-tuning overhead. For data and code, see
https://github.com/AIFlames/Fake-Alignment.


---

**[53. [2404.14461] Competition Report: Finding Universal Jailbreak Backdoors in Aligned
  LLMs](https://arxiv.org/pdf/2404.14461.pdf)** (2024-06-07)

*Javier Rando, Francesco Croce, Kryštof Mitka, Stepan Shabalin, Maksym Andriushchenko, Nicolas Flammarion, Florian Tramèr*

  Large language models are aligned to be safe, preventing users from
generating harmful content like misinformation or instructions for illegal
activities. However, previous work has shown that the alignment process is
vulnerable to poisoning attacks. Adversaries can manipulate the safety training
data to inject backdoors that act like a universal sudo command: adding the
backdoor string to any prompt enables harmful responses from models that,
otherwise, behave safely. Our competition, co-located at IEEE SaTML 2024,
challenged participants to find universal backdoors in several large language
models. This report summarizes the key findings and promising ideas for future
research.


---

**[54. [2502.20285] Conformal Tail Risk Control for Large Language Model Alignment](https://arxiv.org/pdf/2502.20285.pdf)** (2025-02-28)

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

**[55. [2301.11664] Automatic Alignment in Higher-Order Probabilistic Programming Languages](https://arxiv.org/pdf/2301.11664.pdf)** (2023-05-05)

*Daniel Lundén, Gizem Çaylak, Fredrik Ronquist, David Broman*

  Probabilistic Programming Languages (PPLs) allow users to encode statistical
inference problems and automatically apply an inference algorithm to solve
them. Popular inference algorithms for PPLs, such as sequential Monte Carlo
(SMC) and Markov chain Monte Carlo (MCMC), are built around checkpoints --
relevant events for the inference algorithm during the execution of a
probabilistic program. Deciding the location of checkpoints is, in current
PPLs, not done optimally. To solve this problem, we present a static analysis
technique that automatically determines checkpoints in programs, relieving PPL
users of this task. The analysis identifies a set of checkpoints that execute
in the same order in every program run -- they are aligned. We formalize
alignment, prove the correctness of the analysis, and implement the analysis as
part of the higher-order functional PPL Miking CorePPL. By utilizing the
alignment analysis, we design two novel inference algorithm variants: aligned
SMC and aligned lightweight MCMC. We show, through real-world experiments, that
they significantly improve inference execution time and accuracy compared to
standard PPL versions of SMC and MCMC.


---

**[56. [2405.13820] Towards Comprehensive Post Safety Alignment of Large Language Models via
  Safety Patching](https://arxiv.org/pdf/2405.13820.pdf)** (2024-12-18)

*Weixiang Zhao, Yulin Hu, Zhuojun Li, Yang Deng, Jiahe Guo, Xingyu Sui, Yanyan Zhao, Bing Qin, Tat-Seng Chua, Ting Liu*

  Safety alignment of large language models (LLMs) has been gaining increasing
attention. However, current safety-aligned LLMs suffer from the fragile and
imbalanced safety mechanisms, which can still be induced to generate unsafe
responses, exhibit over-safety by rejecting safe user inputs, and fail to
preserve general utility after safety alignment. To this end, we propose a
novel post safety alignment (PSA) method to address these inherent and emerging
safety challenges, including safety enhancement, over-safety mitigation, and
utility preservation. In specific, we introduce \textsc{SafePatching}, a novel
framework for comprehensive PSA, where two distinct safety patches are
developed on the harmful data to enhance safety and mitigate over-safety
concerns, and then seamlessly integrated into the target LLM backbone without
compromising its utility. Extensive experiments on four representative aligned
LLMs, including LLaMA-2/3, Gemma and Mistral, show that \textsc{SafePatching}
achieves a more comprehensive PSA than baseline methods, further optimizing the
balance between being helpful and harmless in current aligned LLMs. Also,
\textsc{SafePatching} demonstrates its superiority in continual PSA scenarios.


---

**[57. [2410.10862] Superficial Safety Alignment Hypothesis](https://arxiv.org/pdf/2410.10862.pdf)** (2024-10-16)

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

**[58. [2411.18688] Immune: Improving Safety Against Jailbreaks in Multi-modal LLMs via
  Inference-Time Alignment](https://arxiv.org/pdf/2411.18688.pdf)** (2025-03-21)

*Soumya Suvra Ghosal, Souradip Chakraborty, Vaibhav Singh, Tianrui Guan, Mengdi Wang, Ahmad Beirami, Furong Huang, Alvaro Velasquez, Dinesh Manocha, Amrit Singh Bedi*

  With the widespread deployment of Multimodal Large Language Models (MLLMs)
for visual-reasoning tasks, improving their safety has become crucial. Recent
research indicates that despite training-time safety alignment, these models
remain vulnerable to jailbreak attacks. In this work, we first highlight an
important safety gap to describe that alignment achieved solely through safety
training may be insufficient against jailbreak attacks. To address this
vulnerability, we propose Immune, an inference-time defense framework that
leverages a safe reward model through controlled decoding to defend against
jailbreak attacks. Additionally, we provide a mathematical characterization of
Immune, offering insights on why it improves safety against jailbreaks.
Extensive evaluations on diverse jailbreak benchmarks using recent MLLMs reveal
that Immune effectively enhances model safety while preserving the model's
original capabilities. For instance, against text-based jailbreak attacks on
LLaVA-1.6, Immune reduces the attack success rate by 57.82% and 16.78% compared
to the base MLLM and state-of-the-art defense strategy, respectively.


---

**[59. [2504.12553] ELAB: Extensive LLM Alignment Benchmark in Persian Language](https://arxiv.org/pdf/2504.12553.pdf)** (2025-04-18)

*Zahra Pourbahman, Fatemeh Rajabi, Mohammadhossein Sadeghi, Omid Ghahroodi, Somaye Bakhshaei, Arash Amini, Reza Kazemi, Mahdieh Soleymani Baghshah*

  This paper presents a comprehensive evaluation framework for aligning Persian
Large Language Models (LLMs) with critical ethical dimensions, including
safety, fairness, and social norms. It addresses the gaps in existing LLM
evaluation frameworks by adapting them to Persian linguistic and cultural
contexts. This benchmark creates three types of Persian-language benchmarks:
(i) translated data, (ii) new data generated synthetically, and (iii) new
naturally collected data. We translate Anthropic Red Teaming data, AdvBench,
HarmBench, and DecodingTrust into Persian. Furthermore, we create
ProhibiBench-fa, SafeBench-fa, FairBench-fa, and SocialBench-fa as new datasets
to address harmful and prohibited content in indigenous culture. Moreover, we
collect extensive dataset as GuardBench-fa to consider Persian cultural norms.
By combining these datasets, our work establishes a unified framework for
evaluating Persian LLMs, offering a new approach to culturally grounded
alignment evaluation. A systematic evaluation of Persian LLMs is performed
across the three alignment aspects: safety (avoiding harmful content), fairness
(mitigating biases), and social norms (adhering to culturally accepted
behaviors). We present a publicly available leaderboard that benchmarks Persian
LLMs with respect to safety, fairness, and social norms at:
https://huggingface.co/spaces/MCILAB/LLM_Alignment_Evaluation.


---

**[60. [2501.01765] SaLoRA: Safety-Alignment Preserved Low-Rank Adaptation](https://arxiv.org/pdf/2501.01765.pdf)** (2025-01-06)

*Mingjie Li, Wai Man Si, Michael Backes, Yang Zhang, Yisen Wang*

  As advancements in large language models (LLMs) continue and the demand for
personalized models increases, parameter-efficient fine-tuning (PEFT) methods
(e.g., LoRA) will become essential due to their efficiency in reducing
computation costs. However, recent studies have raised alarming concerns that
LoRA fine-tuning could potentially compromise the safety alignment in LLMs,
posing significant risks for the model owner. In this paper, we first
investigate the underlying mechanism by analyzing the changes in safety
alignment related features before and after fine-tuning. Then, we propose a
fixed safety module calculated by safety data and a task-specific
initialization for trainable parameters in low-rank adaptations, termed
Safety-alignment preserved Low-Rank Adaptation (SaLoRA). Unlike previous LoRA
methods and their variants, SaLoRA enables targeted modifications to LLMs
without disrupting their original alignments. Our experiments show that SaLoRA
outperforms various adapters-based approaches across various evaluation metrics
in different fine-tuning tasks.


---

**[61. [2311.06697] Trusted Source Alignment in Large Language Models](https://arxiv.org/pdf/2311.06697.pdf)** (2023-11-14)

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

**[62. [2401.09796] A Fast, Performant, Secure Distributed Training Framework For Large
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

**[63. [2502.09723] Making Them a Malicious Database: Exploiting Query Code to Jailbreak
  Aligned Large Language Models](https://arxiv.org/pdf/2502.09723.pdf)** (2025-02-21)

*Qingsong Zou, Jingyu Xiao, Qing Li, Zhi Yan, Yuhang Wang, Li Xu, Wenxuan Wang, Kuofeng Gao, Ruoyu Li, Yong Jiang*

  Recent advances in large language models (LLMs) have demonstrated remarkable
potential in the field of natural language processing. Unfortunately, LLMs face
significant security and ethical risks. Although techniques such as safety
alignment are developed for defense, prior researches reveal the possibility of
bypassing such defenses through well-designed jailbreak attacks. In this paper,
we propose QueryAttack, a novel framework to examine the generalizability of
safety alignment. By treating LLMs as knowledge databases, we translate
malicious queries in natural language into structured non-natural query
language to bypass the safety alignment mechanisms of LLMs. We conduct
extensive experiments on mainstream LLMs, and the results show that QueryAttack
not only can achieve high attack success rates (ASRs), but also can jailbreak
various defense methods. Furthermore, we tailor a defense method against
QueryAttack, which can reduce ASR by up to 64% on GPT-4-1106. Our code is
available at https://github.com/horizonsinzqs/QueryAttack.


---

**[64. [2405.16833] Safe LoRA: the Silver Lining of Reducing Safety Risks when Fine-tuning
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

**[65. [2310.10049] FATE-LLM: A Industrial Grade Federated Learning Framework for Large
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

**[66. [2502.13095] Understanding and Rectifying Safety Perception Distortion in VLMs](https://arxiv.org/pdf/2502.13095.pdf)** (2025-02-19)

*Xiaohan Zou, Jian Kang, George Kesidis, Lu Lin*

  Recent studies reveal that vision-language models (VLMs) become more
susceptible to harmful requests and jailbreak attacks after integrating the
vision modality, exhibiting greater vulnerability than their text-only LLM
backbones. To uncover the root cause of this phenomenon, we conduct an in-depth
analysis and identify a key issue: multimodal inputs introduce an
modality-induced activation shift toward a "safer" direction compared to their
text-only counterparts, leading VLMs to systematically overestimate the safety
of harmful inputs. We refer to this issue as safety perception distortion. To
mitigate such distortion, we propose Activation Shift Disentanglement and
Calibration (ShiftDC), a training-free method that decomposes and calibrates
the modality-induced activation shift to reduce the impact of modality on
safety. By isolating and removing the safety-relevant component, ShiftDC
restores the inherent safety alignment of the LLM backbone while preserving the
vision-language capabilities of VLMs. Empirical results demonstrate that
ShiftDC significantly enhances alignment performance on safety benchmarks
without impairing model utility.


---

**[67. [2503.05021] Safety is Not Only About Refusal: Reasoning-Enhanced Fine-tuning for
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

**[68. [2311.07689] MART: Improving LLM Safety with Multi-round Automatic Red-Teaming](https://arxiv.org/pdf/2311.07689.pdf)** (2023-11-15)

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

**[69. [2502.03699] LLM Alignment as Retriever Optimization: An Information Retrieval
  Perspective](https://arxiv.org/pdf/2502.03699.pdf)** (2025-02-07)

*Bowen Jin, Jinsung Yoon, Zhen Qin, Ziqi Wang, Wei Xiong, Yu Meng, Jiawei Han, Sercan O. Arik*

  Large Language Models (LLMs) have revolutionized artificial intelligence with
capabilities in reasoning, coding, and communication, driving innovation across
industries. Their true potential depends on effective alignment to ensure
correct, trustworthy and ethical behavior, addressing challenges like
misinformation, hallucinations, bias and misuse. While existing Reinforcement
Learning (RL)-based alignment methods are notoriously complex, direct
optimization approaches offer a simpler alternative. In this work, we introduce
a novel direct optimization approach for LLM alignment by drawing on
established Information Retrieval (IR) principles. We present a systematic
framework that bridges LLM alignment and IR methodologies, mapping LLM
generation and reward models to IR's retriever-reranker paradigm. Building on
this foundation, we propose LLM Alignment as Retriever Preference Optimization
(LarPO), a new alignment method that enhances overall alignment quality.
Extensive experiments validate LarPO's effectiveness with 38.9 % and 13.7 %
averaged improvement on AlpacaEval2 and MixEval-Hard respectively. Our work
opens new avenues for advancing LLM alignment by integrating IR foundations,
offering a promising direction for future research.


---

**[70. [2410.10343] Locking Down the Finetuned LLMs Safety](https://arxiv.org/pdf/2410.10343.pdf)** (2024-10-15)

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

**[71. [2312.00575] Instruction-tuning Aligns LLMs to the Human Brain](https://arxiv.org/pdf/2312.00575.pdf)** (2024-08-12)

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

**[72. [2411.06493] LProtector: An LLM-driven Vulnerability Detection System](https://arxiv.org/pdf/2411.06493.pdf)** (2024-11-15)

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

**[73. [2502.08922] Self-Consistency of the Internal Reward Models Improves Self-Rewarding
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

**[74. [2406.15279] Safe Inputs but Unsafe Output: Benchmarking Cross-modality Safety
  Alignment of Large Vision-Language Model](https://arxiv.org/pdf/2406.15279.pdf)** (2025-02-18)

*Siyin Wang, Xingsong Ye, Qinyuan Cheng, Junwen Duan, Shimin Li, Jinlan Fu, Xipeng Qiu, Xuanjing Huang*

  As Artificial General Intelligence (AGI) becomes increasingly integrated into
various facets of human life, ensuring the safety and ethical alignment of such
systems is paramount. Previous studies primarily focus on single-modality
threats, which may not suffice given the integrated and complex nature of
cross-modality interactions. We introduce a novel safety alignment challenge
called Safe Inputs but Unsafe Output (SIUO) to evaluate cross-modality safety
alignment. Specifically, it considers cases where single modalities are safe
independently but could potentially lead to unsafe or unethical outputs when
combined. To empirically investigate this problem, we developed the SIUO, a
cross-modality benchmark encompassing 9 critical safety domains, such as
self-harm, illegal activities, and privacy violations. Our findings reveal
substantial safety vulnerabilities in both closed- and open-source LVLMs, such
as GPT-4V and LLaVA, underscoring the inadequacy of current models to reliably
interpret and respond to complex, real-world scenarios.


---

**[75. [2406.11285] Self and Cross-Model Distillation for LLMs: Effective Methods for
  Refusal Pattern Alignment](https://arxiv.org/pdf/2406.11285.pdf)** (2024-12-03)

*Jie Li, Yi Liu, Chongyang Liu, Xiaoning Ren, Ling Shi, Weisong Sun, Yinxing Xue*

  Large Language Models (LLMs) like OpenAI's GPT series, Anthropic's Claude,
and Meta's LLaMa have shown remarkable capabilities in text generation.
However, their susceptibility to toxic prompts presents significant security
challenges. This paper investigates alignment techniques, including Supervised
Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF), to
mitigate these risks. We conduct an empirical study on refusal patterns across
nine LLMs, revealing that models with uniform refusal patterns, such as
Claude3, exhibit higher security. Based on these findings, we propose
self-distilling and cross-model distilling methods to enhance LLM security. Our
results show that these methods significantly improve refusal rates and reduce
unsafe content, with cross-model distilling achieving refusal rates close to
Claude3's 94.51%. These findings underscore the potential of distillation-based
alignment in securing LLMs against toxic prompts.


---

**[76. [2309.01446] Open Sesame! Universal Black Box Jailbreaking of Large Language Models](https://arxiv.org/pdf/2309.01446.pdf)** (2024-08-06)

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

**[77. [2408.00307] ABC Align: Large Language Model Alignment for Safety & Accuracy](https://arxiv.org/pdf/2408.00307.pdf)** (2024-08-02)

*Gareth Seneque, Lap-Hang Ho, Ariel Kuperman, Nafise Erfanian Saeedi, Jeffrey Molendijk*

  Alignment of Large Language Models (LLMs) remains an unsolved problem. Human
preferences are highly distributed and can be captured at multiple levels of
abstraction, from the individual to diverse populations. Organisational
preferences, represented by standards and principles, are defined to mitigate
reputational risk or meet legislative obligations. In this paper, we present
ABC Align, a novel alignment methodology for LLMs that enables integration of
the standards and preferences of a large media organisation into the LLM
itself. We combine a set of data and methods that build on recent breakthroughs
in synthetic data generation, preference optimisation, and post-training model
quantisation. Our unified approach mitigates bias and improves accuracy, while
preserving reasoning capability, as measured against standard benchmarks.


---

**[78. [2410.17875] Understanding Layer Significance in LLM Alignment](https://arxiv.org/pdf/2410.17875.pdf)** (2025-04-09)

*Guangyuan Shi, Zexin Lu, Xiaoyu Dong, Wenlong Zhang, Xuanyu Zhang, Yujie Feng, Xiao-Ming Wu*

  Aligning large language models (LLMs) through supervised fine-tuning is
essential for tailoring them to specific applications. Recent studies suggest
that alignment primarily adjusts a model's presentation style rather than its
foundational knowledge, indicating that only certain components of the model
are significantly impacted. To uncover how alignment affects model behavior at
a granular level, we propose identifying which layers within LLMs are most
critical to the alignment process. Our approach, named ILA, involves learning a
binary mask for the parameter changes in each layer during alignment, as an
indicator of layer significance. Experimental results reveal that, despite
substantial differences in alignment datasets, the important layers of a model
identified by ILA exhibit nearly 90\% overlap, highlighting fundamental
patterns in LLM alignment. The results also indicate that freezing
non-essential layers improves overall model performance, while selectively
tuning the most critical layers significantly enhances fine-tuning efficiency
with minimal performance loss. Finally, we discuss how these findings extend
from LLM alignment to reasoning.


---

**[79. [2412.16686] NILE: Internal Consistency Alignment in Large Language Models](https://arxiv.org/pdf/2412.16686.pdf)** (2024-12-24)

*Minda Hu, Qiyuan Zhang, Yufei Wang, Bowei He, Hongru Wang, Jingyan Zhou, Liangyou Li, Yasheng Wang, Chen Ma, Irwin King*

  As a crucial step to enhance LLMs alignment with human intentions,
Instruction Fine-Tuning (IFT) has a high demand on dataset quality. However,
existing IFT datasets often contain knowledge that is inconsistent with LLMs'
internal knowledge learned from the pre-training phase, which can greatly
affect the efficacy of IFT. To address this issue, we introduce NILE (iNternal
consIstency aLignmEnt) framework, aimed at optimizing IFT datasets to unlock
LLMs' capability further. NILE operates by eliciting target pre-trained LLM's
internal knowledge corresponding to instruction data. The internal knowledge is
leveraged to revise the answer in IFT datasets. Additionally, we propose a
novel Internal Consistency Filtering (ICF) method to filter training samples,
ensuring its high consistency with LLM's internal knowledge. Our experiments
demonstrate that NILE-aligned IFT datasets sharply boost LLM performance across
multiple LLM ability evaluation datasets, achieving up to 66.6% gain on
Arena-Hard and 68.5% on Alpaca-Eval V2. Further analysis confirms that each
component of the NILE}framework contributes to these substantial performance
improvements, and provides compelling evidence that dataset consistency with
pre-trained internal knowledge is pivotal for maximizing LLM potential.


---

**[80. [2411.12882] ProSec: Fortifying Code LLMs with Proactive Security Alignment](https://arxiv.org/pdf/2411.12882.pdf)** (2025-02-12)

*Xiangzhe Xu, Zian Su, Jinyao Guo, Kaiyuan Zhang, Zhenting Wang, Xiangyu Zhang*

  Recent advances in code-specific large language models (LLMs) have greatly
enhanced code generation and refinement capabilities. However, the safety of
code LLMs remains under-explored, posing potential risks as insecure code
generated by these models may introduce vulnerabilities into real-world
systems. Previous work proposes to collect security-focused instruction-tuning
dataset from real-world vulnerabilities. It is constrained by the data sparsity
of vulnerable code, and has limited applicability in the iterative
post-training workflows of modern LLMs. In this paper, we propose ProSec, a
novel proactive security alignment approach designed to align code LLMs with
secure coding practices. ProSec systematically exposes the vulnerabilities in a
code LLM by synthesizing error-inducing coding scenarios from Common Weakness
Enumerations (CWEs), and generates fixes to vulnerable code snippets, allowing
the model to learn secure practices through advanced preference learning
objectives. The scenarios synthesized by ProSec triggers 25 times more
vulnerable code than a normal instruction-tuning dataset, resulting in a
security-focused alignment dataset 7 times larger than the previous work.
Experiments show that models trained with ProSec are 25.2% to 91.4% more secure
compared to previous work without degrading models' utility.


---

**[81. [2404.00486] Dialectical Alignment: Resolving the Tension of 3H and Security Threats
  of LLMs](https://arxiv.org/pdf/2404.00486.pdf)** (2024-04-02)

*Shu Yang, Jiayuan Su, Han Jiang, Mengdi Li, Keyuan Cheng, Muhammad Asif Ali, Lijie Hu, Di Wang*

  With the rise of large language models (LLMs), ensuring they embody the
principles of being helpful, honest, and harmless (3H), known as Human
Alignment, becomes crucial. While existing alignment methods like RLHF, DPO,
etc., effectively fine-tune LLMs to match preferences in the preference
dataset, they often lead LLMs to highly receptive human input and external
evidence, even when this information is poisoned. This leads to a tendency for
LLMs to be Adaptive Chameleons when external evidence conflicts with their
parametric memory. This exacerbates the risk of LLM being attacked by external
poisoned data, which poses a significant security risk to LLM system
applications such as Retrieval-augmented generation (RAG). To address the
challenge, we propose a novel framework: Dialectical Alignment (DA), which (1)
utilizes AI feedback to identify optimal strategies for LLMs to navigate
inter-context conflicts and context-memory conflicts with different external
evidence in context window (i.e., different ratios of poisoned factual
contexts); (2) constructs the SFT dataset as well as the preference dataset
based on the AI feedback and strategies above; (3) uses the above datasets for
LLM alignment to defense poisoned context attack while preserving the
effectiveness of in-context knowledge editing. Our experiments show that the
dialectical alignment model improves poisoned data attack defense by 20 and
does not require any additional prompt engineering or prior declaration of
``you may be attacked`` to the LLMs' context window.


---

**[82. [2311.04155] Black-Box Prompt Optimization: Aligning Large Language Models without
  Model Training](https://arxiv.org/pdf/2311.04155.pdf)** (2024-06-24)

*Jiale Cheng, Xiao Liu, Kehan Zheng, Pei Ke, Hongning Wang, Yuxiao Dong, Jie Tang, Minlie Huang*

  Large language models (LLMs) have shown impressive success in various
applications. However, these models are often not well aligned with human
intents, which calls for additional treatments on them; that is, the alignment
problem. To make LLMs better follow user instructions, existing alignment
methods primarily focus on further training them. However, the extra training
of LLMs is usually expensive in terms of GPU computing; even worse, some LLMs
are not accessible for user-demanded training, such as GPTs. In this work, we
take a different perspective -- Black-Box Prompt Optimization (BPO) -- to
perform alignments. The idea is to optimize user prompts to suit LLMs' input
understanding, so as to best realize users' intents without updating LLMs'
parameters. BPO leverages human preferences to optimize prompts, thus making it
superior to LLM (e.g., ChatGPT) as a prompt engineer. Moreover, BPO is
model-agnostic, and the empirical results demonstrate that the BPO-aligned
ChatGPT yields a 22% increase in the win rate against its original version and
10% for GPT-4. Notably, the BPO-aligned LLMs can outperform the same models
aligned by PPO and DPO, and it also brings additional performance gains when
combining BPO with PPO or DPO. Code and datasets are released at
https://github.com/thu-coai/BPO.


---

**[83. [2411.18948] RevPRAG: Revealing Poisoning Attacks in Retrieval-Augmented Generation
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

**[84. [2412.08201] Model-Editing-Based Jailbreak against Safety-aligned Large Language
  Models](https://arxiv.org/pdf/2412.08201.pdf)** (2024-12-12)

*Yuxi Li, Zhibo Zhang, Kailong Wang, Ling Shi, Haoyu Wang*

  Large Language Models (LLMs) have transformed numerous fields by enabling
advanced natural language interactions but remain susceptible to critical
vulnerabilities, particularly jailbreak attacks. Current jailbreak techniques,
while effective, often depend on input modifications, making them detectable
and limiting their stealth and scalability. This paper presents Targeted Model
Editing (TME), a novel white-box approach that bypasses safety filters by
minimally altering internal model structures while preserving the model's
intended functionalities. TME identifies and removes safety-critical
transformations (SCTs) embedded in model matrices, enabling malicious queries
to bypass restrictions without input modifications. By analyzing distinct
activation patterns between safe and unsafe queries, TME isolates and
approximates SCTs through an optimization process. Implemented in the D-LLM
framework, our method achieves an average Attack Success Rate (ASR) of 84.86%
on four mainstream open-source LLMs, maintaining high performance. Unlike
existing methods, D-LLM eliminates the need for specific triggers or harmful
response collections, offering a stealthier and more effective jailbreak
strategy. This work reveals a covert and robust threat vector in LLM security
and emphasizes the need for stronger safeguards in model safety alignment.


---

**[85. [2411.14487] Ensuring Safety and Trust: Analyzing the Risks of Large Language Models
  in Medicine](https://arxiv.org/pdf/2411.14487.pdf)** (2024-11-25)

*Yifan Yang, Qiao Jin, Robert Leaman, Xiaoyu Liu, Guangzhi Xiong, Maame Sarfo-Gyamfi, Changlin Gong, Santiago Ferrière-Steinert, W. John Wilbur, Xiaojun Li, Jiaxin Yuan, Bang An, Kelvin S. Castro, Francisco Erramuspe Álvarez, Matías Stockle, Aidong Zhang, Furong Huang, Zhiyong Lu*

  The remarkable capabilities of Large Language Models (LLMs) make them
increasingly compelling for adoption in real-world healthcare applications.
However, the risks associated with using LLMs in medical applications have not
been systematically characterized. We propose using five key principles for
safe and trustworthy medical AI: Truthfulness, Resilience, Fairness,
Robustness, and Privacy, along with ten specific aspects. Under this
comprehensive framework, we introduce a novel MedGuard benchmark with 1,000
expert-verified questions. Our evaluation of 11 commonly used LLMs shows that
the current language models, regardless of their safety alignment mechanisms,
generally perform poorly on most of our benchmarks, particularly when compared
to the high performance of human physicians. Despite recent reports indicate
that advanced LLMs like ChatGPT can match or even exceed human performance in
various medical tasks, this study underscores a significant safety gap,
highlighting the crucial need for human oversight and the implementation of AI
safety guardrails.


---

**[86. [2503.20228] TeleLoRA: Teleporting Model-Specific Alignment Across LLMs](https://arxiv.org/pdf/2503.20228.pdf)** (2025-03-27)

*Xiao Lin, Manoj Acharya, Anirban Roy, Susmit Jha*

  Mitigating Trojans in Large Language Models (LLMs) is one of many tasks where
alignment data is LLM specific, as different LLMs have different Trojan
triggers and trigger behaviors to be removed. In this paper, we introduce
TeleLoRA (Teleporting Low-Rank Adaptation), a novel framework that synergizes
model-specific alignment data across multiple LLMs to enable zero-shot Trojan
mitigation on unseen LLMs without alignment data. TeleLoRA learns a unified
generator of LoRA adapter weights by leveraging local activation information
across multiple LLMs. This generator is designed to be permutation symmetric to
generalize across models with different architectures and sizes. We optimize
the model design for memory efficiency, making it feasible to learn with
large-scale LLMs with minimal computational resources. Experiments on LLM
Trojan mitigation benchmarks demonstrate that TeleLoRA effectively reduces
attack success rates while preserving the benign performance of the models.


---

**[87. [2402.18540] Keeping LLMs Aligned After Fine-tuning: The Crucial Role of Prompt
  Templates](https://arxiv.org/pdf/2402.18540.pdf)** (2025-01-20)

*Kaifeng Lyu, Haoyu Zhao, Xinran Gu, Dingli Yu, Anirudh Goyal, Sanjeev Arora*

  Public LLMs such as the Llama 2-Chat underwent alignment training and were
considered safe. Recently Qi et al. [2024] reported that even benign
fine-tuning on seemingly safe datasets can give rise to unsafe behaviors in the
models. The current paper is about methods and best practices to mitigate such
loss of alignment. We focus on the setting where a public model is fine-tuned
before serving users for specific usage, where the model should improve on the
downstream task while maintaining alignment. Through extensive experiments on
several chat models (Meta's Llama 2-Chat, Mistral AI's Mistral 7B Instruct
v0.2, and OpenAI's GPT-3.5 Turbo), this paper uncovers that the prompt
templates used during fine-tuning and inference play a crucial role in
preserving safety alignment, and proposes the ``Pure Tuning, Safe Testing''
(PTST) strategy -- fine-tune models without a safety prompt, but include it at
test time. This seemingly counterintuitive strategy incorporates an intended
distribution shift to encourage alignment preservation. Fine-tuning experiments
on GSM8K, ChatDoctor, and OpenOrca show that PTST significantly reduces the
rise of unsafe behaviors.


---

**[88. [2407.07342] Multilingual Blending: LLM Safety Alignment Evaluation with Language
  Mixture](https://arxiv.org/pdf/2407.07342.pdf)** (2024-07-11)

*Jiayang Song, Yuheng Huang, Zhehua Zhou, Lei Ma*

  As safety remains a crucial concern throughout the development lifecycle of
Large Language Models (LLMs), researchers and industrial practitioners have
increasingly focused on safeguarding and aligning LLM behaviors with human
preferences and ethical standards. LLMs, trained on extensive multilingual
corpora, exhibit powerful generalization abilities across diverse languages and
domains. However, current safety alignment practices predominantly focus on
single-language scenarios, which leaves their effectiveness in complex
multilingual contexts, especially for those complex mixed-language formats,
largely unexplored. In this study, we introduce Multilingual Blending, a
mixed-language query-response scheme designed to evaluate the safety alignment
of various state-of-the-art LLMs (e.g., GPT-4o, GPT-3.5, Llama3) under
sophisticated, multilingual conditions. We further investigate language
patterns such as language availability, morphology, and language family that
could impact the effectiveness of Multilingual Blending in compromising the
safeguards of LLMs. Our experimental results show that, without meticulously
crafted prompt templates, Multilingual Blending significantly amplifies the
detriment of malicious queries, leading to dramatically increased bypass rates
in LLM safety alignment (67.23% on GPT-3.5 and 40.34% on GPT-4o), far exceeding
those of single-language baselines. Moreover, the performance of Multilingual
Blending varies notably based on intrinsic linguistic properties, with
languages of different morphology and from diverse families being more prone to
evading safety alignments. These findings underscore the necessity of
evaluating LLMs and developing corresponding safety alignment strategies in a
complex, multilingual context to align with their superior cross-language
generalization capabilities.


---

**[89. [2502.16691] Toward Responsible Federated Large Language Models: Leveraging a Safety
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

**[90. [2411.08733] Dynamic Rewarding with Prompt Optimization Enables Tuning-free
  Self-Alignment of Language Models](https://arxiv.org/pdf/2411.08733.pdf)** (2024-11-15)

*Somanshu Singla, Zhen Wang, Tianyang Liu, Abdullah Ashfaq, Zhiting Hu, Eric P. Xing*

  Aligning Large Language Models (LLMs) traditionally relies on costly training
and human preference annotations. Self-alignment seeks to reduce these expenses
by enabling models to align themselves. To further lower costs and achieve
alignment without any expensive tuning or annotations, we introduce a new
tuning-free approach for self-alignment, Dynamic Rewarding with Prompt
Optimization (DRPO). Our approach leverages a search-based optimization
framework that allows LLMs to iteratively self-improve and craft the optimal
alignment instructions, all without additional training or human intervention.
The core of DRPO is a dynamic rewarding mechanism, which identifies and
rectifies model-specific alignment weaknesses, allowing LLMs to adapt
efficiently to diverse alignment challenges. Empirical evaluations on eight
recent LLMs, both open- and closed-sourced, demonstrate that DRPO significantly
enhances alignment performance, with base models outperforming their
SFT/RLHF-tuned counterparts. Moreover, the prompts automatically optimized by
DRPO surpass those curated by human experts, further validating the
effectiveness of our approach. Our findings highlight the great potential of
current LLMs to achieve adaptive self-alignment through inference-time
optimization, complementing tuning-based alignment methods.


---

**[91. [2304.04521] GL-MCM: Global and Local Maximum Concept Matching for Zero-Shot
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

**[92. [2405.09055] A safety realignment framework via subspace-oriented model fusion for
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

**[93. [2501.18532] Differentially Private Steering for Large Language Model Alignment](https://arxiv.org/pdf/2501.18532.pdf)** (2025-03-21)

*Anmol Goel, Yaxi Hu, Iryna Gurevych, Amartya Sanyal*

  Aligning Large Language Models (LLMs) with human values and away from
undesirable behaviors (such as hallucination) has become increasingly
important. Recently, steering LLMs towards a desired behavior via activation
editing has emerged as an effective method to mitigate harmful generations at
inference-time. Activation editing modifies LLM representations by preserving
information from positive demonstrations (e.g., truthful) and minimising
information from negative demonstrations (e.g., hallucinations). When these
demonstrations come from a private dataset, the aligned LLM may leak private
information contained in those private samples. In this work, we present the
first study of aligning LLM behavior with private datasets. Our work proposes
the Private Steering for LLM Alignment (PSA) algorithm to edit LLM activations
with differential privacy (DP) guarantees. We conduct extensive experiments on
seven different benchmarks with open-source LLMs of different sizes (0.5B to
7B) and model families (LlaMa, Qwen, Mistral and Gemma). Our results show that
PSA achieves DP guarantees for LLM alignment with minimal loss in performance,
including alignment metrics, open-ended text generation quality, and
general-purpose reasoning. We also develop the first Membership Inference
Attack (MIA) for evaluating and auditing the empirical privacy for the problem
of LLM steering via activation editing. Our experiments support the theoretical
guarantees by showing improved guarantees for our PSA algorithm compared to
several existing non-private techniques.


---

**[94. [2502.00669] Safety Alignment Depth in Large Language Models: A Markov Chain
  Perspective](https://arxiv.org/pdf/2502.00669.pdf)** (2025-02-04)

*Ching-Chia Kao, Chia-Mu Yu, Chun-Shien Lu, Chu-Song Chen*

  Large Language Models (LLMs) are increasingly adopted in high-stakes
scenarios, yet their safety mechanisms often remain fragile. Simple jailbreak
prompts or even benign fine-tuning can bypass these protocols, underscoring the
need to understand where and how they fail. Recent findings suggest that
vulnerabilities emerge when alignment is confined to only the initial output
tokens. Unfortunately, even with the introduction of deep safety alignment,
determining the optimal safety depth remains an unresolved challenge. By
leveraging the equivalence between autoregressive language models and Markov
chains, this paper offers the first theoretical result on how to identify the
ideal depth for safety alignment, and demonstrates how permutation-based data
augmentation can tighten these bounds. Crucially, we reveal a fundamental
interaction between alignment depth and ensemble width-indicating that broader
ensembles can compensate for shallower alignments. These insights provide a
theoretical foundation for designing more robust, scalable safety strategies
that complement existing alignment approaches, opening new avenues for research
into safer, more reliable LLMs.


---

**[95. [2410.18210] Towards Understanding the Fragility of Multilingual LLMs against
  Fine-Tuning Attacks](https://arxiv.org/pdf/2410.18210.pdf)** (2025-03-03)

*Samuele Poppi, Zheng-Xin Yong, Yifei He, Bobbie Chern, Han Zhao, Aobo Yang, Jianfeng Chi*

  Recent advancements in Large Language Models (LLMs) have sparked widespread
concerns about their safety. Recent work demonstrates that safety alignment of
LLMs can be easily removed by fine-tuning with a few adversarially chosen
instruction-following examples, i.e., fine-tuning attacks. We take a further
step to understand fine-tuning attacks in multilingual LLMs. We first discover
cross-lingual generalization of fine-tuning attacks: using a few adversarially
chosen instruction-following examples in one language, multilingual LLMs can
also be easily compromised (e.g., multilingual LLMs fail to refuse harmful
prompts in other languages). Motivated by this finding, we hypothesize that
safety-related information is language-agnostic and propose a new method termed
Safety Information Localization (SIL) to identify the safety-related
information in the model parameter space. Through SIL, we validate this
hypothesis and find that only changing 20% of weight parameters in fine-tuning
attacks can break safety alignment across all languages. Furthermore, we
provide evidence to the alternative pathways hypothesis for why freezing
safety-related parameters does not prevent fine-tuning attacks, and we
demonstrate that our attack vector can still jailbreak LLMs adapted to new
languages.


---

**[96. [2503.00555] Safety Tax: Safety Alignment Makes Your Large Reasoning Models Less
  Reasonable](https://arxiv.org/pdf/2503.00555.pdf)** (2025-03-04)

*Tiansheng Huang, Sihao Hu, Fatih Ilhan, Selim Furkan Tekin, Zachary Yahn, Yichang Xu, Ling Liu*

  Safety alignment is an important procedure before the official deployment of
a Large Language Model (LLM). While safety alignment has been extensively
studied for LLM, there is still a large research gap for Large Reasoning Models
(LRMs) that equip with improved reasoning capability. We in this paper
systematically examine a simplified pipeline for producing safety aligned LRMs.
With our evaluation of various LRMs, we deliver two main findings: i) Safety
alignment can be done upon the LRM to restore its safety capability. ii) Safety
alignment leads to a degradation of the reasoning capability of LRMs. The two
findings show that there exists a trade-off between reasoning and safety
capability with the sequential LRM production pipeline. The discovered
trade-off, which we name Safety Tax, should shed light on future endeavors of
safety research on LRMs. As a by-product, we curate a dataset called
DirectRefusal, which might serve as an alternative dataset for safety
alignment. Our source code is available at
https://github.com/git-disl/Safety-Tax.


---

**[97. [2402.09283] Attacks, Defenses and Evaluations for LLM Conversation Safety: A Survey](https://arxiv.org/pdf/2402.09283.pdf)** (2024-03-28)

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

**[98. [2504.02725] ERPO: Advancing Safety Alignment via Ex-Ante Reasoning Preference
  Optimization](https://arxiv.org/pdf/2504.02725.pdf)** (2025-04-04)

*Kehua Feng, Keyan Ding, Jing Yu, Menghan Li, Yuhao Wang, Tong Xu, Xinda Wang, Qiang Zhang, Huajun Chen*

  Recent advancements in large language models (LLMs) have accelerated progress
toward artificial general intelligence, yet their potential to generate harmful
content poses critical safety challenges. Existing alignment methods often
struggle to cover diverse safety scenarios and remain vulnerable to adversarial
attacks. In this work, we propose Ex-Ante Reasoning Preference Optimization
(ERPO), a novel safety alignment framework that equips LLMs with explicit
preemptive reasoning through Chain-of-Thought and provides clear evidence for
safety judgments by embedding predefined safety rules. Specifically, our
approach consists of three stages: first, equipping the model with Ex-Ante
reasoning through supervised fine-tuning (SFT) using a constructed reasoning
module; second, enhancing safety, usefulness, and efficiency via Direct
Preference Optimization (DPO); and third, mitigating inference latency with a
length-controlled iterative preference optimization strategy. Experiments on
multiple open-source LLMs demonstrate that ERPO significantly enhances safety
performance while maintaining response efficiency.


---

**[99. [2308.12587] Grounded Entity-Landmark Adaptive Pre-training for Vision-and-Language
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

**[100. [2108.04926] First Order Locally Orderless Registration](https://arxiv.org/pdf/2108.04926.pdf)** (2021-08-12)

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

**[101. [2409.19134] Confidential Prompting: Protecting User Prompts from Cloud LLM Providers](https://arxiv.org/pdf/2409.19134.pdf)** (2025-03-05)

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

**[102. [2502.11533] Be Cautious When Merging Unfamiliar LLMs: A Phishing Model Capable of
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

**[103. [2312.17535] Olapa-MCoT: Enhancing the Chinese Mathematical Reasoning Capability of
  LLMs](https://arxiv.org/pdf/2312.17535.pdf)** (2024-01-01)

*Shaojie Zhu, Zhaobin Wang, Chengxiang Zhuo, Hui Lu, Bo Hu, Zang Li*

  CoT (Chain-of-Thought) is a way to solve reasoning problems for LLMs .
Recently, many researches appear for improving the CoT capability of LLMs. In
this work, we also proposed Olapa-MCoT, which is a LLMs based on llama2-13B PLM
for finetuning and alignment learning. During the alignment training, we
proposed the SimRRHF algorithm and Incorrect Data Relearning and mainly focused
on optimizing the Chinese mathematical reasoning ability of Olapa-MCoT. The
experiment achieved significant results, with the accuracy of Chinese
mathematical reasoning up to 50%, 36% rise compared to llama2-13B. In addition,
the accuracy of English reasoning ability also increased by nearly 4%.


---

**[104. [2406.12168] BPO: Staying Close to the Behavior LLM Creates Better Online LLM
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

**[105. [2310.06387] Jailbreak and Guard Aligned Language Models with Only Few In-Context
  Demonstrations](https://arxiv.org/pdf/2310.06387.pdf)** (2024-05-28)

*Zeming Wei, Yifei Wang, Ang Li, Yichuan Mo, Yisen Wang*

  Large Language Models (LLMs) have shown remarkable success in various tasks,
yet their safety and the risk of generating harmful content remain pressing
concerns. In this paper, we delve into the potential of In-Context Learning
(ICL) to modulate the alignment of LLMs. Specifically, we propose the
In-Context Attack (ICA) which employs harmful demonstrations to subvert LLMs,
and the In-Context Defense (ICD) which bolsters model resilience through
examples that demonstrate refusal to produce harmful responses. We offer
theoretical insights to elucidate how a limited set of in-context
demonstrations can pivotally influence the safety alignment of LLMs. Through
extensive experiments, we demonstrate the efficacy of ICA and ICD in
respectively elevating and mitigating the success rates of jailbreaking
prompts. Our findings illuminate the profound influence of ICL on LLM behavior,
opening new avenues for improving the safety of LLMs.


---

**[106. [2503.00187] Steering Dialogue Dynamics for Robustness against Multi-turn
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

**[107. [2412.06483] SafeWorld: Geo-Diverse Safety Alignment](https://arxiv.org/pdf/2412.06483.pdf)** (2024-12-10)

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

**[108. [2504.09757] Alleviating the Fear of Losing Alignment in LLM Fine-tuning](https://arxiv.org/pdf/2504.09757.pdf)** (2025-04-15)

*Kang Yang, Guanhong Tao, Xun Chen, Jun Xu*

  Large language models (LLMs) have demonstrated revolutionary capabilities in
understanding complex contexts and performing a wide range of tasks. However,
LLMs can also answer questions that are unethical or harmful, raising concerns
about their applications. To regulate LLMs' responses to such questions, a
training strategy called \textit{alignment} can help. Yet, alignment can be
unexpectedly compromised when fine-tuning an LLM for downstream tasks. This
paper focuses on recovering the alignment lost during fine-tuning.
  We observe that there are two distinct directions inherent in an aligned LLM:
the \textit{aligned direction} and the \textit{harmful direction}. An LLM is
inclined to answer questions in the aligned direction while refusing queries in
the harmful direction. Therefore, we propose to recover the harmful direction
of the fine-tuned model that has been compromised. Specifically, we restore a
small subset of the fine-tuned model's weight parameters from the original
aligned model using gradient descent. We also introduce a rollback mechanism to
avoid aggressive recovery and maintain downstream task performance. Our
evaluation on 125 fine-tuned LLMs demonstrates that our method can reduce their
harmful rate (percentage of answering harmful questions) from 33.25\% to
1.74\%, without sacrificing task performance much. In contrast, the existing
methods either only reduce the harmful rate to a limited extent or
significantly impact the normal functionality. Our code is available at
https://github.com/kangyangWHU/LLMAlignment


---

**[109. [2408.01460] LocalValueBench: A Collaboratively Built and Extensible Benchmark for
  Evaluating Localized Value Alignment and Ethical Safety in Large Language
  Models](https://arxiv.org/pdf/2408.01460.pdf)** (2024-08-06)

*Gwenyth Isobel Meadows, Nicholas Wai Long Lau, Eva Adelina Susanto, Chi Lok Yu, Aditya Paul*

  The proliferation of large language models (LLMs) requires robust evaluation
of their alignment with local values and ethical standards, especially as
existing benchmarks often reflect the cultural, legal, and ideological values
of their creators. \textsc{LocalValueBench}, introduced in this paper, is an
extensible benchmark designed to assess LLMs' adherence to Australian values,
and provides a framework for regulators worldwide to develop their own LLM
benchmarks for local value alignment. Employing a novel typology for ethical
reasoning and an interrogation approach, we curated comprehensive questions and
utilized prompt engineering strategies to probe LLMs' value alignment. Our
evaluation criteria quantified deviations from local values, ensuring a
rigorous assessment process. Comparative analysis of three commercial LLMs by
USA vendors revealed significant insights into their effectiveness and
limitations, demonstrating the critical importance of value alignment. This
study offers valuable tools and methodologies for regulators to create tailored
benchmarks, highlighting avenues for future research to enhance ethical AI
development.


---

**[110. [2412.15265] Chinese SafetyQA: A Safety Short-form Factuality Benchmark for Large
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

**[111. [2310.03693] Fine-tuning Aligned Language Models Compromises Safety, Even When Users
  Do Not Intend To!](https://arxiv.org/pdf/2310.03693.pdf)** (2023-10-06)

*Xiangyu Qi, Yi Zeng, Tinghao Xie, Pin-Yu Chen, Ruoxi Jia, Prateek Mittal, Peter Henderson*

  Optimizing large language models (LLMs) for downstream use cases often
involves the customization of pre-trained LLMs through further fine-tuning.
Meta's open release of Llama models and OpenAI's APIs for fine-tuning GPT-3.5
Turbo on custom datasets also encourage this practice. But, what are the safety
costs associated with such custom fine-tuning? We note that while existing
safety alignment infrastructures can restrict harmful behaviors of LLMs at
inference time, they do not cover safety risks when fine-tuning privileges are
extended to end-users. Our red teaming studies find that the safety alignment
of LLMs can be compromised by fine-tuning with only a few adversarially
designed training examples. For instance, we jailbreak GPT-3.5 Turbo's safety
guardrails by fine-tuning it on only 10 such examples at a cost of less than
$0.20 via OpenAI's APIs, making the model responsive to nearly any harmful
instructions. Disconcertingly, our research also reveals that, even without
malicious intent, simply fine-tuning with benign and commonly used datasets can
also inadvertently degrade the safety alignment of LLMs, though to a lesser
extent. These findings suggest that fine-tuning aligned LLMs introduces new
safety risks that current safety infrastructures fall short of addressing --
even if a model's initial safety alignment is impeccable, it is not necessarily
to be maintained after custom fine-tuning. We outline and critically analyze
potential mitigations and advocate for further research efforts toward
reinforcing safety protocols for the custom fine-tuning of aligned LLMs.


---

**[112. [2502.06884] Learning Conformal Abstention Policies for Adaptive Risk Management in
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

**[113. [2403.08200] Prototyping and Experimental Results for Environment-Aware Millimeter
  Wave Beam Alignment via Channel Knowledge Map](https://arxiv.org/pdf/2403.08200.pdf)** (2024-03-14)

*Zhuoyin Dai, Di Wu, Zhenjun Dong, Kun Li, Dingyang Ding, Sihan Wang, Yong Zeng*

  Channel knowledge map (CKM), which aims to directly reflect the intrinsic
channel properties of the local wireless environment, is a novel technique for
achieving environmentaware communication. In this paper, to alleviate the large
training overhead in millimeter wave (mmWave) beam alignment, an
environment-aware and training-free beam alignment prototype is established
based on a typical CKM, termed beam index map (BIM). To this end, a general CKM
construction method is first presented, and an indoor BIM is constructed
offline to learn the candidate transmit and receive beam index pairs for each
grid in the experimental area. Furthermore, based on the location information
of the receiver (or the dynamic obstacles) from the ultra-wide band (UWB)
positioning system, the established BIM is used to achieve training-free beam
alignment by directly providing the beam indexes for the transmitter and
receiver. Three typical scenarios are considered in the experiment, including
quasi-static environment with line-of-sight (LoS) link, quasistatic environment
without LoS link and dynamic environment. Besides, the receiver orientation
measured from the gyroscope is also used to help CKM predict more accurate beam
indexes. The experiment results show that compared with the benchmark
location-based beam alignment strategy, the CKM-based beam alignment strategy
can achieve much higher received power, which is close to that achieved by
exhaustive beam search, but with significantly reduced training overhead.


---

**[114. [2312.06924] Safety Alignment in NLP Tasks: Weakly Aligned Summarization as an
  In-Context Attack](https://arxiv.org/pdf/2312.06924.pdf)** (2024-06-10)

*Yu Fu, Yufei Li, Wen Xiao, Cong Liu, Yue Dong*

  Recent developments in balancing the usefulness and safety of Large Language
Models (LLMs) have raised a critical question: Are mainstream NLP tasks
adequately aligned with safety consideration? Our study, focusing on
safety-sensitive documents obtained through adversarial attacks, reveals
significant disparities in the safety alignment of various NLP tasks. For
instance, LLMs can effectively summarize malicious long documents but often
refuse to translate them. This discrepancy highlights a previously unidentified
vulnerability: attacks exploiting tasks with weaker safety alignment, like
summarization, can potentially compromise the integrity of tasks traditionally
deemed more robust, such as translation and question-answering (QA). Moreover,
the concurrent use of multiple NLP tasks with lesser safety alignment increases
the risk of LLMs inadvertently processing harmful content. We demonstrate these
vulnerabilities in various safety-aligned LLMs, particularly Llama2 models,
Gemini and GPT-4, indicating an urgent need for strengthening safety alignments
across a broad spectrum of NLP tasks.


---

**[115. [2501.18632] Towards Safe AI Clinicians: A Comprehensive Study on Large Language
  Model Jailbreaking in Healthcare](https://arxiv.org/pdf/2501.18632.pdf)** (2025-03-05)

*Hang Zhang, Qian Lou, Yanshan Wang*

  Large language models (LLMs) are increasingly utilized in healthcare
applications. However, their deployment in clinical practice raises significant
safety concerns, including the potential spread of harmful information. This
study systematically assesses the vulnerabilities of seven LLMs to three
advanced black-box jailbreaking techniques within medical contexts. To quantify
the effectiveness of these techniques, we propose an automated and
domain-adapted agentic evaluation pipeline. Experiment results indicate that
leading commercial and open-source LLMs are highly vulnerable to medical
jailbreaking attacks. To bolster model safety and reliability, we further
investigate the effectiveness of Continual Fine-Tuning (CFT) in defending
against medical adversarial attacks. Our findings underscore the necessity for
evolving attack methods evaluation, domain-specific safety alignment, and LLM
safety-utility balancing. This research offers actionable insights for
advancing the safety and reliability of AI clinicians, contributing to ethical
and effective AI deployment in healthcare.


---

**[116. [2406.17923] PAFT: A Parallel Training Paradigm for Effective LLM Fine-Tuning](https://arxiv.org/pdf/2406.17923.pdf)** (2024-06-27)

*Claire  Shiva Kumar Pentyala, Claire  Zhichao Wang, Claire  Bin Bi, Claire  Kiran Ramnath, Claire  Xiang-Bo Mao, Claire  Regunathan Radhakrishnan, Claire  Sitaram Asur, Claire   Na, Cheng*

  Large language models (LLMs) have shown remarkable abilities in diverse
natural language processing (NLP) tasks. The LLMs generally undergo supervised
fine-tuning (SFT) followed by preference alignment to be usable in downstream
applications. However, this sequential training pipeline leads to alignment tax
that degrades the LLM performance.
  This paper introduces PAFT, a new PArallel training paradigm for effective
LLM Fine-Tuning, which independently performs SFT and preference alignment
(e.g., DPO and ORPO, etc.) with the same pre-trained model on respective
datasets. The model produced by SFT and the model from preference alignment are
then merged into a final model by parameter fusing for use in downstream
applications. This work reveals important findings that preference alignment
like DPO naturally results in a sparse model while SFT leads to a natural dense
model which needs to be sparsified for effective model merging. This paper
introduces an effective interference resolution which reduces the redundancy by
sparsifying the delta parameters. The LLM resulted from the new training
paradigm achieved Rank #1 on the HuggingFace Open LLM Leaderboard.
Comprehensive evaluation shows the effectiveness of the parallel training
paradigm.


---

**[117. [2410.13903] CoreGuard: Safeguarding Foundational Capabilities of LLMs Against Model
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

**[118. [2503.03710] Improving LLM Safety Alignment with Dual-Objective Optimization](https://arxiv.org/pdf/2503.03710.pdf)** (2025-03-06)

*Xuandong Zhao, Will Cai, Tianneng Shi, David Huang, Licong Lin, Song Mei, Dawn Song*

  Existing training-time safety alignment techniques for large language models
(LLMs) remain vulnerable to jailbreak attacks. Direct preference optimization
(DPO), a widely deployed alignment method, exhibits limitations in both
experimental and theoretical contexts as its loss function proves suboptimal
for refusal learning. Through gradient-based analysis, we identify these
shortcomings and propose an improved safety alignment that disentangles DPO
objectives into two components: (1) robust refusal training, which encourages
refusal even when partial unsafe generations are produced, and (2) targeted
unlearning of harmful knowledge. This approach significantly increases LLM
robustness against a wide range of jailbreak attacks, including prefilling,
suffix, and multi-turn attacks across both in-distribution and
out-of-distribution scenarios. Furthermore, we introduce a method to emphasize
critical refusal tokens by incorporating a reward-based token-level weighting
mechanism for refusal learning, which further improves the robustness against
adversarial exploits. Our research also suggests that robustness to jailbreak
attacks is correlated with token distribution shifts in the training process
and internal representations of refusal and harmful tokens, offering valuable
directions for future research in LLM safety alignment. The code is available
at https://github.com/wicai24/DOOR-Alignment


---

**[119. [2402.01706] MULTIVERSE: Exposing Large Language Model Alignment Problems in Diverse
  Worlds](https://arxiv.org/pdf/2402.01706.pdf)** (2024-02-06)

*Xiaolong Jin, Zhuo Zhang, Xiangyu Zhang*

  Large Language Model (LLM) alignment aims to ensure that LLM outputs match
with human values. Researchers have demonstrated the severity of alignment
problems with a large spectrum of jailbreak techniques that can induce LLMs to
produce malicious content during conversations. Finding the corresponding
jailbreaking prompts usually requires substantial human intelligence or
computation resources. In this paper, we report that LLMs have different levels
of alignment in various contexts. As such, by systematically constructing many
contexts, called worlds, leveraging a Domain Specific Language describing
possible worlds (e.g., time, location, characters, actions and languages) and
the corresponding compiler, we can cost-effectively expose latent alignment
issues. Given the low cost of our method, we are able to conduct a large scale
study regarding LLM alignment issues in different worlds. Our results show that
our method outperforms the-state-of-the-art jailbreaking techniques on both
effectiveness and efficiency. In addition, our results indicate that existing
LLMs are extremely vulnerable to nesting worlds and programming language
worlds. They imply that existing alignment training focuses on the real-world
and is lacking in various (virtual) worlds where LLMs can be exploited.


---

**[120. [2503.09925] PluralLLM: Pluralistic Alignment in LLMs via Federated Learning](https://arxiv.org/pdf/2503.09925.pdf)** (2025-03-14)

*Mahmoud Srewa, Tianyu Zhao, Salma Elmalaki*

  Ensuring Large Language Models (LLMs) align with diverse human preferences
while preserving privacy and fairness remains a challenge. Existing methods,
such as Reinforcement Learning from Human Feedback (RLHF), rely on centralized
data collection, making them computationally expensive and privacy-invasive. We
introduce PluralLLM a federated learning-based approach that enables multiple
user groups to collaboratively train a transformer-based preference predictor
without sharing sensitive data, which can also serve as a reward model for
aligning LLMs. Our method leverages Federated Averaging (FedAvg) to aggregate
preference updates efficiently, achieving 46% faster convergence, a 4%
improvement in alignment scores, and nearly the same group fairness measure as
in centralized training. Evaluated on a Q/A preference alignment task,
PluralLLM demonstrates that federated preference learning offers a scalable and
privacy-preserving alternative for aligning LLMs with diverse human values.


---

**[121. [2310.04451] AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language
  Models](https://arxiv.org/pdf/2310.04451.pdf)** (2024-03-22)

*Xiaogeng Liu, Nan Xu, Muhao Chen, Chaowei Xiao*

  The aligned Large Language Models (LLMs) are powerful language understanding
and decision-making tools that are created through extensive alignment with
human feedback. However, these large models remain susceptible to jailbreak
attacks, where adversaries manipulate prompts to elicit malicious outputs that
should not be given by aligned LLMs. Investigating jailbreak prompts can lead
us to delve into the limitations of LLMs and further guide us to secure them.
Unfortunately, existing jailbreak techniques suffer from either (1) scalability
issues, where attacks heavily rely on manual crafting of prompts, or (2)
stealthiness problems, as attacks depend on token-based algorithms to generate
prompts that are often semantically meaningless, making them susceptible to
detection through basic perplexity testing. In light of these challenges, we
intend to answer this question: Can we develop an approach that can
automatically generate stealthy jailbreak prompts? In this paper, we introduce
AutoDAN, a novel jailbreak attack against aligned LLMs. AutoDAN can
automatically generate stealthy jailbreak prompts by the carefully designed
hierarchical genetic algorithm. Extensive evaluations demonstrate that AutoDAN
not only automates the process while preserving semantic meaningfulness, but
also demonstrates superior attack strength in cross-model transferability, and
cross-sample universality compared with the baseline. Moreover, we also compare
AutoDAN with perplexity-based defense methods and show that AutoDAN can bypass
them effectively.


---

**[122. [2504.03174] Multi-lingual Multi-turn Automated Red Teaming for LLMs](https://arxiv.org/pdf/2504.03174.pdf)** (2025-04-07)

*Abhishek Singhania, Christophe Dupuy, Shivam Mangale, Amani Namboori*

  Language Model Models (LLMs) have improved dramatically in the past few
years, increasing their adoption and the scope of their capabilities over time.
A significant amount of work is dedicated to ``model alignment'', i.e.,
preventing LLMs to generate unsafe responses when deployed into customer-facing
applications. One popular method to evaluate safety risks is
\textit{red-teaming}, where agents attempt to bypass alignment by crafting
elaborate prompts that trigger unsafe responses from a model. Standard
human-driven red-teaming is costly, time-consuming and rarely covers all the
recent features (e.g., multi-lingual, multi-modal aspects), while proposed
automation methods only cover a small subset of LLMs capabilities (i.e.,
English or single-turn). We present Multi-lingual Multi-turn Automated Red
Teaming (\textbf{MM-ART}), a method to fully automate conversational,
multi-lingual red-teaming operations and quickly identify prompts leading to
unsafe responses. Through extensive experiments on different languages, we show
the studied LLMs are on average 71\% more vulnerable after a 5-turn
conversation in English than after the initial turn. For conversations in
non-English languages, models display up to 195\% more safety vulnerabilities
than the standard single-turn English approach, confirming the need for
automated red-teaming methods matching LLMs capabilities.


---

**[123. [2312.12321] Bypassing the Safety Training of Open-Source LLMs with Priming Attacks](https://arxiv.org/pdf/2312.12321.pdf)** (2024-05-20)

*Jason Vega, Isha Chaudhary, Changming Xu, Gagandeep Singh*

  With the recent surge in popularity of LLMs has come an ever-increasing need
for LLM safety training. In this paper, we investigate the fragility of SOTA
open-source LLMs under simple, optimization-free attacks we refer to as
$\textit{priming attacks}$, which are easy to execute and effectively bypass
alignment from safety training. Our proposed attack improves the Attack Success
Rate on Harmful Behaviors, as measured by Llama Guard, by up to $3.3\times$
compared to baselines. Source code and data are available at
https://github.com/uiuc-focal-lab/llm-priming-attacks.


---

**[124. [2501.10915] LegalGuardian: A Privacy-Preserving Framework for Secure Integration of
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

**[125. [2311.06154] A Last-Level Defense for Application Integrity and Confidentiality](https://arxiv.org/pdf/2311.06154.pdf)** (2023-11-13)

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

**[126. [2412.12192] No Free Lunch for Defending Against Prefilling Attack by In-Context
  Learning](https://arxiv.org/pdf/2412.12192.pdf)** (2024-12-18)

*Zhiyu Xue, Guangliang Liu, Bocheng Chen, Kristen Marie Johnson, Ramtin Pedarsani*

  The security of Large Language Models (LLMs) has become an important research
topic since the emergence of ChatGPT. Though there have been various effective
methods to defend against jailbreak attacks, prefilling attacks remain an
unsolved and popular threat against open-sourced LLMs. In-Context Learning
(ICL) offers a computationally efficient defense against various jailbreak
attacks, yet no effective ICL methods have been developed to counter prefilling
attacks. In this paper, we: (1) show that ICL can effectively defend against
prefilling jailbreak attacks by employing adversative sentence structures
within demonstrations; (2) characterize the effectiveness of this defense
through the lens of model size, number of demonstrations, over-defense,
integration with other jailbreak attacks, and the presence of safety alignment.
Given the experimental results and our analysis, we conclude that there is no
free lunch for defending against prefilling jailbreak attacks with ICL. On the
one hand, current safety alignment methods fail to mitigate prefilling
jailbreak attacks, but adversative structures within ICL demonstrations provide
robust defense across various model sizes and complex jailbreak attacks. On the
other hand, LLMs exhibit similar over-defensiveness when utilizing ICL
demonstrations with adversative structures, and this behavior appears to be
independent of model size.


---

**[127. [2403.08319] Knowledge Conflicts for LLMs: A Survey](https://arxiv.org/pdf/2403.08319.pdf)** (2024-06-25)

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

**[128. [2503.21598] Prompt, Divide, and Conquer: Bypassing Large Language Model Safety
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

**[129. [2402.11753] ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs](https://arxiv.org/pdf/2402.11753.pdf)** (2024-06-10)

*Fengqing Jiang, Zhangchen Xu, Luyao Niu, Zhen Xiang, Bhaskar Ramasubramanian, Bo Li, Radha Poovendran*

  Safety is critical to the usage of large language models (LLMs). Multiple
techniques such as data filtering and supervised fine-tuning have been
developed to strengthen LLM safety. However, currently known techniques presume
that corpora used for safety alignment of LLMs are solely interpreted by
semantics. This assumption, however, does not hold in real-world applications,
which leads to severe vulnerabilities in LLMs. For example, users of forums
often use ASCII art, a form of text-based art, to convey image information. In
this paper, we propose a novel ASCII art-based jailbreak attack and introduce a
comprehensive benchmark Vision-in-Text Challenge (ViTC) to evaluate the
capabilities of LLMs in recognizing prompts that cannot be solely interpreted
by semantics. We show that five SOTA LLMs (GPT-3.5, GPT-4, Gemini, Claude, and
Llama2) struggle to recognize prompts provided in the form of ASCII art. Based
on this observation, we develop the jailbreak attack ArtPrompt, which leverages
the poor performance of LLMs in recognizing ASCII art to bypass safety measures
and elicit undesired behaviors from LLMs. ArtPrompt only requires black-box
access to the victim LLMs, making it a practical attack. We evaluate ArtPrompt
on five SOTA LLMs, and show that ArtPrompt can effectively and efficiently
induce undesired behaviors from all five LLMs. Our code is available at
https://github.com/uw-nsl/ArtPrompt.


---

**[130. [2502.08657] Refining Positive and Toxic Samples for Dual Safety Self-Alignment of
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

**[131. [2410.12298] Pyramid-Driven Alignment: Pyramid Principle Guided Integration of Large
  Language Models and Knowledge Graphs](https://arxiv.org/pdf/2410.12298.pdf)** (2024-10-18)

*Lei Sun, Xinchen Wang, Youdi Li*

  Large Language Models (LLMs) possess impressive reasoning abilities but are
prone to generating incorrect information, often referred to as hallucinations.
While incorporating external Knowledge Graphs (KGs) can partially mitigate this
issue, existing methods primarily treat KGs as static knowledge repositories,
overlooking the critical disparity between KG and LLM knowledge, and failing to
fully exploit the reasoning capabilities inherent in KGs. To address these
limitations, we propose Pyramid-Driven Alignment (PDA), a novel framework for
seamlessly integrating LLMs with KGs. PDA utilizes Pyramid Principle analysis
to construct a hierarchical pyramid structure. This structure is designed to
reflect the input question and generate more validated deductive knowledge,
thereby enhancing the alignment of LLMs and KGs and ensuring more cohesive
integration. Furthermore, PDA employs a recursive mechanism to harness the
underlying reasoning abilities of KGs, resulting in more accurate knowledge
retrieval for question-answering tasks. Our experimental results reveal a
substantial performance advantage of PDA over state-of-the-art baselines, with
improvements reaching 26.70% and 26.78%.


---

**[132. [2405.08619] ALMol: Aligned Language-Molecule Translation LLMs through Offline
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

**[133. [2502.09674] The Hidden Dimensions of LLM Alignment: A Multi-Dimensional Safety
  Analysis](https://arxiv.org/pdf/2502.09674.pdf)** (2025-02-19)

*Wenbo Pan, Zhichao Liu, Qiguang Chen, Xiangyang Zhou, Haining Yu, Xiaohua Jia*

  Large Language Models' safety-aligned behaviors, such as refusing harmful
queries, can be represented by linear directions in activation space. Previous
research modeled safety behavior with a single direction, limiting mechanistic
understanding to an isolated safety feature. In this work, we discover that
safety-aligned behavior is jointly controlled by multi-dimensional directions.
Namely, we study the vector space of representation shifts during safety
fine-tuning on Llama 3 8B for refusing jailbreaks. By studying orthogonal
directions in the space, we first find that a dominant direction governs the
model's refusal behavior, while multiple smaller directions represent distinct
and interpretable features like hypothetical narrative and role-playing. We
then measure how different directions promote or suppress the dominant
direction, showing the important role of secondary directions in shaping the
model's refusal representation. Finally, we demonstrate that removing certain
trigger tokens in harmful queries can mitigate these directions to bypass the
learned safety capability, providing new insights on understanding safety
alignment vulnerability from a multi-dimensional perspective. Code and
artifacts are available at https://github.com/BMPixel/safety-residual-space.


---

**[134. [2504.09466] AdaSteer: Your Aligned LLM is Inherently an Adaptive Jailbreak Defender](https://arxiv.org/pdf/2504.09466.pdf)** (2025-04-15)

*Weixiang Zhao, Jiahe Guo, Yulin Hu, Yang Deng, An Zhang, Xingyu Sui, Xinyang Han, Yanyan Zhao, Bing Qin, Tat-Seng Chua, Ting Liu*

  Despite extensive efforts in safety alignment, large language models (LLMs)
remain vulnerable to jailbreak attacks. Activation steering offers a
training-free defense method but relies on fixed steering coefficients,
resulting in suboptimal protection and increased false rejections of benign
inputs. To address this, we propose AdaSteer, an adaptive activation steering
method that dynamically adjusts model behavior based on input characteristics.
We identify two key properties: Rejection Law (R-Law), which shows that
stronger steering is needed for jailbreak inputs opposing the rejection
direction, and Harmfulness Law (H-Law), which differentiates adversarial and
benign inputs. AdaSteer steers input representations along both the Rejection
Direction (RD) and Harmfulness Direction (HD), with adaptive coefficients
learned via logistic regression, ensuring robust jailbreak defense while
preserving benign input handling. Experiments on LLaMA-3.1, Gemma-2, and
Qwen2.5 show that AdaSteer outperforms baseline methods across multiple
jailbreak attacks with minimal impact on utility. Our results highlight the
potential of interpretable model internals for real-time, flexible safety
enforcement in LLMs.


---

**[135. [2310.09639] DPZero: Private Fine-Tuning of Language Models without Backpropagation](https://arxiv.org/pdf/2310.09639.pdf)** (2024-06-07)

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

**[136. [2405.06237] Risks of Practicing Large Language Models in Smart Grid: Threat Modeling
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

**[137. [2409.15623] Safe Guard: an LLM-agent for Real-time Voice-based Hate Speech Detection
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

**[138. [2501.16378] Internal Activation Revision: Safeguarding Vision Language Models
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

**[139. [2504.04151] STEP: Staged Parameter-Efficient Pre-training for Large Language Models](https://arxiv.org/pdf/2504.04151.pdf)** (2025-04-08)

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

**[140. [2306.07096] Global and Local Semantic Completion Learning for Vision-Language
  Pre-training](https://arxiv.org/pdf/2306.07096.pdf)** (2023-12-07)

*Rong-Cheng Tu, Yatai Ji, Jie Jiang, Weijie Kong, Chengfei Cai, Wenzhe Zhao, Hongfa Wang, Yujiu Yang, Wei Liu*

  Cross-modal alignment plays a crucial role in vision-language pre-training
(VLP) models, enabling them to capture meaningful associations across different
modalities. For this purpose, numerous masked modeling tasks have been proposed
for VLP to further promote cross-modal interactions. The core idea of previous
masked modeling tasks is to focus on reconstructing the masked tokens based on
visible context for learning local-local alignment. However, most of them pay
little attention to the global semantic features generated for the masked data,
resulting in a limited cross-modal alignment ability of global representations
to local features of the other modality. Therefore, in this paper, we propose a
novel Global and Local Semantic Completion Learning (GLSCL) task to facilitate
global-local alignment and local-local alignment simultaneously. Specifically,
the GLSCL task complements the missing semantics of masked data and recovers
global and local features by cross-modal interactions. Our GLSCL consists of
masked global semantic completion (MGSC) and masked local token completion
(MLTC). MGSC promotes learning more representative global features, which have
a great impact on the performance of downstream tasks, while MLTC reconstructs
modal-fusion local tokens, further enhancing accurate comprehension of
multimodal data. To evaluate the proposed approaches on cross-modal alignment,
we develop a validation benchmark called ALIGN-BENCH. Moreover, we present a
flexible vision encoder, enabling our model to simultaneously perform
image-text and video-text multimodal tasks. Experimental results show that our
proposed method obtains state-of-the-art performance on various vision-language
benchmarks, such as visual question answering, image-text retrieval, and
video-text retrieval.


---

**[141. [2402.12343] Emulated Disalignment: Safety Alignment for Large Language Models May
  Backfire!](https://arxiv.org/pdf/2402.12343.pdf)** (2024-06-07)

*Zhanhui Zhou, Jie Liu, Zhichen Dong, Jiaheng Liu, Chao Yang, Wanli Ouyang, Yu Qiao*

  Large language models (LLMs) undergo safety alignment to ensure safe
conversations with humans. However, this paper introduces a training-free
attack method capable of reversing safety alignment, converting the outcomes of
stronger alignment into greater potential for harm by accessing only LLM output
token distributions. Specifically, our method achieves this reversal by
contrasting the output token distribution of a safety-aligned language model
(e.g., Llama-2-chat) against its pre-trained version (e.g., Llama-2), so that
the token predictions are shifted towards the opposite direction of safety
alignment. We name this method emulated disalignment (ED) because sampling from
this contrastive distribution provably emulates the result of fine-tuning to
minimize a safety reward. Our experiments with ED across three evaluation
datasets and four model families (Llama-1, Llama-2, Mistral, and Alpaca) show
that ED doubles the harmfulness of pre-trained models and outperforms strong
baselines, achieving the highest harmful rates in 43 out of 48 evaluation
subsets by a large margin. Eventually, given ED's reliance on language model
output token distributions, which particularly compromises open-source models,
our findings highlight the need to reassess the open accessibility of language
models, even if they have been safety-aligned. Code is available at
https://github.com/ZHZisZZ/emulated-disalignment.


---

**[142. [2410.15334] Modality-Fair Preference Optimization for Trustworthy MLLM Alignment](https://arxiv.org/pdf/2410.15334.pdf)** (2024-10-22)

*Songtao Jiang, Yan Zhang, Ruizhe Chen, Yeying Jin, Zuozhu Liu*

  Direct Preference Optimization (DPO) is effective for aligning large language
models (LLMs), but when applied to multimodal models (MLLMs), it often favors
text over image information, leading to unreliable outputs and visual
hallucinations. To address this, we propose Modality-Fair Preference
Optimization (MFPO) to balance text and image preferences. First, we found that
the lack of image-related rewards in preference data biases optimization toward
text, so we created automated, fine-grained image preference data to correct
this. Then, we designed a learning objective to ensure the model captures both
text and image preferences while maintaining high-quality outputs. Finally, we
use a multi-stage alignment approach to stabilize training and improve learning
across both modalities. Extensive experiments demonstrate that MFPO
significantly enhances MLLM trustworthiness. On models like LLaVA-v1.5 (7B,
13B), our approach reduces hallucinations substantially. On the 7B model, MFPO
outperforms GPT-4V and achieves a nearly 40\% improvement over previous methods
on Object HalBench, as well as achieving state-of-the-art performance on both
Object HalBench and AMBER when combined with the latest LLaVA-v1.6. Code will
be released.


---

**[143. [2408.10668] Probing the Safety Response Boundary of Large Language Models via Unsafe
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

**[144. [2502.11681] RIDE: Enhancing Large Language Model Alignment through Restyled
  In-Context Learning Demonstration Exemplars](https://arxiv.org/pdf/2502.11681.pdf)** (2025-03-06)

*Yuncheng Hua, Lizhen Qu, Zhuang Li, Hao Xue, Flora D. Salim, Gholamreza Haffari*

  Alignment tuning is crucial for ensuring large language models (LLMs) behave
ethically and helpfully. Current alignment approaches require high-quality
annotations and significant training resources. This paper proposes a low-cost,
tuning-free method using in-context learning (ICL) to enhance LLM alignment.
Through an analysis of high-quality ICL demos, we identified style as a key
factor influencing LLM alignment capabilities and explicitly restyled ICL
exemplars based on this stylistic framework. Additionally, we combined the
restyled demos to achieve a balance between the two conflicting aspects of LLM
alignment--factuality and safety. We packaged the restyled examples as prompts
to trigger few-shot learning, improving LLM alignment. Compared to the best
baseline approach, with an average score of 5.00 as the maximum, our method
achieves a maximum 0.10 increase on the Alpaca task (from 4.50 to 4.60), a 0.22
enhancement on the Just-eval benchmark (from 4.34 to 4.56), and a maximum
improvement of 0.32 (from 3.53 to 3.85) on the MT-Bench dataset. We release the
code and data at https://github.com/AnonymousCode-ComputerScience/RIDE.


---

**[145. [2408.11491] SCANS: Mitigating the Exaggerated Safety for LLMs via Safety-Conscious
  Activation Steering](https://arxiv.org/pdf/2408.11491.pdf)** (2024-12-18)

*Zouying Cao, Yifei Yang, Hai Zhao*

  Safety alignment is indispensable for Large Language Models (LLMs) to defend
threats from malicious instructions. However, recent researches reveal
safety-aligned LLMs prone to reject benign queries due to the exaggerated
safety issue, limiting their helpfulness. In this paper, we propose a
Safety-Conscious Activation Steering (SCANS) method to mitigate the exaggerated
safety concerns in aligned LLMs. First, SCANS extracts the refusal steering
vectors within the activation space and utilizes vocabulary projection to
anchor some specific safety-critical layers which influence model refusal
behavior. Second, by tracking the hidden state transition, SCANS identifies the
steering direction and steers the model behavior accordingly, achieving a
balance between exaggerated safety and adequate safety. Experiments show that
SCANS achieves new state-of-the-art performance on XSTest and OKTest
benchmarks, without impairing their defense capability against harmful queries
and maintaining almost unchanged model capability.


---

**[146. [2403.09972] Think Twice Before Trusting: Self-Detection for Large Language Models
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

**[147. [2409.19091] System-Level Defense against Indirect Prompt Injection Attacks: An
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

**[148. [2411.09689] LLM Hallucination Reasoning with Zero-shot Knowledge Test](https://arxiv.org/pdf/2411.09689.pdf)** (2024-11-15)

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

**[149. [2412.10423] Look Before You Leap: Enhancing Attention and Vigilance Regarding
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

**[150. [2410.20142] Mask-based Membership Inference Attacks for Retrieval-Augmented
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

**[151. [2404.11338] LLMs for Cyber Security: New Opportunities](https://arxiv.org/pdf/2404.11338.pdf)** (2024-04-18)

*Dinil Mon Divakaran, Sai Teja Peddinti*

  Large language models (LLMs) are a class of powerful and versatile models
that are beneficial to many industries. With the emergence of LLMs, we take a
fresh look at cyber security, specifically exploring and summarizing the
potential of LLMs in addressing challenging problems in the security and safety
domains.


---

**[152. [2503.03146] PriFFT: Privacy-preserving Federated Fine-tuning of Large Language
  Models via Function Secret Sharing](https://arxiv.org/pdf/2503.03146.pdf)** (2025-03-06)

*Zhichao You, Xuewen Dong, Ke Cheng, Xutong Mu, Jiaxuan Fu, Shiyang Ma, Qiang Qu, Yulong Shen*

  Fine-tuning large language models (LLMs) raises privacy concerns due to the
risk of exposing sensitive training data. Federated learning (FL) mitigates
this risk by keeping training samples on local devices, but recent studies show
that adversaries can still infer private information from model updates in FL.
Additionally, LLM parameters are typically shared publicly during federated
fine-tuning, while developers are often reluctant to disclose these parameters,
posing further security challenges. Inspired by the above problems, we propose
PriFFT, a privacy-preserving federated fine-tuning mechanism, to protect both
the model updates and parameters. In PriFFT, clients and the server share model
inputs and parameters by secret sharing, performing secure fine-tuning on
shared values without accessing plaintext data. Due to considerable LLM
parameters, privacy-preserving federated fine-tuning invokes complex secure
calculations and requires substantial communication and computation resources.
To optimize the efficiency of privacy-preserving federated fine-tuning of LLMs,
we introduce function secret-sharing protocols for various operations,
including reciprocal calculation, tensor products, natural exponentiation,
softmax, hyperbolic tangent, and dropout. The proposed protocols achieve up to
4.02X speed improvement and reduce 7.19X communication overhead compared to the
implementation based on existing secret sharing methods. Besides, PriFFT
achieves a 2.23X speed improvement and reduces 4.08X communication overhead in
privacy-preserving fine-tuning without accuracy drop compared to the existing
secret sharing methods.


---

**[153. [2406.13940] AutoCAP: Towards Automatic Cross-lingual Alignment Planning for
  Zero-shot Chain-of-Thought](https://arxiv.org/pdf/2406.13940.pdf)** (2024-06-21)

*Yongheng Zhang, Qiguang Chen, Min Li, Wanxiang Che, Libo Qin*

  Cross-lingual chain-of-thought can effectively complete reasoning tasks
across languages, which gains increasing attention. Recently, dominant
approaches in the literature improve cross-lingual alignment capabilities by
integrating reasoning knowledge from different languages. Despite achieving
excellent performance, current methods still have two main challenges: (1)
Manual language specification: They still highly rely on manually selecting the
languages to integrate, severely affecting their generalizability; (2) Static
weight allocation: Current methods simply integrate all languages equally. In
fact, different language reasoning paths should have different weights to
achieve better complementation and integration. Motivated by this, we introduce
an Automatic Cross-lingual Alignment Planning (AutoCAP) for zero-shot
chain-of-thought to address the above challenges. The core of AutoCAP consists
of two components: (1) Automatic Language Selection Prompting to guide LLMs to
select appropriate languages and (2) Automatic Weight Allocation Prompting to
automatically allocate alignment weight scores to each reasoning path.
Extensive experiments on several benchmarks reveal that AutoCAP achieves
state-of-the-art performance, surpassing previous methods that required manual
effort.


---

**[154. [2406.08754] StructuralSleight: Automated Jailbreak Attacks on Large Language Models
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

**[155. [2411.07021] Invar-RAG: Invariant LLM-aligned Retrieval for Better Generation](https://arxiv.org/pdf/2411.07021.pdf)** (2024-11-19)

*Ziwei Liu, Liang Zhang, Qian Li, Jianghua Wu, Guangxu Zhu*

  Retrieval-augmented generation (RAG) has shown impressive capability in
providing reliable answer predictions and addressing hallucination problems. A
typical RAG implementation uses powerful retrieval models to extract external
information and large language models (LLMs) to generate answers. In contrast,
recent LLM-based retrieval has gained attention for its substantial
improvements in information retrieval (IR) due to the LLMs' semantic
understanding capability. However, directly applying LLM to RAG systems
presents challenges. This may cause feature locality problems as massive
parametric knowledge can hinder effective usage of global information across
the corpus; for example, an LLM-based retriever often inputs document summaries
instead of full documents. Moreover, various pre-trained tasks in LLMs
introduce variance, further weakening performance as a retriever.
  To address these issues, we propose a novel two-stage fine-tuning
architecture called Invar-RAG. In the retrieval stage, an LLM-based retriever
is constructed by integrating LoRA-based representation learning to tackle
feature locality issues. To enhance retrieval performance, we develop two
patterns (invariant and variant patterns) and an invariance loss to reduce LLM
variance. In the generation stage, a refined fine-tuning method is employed to
improve LLM accuracy in generating answers based on retrieved information.
Experimental results show that Invar-RAG significantly outperforms existing
baselines across three open-domain question answering (ODQA) datasets. Code is
available in the Supplementary Material for reproducibility.


---

**[156. [2410.18469] Iterative Self-Tuning LLMs for Enhanced Jailbreaking Capabilities](https://arxiv.org/pdf/2410.18469.pdf)** (2025-03-13)

*Chung-En Sun, Xiaodong Liu, Weiwei Yang, Tsui-Wei Weng, Hao Cheng, Aidan San, Michel Galley, Jianfeng Gao*

  Recent research has shown that Large Language Models (LLMs) are vulnerable to
automated jailbreak attacks, where adversarial suffixes crafted by algorithms
appended to harmful queries bypass safety alignment and trigger unintended
responses. Current methods for generating these suffixes are computationally
expensive and have low Attack Success Rates (ASR), especially against
well-aligned models like Llama2 and Llama3. To overcome these limitations, we
introduce ADV-LLM, an iterative self-tuning process that crafts adversarial
LLMs with enhanced jailbreak ability. Our framework significantly reduces the
computational cost of generating adversarial suffixes while achieving nearly
100\% ASR on various open-source LLMs. Moreover, it exhibits strong attack
transferability to closed-source models, achieving 99\% ASR on GPT-3.5 and 49\%
ASR on GPT-4, despite being optimized solely on Llama3. Beyond improving
jailbreak ability, ADV-LLM provides valuable insights for future safety
alignment research through its ability to generate large datasets for studying
LLM safety.


---

**[157. [2411.02957] Embedding Safety into RL: A New Take on Trust Region Methods](https://arxiv.org/pdf/2411.02957.pdf)** (2025-02-05)

*Nikola Milosevic, Johannes Müller, Nico Scherf*

  Reinforcement Learning (RL) agents can solve diverse tasks but often exhibit
unsafe behavior. Constrained Markov Decision Processes (CMDPs) address this by
enforcing safety constraints, yet existing methods either sacrifice reward
maximization or allow unsafe training. We introduce Constrained Trust Region
Policy Optimization (C-TRPO), which reshapes the policy space geometry to
ensure trust regions contain only safe policies, guaranteeing constraint
satisfaction throughout training. We analyze its theoretical properties and
connections to TRPO, Natural Policy Gradient (NPG), and Constrained Policy
Optimization (CPO). Experiments show that C-TRPO reduces constraint violations
while maintaining competitive returns.


---

**[158. [2410.09102] Instructional Segment Embedding: Improving LLM Safety with Instruction
  Hierarchy](https://arxiv.org/pdf/2410.09102.pdf)** (2025-03-04)

*Tong Wu, Shujian Zhang, Kaiqiang Song, Silei Xu, Sanqiang Zhao, Ravi Agrawal, Sathish Reddy Indurthi, Chong Xiang, Prateek Mittal, Wenxuan Zhou*

  Large Language Models (LLMs) are susceptible to security and safety threats,
such as prompt injection, prompt extraction, and harmful requests. One major
cause of these vulnerabilities is the lack of an instruction hierarchy. Modern
LLM architectures treat all inputs equally, failing to distinguish between and
prioritize various types of instructions, such as system messages, user
prompts, and data. As a result, lower-priority user prompts may override more
critical system instructions, including safety protocols. Existing approaches
to achieving instruction hierarchy, such as delimiters and instruction-based
training, do not address this issue at the architectural level. We introduce
the Instructional Segment Embedding (ISE) technique, inspired by BERT, to
modern large language models, which embeds instruction priority information
directly into the model. This approach enables models to explicitly
differentiate and prioritize various instruction types, significantly improving
safety against malicious prompts that attempt to override priority rules. Our
experiments on the Structured Query and Instruction Hierarchy benchmarks
demonstrate an average robust accuracy increase of up to 15.75% and 18.68%,
respectively. Furthermore, we observe an improvement in instruction-following
capability of up to 4.1% evaluated on AlpacaEval. Overall, our approach offers
a promising direction for enhancing the safety and effectiveness of LLM
architectures.


---

**[159. [2503.21813] OAEI-LLM-T: A TBox Benchmark Dataset for Understanding LLM
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

**[160. [2401.10768] Knowledge Verification to Nip Hallucination in the Bud](https://arxiv.org/pdf/2401.10768.pdf)** (2024-09-24)

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

**[161. [2412.15035] LLMs Lost in Translation: M-ALERT uncovers Cross-Linguistic Safety Gaps](https://arxiv.org/pdf/2412.15035.pdf)** (2025-04-02)

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

**[162. [2410.21438] UFT: Unifying Fine-Tuning of SFT and RLHF/DPO/UNA through a Generalized
  Implicit Reward Function](https://arxiv.org/pdf/2410.21438.pdf)** (2025-04-08)

*Zhichao Wang, Bin Bi, Zixu Zhu, Xiangbo Mao, Jun Wang, Shiyu Wang*

  By pretraining on trillions of tokens, an LLM gains the capability of text
generation. However, to enhance its utility and reduce potential harm, SFT and
alignment are applied sequentially to the pretrained model. Due to the
differing nature and objective functions of SFT and alignment, catastrophic
forgetting has become a significant issue. To address this, we introduce
Unified Fine-Tuning (UFT), which integrates SFT and alignment into a single
training stage using the same objective and loss functions through an implicit
reward function. Our experimental results demonstrate that UFT outperforms SFT
on instruction-tuning data alone. Moreover, when combining instruction-tuning
data with alignment data, UFT effectively prevents catastrophic forgetting
across these two stages and shows a clear advantage over sequentially applying
SFT and alignment. This is evident in the significant improvements observed in
the \textbf{ifeval} task for instruction-following and the \textbf{truthful-qa}
task for factuality. The proposed general fine-tuning framework UFT establishes
an effective and efficient pretraining-UFT paradigm for LLM training.


---

**[163. [2503.17239] SafeMERGE: Preserving Safety Alignment in Fine-Tuned Large Language
  Models via Selective Layer-Wise Model Merging](https://arxiv.org/pdf/2503.17239.pdf)** (2025-03-24)

*Aladin Djuhera, Swanand Ravindra Kadhe, Farhan Ahmed, Syed Zawad, Holger Boche*

  Fine-tuning large language models (LLMs) on downstream tasks can
inadvertently erode their safety alignment, even for benign fine-tuning
datasets. We address this challenge by proposing SafeMERGE, a post-fine-tuning
framework that preserves safety while maintaining task utility. It achieves
this by selectively merging fine-tuned and safety-aligned model layers only
when those deviate from safe behavior, measured by a cosine similarity
criterion. We evaluate SafeMERGE against other fine-tuning- and
post-fine-tuning-stage approaches for Llama-2-7B-Chat and Qwen-2-7B-Instruct
models on GSM8K and PubMedQA tasks while exploring different merging
strategies. We find that SafeMERGE consistently reduces harmful outputs
compared to other baselines without significantly sacrificing performance,
sometimes even enhancing it. The results suggest that our selective,
subspace-guided, and per-layer merging method provides an effective safeguard
against the inadvertent loss of safety in fine-tuned LLMs while outperforming
simpler post-fine-tuning-stage defenses.


---

**[164. [2406.11801] Safety Arithmetic: A Framework for Test-time Safety Alignment of
  Language Models by Steering Parameters and Activations](https://arxiv.org/pdf/2406.11801.pdf)** (2024-10-29)

*Rima Hazra, Sayan Layek, Somnath Banerjee, Soujanya Poria*

  Ensuring the safe alignment of large language models (LLMs) with human values
is critical as they become integral to applications like translation and
question answering. Current alignment methods struggle with dynamic user
intentions and complex objectives, making models vulnerable to generating
harmful content. We propose Safety Arithmetic, a training-free framework
enhancing LLM safety across different scenarios: Base models, Supervised
fine-tuned models (SFT), and Edited models. Safety Arithmetic involves Harm
Direction Removal to avoid harmful content and Safety Alignment to promote safe
responses. Additionally, we present NoIntentEdit, a dataset highlighting edit
instances that could compromise model safety if used unintentionally. Our
experiments show that Safety Arithmetic significantly improves safety measures,
reduces over-safety, and maintains model utility, outperforming existing
methods in ensuring safe content generation.


---

**[165. [2306.11507] TrustGPT: A Benchmark for Trustworthy and Responsible Large Language
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

**[166. [2501.12895] Test-Time Preference Optimization: On-the-Fly Alignment via Iterative
  Textual Feedback](https://arxiv.org/pdf/2501.12895.pdf)** (2025-01-23)

*Yafu Li, Xuyang Hu, Xiaoye Qu, Linjie Li, Yu Cheng*

  Large language models (LLMs) demonstrate impressive performance but lack the
flexibility to adapt to human preferences quickly without retraining. In this
work, we introduce Test-time Preference Optimization (TPO), a framework that
aligns LLM outputs with human preferences during inference, removing the need
to update model parameters. Rather than relying on purely numerical rewards,
TPO translates reward signals into textual critiques and uses them as textual
rewards to iteratively refine its response. Evaluations on benchmarks covering
instruction following, preference alignment, safety, and mathematics reveal
that TPO progressively improves alignment with human preferences. Notably,
after only a few TPO steps, the initially unaligned Llama-3.1-70B-SFT model can
surpass the aligned counterpart, Llama-3.1-70B-Instruct. Furthermore, TPO
scales efficiently with both the search width and depth during inference.
Through case studies, we illustrate how TPO exploits the innate capacity of LLM
to interpret and act upon reward signals. Our findings establish TPO as a
practical, lightweight alternative for test-time preference optimization,
achieving alignment on the fly. Our code is publicly available at
https://github.com/yafuly/TPO.


---

**[167. [2406.10040] FZI-WIM at SemEval-2024 Task 2: Self-Consistent CoT for Complex NLI in
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

**[168. [2504.13134] Energy-Based Reward Models for Robust Language Model Alignment](https://arxiv.org/pdf/2504.13134.pdf)** (2025-04-18)

*Anamika Lochab, Ruqi Zhang*

  Reward models (RMs) are essential for aligning Large Language Models (LLMs)
with human preferences. However, they often struggle with capturing complex
human preferences and generalizing to unseen data. To address these challenges,
we introduce Energy-Based Reward Model (EBRM), a lightweight post-hoc
refinement framework that enhances RM robustness and generalization. EBRM
models the reward distribution explicitly, capturing uncertainty in human
preferences and mitigating the impact of noisy or misaligned annotations. It
achieves this through conflict-aware data filtering, label-noise-aware
contrastive training, and hybrid initialization. Notably, EBRM enhances RMs
without retraining, making it computationally efficient and adaptable across
different models and tasks. Empirical evaluations on RM benchmarks demonstrate
significant improvements in both robustness and generalization, achieving up to
a 5.97% improvement in safety-critical alignment tasks compared to standard
RMs. Furthermore, reinforcement learning experiments confirm that our refined
rewards enhance alignment quality, effectively delaying reward hacking. These
results demonstrate our approach as a scalable and effective enhancement for
existing RMs and alignment pipelines. The code is available at EBRM.


---

**[169. [2502.11555] Equilibrate RLHF: Towards Balancing Helpfulness-Safety Trade-off in
  Large Language Models](https://arxiv.org/pdf/2502.11555.pdf)** (2025-02-18)

*Yingshui Tan, Yilei Jiang, Yanshi Li, Jiaheng Liu, Xingyuan Bu, Wenbo Su, Xiangyu Yue, Xiaoyong Zhu, Bo Zheng*

  Fine-tuning large language models (LLMs) based on human preferences, commonly
achieved through reinforcement learning from human feedback (RLHF), has been
effective in improving their performance. However, maintaining LLM safety
throughout the fine-tuning process remains a significant challenge, as
resolving conflicts between safety and helpfulness can be non-trivial.
Typically, the safety alignment of LLM is trained on data with safety-related
categories. However, our experiments find that naively increasing the scale of
safety training data usually leads the LLMs to an ``overly safe'' state rather
than a ``truly safe'' state, boosting the refusal rate through extensive
safety-aligned data without genuinely understanding the requirements for safe
responses. Such an approach can inadvertently diminish the models' helpfulness.
To understand the phenomenon, we first investigate the role of safety data by
categorizing them into three different groups, and observe that each group
behaves differently as training data scales up. To boost the balance between
safety and helpfulness, we propose an Equilibrate RLHF framework including a
Fine-grained Data-centric (FDC) approach that achieves better safety alignment
even with fewer training data, and an Adaptive Message-wise Alignment (AMA)
approach, which selectively highlight the key segments through a gradient
masking strategy. Extensive experimental results demonstrate that our approach
significantly enhances the safety alignment of LLMs while balancing safety and
helpfulness.


---

**[170. [2411.01245] PMoL: Parameter Efficient MoE for Preference Mixing of LLM Alignment](https://arxiv.org/pdf/2411.01245.pdf)** (2024-11-05)

*Dongxu Liu, Bing Xu, Yinzhuo Chen, Bufan Xu, Wenpeng Lu, Muyun Yang, Tiejun Zhao*

  Reinforcement Learning from Human Feedback (RLHF) has been proven to be an
effective method for preference alignment of large language models (LLMs) and
is widely used in the post-training process of LLMs. However, RLHF struggles
with handling multiple competing preferences. This leads to a decrease in the
alignment of LLMs with human preferences. To address this issue, we propose
Preference Mixture of LoRAs (PMoL) from the perspective of model architecture,
which can adapt to any number of preferences to mix. PMoL combines Mixture of
Experts (MoE) and Low Rank Adaptor (LoRA). This architecture is innovatively
applied to the research of preference alignment and has achieved significant
performance improvement. The expert group soft loss is used to enable MoE with
the ability to mix preferences. Through comprehensive evaluation by the reward
model and GPT-4o, the experiment results show that PMoL has superior preference
mixing capabilities compared to baseline methods. PMoL achieves better
preference alignment with lower training costs.


---

**[171. [2411.15736] Enhancing Few-Shot Out-of-Distribution Detection with Gradient Aligned
  Context Optimization](https://arxiv.org/pdf/2411.15736.pdf)** (2024-11-26)

*Baoshun Tong, Kaiyu Song, Hanjiang Lai*

  Few-shot out-of-distribution (OOD) detection aims to detect OOD images from
unseen classes with only a few labeled in-distribution (ID) images. To detect
OOD images and classify ID samples, prior methods have been proposed by
regarding the background regions of ID samples as the OOD knowledge and
performing OOD regularization and ID classification optimization. However, the
gradient conflict still exists between ID classification optimization and OOD
regularization caused by biased recognition. To address this issue, we present
Gradient Aligned Context Optimization (GaCoOp) to mitigate this gradient
conflict. Specifically, we decompose the optimization gradient to identify the
scenario when the conflict occurs. Then we alleviate the conflict in inner ID
samples and optimize the prompts via leveraging gradient projection. Extensive
experiments over the large-scale ImageNet OOD detection benchmark demonstrate
that our GaCoOp can effectively mitigate the conflict and achieve great
performance. Code will be available at https://github.com/BaoshunWq/ood-GaCoOp.


---

**[172. [2501.18280] Jailbreaking LLMs' Safeguard with Universal Magic Words for Text
  Embedding Models](https://arxiv.org/pdf/2501.18280.pdf)** (2025-02-11)

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

**[173. [2408.10608] Promoting Equality in Large Language Models: Identifying and Mitigating
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

**[174. [2402.15018] Unintended Impacts of LLM Alignment on Global Representation](https://arxiv.org/pdf/2402.15018.pdf)** (2024-06-10)

*Michael J. Ryan, William Held, Diyi Yang*

  Before being deployed for user-facing applications, developers align Large
Language Models (LLMs) to user preferences through a variety of procedures,
such as Reinforcement Learning From Human Feedback (RLHF) and Direct Preference
Optimization (DPO). Current evaluations of these procedures focus on benchmarks
of instruction following, reasoning, and truthfulness. However, human
preferences are not universal, and aligning to specific preference sets may
have unintended effects. We explore how alignment impacts performance along
three axes of global representation: English dialects, multilingualism, and
opinions from and about countries worldwide. Our results show that current
alignment procedures create disparities between English dialects and global
opinions. We find alignment improves capabilities in several languages. We
conclude by discussing design decisions that led to these unintended impacts
and recommendations for more equitable preference tuning. We make our code and
data publicly available on Github.


---

**[175. [2411.06899] LongSafety: Enhance Safety for Long-Context LLMs](https://arxiv.org/pdf/2411.06899.pdf)** (2025-02-28)

*Mianqiu Huang, Xiaoran Liu, Shaojun Zhou, Mozhi Zhang, Qipeng Guo, Linyang Li, Chenkun Tan, Yang Gao, Pengyu Wang, Linlin Li, Qun Liu, Yaqian Zhou, Xipeng Qiu, Xuanjing Huang*

  Recent advancements in model architectures and length extrapolation
techniques have significantly extended the context length of large language
models (LLMs), paving the way for their application in increasingly complex
tasks. However, despite the growing capabilities of long-context LLMs, the
safety issues in long-context scenarios remain underexplored. While safety
alignment in short context has been widely studied, the safety concerns of
long-context LLMs have not been adequately addressed. In this work, we
introduce \textbf{LongSafety}, a comprehensive safety alignment dataset for
long-context LLMs, containing 10 tasks and 17k samples, with an average length
of 40.9k tokens. Our experiments demonstrate that training with LongSafety can
enhance long-context safety performance while enhancing short-context safety
and preserving general capabilities. Furthermore, we demonstrate that
long-context safety does not equal long-context alignment with short-context
safety data and LongSafety has generalizing capabilities in context length and
long-context safety scenarios.


---

**[176. [2502.12970] Reasoning-to-Defend: Safety-Aware Reasoning Can Defend Large Language
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

**[177. [2309.11830] Goal-Oriented Prompt Attack and Safety Evaluation for LLMs](https://arxiv.org/pdf/2309.11830.pdf)** (2023-12-11)

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

**[178. [2502.14215] Towards Secure Program Partitioning for Smart Contracts with LLM's
  In-Context Learning](https://arxiv.org/pdf/2502.14215.pdf)** (2025-02-21)

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

**[179. [2410.14276] EcomEdit: An Automated E-commerce Knowledge Editing Framework for
  Enhanced Product and Purchase Intention Understanding](https://arxiv.org/pdf/2410.14276.pdf)** (2024-10-21)

*Ching Ming Samuel Lau, Weiqi Wang, Haochen Shi, Baixuan Xu, Jiaxin Bai, Yangqiu Song*

  Knowledge Editing (KE) aims to correct and update factual information in
Large Language Models (LLMs) to ensure accuracy and relevance without
computationally expensive fine-tuning. Though it has been proven effective in
several domains, limited work has focused on its application within the
e-commerce sector. However, there are naturally occurring scenarios that make
KE necessary in this domain, such as the timely updating of product features
and trending purchase intentions by customers, which necessitate further
exploration. In this paper, we pioneer the application of KE in the e-commerce
domain by presenting ECOMEDIT, an automated e-commerce knowledge editing
framework tailored for e-commerce-related knowledge and tasks. Our framework
leverages more powerful LLMs as judges to enable automatic knowledge conflict
detection and incorporates conceptualization to enhance the semantic coverage
of the knowledge to be edited. Through extensive experiments, we demonstrate
the effectiveness of ECOMEDIT in improving LLMs' understanding of product
descriptions and purchase intentions. We also show that LLMs, after our
editing, can achieve stronger performance on downstream e-commerce tasks.


---

**[180. [2405.17374] Navigating the Safety Landscape: Measuring Risks in Finetuning Large
  Language Models](https://arxiv.org/pdf/2405.17374.pdf)** (2024-11-01)

*ShengYun Peng, Pin-Yu Chen, Matthew Hull, Duen Horng Chau*

  Safety alignment is crucial to ensure that large language models (LLMs)
behave in ways that align with human preferences and prevent harmful actions
during inference. However, recent studies show that the alignment can be easily
compromised through finetuning with only a few adversarially designed training
examples. We aim to measure the risks in finetuning LLMs through navigating the
LLM safety landscape. We discover a new phenomenon observed universally in the
model parameter space of popular open-source LLMs, termed as "safety basin":
random perturbations to model weights maintain the safety level of the original
aligned model within its local neighborhood. However, outside this local
region, safety is fully compromised, exhibiting a sharp, step-like drop. This
safety basin contrasts sharply with the LLM capability landscape, where model
performance peaks at the origin and gradually declines as random perturbation
increases. Our discovery inspires us to propose the new VISAGE safety metric
that measures the safety in LLM finetuning by probing its safety landscape.
Visualizing the safety landscape of the aligned model enables us to understand
how finetuning compromises safety by dragging the model away from the safety
basin. The LLM safety landscape also highlights the system prompt's critical
role in protecting a model, and that such protection transfers to its perturbed
variants within the safety basin. These observations from our safety landscape
research provide new insights for future work on LLM safety community. Our code
is publicly available at https://github.com/ShengYun-Peng/llm-landscape.


---

**[181. [2406.09136] Chain of Preference Optimization: Improving Chain-of-Thought Reasoning
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

**[182. [2402.16444] ShieldLM: Empowering LLMs as Aligned, Customizable and Explainable
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

**[183. [2504.05689] Separator Injection Attack: Uncovering Dialogue Biases in Large Language
  Models Caused by Role Separators](https://arxiv.org/pdf/2504.05689.pdf)** (2025-04-09)

*Xitao Li, Haijun Wang, Jiang Wu, Ting Liu*

  Conversational large language models (LLMs) have gained widespread attention
due to their instruction-following capabilities. To ensure conversational LLMs
follow instructions, role separators are employed to distinguish between
different participants in a conversation. However, incorporating role
separators introduces potential vulnerabilities. Misusing roles can lead to
prompt injection attacks, which can easily misalign the model's behavior with
the user's intentions, raising significant security concerns. Although various
prompt injection attacks have been proposed, recent research has largely
overlooked the impact of role separators on safety. This highlights the
critical need to thoroughly understand the systemic weaknesses in dialogue
systems caused by role separators. This paper identifies modeling weaknesses
caused by role separators. Specifically, we observe a strong positional bias
associated with role separators, which is inherent in the format of dialogue
modeling and can be triggered by the insertion of role separators. We further
develop the Separators Injection Attack (SIA), a new orthometric attack based
on role separators. The experiment results show that SIA is efficient and
extensive in manipulating model behavior with an average gain of 18.2% for
manual methods and enhances the attack success rate to 100% with automatic
methods.


---

**[184. [2410.08431] oRetrieval Augmented Generation for 10 Large Language Models and its
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

**[185. [2402.05624] Efficient Models for the Detection of Hate, Abuse and Profanity](https://arxiv.org/pdf/2402.05624.pdf)** (2024-02-09)

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

**[186. [2409.09288] Generating API Parameter Security Rules with LLM for API Misuse
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

**[187. [2307.14539] Jailbreak in pieces: Compositional Adversarial Attacks on Multi-Modal
  Language Models](https://arxiv.org/pdf/2307.14539.pdf)** (2023-10-12)

*Erfan Shayegani, Yue Dong, Nael Abu-Ghazaleh*

  We introduce new jailbreak attacks on vision language models (VLMs), which
use aligned LLMs and are resilient to text-only jailbreak attacks.
Specifically, we develop cross-modality attacks on alignment where we pair
adversarial images going through the vision encoder with textual prompts to
break the alignment of the language model. Our attacks employ a novel
compositional strategy that combines an image, adversarially targeted towards
toxic embeddings, with generic prompts to accomplish the jailbreak. Thus, the
LLM draws the context to answer the generic prompt from the adversarial image.
The generation of benign-appearing adversarial images leverages a novel
embedding-space-based methodology, operating with no access to the LLM model.
Instead, the attacks require access only to the vision encoder and utilize one
of our four embedding space targeting strategies. By not requiring access to
the LLM, the attacks lower the entry barrier for attackers, particularly when
vision encoders such as CLIP are embedded in closed-source LLMs. The attacks
achieve a high success rate across different VLMs, highlighting the risk of
cross-modality alignment vulnerabilities, and the need for new alignment
approaches for multi-modal models.


---

**[188. [2406.17663] LLM-ARC: Enhancing LLMs with an Automated Reasoning Critic](https://arxiv.org/pdf/2406.17663.pdf)** (2024-07-22)

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

**[189. [2410.15483] Mitigating Forgetting in LLM Supervised Fine-Tuning and Preference
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
