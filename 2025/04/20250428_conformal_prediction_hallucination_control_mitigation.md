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

**[2. [2401.11974] Cross-Validation Conformal Risk Control](https://arxiv.org/pdf/2401.11974.pdf)** (2024-05-02)

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

**[3. [2501.17295] Mitigating Hallucinated Translations in Large Language Models with
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

**[4. [2410.11414] ReDeEP: Detecting Hallucination in Retrieval-Augmented Generation via
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

**[5. [2409.15548] Beyond Conformal Predictors: Adaptive Conformal Inference with
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

**[6. [2407.17377] Entropy Reweighted Conformal Classification](https://arxiv.org/pdf/2407.17377.pdf)** (2024-07-25)

*Rui Luo, Nicolo Colombo*

  Conformal Prediction (CP) is a powerful framework for constructing prediction
sets with guaranteed coverage. However, recent studies have shown that
integrating confidence calibration with CP can lead to a degradation in
efficiency. In this paper, We propose an adaptive approach that considers the
classifier's uncertainty and employs entropy-based reweighting to enhance the
efficiency of prediction sets for conformal classification. Our experimental
results demonstrate that this method significantly improves efficiency.


---

**[7. [2212.05765] Information-Theoretic Text Hallucination Reduction for Video-grounded
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

**[8. [2411.10436] Mitigating Hallucination in Multimodal Large Language Model via
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

**[9. [2403.00425] HALC: Object Hallucination Reduction via Adaptive Focal-Contrast
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

**[10. [2405.02140] An Information Theoretic Perspective on Conformal Prediction](https://arxiv.org/pdf/2405.02140.pdf)** (2025-02-18)

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

**[11. [2410.06494] Conformal Prediction: A Data Perspective](https://arxiv.org/pdf/2410.06494.pdf)** (2025-03-12)

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

**[12. [2301.02424] Conformal Loss-Controlling Prediction](https://arxiv.org/pdf/2301.02424.pdf)** (2024-01-24)

*Di Wang, Ping Wang, Zhong Ji, Xiaojun Yang, Hongyue Li*

  Conformal prediction is a learning framework controlling prediction coverage
of prediction sets, which can be built on any learning algorithm for point
prediction. This work proposes a learning framework named conformal
loss-controlling prediction, which extends conformal prediction to the
situation where the value of a loss function needs to be controlled. Different
from existing works about risk-controlling prediction sets and conformal risk
control with the purpose of controlling the expected values of loss functions,
the proposed approach in this paper focuses on the loss for any test object,
which is an extension of conformal prediction from miscoverage loss to some
general loss. The controlling guarantee is proved under the assumption of
exchangeability of data in finite-sample cases and the framework is tested
empirically for classification with a class-varying loss and statistical
postprocessing of numerical weather forecasting applications, which are
introduced as point-wise classification and point-wise regression problems. All
theoretical analysis and experimental results confirm the effectiveness of our
loss-controlling approach.


---

**[13. [2409.07902] Conformal Distributed Remote Inference in Sensor Networks Under
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

**[14. [2412.13817] Nullu: Mitigating Object Hallucinations in Large Vision-Language Models
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

**[15. [2504.12189] Leave-One-Out Stable Conformal Prediction](https://arxiv.org/pdf/2504.12189.pdf)** (2025-04-17)

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

**[16. [2405.18723] Conformal Depression Prediction](https://arxiv.org/pdf/2405.18723.pdf)** (2024-08-28)

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

**[17. [2310.02863] Conformal Predictions for Longitudinal Data](https://arxiv.org/pdf/2310.02863.pdf)** (2023-10-05)

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

**[18. [2310.18794] Sequence-Level Certainty Reduces Hallucination In Knowledge-Grounded
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

**[19. [2503.06149] Wireless Hallucination in Generative AI-enabled Communications:
  Concepts, Issues, and Solutions](https://arxiv.org/pdf/2503.06149.pdf)** (2025-03-11)

*Xudong Wang, Jiacheng Wang, Lei Feng, Dusit Niyato, Ruichen Zhang, Jiawen Kang, Zehui Xiong, Hongyang Du, Shiwen Mao*

  Generative AI (GenAI) is driving the intelligence of wireless communications.
Due to data limitations, random generation, and dynamic environments, GenAI may
generate channel information or optimization strategies that violate physical
laws or deviate from actual real-world requirements. We refer to this
phenomenon as wireless hallucination, which results in invalid channel
information, spectrum wastage, and low communication reliability but remains
underexplored. To address this gap, this article provides a comprehensive
concept of wireless hallucinations in GenAI-driven communications, focusing on
hallucination mitigation. Specifically, we first introduce the fundamental,
analyze its causes based on the GenAI workflow, and propose mitigation
solutions at the data, model, and post-generation levels. Then, we
systematically examines representative hallucination scenarios in GenAI-enabled
communications and their corresponding solutions. Finally, we propose a novel
integrated mitigation solution for GenAI-based channel estimation. At the data
level, we establish a channel estimation hallucination dataset and employ
generative adversarial networks (GANs)-based data augmentation. Additionally,
we incorporate attention mechanisms and large language models (LLMs) to enhance
both training and inference performance. Experimental results demonstrate that
the proposed hybrid solutions reduce the normalized mean square error (NMSE) by
0.19, effectively reducing wireless hallucinations.


---

**[20. [2501.18991] Optimal Transport-based Conformal Prediction](https://arxiv.org/pdf/2501.18991.pdf)** (2025-02-03)

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

**[21. [2402.05203] Bellman Conformal Inference: Calibrating Prediction Intervals For Time
  Series](https://arxiv.org/pdf/2402.05203.pdf)** (2024-02-12)

*Zitong Yang, Emmanuel Candès, Lihua Lei*

  We introduce Bellman Conformal Inference (BCI), a framework that wraps around
any time series forecasting models and provides approximately calibrated
prediction intervals. Unlike existing methods, BCI is able to leverage
multi-step ahead forecasts and explicitly optimize the average interval lengths
by solving a one-dimensional stochastic control problem (SCP) at each time
step. In particular, we use the dynamic programming algorithm to find the
optimal policy for the SCP. We prove that BCI achieves long-term coverage under
arbitrary distribution shifts and temporal dependence, even with poor
multi-step ahead forecasts. We find empirically that BCI avoids uninformative
intervals that have infinite lengths and generates substantially shorter
prediction intervals in multiple applications when compared with existing
methods.


---

**[22. [2309.05922] A Survey of Hallucination in Large Foundation Models](https://arxiv.org/pdf/2309.05922.pdf)** (2023-09-13)

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

**[23. [2010.08098] Agile Robot Navigation through Hallucinated Learning and Sober
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

**[24. [2307.16360] Probabilistically robust conformal prediction](https://arxiv.org/pdf/2307.16360.pdf)** (2023-08-01)

*Subhankar Ghosh, Yuanjie Shi, Taha Belkhouja, Yan Yan, Jana Doppa, Brian Jones*

  Conformal prediction (CP) is a framework to quantify uncertainty of machine
learning classifiers including deep neural networks. Given a testing example
and a trained classifier, CP produces a prediction set of candidate labels with
a user-specified coverage (i.e., true class label is contained with high
probability). Almost all the existing work on CP assumes clean testing data and
there is not much known about the robustness of CP algorithms w.r.t
natural/adversarial perturbations to testing examples. This paper studies the
problem of probabilistically robust conformal prediction (PRCP) which ensures
robustness to most perturbations around clean input examples. PRCP generalizes
the standard CP (cannot handle perturbations) and adversarially robust CP
(ensures robustness w.r.t worst-case perturbations) to achieve better
trade-offs between nominal performance and robustness. We propose a novel
adaptive PRCP (aPRCP) algorithm to achieve probabilistically robust coverage.
The key idea behind aPRCP is to determine two parallel thresholds, one for data
samples and another one for the perturbations on data (aka
"quantile-of-quantile" design). We provide theoretical analysis to show that
aPRCP algorithm achieves robust coverage. Our experiments on CIFAR-10,
CIFAR-100, and ImageNet datasets using deep neural networks demonstrate that
aPRCP achieves better trade-offs than state-of-the-art CP and adversarially
robust CP algorithms.


---

**[25. [2503.05757] Uncertainty-Aware Fusion: An Ensemble Framework for Mitigating
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

**[26. [2407.15441] Developing a Reliable, Fast, General-Purpose Hallucination Detection and
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

**[27. [2504.12137] Efficient Contrastive Decoding with Probabilistic Hallucination
  Detection - Mitigating Hallucinations in Large Vision Language Models -](https://arxiv.org/pdf/2504.12137.pdf)** (2025-04-17)

*Laura Fieback, Nishilkumar Balar, Jakob Spiegelberg, Hanno Gottschalk*

  Despite recent advances in Large Vision Language Models (LVLMs), these models
still suffer from generating hallucinatory responses that do not align with the
visual input provided. To mitigate such hallucinations, we introduce Efficient
Contrastive Decoding (ECD), a simple method that leverages probabilistic
hallucination detection to shift the output distribution towards contextually
accurate answers at inference time. By contrasting token probabilities and
hallucination scores, ECD subtracts hallucinated concepts from the original
distribution, effectively suppressing hallucinations. Notably, our proposed
method can be applied to any open-source LVLM and does not require additional
LVLM training. We evaluate our method on several benchmark datasets and across
different LVLMs. Our experiments show that ECD effectively mitigates
hallucinations, outperforming state-of-the-art methods with respect to
performance on LVLM benchmarks and computation time.


---

**[28. [2410.11701] Magnifier Prompt: Tackling Multimodal Hallucination via Extremely Simple
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

**[29. [2503.08342] Attention Reallocation: Towards Zero-cost and Controllable Hallucination
  Mitigation of MLLMs](https://arxiv.org/pdf/2503.08342.pdf)** (2025-03-13)

*Chongjun Tu, Peng Ye, Dongzhan Zhou, Lei Bai, Gang Yu, Tao Chen, Wanli Ouyang*

  Multi-Modal Large Language Models (MLLMs) stand out in various tasks but
still struggle with hallucinations. While recent training-free mitigation
methods mostly introduce additional inference overhead via retrospection
strategy and contrastive decoding, we propose attention reallocation (AttnReal)
to mitigate hallucinations with nearly zero extra cost. Our approach is
motivated by the key observations that, MLLM's unreasonable attention
distribution causes features to be dominated by historical output tokens, which
further contributes to hallucinated responses because of the distribution gap
between different token types. Based on the observations, AttnReal recycles
excessive attention from output tokens and reallocates it to visual tokens,
which reduces MLLM's reliance on language priors and ensures the decoding
process depends more on the visual inputs. More interestingly, we find that, by
controlling the intensity of AttnReal, we can achieve a wide-range trade-off
between the response faithfulness and overall performance. Comprehensive
results from different benchmarks validate the effectiveness of AttnReal across
six open-source MLLMs and three decoding strategies.


---

**[30. [2407.09417] Mitigating Entity-Level Hallucination in Large Language Models](https://arxiv.org/pdf/2407.09417.pdf)** (2024-07-23)

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

**[31. [2502.07497] On Training-Conditional Conformal Prediction and Binomial Proportion
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

**[32. [2502.12964] Trust Me, I'm Wrong: High-Certainty Hallucinations in LLMs](https://arxiv.org/pdf/2502.12964.pdf)** (2025-02-19)

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

**[33. [2502.01056] Mitigating Hallucinations in Large Vision-Language Models with Internal
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

**[34. [2408.15533] LRP4RAG: Detecting Hallucinations in Retrieval-Augmented Generation via
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

**[35. [2412.04235] Addressing Hallucinations with RAG and NMISS in Italian Healthcare LLM
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

**[36. [2501.08963] Training-Aware Risk Control for Intensity Modulated Radiation Therapies
  Quality Assurance with Conformal Prediction](https://arxiv.org/pdf/2501.08963.pdf)** (2025-01-16)

*Kevin He, David Adam, Sarah Han-Oh, Anqi Liu*

  Measurement quality assurance (QA) practices play a key role in the safe use
of Intensity Modulated Radiation Therapies (IMRT) for cancer treatment. These
practices have reduced measurement-based IMRT QA failure below 1%. However,
these practices are time and labor intensive which can lead to delays in
patient care. In this study, we examine how conformal prediction methodologies
can be used to robustly triage plans. We propose a new training-aware conformal
risk control method by combining the benefit of conformal risk control and
conformal training. We incorporate the decision making thresholds based on the
gamma passing rate, along with the risk functions used in clinical evaluation,
into the design of the risk control framework. Our method achieves high
sensitivity and specificity and significantly reduces the number of plans
needing measurement without generating a huge confidence interval. Our results
demonstrate the validity and applicability of conformal prediction methods for
improving efficiency and reducing the workload of the IMRT QA process.


---

**[37. [2309.16781] Hallucination Reduction in Long Input Text Summarization](https://arxiv.org/pdf/2309.16781.pdf)** (2023-10-02)

*Tohida Rehman, Ronit Mandal, Abhishek Agarwal, Debarshi Kumar Sanyal*

  Hallucination in text summarization refers to the phenomenon where the model
generates information that is not supported by the input source document.
Hallucination poses significant obstacles to the accuracy and reliability of
the generated summaries. In this paper, we aim to reduce hallucinated outputs
or hallucinations in summaries of long-form text documents. We have used the
PubMed dataset, which contains long scientific research documents and their
abstracts. We have incorporated the techniques of data filtering and joint
entity and summary generation (JAENS) in the fine-tuning of the Longformer
Encoder-Decoder (LED) model to minimize hallucinations and thereby improve the
quality of the generated summary. We have used the following metrics to measure
factual consistency at the entity level: precision-source, and F1-target. Our
experiments show that the fine-tuned LED model performs well in generating the
paper abstract. Data filtering techniques based on some preprocessing steps
reduce entity-level hallucinations in the generated summaries in terms of some
of the factual consistency metrics.


---

**[38. [2412.05223] 100% Elimination of Hallucinations on RAGTruth for GPT-4 and GPT-3.5
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

**[39. [2402.03744] INSIDE: LLMs' Internal States Retain the Power of Hallucination
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

**[40. [2503.17229] FactSelfCheck: Fact-Level Black-Box Hallucination Detection for LLMs](https://arxiv.org/pdf/2503.17229.pdf)** (2025-03-24)

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

**[41. [2502.05911] GRAIT: Gradient-Driven Refusal-Aware Instruction Tuning for Effective
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

**[42. [2403.19082] Enhancing Conformal Prediction Using E-Test Statistics](https://arxiv.org/pdf/2403.19082.pdf)** (2024-03-29)

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

**[43. [2405.18942] Verifiably Robust Conformal Prediction](https://arxiv.org/pdf/2405.18942.pdf)** (2024-11-19)

*Linus Jeary, Tom Kuipers, Mehran Hosseini, Nicola Paoletti*

  Conformal Prediction (CP) is a popular uncertainty quantification method that
provides distribution-free, statistically valid prediction sets, assuming that
training and test data are exchangeable. In such a case, CP's prediction sets
are guaranteed to cover the (unknown) true test output with a user-specified
probability. Nevertheless, this guarantee is violated when the data is
subjected to adversarial attacks, which often result in a significant loss of
coverage. Recently, several approaches have been put forward to recover CP
guarantees in this setting. These approaches leverage variations of randomised
smoothing to produce conservative sets which account for the effect of the
adversarial perturbations. They are, however, limited in that they only support
$\ell^2$-bounded perturbations and classification tasks. This paper introduces
VRCP (Verifiably Robust Conformal Prediction), a new framework that leverages
recent neural network verification methods to recover coverage guarantees under
adversarial attacks. Our VRCP method is the first to support perturbations
bounded by arbitrary norms including $\ell^1$, $\ell^2$, and $\ell^\infty$, as
well as regression tasks. We evaluate and compare our approach on image
classification tasks (CIFAR10, CIFAR100, and TinyImageNet) and regression tasks
for deep reinforcement learning environments. In every case, VRCP achieves
above nominal coverage and yields significantly more efficient and informative
prediction regions than the SotA.


---

**[44. [2501.02699] EAGLE: Enhanced Visual Grounding Minimizes Hallucinations in
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

**[45. [2502.14105] Conformal Prediction under L\'evy-Prokhorov Distribution Shifts:
  Robustness to Local and Global Perturbations](https://arxiv.org/pdf/2502.14105.pdf)** (2025-02-21)

*Liviu Aolaritei, Michael I. Jordan, Youssef Marzouk, Zheyu Oliver Wang, Julie Zhu*

  Conformal prediction provides a powerful framework for constructing
prediction intervals with finite-sample guarantees, yet its robustness under
distribution shifts remains a significant challenge. This paper addresses this
limitation by modeling distribution shifts using L\'evy-Prokhorov (LP)
ambiguity sets, which capture both local and global perturbations. We provide a
self-contained overview of LP ambiguity sets and their connections to popular
metrics such as Wasserstein and Total Variation. We show that the link between
conformal prediction and LP ambiguity sets is a natural one: by propagating the
LP ambiguity set through the scoring function, we reduce complex
high-dimensional distribution shifts to manageable one-dimensional distribution
shifts, enabling exact quantification of worst-case quantiles and coverage.
Building on this analysis, we construct robust conformal prediction intervals
that remain valid under distribution shifts, explicitly linking LP parameters
to interval width and confidence levels. Experimental results on real-world
datasets demonstrate the effectiveness of the proposed approach.


---

**[46. [2406.06818] Conformal Prediction for Class-wise Coverage via Augmented Label Rank
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

**[47. [2412.06007] Hallucination-aware Optimization for Large Language Model-empowered
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

**[48. [2310.03951] Chain of Natural Language Inference for Reducing Large Language Model
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

**[49. [2503.04615] HalluCounter: Reference-free LLM Hallucination Detection in the Wild!](https://arxiv.org/pdf/2503.04615.pdf)** (2025-03-07)

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

**[50. [2408.16729] Prediction-Feedback DETR for Temporal Action Detection](https://arxiv.org/pdf/2408.16729.pdf)** (2024-12-20)

*Jihwan Kim, Miso Lee, Cheol-Ho Cho, Jihyun Lee, Jae-Pil Heo*

  Temporal Action Detection (TAD) is fundamental yet challenging for real-world
video applications. Leveraging the unique benefits of transformers, various
DETR-based approaches have been adopted in TAD. However, it has recently been
identified that the attention collapse in self-attention causes the performance
degradation of DETR for TAD. Building upon previous research, this paper newly
addresses the attention collapse problem in cross-attention within DETR-based
TAD methods. Moreover, our findings reveal that cross-attention exhibits
patterns distinct from predictions, indicating a short-cut phenomenon. To
resolve this, we propose a new framework, Prediction-Feedback DETR (Pred-DETR),
which utilizes predictions to restore the collapse and align the cross- and
self-attention with predictions. Specifically, we devise novel
prediction-feedback objectives using guidance from the relations of the
predictions. As a result, Pred-DETR significantly alleviates the collapse and
achieves state-of-the-art performance among DETR-based methods on various
challenging benchmarks including THUMOS14, ActivityNet-v1.3, HACS, and
FineAction.


---

**[51. [2504.12314] How to Detect and Defeat Molecular Mirage: A Metric-Driven Benchmark for
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

**[52. [2210.10254] Safe Planning in Dynamic Environments using Conformal Prediction](https://arxiv.org/pdf/2210.10254.pdf)** (2023-06-09)

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

**[53. [2502.15079] Can Hallucination Correction Improve Video-Language Alignment?](https://arxiv.org/pdf/2502.15079.pdf)** (2025-02-24)

*Lingjun Zhao, Mingyang Xie, Paola Cascante-Bonilla, III Hal Daumé, Kwonjoon Lee*

  Large Vision-Language Models often generate hallucinated content that is not
grounded in its visual inputs. While prior work focuses on mitigating
hallucinations, we instead explore leveraging hallucination correction as a
training objective to improve video-language alignment. We introduce HACA, a
self-training framework learning to correct hallucinations in descriptions that
do not align with the video content. By identifying and correcting
inconsistencies, HACA enhances the model's ability to align video and textual
representations for spatio-temporal reasoning. Our experimental results show
consistent gains in video-caption binding and text-to-video retrieval tasks,
demonstrating that hallucination correction-inspired tasks serve as an
effective strategy for improving vision and language alignment.


---

**[54. [2411.02712] V-DPO: Mitigating Hallucination in Large Vision Language Models via
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

**[55. [2504.07069] HalluciNot: Hallucination Detection Through Context and Common Knowledge
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

**[56. [2402.08680] Mitigating Object Hallucination in Large Vision-Language Models via
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

**[57. [2411.04847] Prompt-Guided Internal States for Hallucination Detection of Large
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

**[58. [2304.01075] Conformal Prediction Regions for Time Series using Linear
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

**[59. [2406.06496] Direct Preference Optimization for Suppressing Hallucinated Prior Exams
  in Radiology Report Generation](https://arxiv.org/pdf/2406.06496.pdf)** (2024-06-18)

*Oishi Banerjee, Hong-Yu Zhou, Subathra Adithan, Stephen Kwak, Kay Wu, Pranav Rajpurkar*

  Recent advances in generative vision-language models (VLMs) have exciting
potential implications for AI in radiology, yet VLMs are also known to produce
hallucinations, nonsensical text, and other unwanted behaviors that can waste
clinicians' time and cause patient harm. Drawing on recent work on direct
preference optimization (DPO), we propose a simple method for modifying the
behavior of pretrained VLMs performing radiology report generation by
suppressing unwanted types of generations. We apply our method to the
prevention of hallucinations of prior exams, addressing a long-established
problem behavior in models performing chest X-ray report generation. Across our
experiments, we find that DPO fine-tuning achieves a 3.2-4.8x reduction in
lines hallucinating prior exams while maintaining model performance on clinical
accuracy metrics. Our work is, to the best of our knowledge, the first work to
apply DPO to medical VLMs, providing a data- and compute- efficient way to
suppress problem behaviors while maintaining overall clinical accuracy.


---

**[60. [2311.01740] SAC3: Reliable Hallucination Detection in Black-Box Language Models via
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

**[61. [2502.13416] Detecting LLM Fact-conflicting Hallucinations Enhanced by
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

**[62. [2410.19493] Conditional Hallucinations for Image Compression](https://arxiv.org/pdf/2410.19493.pdf)** (2025-03-07)

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

**[63. [2402.09801] EFUF: Efficient Fine-grained Unlearning Framework for Mitigating
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

**[64. [2411.09689] LLM Hallucination Reasoning with Zero-shot Knowledge Test](https://arxiv.org/pdf/2411.09689.pdf)** (2024-11-15)

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

**[65. [2407.20999] MoFO: Momentum-Filtered Optimizer for Mitigating Forgetting in LLM
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

**[66. [2502.03695] Reduce Lap Time for Autonomous Racing with Curvature-Integrated MPCC
  Local Trajectory Planning Method](https://arxiv.org/pdf/2502.03695.pdf)** (2025-03-25)

*Zhouheng Li, Lei Xie, Cheng Hu, Hongye Su*

  The widespread application of autonomous driving technology has significantly
advanced the field of autonomous racing. Model Predictive Contouring Control
(MPCC) is a highly effective local trajectory planning method for autonomous
racing. However, the traditional MPCC method struggles with racetracks that
have significant curvature changes, limiting the performance of the vehicle
during autonomous racing. To address this issue, we propose a
curvature-integrated MPCC (CiMPCC) local trajectory planning method for
autonomous racing. This method optimizes the velocity of the local trajectory
based on the curvature of the racetrack centerline. The specific implementation
involves mapping the curvature of the racetrack centerline to a reference
velocity profile, which is then incorporated into the cost function for
optimizing the velocity of the local trajectory. This reference velocity
profile is created by normalizing and mapping the curvature of the racetrack
centerline, thereby ensuring efficient and performance-oriented local
trajectory planning in racetracks with significant curvature. The proposed
CiMPCC method has been experimented on a self-built 1:10 scale F1TENTH racing
vehicle deployed with ROS platform. The experimental results demonstrate that
the proposed method achieves outstanding results on a challenging racetrack
with sharp curvature, improving the overall lap time by 11.4%-12.5% compared to
other autonomous racing trajectory planning methods. Our code is available at
https://github.com/zhouhengli/CiMPCC.


---

**[67. [2308.15126] Evaluation and Analysis of Hallucination in Large Vision-Language Models](https://arxiv.org/pdf/2308.15126.pdf)** (2023-10-11)

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

**[68. [2210.12496] Bayesian Optimization with Conformal Prediction Sets](https://arxiv.org/pdf/2210.12496.pdf)** (2023-12-13)

*Samuel Stanton, Wesley Maddox, Andrew Gordon Wilson*

  Bayesian optimization is a coherent, ubiquitous approach to decision-making
under uncertainty, with applications including multi-arm bandits, active
learning, and black-box optimization. Bayesian optimization selects decisions
(i.e. objective function queries) with maximal expected utility with respect to
the posterior distribution of a Bayesian model, which quantifies reducible,
epistemic uncertainty about query outcomes. In practice, subjectively
implausible outcomes can occur regularly for two reasons: 1) model
misspecification and 2) covariate shift. Conformal prediction is an uncertainty
quantification method with coverage guarantees even for misspecified models and
a simple mechanism to correct for covariate shift. We propose conformal
Bayesian optimization, which directs queries towards regions of search space
where the model predictions have guaranteed validity, and investigate its
behavior on a suite of black-box optimization tasks and tabular ranking tasks.
In many cases we find that query coverage can be significantly improved without
harming sample-efficiency.


---

**[69. [2403.10492] Mitigating Dialogue Hallucination for Large Vision Language Models via
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

**[70. [2502.11306] Smoothing Out Hallucinations: Mitigating LLM Hallucination with Smoothed
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

**[71. [2504.00447] Egocentric Conformal Prediction for Safe and Efficient Navigation in
  Dynamic Cluttered Environments](https://arxiv.org/pdf/2504.00447.pdf)** (2025-04-02)

*Jaeuk Shin, Jungjin Lee, Insoon Yang*

  Conformal prediction (CP) has emerged as a powerful tool in robotics and
control, thanks to its ability to calibrate complex, data-driven models with
formal guarantees. However, in robot navigation tasks, existing CP-based
methods often decouple prediction from control, evaluating models without
considering whether prediction errors actually compromise safety. Consequently,
ego-vehicles may become overly conservative or even immobilized when all
potential trajectories appear infeasible. To address this issue, we propose a
novel CP-based navigation framework that responds exclusively to
safety-critical prediction errors. Our approach introduces egocentric score
functions that quantify how much closer obstacles are to a candidate vehicle
position than anticipated. These score functions are then integrated into a
model predictive control scheme, wherein each candidate state is individually
evaluated for safety. Combined with an adaptive CP mechanism, our framework
dynamically adjusts to changes in obstacle motion without resorting to
unnecessary conservatism. Theoretical analyses indicate that our method
outperforms existing CP-based approaches in terms of cost-efficiency while
maintaining the desired safety levels, as further validated through experiments
on real-world datasets featuring densely populated pedestrian environments.


---

**[72. [2402.04344] Does confidence calibration improve conformal prediction?](https://arxiv.org/pdf/2402.04344.pdf)** (2024-12-24)

*Huajun Xi, Jianguo Huang, Kangdao Liu, Lei Feng, Hongxin Wei*

  Conformal prediction is an emerging technique for uncertainty quantification
that constructs prediction sets guaranteed to contain the true label with a
predefined probability. Previous works often employ temperature scaling to
calibrate classifiers, assuming that confidence calibration benefits conformal
prediction. However, the specific impact of confidence calibration on conformal
prediction remains underexplored. In this work, we make two key discoveries
about the impact of confidence calibration methods on adaptive conformal
prediction. Firstly, we empirically show that current confidence calibration
methods (e.g., temperature scaling) typically lead to larger prediction sets in
adaptive conformal prediction. Secondly, by investigating the role of
temperature value, we observe that high-confidence predictions can enhance the
efficiency of adaptive conformal prediction. Theoretically, we prove that
predictions with higher confidence result in smaller prediction sets on
expectation. This finding implies that the rescaling parameters in these
calibration methods, when optimized with cross-entropy loss, might counteract
the goal of generating efficient prediction sets. To address this issue, we
propose Conformal Temperature Scaling (ConfTS), a variant of temperature
scaling with a novel loss function designed to enhance the efficiency of
prediction sets. This approach can be extended to optimize the parameters of
other post-hoc methods of confidence calibration. Extensive experiments
demonstrate that our method improves existing adaptive conformal prediction
methods in classification tasks, especially with LLMs.


---

**[73. [2503.17395] CP-NCBF: A Conformal Prediction-based Approach to Synthesize Verified
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

**[74. [2407.04121] Hallucination Detection: Robustly Discerning Reliable Answers in Large
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

**[75. [2406.15927] Semantic Entropy Probes: Robust and Cheap Hallucination Detection in
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

**[76. [2412.12597] Distribution-Free Uncertainty Quantification in Mechanical Ventilation
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

**[77. [2501.19164] Poison as Cure: Visual Noise for Mitigating Object Hallucinations in
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

**[78. [2503.13107] ClearSight: Visual Signal Enhancement for Object Hallucination
  Mitigation in Multimodal Large language Models](https://arxiv.org/pdf/2503.13107.pdf)** (2025-03-18)

*Hao Yin, Guangzong Si, Zilei Wang*

  Contrastive decoding strategies are widely used to mitigate object
hallucinations in multimodal large language models (MLLMs). By reducing
over-reliance on language priors, these strategies ensure that generated
content remains closely grounded in visual inputs, producing contextually
accurate outputs. Since contrastive decoding requires no additional training or
external tools, it offers both computational efficiency and versatility, making
it highly attractive. However, these methods present two main limitations: (1)
bluntly suppressing language priors can compromise coherence and accuracy of
generated content, and (2) processing contrastive inputs adds computational
load, significantly slowing inference speed. To address these challenges, we
propose Visual Amplification Fusion (VAF), a plug-and-play technique that
enhances attention to visual signals within the model's middle layers, where
modality fusion predominantly occurs. This approach enables more effective
capture of visual features, reducing the model's bias toward language modality.
Experimental results demonstrate that VAF significantly reduces hallucinations
across various MLLMs without affecting inference speed, while maintaining
coherence and accuracy in generated outputs.


---

**[79. [2411.15224] Parameter Efficient Mamba Tuning via Projector-targeted Diagonal-centric
  Linear Transformation](https://arxiv.org/pdf/2411.15224.pdf)** (2025-03-25)

*Seokil Ham, Hee-Seon Kim, Sangmin Woo, Changick Kim*

  Despite the growing interest in Mamba architecture as a potential replacement
for Transformer architecture, parameter-efficient fine-tuning (PEFT) approaches
for Mamba remain largely unexplored. In our study, we introduce two key
insights-driven strategies for PEFT in Mamba architecture: (1) While
state-space models (SSMs) have been regarded as the cornerstone of Mamba
architecture, then expected to play a primary role in transfer learning, our
findings reveal that Projectors -- not SSMs -- are the predominant contributors
to transfer learning. (2) Based on our observation, we propose a novel PEFT
method specialized to Mamba architecture: Projector-targeted Diagonal-centric
Linear Transformation (ProDiaL). ProDiaL focuses on optimizing only the
pretrained Projectors for new tasks through diagonal-centric linear
transformation matrices, without directly fine-tuning the Projector weights.
This targeted approach allows efficient task adaptation, utilizing less than 1%
of the total parameters, and exhibits strong performance across both vision and
language Mamba models, highlighting its versatility and effectiveness.


---

**[80. [2502.02998] Conformal Uncertainty Indicator for Continual Test-Time Adaptation](https://arxiv.org/pdf/2502.02998.pdf)** (2025-02-06)

*Fan Lyu, Hanyu Zhao, Ziqi Shi, Ye Liu, Fuyuan Hu, Zhang Zhang, Liang Wang*

  Continual Test-Time Adaptation (CTTA) aims to adapt models to sequentially
changing domains during testing, relying on pseudo-labels for self-adaptation.
However, incorrect pseudo-labels can accumulate, leading to performance
degradation. To address this, we propose a Conformal Uncertainty Indicator
(CUI) for CTTA, leveraging Conformal Prediction (CP) to generate prediction
sets that include the true label with a specified coverage probability. Since
domain shifts can lower the coverage than expected, making CP unreliable, we
dynamically compensate for the coverage by measuring both domain and data
differences. Reliable pseudo-labels from CP are then selectively utilized to
enhance adaptation. Experiments confirm that CUI effectively estimates
uncertainty and improves adaptation performance across various existing CTTA
methods.


---

**[81. [2407.15130] DOPRA: Decoding Over-accumulation Penalization and Re-allocation in
  Specific Weighting Layer](https://arxiv.org/pdf/2407.15130.pdf)** (2024-07-24)

*Jinfeng Wei, Xiaofeng Zhang*

  In this work, we introduce DOPRA, a novel approach designed to mitigate
hallucinations in multi-modal large language models (MLLMs). Unlike existing
solutions that typically involve costly supplementary training data or the
integration of external knowledge sources, DOPRA innovatively addresses
hallucinations by decoding specific weighted layer penalties and
redistribution, offering an economical and effective solution without
additional resources. DOPRA is grounded in unique insights into the intrinsic
mechanisms controlling hallucinations within MLLMs, especially the models'
tendency to over-rely on a subset of summary tokens in the self-attention
matrix, neglecting critical image-related information. This phenomenon is
particularly pronounced in certain strata. To counteract this over-reliance,
DOPRA employs a strategy of weighted overlay penalties and redistribution in
specific layers, such as the 12th layer, during the decoding process.
Furthermore, DOPRA includes a retrospective allocation process that re-examines
the sequence of generated tokens, allowing the algorithm to reallocate token
selection to better align with the actual image content, thereby reducing the
incidence of hallucinatory descriptions in auto-generated captions. Overall,
DOPRA represents a significant step forward in improving the output quality of
MLLMs by systematically reducing hallucinations through targeted adjustments
during the decoding process.


---

**[82. [2502.16872] Mitigating Hallucinations in Diffusion Models through Adaptive Attention
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

**[83. [2503.03106] Monitoring Decoding: Mitigating Hallucination via Evaluating the
  Factuality of Partial Response during Generation](https://arxiv.org/pdf/2503.03106.pdf)** (2025-03-06)

*Yurui Chang, Bochuan Cao, Lu Lin*

  While large language models have demonstrated exceptional performance across
a wide range of tasks, they remain susceptible to hallucinations -- generating
plausible yet factually incorrect contents. Existing methods to mitigating such
risk often rely on sampling multiple full-length generations, which introduces
significant response latency and becomes ineffective when the model
consistently produces hallucinated outputs with high confidence. To address
these limitations, we introduce Monitoring Decoding (MD), a novel framework
that dynamically monitors the generation process and selectively applies
in-process interventions, focusing on revising crucial tokens responsible for
hallucinations. Instead of waiting until completion of multiple full-length
generations, we identify hallucination-prone tokens during generation using a
monitor function, and further refine these tokens through a tree-based decoding
strategy. This approach ensures an enhanced factual accuracy and coherence in
the generated output while maintaining efficiency. Experimental results
demonstrate that MD consistently outperforms self-consistency-based approaches
in both effectiveness and efficiency, achieving higher factual accuracy while
significantly reducing computational overhead.


---

**[84. [2104.04250] Periodic Load Rejection for Floating Offshore Wind Turbines via
  Constrained Subspace Predictive Repetitive Control](https://arxiv.org/pdf/2104.04250.pdf)** (2021-04-12)

*Yichao Liu, Riccardo M. G. Ferrari, Jan-Willem van Wingerden*

  Individual Pitch Control (IPC) is an effective control strategy to mitigate
the blade loads on large-scale wind turbines. Since IPC usually requires high
pitch actuation, the safety constraints of the pitch actuator should be taken
into account when designing the controller. This paper introduces a constrained
Subspace Predictive Repetitive Control (SPRC) approach, which considers the
limitation of blade pitch angle and pitch rate. To fulfill this goal, a model
predictive control scheme is implemented in the fully data-driven SPRC approach
to incorporate the physical limitations of the pitch actuator in the control
problem formulation. An optimal control law subjected to constraints is then
formulated so that future constraint violations are anticipated and prevented.
Case studies show that the developed constrained SPRC reduces the pitch
activities necessary to mitigate the blade loads when experiencing wind
turbulence and abrupt wind gusts. More importantly, the approach allows the
wind farm operator to design conservative bounds for the pitch actuator
constraints that satisfies safety limitations, design specifications and
physical restrictions. This will help to alleviate the cyclic fatigue loads on
the actuators, increase the structural reliability and extend the lifespan of
the pitch control system.


---

**[85. [2406.07457] Estimating the Hallucination Rate of Generative AI](https://arxiv.org/pdf/2406.07457.pdf)** (2024-12-10)

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

**[86. [2407.09165] Robust Yet Efficient Conformal Prediction Sets](https://arxiv.org/pdf/2407.09165.pdf)** (2024-07-15)

*Soroush H. Zargarbashi, Mohammad Sadegh Akhondzadeh, Aleksandar Bojchevski*

  Conformal prediction (CP) can convert any model's output into prediction sets
guaranteed to include the true label with any user-specified probability.
However, same as the model itself, CP is vulnerable to adversarial test
examples (evasion) and perturbed calibration data (poisoning). We derive
provably robust sets by bounding the worst-case change in conformity scores.
Our tighter bounds lead to more efficient sets. We cover both continuous and
discrete (sparse) data and our guarantees work both for evasion and poisoning
attacks (on both features and labels).


---

**[87. [2502.20034] Vision-Encoders (Already) Know What They See: Mitigating Object
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

**[88. [2411.01558] Adaptive Conformal Inference by Particle Filtering under Hidden Markov
  Models](https://arxiv.org/pdf/2411.01558.pdf)** (2024-11-05)

*Xiaoyi Su, Zhixin Zhou, Rui Luo*

  Conformal inference is a statistical method used to construct prediction sets
for point predictors, providing reliable uncertainty quantification with
probability guarantees. This method utilizes historical labeled data to
estimate the conformity or nonconformity between predictions and true labels.
However, conducting conformal inference for hidden states under hidden Markov
models (HMMs) presents a significant challenge, as the hidden state data is
unavailable, resulting in the absence of a true label set to serve as a
conformal calibration set. This paper proposes an adaptive conformal inference
framework that leverages a particle filtering approach to address this issue.
Rather than directly focusing on the unobservable hidden state, we innovatively
use weighted particles as an approximation of the actual posterior distribution
of the hidden state. Our goal is to produce prediction sets that encompass
these particles to achieve a specific aggregate weight sum, referred to as the
aggregated coverage level. The proposed framework can adapt online to the
time-varying distribution of data and achieve the defined marginal aggregated
coverage level in both one-step and multi-step inference over the long term. We
verify the effectiveness of this approach through a real-time target
localization simulation study.


---

**[89. [2412.20167] Conformal Risk Control for Pulmonary Nodule Detection](https://arxiv.org/pdf/2412.20167.pdf)** (2024-12-31)

*Roel Hulsman, Valentin Comte, Lorenzo Bertolini, Tobias Wiesenthal, Antonio Puertas Gallardo, Mario Ceresa*

  Quantitative tools are increasingly appealing for decision support in
healthcare, driven by the growing capabilities of advanced AI systems. However,
understanding the predictive uncertainties surrounding a tool's output is
crucial for decision-makers to ensure reliable and transparent decisions. In
this paper, we present a case study on pulmonary nodule detection for lung
cancer screening, enhancing an advanced detection model with an uncertainty
quantification technique called conformal risk control (CRC). We demonstrate
that prediction sets with conformal guarantees are attractive measures of
predictive uncertainty in the safety-critical healthcare domain, allowing
end-users to achieve arbitrary validity by trading off false positives and
providing formal statistical guarantees on model performance. Among
ground-truth nodules annotated by at least three radiologists, our model
achieves a sensitivity that is competitive with that generally achieved by
individual radiologists, with a slight increase in false positives.
Furthermore, we illustrate the risks of using off-the-shelve prediction models
when faced with ontological uncertainty, such as when radiologists disagree on
what constitutes the ground truth on pulmonary nodules.


---

**[90. [2401.10768] Knowledge Verification to Nip Hallucination in the Bud](https://arxiv.org/pdf/2401.10768.pdf)** (2024-09-24)

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

**[91. [2504.08809] Decoupling Contrastive Decoding: Robust Hallucination Mitigation in
  Multimodal Large Language Models](https://arxiv.org/pdf/2504.08809.pdf)** (2025-04-15)

*Wei Chen, Xin Yan, Bin Wen, Fan Yang, Tingting Gao, Di Zhang, Long Chen*

  Although multimodal large language models (MLLMs) exhibit remarkable
reasoning capabilities on complex multimodal understanding tasks, they still
suffer from the notorious hallucination issue: generating outputs misaligned
with obvious visual or factual evidence. Currently, training-based solutions,
like direct preference optimization (DPO), leverage paired preference data to
suppress hallucinations. However, they risk sacrificing general reasoning
capabilities due to the likelihood displacement. Meanwhile, training-free
solutions, like contrastive decoding, achieve this goal by subtracting the
estimated hallucination pattern from a distorted input. Yet, these handcrafted
perturbations (e.g., add noise to images) may poorly capture authentic
hallucination patterns. To avoid these weaknesses of existing methods, and
realize robust hallucination mitigation (i.e., maintaining general reasoning
performance), we propose a novel framework: Decoupling Contrastive Decoding
(DCD). Specifically, DCD decouples the learning of positive and negative
samples in preference datasets, and trains separate positive and negative image
projections within the MLLM. The negative projection implicitly models real
hallucination patterns, which enables vision-aware negative images in the
contrastive decoding inference stage. Our DCD alleviates likelihood
displacement by avoiding pairwise optimization and generalizes robustly without
handcrafted degradation. Extensive ablations across hallucination benchmarks
and general reasoning tasks demonstrate the effectiveness of DCD, i.e., it
matches DPO's hallucination suppression while preserving general capabilities
and outperforms the handcrafted contrastive decoding methods.


---

**[92. [2406.11267] Mitigating Large Language Model Hallucination with Faithful Finetuning](https://arxiv.org/pdf/2406.11267.pdf)** (2024-06-18)

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

**[93. [2212.02712] Improved Beam Search for Hallucination Mitigation in Abstractive
  Summarization](https://arxiv.org/pdf/2212.02712.pdf)** (2023-11-15)

*Arvind Krishna Sridhar, Erik Visser*

  Advancement in large pretrained language models has significantly improved
their performance for conditional language generation tasks including
summarization albeit with hallucinations. To reduce hallucinations,
conventional methods proposed improving beam search or using a fact checker as
a postprocessing step. In this paper, we investigate the use of the Natural
Language Inference (NLI) entailment metric to detect and prevent hallucinations
in summary generation. We propose an NLI-assisted beam re-ranking mechanism by
computing entailment probability scores between the input context and
summarization model-generated beams during saliency-enhanced greedy decoding.
Moreover, a diversity metric is introduced to compare its effectiveness against
vanilla beam search. Our proposed algorithm significantly outperforms vanilla
beam decoding on XSum and CNN/DM datasets.


---

**[94. [2502.03023] Parametric Scaling Law of Tuning Bias in Conformal Prediction](https://arxiv.org/pdf/2502.03023.pdf)** (2025-02-06)

*Hao Zeng, Kangdao Liu, Bingyi Jing, Hongxin Wei*

  Conformal prediction is a popular framework of uncertainty quantification
that constructs prediction sets with coverage guarantees. To uphold the
exchangeability assumption, many conformal prediction methods necessitate an
additional holdout set for parameter tuning. Yet, the impact of violating this
principle on coverage remains underexplored, making it ambiguous in practical
applications. In this work, we empirically find that the tuning bias - the
coverage gap introduced by leveraging the same dataset for tuning and
calibration, is negligible for simple parameter tuning in many conformal
prediction methods. In particular, we observe the scaling law of the tuning
bias: this bias increases with parameter space complexity and decreases with
calibration set size. Formally, we establish a theoretical framework to
quantify the tuning bias and provide rigorous proof for the scaling law of the
tuning bias by deriving its upper bound. In the end, we discuss how to reduce
the tuning bias, guided by the theories we developed.


---

**[95. [2503.10345] Mirror Online Conformal Prediction with Intermittent Feedback](https://arxiv.org/pdf/2503.10345.pdf)** (2025-03-18)

*Bowen Wang, Matteo Zecchin, Osvaldo Simeone*

  Online conformal prediction enables the runtime calibration of a pre-trained
artificial intelligence model using feedback on its performance. Calibration is
achieved through set predictions that are updated via online rules so as to
ensure long-term coverage guarantees. While recent research has demonstrated
the benefits of incorporating prior knowledge into the calibration process,
this has come at the cost of replacing coverage guarantees with less tangible
regret guarantees based on the quantile loss. This work introduces intermittent
mirror online conformal prediction (IM-OCP), a novel runtime calibration
framework that integrates prior knowledge, while maintaining long-term coverage
and achieving sub-linear regret. IM-OCP features closed-form updates with
minimal memory complexity, and is designed to operate under potentially
intermittent feedback.


---

**[96. [2205.14317] A Confidence Machine for Sparse High-Order Interaction Model](https://arxiv.org/pdf/2205.14317.pdf)** (2022-11-02)

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

**[97. [2503.21813] OAEI-LLM-T: A TBox Benchmark Dataset for Understanding LLM
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

**[98. [2411.17387] Robust Bayesian Optimization via Localized Online Conformal Prediction](https://arxiv.org/pdf/2411.17387.pdf)** (2025-04-02)

*Dongwon Kim, Matteo Zecchin, Sangwoo Park, Joonhyuk Kang, Osvaldo Simeone*

  Bayesian optimization (BO) is a sequential approach for optimizing black-box
objective functions using zeroth-order noisy observations. In BO, Gaussian
processes (GPs) are employed as probabilistic surrogate models to estimate the
objective function based on past observations, guiding the selection of future
queries to maximize utility. However, the performance of BO heavily relies on
the quality of these probabilistic estimates, which can deteriorate
significantly under model misspecification. To address this issue, we introduce
localized online conformal prediction-based Bayesian optimization (LOCBO), a BO
algorithm that calibrates the GP model through localized online conformal
prediction (CP). LOCBO corrects the GP likelihood based on predictive sets
produced by LOCBO, and the corrected GP likelihood is then denoised to obtain a
calibrated posterior distribution on the objective function. The likelihood
calibration step leverages an input-dependent calibration threshold to tailor
coverage guarantees to different regions of the input space. Under minimal
noise assumptions, we provide theoretical performance guarantees for LOCBO's
iterates that hold for the unobserved objective function. These theoretical
findings are validated through experiments on synthetic and real-world
optimization tasks, demonstrating that LOCBO consistently outperforms
state-of-the-art BO algorithms in the presence of model misspecification.


---

**[99. [2402.10612] Retrieve Only When It Needs: Adaptive Retrieval Augmentation for
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

**[100. [2502.06884] Learning Conformal Abstention Policies for Adaptive Risk Management in
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

**[101. [2409.11353] THaMES: An End-to-End Tool for Hallucination Mitigation and Evaluation
  in Large Language Models](https://arxiv.org/pdf/2409.11353.pdf)** (2025-01-22)

*Mengfei Liang, Archish Arun, Zekun Wu, Cristian Munoz, Jonathan Lutch, Emre Kazim, Adriano Koshiyama, Philip Treleaven*

  Hallucination, the generation of factually incorrect content, is a growing
challenge in Large Language Models (LLMs). Existing detection and mitigation
methods are often isolated and insufficient for domain-specific needs, lacking
a standardized pipeline. This paper introduces THaMES (Tool for Hallucination
Mitigations and EvaluationS), an integrated framework and library addressing
this gap. THaMES offers an end-to-end solution for evaluating and mitigating
hallucinations in LLMs, featuring automated test set generation, multifaceted
benchmarking, and adaptable mitigation strategies. It automates test set
creation from any corpus, ensuring high data quality, diversity, and
cost-efficiency through techniques like batch processing, weighted sampling,
and counterfactual validation. THaMES assesses a model's ability to detect and
reduce hallucinations across various tasks, including text generation and
binary classification, applying optimal mitigation strategies like In-Context
Learning (ICL), Retrieval Augmented Generation (RAG), and Parameter-Efficient
Fine-tuning (PEFT). Evaluations of state-of-the-art LLMs using a knowledge base
of academic papers, political news, and Wikipedia reveal that commercial models
like GPT-4o benefit more from RAG than ICL, while open-weight models like
Llama-3.1-8B-Instruct and Mistral-Nemo gain more from ICL. Additionally, PEFT
significantly enhances the performance of Llama-3.1-8B-Instruct in both
evaluation tasks.


---

**[102. [2410.19385] Investigating the Role of Prompting and External Tools in Hallucination
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

**[103. [2405.01976] Conformal Prediction for Natural Language Processing: A Survey](https://arxiv.org/pdf/2405.01976.pdf)** (2024-05-06)

*Margarida M. Campos, António Farinhas, Chrysoula Zerva, Mário A. T. Figueiredo, André F. T. Martins*

  The rapid proliferation of large language models and natural language
processing (NLP) applications creates a crucial need for uncertainty
quantification to mitigate risks such as hallucinations and to enhance
decision-making reliability in critical applications. Conformal prediction is
emerging as a theoretically sound and practically useful framework, combining
flexibility with strong statistical guarantees. Its model-agnostic and
distribution-free nature makes it particularly promising to address the current
shortcomings of NLP systems that stem from the absence of uncertainty
quantification. This paper provides a comprehensive survey of conformal
prediction techniques, their guarantees, and existing applications in NLP,
pointing to directions for future research and open challenges.


---

**[104. [2305.13632] Detecting and Mitigating Hallucinations in Multilingual Summarisation](https://arxiv.org/pdf/2305.13632.pdf)** (2023-10-27)

*Yifu Qiu, Yftah Ziser, Anna Korhonen, Edoardo M. Ponti, Shay B. Cohen*

  Hallucinations pose a significant challenge to the reliability of neural
models for abstractive summarisation. While automatically generated summaries
may be fluent, they often lack faithfulness to the original document. This
issue becomes even more pronounced in low-resource settings, such as
cross-lingual transfer. With the existing faithful metrics focusing on English,
even measuring the extent of this phenomenon in cross-lingual settings is hard.
To address this, we first develop a novel metric, mFACT, evaluating the
faithfulness of non-English summaries, leveraging translation-based transfer
from multiple English faithfulness metrics. We then propose a simple but
effective method to reduce hallucinations with a cross-lingual transfer, which
weighs the loss of each training example by its faithfulness score. Through
extensive experiments in multiple languages, we demonstrate that mFACT is the
metric that is most suited to detect hallucinations. Moreover, we find that our
proposed loss weighting method drastically increases both performance and
faithfulness according to both automatic and human evaluation when compared to
strong baselines for cross-lingual transfer such as MAD-X. Our code and dataset
are available at https://github.com/yfqiu-nlp/mfact-summ.


---

**[105. [2501.14544] Distributed Conformal Prediction via Message Passing](https://arxiv.org/pdf/2501.14544.pdf)** (2025-01-27)

*Haifeng Wen, Hong Xing, Osvaldo Simeone*

  Post-hoc calibration of pre-trained models is critical for ensuring reliable
inference, especially in safety-critical domains such as healthcare. Conformal
Prediction (CP) offers a robust post-hoc calibration framework, providing
distribution-free statistical coverage guarantees for prediction sets by
leveraging held-out datasets. In this work, we address a decentralized setting
where each device has limited calibration data and can communicate only with
its neighbors over an arbitrary graph topology. We propose two
message-passing-based approaches for achieving reliable inference via CP:
quantile-based distributed conformal prediction (Q-DCP) and histogram-based
distributed conformal prediction (H-DCP). Q-DCP employs distributed quantile
regression enhanced with tailored smoothing and regularization terms to
accelerate convergence, while H-DCP uses a consensus-based histogram estimation
approach. Through extensive experiments, we investigate the trade-offs between
hyperparameter tuning requirements, communication overhead, coverage
guarantees, and prediction set sizes across different network topologies.


---

**[106. [2502.19209] Bi'an: A Bilingual Benchmark and Model for Hallucination Detection in
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

**[107. [2502.17125] LettuceDetect: A Hallucination Detection Framework for RAG Applications](https://arxiv.org/pdf/2502.17125.pdf)** (2025-02-25)

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

**[108. [2503.21157] Real-Time Evaluation Models for RAG: Who Detects Hallucinations Best?](https://arxiv.org/pdf/2503.21157.pdf)** (2025-04-08)

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

**[109. [2402.11875] M2K-VDG: Model-Adaptive Multimodal Knowledge Anchor Enhanced
  Video-grounded Dialogue Generation](https://arxiv.org/pdf/2402.11875.pdf)** (2024-02-20)

*Hongcheng Liu, Pingjie Wang, Yu Wang, Yanfeng Wang*

  Video-grounded dialogue generation (VDG) requires the system to generate a
fluent and accurate answer based on multimodal knowledge. However, the
difficulty in multimodal knowledge utilization brings serious hallucinations to
VDG models in practice. Although previous works mitigate the hallucination in a
variety of ways, they hardly take notice of the importance of the multimodal
knowledge anchor answer tokens. In this paper, we reveal via perplexity that
different VDG models experience varying hallucinations and exhibit diverse
anchor tokens. Based on this observation, we propose M2K-VDG, a model-adaptive
multimodal knowledge anchor enhancement framework for hallucination reduction.
Furthermore, we introduce the counterfactual effect for more accurate anchor
token detection. The experimental results on three popular benchmarks exhibit
the superiority of our approach over state-of-the-art methods, demonstrating
its effectiveness in reducing hallucinations.


---

**[110. [2310.10003] Conformal Contextual Robust Optimization](https://arxiv.org/pdf/2310.10003.pdf)** (2023-10-17)

*Yash Patel, Sahana Rayan, Ambuj Tewari*

  Data-driven approaches to predict-then-optimize decision-making problems seek
to mitigate the risk of uncertainty region misspecification in safety-critical
settings. Current approaches, however, suffer from considering overly
conservative uncertainty regions, often resulting in suboptimal decisionmaking.
To this end, we propose Conformal-Predict-Then-Optimize (CPO), a framework for
leveraging highly informative, nonconvex conformal prediction regions over
high-dimensional spaces based on conditional generative models, which have the
desired distribution-free coverage guarantees. Despite guaranteeing robustness,
such black-box optimization procedures alone inspire little confidence owing to
the lack of explanation of why a particular decision was found to be optimal.
We, therefore, augment CPO to additionally provide semantically meaningful
visual summaries of the uncertainty regions to give qualitative intuition for
the optimal decision. We highlight the CPO framework by demonstrating results
on a suite of simulation-based inference benchmark tasks and a vehicle routing
task based on probabilistic weather prediction.


---

**[111. [2407.03282] LLM Internal States Reveal Hallucination Risk Faced With a Query](https://arxiv.org/pdf/2407.03282.pdf)** (2024-10-01)

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

**[112. [2502.11948] Can Your Uncertainty Scores Detect Hallucinated Entity?](https://arxiv.org/pdf/2502.11948.pdf)** (2025-02-18)

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

**[113. [2409.16494] A Unified Hallucination Mitigation Framework for Large Vision-Language
  Models](https://arxiv.org/pdf/2409.16494.pdf)** (2024-09-26)

*Yue Chang, Liqiang Jing, Xiaopeng Zhang, Yue Zhang*

  Hallucination is a common problem for Large Vision-Language Models (LVLMs)
with long generations which is difficult to eradicate. The generation with
hallucinations is partially inconsistent with the image content. To mitigate
hallucination, current studies either focus on the process of model inference
or the results of model generation, but the solutions they design sometimes do
not deal appropriately with various types of queries and the hallucinations of
the generations about these queries. To accurately deal with various
hallucinations, we present a unified framework, Dentist, for hallucination
mitigation. The core step is to first classify the queries, then perform
different processes of hallucination mitigation based on the classification
result, just like a dentist first observes the teeth and then makes a plan. In
a simple deployment, Dentist can classify queries as perception or reasoning
and easily mitigate potential hallucinations in answers which has been
demonstrated in our experiments. On MMbench, we achieve a 13.44%/10.2%/15.8%
improvement in accuracy on Image Quality, a Coarse Perception visual question
answering (VQA) task, over the baseline InstructBLIP/LLaVA/VisualGLM.


---

**[114. [2406.17260] Mitigating Hallucination in Fictional Character Role-Play](https://arxiv.org/pdf/2406.17260.pdf)** (2024-11-12)

*Nafis Sadeq, Zhouhang Xie, Byungkyu Kang, Prarit Lamba, Xiang Gao, Julian McAuley*

  Role-playing has wide-ranging applications in customer support, embodied
agents, and computational social science. The influence of parametric world
knowledge of large language models (LLMs) often causes role-playing characters
to act out of character and to hallucinate about things outside the scope of
their knowledge. In this work, we focus on the evaluation and mitigation of
hallucination in fictional character role-play. We introduce a dataset with
over 2,000 characters and 72,000 interviews, including 18,000 adversarial
questions. We propose RoleFact, a role-playing method that mitigates
hallucination by modulating the influence of parametric knowledge using a
pre-calibrated confidence threshold. Experiments show that the proposed method
improves the factual precision of generated responses by 18% for adversarial
questions with a 44% reduction in temporal hallucination for time-sensitive
interviews. The code and the dataset are available at
https://github.com/NafisSadeq/rolefact.git.


---

**[115. [2410.14748] ETF: An Entity Tracing Framework for Hallucination Detection in Code
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

**[116. [2401.03403] Deep peak property learning for efficient chiral molecules ECD spectra
  prediction](https://arxiv.org/pdf/2401.03403.pdf)** (2024-01-09)

*Hao Li, Da Long, Li Yuan, Yonghong Tian, Xinchang Wang, Fanyang Mo*

  Chiral molecule assignation is crucial for asymmetric catalysis, functional
materials, and the drug industry. The conventional approach requires
theoretical calculations of electronic circular dichroism (ECD) spectra, which
is time-consuming and costly. To speed up this process, we have incorporated
deep learning techniques for the ECD prediction. We first set up a large-scale
dataset of Chiral Molecular ECD spectra (CMCDS) with calculated ECD spectra. We
further develop the ECDFormer model, a Transformer-based model to learn the
chiral molecular representations and predict corresponding ECD spectra with
improved efficiency and accuracy. Unlike other models for spectrum prediction,
our ECDFormer creatively focused on peak properties rather than the whole
spectrum sequence for prediction, inspired by the scenario of chiral molecule
assignation. Specifically, ECDFormer predicts the peak properties, including
number, position, and symbol, then renders the ECD spectra from these peak
properties, which significantly outperforms other models in ECD prediction, Our
ECDFormer reduces the time of acquiring ECD spectra from 1-100 hours per
molecule to 1.5s.


---

**[117. [2408.15037] Evidence-Enhanced Triplet Generation Framework for Hallucination
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

**[118. [2403.14401] Pensieve: Retrospect-then-Compare Mitigates Visual Hallucination](https://arxiv.org/pdf/2403.14401.pdf)** (2024-09-04)

*Dingchen Yang, Bowen Cao, Guang Chen, Changjun Jiang*

  Multi-modal Large Language Models (MLLMs) demonstrate remarkable success
across various vision-language tasks. However, they suffer from visual
hallucination, where the generated responses diverge from the provided image.
Are MLLMs oblivious to the accurate visual cues when they hallucinate? Our
investigation reveals that the visual branch may equally advocate both accurate
and erroneous content. To address this issue, we propose Pensieve, a
training-free method that leverages the analogous visual hallucinations, which
are induced by images sharing common semantic and appearance characteristics,
to mitigate hallucination. Specifically, Pensieve enables MLLMs to retrospect
relevant images as references and compare their visual content with the test
image via confidence score subtraction. Moreover, our paradigm balances the
effects of addressing errors from both the visual and textual branches by
adaptively scaling the subtracted scores. Experiments on Whoops, LLaVA Bench,
POPE, and MME demonstrate the efficacy of Pensieve in mitigating visual
hallucination, surpassing other advanced decoding strategies. Pensieve also
aids MLLMs in identifying visual details and enhance the specificity of
generated image descriptions.


---

**[119. [2410.19077] Target Strangeness: A Novel Conformal Prediction Difficulty Estimator](https://arxiv.org/pdf/2410.19077.pdf)** (2024-10-28)

*Alexis Bose, Jonathan Ethier, Paul Guinand*

  This paper introduces Target Strangeness, a novel difficulty estimator for
conformal prediction (CP) that offers an alternative approach for normalizing
prediction intervals (PIs). By assessing how atypical a prediction is within
the context of its nearest neighbours' target distribution, Target Strangeness
can surpass the current state-of-the-art performance. This novel difficulty
estimator is evaluated against others in the context of several conformal
regression experiments.


---

**[120. [2504.08020] Learning Fine-grained Domain Generalization via Hyperbolic State Space
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

**[121. [2206.06584] Probabilistic Conformal Prediction Using Conditional Random Samples](https://arxiv.org/pdf/2206.06584.pdf)** (2022-06-22)

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

**[122. [2501.12749] Estimating the Conformal Prediction Threshold from Noisy Labels](https://arxiv.org/pdf/2501.12749.pdf)** (2025-01-23)

*Coby Penso, Jacob Goldberger, Ethan Fetaya*

  Conformal Prediction (CP) is a method to control prediction uncertainty by
producing a small prediction set, ensuring a predetermined probability that the
true class lies within this set. This is commonly done by defining a score,
based on the model predictions, and setting a threshold on this score using a
validation set. In this study, we address the problem of CP calibration when we
only have access to a validation set with noisy labels. We show how we can
estimate the noise-free conformal threshold based on the noisy labeled data.
Our solution is flexible and can accommodate various modeling assumptions
regarding the label contamination process, without needing any information
about the underlying data distribution or the internal mechanisms of the
machine learning classifier. We develop a coverage guarantee for uniform noise
that is effective even in tasks with a large number of classes. We dub our
approach Noise-Aware Conformal Prediction (NACP) and show on several natural
and medical image classification datasets, including ImageNet, that it
significantly outperforms current noisy label methods and achieves results
comparable to those obtained with a clean validation set.


---

**[123. [2402.09733] Do LLMs Know about Hallucination? An Empirical Investigation of LLM's
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

**[124. [2312.15576] Reducing LLM Hallucinations using Epistemic Neural Networks](https://arxiv.org/pdf/2312.15576.pdf)** (2023-12-27)

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

**[125. [2407.00569] Investigating and Mitigating the Multimodal Hallucination Snowballing in
  Large Vision-Language Models](https://arxiv.org/pdf/2407.00569.pdf)** (2024-08-06)

*Weihong Zhong, Xiaocheng Feng, Liang Zhao, Qiming Li, Lei Huang, Yuxuan Gu, Weitao Ma, Yuan Xu, Bing Qin*

  Though advanced in understanding visual information with human languages,
Large Vision-Language Models (LVLMs) still suffer from multimodal
hallucinations. A natural concern is that during multimodal interaction, the
generated hallucinations could influence the LVLMs' subsequent generation.
Thus, we raise a question: When presented with a query relevant to the
previously generated hallucination, will LVLMs be misled and respond
incorrectly, even though the ground visual information exists? To answer this,
we propose a framework called MMHalSnowball to evaluate LVLMs' behaviors when
encountering generated hallucinations, where LVLMs are required to answer
specific visual questions within a curated hallucinatory conversation.
Crucially, our experiment shows that the performance of open-source LVLMs drops
by at least $31\%$, indicating that LVLMs are prone to accept the generated
hallucinations and make false claims that they would not have supported without
distractions. We term this phenomenon Multimodal Hallucination Snowballing. To
mitigate this, we further propose a training-free method called Residual Visual
Decoding, where we revise the output distribution of LVLMs with the one derived
from the residual visual input, providing models with direct access to the
visual information. Experiments show that our method can mitigate more than
$24\%$ of the snowballed multimodal hallucination while maintaining
capabilities.


---

**[126. [2207.03343] Refutation of Spectral Graph Theory Conjectures with Monte Carlo Search](https://arxiv.org/pdf/2207.03343.pdf)** (2022-08-04)

*Milo Roucairol, Tristan Cazenave*

  We demonstrate how Monte Carlo Search (MCS) algorithms, namely Nested Monte
Carlo Search (NMCS) and Nested Rollout Policy Adaptation (NRPA), can be used to
build graphs and find counter-examples to spectral graph theory conjectures in
minutes.


---

**[127. [2501.01926] Mitigating Hallucination for Large Vision Language Model by
  Inter-Modality Correlation Calibration Decoding](https://arxiv.org/pdf/2501.01926.pdf)** (2025-03-13)

*Jiaming Li, Jiacheng Zhang, Zequn Jie, Lin Ma, Guanbin Li*

  Large vision-language models (LVLMs) have shown remarkable capabilities in
visual-language understanding for downstream multi-modal tasks. Despite their
success, LVLMs still suffer from generating hallucinations in complex
generation tasks, leading to inconsistencies between visual inputs and
generated content. To address this issue, some approaches have introduced
inference-time interventions, such as contrastive decoding and attention
rectification, to reduce overreliance on language priors. However, these
approaches overlook hallucinations stemming from spurious inter-modality
correlations. In this paper, we propose an Inter-Modality Correlation
Calibration Decoding (IMCCD) method to mitigate hallucinations in LVLMs in a
training-free manner. In this method, we design a Cross-Modal Value-Enhanced
Decoding(CMVED) module to alleviate hallucination by a novel contrastive
decoding mechanism. During the estimation of distorted distribution, CMVED
masks the value vectors associated with significant cross-modal attention
weights, which address both uni-modality overreliance and misleading
inter-modality correlations. Additionally, a Content-Driven Attention
Refinement(CDAR) module refines cross-modal attention weights, guiding LVLMs to
focus on important visual content. Experimental results on diverse
hallucination benchmarks validate the superiority of our method over existing
state-of-the-art techniques in reducing hallucinations in LVLM text generation.
Our code will be available at https://github.com/lijm48/IMCCD.


---

**[128. [2404.02935] KnowHalu: Hallucination Detection via Multi-Form Knowledge Based Factual
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

**[129. [2108.09793] From Agile Ground to Aerial Navigation: Learning from Learned
  Hallucination](https://arxiv.org/pdf/2108.09793.pdf)** (2021-08-24)

*Zizhao Wang, Xuesu Xiao, Alexander J Nettekoven, Kadhiravan Umasankar, Anika Singh, Sriram Bommakanti, Ufuk Topcu, Peter Stone*

  This paper presents a self-supervised Learning from Learned Hallucination
(LfLH) method to learn fast and reactive motion planners for ground and aerial
robots to navigate through highly constrained environments. The recent Learning
from Hallucination (LfH) paradigm for autonomous navigation executes motion
plans by random exploration in completely safe obstacle-free spaces, uses
hand-crafted hallucination techniques to add imaginary obstacles to the robot's
perception, and then learns motion planners to navigate in realistic,
highly-constrained, dangerous spaces. However, current hand-crafted
hallucination techniques need to be tailored for specific robot types (e.g., a
differential drive ground vehicle), and use approximations heavily dependent on
certain assumptions (e.g., a short planning horizon). In this work, instead of
manually designing hallucination functions, LfLH learns to hallucinate obstacle
configurations, where the motion plans from random exploration in open space
are optimal, in a self-supervised manner. LfLH is robust to different robot
types and does not make assumptions about the planning horizon. Evaluated in
both simulated and physical environments with a ground and an aerial robot,
LfLH outperforms or performs comparably to previous hallucination approaches,
along with sampling- and optimization-based classical methods.


---

**[130. [2406.11514] Counterfactual Debating with Preset Stances for Hallucination
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

**[131. [2502.13490] What are Models Thinking about? Understanding Large Language Model
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

**[132. [2503.03032] SAFE: A Sparse Autoencoder-Based Framework for Robust Query Enrichment
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

**[133. [2410.18860] DeCoRe: Decoding by Contrasting Retrieval Heads to Mitigate
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

**[134. [2405.09886] MTLComb: multi-task learning combining regression and classification
  tasks for joint feature selection](https://arxiv.org/pdf/2405.09886.pdf)** (2024-05-17)

*Han Cao, Sivanesan Rajan, Bianka Hahn, Ersoy Kocak, Daniel Durstewitz, Emanuel Schwarz, Verena Schneider-Lindner*

  Multi-task learning (MTL) is a learning paradigm that enables the
simultaneous training of multiple communicating algorithms. Although MTL has
been successfully applied to ether regression or classification tasks alone,
incorporating mixed types of tasks into a unified MTL framework remains
challenging, primarily due to variations in the magnitudes of losses associated
with different tasks. This challenge, particularly evident in MTL applications
with joint feature selection, often results in biased selections. To overcome
this obstacle, we propose a provable loss weighting scheme that analytically
determines the optimal weights for balancing regression and classification
tasks. This scheme significantly mitigates the otherwise biased feature
selection. Building upon this scheme, we introduce MTLComb, an MTL algorithm
and software package encompassing optimization procedures, training protocols,
and hyperparameter estimation procedures. MTLComb is designed for learning
shared predictors among tasks of mixed types. To showcase the efficacy of
MTLComb, we conduct tests on both simulated data and biomedical studies
pertaining to sepsis and schizophrenia.


---

**[135. [2503.01670] Evaluating LLMs' Assessment of Mixed-Context Hallucination Through the
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

**[136. [2410.15778] Reducing Hallucinations in Vision-Language Models via Latent Space
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

**[137. [2501.18363] Robust Online Conformal Prediction under Uniform Label Noise](https://arxiv.org/pdf/2501.18363.pdf)** (2025-02-04)

*Huajun Xi, Kangdao Liu, Hao Zeng, Wenguang Sun, Hongxin Wei*

  Conformal prediction is an emerging technique for uncertainty quantification
that constructs prediction sets guaranteed to contain the true label with a
predefined probability. Recent work develops online conformal prediction
methods that adaptively construct prediction sets to accommodate distribution
shifts. However, existing algorithms typically assume perfect label accuracy
which rarely holds in practice. In this work, we investigate the robustness of
online conformal prediction under uniform label noise with a known noise rate,
in both constant and dynamic learning rate schedules. We show that label noise
causes a persistent gap between the actual mis-coverage rate and the desired
rate $\alpha$, leading to either overestimated or underestimated coverage
guarantees. To address this issue, we propose Noise Robust Online Conformal
Prediction (dubbed NR-OCP) by updating the threshold with a novel robust
pinball loss, which provides an unbiased estimate of clean pinball loss without
requiring ground-truth labels. Our theoretical analysis shows that NR-OCP
eliminates the coverage gap in both constant and dynamic learning rate
schedules, achieving a convergence rate of $\mathcal{O}(T^{-1/2})$ for both
empirical and expected coverage errors under uniform label noise. Extensive
experiments demonstrate the effectiveness of our method by achieving both
precise coverage and improved efficiency.


---

**[138. [2403.01548] In-Context Sharpness as Alerts: An Inner Representation Perspective for
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

**[139. [2503.08340] Online Conformal Compression for Zero-Delay Communication with
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

**[140. [2402.15300] Seeing is Believing: Mitigating Hallucination in Large Vision-Language
  Models via CLIP-Guided Decoding](https://arxiv.org/pdf/2402.15300.pdf)** (2024-04-24)

*Ailin Deng, Zhirui Chen, Bryan Hooi*

  Large Vision-Language Models (LVLMs) are susceptible to object
hallucinations, an issue in which their generated text contains non-existent
objects, greatly limiting their reliability and practicality. Current
approaches often rely on the model's token likelihoods or other internal
information, instruction tuning on additional datasets, or incorporating
complex external tools. We first perform empirical analysis on sentence-level
LVLM hallucination, finding that CLIP similarity to the image acts as a
stronger and more robust indicator of hallucination compared to token
likelihoods. Motivated by this, we introduce our CLIP-Guided Decoding (CGD)
approach, a straightforward but effective training-free approach to reduce
object hallucination at decoding time. CGD uses CLIP to guide the model's
decoding process by enhancing visual grounding of generated text with the
image. Experiments demonstrate that CGD effectively mitigates object
hallucination across multiple LVLM families while preserving the utility of
text generation. Codes are available at
https://github.com/d-ailin/CLIP-Guided-Decoding.


---

**[141. [2406.12053] InternalInspector $I^2$: Robust Confidence Estimation in LLMs through
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

**[142. [2410.15926] Mitigating Object Hallucination via Concentric Causal Attention](https://arxiv.org/pdf/2410.15926.pdf)** (2024-10-22)

*Yun Xing, Yiheng Li, Ivan Laptev, Shijian Lu*

  Recent Large Vision Language Models (LVLMs) present remarkable zero-shot
conversational and reasoning capabilities given multimodal queries.
Nevertheless, they suffer from object hallucination, a phenomenon where LVLMs
are prone to generate textual responses not factually aligned with image
inputs. Our pilot study reveals that object hallucination is closely tied with
Rotary Position Encoding (RoPE), a widely adopted positional dependency
modeling design in existing LVLMs. Due to the long-term decay in RoPE, LVLMs
tend to hallucinate more when relevant visual cues are distant from instruction
tokens in the multimodal input sequence. Additionally, we observe a similar
effect when reversing the sequential order of visual tokens during multimodal
alignment. Our tests indicate that long-term decay in RoPE poses challenges to
LVLMs while capturing visual-instruction interactions across long distances. We
propose Concentric Causal Attention (CCA), a simple yet effective positional
alignment strategy that mitigates the impact of RoPE long-term decay in LVLMs
by naturally reducing relative distance between visual and instruction tokens.
With CCA, visual tokens can better interact with instruction tokens, thereby
enhancing model's perception capability and alleviating object hallucination.
Without bells and whistles, our positional alignment method surpasses existing
hallucination mitigation strategies by large margins on multiple object
hallucination benchmarks.


---

**[143. [2411.12591] Thinking Before Looking: Improving Multimodal LLM Reasoning via
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

**[144. [2308.14239] Quantum Next Generation Reservoir Computing: An Efficient Quantum
  Algorithm for Forecasting Quantum Dynamics](https://arxiv.org/pdf/2308.14239.pdf)** (2025-01-24)

*Apimuk Sornsaeng, Ninnat Dangniam, Thiparat Chotibut*

  Next Generation Reservoir Computing (NG-RC) is a modern class of model-free
machine learning that enables an accurate forecasting of time series data
generated by dynamical systems. We demonstrate that NG-RC can accurately
predict full many-body quantum dynamics in both integrable and chaotic systems.
This is in contrast to the conventional application of reservoir computing that
concentrates on the prediction of the dynamics of observables. In addition, we
apply a technique which we refer to as skipping ahead to predict far future
states accurately without the need to extract information about the
intermediate states. However, adopting a classical NG-RC for many-body quantum
dynamics prediction is computationally prohibitive due to the large Hilbert
space of sample input data. In this work, we propose an end-to-end quantum
algorithm for many-body quantum dynamics forecasting with a quantum
computational speedup via the block-encoding technique. This proposal presents
an efficient model-free quantum scheme to forecast quantum dynamics coherently,
bypassing inductive biases incurred in a model-based approach.


---

**[145. [2405.15356] Alleviating Hallucinations in Large Vision-Language Models through
  Hallucination-Induced Optimization](https://arxiv.org/pdf/2405.15356.pdf)** (2025-04-02)

*Xinyu Lyu, Beitao Chen, Lianli Gao, Jingkuan Song, Heng Tao Shen*

  Although Large Visual Language Models (LVLMs) have demonstrated exceptional
abilities in understanding multimodal data, they invariably suffer from
hallucinations, leading to a disconnect between the generated text and the
corresponding images. Almost all current visual contrastive decoding methods
attempt to mitigate these hallucinations by introducing visual uncertainty
information that appropriately widens the contrastive logits gap between
hallucinatory and targeted ones. However, due to uncontrollable nature of the
global visual uncertainty, they struggle to precisely induce the hallucinatory
tokens, which severely limits their effectiveness in mitigating hallucinations
and may even lead to the generation of undesired hallucinations. To tackle this
issue, we conducted the theoretical analysis to promote the effectiveness of
contrast decoding. Building on this insight, we introduce a novel optimization
strategy named Hallucination-Induced Optimization (HIO). This strategy seeks to
amplify the contrast between hallucinatory and targeted tokens relying on a
fine-tuned theoretical preference model (i.e., Contrary Bradley-Terry Model),
thereby facilitating efficient contrast decoding to alleviate hallucinations in
LVLMs. Extensive experimental research demonstrates that our HIO strategy can
effectively reduce hallucinations in LVLMs, outperforming state-of-the-art
methods across various benchmarks.


---

**[146. [2504.10198] DioR: Adaptive Cognitive Detection and Contextual Retrieval Optimization
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

**[147. [2502.06221] Interaction-aware Conformal Prediction for Crowd Navigation](https://arxiv.org/pdf/2502.06221.pdf)** (2025-02-11)

*Zhe Huang, Tianchen Ji, Heling Zhang, Fatemeh Cheraghi Pouria, Katherine Driggs-Campbell, Roy Dong*

  During crowd navigation, robot motion plan needs to consider human motion
uncertainty, and the human motion uncertainty is dependent on the robot motion
plan. We introduce Interaction-aware Conformal Prediction (ICP) to alternate
uncertainty-aware robot motion planning and decision-dependent human motion
uncertainty quantification. ICP is composed of a trajectory predictor to
predict human trajectories, a model predictive controller to plan robot motion
with confidence interval radii added for probabilistic safety, a human
simulator to collect human trajectory calibration dataset conditioned on the
planned robot motion, and a conformal prediction module to quantify trajectory
prediction error on the decision-dependent calibration dataset. Crowd
navigation simulation experiments show that ICP strikes a good balance of
performance among navigation efficiency, social awareness, and uncertainty
quantification compared to previous works. ICP generalizes well to navigation
tasks under various crowd densities. The fast runtime and efficient memory
usage make ICP practical for real-world applications. Code is available at
https://github.com/tedhuang96/icp.


---

**[148. [2306.14565] Mitigating Hallucination in Large Multi-Modal Models via Robust
  Instruction Tuning](https://arxiv.org/pdf/2306.14565.pdf)** (2024-03-21)

*Fuxiao Liu, Kevin Lin, Linjie Li, Jianfeng Wang, Yaser Yacoob, Lijuan Wang*

  Despite the promising progress in multi-modal tasks, current large
multi-modal models (LMMs) are prone to hallucinating inconsistent descriptions
with respect to the associated image and human instructions. This paper
addresses this issue by introducing the first large and diverse visual
instruction tuning dataset, named Large-scale Robust Visual (LRV)-Instruction.
Our dataset comprises 400k visual instructions generated by GPT4, covering 16
vision-and-language tasks with open-ended instructions and answers. Unlike
existing studies that primarily focus on positive instruction samples, we
design LRV-Instruction to include both positive and negative instructions for
more robust visual instruction tuning. Our negative instructions are designed
at three semantic levels: (i) Nonexistent Object Manipulation, (ii) Existent
Object Manipulation and (iii) Knowledge Manipulation. To efficiently measure
the hallucination generated by LMMs, we propose GPT4-Assisted Visual
Instruction Evaluation (GAVIE), a stable approach to evaluate visual
instruction tuning like human experts. GAVIE does not require human-annotated
groundtruth answers and can adapt to diverse instruction formats. We conduct
comprehensive experiments to investigate the hallucination of LMMs. Our results
demonstrate existing LMMs exhibit significant hallucinations when presented
with our negative instructions, particularly Existent Object and Knowledge
Manipulation instructions. Moreover, we successfully mitigate hallucination by
finetuning MiniGPT4 and mPLUG-Owl on LRV-Instruction while improving
performance on several public datasets compared to state-of-the-art methods.
Additionally, we observed that a balanced ratio of positive and negative
instances in the training data leads to a more robust model. Code and data are
available at https://github.com/FuxiaoLiu/LRV-Instruction.


---

**[149. [2402.19103] Whispers that Shake Foundations: Analyzing and Mitigating False Premise
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

**[150. [2501.00673] Controlled Causal Hallucinations Can Estimate Phantom Nodes in
  Multiexpert Mixtures of Fuzzy Cognitive Maps](https://arxiv.org/pdf/2501.00673.pdf)** (2025-01-03)

*Akash Kumar Panda, Bart Kosko*

  An adaptive multiexpert mixture of feedback causal models can approximate
missing or phantom nodes in large-scale causal models. The result gives a
scalable form of \emph{big knowledge}. The mixed model approximates a sampled
dynamical system by approximating its main limit-cycle equilibria. Each expert
first draws a fuzzy cognitive map (FCM) with at least one missing causal node
or variable. FCMs are directed signed partial-causality cyclic graphs. They mix
naturally through convex combination to produce a new causal feedback FCM.
Supervised learning helps each expert FCM estimate its phantom node by
comparing the FCM's partial equilibrium with the complete multi-node
equilibrium. Such phantom-node estimation allows partial control over these
causal hallucinations and helps approximate the future trajectory of the
dynamical system. But the approximation can be computationally heavy. Mixing
the tuned expert FCMs gives a practical way to find several phantom nodes and
thereby better approximate the feedback system's true equilibrium behavior.


---

**[151. [2406.01920] CODE: Contrasting Self-generated Description to Combat Hallucination in
  Large Multi-modal Models](https://arxiv.org/pdf/2406.01920.pdf)** (2024-06-05)

*Junho Kim, Hyunjun Kim, Yeonju Kim, Yong Man Ro*

  Large Multi-modal Models (LMMs) have recently demonstrated remarkable
abilities in visual context understanding and coherent response generation.
However, alongside these advancements, the issue of hallucinations has emerged
as a significant challenge, producing erroneous responses that are unrelated to
the visual contents. In this paper, we introduce a novel contrastive-based
decoding method, COuntering DEscription Contrastive Decoding (CODE), which
leverages self-generated descriptions as contrasting references during the
decoding phase of LMMs to address hallucination issues. CODE utilizes the
comprehensive descriptions from model itself as visual counterpart to correct
and improve response alignment with actual visual content. By dynamically
adjusting the information flow and distribution of next-token predictions in
the LMM's vocabulary, CODE enhances the coherence and informativeness of
generated responses. Extensive experiments demonstrate that our method
significantly reduces hallucinations and improves cross-modal consistency
across various benchmarks and cutting-edge LMMs. Our method provides a simple
yet effective decoding strategy that can be integrated to existing LMM
frameworks without additional training.


---

**[152. [2408.13808] Towards Reliable Medical Question Answering: Techniques and Challenges
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

**[153. [2410.06304] FG-PRM: Fine-grained Hallucination Detection and Mitigation in Language
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

**[154. [2501.12975] OnionEval: An Unified Evaluation of Fact-conflicting Hallucination for
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

**[155. [2205.01307] Embedding Hallucination for Few-Shot Language Fine-tuning](https://arxiv.org/pdf/2205.01307.pdf)** (2022-05-04)

*Yiren Jian, Chongyang Gao, Soroush Vosoughi*

  Few-shot language learners adapt knowledge from a pre-trained model to
recognize novel classes from a few-labeled sentences. In such settings,
fine-tuning a pre-trained language model can cause severe over-fitting. In this
paper, we propose an Embedding Hallucination (EmbedHalluc) method, which
generates auxiliary embedding-label pairs to expand the fine-tuning dataset.
The hallucinator is trained by playing an adversarial game with the
discriminator, such that the hallucinated embedding is indiscriminative to the
real ones in the fine-tuning dataset. By training with the extended dataset,
the language learner effectively learns from the diverse hallucinated
embeddings to overcome the over-fitting issue. Experiments demonstrate that our
proposed method is effective in a wide range of language tasks, outperforming
current fine-tuning methods. Further, we show that EmbedHalluc outperforms
other methods that address this over-fitting problem, such as common data
augmentation, semi-supervised pseudo-labeling, and regularization. The code
will be made available at: https://github.com/yiren-jian/EmbedHalluc.


---

**[156. [2407.16908] Generation Constraint Scaling Can Mitigate Hallucination](https://arxiv.org/pdf/2407.16908.pdf)** (2024-07-25)

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

**[157. [2401.11810] Generalization and Informativeness of Conformal Prediction](https://arxiv.org/pdf/2401.11810.pdf)** (2024-01-23)

*Matteo Zecchin, Sangwoo Park, Osvaldo Simeone, Fredrik Hellström*

  The safe integration of machine learning modules in decision-making processes
hinges on their ability to quantify uncertainty. A popular technique to achieve
this goal is conformal prediction (CP), which transforms an arbitrary base
predictor into a set predictor with coverage guarantees. While CP certifies the
predicted set to contain the target quantity with a user-defined tolerance, it
does not provide control over the average size of the predicted sets, i.e.,
over the informativeness of the prediction. In this work, a theoretical
connection is established between the generalization properties of the base
predictor and the informativeness of the resulting CP prediction sets. To this
end, an upper bound is derived on the expected size of the CP set predictor
that builds on generalization error bounds for the base predictor. The derived
upper bound provides insights into the dependence of the average size of the CP
set predictor on the amount of calibration data, the target reliability, and
the generalization performance of the base predictor. The theoretical insights
are validated using simple numerical regression and classification tasks.


---

**[158. [2411.15736] Enhancing Few-Shot Out-of-Distribution Detection with Gradient Aligned
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

**[159. [2401.08358] Hallucination Detection and Hallucination Mitigation: An Investigation](https://arxiv.org/pdf/2401.08358.pdf)** (2024-01-17)

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

**[160. [2503.20134] DRPA-MPPI: Dynamic Repulsive Potential Augmented MPPI for Reactive
  Navigation in Unstructured Environments](https://arxiv.org/pdf/2503.20134.pdf)** (2025-03-27)

*Takahiro Fuke, Masafumi Endo, Kohei Honda, Genya Ishigami*

  Reactive mobile robot navigation in unstructured environments is challenging
when robots encounter unexpected obstacles that invalidate previously planned
trajectories. Model predictive path integral control (MPPI) enables reactive
planning, but still suffers from limited prediction horizons that lead to local
minima traps near obstacles. Current solutions rely on heuristic cost design or
scenario-specific pre-training, which often limits their adaptability to new
environments. We introduce dynamic repulsive potential augmented MPPI
(DRPA-MPPI), which dynamically detects potential entrapments on the predicted
trajectories. Upon detecting local minima, DRPA-MPPI automatically switches
between standard goal-oriented optimization and a modified cost function that
generates repulsive forces away from local minima. Comprehensive testing in
simulated obstacle-rich environments confirms DRPA-MPPI's superior navigation
performance and safety compared to conventional methods with less computational
burden.


---

**[161. [2504.05946] InstructMPC: A Human-LLM-in-the-Loop Framework for Context-Aware Control](https://arxiv.org/pdf/2504.05946.pdf)** (2025-04-15)

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

**[162. [2502.12601] COPU: Conformal Prediction for Uncertainty Quantification in Natural
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

**[163. [2410.12278] Controlled Automatic Task-Specific Synthetic Data Generation for
  Hallucination Detection](https://arxiv.org/pdf/2410.12278.pdf)** (2024-10-17)

*Yong Xie, Karan Aggarwal, Aitzaz Ahmad, Stephen Lau*

  We present a novel approach to automatically generate non-trivial
task-specific synthetic datasets for hallucination detection. Our approach
features a two-step generation-selection pipeline, using hallucination pattern
guidance and a language style alignment during generation. Hallucination
pattern guidance leverages the most important task-specific hallucination
patterns while language style alignment aligns the style of the synthetic
dataset with benchmark text. To obtain robust supervised detectors from
synthetic datasets, we also adopt a data mixture strategy to improve
performance robustness and generalization. Our results on three datasets show
that our generated hallucination text is more closely aligned with
non-hallucinated text versus baselines, to train hallucination detectors with
better generalization. Our hallucination detectors trained on synthetic
datasets outperform in-context-learning (ICL)-based detectors by a large margin
of 32%. Our extensive experiments confirm the benefits of our approach with
cross-task and cross-generator generalization. Our data-mixture-based training
further improves the generalization and robustness of hallucination detection.


---

**[164. [2502.17264] Kandinsky Conformal Prediction: Beyond Class- and Covariate-Conditional
  Coverage](https://arxiv.org/pdf/2502.17264.pdf)** (2025-02-25)

*Konstantina Bairaktari, Jiayun Wu, Zhiwei Steven Wu*

  Conformal prediction is a powerful distribution-free framework for
constructing prediction sets with coverage guarantees. Classical methods, such
as split conformal prediction, provide marginal coverage, ensuring that the
prediction set contains the label of a random test point with a target
probability. However, these guarantees may not hold uniformly across different
subpopulations, leading to disparities in coverage. Prior work has explored
coverage guarantees conditioned on events related to the covariates and label
of the test point. We present Kandinsky conformal prediction, a framework that
significantly expands the scope of conditional coverage guarantees. In contrast
to Mondrian conformal prediction, which restricts its coverage guarantees to
disjoint groups -- reminiscent of the rigid, structured grids of Piet
Mondrian's art -- our framework flexibly handles overlapping and fractional
group memberships defined jointly on covariates and labels, reflecting the
layered, intersecting forms in Wassily Kandinsky's compositions. Our algorithm
unifies and extends existing methods, encompassing covariate-based group
conditional, class conditional, and Mondrian conformal prediction as special
cases, while achieving a minimax-optimal high-probability conditional coverage
bound. Finally, we demonstrate the practicality of our approach through
empirical evaluation on real-world datasets.


---

**[165. [2411.17265] A Topic-level Self-Correctional Approach to Mitigate Hallucinations in
  MLLMs](https://arxiv.org/pdf/2411.17265.pdf)** (2024-12-10)

*Lehan He, Zeren Chen, Zhelun Shi, Tianyu Yu, Jing Shao, Lu Sheng*

  Aligning the behaviors of Multimodal Large Language Models (MLLMs) with human
preferences is crucial for developing robust and trustworthy AI systems. While
recent attempts have employed human experts or powerful auxiliary AI systems to
provide more accurate preference feedback, such as determining the preferable
responses from MLLMs or directly rewriting hallucination-free responses,
extensive resource overhead compromise the scalability of the feedback
collection. In this work, we introduce Topic-level Preference Overwriting
(TPO), a self-correctional approach that guide the model itself to mitigate its
own hallucination at the topic level. Through a deconfounded strategy that
replaces each topic within the response with the best or worst alternatives
generated by the model itself, TPO creates more contrasting pairwise preference
feedback, enhancing the feedback quality without human or proprietary model
intervention. Notably, the experimental results demonstrate proposed TPO
achieves state-of-the-art performance in trustworthiness, significantly
reducing the object hallucinations by 92% and overall hallucinations by 38%.
Code, model and dataset are available now.


---

**[166. [2311.12688] On the Out-of-Distribution Coverage of Combining Split Conformal
  Prediction and Bayesian Deep Learning](https://arxiv.org/pdf/2311.12688.pdf)** (2024-03-08)

*Paul Scemama, Ariel Kapusta*

  Bayesian deep learning and conformal prediction are two methods that have
been used to convey uncertainty and increase safety in machine learning
systems. We focus on combining Bayesian deep learning with split conformal
prediction and how this combination effects out-of-distribution coverage;
particularly in the case of multiclass image classification. We suggest that if
the model is generally underconfident on the calibration set, then the
resultant conformal sets may exhibit worse out-of-distribution coverage
compared to simple predictive credible sets. Conversely, if the model is
overconfident on the calibration set, the use of conformal prediction may
improve out-of-distribution coverage. We evaluate prediction sets as a result
of combining split conformal methods and neural networks trained with (i)
stochastic gradient descent, (ii) deep ensembles, and (iii) mean-field
variational inference. Our results suggest that combining Bayesian deep
learning models with split conformal prediction can, in some cases, cause
unintended consequences such as reducing out-of-distribution coverage.


---

**[167. [2502.03609] Multivariate Conformal Prediction using Optimal Transport](https://arxiv.org/pdf/2502.03609.pdf)** (2025-02-07)

*Michal Klein, Louis Bethune, Eugene Ndiaye, Marco Cuturi*

  Conformal prediction (CP) quantifies the uncertainty of machine learning
models by constructing sets of plausible outputs. These sets are constructed by
leveraging a so-called conformity score, a quantity computed using the input
point of interest, a prediction model, and past observations. CP sets are then
obtained by evaluating the conformity score of all possible outputs, and
selecting them according to the rank of their scores. Due to this ranking step,
most CP approaches rely on a score functions that are univariate. The challenge
in extending these scores to multivariate spaces lies in the fact that no
canonical order for vectors exists. To address this, we leverage a natural
extension of multivariate score ranking based on optimal transport (OT). Our
method, OTCP, offers a principled framework for constructing conformal
prediction sets in multidimensional settings, preserving distribution-free
coverage guarantees with finite data samples. We demonstrate tangible gains in
a benchmark dataset of multivariate regression problems and address
computational \& statistical trade-offs that arise when estimating conformity
scores through OT maps.


---

**[168. [2502.18342] BRIDO: Bringing Democratic Order to Abstractive Summarization](https://arxiv.org/pdf/2502.18342.pdf)** (2025-02-26)

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

**[169. [2403.18051] Supervisory Prompt Training](https://arxiv.org/pdf/2403.18051.pdf)** (2024-03-28)

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
