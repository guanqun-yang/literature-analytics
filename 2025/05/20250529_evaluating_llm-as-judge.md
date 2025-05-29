**[1. [2410.11594] Black-box Uncertainty Quantification Method for LLM-as-a-Judge](https://arxiv.org/pdf/2410.11594.pdf)** (2024-10-16)

*Nico Wagner, Michael Desmond, Rahul Nair, Zahra Ashktorab, Elizabeth M. Daly, Qian Pan, Martín Santillán Cooper, James M. Johnson, Werner Geyer*

  LLM-as-a-Judge is a widely used method for evaluating the performance of
Large Language Models (LLMs) across various tasks. We address the challenge of
quantifying the uncertainty of LLM-as-a-Judge evaluations. While uncertainty
quantification has been well-studied in other domains, applying it effectively
to LLMs poses unique challenges due to their complex decision-making
capabilities and computational demands. In this paper, we introduce a novel
method for quantifying uncertainty designed to enhance the trustworthiness of
LLM-as-a-Judge evaluations. The method quantifies uncertainty by analyzing the
relationships between generated assessments and possible ratings. By
cross-evaluating these relationships and constructing a confusion matrix based
on token probabilities, the method derives labels of high or low uncertainty.
We evaluate our method across multiple benchmarks, demonstrating a strong
correlation between the accuracy of LLM evaluations and the derived uncertainty
scores. Our findings suggest that this method can significantly improve the
reliability and consistency of LLM-as-a-Judge evaluations.


---

**[2. [2411.16594] From Generation to Judgment: Opportunities and Challenges of
  LLM-as-a-judge](https://arxiv.org/pdf/2411.16594.pdf)** (2025-02-07)

*Dawei Li, Bohan Jiang, Liangjie Huang, Alimohammad Beigi, Chengshuai Zhao, Zhen Tan, Amrita Bhattacharjee, Yuxuan Jiang, Canyu Chen, Tianhao Wu, Kai Shu, Lu Cheng, Huan Liu*

  Assessment and evaluation have long been critical challenges in artificial
intelligence (AI) and natural language processing (NLP). However, traditional
methods, whether matching-based or embedding-based, often fall short of judging
subtle attributes and delivering satisfactory results. Recent advancements in
Large Language Models (LLMs) inspire the "LLM-as-a-judge" paradigm, where LLMs
are leveraged to perform scoring, ranking, or selection across various tasks
and applications. This paper provides a comprehensive survey of LLM-based
judgment and assessment, offering an in-depth overview to advance this emerging
field. We begin by giving detailed definitions from both input and output
perspectives. Then we introduce a comprehensive taxonomy to explore
LLM-as-a-judge from three dimensions: what to judge, how to judge and where to
judge. Finally, we compile benchmarks for evaluating LLM-as-a-judge and
highlight key challenges and promising directions, aiming to provide valuable
insights and inspire future research in this promising research area. Paper
list and more resources about LLM-as-a-judge can be found at
https://github.com/llm-as-a-judge/Awesome-LLM-as-a-judge and
https://llm-as-a-judge.github.io.


---

**[3. [2501.00274] LLM-Rubric: A Multidimensional, Calibrated Approach to Automated
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

**[4. [2404.11086] ViLLM-Eval: A Comprehensive Evaluation Suite for Vietnamese Large
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

**[5. [2502.18817] Judge as A Judge: Improving the Evaluation of Retrieval-Augmented
  Generation through the Judge-Consistency of Large Language Models](https://arxiv.org/pdf/2502.18817.pdf)** (2025-02-27)

*Shuliang Liu, Xinze Li, Zhenghao Liu, Yukun Yan, Cheng Yang, Zheni Zeng, Zhiyuan Liu, Maosong Sun, Ge Yu*

  Retrieval-Augmented Generation (RAG) has proven its effectiveness in
alleviating hallucinations for Large Language Models (LLMs). However, existing
automated evaluation metrics cannot fairly evaluate the outputs generated by
RAG models during training and evaluation. LLM-based judgment models provide
the potential to produce high-quality judgments, but they are highly sensitive
to evaluation prompts, leading to inconsistencies when judging the output of
RAG models. This paper introduces the Judge-Consistency (ConsJudge) method,
which aims to enhance LLMs to generate more accurate evaluations for RAG
models. Specifically, ConsJudge prompts LLMs to generate different judgments
based on various combinations of judgment dimensions, utilize the
judge-consistency to evaluate these judgments and select the accepted and
rejected judgments for DPO training. Our experiments show that ConsJudge can
effectively provide more accurate judgments for optimizing RAG models across
various RAG models and datasets. Further analysis reveals that judgments
generated by ConsJudge have a high agreement with the superior LLM. All codes
are available at https://github.com/OpenBMB/ConsJudge.


---

**[6. [2407.03479] Human-Centered Design Recommendations for LLM-as-a-Judge](https://arxiv.org/pdf/2407.03479.pdf)** (2024-07-08)

*Qian Pan, Zahra Ashktorab, Michael Desmond, Martin Santillan Cooper, James Johnson, Rahul Nair, Elizabeth Daly, Werner Geyer*

  Traditional reference-based metrics, such as BLEU and ROUGE, are less
effective for assessing outputs from Large Language Models (LLMs) that produce
highly creative or superior-quality text, or in situations where reference
outputs are unavailable. While human evaluation remains an option, it is costly
and difficult to scale. Recent work using LLMs as evaluators (LLM-as-a-judge)
is promising, but trust and reliability remain a significant concern.
Integrating human input is crucial to ensure criteria used to evaluate are
aligned with the human's intent, and evaluations are robust and consistent.
This paper presents a user study of a design exploration called EvaluLLM, that
enables users to leverage LLMs as customizable judges, promoting human
involvement to balance trust and cost-saving potential with caution. Through
interviews with eight domain experts, we identified the need for assistance in
developing effective evaluation criteria aligning the LLM-as-a-judge with
practitioners' preferences and expectations. We offer findings and design
recommendations to optimize human-assisted LLM-as-judge systems.


---

**[7. [2408.10718] CodeJudge-Eval: Can Large Language Models be Good Judges in Code
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

**[8. [2306.05685] Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/pdf/2306.05685.pdf)** (2023-12-27)

*Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica*

  Evaluating large language model (LLM) based chat assistants is challenging
due to their broad capabilities and the inadequacy of existing benchmarks in
measuring human preferences. To address this, we explore using strong LLMs as
judges to evaluate these models on more open-ended questions. We examine the
usage and limitations of LLM-as-a-judge, including position, verbosity, and
self-enhancement biases, as well as limited reasoning ability, and propose
solutions to mitigate some of them. We then verify the agreement between LLM
judges and human preferences by introducing two benchmarks: MT-bench, a
multi-turn question set; and Chatbot Arena, a crowdsourced battle platform. Our
results reveal that strong LLM judges like GPT-4 can match both controlled and
crowdsourced human preferences well, achieving over 80% agreement, the same
level of agreement between humans. Hence, LLM-as-a-judge is a scalable and
explainable way to approximate human preferences, which are otherwise very
expensive to obtain. Additionally, we show our benchmark and traditional
benchmarks complement each other by evaluating several variants of LLaMA and
Vicuna. The MT-bench questions, 3K expert votes, and 30K conversations with
human preferences are publicly available at
https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge.


---

**[9. [2305.13711] LLM-Eval: Unified Multi-Dimensional Automatic Evaluation for Open-Domain
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

**[10. [2406.17663] LLM-ARC: Enhancing LLMs with an Automated Reasoning Critic](https://arxiv.org/pdf/2406.17663.pdf)** (2024-07-22)

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

**[11. [2412.00543] Evaluating the Consistency of LLM Evaluators](https://arxiv.org/pdf/2412.00543.pdf)** (2024-12-03)

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

**[12. [2410.12265] An Automatic and Cost-Efficient Peer-Review Framework for Language
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

**[13. [2503.03064] Improving LLM-as-a-Judge Inference with the Judgment Distribution](https://arxiv.org/pdf/2503.03064.pdf)** (2025-03-06)

*Victor Wang, Michael J. Q. Zhang, Eunsol Choi*

  Using language models to scalably approximate human preferences on text
quality (LLM-as-a-judge) has become a standard practice applicable to many
tasks. A judgment is often extracted from the judge's textual output alone,
typically with greedy decoding. However, LLM judges naturally provide
distributions over judgment tokens, inviting a breadth of inference methods for
extracting fine-grained preferences. We find that taking the mean of the
judgment distribution consistently outperforms taking the mode (i.e. greedy
decoding) in all evaluation settings (i.e. pointwise, pairwise, and listwise).
We further explore novel methods of deriving preferences from judgment
distributions, and find that methods incorporating risk aversion often improve
performance. Lastly, we analyze LLM-as-a-judge paired with chain-of-thought
(CoT) prompting, showing that CoT can collapse the spread of the judgment
distribution, often harming performance. Our findings suggest leveraging
distributional output can improve LLM-as-a-judge, as opposed to using the text
interface alone.


---

**[14. [2309.11325] DISC-LawLLM: Fine-tuning Large Language Models for Intelligent Legal
  Services](https://arxiv.org/pdf/2309.11325.pdf)** (2023-09-26)

*Shengbin Yue, Wei Chen, Siyuan Wang, Bingxuan Li, Chenchen Shen, Shujun Liu, Yuxuan Zhou, Yao Xiao, Song Yun, Xuanjing Huang, Zhongyu Wei*

  We propose DISC-LawLLM, an intelligent legal system utilizing large language
models (LLMs) to provide a wide range of legal services. We adopt legal
syllogism prompting strategies to construct supervised fine-tuning datasets in
the Chinese Judicial domain and fine-tune LLMs with legal reasoning capability.
We augment LLMs with a retrieval module to enhance models' ability to access
and utilize external legal knowledge. A comprehensive legal benchmark,
DISC-Law-Eval, is presented to evaluate intelligent legal systems from both
objective and subjective dimensions. Quantitative and qualitative results on
DISC-Law-Eval demonstrate the effectiveness of our system in serving various
users across diverse legal scenarios. The detailed resources are available at
https://github.com/FudanDISC/DISC-LawLLM.


---

**[15. [2312.07910] PromptBench: A Unified Library for Evaluation of Large Language Models](https://arxiv.org/pdf/2312.07910.pdf)** (2024-08-21)

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

**[16. [2504.11972] LLM-as-a-Judge: Reassessing the Performance of LLMs in Extractive QA](https://arxiv.org/pdf/2504.11972.pdf)** (2025-04-17)

*Xanh Ho, Jiahao Huang, Florian Boudin, Akiko Aizawa*

  Extractive reading comprehension question answering (QA) datasets are
typically evaluated using Exact Match (EM) and F1-score, but these metrics
often fail to fully capture model performance. With the success of large
language models (LLMs), they have been employed in various tasks, including
serving as judges (LLM-as-a-judge). In this paper, we reassess the performance
of QA models using LLM-as-a-judge across four reading comprehension QA
datasets. We examine different families of LLMs and various answer types to
evaluate the effectiveness of LLM-as-a-judge in these tasks. Our results show
that LLM-as-a-judge is highly correlated with human judgments and can replace
traditional EM/F1 metrics. By using LLM-as-a-judge, the correlation with human
judgments improves significantly, from 0.17 (EM) and 0.36 (F1-score) to 0.85.
These findings confirm that EM and F1 metrics underestimate the true
performance of the QA models. While LLM-as-a-judge is not perfect for more
difficult answer types (e.g., job), it still outperforms EM/F1, and we observe
no bias issues, such as self-preference, when the same model is used for both
the QA and judgment tasks.


---

**[17. [2408.08781] Evaluating the Evaluator: Measuring LLMs' Adherence to Task Evaluation
  Instructions](https://arxiv.org/pdf/2408.08781.pdf)** (2024-08-19)

*Bhuvanashree Murugadoss, Christian Poelitz, Ian Drosos, Vu Le, Nick McKenna, Carina Suzana Negreanu, Chris Parnin, Advait Sarkar*

  LLMs-as-a-judge is a recently popularized method which replaces human
judgements in task evaluation (Zheng et al. 2024) with automatic evaluation
using LLMs. Due to widespread use of RLHF (Reinforcement Learning from Human
Feedback), state-of-the-art LLMs like GPT4 and Llama3 are expected to have
strong alignment with human preferences when prompted for a quality judgement,
such as the coherence of a text. While this seems beneficial, it is not clear
whether the assessments by an LLM-as-a-judge constitute only an evaluation
based on the instructions in the prompts, or reflect its preference for
high-quality data similar to its fine-tune data. To investigate how much
influence prompting the LLMs-as-a-judge has on the alignment of AI judgements
to human judgements, we analyze prompts with increasing levels of instructions
about the target quality of an evaluation, for several LLMs-as-a-judge.
Further, we compare to a prompt-free method using model perplexity as a quality
measure instead. We aggregate a taxonomy of quality criteria commonly used
across state-of-the-art evaluations with LLMs and provide this as a rigorous
benchmark of models as judges. Overall, we show that the LLMs-as-a-judge
benefit only little from highly detailed instructions in prompts and that
perplexity can sometimes align better with human judgements than prompting,
especially on textual quality.


---

**[18. [2409.04168] From Calculation to Adjudication: Examining LLM judges on Mathematical
  Reasoning Tasks](https://arxiv.org/pdf/2409.04168.pdf)** (2024-09-09)

*Andreas Stephan, Dawei Zhu, Matthias Aßenmacher, Xiaoyu Shen, Benjamin Roth*

  To reduce the need for human annotations, large language models (LLMs) have
been proposed as judges of the quality of other candidate models. LLM judges
are typically evaluated by measuring the correlation with human judgments on
generation tasks such as summarization or machine translation. In contrast, we
study LLM judges on mathematical reasoning tasks. These tasks require
multi-step reasoning, and the correctness of their solutions is verifiable,
enabling a more objective evaluation. We perform a detailed performance
analysis and find that the used judges are mostly unable to improve task
performance but are able to pick the better model. Our analysis uncovers a
strong correlation between judgment performance and the candidate model task
performance. We observe that judges tend to choose the model of higher quality
even if its answer is incorrect. Further, we show that it is possible to use
statistics, such as the task performances of the individual models, to predict
judgment performance. In an ablation, we either swap or mask the candidate
answers and observe that judges often keep the original judgment, providing
evidence that judges incorporate writing style in their judgments. In summary,
we find that regularities in the judgments are quantifiable using statistical
measures and provide various angles on exploiting them.


---

**[19. [2503.02246] From Code to Courtroom: LLMs as the New Software Judges](https://arxiv.org/pdf/2503.02246.pdf)** (2025-03-05)

*Junda He, Jieke Shi, Terry Yue Zhuo, Christoph Treude, Jiamou Sun, Zhenchang Xing, Xiaoning Du, David Lo*

  Recently, Large Language Models (LLMs) have been increasingly used to
automate SE tasks such as code generation and summarization. However,
evaluating the quality of LLM-generated software artifacts remains challenging.
Human evaluation, while effective, is very costly and time-consuming.
Traditional automated metrics like BLEU rely on high-quality references and
struggle to capture nuanced aspects of software quality, such as readability
and usefulness. In response, the LLM-as-a-Judge paradigm, which employs LLMs
for automated evaluation, has emerged. Given that LLMs are typically trained to
align with human judgment and possess strong coding abilities and reasoning
skills, they hold promise as cost-effective and scalable surrogates for human
evaluators. Nevertheless, LLM-as-a-Judge research in the SE community is still
in its early stages, with many breakthroughs needed.
  This forward-looking SE 2030 paper aims to steer the research community
toward advancing LLM-as-a-Judge for evaluating LLMgenerated software artifacts,
while also sharing potential research paths to achieve this goal. We provide a
literature review of existing SE studies on LLM-as-a-Judge and envision these
frameworks as reliable, robust, and scalable human surrogates capable of
evaluating software artifacts with consistent, multi-faceted assessments by
2030 and beyond. To validate this vision, we analyze the limitations of current
studies, identify key research gaps, and outline a detailed roadmap to guide
future developments of LLM-as-a-Judge in software engineering. While not
intended to be a definitive guide, our work aims to foster further research and
adoption of LLM-as-a-Judge frameworks within the SE community, ultimately
improving the effectiveness and scalability of software artifact evaluation
methods.


---

**[20. [2503.04381] TRACT: Regression-Aware Fine-tuning Meets Chain-of-Thought Reasoning for
  LLM-as-a-Judge](https://arxiv.org/pdf/2503.04381.pdf)** (2025-03-07)

*Cheng-Han Chiang, Hung-yi Lee, Michal Lukasik*

  The LLM-as-a-judge paradigm uses large language models (LLMs) for automated
text evaluation, where a numerical assessment is assigned by an LLM to the
input text following scoring rubrics. Existing methods for LLM-as-a-judge use
cross-entropy (CE) loss for fine-tuning, which neglects the numeric nature of
score prediction. Recent work addresses numerical prediction limitations of LLM
fine-tuning through regression-aware fine-tuning, which, however, does not
consider chain-of-thought (CoT) reasoning for score prediction. In this paper,
we introduce TRACT (Two-stage Regression-Aware fine-tuning with CoT), a method
combining CoT reasoning with regression-aware training. TRACT consists of two
stages: first, seed LLM is fine-tuned to generate CoTs, which serve as
supervision for the second stage fine-tuning. The training objective of TRACT
combines the CE loss for learning the CoT reasoning capabilities, and the
regression-aware loss for the score prediction. Experiments across four
LLM-as-a-judge datasets and two LLMs show that TRACT significantly outperforms
existing methods. Extensive ablation studies validate the importance of each
component in TRACT.


---

**[21. [2502.15094] Judging It, Washing It: Scoring and Greenwashing Corporate Climate
  Disclosures using Large Language Models](https://arxiv.org/pdf/2502.15094.pdf)** (2025-02-24)

*Marianne Chuang, Gabriel Chuang, Cheryl Chuang, John Chuang*

  We study the use of large language models (LLMs) to both evaluate and
greenwash corporate climate disclosures. First, we investigate the use of the
LLM-as-a-Judge (LLMJ) methodology for scoring company-submitted reports on
emissions reduction targets and progress. Second, we probe the behavior of an
LLM when it is prompted to greenwash a response subject to accuracy and length
constraints. Finally, we test the robustness of the LLMJ methodology against
responses that may be greenwashed using an LLM. We find that two LLMJ scoring
systems, numerical rating and pairwise comparison, are effective in
distinguishing high-performing companies from others, with the pairwise
comparison system showing greater robustness against LLM-greenwashed responses.


---

**[22. [2412.14140] GLIDER: Grading LLM Interactions and Decisions using Explainable Ranking](https://arxiv.org/pdf/2412.14140.pdf)** (2024-12-24)

*Darshan Deshpande, Selvan Sunitha Ravi, Sky CH-Wang, Bartosz Mielczarek, Anand Kannappan, Rebecca Qian*

  The LLM-as-judge paradigm is increasingly being adopted for automated
evaluation of model outputs. While LLM judges have shown promise on constrained
evaluation tasks, closed source LLMs display critical shortcomings when
deployed in real world applications due to challenges of fine grained metrics
and explainability, while task specific evaluation models lack cross-domain
generalization. We introduce GLIDER, a powerful 3B evaluator LLM that can score
any text input and associated context on arbitrary user defined criteria.
GLIDER shows higher Pearson's correlation than GPT-4o on FLASK and greatly
outperforms prior evaluation models, achieving comparable performance to LLMs
17x its size. GLIDER supports fine-grained scoring, multilingual reasoning,
span highlighting and was trained on 685 domains and 183 criteria. Extensive
qualitative analysis shows that GLIDER scores are highly correlated with human
judgments, with 91.3% human agreement. We have open-sourced GLIDER to
facilitate future research.


---

**[23. [2412.05579] LLMs-as-Judges: A Comprehensive Survey on LLM-based Evaluation Methods](https://arxiv.org/pdf/2412.05579.pdf)** (2024-12-11)

*Haitao Li, Qian Dong, Junjie Chen, Huixue Su, Yujia Zhou, Qingyao Ai, Ziyi Ye, Yiqun Liu*

  The rapid advancement of Large Language Models (LLMs) has driven their
expanding application across various fields. One of the most promising
applications is their role as evaluators based on natural language responses,
referred to as ''LLMs-as-judges''. This framework has attracted growing
attention from both academia and industry due to their excellent effectiveness,
ability to generalize across tasks, and interpretability in the form of natural
language. This paper presents a comprehensive survey of the LLMs-as-judges
paradigm from five key perspectives: Functionality, Methodology, Applications,
Meta-evaluation, and Limitations. We begin by providing a systematic definition
of LLMs-as-Judges and introduce their functionality (Why use LLM judges?). Then
we address methodology to construct an evaluation system with LLMs (How to use
LLM judges?). Additionally, we investigate the potential domains for their
application (Where to use LLM judges?) and discuss methods for evaluating them
in various contexts (How to evaluate LLM judges?). Finally, we provide a
detailed analysis of the limitations of LLM judges and discuss potential future
directions. Through a structured and comprehensive analysis, we aim aims to
provide insights on the development and application of LLMs-as-judges in both
research and practice. We will continue to maintain the relevant resource list
at https://github.com/CSHaitao/Awesome-LLMs-as-Judges.


---

**[24. [2411.15594] A Survey on LLM-as-a-Judge](https://arxiv.org/pdf/2411.15594.pdf)** (2025-03-11)

*Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, Saizhuo Wang, Kun Zhang, Yuanzhuo Wang, Wen Gao, Lionel Ni, Jian Guo*

  Accurate and consistent evaluation is crucial for decision-making across
numerous fields, yet it remains a challenging task due to inherent
subjectivity, variability, and scale. Large Language Models (LLMs) have
achieved remarkable success across diverse domains, leading to the emergence of
"LLM-as-a-Judge," where LLMs are employed as evaluators for complex tasks. With
their ability to process diverse data types and provide scalable,
cost-effective, and consistent assessments, LLMs present a compelling
alternative to traditional expert-driven evaluations. However, ensuring the
reliability of LLM-as-a-Judge systems remains a significant challenge that
requires careful design and standardization. This paper provides a
comprehensive survey of LLM-as-a-Judge, addressing the core question: How can
reliable LLM-as-a-Judge systems be built? We explore strategies to enhance
reliability, including improving consistency, mitigating biases, and adapting
to diverse assessment scenarios. Additionally, we propose methodologies for
evaluating the reliability of LLM-as-a-Judge systems, supported by a novel
benchmark designed for this purpose. To advance the development and real-world
deployment of LLM-as-a-Judge systems, we also discussed practical applications,
challenges, and future directions. This survey serves as a foundational
reference for researchers and practitioners in this rapidly evolving field.


---

**[25. [2502.01534] Preference Leakage: A Contamination Problem in LLM-as-a-judge](https://arxiv.org/pdf/2502.01534.pdf)** (2025-02-04)

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

**[26. [2410.16848] ETHIC: Evaluating Large Language Models on Long-Context Tasks with High
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

**[27. [2502.16268] ThinkBench: Dynamic Out-of-Distribution Evaluation for Robust LLM
  Reasoning](https://arxiv.org/pdf/2502.16268.pdf)** (2025-02-25)

*Shulin Huang, Linyi Yang, Yan Song, Shuang Chen, Leyang Cui, Ziyu Wan, Qingcheng Zeng, Ying Wen, Kun Shao, Weinan Zhang, Jun Wang, Yue Zhang*

  Evaluating large language models (LLMs) poses significant challenges,
particularly due to issues of data contamination and the leakage of correct
answers. To address these challenges, we introduce ThinkBench, a novel
evaluation framework designed to evaluate LLMs' reasoning capability robustly.
ThinkBench proposes a dynamic data generation method for constructing
out-of-distribution (OOD) datasets and offers an OOD dataset that contains
2,912 samples drawn from reasoning tasks. ThinkBench unifies the evaluation of
reasoning models and non-reasoning models. We evaluate 16 LLMs and 4 PRMs under
identical experimental conditions and show that most of the LLMs' performance
are far from robust and they face a certain level of data leakage. By
dynamically generating OOD datasets, ThinkBench effectively provides a reliable
evaluation of LLMs and reduces the impact of data contamination.


---

**[28. [2502.06193] Can LLMs Replace Human Evaluators? An Empirical Study of LLM-as-a-Judge
  in Software Engineering](https://arxiv.org/pdf/2502.06193.pdf)** (2025-04-11)

*Ruiqi Wang, Jiyu Guo, Cuiyun Gao, Guodong Fan, Chun Yong Chong, Xin Xia*

  Recently, large language models (LLMs) have been deployed to tackle various
software engineering (SE) tasks like code generation, significantly advancing
the automation of SE tasks. However, assessing the quality of these
LLM-generated code and text remains challenging. The commonly used Pass@k
metric necessitates extensive unit tests and configured environments, demands a
high labor cost, and is not suitable for evaluating LLM-generated text.
Conventional metrics like BLEU, which measure only lexical rather than semantic
similarity, have also come under scrutiny. In response, a new trend has emerged
to employ LLMs for automated evaluation, known as LLM-as-a-judge. These
LLM-as-a-judge methods are claimed to better mimic human assessment than
conventional metrics without relying on high-quality reference answers.
Nevertheless, their exact human alignment in SE tasks remains unexplored. In
this paper, we empirically explore LLM-as-a-judge methods for evaluating SE
tasks, focusing on their alignment with human judgments. We select seven
LLM-as-a-judge methods that utilize general-purpose LLMs, alongside two LLMs
specifically fine-tuned for evaluation. After generating and manually scoring
LLM responses on three recent SE datasets of code translation, code generation,
and code summarization, we then prompt these methods to evaluate each response.
Finally, we compare the scores generated by these methods with human
evaluation. The results indicate that output-based methods reach the highest
Pearson correlation of 81.32 and 68.51 with human scores in code translation
and generation, achieving near-human evaluation, noticeably outperforming
ChrF++, one of the best conventional metrics, at 34.23 and 64.92. Such
output-based methods prompt LLMs to output judgments directly, and exhibit more
balanced score distributions that resemble human score patterns. Finally, we
provide...


---

**[29. [2410.12784] JudgeBench: A Benchmark for Evaluating LLM-based Judges](https://arxiv.org/pdf/2410.12784.pdf)** (2025-04-08)

*Sijun Tan, Siyuan Zhuang, Kyle Montgomery, William Y. Tang, Alejandro Cuadron, Chenguang Wang, Raluca Ada Popa, Ion Stoica*

  LLM-based judges have emerged as a scalable alternative to human evaluation
and are increasingly used to assess, compare, and improve models. However, the
reliability of LLM-based judges themselves is rarely scrutinized. As LLMs
become more advanced, their responses grow more sophisticated, requiring
stronger judges to evaluate them. Existing benchmarks primarily focus on a
judge's alignment with human preferences, but often fail to account for more
challenging tasks where crowdsourced human preference is a poor indicator of
factual and logical correctness. To address this, we propose a novel evaluation
framework to objectively evaluate LLM-based judges. Based on this framework, we
propose JudgeBench, a benchmark for evaluating LLM-based judges on challenging
response pairs spanning knowledge, reasoning, math, and coding. JudgeBench
leverages a novel pipeline for converting existing difficult datasets into
challenging response pairs with preference labels reflecting objective
correctness. Our comprehensive evaluation on a collection of prompted judges,
fine-tuned judges, multi-agent judges, and reward models shows that JudgeBench
poses a significantly greater challenge than previous benchmarks, with many
strong models (e.g., GPT-4o) performing just slightly better than random
guessing. Overall, JudgeBench offers a reliable platform for assessing
increasingly advanced LLM-based judges. Data and code are available at
https://github.com/ScalerLab/JudgeBench.


---

**[30. [2410.20266] Limitations of the LLM-as-a-Judge Approach for Evaluating LLM Outputs in
  Expert Knowledge Tasks](https://arxiv.org/pdf/2410.20266.pdf)** (2024-10-29)

*Annalisa Szymanski, Noah Ziems, Heather A. Eicher-Miller, Toby Jia-Jun Li, Meng Jiang, Ronald A. Metoyer*

  The potential of using Large Language Models (LLMs) themselves to evaluate
LLM outputs offers a promising method for assessing model performance across
various contexts. Previous research indicates that LLM-as-a-judge exhibits a
strong correlation with human judges in the context of general instruction
following. However, for instructions that require specialized knowledge, the
validity of using LLMs as judges remains uncertain. In our study, we applied a
mixed-methods approach, conducting pairwise comparisons in which both subject
matter experts (SMEs) and LLMs evaluated outputs from domain-specific tasks. We
focused on two distinct fields: dietetics, with registered dietitian experts,
and mental health, with clinical psychologist experts. Our results showed that
SMEs agreed with LLM judges 68% of the time in the dietetics domain and 64% in
mental health when evaluating overall preference. Additionally, the results
indicated variations in SME-LLM agreement across domain-specific aspect
questions. Our findings emphasize the importance of keeping human experts in
the evaluation process, as LLMs alone may not provide the depth of
understanding required for complex, knowledge specific tasks. We also explore
the implications of LLM evaluations across different domains and discuss how
these insights can inform the design of evaluation workflows that ensure better
alignment between human experts and LLMs in interactive systems.


---

**[31. [2403.02839] An Empirical Study of LLM-as-a-Judge for LLM Evaluation: Fine-tuned
  Judge Model is not a General Substitute for GPT-4](https://arxiv.org/pdf/2403.02839.pdf)** (2024-11-06)

*Hui Huang, Yingqi Qu, Xingyuan Bu, Hongli Zhou, Jing Liu, Muyun Yang, Bing Xu, Tiejun Zhao*

  Recently, there has been a growing trend of utilizing Large Language Model
(LLM) to evaluate the quality of other LLMs. Many studies have employed
proprietary close-sourced models, especially GPT-4, as the evaluator.
Alternatively, other works have fine-tuned judge models based on open-source
LLMs as the evaluator. While the fine-tuned judge models are claimed to achieve
comparable evaluation capability with GPT-4, in this work, we conduct an
empirical study of judge models. Our findings indicate that although the
fine-tuned judge models achieve high performance on in-domain test sets, even
surpassing GPT-4, they underperform GPT-4 across several dimensions, including
generalizability, fairness, aspect-specific evaluation, and scalability. We
also reveal that the fine-tuned judge model inherently operates as a
task-specific classifier, consequently imposing the limitations. Finally, we
introduce a integrated method, leveraging GPT-4 to compensate for the
limitations and improve the fine-tuned judges. Experiment results show our
method achieves accuracy on par with GPT-4 with only 50% of the API expense.


---

**[32. [2504.07440] Revisiting LLM Evaluation through Mechanism Interpretability: a New
  Metric and Model Utility Law](https://arxiv.org/pdf/2504.07440.pdf)** (2025-04-11)

*Yixin Cao, Jiahao Ying, Yaoning Wang, Xipeng Qiu, Xuanjing Huang, Yugang Jiang*

  Large Language Models (LLMs) have become indispensable across academia,
industry, and daily applications, yet current evaluation methods struggle to
keep pace with their rapid development. In this paper, we analyze the core
limitations of traditional evaluation pipelines and propose a novel metric, the
Model Utilization Index (MUI), which introduces mechanism interpretability
techniques to complement traditional performance metrics. MUI quantifies the
extent to which a model leverages its capabilities to complete tasks. The core
idea is that to assess an LLM's overall ability, we must evaluate not only its
task performance but also the effort expended to achieve the outcome. Our
extensive experiments reveal an inverse relationship between MUI and
performance, from which we deduce a common trend observed in popular LLMs,
which we term the Utility Law. Based on this, we derive four corollaries that
address key challenges, including training judgement, the issue of data
contamination, fairness in model comparison, and data diversity. We hope that
our survey, novel metric, and utility law will foster mutual advancement in
both evaluation and mechanism interpretability. Our code can be found at
https://github.com/ALEX-nlp/MUI-Eva.


---

**[33. [2501.06211] FLAME: Financial Large-Language Model Assessment and Metrics Evaluation](https://arxiv.org/pdf/2501.06211.pdf)** (2025-01-14)

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

**[34. [2312.14033] T-Eval: Evaluating the Tool Utilization Capability of Large Language
  Models Step by Step](https://arxiv.org/pdf/2312.14033.pdf)** (2024-01-17)

*Zehui Chen, Weihua Du, Wenwei Zhang, Kuikun Liu, Jiangning Liu, Miao Zheng, Jingming Zhuo, Songyang Zhang, Dahua Lin, Kai Chen, Feng Zhao*

  Large language models (LLM) have achieved remarkable performance on various
NLP tasks and are augmented by tools for broader applications. Yet, how to
evaluate and analyze the tool-utilization capability of LLMs is still
under-explored. In contrast to previous works that evaluate models
holistically, we comprehensively decompose the tool utilization into multiple
sub-processes, including instruction following, planning, reasoning, retrieval,
understanding, and review. Based on that, we further introduce T-Eval to
evaluate the tool utilization capability step by step. T-Eval disentangles the
tool utilization evaluation into several sub-domains along model capabilities,
facilitating the inner understanding of both holistic and isolated competency
of LLMs. We conduct extensive experiments on T-Eval and in-depth analysis of
various LLMs. T-Eval not only exhibits consistency with the outcome-oriented
evaluation but also provides a more fine-grained analysis of the capabilities
of LLMs, providing a new perspective in LLM evaluation on tool-utilization
ability. The benchmark will be available at
https://github.com/open-compass/T-Eval.


---

**[35. [2503.05142] RocketEval: Efficient Automated LLM Evaluation via Grading Checklist](https://arxiv.org/pdf/2503.05142.pdf)** (2025-03-10)

*Tianjun Wei, Wei Wen, Ruizhi Qiao, Xing Sun, Jianghong Ma*

  Evaluating large language models (LLMs) in diverse and challenging scenarios
is essential to align them with human preferences. To mitigate the prohibitive
costs associated with human evaluations, utilizing a powerful LLM as a judge
has emerged as a favored approach. Nevertheless, this methodology encounters
several challenges, including substantial expenses, concerns regarding privacy
and security, and reproducibility. In this paper, we propose a straightforward,
replicable, and accurate automated evaluation method by leveraging a
lightweight LLM as the judge, named RocketEval. Initially, we identify that the
performance disparity between lightweight and powerful LLMs in evaluation tasks
primarily stems from their ability to conduct comprehensive analyses, which is
not easily enhanced through techniques such as chain-of-thought reasoning. By
reframing the evaluation task as a multi-faceted Q&A using an instance-specific
checklist, we demonstrate that the limited judgment accuracy of lightweight
LLMs is largely attributes to high uncertainty and positional bias. To address
these challenges, we introduce an automated evaluation process grounded in
checklist grading, which is designed to accommodate a variety of scenarios and
questions. This process encompasses the creation of checklists, the grading of
these checklists by lightweight LLMs, and the reweighting of checklist items to
align with the supervised annotations. Our experiments carried out on the
automated evaluation benchmarks, MT-Bench and WildBench datasets, reveal that
RocketEval, when using Gemma-2-2B as the judge, achieves a high correlation
(0.965) with human preferences, which is comparable to GPT-4o. Moreover,
RocketEval provides a cost reduction exceeding 50-fold for large-scale
evaluation and comparison scenarios. Our code is available at
https://github.com/Joinn99/RocketEval-ICLR .


---

**[36. [2403.17710] Optimization-based Prompt Injection Attack to LLM-as-a-Judge](https://arxiv.org/pdf/2403.17710.pdf)** (2025-03-04)

*Jiawen Shi, Zenghui Yuan, Yinuo Liu, Yue Huang, Pan Zhou, Lichao Sun, Neil Zhenqiang Gong*

  LLM-as-a-Judge uses a large language model (LLM) to select the best response
from a set of candidates for a given question. LLM-as-a-Judge has many
applications such as LLM-powered search, reinforcement learning with AI
feedback (RLAIF), and tool selection. In this work, we propose JudgeDeceiver,
an optimization-based prompt injection attack to LLM-as-a-Judge. JudgeDeceiver
injects a carefully crafted sequence into an attacker-controlled candidate
response such that LLM-as-a-Judge selects the candidate response for an
attacker-chosen question no matter what other candidate responses are.
Specifically, we formulate finding such sequence as an optimization problem and
propose a gradient based method to approximately solve it. Our extensive
evaluation shows that JudgeDeceive is highly effective, and is much more
effective than existing prompt injection attacks that manually craft the
injected sequences and jailbreak attacks when extended to our problem. We also
show the effectiveness of JudgeDeceiver in three case studies, i.e.,
LLM-powered search, RLAIF, and tool selection. Moreover, we consider defenses
including known-answer detection, perplexity detection, and perplexity windowed
detection. Our results show these defenses are insufficient, highlighting the
urgent need for developing new defense strategies. Our implementation is
available at this repository: https://github.com/ShiJiawenwen/JudgeDeceiver.


---

**[37. [2503.08542] DAFE: LLM-Based Evaluation Through Dynamic Arbitration for Free-Form
  Question-Answering](https://arxiv.org/pdf/2503.08542.pdf)** (2025-03-12)

*Sher Badshah, Hassan Sajjad*

  Evaluating Large Language Models (LLMs) free-form generated responses remains
a challenge due to their diverse and open-ended nature. Traditional supervised
signal-based automatic metrics fail to capture semantic equivalence or handle
the variability of open-ended responses, while human evaluation, though
reliable, is resource-intensive. Leveraging LLMs as evaluators offers a
promising alternative due to their strong language understanding and
instruction-following capabilities. Taking advantage of these capabilities, we
propose the Dynamic Arbitration Framework for Evaluation (DAFE), which employs
two primary LLM-as-judges and engages a third arbitrator only in cases of
disagreements. This selective arbitration prioritizes evaluation reliability
while reducing unnecessary computational demands compared to conventional
majority voting. DAFE utilizes task-specific reference answers with dynamic
arbitration to enhance judgment accuracy, resulting in significant improvements
in evaluation metrics such as Macro F1 and Cohen's Kappa. Through experiments,
including a comprehensive human evaluation, we demonstrate DAFE's ability to
provide consistent, scalable, and resource-efficient assessments, establishing
it as a robust framework for evaluating free-form model outputs.


---

**[38. [2406.12319] The Comparative Trap: Pairwise Comparisons Amplifies Biased Preferences
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

**[39. [2406.11044] Evaluating the Performance of Large Language Models via Debates](https://arxiv.org/pdf/2406.11044.pdf)** (2025-02-11)

*Behrad Moniri, Hamed Hassani, Edgar Dobriban*

  Large Language Models (LLMs) are rapidly evolving and impacting various
fields, necessitating the development of effective methods to evaluate and
compare their performance. Most current approaches for performance evaluation
are either based on fixed, domain-specific questions that lack the flexibility
required in many real-world applications, or rely on human input, making them
unscalable. To address these issues, we propose an automated benchmarking
framework based on debates between LLMs, judged by another LLM. This method
assesses not only domain knowledge, but also skills such as argumentative
reasoning and inconsistency recognition. We evaluate the performance of various
state-of-the-art LLMs using the debate framework and achieve rankings that
align closely with popular rankings based on human input, eliminating the need
for costly human crowdsourcing.


---

**[40. [2402.14016] Is LLM-as-a-Judge Robust? Investigating Universal Adversarial Attacks on
  Zero-shot LLM Assessment](https://arxiv.org/pdf/2402.14016.pdf)** (2024-07-08)

*Vyas Raina, Adian Liusie, Mark Gales*

  Large Language Models (LLMs) are powerful zero-shot assessors used in
real-world situations such as assessing written exams and benchmarking systems.
Despite these critical applications, no existing work has analyzed the
vulnerability of judge-LLMs to adversarial manipulation. This work presents the
first study on the adversarial robustness of assessment LLMs, where we
demonstrate that short universal adversarial phrases can be concatenated to
deceive judge LLMs to predict inflated scores. Since adversaries may not know
or have access to the judge-LLMs, we propose a simple surrogate attack where a
surrogate model is first attacked, and the learned attack phrase then
transferred to unknown judge-LLMs. We propose a practical algorithm to
determine the short universal attack phrases and demonstrate that when
transferred to unseen models, scores can be drastically inflated such that
irrespective of the assessed text, maximum scores are predicted. It is found
that judge-LLMs are significantly more susceptible to these adversarial attacks
when used for absolute scoring, as opposed to comparative assessment. Our
findings raise concerns on the reliability of LLM-as-a-judge methods, and
emphasize the importance of addressing vulnerabilities in LLM assessment
methods before deployment in high-stakes real-world scenarios.


---

**[41. [2501.03491] Can LLMs Design Good Questions Based on Context?](https://arxiv.org/pdf/2501.03491.pdf)** (2025-01-08)

*Yueheng Zhang, Xiaoyuan Liu, Yiyou Sun, Atheer Alharbi, Hend Alzahrani, Basel Alomair, Dawn Song*

  This paper evaluates questions generated by LLMs from context, comparing them
to human-generated questions across six dimensions. We introduce an automated
LLM-based evaluation method, focusing on aspects like question length, type,
context coverage, and answerability. Our findings highlight unique
characteristics of LLM-generated questions, contributing insights that can
support further research in question quality and downstream applications.


---

**[42. [2404.18796] Replacing Judges with Juries: Evaluating LLM Generations with a Panel of
  Diverse Models](https://arxiv.org/pdf/2404.18796.pdf)** (2024-05-02)

*Pat Verga, Sebastian Hofstatter, Sophia Althammer, Yixuan Su, Aleksandra Piktus, Arkady Arkhangorodsky, Minjie Xu, Naomi White, Patrick Lewis*

  As Large Language Models (LLMs) have become more advanced, they have outpaced
our abilities to accurately evaluate their quality. Not only is finding data to
adequately probe particular model properties difficult, but evaluating the
correctness of a model's freeform generation alone is a challenge. To address
this, many evaluations now rely on using LLMs themselves as judges to score the
quality of outputs from other LLMs. Evaluations most commonly use a single
large model like GPT4. While this method has grown in popularity, it is costly,
has been shown to introduce intramodel bias, and in this work, we find that
very large models are often unnecessary. We propose instead to evaluate models
using a Panel of LLm evaluators (PoLL). Across three distinct judge settings
and spanning six different datasets, we find that using a PoLL composed of a
larger number of smaller models outperforms a single large judge, exhibits less
intra-model bias due to its composition of disjoint model families, and does so
while being over seven times less expensive.


---

**[43. [2407.01212] EconNLI: Evaluating Large Language Models on Economics Reasoning](https://arxiv.org/pdf/2407.01212.pdf)** (2024-07-02)

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

**[44. [2502.11689] Improve LLM-as-a-Judge Ability as a General Ability](https://arxiv.org/pdf/2502.11689.pdf)** (2025-02-18)

*Jiachen Yu, Shaoning Sun, Xiaohui Hu, Jiaxu Yan, Kaidong Yu, Xuelong Li*

  LLM-as-a-Judge leverages the generative and reasoning capabilities of large
language models (LLMs) to evaluate LLM responses across diverse scenarios,
providing accurate preference signals. This approach plays a vital role in
aligning LLMs with human values, ensuring ethical and reliable AI outputs that
align with societal norms. Recent studies have raised many methods to train LLM
as generative judges, but most of them are data consuming or lack accuracy, and
only focus on LLM's judge ability. In this work, we regard judge ability as a
general ability of LLM and implement a two-stage training approach, comprising
supervised fine-tuning (SFT) warm-up and direct preference optimization (DPO)
enhancement, to achieve judge style adaptation and improve judgment accuracy.
Additionally, we introduce an efficient data synthesis method to generate
judgmental content. Experimental results demonstrate that our approach,
utilizing only about 2% to 40% of the data required by other methods, achieves
SOTA performance on RewardBench. Furthermore, our training method enhances the
general capabilities of the model by constructing complicated judge task, and
the judge signals provided by our model have significantly enhanced the
downstream DPO training performance of our internal models in our test to
optimize policy model with Judge Model. We also open-source our model weights
and training data to facilitate further research.


---

**[45. [2404.04302] CBR-RAG: Case-Based Reasoning for Retrieval Augmented Generation in LLMs
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

**[46. [2502.02988] Training an LLM-as-a-Judge Model: Pipeline, Insights, and Practical
  Lessons](https://arxiv.org/pdf/2502.02988.pdf)** (2025-02-06)

*Renjun Hu, Yi Cheng, Libin Meng, Jiaxin Xia, Yi Zong, Xing Shi, Wei Lin*

  The rapid advancement of large language models (LLMs) has opened new
possibilities for their adoption as evaluative judges. This paper introduces
Themis, a fine-tuned LLM judge that delivers sophisticated context-aware
evaluations. We provide a comprehensive overview of the development pipeline
for Themis, highlighting its scenario-dependent evaluation prompts and two
novel methods for controlled instruction generation. These designs enable
Themis to effectively distill evaluative skills from teacher models, while
retaining flexibility for continuous development. We introduce two
human-labeled benchmarks for meta-evaluation, demonstrating that Themis can
achieve high alignment with human preferences in an economical manner.
Additionally, we explore insights into the LLM-as-a-judge paradigm, revealing
nuances in performance and the varied effects of reference answers. Notably, we
observe that pure knowledge distillation from strong LLMs, though common, does
not guarantee performance improvement through scaling. We propose a mitigation
strategy based on instruction-following difficulty. Furthermore, we provide
practical guidelines covering data balancing, prompt customization,
multi-objective training, and metric aggregation. We aim for our method and
findings, along with the fine-tuning data, benchmarks, and model checkpoints,
to support future research and development in this area.


---

**[47. [2501.10970] The Alternative Annotator Test for LLM-as-a-Judge: How to Statistically
  Justify Replacing Human Annotators with LLMs](https://arxiv.org/pdf/2501.10970.pdf)** (2025-02-06)

*Nitay Calderon, Roi Reichart, Rotem Dror*

  The "LLM-as-a-judge" paradigm employs Large Language Models (LLMs) as
annotators and evaluators in tasks traditionally performed by humans. LLM
annotations are widely used, not only in NLP research but also in fields like
medicine, psychology, and social science. Despite their role in shaping study
results and insights, there is no standard or rigorous procedure to determine
whether LLMs can replace human annotators. In this paper, we propose a novel
statistical procedure -- the Alternative Annotator Test (alt-test) -- that
requires only a modest subset of annotated examples to justify using LLM
annotations. Additionally, we introduce a versatile and interpretable measure
for comparing LLM judges. To demonstrate our procedure, we curated a diverse
collection of ten datasets, consisting of language and vision-language tasks,
and conducted experiments with six LLMs and four prompting techniques. Our
results show that LLMs can sometimes replace humans with closed-source LLMs
(such as GPT-4o), outperforming open-source LLMs, and that prompting techniques
yield judges of varying quality. We hope this study encourages more rigorous
and reliable practices.


---

**[48. [2406.07791] Judging the Judges: A Systematic Study of Position Bias in
  LLM-as-a-Judge](https://arxiv.org/pdf/2406.07791.pdf)** (2025-04-18)

*Lin Shi, Chiyu Ma, Wenhua Liang, Xingjian Diao, Weicheng Ma, Soroush Vosoughi*

  LLM-as-a-Judge has emerged as a promising alternative to human evaluators
across various tasks, yet inherent biases - particularly position bias, the
tendency to favor solutions based on their position within the prompt -
compromise its reliability. This exploratory study evaluates position bias in
LLM judges across pairwise and list-wise comparison settings, introducing three
metrics: repetition stability, position consistency, and preference fairness.
Our experiments, involving 15 LLM judges across MTBench and DevBench with 22
tasks and approximately 40 solution-generating models, result in over 150,000
evaluation instances. We identify Judge-Level, Candidate-Level, and Task-Level
factors contributing to bias. The findings confirm that position bias is not
due to random chance and varies significantly across judges and tasks. While
position bias is weakly influenced by the length of prompt components, it is
strongly affected by the quality gap between solutions. Our agreement and
disagreement analysis among judges further provides insights into the
distribution of judging difficulty across the dataset, and highlights the
potential for dataset modifications.


---

**[49. [2503.21157] Real-Time Evaluation Models for RAG: Who Detects Hallucinations Best?](https://arxiv.org/pdf/2503.21157.pdf)** (2025-04-08)

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

**[50. [2408.08688] The Fellowship of the LLMs: Multi-Agent Workflows for Synthetic
  Preference Optimization Dataset Generation](https://arxiv.org/pdf/2408.08688.pdf)** (2024-10-17)

*Samee Arif, Sualeha Farid, Abdul Hameed Azeemi, Awais Athar, Agha Ali Raza*

  This paper presents a novel methodology for generating synthetic Preference
Optimization (PO) datasets using multi-agent workflows. We evaluate the
effectiveness and potential of these workflows in automating and enhancing the
dataset generation process. PO dataset generation requires two modules: (1)
response evaluation, and (2) response generation. In the response evaluation
module, the responses from Large Language Models (LLMs) are evaluated and
ranked - a task typically carried out by human annotators that we automate
using LLMs. We assess the response evaluation module in a 2 step process. In
step 1, we assess LLMs as evaluators using three distinct prompting strategies.
In step 2, we apply the winning prompting strategy to compare the performance
of LLM-as-a-Judge, LLMs-as-a-Jury, and LLM Debate. Our evaluation shows that
GPT-4o-as-a-Judge is more consistent across all datasets. For the response
generation module, we use the identified LLM evaluator configuration and
compare different configurations of the LLM Feedback Loop. We use the win rate
to determine the best multi-agent configuration for generation. Experimenting
with various configurations, we find that the LLM Feedback Loop, with Llama as
the generator and Gemma as the reviewer, achieves a notable 71.8% and 73.8% win
rate over single-agent Llama and Gemma, respectively. After identifying the
best configurations for both modules, we generate our PO datasets using the
above pipeline.


---

**[51. [2504.00050] JudgeLRM: Large Reasoning Models as a Judge](https://arxiv.org/pdf/2504.00050.pdf)** (2025-04-02)

*Nuo Chen, Zhiyuan Hu, Qingyun Zou, Jiaying Wu, Qian Wang, Bryan Hooi, Bingsheng He*

  The rise of Large Language Models (LLMs) as evaluators offers a scalable
alternative to human annotation, yet existing Supervised Fine-Tuning (SFT) for
judges approaches often fall short in domains requiring complex reasoning. In
this work, we investigate whether LLM judges truly benefit from enhanced
reasoning capabilities. Through a detailed analysis of reasoning requirements
across evaluation tasks, we reveal a negative correlation between SFT
performance gains and the proportion of reasoning-demanding samples -
highlighting the limitations of SFT in such scenarios. To address this, we
introduce JudgeLRM, a family of judgment-oriented LLMs trained using
reinforcement learning (RL) with judge-wise, outcome-driven rewards. JudgeLRM
models consistently outperform both SFT-tuned and state-of-the-art reasoning
models. Notably, JudgeLRM-3B surpasses GPT-4, and JudgeLRM-7B outperforms
DeepSeek-R1 by 2.79% in F1 score, particularly excelling in judge tasks
requiring deep reasoning.


---

**[52. [2410.03608] TICKing All the Boxes: Generated Checklists Improve LLM Evaluation and
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

**[53. [2503.22968] HRET: A Self-Evolving LLM Evaluation Toolkit for Korean](https://arxiv.org/pdf/2503.22968.pdf)** (2025-04-02)

*Hanwool Lee, Soo Yong Kim, Dasol Choi, SangWon Baek, Seunghyeok Hong, Ilgyun Jeong, Inseon Hwang, Naeun Lee, Guijin Son*

  Recent advancements in Korean large language models (LLMs) have spurred
numerous benchmarks and evaluation methodologies, yet the lack of a
standardized evaluation framework has led to inconsistent results and limited
comparability. To address this, we introduce HRET Haerae Evaluation Toolkit, an
open-source, self-evolving evaluation framework tailored specifically for
Korean LLMs. HRET unifies diverse evaluation methods, including logit-based
scoring, exact-match, language-inconsistency penalization, and LLM-as-a-Judge
assessments. Its modular, registry-based architecture integrates major
benchmarks (HAE-RAE Bench, KMMLU, KUDGE, HRM8K) and multiple inference backends
(vLLM, HuggingFace, OpenAI-compatible endpoints). With automated pipelines for
continuous evolution, HRET provides a robust foundation for reproducible, fair,
and transparent Korean NLP research.


---

**[54. [2408.00122] A Course Shared Task on Evaluating LLM Output for Clinical Questions](https://arxiv.org/pdf/2408.00122.pdf)** (2024-08-02)

*Yufang Hou, Thy Thy Tran, Doan Nam Long Vu, Yiwen Cao, Kai Li, Lukas Rohde, Iryna Gurevych*

  This paper presents a shared task that we organized at the Foundations of
Language Technology (FoLT) course in 2023/2024 at the Technical University of
Darmstadt, which focuses on evaluating the output of Large Language Models
(LLMs) in generating harmful answers to health-related clinical questions. We
describe the task design considerations and report the feedback we received
from the students. We expect the task and the findings reported in this paper
to be relevant for instructors teaching natural language processing (NLP) and
designing course assignments.


---

**[55. [2406.19065] STBench: Assessing the Ability of Large Language Models in
  Spatio-Temporal Analysis](https://arxiv.org/pdf/2406.19065.pdf)** (2024-06-28)

*Wenbin Li, Di Yao, Ruibo Zhao, Wenjie Chen, Zijie Xu, Chengxue Luo, Chang Gong, Quanliang Jing, Haining Tan, Jingping Bi*

  The rapid evolution of large language models (LLMs) holds promise for
reforming the methodology of spatio-temporal data mining. However, current
works for evaluating the spatio-temporal understanding capability of LLMs are
somewhat limited and biased. These works either fail to incorporate the latest
language models or only focus on assessing the memorized spatio-temporal
knowledge. To address this gap, this paper dissects LLMs' capability of
spatio-temporal data into four distinct dimensions: knowledge comprehension,
spatio-temporal reasoning, accurate computation, and downstream applications.
We curate several natural language question-answer tasks for each category and
build the benchmark dataset, namely STBench, containing 13 distinct tasks and
over 60,000 QA pairs. Moreover, we have assessed the capabilities of 13 LLMs,
such as GPT-4o, Gemma and Mistral. Experimental results reveal that existing
LLMs show remarkable performance on knowledge comprehension and spatio-temporal
reasoning tasks, with potential for further enhancement on other tasks through
in-context learning, chain-of-though prompting, and fine-tuning. The code and
datasets of STBench are released on https://github.com/LwbXc/STBench.


---

**[56. [2410.05193] RevisEval: Improving LLM-as-a-Judge via Response-Adapted References](https://arxiv.org/pdf/2410.05193.pdf)** (2025-04-08)

*Qiyuan Zhang, Yufei Wang, Tiezheng YU, Yuxin Jiang, Chuhan Wu, Liangyou Li, Yasheng Wang, Xin Jiang, Lifeng Shang, Ruiming Tang, Fuyuan Lyu, Chen Ma*

  With significant efforts in recent studies, LLM-as-a-Judge has become a
cost-effective alternative to human evaluation for assessing text generation
quality in a wide range of tasks. However, there still remains a reliability
gap between LLM-as-a-Judge and human evaluation. One important reason is the
lack of guided oracles in the evaluation process. Motivated by the role of
reference pervasively used in classic text evaluation, we introduce RevisEval,
a novel text generation evaluation paradigm via the response-adapted
references. RevisEval is driven by the key observation that an ideal reference
should maintain the necessary relevance to the response to be evaluated.
Specifically, RevisEval leverages the text revision capabilities of large
language models (LLMs) to adaptively revise the response, then treat the
revised text as the reference (response-adapted reference) for the subsequent
evaluation. Extensive experiments demonstrate that RevisEval outperforms
traditional reference-free and reference-based evaluation paradigms that use
LLM-as-a-Judge across NLG tasks and open-ended instruction-following tasks.
More importantly, our response-adapted references can further boost the
classical text metrics, e.g., BLEU and BERTScore, compared to traditional
references and even rival the LLM-as-a-Judge. A detailed analysis is also
conducted to confirm RevisEval's effectiveness in bias reduction, the impact of
inference cost, and reference relevance.


---

**[57. [2410.02425] LLM-Pilot: Characterize and Optimize Performance of your LLM Inference
  Services](https://arxiv.org/pdf/2410.02425.pdf)** (2024-10-04)

*Małgorzata Łazuka, Andreea Anghel, Thomas Parnell*

  As Large Language Models (LLMs) are rapidly growing in popularity, LLM
inference services must be able to serve requests from thousands of users while
satisfying performance requirements. The performance of an LLM inference
service is largely determined by the hardware onto which it is deployed, but
understanding of which hardware will deliver on performance requirements
remains challenging. In this work we present LLM-Pilot - a first-of-its-kind
system for characterizing and predicting performance of LLM inference services.
LLM-Pilot performs benchmarking of LLM inference services, under a realistic
workload, across a variety of GPUs, and optimizes the service configuration for
each considered GPU to maximize performance. Finally, using this
characterization data, LLM-Pilot learns a predictive model, which can be used
to recommend the most cost-effective hardware for a previously unseen LLM.
Compared to existing methods, LLM-Pilot can deliver on performance requirements
33% more frequently, whilst reducing costs by 60% on average.


---

**[58. [2503.02374] MedEthicEval: Evaluating Large Language Models Based on Chinese Medical
  Ethics](https://arxiv.org/pdf/2503.02374.pdf)** (2025-03-05)

*Haoan Jin, Jiacheng Shi, Hanhui Xu, Kenny Q. Zhu, Mengyue Wu*

  Large language models (LLMs) demonstrate significant potential in advancing
medical applications, yet their capabilities in addressing medical ethics
challenges remain underexplored. This paper introduces MedEthicEval, a novel
benchmark designed to systematically evaluate LLMs in the domain of medical
ethics. Our framework encompasses two key components: knowledge, assessing the
models' grasp of medical ethics principles, and application, focusing on their
ability to apply these principles across diverse scenarios. To support this
benchmark, we consulted with medical ethics researchers and developed three
datasets addressing distinct ethical challenges: blatant violations of medical
ethics, priority dilemmas with clear inclinations, and equilibrium dilemmas
without obvious resolutions. MedEthicEval serves as a critical tool for
understanding LLMs' ethical reasoning in healthcare, paving the way for their
responsible and effective use in medical contexts.


---

**[59. [2502.19209] Bi'an: A Bilingual Benchmark and Model for Hallucination Detection in
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

**[60. [2408.09831] Ranking Generated Answers: On the Agreement of Retrieval Models with
  Humans on Consumer Health Questions](https://arxiv.org/pdf/2408.09831.pdf)** (2025-01-20)

*Sebastian Heineking, Jonas Probst, Daniel Steinbach, Martin Potthast, Harrisen Scells*

  Evaluating the output of generative large language models (LLMs) is
challenging and difficult to scale. Many evaluations of LLMs focus on tasks
such as single-choice question-answering or text classification. These tasks
are not suitable for assessing open-ended question-answering capabilities,
which are critical in domains where expertise is required. One such domain is
health, where misleading or incorrect answers can have a negative impact on a
user's well-being. Using human experts to evaluate the quality of LLM answers
is generally considered the gold standard, but expert annotation is costly and
slow. We present a method for evaluating LLM answers that uses ranking models
trained on annotated document collections as a substitute for explicit
relevance judgements and apply it to the CLEF 2021 eHealth dataset. In a user
study, our method correlates with the preferences of a human expert (Kendall's
$\tau=0.64$). It is also consistent with previous findings in that the quality
of generated answers improves with the size of the model and more sophisticated
prompting strategies.


---

**[61. [2407.00215] LLM Critics Help Catch LLM Bugs](https://arxiv.org/pdf/2407.00215.pdf)** (2024-07-02)

*Nat McAleese, Rai Michael Pokorny, Juan Felipe Ceron Uribe, Evgenia Nitishinskaya, Maja Trebacz, Jan Leike*

  Reinforcement learning from human feedback (RLHF) is fundamentally limited by
the capacity of humans to correctly evaluate model output. To improve human
evaluation ability and overcome that limitation this work trains "critic"
models that help humans to more accurately evaluate model-written code. These
critics are themselves LLMs trained with RLHF to write natural language
feedback highlighting problems in code from real-world assistant tasks. On code
containing naturally occurring LLM errors model-written critiques are preferred
over human critiques in 63% of cases, and human evaluation finds that models
catch more bugs than human contractors paid for code review. We further confirm
that our fine-tuned LLM critics can successfully identify hundreds of errors in
ChatGPT training data rated as "flawless", even though the majority of those
tasks are non-code tasks and thus out-of-distribution for the critic model.
Critics can have limitations of their own, including hallucinated bugs that
could mislead humans into making mistakes they might have otherwise avoided,
but human-machine teams of critics and contractors catch similar numbers of
bugs to LLM critics while hallucinating less than LLMs alone.


---

**[62. [2503.16040] Evaluating Test-Time Scaling LLMs for Legal Reasoning: OpenAI o1,
  DeepSeek-R1, and Beyond](https://arxiv.org/pdf/2503.16040.pdf)** (2025-03-21)

*Yaoyao Yu, Leilei Gan, Yinghao Hu, Bin Wei, Kun Kuang, Fei Wu*

  Recently, Test-Time Scaling Large Language Models (LLMs), such as DeepSeek-R1
and OpenAI o1, have demonstrated exceptional capabilities across various
domains and tasks, particularly in reasoning. While these models have shown
impressive performance on general language tasks, their effectiveness in
specialized fields like legal remains unclear. To address this, we present a
preliminary evaluation of LLMs in various legal scenarios, covering both
Chinese and English legal tasks. Our analysis includes 9 LLMs and 17 legal
tasks, with a focus on newly published and more complex challenges such as
multi-defendant legal judgments and legal argument reasoning. Our findings
indicate that, despite DeepSeek-R1 and OpenAI o1 being among the most powerful
models, their legal reasoning capabilities are still lacking. Specifically,
these models score below 80\% on seven Chinese legal reasoning tasks and below
80\% on two English legal reasoning tasks. This suggests that, even among the
most advanced reasoning models, legal reasoning abilities remain
underdeveloped.


---

**[63. [2406.08747] StreamBench: Towards Benchmarking Continuous Improvement of Language
  Agents](https://arxiv.org/pdf/2406.08747.pdf)** (2024-11-01)

*Cheng-Kuang Wu, Zhi Rui Tam, Chieh-Yen Lin, Yun-Nung Chen, Hung-yi Lee*

  Recent works have shown that large language model (LLM) agents are able to
improve themselves from experience, which is an important ability for
continuous enhancement post-deployment. However, existing benchmarks primarily
evaluate their innate capabilities and do not assess their ability to improve
over time. To address this gap, we introduce StreamBench, a pioneering
benchmark designed to evaluate the continuous improvement of LLM agents over an
input-feedback sequence. StreamBench simulates an online learning environment
where LLMs receive a continuous flow of feedback stream and iteratively enhance
their performance. In addition, we propose several simple yet effective
baselines for improving LLMs on StreamBench, and provide a comprehensive
analysis to identify critical components that contribute to successful
streaming strategies. Our work serves as a stepping stone towards developing
effective online learning strategies for LLMs, paving the way for more adaptive
AI systems in streaming scenarios. Source code:
https://github.com/stream-bench/stream-bench. Benchmark website:
https://stream-bench.github.io.


---

**[64. [2502.19064] Can Large Language Models Outperform Non-Experts in Poetry Evaluation? A
  Comparative Study Using the Consensual Assessment Technique](https://arxiv.org/pdf/2502.19064.pdf)** (2025-02-27)

*Piotr Sawicki, Marek Grześ, Dan Brown, Fabrício Góes*

  The Consensual Assessment Technique (CAT) evaluates creativity through
holistic expert judgments. We investigate the use of two advanced Large
Language Models (LLMs), Claude-3-Opus and GPT-4o, to evaluate poetry by a
methodology inspired by the CAT. Using a dataset of 90 poems, we found that
these LLMs can surpass the results achieved by non-expert human judges at
matching a ground truth based on publication venue, particularly when assessing
smaller subsets of poems. Claude-3-Opus exhibited slightly superior performance
than GPT-4o. We show that LLMs are viable tools for accurately assessing
poetry, paving the way for their broader application into other creative
domains.


---

**[65. [2408.13006] Systematic Evaluation of LLM-as-a-Judge in LLM Alignment Tasks:
  Explainable Metrics and Diverse Prompt Templates](https://arxiv.org/pdf/2408.13006.pdf)** (2025-04-01)

*Hui Wei, Shenghua He, Tian Xia, Fei Liu, Andy Wong, Jingyang Lin, Mei Han*

  LLM-as-a-Judge has been widely applied to evaluate and compare different LLM
alignmnet approaches (e.g., RLHF and DPO). However, concerns regarding its
reliability have emerged, due to LLM judges' biases and inconsistent
decision-making. Previous research has developed evaluation frameworks to
assess reliability of LLM judges and their alignment with human preferences.
However, the employed evaluation metrics often lack adequate explainability and
fail to address LLM internal inconsistency. Additionally, existing studies
inadequately explore the impact of various prompt templates when applying
LLM-as-a-Judge methods, leading to potentially inconsistent comparisons between
different alignment algorithms. In this work, we systematically evaluate
LLM-as-a-Judge on alignment tasks by defining more theoretically interpretable
evaluation metrics and explicitly mitigating LLM internal inconsistency from
reliability metrics. We develop an open-source framework to evaluate, compare,
and visualize the reliability and alignment of LLM judges, which facilitates
practitioners to choose LLM judges for alignment tasks. In the experiments, we
examine effects of diverse prompt templates on LLM-judge reliability and also
demonstrate our developed framework by comparing various LLM judges on two
common alignment datasets (i.e., TL;DR Summarization and HH-RLHF-Helpfulness).
Our results indicate a significant impact of prompt templates on LLM judge
performance, as well as a mediocre alignment level between the tested LLM
judges and human evaluators.


---

**[66. [2503.05061] No Free Labels: Limitations of LLM-as-a-Judge Without Human Grounding](https://arxiv.org/pdf/2503.05061.pdf)** (2025-03-10)

*Michael Krumdick, Charles Lovering, Varshini Reddy, Seth Ebner, Chris Tanner*

  LLM-as-a-Judge is a framework that uses an LLM (large language model) to
evaluate the quality of natural language text - typically text that is also
generated by an LLM. This framework holds great promise due to its relative
low-cost, ease of use, and strong correlations with human stylistic
preferences. However, LLM Judges have been shown to exhibit biases that can
distort their judgments. We evaluate how well LLM Judges can grade whether a
given response to a conversational question is correct, an ability crucial to
soundly estimating the overall response quality. To do so, we create and
publicly release a human-annotated dataset with labels of correctness for 1,200
LLM responses. We source questions from a combination of existing datasets and
a novel, challenging benchmark (BFF-Bench) created for this analysis. We
demonstrate a strong connection between an LLM's ability to correctly answer a
question and grade responses to that question. Although aggregate level
statistics might imply a judge has high agreement with human annotators, it
will struggle on the subset of questions it could not answer. To address this
issue, we recommend a simple solution: provide the judge with a correct,
human-written reference answer. We perform an in-depth analysis on how
reference quality can affect the performance of an LLM Judge. We show that
providing a weaker judge (e.g. Qwen 2.5 7B) with higher quality references
reaches better agreement with human annotators than a stronger judge (e.g.
GPT-4o) with synthetic references.


---

**[67. [2502.11393] HellaSwag-Pro: A Large-Scale Bilingual Benchmark for Evaluating the
  Robustness of LLMs in Commonsense Reasoning](https://arxiv.org/pdf/2502.11393.pdf)** (2025-02-18)

*Xiaoyuan Li, Moxin Li, Rui Men, Yichang Zhang, Keqin Bao, Wenjie Wang, Fuli Feng, Dayiheng Liu, Junyang Lin*

  Large language models (LLMs) have shown remarkable capabilities in
commonsense reasoning; however, some variations in questions can trigger
incorrect responses. Do these models truly understand commonsense knowledge, or
just memorize expression patterns? To investigate this question, we present the
first extensive robustness evaluation of LLMs in commonsense reasoning. We
introduce HellaSwag-Pro, a large-scale bilingual benchmark consisting of 11,200
cases, by designing and compiling seven types of question variants. To
construct this benchmark, we propose a two-stage method to develop Chinese
HellaSwag, a finely annotated dataset comprising 12,000 instances across 56
categories. We conduct extensive experiments on 41 representative LLMs,
revealing that these LLMs are far from robust in commonsense reasoning.
Furthermore, this robustness varies depending on the language in which the LLM
is tested. This work establishes a high-quality evaluation benchmark, with
extensive experiments offering valuable insights to the community in
commonsense reasoning for LLMs.


---

**[68. [2501.03112] LangFair: A Python Package for Assessing Bias and Fairness in Large
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

**[69. [2405.05894] Efficient LLM Comparative Assessment: a Product of Experts Framework for
  Pairwise Comparisons](https://arxiv.org/pdf/2405.05894.pdf)** (2024-11-13)

*Adian Liusie, Vatsal Raina, Yassir Fathullah, Mark Gales*

  LLM-as-a-judge approaches are a practical and effective way of assessing a
range of text tasks. However, when using pairwise comparisons to rank a set of
candidates, the computational cost scales quadratically with the number of
candidates, which has practical limitations. This paper introduces a Product of
Expert (PoE) framework for efficient LLM Comparative Assessment. Here
individual comparisons are considered experts that provide information on a
pair's score difference. The PoE framework combines the information from these
experts to yield an expression that can be maximized with respect to the
underlying set of candidates, and is highly flexible where any form of expert
can be assumed. When Gaussian experts are used one can derive simple
closed-form solutions for the optimal candidate ranking, and expressions for
selecting which comparisons should be made to maximize the probability of this
ranking. Our approach enables efficient comparative assessment, where by using
only a small subset of the possible comparisons, one can generate score
predictions that correlate well with human judgements. We evaluate the approach
on multiple NLG tasks and demonstrate that our framework can yield considerable
computational savings when performing pairwise comparative assessment. With
many candidate texts, using as few as 2% of comparisons the PoE solution can
achieve similar performance to when all comparisons are used.


---

**[70. [2410.21819] Self-Preference Bias in LLM-as-a-Judge](https://arxiv.org/pdf/2410.21819.pdf)** (2024-10-30)

*Koki Wataoka, Tsubasa Takahashi, Ryokan Ri*

  Automated evaluation leveraging large language models (LLMs), commonly
referred to as LLM evaluators or LLM-as-a-judge, has been widely used in
measuring the performance of dialogue systems. However, the self-preference
bias in LLMs has posed significant risks, including promoting specific styles
or policies intrinsic to the LLMs. Despite the importance of this issue, there
is a lack of established methods to measure the self-preference bias
quantitatively, and its underlying causes are poorly understood. In this paper,
we introduce a novel quantitative metric to measure the self-preference bias.
Our experimental results demonstrate that GPT-4 exhibits a significant degree
of self-preference bias. To explore the causes, we hypothesize that LLMs may
favor outputs that are more familiar to them, as indicated by lower perplexity.
We analyze the relationship between LLM evaluations and the perplexities of
outputs. Our findings reveal that LLMs assign significantly higher evaluations
to outputs with lower perplexity than human evaluators, regardless of whether
the outputs were self-generated. This suggests that the essence of the bias
lies in perplexity and that the self-preference bias exists because LLMs prefer
texts more familiar to them.


---

**[71. [2407.19594] Meta-Rewarding Language Models: Self-Improving Alignment with
  LLM-as-a-Meta-Judge](https://arxiv.org/pdf/2407.19594.pdf)** (2024-07-31)

*Tianhao Wu, Weizhe Yuan, Olga Golovneva, Jing Xu, Yuandong Tian, Jiantao Jiao, Jason Weston, Sainbayar Sukhbaatar*

  Large Language Models (LLMs) are rapidly surpassing human knowledge in many
domains. While improving these models traditionally relies on costly human
data, recent self-rewarding mechanisms (Yuan et al., 2024) have shown that LLMs
can improve by judging their own responses instead of relying on human
labelers. However, existing methods have primarily focused on improving model
responses rather than judgment capabilities, resulting in rapid saturation
during iterative training. To address this issue, we introduce a novel
Meta-Rewarding step to the self-improvement process, where the model judges its
own judgements and uses that feedback to refine its judgment skills.
Surprisingly, this unsupervised approach improves the model's ability to judge
{\em and} follow instructions, as demonstrated by a win rate improvement of
Llama-3-8B-Instruct from 22.9% to 39.4% on AlpacaEval 2, and 20.6% to 29.1% on
Arena-Hard. These results strongly suggest the potential for self-improving
models without human supervision.


---

**[72. [2410.02054] Comparing Criteria Development Across Domain Experts, Lay Users, and
  Models in Large Language Model Evaluation](https://arxiv.org/pdf/2410.02054.pdf)** (2024-10-04)

*Annalisa Szymanski, Simret Araya Gebreegziabher, Oghenemaro Anuyah, Ronald A. Metoyer, Toby Jia-Jun Li*

  Large Language Models (LLMs) are increasingly utilized for domain-specific
tasks, yet integrating domain expertise into evaluating their outputs remains
challenging. A common approach to evaluating LLMs is to use metrics, or
criteria, which are assertions used to assess performance that help ensure that
their outputs align with domain-specific standards. Previous efforts have
involved developers, lay users, or the LLMs themselves in creating these
criteria, however, evaluation particularly from a domain expertise perspective,
remains understudied. This study explores how domain experts contribute to LLM
evaluation by comparing their criteria with those generated by LLMs and lay
users. We further investigate how the criteria-setting process evolves,
analyzing changes between a priori and a posteriori stages. Our findings
emphasize the importance of involving domain experts early in the evaluation
process while utilizing complementary strengths of lay users and LLMs. We
suggest implications for designing workflows that leverage these strengths at
different evaluation stages.


---

**[73. [2402.09668] How to Train Data-Efficient LLMs](https://arxiv.org/pdf/2402.09668.pdf)** (2024-02-16)

*Noveen Sachdeva, Benjamin Coleman, Wang-Cheng Kang, Jianmo Ni, Lichan Hong, Ed H. Chi, James Caverlee, Julian McAuley, Derek Zhiyuan Cheng*

  The training of large language models (LLMs) is expensive. In this paper, we
study data-efficient approaches for pre-training LLMs, i.e., techniques that
aim to optimize the Pareto frontier of model quality and training resource/data
consumption. We seek to understand the tradeoffs associated with data selection
routines based on (i) expensive-to-compute data-quality estimates, and (ii)
maximization of coverage and diversity-based measures in the feature space. Our
first technique, Ask-LLM, leverages the zero-shot reasoning capabilities of
instruction-tuned LLMs to directly assess the quality of a training example. To
target coverage, we propose Density sampling, which models the data
distribution to select a diverse sample. In our comparison of 19 samplers,
involving hundreds of evaluation tasks and pre-training runs, we find that
Ask-LLM and Density are the best methods in their respective categories.
Coverage sampling can recover the performance of the full data, while models
trained on Ask-LLM data consistently outperform full-data training -- even when
we reject 90% of the original dataset, while converging up to 70% faster.


---

**[74. [2501.17178] Tuning LLM Judge Design Decisions for 1/1000 of the Cost](https://arxiv.org/pdf/2501.17178.pdf)** (2025-03-19)

*David Salinas, Omar Swelam, Frank Hutter*

  Evaluating Large Language Models (LLMs) often requires costly human
annotations. To address this, LLM-based judges have been proposed, which
compare the outputs of two LLMs enabling the ranking of models without human
intervention. While several approaches have been proposed, many confounding
factors are present between different papers. For instance the model, the
prompt and other hyperparameters are typically changed at the same time making
apple-to-apple comparisons challenging. In this paper, we propose to
systematically analyze and tune hyperparameter of LLM judges. To alleviate the
high cost of evaluating a judge, we propose to leverage multi-objective
multi-fidelity which allows to find judges that trades accuracy for cost and
also reduce significantly the cost of the search. Our method identifies judges
that not only outperform existing benchmarks in accuracy and cost-efficiency
but also utilize open-weight models, ensuring greater accessibility and
reproducibility.


---

**[75. [2305.08322] C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for
  Foundation Models](https://arxiv.org/pdf/2305.08322.pdf)** (2023-11-07)

*Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Jiayi Lei, Yao Fu, Maosong Sun, Junxian He*

  New NLP benchmarks are urgently needed to align with the rapid development of
large language models (LLMs). We present C-Eval, the first comprehensive
Chinese evaluation suite designed to assess advanced knowledge and reasoning
abilities of foundation models in a Chinese context. C-Eval comprises
multiple-choice questions across four difficulty levels: middle school, high
school, college, and professional. The questions span 52 diverse disciplines,
ranging from humanities to science and engineering. C-Eval is accompanied by
C-Eval Hard, a subset of very challenging subjects in C-Eval that requires
advanced reasoning abilities to solve. We conduct a comprehensive evaluation of
the most advanced LLMs on C-Eval, including both English- and Chinese-oriented
models. Results indicate that only GPT-4 could achieve an average accuracy of
over 60%, suggesting that there is still significant room for improvement for
current LLMs. We anticipate C-Eval will help analyze important strengths and
shortcomings of foundation models, and foster their development and growth for
Chinese users.


---

**[76. [2411.05897] Humans and Large Language Models in Clinical Decision Support: A Study
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

**[77. [2501.18099] Learning to Plan & Reason for Evaluation with Thinking-LLM-as-a-Judge](https://arxiv.org/pdf/2501.18099.pdf)** (2025-01-31)

*Swarnadeep Saha, Xian Li, Marjan Ghazvininejad, Jason Weston, Tianlu Wang*

  LLM-as-a-Judge models generate chain-of-thought (CoT) sequences intended to
capture the step-bystep reasoning process that underlies the final evaluation
of a response. However, due to the lack of human annotated CoTs for evaluation,
the required components and structure of effective reasoning traces remain
understudied. Consequently, previous approaches often (1) constrain reasoning
traces to hand-designed components, such as a list of criteria, reference
answers, or verification questions and (2) structure them such that planning is
intertwined with the reasoning for evaluation. In this work, we propose
EvalPlanner, a preference optimization algorithm for Thinking-LLM-as-a-Judge
that first generates an unconstrained evaluation plan, followed by its
execution, and then the final judgment. In a self-training loop, EvalPlanner
iteratively optimizes over synthetically constructed evaluation plans and
executions, leading to better final verdicts. Our method achieves a new
state-of-the-art performance for generative reward models on RewardBench (with
a score of 93.9), despite being trained on fewer amount of, and synthetically
generated, preference pairs. Additional experiments on other benchmarks like
RM-Bench, JudgeBench, and FollowBenchEval further highlight the utility of both
planning and reasoning for building robust LLM-as-a-Judge reasoning models.


---

**[78. [2501.15595] SedarEval: Automated Evaluation using Self-Adaptive Rubrics](https://arxiv.org/pdf/2501.15595.pdf)** (2025-01-28)

*Zhiyuan Fan, Weinong Wang, Xing Wu, Debing Zhang*

  The evaluation paradigm of LLM-as-judge gains popularity due to its
significant reduction in human labor and time costs. This approach utilizes one
or more large language models (LLMs) to assess the quality of outputs from
other LLMs. However, existing methods rely on generic scoring rubrics that fail
to consider the specificities of each question and its problem-solving process,
compromising precision and stability in assessments. Inspired by human
examination scoring processes, we propose a new evaluation paradigm based on
self-adaptive rubrics. Specifically, we create detailed scoring rubrics for
each question, capturing the primary and secondary criteria in a structured
format of scoring and deduction points that mimic a human evaluator's
analytical process. Building on this paradigm, we further develop a novel
benchmark called SedarEval, which covers a range of domains including long-tail
knowledge, mathematics, coding, and logical reasoning. SedarEval consists of
1,000 meticulously crafted questions, each with its own self-adaptive rubric.
To further streamline the evaluation, we train a specialized evaluator language
model (evaluator LM) to supplant human graders. Using the same training data,
our evaluator LM achieves a higher concordance rate with human grading results
than other paradigms, including GPT-4, highlighting the superiority and
efficiency of our approach. We release our dataset at
https://github.com/wwn1233/sedareval.


---

**[79. [2410.13153] Better to Ask in English: Evaluation of Large Language Models on
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

**[80. [2502.18018] Verdict: A Library for Scaling Judge-Time Compute](https://arxiv.org/pdf/2502.18018.pdf)** (2025-02-26)

*Nimit Kalra, Leonard Tang*

  The use of LLMs as automated judges ("LLM-as-a-judge") is now widespread, yet
standard judges suffer from a multitude of reliability issues. To address these
challenges, we introduce Verdict, an open-source library for scaling judge-time
compute to enhance the accuracy, reliability, and interpretability of automated
evaluators. Verdict leverages the composition of modular reasoning units --
such as verification, debate, and aggregation -- and increased inference-time
compute to improve LLM judge quality. Across a variety of challenging tasks
such as content moderation, fact-checking, and hallucination detection, Verdict
judges achieve state-of-the-art (SOTA) or near-SOTA performance, surpassing
orders-of-magnitude larger fine-tuned judges, prompted judges, and reasoning
models. Ultimately, we hope Verdict serves as a useful framework for
researchers and practitioners building scalable, interpretable, and reliable
LLM-based evaluators.


---

**[81. [2306.11879] Open-Domain Text Evaluation via Contrastive Distribution Methods](https://arxiv.org/pdf/2306.11879.pdf)** (2024-06-11)

*Sidi Lu, Hongyi Liu, Asli Celikyilmaz, Tianlu Wang, Nanyun Peng*

  Recent advancements in open-domain text generation, driven by the power of
large pre-trained language models (LLMs), have demonstrated remarkable
performance. However, assessing these models' generation quality remains a
challenge. In this paper, we introduce a novel method for evaluating
open-domain text generation called Contrastive Distribution Methods (CDM).
Leveraging the connection between increasing model parameters and enhanced LLM
performance, CDM creates a mapping from the _contrast_ of two probabilistic
distributions -- one known to be superior to the other -- to quality measures.
We investigate CDM for open-domain text generation evaluation under two
paradigms: 1) _Generative_ CDM, which harnesses the contrast of two language
models' distributions to generate synthetic examples for training
discriminator-based metrics; 2) _Discriminative_ CDM, which directly uses
distribution disparities between two language models for evaluation. Our
experiments on coherence evaluation for multi-turn dialogue and commonsense
evaluation for controllable generation demonstrate CDM's superior correlate
with human judgment than existing automatic evaluation metrics, highlighting
the strong performance and generalizability of our approach.


---

**[82. [2404.00942] Evaluating the Factuality of Large Language Models using Large-Scale
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

**[83. [2407.04873] Evaluating Language Models for Generating and Judging Programming
  Feedback](https://arxiv.org/pdf/2407.04873.pdf)** (2024-11-25)

*Charles Koutcheme, Nicola Dainese, Arto Hellas, Sami Sarsa, Juho Leinonen, Syed Ashraf, Paul Denny*

  The emergence of large language models (LLMs) has transformed research and
practice across a wide range of domains. Within the computing education
research (CER) domain, LLMs have garnered significant attention, particularly
in the context of learning programming. Much of the work on LLMs in CER,
however, has focused on applying and evaluating proprietary models. In this
article, we evaluate the efficiency of open-source LLMs in generating
high-quality feedback for programming assignments and judging the quality of
programming feedback, contrasting the results with proprietary models. Our
evaluations on a dataset of students' submissions to introductory Python
programming exercises suggest that state-of-the-art open-source LLMs are nearly
on par with proprietary models in both generating and assessing programming
feedback. Additionally, we demonstrate the efficiency of smaller LLMs in these
tasks and highlight the wide range of LLMs accessible, even for free, to
educators and practitioners.


---

**[84. [2401.14869] F-Eval: Assessing Fundamental Abilities with Refined Evaluation Methods](https://arxiv.org/pdf/2401.14869.pdf)** (2024-08-21)

*Yu Sun, Keyu Chen, Shujie Wang, Peiji Li, Qipeng Guo, Hang Yan, Xipeng Qiu, Xuanjing Huang, Dahua Lin*

  Large language models (LLMs) garner significant attention for their
unprecedented performance, leading to an increasing number of researches
evaluating LLMs. However, these evaluation benchmarks are limited to assessing
the instruction-following capabilities, overlooking the fundamental abilities
that emerge during the pre-training stage. Previous subjective evaluation
methods mainly reply on scoring by API models. However, in the absence of
references, large models have shown limited ability to discern subtle
differences. To bridge the gap, we propose F-Eval, a bilingual evaluation
benchmark to evaluate the fundamental abilities, including expression,
commonsense and logic. The tasks in F-Eval include multi-choice objective
tasks, open-ended objective tasks, reference-based subjective tasks and
reference-free subjective tasks. For reference-free subjective tasks, we devise
new evaluation methods, serving as alternatives to scoring by API models. We
conduct evaluations on 13 advanced LLMs. Results show that our evaluation
methods show higher correlation coefficients and larger distinction than other
evaluators. Additionally, we discuss the influence of different model sizes,
dimensions, and normalization methods. We anticipate that F-Eval will
facilitate the study of LLMs' fundamental abilities.


---

**[85. [2403.11152] Evaluation Ethics of LLMs in Legal Domain](https://arxiv.org/pdf/2403.11152.pdf)** (2024-03-19)

*Ruizhe Zhang, Haitao Li, Yueyue Wu, Qingyao Ai, Yiqun Liu, Min Zhang, Shaoping Ma*

  In recent years, the utilization of large language models for natural
language dialogue has gained momentum, leading to their widespread adoption
across various domains. However, their universal competence in addressing
challenges specific to specialized fields such as law remains a subject of
scrutiny. The incorporation of legal ethics into the model has been overlooked
by researchers. We asserts that rigorous ethic evaluation is essential to
ensure the effective integration of large language models in legal domains,
emphasizing the need to assess domain-specific proficiency and domain-specific
ethic. To address this, we propose a novelty evaluation methodology, utilizing
authentic legal cases to evaluate the fundamental language abilities,
specialized legal knowledge and legal robustness of large language models
(LLMs). The findings from our comprehensive evaluation contribute significantly
to the academic discourse surrounding the suitability and performance of large
language models in legal domains.


---

**[86. [2310.05191] LLM-as-a-tutor in EFL Writing Education: Focusing on Evaluation of
  Student-LLM Interaction](https://arxiv.org/pdf/2310.05191.pdf)** (2024-09-04)

*Jieun Han, Haneul Yoo, Junho Myung, Minsun Kim, Hyunseung Lim, Yoonsu Kim, Tak Yeon Lee, Hwajung Hong, Juho Kim, So-Yeon Ahn, Alice Oh*

  In the context of English as a Foreign Language (EFL) writing education,
LLM-as-a-tutor can assist students by providing real-time feedback on their
essays. However, challenges arise in assessing LLM-as-a-tutor due to differing
standards between educational and general use cases. To bridge this gap, we
integrate pedagogical principles to assess student-LLM interaction. First, we
explore how LLMs can function as English tutors, providing effective essay
feedback tailored to students. Second, we propose three metrics to evaluate
LLM-as-a-tutor specifically designed for EFL writing education, emphasizing
pedagogical aspects. In this process, EFL experts evaluate the feedback from
LLM-as-a-tutor regarding quality and characteristics. On the other hand, EFL
learners assess their learning outcomes from interaction with LLM-as-a-tutor.
This approach lays the groundwork for developing LLMs-as-a-tutor tailored to
the needs of EFL learners, advancing the effectiveness of writing education in
this context.


---

**[87. [2407.07666] A Proposed S.C.O.R.E. Evaluation Framework for Large Language Models :
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

**[88. [2402.01733] Development and Testing of Retrieval Augmented Generation in Large
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

**[89. [2502.10709] An Empirical Analysis of Uncertainty in Large Language Model Evaluations](https://arxiv.org/pdf/2502.10709.pdf)** (2025-03-04)

*Qiujie Xie, Qingqiu Li, Zhuohao Yu, Yuejie Zhang, Yue Zhang, Linyi Yang*

  As LLM-as-a-Judge emerges as a new paradigm for assessing large language
models (LLMs), concerns have been raised regarding the alignment, bias, and
stability of LLM evaluators. While substantial work has focused on alignment
and bias, little research has concentrated on the stability of LLM evaluators.
In this paper, we conduct extensive experiments involving 9 widely used LLM
evaluators across 2 different evaluation settings to investigate the
uncertainty in model-based LLM evaluations. We pinpoint that LLM evaluators
exhibit varying uncertainty based on model families and sizes. With careful
comparative analyses, we find that employing special prompting strategies,
whether during inference or post-training, can alleviate evaluation uncertainty
to some extent. By utilizing uncertainty to enhance LLM's reliability and
detection capability in Out-Of-Distribution (OOD) data, we further fine-tune an
uncertainty-aware LLM evaluator named ConfiLM using a human-annotated
fine-tuning set and assess ConfiLM's OOD evaluation ability on a manually
designed test set sourced from the 2024 Olympics. Experimental results
demonstrate that incorporating uncertainty as additional information during the
fine-tuning phase can largely improve the model's evaluation performance in OOD
scenarios. The code and data are released at:
https://github.com/hasakiXie123/LLM-Evaluator-Uncertainty.


---

**[90. [2406.16801] RES-Q: Evaluating Code-Editing Large Language Model Systems at the
  Repository Scale](https://arxiv.org/pdf/2406.16801.pdf)** (2024-06-27)

*Beck LaBash, August Rosedale, Alex Reents, Lucas Negritto, Colin Wiel*

  The instruction-following ability of Large Language Models (LLMs) has
cultivated a class of LLM-based systems capable of approaching complex tasks
such as making edits to large code repositories. Due to the high sensitivity
and unpredictability of LLM behavior in response to changes in prompting,
robust evaluation tools are needed to drive future iteration of these systems.
We propose RES-Q, a natural language instruction-based benchmark for evaluating
$\textbf{R}$epository $\textbf{E}$diting $\textbf{S}$ystems, which consists of
100 handcrafted repository editing tasks derived from real GitHub commits.
Given an edit instruction and a code repository, RES-Q evaluates an LLM
system's ability to interpret the instruction, navigate the repository to
gather relevant information, and construct an appropriate edit that satisfies
the specified criteria. We argue that evaluating LLMs in this way addresses
issues with traditional benchmarks and provides a more holistic assessment of a
model's abilities. We evaluate various state-of-the-art LLMs as language agents
in a repository-editing system built on Qurrent OS, our language agent
development software. Despite their 1% pass@1 performance difference on
HumanEval, we find Claude Sonnet 3.5 outperforms GPT-4o by 12% pass@1 on RES-Q,
indicating RES-Q's capacity to differentiate model capability as traditional
benchmarks approach saturation. We further investigate token efficiency,
performance relationships with existing benchmarks, and interesting disparities
between closed and open-source LLMs. Code and dataset are available at
https://github.com/Qurrent-AI/RES-Q.


---

**[91. [2411.07037] LIFBench: Evaluating the Instruction Following Performance and Stability
  of Large Language Models in Long-Context Scenarios](https://arxiv.org/pdf/2411.07037.pdf)** (2024-12-17)

*Xiaodong Wu, Minhao Wang, Yichen Liu, Xiaoming Shi, He Yan, Xiangju Lu, Junmin Zhu, Wei Zhang*

  As Large Language Models (LLMs) evolve in natural language processing (NLP),
their ability to stably follow instructions in long-context inputs has become
critical for real-world applications. However, existing benchmarks seldom focus
on instruction-following in long-context scenarios or stability on different
inputs. To bridge this gap, we introduce LIFBench, a scalable dataset designed
to evaluate LLMs' instruction-following capabilities and stability across long
contexts. LIFBench comprises three long-context scenarios and eleven diverse
tasks, featuring 2,766 instructions generated through an automated expansion
method across three dimensions: length, expression, and variables. For
evaluation, we propose LIFEval, a rubric-based assessment method that enables
precise, automated scoring of complex LLM responses without reliance on
LLM-assisted assessments or human judgment. This method allows for a
comprehensive analysis of model performance and stability from multiple
perspectives. We conduct detailed experiments on 20 prominent LLMs across six
length intervals. Our work contributes LIFBench and LIFEval as robust tools for
assessing LLM performance in complex and long-context settings, offering
valuable insights to guide future advancements in LLM development.


---

**[92. [2409.11239] LLM-as-a-Judge & Reward Model: What They Can and Cannot Do](https://arxiv.org/pdf/2409.11239.pdf)** (2024-10-03)

*Guijin Son, Hyunwoo Ko, Hoyoung Lee, Yewon Kim, Seunghyeok Hong*

  LLM-as-a-Judge and reward models are widely used alternatives of
multiple-choice questions or human annotators for large language model (LLM)
evaluation. Their efficacy shines in evaluating long-form responses, serving a
critical role as evaluators of leaderboards and as proxies to align LLMs via
reinforcement learning. However, despite their popularity, their effectiveness
in diverse contexts, such as non-English prompts, factual verification, or
challenging questions, remains unexplored. In this paper, we conduct a
comprehensive analysis of automated evaluators, reporting several key findings
on their behavior. First, we discover that English evaluation capabilities
significantly influence language-specific evaluation capabilities, often more
than the language proficiency itself, enabling evaluators trained in English to
easily transfer their skills to other languages. Second, we identify critical
shortcomings, where LLMs fail to detect and penalize errors, such as factual
inaccuracies, cultural misrepresentations, and the presence of unwanted
language. Finally, we find that state-of-the-art evaluators struggle with
challenging prompts, in either English or Korean, underscoring their
limitations in assessing or generating complex reasoning questions. We release
the dataset and codes used.


---

**[93. [2503.06029] SmartBench: Is Your LLM Truly a Good Chinese Smartphone Assistant?](https://arxiv.org/pdf/2503.06029.pdf)** (2025-03-11)

*Xudong Lu, Haohao Gao, Renshou Wu, Shuai Ren, Xiaoxin Chen, Hongsheng Li, Fangyuan Li*

  Large Language Models (LLMs) have become integral to daily life, especially
advancing as intelligent assistants through on-device deployment on
smartphones. However, existing LLM evaluation benchmarks predominantly focus on
objective tasks like mathematics and coding in English, which do not
necessarily reflect the practical use cases of on-device LLMs in real-world
mobile scenarios, especially for Chinese users. To address these gaps, we
introduce SmartBench, the first benchmark designed to evaluate the capabilities
of on-device LLMs in Chinese mobile contexts. We analyze functionalities
provided by representative smartphone manufacturers and divide them into five
categories: text summarization, text Q\&A, information extraction, content
creation, and notification management, further detailed into 20 specific tasks.
For each task, we construct high-quality datasets comprising 50 to 200
question-answer pairs that reflect everyday mobile interactions, and we develop
automated evaluation criteria tailored for these tasks. We conduct
comprehensive evaluations of on-device LLMs and MLLMs using SmartBench and also
assess their performance after quantized deployment on real smartphone NPUs.
Our contributions provide a standardized framework for evaluating on-device
LLMs in Chinese, promoting further development and optimization in this
critical area. Code and data will be available at
https://github.com/Lucky-Lance/SmartBench.


---

**[94. [2410.09893] RMB: Comprehensively Benchmarking Reward Models in LLM Alignment](https://arxiv.org/pdf/2410.09893.pdf)** (2025-04-07)

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

**[95. [2406.09136] Chain of Preference Optimization: Improving Chain-of-Thought Reasoning
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

**[96. [2401.05399] Automated Assessment of Students' Code Comprehension using LLMs](https://arxiv.org/pdf/2401.05399.pdf)** (2024-01-12)

*Priti Oli, Rabin Banjade, Jeevan Chapagain, Vasile Rus*

  Assessing student's answers and in particular natural language answers is a
crucial challenge in the field of education. Advances in machine learning,
including transformer-based models such as Large Language Models(LLMs), have
led to significant progress in various natural language tasks. Nevertheless,
amidst the growing trend of evaluating LLMs across diverse tasks, evaluating
LLMs in the realm of automated answer assesment has not received much
attention. To address this gap, we explore the potential of using LLMs for
automated assessment of student's short and open-ended answer. Particularly, we
use LLMs to compare students' explanations with expert explanations in the
context of line-by-line explanations of computer programs.
  For comparison purposes, we assess both Large Language Models (LLMs) and
encoder-based Semantic Textual Similarity (STS) models in the context of
assessing the correctness of students' explanation of computer code. Our
findings indicate that LLMs, when prompted in few-shot and chain-of-thought
setting perform comparable to fine-tuned encoder-based models in evaluating
students' short answers in programming domain.


---

**[97. [2406.13805] WikiContradict: A Benchmark for Evaluating LLMs on Real-World Knowledge
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

**[98. [2309.16938] "I'd Like to Have an Argument, Please": Argumentative Reasoning in Large
  Language Models](https://arxiv.org/pdf/2309.16938.pdf)** (2024-06-11)

*Adrian de Wynter, Tangming Yuan*

  We evaluate two large language models (LLMs) ability to perform argumentative
reasoning. We experiment with argument mining (AM) and argument pair extraction
(APE), and evaluate the LLMs' ability to recognize arguments under
progressively more abstract input and output (I/O) representations (e.g.,
arbitrary label sets, graphs, etc.). Unlike the well-known evaluation of prompt
phrasings, abstraction evaluation retains the prompt's phrasing but tests
reasoning capabilities. We find that scoring-wise the LLMs match or surpass the
SOTA in AM and APE, and under certain I/O abstractions LLMs perform well, even
beating chain-of-thought--we call this symbolic prompting. However, statistical
analysis on the LLMs outputs when subject to small, yet still human-readable,
alterations in the I/O representations (e.g., asking for BIO tags as opposed to
line numbers) showed that the models are not performing reasoning. This
suggests that LLM applications to some tasks, such as data labelling and paper
reviewing, must be done with care.


---

**[99. [2311.10733] Proceedings of the 3rd International Workshop on Mining and Learning in
  the Legal Domain (MLLD-23)](https://arxiv.org/pdf/2311.10733.pdf)** (2023-11-21)

*Masoud Makrehchi, Dell Zhang, Alina Petrova, John Armour*

  This is the Proceedings of the 3rd International Workshop on Mining and
Learning in the Legal Domain (MLLD-23) which took place in conjunction with the
32nd ACM International Conference on Information and Knowledge Management
(CIKM-2023) at the University of Birmingham, Birmingham, UK on Sunday 22nd
October 2023.


---

**[100. [2305.07609] Is ChatGPT Fair for Recommendation? Evaluating Fairness in Large
  Language Model Recommendation](https://arxiv.org/pdf/2305.07609.pdf)** (2023-10-18)

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

**[101. [2502.14074] Investigating Non-Transitivity in LLM-as-a-Judge](https://arxiv.org/pdf/2502.14074.pdf)** (2025-03-07)

*Yi Xu, Laura Ruis, Tim Rocktäschel, Robert Kirk*

  Automatic evaluation methods based on large language models (LLMs) are
emerging as the standard tool for assessing the instruction-following abilities
of LLM-based agents. The most common method in this paradigm, pairwise
comparisons with a baseline model, critically depends on the assumption of
transitive preferences. However, the validity of this assumption remains
largely unexplored. In this study, we investigate the presence of
non-transitivity within the AlpacaEval framework and analyze its effects on
model rankings. We find that LLM judges exhibit non-transitive preferences,
leading to rankings that are sensitive to the choice of the baseline model. To
mitigate this issue, we show that round-robin tournaments combined with
Bradley-Terry models of preference can produce more reliable rankings. Notably,
our method increases both the Spearman correlation and the Kendall correlation
with Chatbot Arena (95.0% -> 96.4% and 82.1% -> 86.3% respectively). To address
the computational cost of round-robin tournaments, we propose Swiss-Wise
Iterative Matchmaking (Swim) tournaments, using a dynamic matching strategy to
capture the benefits of round-robin tournaments while maintaining computational
efficiency.


---

**[102. [2409.07871] Objection Overruled! Lay People can Distinguish Large Language Models
  from Lawyers, but still Favour Advice from an LLM](https://arxiv.org/pdf/2409.07871.pdf)** (2025-03-24)

*Eike Schneiders, Tina Seabrooke, Joshua Krook, Richard Hyde, Natalie Leesakul, Jeremie Clos, Joel Fischer*

  Large Language Models (LLMs) are seemingly infiltrating every domain, and the
legal context is no exception. In this paper, we present the results of three
experiments (total N = 288) that investigated lay people's willingness to act
upon, and their ability to discriminate between, LLM- and lawyer-generated
legal advice. In Experiment 1, participants judged their willingness to act on
legal advice when the source of the advice was either known or unknown. When
the advice source was unknown, participants indicated that they were
significantly more willing to act on the LLM-generated advice. The result of
the source unknown condition was replicated in Experiment 2. Intriguingly,
despite participants indicating higher willingness to act on LLM-generated
advice in Experiments 1 and 2, participants discriminated between the LLM- and
lawyer-generated texts significantly above chance-level in Experiment 3.
Lastly, we discuss potential explanations and risks of our findings,
limitations and future work.


---

**[103. [2412.17259] LegalAgentBench: Evaluating LLM Agents in Legal Domain](https://arxiv.org/pdf/2412.17259.pdf)** (2024-12-24)

*Haitao Li, Junjie Chen, Jingli Yang, Qingyao Ai, Wei Jia, Youfeng Liu, Kai Lin, Yueyue Wu, Guozhi Yuan, Yiran Hu, Wuyue Wang, Yiqun Liu, Minlie Huang*

  With the increasing intelligence and autonomy of LLM agents, their potential
applications in the legal domain are becoming increasingly apparent. However,
existing general-domain benchmarks cannot fully capture the complexity and
subtle nuances of real-world judicial cognition and decision-making. Therefore,
we propose LegalAgentBench, a comprehensive benchmark specifically designed to
evaluate LLM Agents in the Chinese legal domain. LegalAgentBench includes 17
corpora from real-world legal scenarios and provides 37 tools for interacting
with external knowledge. We designed a scalable task construction framework and
carefully annotated 300 tasks. These tasks span various types, including
multi-hop reasoning and writing, and range across different difficulty levels,
effectively reflecting the complexity of real-world legal scenarios. Moreover,
beyond evaluating final success, LegalAgentBench incorporates keyword analysis
during intermediate processes to calculate progress rates, enabling more
fine-grained evaluation. We evaluated eight popular LLMs, highlighting the
strengths, limitations, and potential areas for improvement of existing models
and methods. LegalAgentBench sets a new benchmark for the practical application
of LLMs in the legal domain, with its code and data available at
\url{https://github.com/CSHaitao/LegalAgentBench}.


---

**[104. [2410.02736] Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge](https://arxiv.org/pdf/2410.02736.pdf)** (2024-10-07)

*Jiayi Ye, Yanbo Wang, Yue Huang, Dongping Chen, Qihui Zhang, Nuno Moniz, Tian Gao, Werner Geyer, Chao Huang, Pin-Yu Chen, Nitesh V Chawla, Xiangliang Zhang*

  LLM-as-a-Judge has been widely utilized as an evaluation method in various
benchmarks and served as supervised rewards in model training. However, despite
their excellence in many domains, potential issues are under-explored,
undermining their reliability and the scope of their utility. Therefore, we
identify 12 key potential biases and propose a new automated bias
quantification framework-CALM-which systematically quantifies and analyzes each
type of bias in LLM-as-a-Judge by using automated and principle-guided
modification. Our experiments cover multiple popular language models, and the
results indicate that while advanced models have achieved commendable overall
performance, significant biases persist in certain specific tasks. Empirical
results suggest that there remains room for improvement in the reliability of
LLM-as-a-Judge. Moreover, we also discuss the explicit and implicit influence
of these biases and give some suggestions for the reliable application of
LLM-as-a-Judge. Our work highlights the need for stakeholders to address these
issues and remind users to exercise caution in LLM-as-a-Judge applications.


---

**[105. [2310.15205] DISC-FinLLM: A Chinese Financial Large Language Model based on Multiple
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

**[106. [2404.12273] FedEval-LLM: Federated Evaluation of Large Language Models on Downstream
  Tasks with Collective Wisdom](https://arxiv.org/pdf/2404.12273.pdf)** (2024-04-19)

*Yuanqin He, Yan Kang, Lixin Fan, Qiang Yang*

  Federated Learning (FL) has emerged as a promising solution for collaborative
training of large language models (LLMs). However, the integration of LLMs into
FL introduces new challenges, particularly concerning the evaluation of LLMs.
Traditional evaluation methods that rely on labeled test sets and
similarity-based metrics cover only a subset of the acceptable answers, thereby
failing to accurately reflect the performance of LLMs on generative tasks.
Meanwhile, although automatic evaluation methods that leverage advanced LLMs
present potential, they face critical risks of data leakage due to the need to
transmit data to external servers and suboptimal performance on downstream
tasks due to the lack of domain knowledge. To address these issues, we propose
a Federated Evaluation framework of Large Language Models, named FedEval-LLM,
that provides reliable performance measurements of LLMs on downstream tasks
without the reliance on labeled test sets and external tools, thus ensuring
strong privacy-preserving capability. FedEval-LLM leverages a consortium of
personalized LLMs from participants as referees to provide domain knowledge and
collective evaluation capability, thus aligning to the respective downstream
tasks and mitigating uncertainties and biases associated with a single referee.
Experimental results demonstrate a significant improvement in the evaluation
capability of personalized evaluation models on downstream tasks. When applied
to FL, these evaluation models exhibit strong agreement with human preference
and RougeL-score on meticulously curated test sets. FedEval-LLM effectively
overcomes the limitations of traditional metrics and the reliance on external
services, making it a promising framework for the evaluation of LLMs within
collaborative training scenarios.


---

**[107. [2410.04601] ProtoMed-LLM: An Automatic Evaluation Framework for Large Language
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

**[108. [2405.05444] Evaluating Students' Open-ended Written Responses with LLMs: Using the
  RAG Framework for GPT-3.5, GPT-4, Claude-3, and Mistral-Large](https://arxiv.org/pdf/2405.05444.pdf)** (2024-05-10)

*Jussi S. Jauhiainen, Agustín Garagorry Guerra*

  Evaluating open-ended written examination responses from students is an
essential yet time-intensive task for educators, requiring a high degree of
effort, consistency, and precision. Recent developments in Large Language
Models (LLMs) present a promising opportunity to balance the need for thorough
evaluation with efficient use of educators' time. In our study, we explore the
effectiveness of LLMs ChatGPT-3.5, ChatGPT-4, Claude-3, and Mistral-Large in
assessing university students' open-ended answers to questions made about
reference material they have studied. Each model was instructed to evaluate 54
answers repeatedly under two conditions: 10 times (10-shot) with a temperature
setting of 0.0 and 10 times with a temperature of 0.5, expecting a total of
1,080 evaluations per model and 4,320 evaluations across all models. The RAG
(Retrieval Augmented Generation) framework was used as the framework to make
the LLMs to process the evaluation of the answers. As of spring 2024, our
analysis revealed notable variations in consistency and the grading outcomes
provided by studied LLMs. There is a need to comprehend strengths and
weaknesses of LLMs in educational settings for evaluating open-ended written
responses. Further comparative research is essential to determine the accuracy
and cost-effectiveness of using LLMs for educational assessments.


---

**[109. [2502.20635] Can LLM Assist in the Evaluation of the Quality of Machine Learning
  Explanations?](https://arxiv.org/pdf/2502.20635.pdf)** (2025-03-03)

*Bo Wang, Yiqiao Li, Jianlong Zhou, Fang Chen*

  EXplainable machine learning (XML) has recently emerged to address the
mystery mechanisms of machine learning (ML) systems by interpreting their
'black box' results. Despite the development of various explanation methods,
determining the most suitable XML method for specific ML contexts remains
unclear, highlighting the need for effective evaluation of explanations. The
evaluating capabilities of the Transformer-based large language model (LLM)
present an opportunity to adopt LLM-as-a-Judge for assessing explanations. In
this paper, we propose a workflow that integrates both LLM-based and human
judges for evaluating explanations. We examine how LLM-based judges evaluate
the quality of various explanation methods and compare their evaluation
capabilities to those of human judges within an iris classification scenario,
employing both subjective and objective metrics. We conclude that while
LLM-based judges effectively assess the quality of explanations using
subjective metrics, they are not yet sufficiently developed to replace human
judges in this role.


---

**[110. [2502.09497] Improve LLM-based Automatic Essay Scoring with Linguistic Features](https://arxiv.org/pdf/2502.09497.pdf)** (2025-02-14)

*Zhaoyi Joey Hou, Alejandro Ciuba, Xiang Lorraine Li*

  Automatic Essay Scoring (AES) assigns scores to student essays, reducing the
grading workload for instructors. Developing a scoring system capable of
handling essays across diverse prompts is challenging due to the flexibility
and diverse nature of the writing task. Existing methods typically fall into
two categories: supervised feature-based approaches and large language model
(LLM)-based methods. Supervised feature-based approaches often achieve higher
performance but require resource-intensive training. In contrast, LLM-based
methods are computationally efficient during inference but tend to suffer from
lower performance. This paper combines these approaches by incorporating
linguistic features into LLM-based scoring. Experimental results show that this
hybrid method outperforms baseline models for both in-domain and out-of-domain
writing prompts.


---

**[111. [2312.04916] EE-LLM: Large-Scale Training and Inference of Early-Exit Large Language
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

**[112. [2402.01830] PiCO: Peer Review in LLMs based on the Consistency Optimization](https://arxiv.org/pdf/2402.01830.pdf)** (2025-02-24)

*Kun-Peng Ning, Shuo Yang, Yu-Yang Liu, Jia-Yu Yao, Zhen-Hui Liu, Yong-Hong Tian, Yibing Song, Li Yuan*

  Existing large language models (LLMs) evaluation methods typically focus on
testing the performance on some closed-environment and domain-specific
benchmarks with human annotations. In this paper, we explore a novel
unsupervised evaluation direction, utilizing peer-review mechanisms to measure
LLMs automatically. In this setting, both open-source and closed-source LLMs
lie in the same environment, capable of answering unlabeled questions and
evaluating each other, where each LLM's response score is jointly determined by
other anonymous ones. To obtain the ability hierarchy among these models, we
assign each LLM a learnable capability parameter to adjust the final ranking.
We formalize it as a constrained optimization problem, intending to maximize
the consistency of each LLM's capabilities and scores. The key assumption
behind is that high-level LLM can evaluate others' answers more accurately than
low-level ones, while higher-level LLM can also achieve higher response scores.
Moreover, we propose three metrics called PEN, CIN, and LIS to evaluate the gap
in aligning human rankings. We perform experiments on multiple datasets with
these metrics, validating the effectiveness of the proposed approach.


---

**[113. [2411.04424] Bayesian Calibration of Win Rate Estimation with LLM Evaluators](https://arxiv.org/pdf/2411.04424.pdf)** (2024-12-25)

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

**[114. [2504.01840] LARGE: Legal Retrieval Augmented Generation Evaluation Tool](https://arxiv.org/pdf/2504.01840.pdf)** (2025-04-03)

*Minhu Park, Hongseok Oh, Eunkyung Choi, Wonseok Hwang*

  Recently, building retrieval-augmented generation (RAG) systems to enhance
the capability of large language models (LLMs) has become a common practice.
Especially in the legal domain, previous judicial decisions play a significant
role under the doctrine of stare decisis which emphasizes the importance of
making decisions based on (retrieved) prior documents. However, the overall
performance of RAG system depends on many components: (1) retrieval corpora,
(2) retrieval algorithms, (3) rerankers, (4) LLM backbones, and (5) evaluation
metrics. Here we propose LRAGE, an open-source tool for holistic evaluation of
RAG systems focusing on the legal domain. LRAGE provides GUI and CLI interfaces
to facilitate seamless experiments and investigate how changes in the
aforementioned five components affect the overall accuracy. We validated LRAGE
using multilingual legal benches including Korean (KBL), English (LegalBench),
and Chinese (LawBench) by demonstrating how the overall accuracy changes when
varying the five components mentioned above. The source code is available at
https://github.com/hoorangyee/LRAGE.


---

**[115. [2408.11729] LLM4VV: Exploring LLM-as-a-Judge for Validation and Verification
  Testsuites](https://arxiv.org/pdf/2408.11729.pdf)** (2024-09-04)

*Zachariah Sollenberger, Jay Patel, Christian Munley, Aaron Jarmusch, Sunita Chandrasekaran*

  Large Language Models (LLM) are evolving and have significantly
revolutionized the landscape of software development. If used well, they can
significantly accelerate the software development cycle. At the same time, the
community is very cautious of the models being trained on biased or sensitive
data, which can lead to biased outputs along with the inadvertent release of
confidential information. Additionally, the carbon footprints and the
un-explainability of these black box models continue to raise questions about
the usability of LLMs.
  With the abundance of opportunities LLMs have to offer, this paper explores
the idea of judging tests used to evaluate compiler implementations of
directive-based programming models as well as probe into the black box of LLMs.
Based on our results, utilizing an agent-based prompting approach and setting
up a validation pipeline structure drastically increased the quality of
DeepSeek Coder, the LLM chosen for the evaluation purposes.


---

**[116. [2411.01996] Culinary Class Wars: Evaluating LLMs using ASH in Cuisine Transfer Task](https://arxiv.org/pdf/2411.01996.pdf)** (2024-11-05)

*Hoonick Lee, Mogan Gim, Donghyeon Park, Donghee Choi, Jaewoo Kang*

  The advent of Large Language Models (LLMs) have shown promise in various
creative domains, including culinary arts. However, many LLMs still struggle to
deliver the desired level of culinary creativity, especially when tasked with
adapting recipes to meet specific cultural requirements. This study focuses on
cuisine transfer-applying elements of one cuisine to another-to assess LLMs'
culinary creativity. We employ a diverse set of LLMs to generate and evaluate
culturally adapted recipes, comparing their evaluations against LLM and human
judgments. We introduce the ASH (authenticity, sensitivity, harmony) benchmark
to evaluate LLMs' recipe generation abilities in the cuisine transfer task,
assessing their cultural accuracy and creativity in the culinary domain. Our
findings reveal crucial insights into both generative and evaluative
capabilities of LLMs in the culinary domain, highlighting strengths and
limitations in understanding and applying cultural nuances in recipe creation.
The code and dataset used in this project will be openly available in
\url{http://github.com/dmis-lab/CulinaryASH}.


---

**[117. [2407.10582] Boosting Zero-Shot Crosslingual Performance using LLM-Based
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

**[118. [2401.17072] SemScore: Automated Evaluation of Instruction-Tuned LLMs based on
  Semantic Textual Similarity](https://arxiv.org/pdf/2401.17072.pdf)** (2024-02-06)

*Ansar Aynetdinov, Alan Akbik*

  Instruction-tuned Large Language Models (LLMs) have recently showcased
remarkable advancements in their ability to generate fitting responses to
natural language instructions. However, many current works rely on manual
evaluation to judge the quality of generated responses. Since such manual
evaluation is time-consuming, it does not easily scale to the evaluation of
multiple models and model variants. In this short paper, we propose a
straightforward but remarkably effective evaluation metric called SemScore, in
which we directly compare model outputs to gold target responses using semantic
textual similarity (STS). We conduct a comparative evaluation of the model
outputs of 12 prominent instruction-tuned LLMs using 8 widely-used evaluation
metrics for text generation. We find that our proposed SemScore metric
outperforms all other, in many cases more complex, evaluation metrics in terms
of correlation to human evaluation. These findings indicate the utility of our
proposed metric for the evaluation of instruction-tuned LLMs.


---

**[119. [2406.08598] Language Model Council: Democratically Benchmarking Foundation Models on
  Highly Subjective Tasks](https://arxiv.org/pdf/2406.08598.pdf)** (2025-03-20)

*Justin Zhao, Flor Miriam Plaza-del-Arco, Benjamin Genchel, Amanda Cercas Curry*

  As Large Language Models (LLMs) continue to evolve, evaluating them remains a
persistent challenge. Many recent evaluations use LLMs as judges to score
outputs from other LLMs, often relying on a single large model like GPT-4o.
However, using a single LLM judge is prone to intra-model bias, and many tasks
- such as those related to emotional intelligence, creative writing, and
persuasiveness - may be too subjective for a single model to judge fairly. We
introduce the Language Model Council (LMC), where a group of LLMs collaborate
to create tests, respond to them, and evaluate each other's responses to
produce a ranking in a democratic fashion. Unlike previous approaches that
focus on reducing cost or bias by using a panel of smaller models, our work
examines the benefits and nuances of a fully inclusive LLM evaluation system.
In a detailed case study on emotional intelligence, we deploy a council of 20
recent LLMs to rank each other on open-ended responses to interpersonal
conflicts. Our results show that the LMC produces rankings that are more
separable and more robust, and through a user study, we show that they are more
consistent with human evaluations than any individual LLM judge. Using all LLMs
for judging can be costly, however, so we use Monte Carlo simulations and
hand-curated sub-councils to study hypothetical council compositions and
discuss the value of the incremental LLM judge.


---

**[120. [2403.06574] AC-EVAL: Evaluating Ancient Chinese Language Understanding in Large
  Language Models](https://arxiv.org/pdf/2403.06574.pdf)** (2024-03-12)

*Yuting Wei, Yuanxing Xu, Xinru Wei, Simin Yang, Yangfu Zhu, Yuqing Li, Di Liu, Bin Wu*

  Given the importance of ancient Chinese in capturing the essence of rich
historical and cultural heritage, the rapid advancements in Large Language
Models (LLMs) necessitate benchmarks that can effectively evaluate their
understanding of ancient contexts. To meet this need, we present AC-EVAL, an
innovative benchmark designed to assess the advanced knowledge and reasoning
capabilities of LLMs within the context of ancient Chinese. AC-EVAL is
structured across three levels of difficulty reflecting different facets of
language comprehension: general historical knowledge, short text understanding,
and long text comprehension. The benchmark comprises 13 tasks, spanning
historical facts, geography, social customs, art, philosophy, classical poetry
and prose, providing a comprehensive assessment framework. Our extensive
evaluation of top-performing LLMs, tailored for both English and Chinese,
reveals a substantial potential for enhancing ancient text comprehension. By
highlighting the strengths and weaknesses of LLMs, AC-EVAL aims to promote
their development and application forward in the realms of ancient Chinese
language education and scholarly research. The AC-EVAL data and evaluation code
are available at https://github.com/yuting-wei/AC-EVAL.


---

**[121. [2305.07507] LeXFiles and LegalLAMA: Facilitating English Multinational Legal
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

**[122. [2504.02881] Better Bill GPT: Comparing Large Language Models against Legal Invoice
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

**[123. [2504.10706] GestureCoach: Rehearsing for Engaging Talks with LLM-Driven Gesture
  Recommendations](https://arxiv.org/pdf/2504.10706.pdf)** (2025-04-16)

*Ashwin Ram, Varsha Suresh, Artin Saberpour Abadian, Vera Demberg, Jürgen Steimle*

  This paper introduces GestureCoach, a system designed to help speakers
deliver more engaging talks by guiding them to gesture effectively during
rehearsal. GestureCoach combines an LLM-driven gesture recommendation model
with a rehearsal interface that proactively cues speakers to gesture
appropriately. Trained on experts' gesturing patterns from TED talks, the model
consists of two modules: an emphasis proposal module, which predicts when to
gesture by identifying gesture-worthy text segments in the presenter notes, and
a gesture identification module, which determines what gesture to use by
retrieving semantically appropriate gestures from a curated gesture database.
Results of a model performance evaluation and user study (N=30) show that the
emphasis proposal module outperforms off-the-shelf LLMs in identifying suitable
gesture regions, and that participants rated the majority of these predicted
regions and their corresponding gestures as highly appropriate. A subsequent
user study (N=10) showed that rehearsing with GestureCoach encouraged speakers
to gesture and significantly increased gesture diversity, resulting in more
engaging talks. We conclude with design implications for future AI-driven
rehearsal systems.


---

**[124. [2412.11536] Let your LLM generate a few tokens and you will reduce the need for
  retrieval](https://arxiv.org/pdf/2412.11536.pdf)** (2024-12-17)

*Hervé Déjean*

  In this paper, we investigate how efficiently large language models (LLM) can
be trained to check whether an answer is already stored in their parametric
memory. We distill an LLM-as-a-judge to compute the IK (I Know) score. We found
that this method is particularly beneficial in the context of
retrieval-assisted augmented generation (RAG), with a respectable accuracy of
80%. It enables a significant reduction (more than 50%) in the number of search
and reranking steps required for certain data sets. We have also introduced the
IK score, which serves as a useful tool for characterising datasets by
facilitating the classification task. Interestingly, through the inclusion of
response tokens as input, our results suggest that only about 20,000 training
samples are required to achieve good performance. The central element of this
work is the use of a teacher model - the LLM as a judge - to generate training
data. We also assess the robustness of the IK classifier by evaluating it with
various types of teachers, including both string-based methods and LLMs, with
the latter providing better results.


---

**[125. [2401.17703] WSC+: Enhancing The Winograd Schema Challenge Using Tree-of-Experts](https://arxiv.org/pdf/2401.17703.pdf)** (2024-02-01)

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

**[126. [2307.13692] ARB: Advanced Reasoning Benchmark for Large Language Models](https://arxiv.org/pdf/2307.13692.pdf)** (2023-07-31)

*Tomohiro Sawada, Daniel Paleka, Alexander Havrilla, Pranav Tadepalli, Paula Vidas, Alexander Kranias, John J. Nay, Kshitij Gupta, Aran Komatsuzaki*

  Large Language Models (LLMs) have demonstrated remarkable performance on
various quantitative reasoning and knowledge benchmarks. However, many of these
benchmarks are losing utility as LLMs get increasingly high scores, despite not
yet reaching expert performance in these domains. We introduce ARB, a novel
benchmark composed of advanced reasoning problems in multiple fields. ARB
presents a more challenging test than prior benchmarks, featuring problems in
mathematics, physics, biology, chemistry, and law. As a subset of ARB, we
introduce a challenging set of math and physics problems which require advanced
symbolic reasoning and domain knowledge. We evaluate recent models such as
GPT-4 and Claude on ARB and demonstrate that current models score well below
50% on more demanding tasks. In order to improve both automatic and assisted
evaluation capabilities, we introduce a rubric-based evaluation approach,
allowing GPT-4 to score its own intermediate reasoning steps. Further, we
conduct a human evaluation of the symbolic subset of ARB, finding promising
agreement between annotators and GPT-4 rubric evaluation scores.


---

**[127. [2502.10868] NitiBench: A Comprehensive Study of LLM Framework Capabilities for Thai
  Legal Question Answering](https://arxiv.org/pdf/2502.10868.pdf)** (2025-03-11)

*Pawitsapak Akarajaradwong, Pirat Pothavorn, Chompakorn Chaksangchaichot, Panuthep Tasawong, Thitiwat Nopparatbundit, Sarana Nutanong*

  The application of large language models (LLMs) in the legal domain holds
significant potential for information retrieval and question answering, yet
Thai legal QA systems face challenges due to a lack of standardized evaluation
benchmarks and the complexity of Thai legal structures. This paper introduces
NitiBench, a benchmark comprising two datasets: the NitiBench-CCL, covering
general Thai financial law, and the NitiBench-Tax, which includes real-world
tax law cases requiring advanced legal reasoning. We evaluate
retrieval-augmented generation (RAG) and long-context LLM-based approaches to
address three key research questions: the impact of domain-specific components
like section-based chunking and cross-referencing, the comparative performance
of different retrievers and LLMs, and the viability of long-context LLMs as an
alternative to RAG. Our results show that section-based chunking significantly
improves retrieval and end-to-end performance, current retrievers struggle with
complex queries, and long-context LLMs still underperform RAG-based systems in
Thai legal QA. To support fair evaluation, we propose tailored multi-label
retrieval metrics and the use of an LLM-as-judge for coverage and contradiction
detection method. These findings highlight the limitations of current Thai
legal NLP solutions and provide a foundation for future research in the field.
We also open-sourced our codes and dataset to available publicly.


---

**[128. [2411.06387] Self-Training Meets Consistency: Improving LLMs' Reasoning with
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

**[129. [2504.02404] AnesBench: Multi-Dimensional Evaluation of LLM Reasoning in
  Anesthesiology](https://arxiv.org/pdf/2504.02404.pdf)** (2025-04-04)

*Xiang Feng, Wentao Jiang, Zengmao Wang, Yong Luo, Pingbo Xu, Baosheng Yu, Hua Jin, Bo Du, Jing Zhang*

  The application of large language models (LLMs) in the medical field has
gained significant attention, yet their reasoning capabilities in more
specialized domains like anesthesiology remain underexplored. In this paper, we
systematically evaluate the reasoning capabilities of LLMs in anesthesiology
and analyze key factors influencing their performance. To this end, we
introduce AnesBench, a cross-lingual benchmark designed to assess
anesthesiology-related reasoning across three levels: factual retrieval (System
1), hybrid reasoning (System 1.x), and complex decision-making (System 2).
Through extensive experiments, we first explore how model characteristics,
including model scale, Chain of Thought (CoT) length, and language
transferability, affect reasoning performance. Then, we further evaluate the
effectiveness of different training strategies, leveraging our curated
anesthesiology-related dataset, including continuous pre-training (CPT) and
supervised fine-tuning (SFT). Additionally, we also investigate how the
test-time reasoning techniques, such as Best-of-N sampling and beam search,
influence reasoning performance, and assess the impact of reasoning-enhanced
model distillation, specifically DeepSeek-R1. We will publicly release
AnesBench, along with our CPT and SFT training datasets and evaluation code at
https://github.com/MiliLab/AnesBench.


---

**[130. [2406.12624] Judging the Judges: Evaluating Alignment and Vulnerabilities in
  LLMs-as-Judges](https://arxiv.org/pdf/2406.12624.pdf)** (2025-01-22)

*Aman Singh Thakur, Kartik Choudhary, Venkat Srinik Ramayapally, Sankaran Vaidyanathan, Dieuwke Hupkes*

  Offering a promising solution to the scalability challenges associated with
human evaluation, the LLM-as-a-judge paradigm is rapidly gaining traction as an
approach to evaluating large language models (LLMs). However, there are still
many open questions about the strengths and weaknesses of this paradigm, and
what potential biases it may hold. In this paper, we present a comprehensive
study of the performance of various LLMs acting as judges, focusing on a clean
scenario in which inter-human agreement is high. Investigating thirteen judge
models of different model sizes and families, judging answers of nine different
'examtaker models' - both base and instruction-tuned - we find that only the
best (and largest) models achieve reasonable alignment with humans. However,
they are still quite far behind inter-human agreement and their assigned scores
may still differ with up to 5 points from human-assigned scores. In terms of
their ranking of the nine exam-taker models, instead, also smaller models and
even the lexical metric contains may provide a reasonable signal. Through error
analysis and other studies, we identify vulnerabilities in judge models, such
as their sensitivity to prompt complexity and length, and a tendency toward
leniency. The fact that even the best judges differ from humans in this
comparatively simple setup suggest that caution may be wise when using judges
in more complex setups. Lastly, our research rediscovers the importance of
using alignment metrics beyond simple percent alignment, showing that judges
with high percent agreement can still assign vastly different scores.


---

**[131. [2504.03846] Do LLM Evaluators Prefer Themselves for a Reason?](https://arxiv.org/pdf/2504.03846.pdf)** (2025-04-08)

*Wei-Lin Chen, Zhepei Wei, Xinyu Zhu, Shi Feng, Yu Meng*

  Large language models (LLMs) are increasingly used as automatic evaluators in
applications such as benchmarking, reward modeling, and self-refinement. Prior
work highlights a potential self-preference bias where LLMs favor their own
generated responses, a tendency often intensifying with model size and
capability. This raises a critical question: Is self-preference detrimental, or
does it simply reflect objectively superior outputs from more capable models?
Disentangling these has been challenging due to the usage of subjective tasks
in previous studies. To address this, we investigate self-preference using
verifiable benchmarks (mathematical reasoning, factual knowledge, code
generation) that allow objective ground-truth assessment. This enables us to
distinguish harmful self-preference (favoring objectively worse responses) from
legitimate self-preference (favoring genuinely superior ones). We conduct
large-scale experiments under controlled evaluation conditions across diverse
model families (e.g., Llama, Qwen, Gemma, Mistral, Phi, GPT, DeepSeek). Our
findings reveal three key insights: (1) Better generators are better judges --
LLM evaluators' accuracy strongly correlates with their task performance, and
much of the self-preference in capable models is legitimate. (2) Harmful
self-preference persists, particularly when evaluator models perform poorly as
generators on specific task instances. Stronger models exhibit more pronounced
harmful bias when they err, though such incorrect generations are less
frequent. (3) Inference-time scaling strategies, such as generating a long
Chain-of-Thought before evaluation, effectively reduce the harmful
self-preference. These results provide a more nuanced understanding of
LLM-based evaluation and practical insights for improving its reliability.


---

**[132. [2410.01553] MedQA-CS: Benchmarking Large Language Models Clinical Skills Using an
  AI-SCE Framework](https://arxiv.org/pdf/2410.01553.pdf)** (2024-10-03)

*Zonghai Yao, Zihao Zhang, Chaolong Tang, Xingyu Bian, Youxia Zhao, Zhichao Yang, Junda Wang, Huixue Zhou, Won Seok Jang, Feiyun Ouyang, Hong Yu*

  Artificial intelligence (AI) and large language models (LLMs) in healthcare
require advanced clinical skills (CS), yet current benchmarks fail to evaluate
these comprehensively. We introduce MedQA-CS, an AI-SCE framework inspired by
medical education's Objective Structured Clinical Examinations (OSCEs), to
address this gap. MedQA-CS evaluates LLMs through two instruction-following
tasks, LLM-as-medical-student and LLM-as-CS-examiner, designed to reflect real
clinical scenarios. Our contributions include developing MedQA-CS, a
comprehensive evaluation framework with publicly available data and expert
annotations, and providing the quantitative and qualitative assessment of LLMs
as reliable judges in CS evaluation. Our experiments show that MedQA-CS is a
more challenging benchmark for evaluating clinical skills than traditional
multiple-choice QA benchmarks (e.g., MedQA). Combined with existing benchmarks,
MedQA-CS enables a more comprehensive evaluation of LLMs' clinical capabilities
for both open- and closed-source LLMs.


---

**[133. [2407.02408] CEB: Compositional Evaluation Benchmark for Fairness in Large Language
  Models](https://arxiv.org/pdf/2407.02408.pdf)** (2025-02-25)

*Song Wang, Peng Wang, Tong Zhou, Yushun Dong, Zhen Tan, Jundong Li*

  As Large Language Models (LLMs) are increasingly deployed to handle various
natural language processing (NLP) tasks, concerns regarding the potential
negative societal impacts of LLM-generated content have also arisen. To
evaluate the biases exhibited by LLMs, researchers have recently proposed a
variety of datasets. However, existing bias evaluation efforts often focus on
only a particular type of bias and employ inconsistent evaluation metrics,
leading to difficulties in comparison across different datasets and LLMs. To
address these limitations, we collect a variety of datasets designed for the
bias evaluation of LLMs, and further propose CEB, a Compositional Evaluation
Benchmark that covers different types of bias across different social groups
and tasks. The curation of CEB is based on our newly proposed compositional
taxonomy, which characterizes each dataset from three dimensions: bias types,
social groups, and tasks. By combining the three dimensions, we develop a
comprehensive evaluation strategy for the bias in LLMs. Our experiments
demonstrate that the levels of bias vary across these dimensions, thereby
providing guidance for the development of specific bias mitigation methods.


---

**[134. [2503.00902] Instruct-of-Reflection: Enhancing Large Language Models Iterative
  Reflection Capabilities via Dynamic-Meta Instruction](https://arxiv.org/pdf/2503.00902.pdf)** (2025-03-04)

*Liping Liu, Chunhong Zhang, Likang Wu, Chuang Zhao, Zheng Hu, Ming He, Jianping Fan*

  Self-reflection for Large Language Models (LLMs) has gained significant
attention. Existing approaches involve models iterating and improving their
previous responses based on LLMs' internal reflection ability or external
feedback. However, recent research has raised doubts about whether intrinsic
self-correction without external feedback may even degrade performance. Based
on our empirical evidence, we find that current static reflection methods may
lead to redundant, drift, and stubborn issues. To mitigate this, we introduce
Instruct-of-Reflection (IoRT), a novel and general reflection framework that
leverages dynamic-meta instruction to enhance the iterative reflection
capability of LLMs. Specifically, we propose the instructor driven by the
meta-thoughts and self-consistency classifier, generates various instructions,
including refresh, stop, and select, to guide the next reflection iteration.
Our experiments demonstrate that IoRT achieves an average improvement of 10.1%
over established baselines in mathematical and commonsense reasoning tasks,
highlighting its efficacy and applicability.


---

**[135. [2309.13701] ALLURE: Auditing and Improving LLM-based Evaluation of Text using
  Iterative In-Context-Learning](https://arxiv.org/pdf/2309.13701.pdf)** (2023-09-28)

*Hosein Hasanbeig, Hiteshi Sharma, Leo Betthauser, Felipe Vieira Frujeri, Ida Momennejad*

  From grading papers to summarizing medical documents, large language models
(LLMs) are evermore used for evaluation of text generated by humans and AI
alike. However, despite their extensive utility, LLMs exhibit distinct failure
modes, necessitating a thorough audit and improvement of their text evaluation
capabilities. Here we introduce ALLURE, a systematic approach to Auditing Large
Language Models Understanding and Reasoning Errors. ALLURE involves comparing
LLM-generated evaluations with annotated data, and iteratively incorporating
instances of significant deviation into the evaluator, which leverages
in-context learning (ICL) to enhance and improve robust evaluation of text by
LLMs. Through this iterative process, we refine the performance of the
evaluator LLM, ultimately reducing reliance on human annotators in the
evaluation process. We anticipate ALLURE to serve diverse applications of LLMs
in various domains related to evaluation of textual data, such as medical
summarization, education, and and productivity.


---

**[136. [2504.13557] Integrating LLMs for Grading and Appeal Resolution in Computer Science
  Education](https://arxiv.org/pdf/2504.13557.pdf)** (2025-04-21)

*I. Aytutuldu, O. Yol, Y. S. Akgul*

  This study explores the integration of Large Language Models (LLMs) into the
grading and appeal resolution process in computer science education. We
introduce AI-PAT, an AI-powered assessment tool that leverages LLMs to evaluate
computer science exams, generate feedback, and address student appeals. AI-PAT
was used to assess over 850 exam submissions and handle 185 appeal cases. Our
multi-model comparison (ChatGPT, Gemini) reveals strong correlations between
model outputs, though significant variability persists depending on
configuration and prompt design. Human graders, while internally consistent,
showed notable inter-rater disagreement, further highlighting subjectivity in
manual evaluation. The appeal process led to grade changes in 74% of cases,
indicating the need for continued refinement of AI evaluation strategies. While
students appreciated the speed and detail of AI feedback, survey responses
revealed trust and fairness concerns. We conclude that AI-PAT offers scalable
benefits for formative assessment and feedback, but must be accompanied by
transparent grading rubrics, human oversight, and appeal mechanisms to ensure
equitable outcomes.


---

**[137. [2503.16515] Highlighting Case Studies in LLM Literature Review of Interdisciplinary
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

**[138. [2411.02451] High-performance automated abstract screening with large language model
  ensembles](https://arxiv.org/pdf/2411.02451.pdf)** (2024-11-25)

*Rohan Sanghera, Arun James Thirunavukarasu, Marc El Khoury, Jessica O'Logbon, Yuqing Chen, Archie Watt, Mustafa Mahmood, Hamid Butt, George Nishimura, Andrew Soltan*

  Large language models (LLMs) excel in tasks requiring processing and
interpretation of input text. Abstract screening is a labour-intensive
component of systematic review involving repetitive application of inclusion
and exclusion criteria on a large volume of studies identified by a literature
search. Here, LLMs (GPT-3.5 Turbo, GPT-4 Turbo, GPT-4o, Llama 3 70B, Gemini 1.5
Pro, and Claude Sonnet 3.5) were trialled on systematic reviews in a full issue
of the Cochrane Library to evaluate their accuracy in zero-shot binary
classification for abstract screening. Trials over a subset of 800 records
identified optimal prompting strategies and demonstrated superior performance
of LLMs to human researchers in terms of sensitivity (LLM-max = 1.000,
human-max = 0.775), precision (LLM-max = 0.927, human-max = 0.911), and
balanced accuracy (LLM-max = 0.904, human-max = 0.865). The best performing
LLM-prompt combinations were trialled across every replicated search result (n
= 119,691), and exhibited consistent sensitivity (range 0.756-1.000) but
diminished precision (range 0.004-0.096). 66 LLM-human and LLM-LLM ensembles
exhibited perfect sensitivity with a maximal precision of 0.458, with less
observed performance drop in larger trials. Significant variation in
performance was observed between reviews, highlighting the importance of
domain-specific validation before deployment. LLMs may reduce the human labour
cost of systematic review with maintained or improved accuracy and sensitivity.
Systematic review is the foundation of evidence synthesis across academic
disciplines, including evidence-based medicine, and LLMs may increase the
efficiency and quality of this mode of research.


---

**[139. [2406.14955] ICLEval: Evaluating In-Context Learning Ability of Large Language Models](https://arxiv.org/pdf/2406.14955.pdf)** (2024-12-10)

*Wentong Chen, Yankai Lin, ZhenHao Zhou, HongYun Huang, Yantao Jia, Zhao Cao, Ji-Rong Wen*

  In-Context Learning (ICL) is a critical capability of Large Language Models
(LLMs) as it empowers them to comprehend and reason across interconnected
inputs. Evaluating the ICL ability of LLMs can enhance their utilization and
deepen our understanding of how this ability is acquired at the training stage.
However, existing evaluation frameworks primarily focus on language abilities
and knowledge, often overlooking the assessment of ICL ability. In this work,
we introduce the ICLEval benchmark to evaluate the ICL abilities of LLMs, which
encompasses two key sub-abilities: exact copying and rule learning. Through the
ICLEval benchmark, we demonstrate that ICL ability is universally present in
different LLMs, and model size is not the sole determinant of ICL efficacy.
Surprisingly, we observe that ICL abilities, particularly copying, develop
early in the pretraining process and stabilize afterward. Our source codes and
benchmark are released at https://github.com/yiye3/ICLEval.


---

**[140. [2310.10049] FATE-LLM: A Industrial Grade Federated Learning Framework for Large
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

**[141. [2410.20774] Are LLM-Judges Robust to Expressions of Uncertainty? Investigating the
  effect of Epistemic Markers on LLM-based Evaluation](https://arxiv.org/pdf/2410.20774.pdf)** (2024-10-29)

*Dongryeol Lee, Yerin Hwang, Yongil Kim, Joonsuk Park, Kyomin Jung*

  In line with the principle of honesty, there has been a growing effort to
train large language models (LLMs) to generate outputs containing epistemic
markers. However, evaluation in the presence of epistemic markers has been
largely overlooked, raising a critical question: Could the use of epistemic
markers in LLM-generated outputs lead to unintended negative consequences? To
address this, we present EMBER, a benchmark designed to assess the robustness
of LLM-judges to epistemic markers in both single and pairwise evaluation
settings. Our findings, based on evaluations using EMBER, reveal that all
tested LLM-judges, including GPT-4o, show a notable lack of robustness in the
presence of epistemic markers. Specifically, we observe a negative bias toward
epistemic markers, with a stronger bias against markers expressing uncertainty.
This suggests that LLM-judges are influenced by the presence of these markers
and do not focus solely on the correctness of the content.


---

**[142. [2502.12468] MCTS-Judge: Test-Time Scaling in LLM-as-a-Judge for Code Correctness
  Evaluation](https://arxiv.org/pdf/2502.12468.pdf)** (2025-02-19)

*Yutong Wang, Pengliang Ji, Chaoqun Yang, Kaixin Li, Ming Hu, Jiaoyang Li, Guillaume Sartoretti*

  The LLM-as-a-Judge paradigm shows promise for evaluating generative content
but lacks reliability in reasoning-intensive scenarios, such as programming.
Inspired by recent advances in reasoning models and shifts in scaling laws, we
pioneer bringing test-time computation into LLM-as-a-Judge, proposing
MCTS-Judge, a resource-efficient, System-2 thinking framework for code
correctness evaluation. MCTS-Judge leverages Monte Carlo Tree Search (MCTS) to
decompose problems into simpler, multi-perspective evaluations. Through a
node-selection strategy that combines self-assessment based on historical
actions in the current trajectory and the Upper Confidence Bound for Trees
based on prior rollouts, MCTS-Judge balances global optimization and refinement
of the current trajectory. We further designed a high-precision,
unit-test-level reward mechanism to encourage the Large Language Model (LLM) to
perform line-by-line analysis. Extensive experiments on three benchmarks and
five LLMs demonstrate the effectiveness of MCTS-Judge, which improves the base
model's accuracy from 41% to 80%, surpassing the o1-series models with 3x fewer
tokens. Further evaluations validate the superiority of its reasoning
trajectory in logic, analytics, thoroughness, and overall quality, while
revealing the test-time scaling law of the LLM-as-a-Judge paradigm.


---

**[143. [2409.14961] UELLM: A Unified and Efficient Approach for LLM Inference Serving](https://arxiv.org/pdf/2409.14961.pdf)** (2024-09-25)

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

**[144. [2407.00379] GraphArena: Evaluating and Exploring Large Language Models on Graph
  Computation](https://arxiv.org/pdf/2407.00379.pdf)** (2025-02-18)

*Jianheng Tang, Qifan Zhang, Yuhan Li, Nuo Chen, Jia Li*

  The ``arms race'' of Large Language Models (LLMs) demands new benchmarks to
examine their progresses. In this paper, we introduce GraphArena, a
benchmarking tool designed to evaluate LLMs on real-world graph computational
problems. It offers a suite of four polynomial-time tasks (e.g., Shortest
Distance) and six NP-complete challenges (e.g., Traveling Salesman Problem).
GraphArena features a rigorous evaluation framework that classifies LLM outputs
as correct, suboptimal (feasible but not optimal), hallucinatory (properly
formatted but infeasible), or missing. Evaluation of over 10 LLMs reveals that
even top-performing LLMs struggle with larger, more complex graph problems and
exhibit hallucination issues. We further explore four potential solutions to
address this issue and improve LLMs on graph computation, including
chain-of-thought prompting, instruction tuning, code writing, and scaling
test-time compute, each demonstrating unique strengths and limitations.
GraphArena complements the existing LLM benchmarks and is open-sourced at
https://github.com/squareRoot3/GraphArena.


---

**[145. [2305.14726] In-Context Demonstration Selection with Cross Entropy Difference](https://arxiv.org/pdf/2305.14726.pdf)** (2023-11-29)

*Dan Iter, Reid Pryzant, Ruochen Xu, Shuohang Wang, Yang Liu, Yichong Xu, Chenguang Zhu*

  Large language models (LLMs) can use in-context demonstrations to improve
performance on zero-shot tasks. However, selecting the best in-context examples
is challenging because model performance can vary widely depending on the
selected examples. We present a cross-entropy difference (CED) method for
selecting in-context demonstrations. Our method is based on the observation
that the effectiveness of in-context demonstrations negatively correlates with
the perplexity of the test example by a language model that was finetuned on
that demonstration. We utilize parameter efficient finetuning to train small
models on training data that are used for computing the cross-entropy
difference between a test example and every candidate in-context demonstration.
This metric is used to rank and select in-context demonstrations independently
for each test input. We evaluate our method on a mix-domain dataset that
combines 8 benchmarks, representing 4 text generation tasks, showing that CED
for in-context demonstration selection can improve performance for a variety of
LLMs.


---

**[146. [2502.20640] LexRAG: Benchmarking Retrieval-Augmented Generation in Multi-Turn Legal
  Consultation Conversation](https://arxiv.org/pdf/2502.20640.pdf)** (2025-03-03)

*Haitao Li, Yifan Chen, Yiran Hu, Qingyao Ai, Junjie Chen, Xiaoyu Yang, Jianhui Yang, Yueyue Wu, Zeyang Liu, Yiqun Liu*

  Retrieval-augmented generation (RAG) has proven highly effective in improving
large language models (LLMs) across various domains. However, there is no
benchmark specifically designed to assess the effectiveness of RAG in the legal
domain, which restricts progress in this area. To fill this gap, we propose
LexRAG, the first benchmark to evaluate RAG systems for multi-turn legal
consultations. LexRAG consists of 1,013 multi-turn dialogue samples and 17,228
candidate legal articles. Each sample is annotated by legal experts and
consists of five rounds of progressive questioning. LexRAG includes two key
tasks: (1) Conversational knowledge retrieval, requiring accurate retrieval of
relevant legal articles based on multi-turn context. (2) Response generation,
focusing on producing legally sound answers. To ensure reliable
reproducibility, we develop LexiT, a legal RAG toolkit that provides a
comprehensive implementation of RAG system components tailored for the legal
domain. Additionally, we introduce an LLM-as-a-judge evaluation pipeline to
enable detailed and effective assessment. Through experimental analysis of
various LLMs and retrieval methods, we reveal the key limitations of existing
RAG systems in handling legal consultation conversations. LexRAG establishes a
new benchmark for the practical application of RAG systems in the legal domain,
with its code and data available at https://github.com/CSHaitao/LexRAG.


---

**[147. [2305.14483] Language Model Self-improvement by Reinforcement Learning Contemplation](https://arxiv.org/pdf/2305.14483.pdf)** (2023-05-25)

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

**[148. [2309.17012] Benchmarking Cognitive Biases in Large Language Models as Evaluators](https://arxiv.org/pdf/2309.17012.pdf)** (2024-09-26)

*Ryan Koo, Minhwa Lee, Vipul Raheja, Jong Inn Park, Zae Myung Kim, Dongyeop Kang*

  Large Language Models are cognitively biased judges. Large Language Models
(LLMs) have recently been shown to be effective as automatic evaluators with
simple prompting and in-context learning. In this work, we assemble 15 LLMs of
four different size ranges and evaluate their output responses by preference
ranking from the other LLMs as evaluators, such as System Star is better than
System Square. We then evaluate the quality of ranking outputs introducing the
Cognitive Bias Benchmark for LLMs as Evaluators (CoBBLEr), a benchmark to
measure six different cognitive biases in LLM evaluation outputs, such as the
Egocentric bias where a model prefers to rank its own outputs highly in
evaluation. We find that LLMs are biased text quality evaluators, exhibiting
strong indications on our bias benchmark (average of 40% of comparisons across
all models) within each of their evaluations that question their robustness as
evaluators. Furthermore, we examine the correlation between human and machine
preferences and calculate the average Rank-Biased Overlap (RBO) score to be
49.6%, indicating that machine preferences are misaligned with humans.
According to our findings, LLMs may still be unable to be utilized for
automatic annotation aligned with human preferences. Our project page is at:
https://minnesotanlp.github.io/cobbler.


---

**[149. [2503.24307] A Systematic Evaluation of LLM Strategies for Mental Health Text
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

**[150. [2307.02288] Performance Comparison of Large Language Models on VNHSGE English
  Dataset: OpenAI ChatGPT, Microsoft Bing Chat, and Google Bard](https://arxiv.org/pdf/2307.02288.pdf)** (2023-07-21)

*Xuan-Quy Dao*

  This paper presents a performance comparison of three large language models
(LLMs), namely OpenAI ChatGPT, Microsoft Bing Chat (BingChat), and Google Bard,
on the VNHSGE English dataset. The performance of BingChat, Bard, and ChatGPT
(GPT-3.5) is 92.4\%, 86\%, and 79.2\%, respectively. The results show that
BingChat is better than ChatGPT and Bard. Therefore, BingChat and Bard can
replace ChatGPT while ChatGPT is not yet officially available in Vietnam. The
results also indicate that BingChat, Bard and ChatGPT outperform Vietnamese
students in English language proficiency. The findings of this study contribute
to the understanding of the potential of LLMs in English language education.
The remarkable performance of ChatGPT, BingChat, and Bard demonstrates their
potential as effective tools for teaching and learning English at the high
school level.


---

**[151. [2412.04947] C$^2$LEVA: Toward Comprehensive and Contamination-Free Language Model
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

**[152. [2502.07912] Elevating Legal LLM Responses: Harnessing Trainable Logical Structures
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

**[153. [2502.03159] PICBench: Benchmarking LLMs for Photonic Integrated Circuits Design](https://arxiv.org/pdf/2502.03159.pdf)** (2025-02-07)

*Yuchao Wu, Xiaofei Yu, Hao Chen, Yang Luo, Yeyu Tong, Yuzhe Ma*

  While large language models (LLMs) have shown remarkable potential in
automating various tasks in digital chip design, the field of Photonic
Integrated Circuits (PICs)-a promising solution to advanced chip
designs-remains relatively unexplored in this context. The design of PICs is
time-consuming and prone to errors due to the extensive and repetitive nature
of code involved in photonic chip design. In this paper, we introduce PICBench,
the first benchmarking and evaluation framework specifically designed to
automate PIC design generation using LLMs, where the generated output takes the
form of a netlist. Our benchmark consists of dozens of meticulously crafted PIC
design problems, spanning from fundamental device designs to more complex
circuit-level designs. It automatically evaluates both the syntax and
functionality of generated PIC designs by comparing simulation outputs with
expert-written solutions, leveraging an open-source simulator. We evaluate a
range of existing LLMs, while also conducting comparative tests on various
prompt engineering techniques to enhance LLM performance in automated PIC
design. The results reveal the challenges and potential of LLMs in the PIC
design domain, offering insights into the key areas that require further
research and development to optimize automation in this field. Our benchmark
and evaluation code is available at https://github.com/PICDA/PICBench.


---

**[154. [2411.10954] Dialectal Toxicity Detection: Evaluating LLM-as-a-Judge Consistency
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

**[155. [2408.09819] CMoralEval: A Moral Evaluation Benchmark for Chinese Large Language
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

**[156. [2503.17599] GPBench: A Comprehensive and Fine-Grained Benchmark for Evaluating Large
  Language Models as General Practitioners](https://arxiv.org/pdf/2503.17599.pdf)** (2025-03-25)

*Zheqing Li, Yiying Yang, Jiping Lang, Wenhao Jiang, Yuhang Zhao, Shuang Li, Dingqian Wang, Zhu Lin, Xuanna Li, Yuze Tang, Jiexian Qiu, Xiaolin Lu, Hongji Yu, Shuang Chen, Yuhua Bi, Xiaofei Zeng, Yixian Chen, Junrong Chen, Lin Yao*

  General practitioners (GPs) serve as the cornerstone of primary healthcare
systems by providing continuous and comprehensive medical services. However,
due to community-oriented nature of their practice, uneven training and
resource gaps, the clinical proficiency among GPs can vary significantly across
regions and healthcare settings. Currently, Large Language Models (LLMs) have
demonstrated great potential in clinical and medical applications, making them
a promising tool for supporting general practice. However, most existing
benchmarks and evaluation frameworks focus on exam-style assessments-typically
multiple-choice question-lack comprehensive assessment sets that accurately
mirror the real-world scenarios encountered by GPs. To evaluate how effectively
LLMs can make decisions in the daily work of GPs, we designed GPBench, which
consists of both test questions from clinical practice and a novel evaluation
framework. The test set includes multiple-choice questions that assess
fundamental knowledge of general practice, as well as realistic, scenario-based
problems. All questions are meticulously annotated by experts, incorporating
rich fine-grained information related to clinical management. The proposed LLM
evaluation framework is based on the competency model for general practice,
providing a comprehensive methodology for assessing LLM performance in
real-world settings. As the first large-model evaluation set targeting GP
decision-making scenarios, GPBench allows us to evaluate current mainstream
LLMs. Expert assessment and evaluation reveal that in areas such as disease
staging, complication recognition, treatment detail, and medication usage,
these models exhibit at least ten major shortcomings. Overall, existing LLMs
are not yet suitable for independent use in real-world GP working scenarios
without human oversight.


---

**[157. [2405.20267] Auto-Arena: Automating LLM Evaluations with Agent Peer Battles and
  Committee Discussions](https://arxiv.org/pdf/2405.20267.pdf)** (2024-10-08)

*Ruochen Zhao, Wenxuan Zhang, Yew Ken Chia, Weiwen Xu, Deli Zhao, Lidong Bing*

  As LLMs continuously evolve, there is an urgent need for a reliable
evaluation method that delivers trustworthy results promptly. Currently, static
benchmarks suffer from inflexibility and unreliability, leading users to prefer
human voting platforms like Chatbot Arena. However, human evaluations require
significant manual effort. To address this, we propose the Auto-Arena, an
innovative framework that automates the entire evaluation process using
LLM-powered agents. Firstly, an LLM examiner generates questions. Then, two LLM
candidates engage in a multi-round peer battle based on individual questions,
aiming at revealing their true performance differences. Finally, a committee of
LLM judges collaboratively discusses and decides the winner, reducing bias and
enhancing fairness. During the peer battles, we observe intriguing scenarios
where the LLM candidates display competitive behaviors and even learn from the
opponents. In our extensive experiments involving 15 recent LLMs, Auto-Arena
shows a 92.14% correlation with human preferences, surpassing all previous
expert-annotated benchmarks without any manual efforts. As a result, Auto-Arena
offers a promising alternative to current human evaluation platforms for
evaluating LLMs automatically.


---

**[158. [2402.01722] Enhancing Large Language Model Performance To Answer Questions and
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

**[159. [2308.03688] AgentBench: Evaluating LLMs as Agents](https://arxiv.org/pdf/2308.03688.pdf)** (2023-10-26)

*Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie Huang, Yuxiao Dong, Jie Tang*

  Large Language Models (LLMs) are becoming increasingly smart and autonomous,
targeting real-world pragmatic missions beyond traditional NLP tasks. As a
result, there has been an urgent need to evaluate LLMs as agents on challenging
tasks in interactive environments. We present AgentBench, a multi-dimensional
evolving benchmark that currently consists of 8 distinct environments to assess
LLM-as-Agent's reasoning and decision-making abilities in a multi-turn
open-ended generation setting. Our extensive test over 27 API-based and
open-sourced (OSS) LLMs shows that, while top commercial LLMs present a strong
ability of acting as agents in complex environments, there is a significant
disparity in performance between them and OSS competitors. We identify the
typical reasons of failures in environments and LLMs, showing that poor
long-term reasoning, decision-making, and instruction following abilities are
the main obstacles for developing usable LLM agents. Training on code and high
quality multi-turn alignment data could improve agent performance. Datasets,
environments, and an integrated evaluation package for AgentBench are released
at \url{https://github.com/THUDM/AgentBench}.


---

**[160. [2502.12501] Crowd Comparative Reasoning: Unlocking Comprehensive Evaluations for
  LLM-as-a-Judge](https://arxiv.org/pdf/2502.12501.pdf)** (2025-04-08)

*Qiyuan Zhang, Yufei Wang, Yuxin Jiang, Liangyou Li, Chuhan Wu, Yasheng Wang, Xin Jiang, Lifeng Shang, Ruiming Tang, Fuyuan Lyu, Chen Ma*

  LLM-as-a-Judge, which generates chain-of-thought (CoT) judgments, has become
a widely adopted auto-evaluation method. However, its reliability is
compromised by the CoT reasoning's inability to capture comprehensive and
deeper details, often leading to incomplete outcomes. Existing methods mainly
rely on majority voting or criteria expansion, which is insufficient to address
the limitation in CoT. We propose Crowd-based Comparative Evaluation, which
introduces additional crowd responses to compare with the candidate responses,
thereby exposing deeper and more comprehensive details within the candidate
responses. This process effectively guides LLM-as-a-Judge to provide a more
detailed CoT judgment. Extensive experiments demonstrate that our approach
enhances evaluation reliability, achieving an average accuracy gain of 6.7%
across five benchmarks. Moreover, our method produces higher-quality CoTs that
facilitate judge distillation and exhibit superior performance in rejection
sampling for supervised fine-tuning (SFT), referred to as crowd rejection
sampling, thereby enabling more efficient SFT. Our analysis confirms that CoTs
generated by ours are more comprehensive and of higher quality, and evaluation
accuracy improves as inference scales.


---

**[161. [2408.02666] Self-Taught Evaluators](https://arxiv.org/pdf/2408.02666.pdf)** (2024-08-09)

*Tianlu Wang, Ilia Kulikov, Olga Golovneva, Ping Yu, Weizhe Yuan, Jane Dwivedi-Yu, Richard Yuanzhe Pang, Maryam Fazel-Zarandi, Jason Weston, Xian Li*

  Model-based evaluation is at the heart of successful model development -- as
a reward model for training, and as a replacement for human evaluation. To
train such evaluators, the standard approach is to collect a large amount of
human preference judgments over model responses, which is costly and the data
becomes stale as models improve. In this work, we present an approach that aims
to im-prove evaluators without human annotations, using synthetic training data
only. Starting from unlabeled instructions, our iterative self-improvement
scheme generates contrasting model outputs and trains an LLM-as-a-Judge to
produce reasoning traces and final judgments, repeating this training at each
new iteration using the improved predictions. Without any labeled preference
data, our Self-Taught Evaluator can improve a strong LLM (Llama3-70B-Instruct)
from 75.4 to 88.3 (88.7 with majority vote) on RewardBench. This outperforms
commonly used LLM judges such as GPT-4 and matches the performance of the
top-performing reward models trained with labeled examples.


---

**[162. [2406.11370] Fairer Preferences Elicit Improved Human-Aligned Large Language Model
  Judgments](https://arxiv.org/pdf/2406.11370.pdf)** (2024-10-15)

*Han Zhou, Xingchen Wan, Yinhong Liu, Nigel Collier, Ivan Vulić, Anna Korhonen*

  Large language models (LLMs) have shown promising abilities as cost-effective
and reference-free evaluators for assessing language generation quality. In
particular, pairwise LLM evaluators, which compare two generated texts and
determine the preferred one, have been employed in a wide range of
applications. However, LLMs exhibit preference biases and worrying sensitivity
to prompt designs. In this work, we first reveal that the predictive preference
of LLMs can be highly brittle and skewed, even with semantically equivalent
instructions. We find that fairer predictive preferences from LLMs consistently
lead to judgments that are better aligned with humans. Motivated by this
phenomenon, we propose an automatic Zero-shot Evaluation-oriented Prompt
Optimization framework, ZEPO, which aims to produce fairer preference decisions
and improve the alignment of LLM evaluators with human judgments. To this end,
we propose a zero-shot learning objective based on the preference decision
fairness. ZEPO demonstrates substantial performance improvements over
state-of-the-art LLM evaluators, without requiring labeled data, on
representative meta-evaluation benchmarks. Our findings underscore the critical
correlation between preference fairness and human alignment, positioning ZEPO
as an efficient prompt optimizer for bridging the gap between LLM evaluators
and human judgments.


---

**[163. [2503.01150] MiLiC-Eval: Benchmarking Multilingual LLMs for China's Minority
  Languages](https://arxiv.org/pdf/2503.01150.pdf)** (2025-03-04)

*Chen Zhang, Mingxu Tao, Zhiyuan Liao, Yansong Feng*

  Large language models (LLMs) excel in high-resource languages but struggle
with low-resource languages (LRLs), particularly those spoken by minority
communities in China, such as Tibetan, Uyghur, Kazakh, and Mongolian. To
systematically track the progress in these languages, we introduce MiLiC-Eval,
a benchmark designed for minority languages in China, featuring 24K instances
across 9 tasks. MiLiC-Eval focuses on underrepresented writing systems and
provides a fine-grained assessment of linguistic and problem-solving skills.
Our evaluation reveals that LLMs perform poorly on syntax-intensive tasks and
multi-script languages. We further demonstrate how MiLiC-Eval can help advance
LRL research in handling diverse writing systems and understanding the process
of language adaptation.


---

**[164. [2402.15987] Likelihood-based Mitigation of Evaluation Bias in Large Language Models](https://arxiv.org/pdf/2402.15987.pdf)** (2024-10-15)

*Masanari Ohi, Masahiro Kaneko, Ryuto Koike, Mengsay Loem, Naoaki Okazaki*

  Large Language Models (LLMs) are widely used to evaluate natural language
generation tasks as automated metrics. However, the likelihood, a measure of
LLM's plausibility for a sentence, can vary due to superficial differences in
sentences, such as word order and sentence structure. It is therefore possible
that there might be a likelihood bias if LLMs are used for evaluation: they
might overrate sentences with higher likelihoods while underrating those with
lower likelihoods. In this paper, we investigate the presence and impact of
likelihood bias in LLM-based evaluators. We also propose a method to mitigate
the likelihood bias. Our method utilizes highly biased instances as few-shot
examples for in-context learning. Our experiments in evaluating the
data-to-text and grammatical error correction tasks reveal that several LLMs we
test display a likelihood bias. Furthermore, our proposed method successfully
mitigates this bias, also improving evaluation performance (in terms of
correlation of models with human scores) significantly.


---

**[165. [2402.14809] CriticBench: Benchmarking LLMs for Critique-Correct Reasoning](https://arxiv.org/pdf/2402.14809.pdf)** (2024-06-04)

*Zicheng Lin, Zhibin Gou, Tian Liang, Ruilin Luo, Haowei Liu, Yujiu Yang*

  The ability of Large Language Models (LLMs) to critique and refine their
reasoning is crucial for their application in evaluation, feedback provision,
and self-improvement. This paper introduces CriticBench, a comprehensive
benchmark designed to assess LLMs' abilities to critique and rectify their
reasoning across a variety of tasks. CriticBench encompasses five reasoning
domains: mathematical, commonsense, symbolic, coding, and algorithmic. It
compiles 15 datasets and incorporates responses from three LLM families.
Utilizing CriticBench, we evaluate and dissect the performance of 17 LLMs in
generation, critique, and correction reasoning, i.e., GQC reasoning. Our
findings reveal: (1) a linear relationship in GQC capabilities, with
critique-focused training markedly enhancing performance; (2) a task-dependent
variation in correction effectiveness, with logic-oriented tasks being more
amenable to correction; (3) GQC knowledge inconsistencies that decrease as
model size increases; and (4) an intriguing inter-model critiquing dynamic,
where stronger models are better at critiquing weaker ones, while weaker models
can surprisingly surpass stronger ones in their self-critique. We hope these
insights into the nuanced critique-correct reasoning of LLMs will foster
further research in LLM critique and self-improvement.


---

**[166. [2502.18532] CuDIP: Enhancing Theorem Proving in LLMs via Curriculum Learning-based
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

**[167. [2403.03883] SaulLM-7B: A pioneering Large Language Model for Law](https://arxiv.org/pdf/2403.03883.pdf)** (2024-03-08)

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

**[168. [2407.11536] Fine-Tuning Medical Language Models for Enhanced Long-Contextual
  Understanding and Domain Expertise](https://arxiv.org/pdf/2407.11536.pdf)** (2024-07-17)

*Qimin Yang, Rongsheng Wang, Jiexin Chen, Runqi Su, Tao Tan*

  Large Language Models (LLMs) have been widely applied in various professional
fields. By fine-tuning the models using domain specific question and answer
datasets, the professional domain knowledge and Q\&A abilities of these models
have significantly improved, for example, medical professional LLMs that use
fine-tuning of doctor-patient Q\&A data exhibit extraordinary disease
diagnostic abilities. However, we observed that despite improvements in
specific domain knowledge, the performance of medical LLM in long-context
understanding has significantly declined, especially compared to general
language models with similar parameters. The purpose of this study is to
investigate the phenomenon of reduced performance in understanding long-context
in medical LLM. We designed a series of experiments to conduct open-book
professional knowledge exams on all models to evaluate their ability to read
long-context. By adjusting the proportion and quantity of general data and
medical data in the process of fine-tuning, we can determine the best data
composition to optimize the professional model and achieve a balance between
long-context performance and specific domain knowledge.


---

**[169. [2406.17588] LongIns: A Challenging Long-context Instruction-based Exam for LLMs](https://arxiv.org/pdf/2406.17588.pdf)** (2024-06-27)

*Shawn Gavin, Tuney Zheng, Jiaheng Liu, Quehry Que, Noah Wang, Jian Yang, Chenchen Zhang, Wenhao Huang, Wenhu Chen, Ge Zhang*

  The long-context capabilities of large language models (LLMs) have been a hot
topic in recent years. To evaluate the performance of LLMs in different
scenarios, various assessment benchmarks have emerged. However, as most of
these benchmarks focus on identifying key information to answer questions,
which mainly requires the retrieval ability of LLMs, these benchmarks can
partially represent the reasoning performance of LLMs from large amounts of
information. Meanwhile, although LLMs often claim to have context windows of
32k, 128k, 200k, or even longer, these benchmarks fail to reveal the actual
supported length of these LLMs. To address these issues, we propose the LongIns
benchmark dataset, a challenging long-context instruction-based exam for LLMs,
which is built based on the existing instruction datasets. Specifically, in our
LongIns, we introduce three evaluation settings: Global Instruction & Single
Task (GIST), Local Instruction & Single Task (LIST), and Local Instruction &
Multiple Tasks (LIMT). Based on LongIns, we perform comprehensive evaluations
on existing LLMs and have the following important findings: (1). The
top-performing GPT-4 with 128k context length performs poorly on the evaluation
context window of 16k in our LongIns. (2). For the multi-hop reasoning ability
of many existing LLMs, significant efforts are still needed under short context
windows (less than 4k).


---

**[170. [2502.13347] Craw4LLM: Efficient Web Crawling for LLM Pretraining](https://arxiv.org/pdf/2502.13347.pdf)** (2025-02-26)

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

**[171. [2311.04933] Evaluating Large Language Models in Ophthalmology](https://arxiv.org/pdf/2311.04933.pdf)** (2023-11-10)

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

**[172. [2410.15737] Who's Who: Large Language Models Meet Knowledge Conflicts in Practice](https://arxiv.org/pdf/2410.15737.pdf)** (2024-10-22)

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

**[173. [2311.09336] LLMRefine: Pinpointing and Refining Large Language Models via
  Fine-Grained Actionable Feedback](https://arxiv.org/pdf/2311.09336.pdf)** (2024-10-28)

*Wenda Xu, Daniel Deutsch, Mara Finkelstein, Juraj Juraska, Biao Zhang, Zhongtao Liu, William Yang Wang, Lei Li, Markus Freitag*

  Recent large language models (LLM) are leveraging human feedback to improve
their generation quality. However, human feedback is costly to obtain,
especially during inference. In this work, we propose LLMRefine, an inference
time optimization method to refine LLM's output. The core idea is to use a
learned fine-grained feedback model to pinpoint defects and guide LLM to refine
them iteratively. Using original LLM as a proposal of edits, LLMRefine searches
for defect-less text via simulated annealing, trading off the exploration and
exploitation. We conduct experiments on three text generation tasks, including
machine translation, long-form question answering (QA), and topical
summarization. LLMRefine consistently outperforms all baseline approaches,
achieving improvements up to 1.7 MetricX points on translation tasks, 8.1
ROUGE-L on ASQA, 2.2 ROUGE-L on topical summarization.


---

**[174. [2307.02762] PRD: Peer Rank and Discussion Improve Large Language Model based
  Evaluations](https://arxiv.org/pdf/2307.02762.pdf)** (2025-01-03)

*Ruosen Li, Teerth Patel, Xinya Du*

  Nowadays, the quality of responses generated by different modern large
language models (LLMs) is hard to evaluate and compare automatically. Recent
studies suggest and predominantly use LLMs for reference-free evaluation of
open-ended question answering. More specifically, they use the recognized
"strongest" LLM as the evaluator, which conducts pairwise comparisons of
candidate models' answers and provides a ranking score. However, this intuitive
method has multiple problems, such as bringing in self-enhancement (favoring
its own answers) and positional bias. We draw insights and lessons from the
educational domain (Cho & MacArthur, 2011; Walsh, 2014) to improve LLM-based
evaluations. Specifically, we propose (1) the peer rank (PR) algorithm that
takes into account each peer LLM's pairwise preferences of all answer pairs,
and outputs a final ranking of models; and (2) peer discussion (PD), where we
prompt two LLMs to discuss and try to reach a mutual agreement on the
preferences of two answers. We conduct experiments on two benchmark datasets.
We find that our approaches achieve higher accuracy and align better with human
judgments. Interestingly, PR can induce a relatively accurate self-ranking of
models under the anonymous setting, where each model's name is unrevealed. Our
work provides space to explore evaluating models that are hard to compare for
humans.


---

**[175. [2503.07041] TCM-3CEval: A Triaxial Benchmark for Assessing Responses from Large
  Language Models in Traditional Chinese Medicine](https://arxiv.org/pdf/2503.07041.pdf)** (2025-03-11)

*Tianai Huang, Lu Lu, Jiayuan Chen, Lihao Liu, Junjun He, Yuping Zhao, Wenchao Tang, Jie Xu*

  Large language models (LLMs) excel in various NLP tasks and modern medicine,
but their evaluation in traditional Chinese medicine (TCM) is underexplored. To
address this, we introduce TCM3CEval, a benchmark assessing LLMs in TCM across
three dimensions: core knowledge mastery, classical text understanding, and
clinical decision-making. We evaluate diverse models, including international
(e.g., GPT-4o), Chinese (e.g., InternLM), and medical-specific (e.g., PLUSE).
Results show a performance hierarchy: all models have limitations in
specialized subdomains like Meridian & Acupoint theory and Various TCM Schools,
revealing gaps between current capabilities and clinical needs. Models with
Chinese linguistic and cultural priors perform better in classical text
interpretation and clinical reasoning. TCM-3CEval sets a standard for AI
evaluation in TCM, offering insights for optimizing LLMs in culturally grounded
medical domains. The benchmark is available on Medbench's TCM track, aiming to
assess LLMs' TCM capabilities in basic knowledge, classic texts, and clinical
decision-making through multidimensional questions and real cases.


---

**[176. [2411.18019] A Real-World Benchmark for Evaluating Fine-Grained Issue Solving
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

**[177. [2411.08324] Are LLMs Prescient? A Continuous Evaluation using Daily News as the
  Oracle](https://arxiv.org/pdf/2411.08324.pdf)** (2024-11-14)

*Hui Dai, Ryan Teehan, Mengye Ren*

  Many existing evaluation benchmarks for Large Language Models (LLMs) quickly
become outdated due to the emergence of new models and training data. These
benchmarks also fall short in assessing how LLM performance changes over time,
as they consist of static questions without a temporal dimension. To address
these limitations, we propose using future event prediction as a continuous
evaluation method to assess LLMs' temporal generalization and forecasting
abilities. Our benchmark, Daily Oracle, automatically generates question-answer
(QA) pairs from daily news, challenging LLMs to predict "future" event
outcomes. Our findings reveal that as pre-training data becomes outdated, LLM
performance degrades over time. While Retrieval Augmented Generation (RAG) has
the potential to enhance prediction accuracy, the performance degradation
pattern persists, highlighting the need for continuous model updates.


---

**[178. [2504.03716] Ethical AI on the Waitlist: Group Fairness Evaluation of LLM-Aided Organ
  Allocation](https://arxiv.org/pdf/2504.03716.pdf)** (2025-04-08)

*Hannah Murray, Brian Hyeongseok Kim, Isabelle Lee, Jason Byun, Dani Yogatama, Evi Micha*

  Large Language Models (LLMs) are becoming ubiquitous, promising automation
even in high-stakes scenarios. However, existing evaluation methods often fall
short -- benchmarks saturate, accuracy-based metrics are overly simplistic, and
many inherently ambiguous problems lack a clear ground truth. Given these
limitations, evaluating fairness becomes complex. To address this, we reframe
fairness evaluation using Borda scores, a method from voting theory, as a
nuanced yet interpretable metric for measuring fairness. Using organ allocation
as a case study, we introduce two tasks: (1) Choose-One and (2) Rank-All. In
Choose-One, LLMs select a single candidate for a kidney, and we assess fairness
across demographics using proportional parity. In Rank-All, LLMs rank all
candidates for a kidney, reflecting real-world allocation processes. Since
traditional fairness metrics do not account for ranking, we propose a novel
application of Borda scoring to capture biases. Our findings highlight the
potential of voting-based metrics to provide a richer, more multifaceted
evaluation of LLM fairness.


---

**[179. [2408.16100] LLMSecCode: Evaluating Large Language Models for Secure Coding](https://arxiv.org/pdf/2408.16100.pdf)** (2024-08-30)

*Anton Rydén, Erik Näslund, Elad Michael Schiller, Magnus Almgren*

  The rapid deployment of Large Language Models (LLMs) requires careful
consideration of their effect on cybersecurity. Our work aims to improve the
selection process of LLMs that are suitable for facilitating Secure Coding
(SC). This raises challenging research questions, such as (RQ1) Which
functionality can streamline the LLM evaluation? (RQ2) What should the
evaluation measure? (RQ3) How to attest that the evaluation process is
impartial? To address these questions, we introduce LLMSecCode, an open-source
evaluation framework designed to assess LLM SC capabilities objectively.
  We validate the LLMSecCode implementation through experiments. When varying
parameters and prompts, we find a 10% and 9% difference in performance,
respectively. We also compare some results to reliable external actors, where
our results show a 5% difference.
  We strive to ensure the ease of use of our open-source framework and
encourage further development by external actors. With LLMSecCode, we hope to
encourage the standardization and benchmarking of LLMs' capabilities in
security-oriented code and tasks.


---

**[180. [2401.16212] Better Call GPT, Comparing Large Language Models Against Lawyers](https://arxiv.org/pdf/2401.16212.pdf)** (2024-01-30)

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

**[181. [2407.16833] Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive
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

**[182. [2502.20747] Measuring Determinism in Large Language Models for Software Code Review](https://arxiv.org/pdf/2502.20747.pdf)** (2025-03-03)

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

**[183. [2310.17631] JudgeLM: Fine-tuned Large Language Models are Scalable Judges](https://arxiv.org/pdf/2310.17631.pdf)** (2025-03-04)

*Lianghui Zhu, Xinggang Wang, Xinlong Wang*

  Evaluating Large Language Models (LLMs) in open-ended scenarios is
challenging because existing benchmarks and metrics can not measure them
comprehensively. To address this problem, we propose to fine-tune LLMs as
scalable judges (JudgeLM) to evaluate LLMs efficiently and effectively in
open-ended benchmarks. We first propose a comprehensive, large-scale,
high-quality dataset containing task seeds, LLMs-generated answers, and
GPT-4-generated judgments for fine-tuning high-performance judges, as well as a
new benchmark for evaluating the judges. We train JudgeLM at different scales
from 7B, 13B, to 33B parameters, and conduct a systematic analysis of its
capabilities and behaviors. We then analyze the key biases in fine-tuning LLM
as a judge and consider them as position bias, knowledge bias, and format bias.
To address these issues, JudgeLM introduces a bag of techniques including swap
augmentation, reference support, and reference drop, which clearly enhance the
judge's performance. JudgeLM obtains the state-of-the-art judge performance on
both the existing PandaLM benchmark and our proposed new benchmark. Our JudgeLM
is efficient and the JudgeLM-7B only needs 3 minutes to judge 5K samples with 8
A100 GPUs. JudgeLM obtains high agreement with the teacher judge, achieving an
agreement exceeding 90% that even surpasses human-to-human agreement. JudgeLM
also demonstrates extended capabilities in being judges of the single answer,
multimodal models, multiple answers, multi-turn chat, etc. Code is available at
https://github.com/baaivision/JudgeLM.


---

**[184. [2503.23566] When LLM Therapists Become Salespeople: Evaluating Large Language Models
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

**[185. [2405.20389] Designing an Evaluation Framework for Large Language Models in Astronomy
  Research](https://arxiv.org/pdf/2405.20389.pdf)** (2024-06-03)

*John F. Wu, Alina Hyk, Kiera McCormick, Christine Ye, Simone Astarita, Elina Baral, Jo Ciuca, Jesse Cranney, Anjalie Field, Kartheik Iyer, Philipp Koehn, Jenn Kotler, Sandor Kruk, Michelle Ntampaka, Charles O'Neill, Joshua E. G. Peek, Sanjib Sharma, Mikaeel Yunus*

  Large Language Models (LLMs) are shifting how scientific research is done. It
is imperative to understand how researchers interact with these models and how
scientific sub-communities like astronomy might benefit from them. However,
there is currently no standard for evaluating the use of LLMs in astronomy.
Therefore, we present the experimental design for an evaluation study on how
astronomy researchers interact with LLMs. We deploy a Slack chatbot that can
answer queries from users via Retrieval-Augmented Generation (RAG); these
responses are grounded in astronomy papers from arXiv. We record and anonymize
user questions and chatbot answers, user upvotes and downvotes to LLM
responses, user feedback to the LLM, and retrieved documents and similarity
scores with the query. Our data collection method will enable future dynamic
evaluations of LLM tools for astronomy.


---

**[186. [2402.17256] Beyond the Known: Investigating LLMs Performance on Out-of-Domain Intent
  Detection](https://arxiv.org/pdf/2402.17256.pdf)** (2024-03-05)

*Pei Wang, Keqing He, Yejie Wang, Xiaoshuai Song, Yutao Mou, Jingang Wang, Yunsen Xian, Xunliang Cai, Weiran Xu*

  Out-of-domain (OOD) intent detection aims to examine whether the user's query
falls outside the predefined domain of the system, which is crucial for the
proper functioning of task-oriented dialogue (TOD) systems. Previous methods
address it by fine-tuning discriminative models. Recently, some studies have
been exploring the application of large language models (LLMs) represented by
ChatGPT to various downstream tasks, but it is still unclear for their ability
on OOD detection task.This paper conducts a comprehensive evaluation of LLMs
under various experimental settings, and then outline the strengths and
weaknesses of LLMs. We find that LLMs exhibit strong zero-shot and few-shot
capabilities, but is still at a disadvantage compared to models fine-tuned with
full resource. More deeply, through a series of additional analysis
experiments, we discuss and summarize the challenges faced by LLMs and provide
guidance for future work including injecting domain knowledge, strengthening
knowledge transfer from IND(In-domain) to OOD, and understanding long
instructions.


---

**[187. [2503.14258] JuDGE: Benchmarking Judgment Document Generation for Chinese Legal
  System](https://arxiv.org/pdf/2503.14258.pdf)** (2025-03-21)

*Weihang Su, Baoqing Yue, Qingyao Ai, Yiran Hu, Jiaqi Li, Changyue Wang, Kaiyuan Zhang, Yueyue Wu, Yiqun Liu*

  This paper introduces JuDGE (Judgment Document Generation Evaluation), a
novel benchmark for evaluating the performance of judgment document generation
in the Chinese legal system. We define the task as generating a complete legal
judgment document from the given factual description of the case. To facilitate
this benchmark, we construct a comprehensive dataset consisting of factual
descriptions from real legal cases, paired with their corresponding full
judgment documents, which serve as the ground truth for evaluating the quality
of generated documents. This dataset is further augmented by two external legal
corpora that provide additional legal knowledge for the task: one comprising
statutes and regulations, and the other consisting of a large collection of
past judgment documents. In collaboration with legal professionals, we
establish a comprehensive automated evaluation framework to assess the quality
of generated judgment documents across various dimensions. We evaluate various
baseline approaches, including few-shot in-context learning, fine-tuning, and a
multi-source retrieval-augmented generation (RAG) approach, using both general
and legal-domain LLMs. The experimental results demonstrate that, while RAG
approaches can effectively improve performance in this task, there is still
substantial room for further improvement. All the codes and datasets are
available at: https://github.com/oneal2000/JuDGE.


---

**[188. [2303.09384] LLMSecEval: A Dataset of Natural Language Prompts for Security
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

**[189. [2306.02561] LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and
  Generative Fusion](https://arxiv.org/pdf/2306.02561.pdf)** (2023-07-04)

*Dongfu Jiang, Xiang Ren, Bill Yuchen Lin*

  We present LLM-Blender, an ensembling framework designed to attain
consistently superior performance by leveraging the diverse strengths of
multiple open-source large language models (LLMs). Our framework consists of
two modules: PairRanker and GenFuser, addressing the observation that optimal
LLMs for different examples can significantly vary. PairRanker employs a
specialized pairwise comparison method to distinguish subtle differences
between candidate outputs. It jointly encodes the input text and a pair of
candidates, using cross-attention encoders to determine the superior one. Our
results demonstrate that PairRanker exhibits the highest correlation with
ChatGPT-based ranking. Then, GenFuser aims to merge the top-ranked candidates,
generating an improved output by capitalizing on their strengths and mitigating
their weaknesses. To facilitate large-scale evaluation, we introduce a
benchmark dataset, MixInstruct, which is a mixture of multiple instruction
datasets featuring oracle pairwise comparisons. Our LLM-Blender significantly
outperform individual LLMs and baseline methods across various metrics,
establishing a substantial performance gap.


---

**[190. [2404.13925] MARIO Eval: Evaluate Your Math LLM with your Math LLM--A mathematical
  dataset evaluation toolkit](https://arxiv.org/pdf/2404.13925.pdf)** (2024-04-23)

*Boning Zhang, Chengxi Li, Kai Fan*

  Large language models (LLMs) have been explored in a variety of reasoning
tasks including solving of mathematical problems. Each math dataset typically
includes its own specially designed evaluation script, which, while suitable
for its intended use, lacks generalizability across different datasets.
Consequently, updates and adaptations to these evaluation tools tend to occur
without being systematically reported, leading to inconsistencies and obstacles
to fair comparison across studies. To bridge this gap, we introduce a
comprehensive mathematical evaluation toolkit that not only utilizes a python
computer algebra system (CAS) for its numerical accuracy, but also integrates
an optional LLM, known for its considerable natural language processing
capabilities. To validate the effectiveness of our toolkit, we manually
annotated two distinct datasets. Our experiments demonstrate that the toolkit
yields more robust evaluation results compared to prior works, even without an
LLM. Furthermore, when an LLM is incorporated, there is a notable enhancement.
The code for our method will be made available at
\url{https://github.com/MARIO-Math-Reasoning/math_evaluation}.


---

**[191. [2404.03543] CodeEditorBench: Evaluating Code Editing Capability of Large Language
  Models](https://arxiv.org/pdf/2404.03543.pdf)** (2025-04-09)

*Jiawei Guo, Ziming Li, Xueling Liu, Kaijing Ma, Tianyu Zheng, Zhouliang Yu, Ding Pan, Yizhi LI, Ruibo Liu, Yue Wang, Shuyue Guo, Xingwei Qu, Xiang Yue, Ge Zhang, Wenhu Chen, Jie Fu*

  Large Language Models (LLMs) for code are rapidly evolving, with code editing
emerging as a critical capability. We introduce CodeEditorBench, an evaluation
framework designed to rigorously assess the performance of LLMs in code editing
tasks, including debugging, translating, polishing, and requirement switching.
Unlike existing benchmarks focusing solely on code generation, CodeEditorBench
emphasizes real-world scenarios and practical aspects of software development.
We curate diverse coding challenges and scenarios from five sources, covering
various programming languages, complexity levels, and editing tasks. Evaluation
of 19 LLMs reveals that closed-source models (particularly Gemini-Ultra and
GPT-4), outperform open-source models in CodeEditorBench, highlighting
differences in model performance based on problem types and prompt
sensitivities. CodeEditorBench aims to catalyze advancements in LLMs by
providing a robust platform for assessing code editing capabilities. We will
release all prompts and datasets to enable the community to expand the dataset
and benchmark emerging LLMs. By introducing CodeEditorBench, we contribute to
the advancement of LLMs in code editing and provide a valuable resource for
researchers and practitioners.


---

**[192. [2402.10866] EcoRank: Budget-Constrained Text Re-ranking Using Large Language Models](https://arxiv.org/pdf/2402.10866.pdf)** (2024-05-29)

*Muhammad Shihab Rashid, Jannat Ara Meem, Yue Dong, Vagelis Hristidis*

  Large Language Models (LLMs) have achieved state-of-the-art performance in
text re-ranking. This process includes queries and candidate passages in the
prompts, utilizing pointwise, listwise, and pairwise prompting strategies. A
limitation of these ranking strategies with LLMs is their cost: the process can
become expensive due to API charges, which are based on the number of input and
output tokens. We study how to maximize the re-ranking performance given a
budget, by navigating the vast search spaces of prompt choices, LLM APIs, and
budget splits. We propose a suite of budget-constrained methods to perform text
re-ranking using a set of LLM APIs. Our most efficient method, called EcoRank,
is a two-layered pipeline that jointly optimizes decisions regarding budget
allocation across prompt strategies and LLM APIs. Our experimental results on
four popular QA and passage reranking datasets show that EcoRank outperforms
other budget-aware supervised and unsupervised baselines.


---

**[193. [2502.16614] CodeCriticBench: A Holistic Code Critique Benchmark for Large Language
  Models](https://arxiv.org/pdf/2502.16614.pdf)** (2025-02-25)

*Alexander Zhang, Marcus Dong, Jiaheng Liu, Wei Zhang, Yejie Wang, Jian Yang, Ge Zhang, Tianyu Liu, Zhongyuan Peng, Yingshui Tan, Yuanxing Zhang, Zhexu Wang, Weixun Wang, Yancheng He, Ken Deng, Wangchunshu Zhou, Wenhao Huang, Zhaoxiang Zhang*

  The critique capacity of Large Language Models (LLMs) is essential for
reasoning abilities, which can provide necessary suggestions (e.g., detailed
analysis and constructive feedback). Therefore, how to evaluate the critique
capacity of LLMs has drawn great attention and several critique benchmarks have
been proposed. However, existing critique benchmarks usually have the following
limitations: (1). Focusing on diverse reasoning tasks in general domains and
insufficient evaluation on code tasks (e.g., only covering code generation
task), where the difficulty of queries is relatively easy (e.g., the code
queries of CriticBench are from Humaneval and MBPP). (2). Lacking comprehensive
evaluation from different dimensions. To address these limitations, we
introduce a holistic code critique benchmark for LLMs called CodeCriticBench.
Specifically, our CodeCriticBench includes two mainstream code tasks (i.e.,
code generation and code QA) with different difficulties. Besides, the
evaluation protocols include basic critique evaluation and advanced critique
evaluation for different characteristics, where fine-grained evaluation
checklists are well-designed for advanced settings. Finally, we conduct
extensive experimental results of existing LLMs, which show the effectiveness
of CodeCriticBench.


---

**[194. [2409.14038] OAEI-LLM: A Benchmark Dataset for Understanding Large Language Model
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

**[195. [2410.08431] oRetrieval Augmented Generation for 10 Large Language Models and its
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

**[196. [2310.05620] LAiW: A Chinese Legal Large Language Models Benchmark](https://arxiv.org/pdf/2310.05620.pdf)** (2024-02-20)

*Yongfu Dai, Duanyu Feng, Jimin Huang, Haochen Jia, Qianqian Xie, Yifang Zhang, Weiguang Han, Wei Tian, Hao Wang*

  General and legal domain LLMs have demonstrated strong performance in various
tasks of LegalAI. However, the current evaluations of these LLMs in LegalAI are
defined by the experts of computer science, lacking consistency with the logic
of legal practice, making it difficult to judge their practical capabilities.
To address this challenge, we are the first to build the Chinese legal LLMs
benchmark LAiW, based on the logic of legal practice. To align with the
thinking process of legal experts and legal practice (syllogism), we divide the
legal capabilities of LLMs from easy to difficult into three levels: basic
information retrieval, legal foundation inference, and complex legal
application. Each level contains multiple tasks to ensure a comprehensive
evaluation. Through automated evaluation of current general and legal domain
LLMs on our benchmark, we indicate that these LLMs may not align with the logic
of legal practice. LLMs seem to be able to directly acquire complex legal
application capabilities but perform poorly in some basic tasks, which may pose
obstacles to their practical application and acceptance by legal experts. To
further confirm the complex legal application capabilities of current LLMs in
legal application scenarios, we also incorporate human evaluation with legal
experts. The results indicate that while LLMs may demonstrate strong
performance, they still require reinforcement of legal logic.


---

**[197. [2408.03907] Decoding Biases: Automated Methods and LLM Judges for Gender Bias
  Detection in Language Models](https://arxiv.org/pdf/2408.03907.pdf)** (2024-08-08)

*Shachi H Kumar, Saurav Sahay, Sahisnu Mazumder, Eda Okur, Ramesh Manuvinakurike, Nicole Beckage, Hsuan Su, Hung-yi Lee, Lama Nachman*

  Large Language Models (LLMs) have excelled at language understanding and
generating human-level text. However, even with supervised training and human
alignment, these LLMs are susceptible to adversarial attacks where malicious
users can prompt the model to generate undesirable text. LLMs also inherently
encode potential biases that can cause various harmful effects during
interactions. Bias evaluation metrics lack standards as well as consensus and
existing methods often rely on human-generated templates and annotations which
are expensive and labor intensive. In this work, we train models to
automatically create adversarial prompts to elicit biased responses from target
LLMs. We present LLM- based bias evaluation metrics and also analyze several
existing automatic evaluation methods and metrics. We analyze the various
nuances of model responses, identify the strengths and weaknesses of model
families, and assess where evaluation methods fall short. We compare these
metrics to human evaluation and validate that the LLM-as-a-Judge metric aligns
with human judgement on bias in response generation.


---

**[198. [2410.16285] Assessing the Performance of Human-Capable LLMs -- Are LLMs Coming for
  Your Job?](https://arxiv.org/pdf/2410.16285.pdf)** (2024-10-23)

*John Mavi, Nathan Summers, Sergio Coronado*

  The current paper presents the development and validation of SelfScore, a
novel benchmark designed to assess the performance of automated Large Language
Model (LLM) agents on help desk and professional consultation tasks. Given the
increasing integration of AI in industries, particularly within customer
service, SelfScore fills a crucial gap by enabling the comparison of automated
agents and human workers. The benchmark evaluates agents on problem complexity
and response helpfulness, ensuring transparency and simplicity in its scoring
system. The study also develops automated LLM agents to assess SelfScore and
explores the benefits of Retrieval-Augmented Generation (RAG) for
domain-specific tasks, demonstrating that automated LLM agents incorporating
RAG outperform those without. All automated LLM agents were observed to perform
better than the human control group. Given these results, the study raises
concerns about the potential displacement of human workers, especially in areas
where AI technologies excel. Ultimately, SelfScore provides a foundational tool
for understanding the impact of AI in help desk environments while advocating
for ethical considerations in the ongoing transition towards automation.


---

**[199. [2407.10499] CIBench: Evaluating Your LLMs with a Code Interpreter Plugin](https://arxiv.org/pdf/2407.10499.pdf)** (2024-11-07)

*Chuyu Zhang, Songyang Zhang, Yingfan Hu, Haowen Shen, Kuikun Liu, Zerun Ma, Fengzhe Zhou, Wenwei Zhang, Xuming He, Dahua Lin, Kai Chen*

  While LLM-Based agents, which use external tools to solve complex problems,
have made significant progress, benchmarking their ability is challenging,
thereby hindering a clear understanding of their limitations. In this paper, we
propose an interactive evaluation framework, named CIBench, to comprehensively
assess LLMs' ability to utilize code interpreters for data science tasks. Our
evaluation framework includes an evaluation dataset and two evaluation modes.
The evaluation dataset is constructed using an LLM-human cooperative approach
and simulates an authentic workflow by leveraging consecutive and interactive
IPython sessions. The two evaluation modes assess LLMs' ability with and
without human assistance. We conduct extensive experiments to analyze the
ability of 24 LLMs on CIBench and provide valuable insights for future LLMs in
code interpreter utilization.


---

**[200. [2410.08437] Autonomous Evaluation of LLMs for Truth Maintenance and Reasoning Tasks](https://arxiv.org/pdf/2410.08437.pdf)** (2025-04-15)

*Rushang Karia, Daniel Bramblett, Daksh Dobhal, Siddharth Srivastava*

  This paper presents AutoEval, a novel benchmark for scaling Large Language
Model (LLM) assessment in formal tasks with clear notions of correctness, such
as truth maintenance in translation and logical reasoning. AutoEval is the
first benchmarking paradigm that offers several key advantages necessary for
scaling objective evaluation of LLMs without human labeling: (a) ability to
evaluate LLMs of increasing sophistication by auto-generating tasks at
different levels of difficulty; (b) auto-generation of ground truth that
eliminates dependence on expensive and time-consuming human annotation; (c) the
use of automatically generated, randomized datasets that mitigate the ability
of successive LLMs to overfit to static datasets used in many contemporary
benchmarks. Empirical analysis shows that an LLM's performance on AutoEval is
highly indicative of its performance on a diverse array of other benchmarks
focusing on translation and reasoning tasks, making it a valuable autonomous
evaluation paradigm in settings where hand-curated datasets can be hard to
obtain and/or update.


---
