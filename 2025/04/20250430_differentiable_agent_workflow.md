**[1. [2307.01085] Some challenges of calibrating differentiable agent-based models](https://arxiv.org/pdf/2307.01085.pdf)** (2023-07-04)

*Arnau Quera-Bofarull, Joel Dyer, Anisoara Calinescu, Michael Wooldridge*

  Agent-based models (ABMs) are a promising approach to modelling and reasoning
about complex systems, yet their application in practice is impeded by their
complexity, discrete nature, and the difficulty of performing parameter
inference and optimisation tasks. This in turn has sparked interest in the
construction of differentiable ABMs as a strategy for combatting these
difficulties, yet a number of challenges remain. In this paper, we discuss and
present experiments that highlight some of these challenges, along with
potential solutions.


---

**[2. [2106.11655] DARTS-PRIME: Regularization and Scheduling Improve Constrained
  Optimization in Differentiable NAS](https://arxiv.org/pdf/2106.11655.pdf)** (2021-10-19)

*Kaitlin Maile, Erwan Lecarpentier, Hervé Luga, Dennis G. Wilson*

  Differentiable Architecture Search (DARTS) is a recent neural architecture
search (NAS) method based on a differentiable relaxation. Due to its success,
numerous variants analyzing and improving parts of the DARTS framework have
recently been proposed. By considering the problem as a constrained bilevel
optimization, we present and analyze DARTS-PRIME, a variant including
improvements to architectural weight update scheduling and regularization
towards discretization. We propose a dynamic schedule based on per-minibatch
network information to make architecture updates more informed, as well as
proximity regularization to promote well-separated discretization. Our results
in multiple domains show that DARTS-PRIME improves both performance and
reliability, comparable to state-of-the-art in differentiable NAS.


---

**[3. [2108.09306] D-DARTS: Distributed Differentiable Architecture Search](https://arxiv.org/pdf/2108.09306.pdf)** (2022-11-02)

*Alexandre Heuillet, Hedi Tabia, Hichem Arioui, Kamal Youcef-Toumi*

  Differentiable ARchiTecture Search (DARTS) is one of the most trending Neural
Architecture Search (NAS) methods. It drastically reduces search cost by
resorting to weight-sharing. However, it also dramatically reduces the search
space, thus excluding potential promising architectures. In this article, we
propose D-DARTS, a solution that addresses this problem by nesting neural
networks at the cell level instead of using weight-sharing to produce more
diversified and specialized architectures. Moreover, we introduce a novel
algorithm that can derive deeper architectures from a few trained cells,
increasing performance and saving computation time. In addition, we also
present an alternative search space (DARTOpti) in which we optimize existing
handcrafted architectures (e.g., ResNet) rather than starting from scratch.
This approach is accompanied by a novel metric that measures the distance
between architectures inside our custom search space. Our solution reaches
competitive performance on multiple computer vision tasks. Code and pretrained
models can be accessed at https://github.com/aheuillet/D-DARTS.


---

**[4. [2409.07129] MVLLaVA: An Intelligent Agent for Unified and Flexible Novel View
  Synthesis](https://arxiv.org/pdf/2409.07129.pdf)** (2024-09-12)

*Hanyu Jiang, Jian Xue, Xing Lan, Guohong Hu, Ke Lu*

  This paper introduces MVLLaVA, an intelligent agent designed for novel view
synthesis tasks. MVLLaVA integrates multiple multi-view diffusion models with a
large multimodal model, LLaVA, enabling it to handle a wide range of tasks
efficiently. MVLLaVA represents a versatile and unified platform that adapts to
diverse input types, including a single image, a descriptive caption, or a
specific change in viewing azimuth, guided by language instructions for
viewpoint generation. We carefully craft task-specific instruction templates,
which are subsequently used to fine-tune LLaVA. As a result, MVLLaVA acquires
the capability to generate novel view images based on user instructions,
demonstrating its flexibility across diverse tasks. Experiments are conducted
to validate the effectiveness of MVLLaVA, demonstrating its robust performance
and versatility in tackling diverse novel view synthesis challenges.


---

**[5. [2104.10450] Making Differentiable Architecture Search less local](https://arxiv.org/pdf/2104.10450.pdf)** (2021-04-22)

*Erik Bodin, Federico Tomasi, Zhenwen Dai*

  Neural architecture search (NAS) is a recent methodology for automating the
design of neural network architectures. Differentiable neural architecture
search (DARTS) is a promising NAS approach that dramatically increases search
efficiency. However, it has been shown to suffer from performance collapse,
where the search often leads to detrimental architectures. Many recent works
try to address this issue of DARTS by identifying indicators for early
stopping, regularising the search objective to reduce the dominance of some
operations, or changing the parameterisation of the search problem. In this
work, we hypothesise that performance collapses can arise from poor local
optima around typical initial architectures and weights. We address this issue
by developing a more global optimisation scheme that is able to better explore
the space without changing the DARTS problem formulation. Our experiments show
that our changes in the search algorithm allow the discovery of architectures
with both better test performance and fewer parameters.


---

**[6. [2311.10751] ProAgent: From Robotic Process Automation to Agentic Process Automation](https://arxiv.org/pdf/2311.10751.pdf)** (2023-11-27)

*Yining Ye, Xin Cong, Shizuo Tian, Jiannan Cao, Hao Wang, Yujia Qin, Yaxi Lu, Heyang Yu, Huadong Wang, Yankai Lin, Zhiyuan Liu, Maosong Sun*

  From ancient water wheels to robotic process automation (RPA), automation
technology has evolved throughout history to liberate human beings from arduous
tasks. Yet, RPA struggles with tasks needing human-like intelligence,
especially in elaborate design of workflow construction and dynamic
decision-making in workflow execution. As Large Language Models (LLMs) have
emerged human-like intelligence, this paper introduces Agentic Process
Automation (APA), a groundbreaking automation paradigm using LLM-based agents
for advanced automation by offloading the human labor to agents associated with
construction and execution. We then instantiate ProAgent, an LLM-based agent
designed to craft workflows from human instructions and make intricate
decisions by coordinating specialized agents. Empirical experiments are
conducted to detail its construction and execution procedure of workflow,
showcasing the feasibility of APA, unveiling the possibility of a new paradigm
of automation driven by agents. Our code is public at
https://github.com/OpenBMB/ProAgent.


---

**[7. [2502.09565] MDCrow: Automating Molecular Dynamics Workflows with Large Language
  Models](https://arxiv.org/pdf/2502.09565.pdf)** (2025-02-14)

*Quintina Campbell, Sam Cox, Jorge Medina, Brittany Watterson, Andrew D. White*

  Molecular dynamics (MD) simulations are essential for understanding
biomolecular systems but remain challenging to automate. Recent advances in
large language models (LLM) have demonstrated success in automating complex
scientific tasks using LLM-based agents. In this paper, we introduce MDCrow, an
agentic LLM assistant capable of automating MD workflows. MDCrow uses
chain-of-thought over 40 expert-designed tools for handling and processing
files, setting up simulations, analyzing the simulation outputs, and retrieving
relevant information from literature and databases. We assess MDCrow's
performance across 25 tasks of varying required subtasks and difficulty, and we
evaluate the agent's robustness to both difficulty and prompt style.
\texttt{gpt-4o} is able to complete complex tasks with low variance, followed
closely by \texttt{llama3-405b}, a compelling open-source model. While prompt
style does not influence the best models' performance, it has significant
effects on smaller models.


---

**[8. [2402.17574] Agent-Pro: Learning to Evolve via Policy-Level Reflection and
  Optimization](https://arxiv.org/pdf/2402.17574.pdf)** (2024-06-10)

*Wenqi Zhang, Ke Tang, Hai Wu, Mengna Wang, Yongliang Shen, Guiyang Hou, Zeqi Tan, Peng Li, Yueting Zhuang, Weiming Lu*

  Large Language Models (LLMs) exhibit robust problem-solving capabilities for
diverse tasks. However, most LLM-based agents are designed as specific task
solvers with sophisticated prompt engineering, rather than agents capable of
learning and evolving through interactions. These task solvers necessitate
manually crafted prompts to inform task rules and regulate LLM behaviors,
inherently incapacitating to address complex dynamic scenarios e.g., large
interactive games. In light of this, we propose Agent-Pro: an LLM-based Agent
with Policy-level Reflection and Optimization that can learn a wealth of
expertise from interactive experiences and progressively elevate its behavioral
policy. Specifically, it involves a dynamic belief generation and reflection
process for policy evolution. Rather than action-level reflection, Agent-Pro
iteratively reflects on past trajectories and beliefs, fine-tuning its
irrational beliefs for a better policy. Moreover, a depth-first search is
employed for policy optimization, ensuring continual enhancement in policy
payoffs. Agent-Pro is evaluated across two games: Blackjack and Texas Hold'em,
outperforming vanilla LLM and specialized models. Our results show Agent-Pro
can learn and evolve in complex and dynamic scenes, which also benefits
numerous LLM-based applications.


---

**[9. [2001.08861] Encoding Physical Constraints in Differentiable Newton-Euler Algorithm](https://arxiv.org/pdf/2001.08861.pdf)** (2020-10-12)

*Giovanni Sutanto, Austin S. Wang, Yixin Lin, Mustafa Mukadam, Gaurav S. Sukhatme, Akshara Rai, Franziska Meier*

  The recursive Newton-Euler Algorithm (RNEA) is a popular technique for
computing the dynamics of robots. RNEA can be framed as a differentiable
computational graph, enabling the dynamics parameters of the robot to be
learned from data via modern auto-differentiation toolboxes. However, the
dynamics parameters learned in this manner can be physically implausible. In
this work, we incorporate physical constraints in the learning by adding
structure to the learned parameters. This results in a framework that can learn
physically plausible dynamics via gradient descent, improving the training
speed as well as generalization of the learned dynamics models. We evaluate our
method on real-time inverse dynamics control tasks on a 7 degree of freedom
robot arm, both in simulation and on the real robot. Our experiments study a
spectrum of structure added to the parameters of the differentiable RNEA
algorithm, and compare their performance and generalization.


---

**[10. [2309.02609] Directionality-Aware Mixture Model Parallel Sampling for Efficient
  Linear Parameter Varying Dynamical System Learning](https://arxiv.org/pdf/2309.02609.pdf)** (2024-03-26)

*Sunan Sun, Haihui Gao, Tianyu Li, Nadia Figueroa*

  The Linear Parameter Varying Dynamical System (LPV-DS) is an effective
approach that learns stable, time-invariant motion policies using statistical
modeling and semi-definite optimization to encode complex motions for reactive
robot control. Despite its strengths, the LPV-DS learning approach faces
challenges in achieving a high model accuracy without compromising the
computational efficiency. To address this, we introduce the
Directionality-Aware Mixture Model (DAMM), a novel statistical model that
applies the Riemannian metric on the n-sphere $\mathbb{S}^n$ to efficiently
blend non-Euclidean directional data with $\mathbb{R}^m$ Euclidean states.
Additionally, we develop a hybrid Markov chain Monte Carlo technique that
combines Gibbs Sampling with Split/Merge Proposal, allowing for parallel
computation to drastically speed up inference. Our extensive empirical tests
demonstrate that LPV-DS integrated with DAMM achieves higher reproduction
accuracy, better model efficiency, and near real-time/online learning compared
to standard estimation methods on various datasets. Lastly, we demonstrate its
suitability for incrementally learning multi-behavior policies in real-world
robot experiments.


---

**[11. [2302.10319] Differentiable Bootstrap Particle Filters for Regime-Switching Models](https://arxiv.org/pdf/2302.10319.pdf)** (2023-05-04)

*Wenhan Li, Xiongjie Chen, Wenwu Wang, Víctor Elvira, Yunpeng Li*

  Differentiable particle filters are an emerging class of particle filtering
methods that use neural networks to construct and learn parametric state-space
models. In real-world applications, both the state dynamics and measurements
can switch between a set of candidate models. For instance, in target tracking,
vehicles can idle, move through traffic, or cruise on motorways, and
measurements are collected in different geographical or weather conditions.
This paper proposes a new differentiable particle filter for regime-switching
state-space models. The method can learn a set of unknown candidate dynamic and
measurement models and track the state posteriors. We evaluate the performance
of the novel algorithm in relevant models, showing its great performance
compared to other competitive algorithms.


---

**[12. [2109.08522] Dynamics-Aware Quality-Diversity for Efficient Learning of Skill
  Repertoires](https://arxiv.org/pdf/2109.08522.pdf)** (2022-07-21)

*Bryan Lim, Luca Grillotti, Lorenzo Bernasconi, Antoine Cully*

  Quality-Diversity (QD) algorithms are powerful exploration algorithms that
allow robots to discover large repertoires of diverse and high-performing
skills. However, QD algorithms are sample inefficient and require millions of
evaluations. In this paper, we propose Dynamics-Aware Quality-Diversity
(DA-QD), a framework to improve the sample efficiency of QD algorithms through
the use of dynamics models. We also show how DA-QD can then be used for
continual acquisition of new skill repertoires. To do so, we incrementally
train a deep dynamics model from experience obtained when performing skill
discovery using QD. We can then perform QD exploration in imagination with an
imagined skill repertoire. We evaluate our approach on three robotic
experiments. First, our experiments show DA-QD is 20 times more sample
efficient than existing QD approaches for skill discovery. Second, we
demonstrate learning an entirely new skill repertoire in imagination to perform
zero-shot learning. Finally, we show how DA-QD is useful and effective for
solving a long horizon navigation task and for damage adaptation in the real
world. Videos and source code are available at:
https://sites.google.com/view/da-qd.


---

**[13. [2503.11301] GNNs as Predictors of Agentic Workflow Performances](https://arxiv.org/pdf/2503.11301.pdf)** (2025-03-17)

*Yuanshuo Zhang, Yuchen Hou, Bohan Tang, Shuo Chen, Muhan Zhang, Xiaowen Dong, Siheng Chen*

  Agentic workflows invoked by Large Language Models (LLMs) have achieved
remarkable success in handling complex tasks. However, optimizing such
workflows is costly and inefficient in real-world applications due to extensive
invocations of LLMs. To fill this gap, this position paper formulates agentic
workflows as computational graphs and advocates Graph Neural Networks (GNNs) as
efficient predictors of agentic workflow performances, avoiding repeated LLM
invocations for evaluation. To empirically ground this position, we construct
FLORA-Bench, a unified platform for benchmarking GNNs for predicting agentic
workflow performances. With extensive experiments, we arrive at the following
conclusion: GNNs are simple yet effective predictors. This conclusion supports
new applications of GNNs and a novel direction towards automating agentic
workflow optimization. All codes, models, and data are available at
https://github.com/youngsoul0731/Flora-Bench.


---

**[14. [2305.13795] Proximal Policy Gradient Arborescence for Quality Diversity
  Reinforcement Learning](https://arxiv.org/pdf/2305.13795.pdf)** (2024-01-31)

*Sumeet Batra, Bryon Tjanaka, Matthew C. Fontaine, Aleksei Petrenko, Stefanos Nikolaidis, Gaurav Sukhatme*

  Training generally capable agents that thoroughly explore their environment
and learn new and diverse skills is a long-term goal of robot learning. Quality
Diversity Reinforcement Learning (QD-RL) is an emerging research area that
blends the best aspects of both fields -- Quality Diversity (QD) provides a
principled form of exploration and produces collections of behaviorally diverse
agents, while Reinforcement Learning (RL) provides a powerful performance
improvement operator enabling generalization across tasks and dynamic
environments. Existing QD-RL approaches have been constrained to sample
efficient, deterministic off-policy RL algorithms and/or evolution strategies,
and struggle with highly stochastic environments. In this work, we, for the
first time, adapt on-policy RL, specifically Proximal Policy Optimization
(PPO), to the Differentiable Quality Diversity (DQD) framework and propose
additional improvements over prior work that enable efficient optimization and
discovery of novel skills on challenging locomotion tasks. Our new algorithm,
Proximal Policy Gradient Arborescence (PPGA), achieves state-of-the-art
results, including a 4x improvement in best reward over baselines on the
challenging humanoid domain.


---

**[15. [2410.18528] PRACT: Optimizing Principled Reasoning and Acting of LLM Agent](https://arxiv.org/pdf/2410.18528.pdf)** (2024-10-25)

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

**[16. [2503.15520] Agent-S: LLM Agentic workflow to automate Standard Operating Procedures](https://arxiv.org/pdf/2503.15520.pdf)** (2025-03-21)

*Mandar Kulkarni*

  AI agents using Large Language Models (LLMs) as foundations have shown
promise in solving complex real-world tasks. In this paper, we propose an
LLM-based agentic workflow for automating Standard Operating Procedures (SOP).
For customer care operations, an SOP defines a logical step-by-step process for
human agents to resolve customer issues. We observe that any step in the SOP
can be categorized as user interaction or API call, while the logical flow in
the SOP defines the navigation. We use LLMs augmented with memory and
environments (API tools, user interface, external knowledge source) for SOP
automation. Our agentic architecture consists of three task-specific LLMs, a
Global Action Repository (GAR), execution memory, and multiple environments.
SOP workflow is written as a simple logical block of text. Based on the current
execution memory and the SOP, the agent chooses the action to execute; it
interacts with an appropriate environment (user/API) to collect observations
and feedback, which are, in turn, inputted to memory to decide the next action.
The agent is designed to be fault-tolerant, where it dynamically decides to
repeat an action or seek input from an external knowledge source. We
demonstrate the efficacy of the proposed agent on the three SOPs from the
e-commerce seller domain. The experimental results validate the agent's
performance under complex real-world scenarios.


---

**[17. [2305.15340] Bayesian calibration of differentiable agent-based models](https://arxiv.org/pdf/2305.15340.pdf)** (2023-05-25)

*Arnau Quera-Bofarull, Ayush Chopra, Anisoara Calinescu, Michael Wooldridge, Joel Dyer*

  Agent-based modelling (ABMing) is a powerful and intuitive approach to
modelling complex systems; however, the intractability of ABMs' likelihood
functions and the non-differentiability of the mathematical operations
comprising these models present a challenge to their use in the real world.
These difficulties have in turn generated research on approximate Bayesian
inference methods for ABMs and on constructing differentiable approximations to
arbitrary ABMs, but little work has been directed towards designing approximate
Bayesian inference techniques for the specific case of differentiable ABMs. In
this work, we aim to address this gap and discuss how generalised variational
inference procedures may be employed to provide misspecification-robust
Bayesian parameter inferences for differentiable ABMs. We demonstrate with
experiments on a differentiable ABM of the COVID-19 pandemic that our approach
can result in accurate inferences, and discuss avenues for future work.


---

**[18. [2303.11103] Sionna RT: Differentiable Ray Tracing for Radio Propagation Modeling](https://arxiv.org/pdf/2303.11103.pdf)** (2023-07-20)

*Jakob Hoydis, Fayçal Aït Aoudia, Sebastian Cammerer, Merlin Nimier-David, Nikolaus Binder, Guillermo Marcus, Alexander Keller*

  Sionna is a GPU-accelerated open-source library for link-level simulations
based on TensorFlow. Since release v0.14 it integrates a differentiable ray
tracer (RT) for the simulation of radio wave propagation. This unique feature
allows for the computation of gradients of the channel impulse response and
other related quantities with respect to many system and environment
parameters, such as material properties, antenna patterns, array geometries, as
well as transmitter and receiver orientations and positions. In this paper, we
outline the key components of Sionna RT and showcase example applications such
as learning radio materials and optimizing transmitter orientations by gradient
descent. While classic ray tracing is a crucial tool for 6G research topics
like reconfigurable intelligent surfaces, integrated sensing and
communications, as well as user localization, differentiable ray tracing is a
key enabler for many novel and exciting research directions, for example,
digital twins.


---

**[19. [2110.00304] Divergence-Regularized Multi-Agent Actor-Critic](https://arxiv.org/pdf/2110.00304.pdf)** (2022-06-22)

*Kefan Su, Zongqing Lu*

  Entropy regularization is a popular method in reinforcement learning (RL).
Although it has many advantages, it alters the RL objective of the original
Markov Decision Process (MDP). Though divergence regularization has been
proposed to settle this problem, it cannot be trivially applied to cooperative
multi-agent reinforcement learning (MARL). In this paper, we investigate
divergence regularization in cooperative MARL and propose a novel off-policy
cooperative MARL framework, divergence-regularized multi-agent actor-critic
(DMAC). Theoretically, we derive the update rule of DMAC which is naturally
off-policy and guarantees monotonic policy improvement and convergence in both
the original MDP and divergence-regularized MDP. We also give a bound of the
discrepancy between the converged policy and optimal policy in the original
MDP. DMAC is a flexible framework and can be combined with many existing MARL
algorithms. Empirically, we evaluate DMAC in a didactic stochastic game and
StarCraft Multi-Agent Challenge and show that DMAC substantially improves the
performance of existing MARL algorithms.


---

**[20. [2006.13561] Differentiable Window for Dynamic Local Attention](https://arxiv.org/pdf/2006.13561.pdf)** (2020-06-25)

*Thanh-Tung Nguyen, Xuan-Phi Nguyen, Shafiq Joty, Xiaoli Li*

  We propose Differentiable Window, a new neural module and general purpose
component for dynamic window selection. While universally applicable, we
demonstrate a compelling use case of utilizing Differentiable Window to improve
standard attention modules by enabling more focused attentions over the input
regions. We propose two variants of Differentiable Window, and integrate them
within the Transformer architecture in two novel ways. We evaluate our proposed
approach on a myriad of NLP tasks, including machine translation, sentiment
analysis, subject-verb agreement and language modeling. Our experimental
results demonstrate consistent and sizable improvements across all tasks.


---

**[21. [2302.05629] Improving Differentiable Architecture Search via Self-Distillation](https://arxiv.org/pdf/2302.05629.pdf)** (2023-09-04)

*Xunyu Zhu, Jian Li, Yong Liu, Weiping Wang*

  Differentiable Architecture Search (DARTS) is a simple yet efficient Neural
Architecture Search (NAS) method. During the search stage, DARTS trains a
supernet by jointly optimizing architecture parameters and network parameters.
During the evaluation stage, DARTS discretizes the supernet to derive the
optimal architecture based on architecture parameters. However, recent research
has shown that during the training process, the supernet tends to converge
towards sharp minima rather than flat minima. This is evidenced by the higher
sharpness of the loss landscape of the supernet, which ultimately leads to a
performance gap between the supernet and the optimal architecture. In this
paper, we propose Self-Distillation Differentiable Neural Architecture Search
(SD-DARTS) to alleviate the discretization gap. We utilize self-distillation to
distill knowledge from previous steps of the supernet to guide its training in
the current step, effectively reducing the sharpness of the supernet's loss and
bridging the performance gap between the supernet and the optimal architecture.
Furthermore, we introduce the concept of voting teachers, where multiple
previous supernets are selected as teachers, and their output probabilities are
aggregated through voting to obtain the final teacher prediction. Experimental
results on real datasets demonstrate the advantages of our novel
self-distillation-based NAS method compared to state-of-the-art alternatives.


---

**[22. [2101.08095] Automatic Differentiation via Effects and Handlers: An Implementation in
  Frank](https://arxiv.org/pdf/2101.08095.pdf)** (2021-01-21)

*Jesse Sigal*

  Automatic differentiation (AD) is an important family of algorithms which
enables derivative based optimization. We show that AD can be simply
implemented with effects and handlers by doing so in the Frank language. By
considering how our implementation behaves in Frank's operational semantics, we
show how our code performs the dynamic creation of programs during evaluation.


---

**[23. [2001.05214] Proceedings of the AAAI-20 Workshop on Intelligent Process Automation
  (IPA-20)](https://arxiv.org/pdf/2001.05214.pdf)** (2021-04-20)

*Dell Zhang, Andre Freitas, Dacheng Tao, Dawn Song*

  This is the Proceedings of the AAAI-20 Workshop on Intelligent Process
Automation (IPA-20) which took place in New York, NY, USA on February 7th 2020.


---

**[24. [2402.05421] DiffTORI: Differentiable Trajectory Optimization for Deep Reinforcement
  and Imitation Learning](https://arxiv.org/pdf/2402.05421.pdf)** (2025-03-05)

*Weikang Wan, Ziyu Wang, Yufei Wang, Zackory Erickson, David Held*

  This paper introduces DiffTORI, which utilizes Differentiable Trajectory
Optimization as the policy representation to generate actions for deep
Reinforcement and Imitation learning. Trajectory optimization is a powerful and
widely used algorithm in control, parameterized by a cost and a dynamics
function. The key to our approach is to leverage the recent progress in
differentiable trajectory optimization, which enables computing the gradients
of the loss with respect to the parameters of trajectory optimization. As a
result, the cost and dynamics functions of trajectory optimization can be
learned end-to-end. DiffTORI addresses the ``objective mismatch'' issue of
prior model-based RL algorithms, as the dynamics model in DiffTORI is learned
to directly maximize task performance by differentiating the policy gradient
loss through the trajectory optimization process. We further benchmark DiffTORI
for imitation learning on standard robotic manipulation task suites with
high-dimensional sensory observations and compare our method to feed-forward
policy classes as well as Energy-Based Models (EBM) and Diffusion. Across 15
model-based RL tasks and 35 imitation learning tasks with high-dimensional
image and point cloud inputs, DiffTORI outperforms prior state-of-the-art
methods in both domains. Our code is available at
https://github.com/wkwan7/DiffTORI.


---

**[25. [2107.06862] Differentiable Programming of Reaction-Diffusion Patterns](https://arxiv.org/pdf/2107.06862.pdf)** (2021-07-15)

*Alexander Mordvintsev, Ettore Randazzo, Eyvind Niklasson*

  Reaction-Diffusion (RD) systems provide a computational framework that
governs many pattern formation processes in nature. Current RD system design
practices boil down to trial-and-error parameter search. We propose a
differentiable optimization method for learning the RD system parameters to
perform example-based texture synthesis on a 2D plane. We do this by
representing the RD system as a variant of Neural Cellular Automata and using
task-specific differentiable loss functions. RD systems generated by our method
exhibit robust, non-trivial 'life-like' behavior.


---

**[26. [2407.13513] Instance Selection for Dynamic Algorithm Configuration with
  Reinforcement Learning: Improving Generalization](https://arxiv.org/pdf/2407.13513.pdf)** (2024-07-19)

*Carolin Benjamins, Gjorgjina Cenikj, Ana Nikolikj, Aditya Mohan, Tome Eftimov, Marius Lindauer*

  Dynamic Algorithm Configuration (DAC) addresses the challenge of dynamically
setting hyperparameters of an algorithm for a diverse set of instances rather
than focusing solely on individual tasks. Agents trained with Deep
Reinforcement Learning (RL) offer a pathway to solve such settings. However,
the limited generalization performance of these agents has significantly
hindered the application in DAC. Our hypothesis is that a potential bias in the
training instances limits generalization capabilities. We take a step towards
mitigating this by selecting a representative subset of training instances to
overcome overrepresentation and then retraining the agent on this subset to
improve its generalization performance. For constructing the meta-features for
the subset selection, we particularly account for the dynamic nature of the RL
agent by computing time series features on trajectories of actions and rewards
generated by the agent's interaction with the environment. Through empirical
evaluations on the Sigmoid and CMA-ES benchmarks from the standard benchmark
library for DAC, called DACBench, we discuss the potentials of our selection
technique compared to training on the entire instance set. Our results
highlight the efficacy of instance selection in refining DAC policies for
diverse instance spaces.


---

**[27. [2412.03572] Navigation World Models](https://arxiv.org/pdf/2412.03572.pdf)** (2025-04-15)

*Amir Bar, Gaoyue Zhou, Danny Tran, Trevor Darrell, Yann LeCun*

  Navigation is a fundamental skill of agents with visual-motor capabilities.
We introduce a Navigation World Model (NWM), a controllable video generation
model that predicts future visual observations based on past observations and
navigation actions. To capture complex environment dynamics, NWM employs a
Conditional Diffusion Transformer (CDiT), trained on a diverse collection of
egocentric videos of both human and robotic agents, and scaled up to 1 billion
parameters. In familiar environments, NWM can plan navigation trajectories by
simulating them and evaluating whether they achieve the desired goal. Unlike
supervised navigation policies with fixed behavior, NWM can dynamically
incorporate constraints during planning. Experiments demonstrate its
effectiveness in planning trajectories from scratch or by ranking trajectories
sampled from an external policy. Furthermore, NWM leverages its learned visual
priors to imagine trajectories in unfamiliar environments from a single input
image, making it a flexible and powerful tool for next-generation navigation
systems.


---

**[28. [2304.01447] Off-Policy Action Anticipation in Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/2304.01447.pdf)** (2023-04-05)

*Ariyan Bighashdel, Daan de Geus, Pavol Jancura, Gijs Dubbelman*

  Learning anticipation in Multi-Agent Reinforcement Learning (MARL) is a
reasoning paradigm where agents anticipate the learning steps of other agents
to improve cooperation among themselves. As MARL uses gradient-based
optimization, learning anticipation requires using Higher-Order Gradients
(HOG), with so-called HOG methods. Existing HOG methods are based on policy
parameter anticipation, i.e., agents anticipate the changes in policy
parameters of other agents. Currently, however, these existing HOG methods have
only been applied to differentiable games or games with small state spaces. In
this work, we demonstrate that in the case of non-differentiable games with
large state spaces, existing HOG methods do not perform well and are
inefficient due to their inherent limitations related to policy parameter
anticipation and multiple sampling stages. To overcome these problems, we
propose Off-Policy Action Anticipation (OffPA2), a novel framework that
approaches learning anticipation through action anticipation, i.e., agents
anticipate the changes in actions of other agents, via off-policy sampling. We
theoretically analyze our proposed OffPA2 and employ it to develop multiple HOG
methods that are applicable to non-differentiable games with large state
spaces. We conduct a large set of experiments and illustrate that our proposed
HOG methods outperform the existing ones regarding efficiency and performance.


---

**[29. [2311.02916] Virtual Action Actor-Critic Framework for Exploration (Student Abstract)](https://arxiv.org/pdf/2311.02916.pdf)** (2023-11-07)

*Bumgeun Park, Taeyoung Kim, Quoc-Vinh Lai-Dang, Dongsoo Har*

  Efficient exploration for an agent is challenging in reinforcement learning
(RL). In this paper, a novel actor-critic framework namely virtual action
actor-critic (VAAC), is proposed to address the challenge of efficient
exploration in RL. This work is inspired by humans' ability to imagine the
potential outcomes of their actions without actually taking them. In order to
emulate this ability, VAAC introduces a new actor called virtual actor (VA),
alongside the conventional actor-critic framework. Unlike the conventional
actor, the VA takes the virtual action to anticipate the next state without
interacting with the environment. With the virtual policy following a Gaussian
distribution, the VA is trained to maximize the anticipated novelty of the
subsequent state resulting from a virtual action. If any next state resulting
from available actions does not exhibit high anticipated novelty, training the
VA leads to an increase in the virtual policy entropy. Hence, high virtual
policy entropy represents that there is no room for exploration. The proposed
VAAC aims to maximize a modified Q function, which combines cumulative rewards
and the negative sum of virtual policy entropy. Experimental results show that
the VAAC improves the exploration performance compared to existing algorithms.


---

**[30. [2409.14432] EM-DARTS: Hierarchical Differentiable Architecture Search for Eye
  Movement Recognition](https://arxiv.org/pdf/2409.14432.pdf)** (2025-01-14)

*Huafeng Qin, Hongyu Zhu, Xin Jin, Xin Yu, Mounim A. El-Yacoubi, Shuqiang Yang*

  Eye movement biometrics has received increasing attention thanks to its
highly secure identification. Although deep learning (DL) models have shown
success in eye movement recognition, their architectures largely rely on human
prior knowledge. Differentiable Neural Architecture Search (DARTS) automates
the manual process of architecture design with high search efficiency. However,
DARTS typically stacks multiple cells to form a convolutional network, which
limits the diversity of architecture. Furthermore, DARTS generally searches for
architectures using shallower networks than those used in the evaluation,
creating a significant disparity in architecture depth between the search and
evaluation phases. To address this issue, we propose EM-DARTS, a hierarchical
differentiable architecture search algorithm to automatically design the DL
architecture for eye movement recognition. First, we define a supernet and
propose a global and local alternate Neural Architecture Search method to
search the optimal architecture alternately with a differentiable neural
architecture search. The local search strategy aims to find an optimal
architecture for different cells while the global search strategy is
responsible for optimizing the architecture of the target network. To minimize
redundancy, transfer entropy is proposed to compute the information amount of
each layer, thereby further simplifying the network search process.
Experimental results on three public datasets demonstrate that the proposed
EM-DARTS is capable of producing an optimal architecture that leads to
state-of-the-art recognition performance, {Specifically, the recognition models
developed using EM-DARTS achieved the lowest EERs of 0.0453 on the GazeBase
dataset, 0.0377 on the JuDo1000 dataset, and 0.1385 on the EMglasses dataset.


---

**[31. [2311.15929] The Common Workflow Scheduler Interface: Status Quo and Future Plans](https://arxiv.org/pdf/2311.15929.pdf)** (2023-11-28)

*Fabian Lehmann, Jonathan Bader, Lauritz Thamsen, Ulf Leser*

  Nowadays, many scientific workflows from different domains, such as Remote
Sensing, Astronomy, and Bioinformatics, are executed on large computing
infrastructures managed by resource managers. Scientific workflow management
systems (SWMS) support the workflow execution and communicate with the
infrastructures' resource managers. However, the communication between SWMS and
resource managers is complicated by a) inconsistent interfaces between SMWS and
resource managers and b) the lack of support for workflow dependencies and
workflow-specific properties.
  To tackle these issues, we developed the Common Workflow Scheduler Interface
(CWSI), a simple yet powerful interface to exchange workflow-related
information between a SWMS and a resource manager, making the resource manager
workflow-aware. The first prototype implementations show that the CWSI can
reduce the makespan already with simple but workflow-aware strategies up to
25%. In this paper, we show how existing workflow resource management research
can be integrated into the CWSI.


---

**[32. [2411.08054] GREI Data Repository AI Taxonomy](https://arxiv.org/pdf/2411.08054.pdf)** (2024-11-14)

*California Digital Library  John Chodacki, figshare  Mark Hanhel, Dataverse  Stefano Iacus, Dryad  Ryan Scherle, Center for Open
  Science  Eric Olson, Center for Open Science  Nici Pfeiffer, Zenodo  Kristi Holmes, Zenodo  Mohammad Hosseini*

  The Generalist Repository Ecosystem Initiative (GREI), funded by the NIH,
developed an AI taxonomy tailored to data repository roles to guide AI
integration across repository management. It categorizes the roles into stages,
including acquisition, validation, organization, enhancement, analysis,
sharing, and user support, providing a structured framework for implementing AI
in repository workflows.


---

**[33. [2012.12264] Digital Annealer for quadratic unconstrained binary optimization: a
  comparative performance analysis](https://arxiv.org/pdf/2012.12264.pdf)** (2020-12-24)

*Oylum Şeker, Neda Tanoumand, Merve Bodur*

  Digital Annealer (DA) is a computer architecture designed for tackling
combinatorial optimization problems formulated as quadratic unconstrained
binary optimization (QUBO) models. In this paper, we present the results of an
extensive computational study to evaluate the performance of DA in a systematic
way in comparison to multiple state-of-the-art solvers for different problem
classes. We examine pure QUBO models, as well as QUBO reformulations of three
constrained problems, namely quadratic assignment, quadratic cycle partition,
and selective graph coloring, with the last two being new applications for DA.
For the selective graph coloring problem, we also present a size reduction
heuristic that significantly increases the number of eligible instances for DA.
Our experimental results show that despite being in its development stage, DA
can provide high-quality solutions quickly and in that regard rivals the state
of the art, particularly for large instances. Moreover, as opposed to
established solvers, within its limit on the number of decision variables, DA's
solution times are not affected by the increase in instance size. These
findings illustrate that DA has the potential to become a successful technology
in tackling combinatorial optimization problems.


---

**[34. [2211.02987] Differentiable Neural Computers with Memory Demon](https://arxiv.org/pdf/2211.02987.pdf)** (2022-11-08)

*Ari Azarafrooz*

  A Differentiable Neural Computer (DNC) is a neural network with an external
memory which allows for iterative content modification via read, write and
delete operations.
  We show that information theoretic properties of the memory contents play an
important role in the performance of such architectures. We introduce a novel
concept of memory demon to DNC architectures which modifies the memory contents
implicitly via additive input encoding. The goal of the memory demon is to
maximize the expected sum of mutual information of the consecutive external
memory contents.


---

**[35. [2108.03894] FIFA: Fast Inference Approximation for Action Segmentation](https://arxiv.org/pdf/2108.03894.pdf)** (2021-08-10)

*Yaser Souri, Yazan Abu Farha, Fabien Despinoy, Gianpiero Francesca, Juergen Gall*

  We introduce FIFA, a fast approximate inference method for action
segmentation and alignment. Unlike previous approaches, FIFA does not rely on
expensive dynamic programming for inference. Instead, it uses an approximate
differentiable energy function that can be minimized using gradient-descent.
FIFA is a general approach that can replace exact inference improving its speed
by more than 5 times while maintaining its performance. FIFA is an anytime
inference algorithm that provides a better speed vs. accuracy trade-off
compared to exact inference. We apply FIFA on top of state-of-the-art
approaches for weakly supervised action segmentation and alignment as well as
fully supervised action segmentation. FIFA achieves state-of-the-art results on
most metrics on two action segmentation datasets.


---

**[36. [2202.10449] Optimal Multi-Agent Path Finding for Precedence Constrained Planning
  Tasks](https://arxiv.org/pdf/2202.10449.pdf)** (2022-02-23)

*Kushal Kedia, Rajat Kumar Jenamani, Aritra Hazra, Partha Pratim Chakrabarti*

  Multi-Agent Path Finding (MAPF) is the problem of finding collision-free
paths for multiple agents from their start locations to end locations. We
consider an extension to this problem, Precedence Constrained Multi-Agent Path
Finding (PC-MAPF), wherein agents are assigned a sequence of planning tasks
that contain precedence constraints between them. PC-MAPF has various
applications, for example in multi-agent pickup and delivery problems where
some objects might require multiple agents to collaboratively pickup and move
them in unison. Precedence constraints also arise in warehouse assembly
problems where before a manufacturing task can begin, its input resources must
be manufactured and delivered. We propose a novel algorithm, Precedence
Constrained Conflict Based Search (PC-CBS), which finds makespan-optimal
solutions for this class of problems. PC-CBS utilizes a Precedence-Constrained
Task-Graph to define valid intervals for each planning task and updates them
when precedence conflicts are encountered. We benchmark the performance of this
algorithm over various warehouse assembly, and multi-agent pickup and delivery
tasks, and use it to evaluate the sub-optimality of a recently proposed
efficient baseline.


---

**[37. [2002.05283] Stabilizing Differentiable Architecture Search via Perturbation-based
  Regularization](https://arxiv.org/pdf/2002.05283.pdf)** (2021-01-19)

*Xiangning Chen, Cho-Jui Hsieh*

  Differentiable architecture search (DARTS) is a prevailing NAS solution to
identify architectures. Based on the continuous relaxation of the architecture
space, DARTS learns a differentiable architecture weight and largely reduces
the search cost. However, its stability has been challenged for yielding
deteriorating architectures as the search proceeds. We find that the
precipitous validation loss landscape, which leads to a dramatic performance
drop when distilling the final architecture, is an essential factor that causes
instability. Based on this observation, we propose a perturbation-based
regularization - SmoothDARTS (SDARTS), to smooth the loss landscape and improve
the generalizability of DARTS-based methods. In particular, our new
formulations stabilize DARTS-based methods by either random smoothing or
adversarial attack. The search trajectory on NAS-Bench-1Shot1 demonstrates the
effectiveness of our approach and due to the improved stability, we achieve
performance gain across various search spaces on 4 datasets. Furthermore, we
mathematically show that SDARTS implicitly regularizes the Hessian norm of the
validation loss, which accounts for a smoother loss landscape and improved
performance.


---

**[38. [2409.02366] The Hidden Costs of Automation: An Empirical Study on GitHub Actions
  Workflow Maintenance](https://arxiv.org/pdf/2409.02366.pdf)** (2024-09-05)

*Pablo Valenzuela-Toledo, Alexandre Bergel, Timo Kehrer, Oscar Nierstrasz*

  GitHub Actions (GA) is an orchestration platform that streamlines the
automatic execution of software engineering tasks such as building, testing,
and deployment. Although GA workflows are the primary means for automation,
according to our experience and observations, human intervention is necessary
to correct defects, update dependencies, or refactor existing workflow files.
In fact, previous research has shown that software artifacts similar to
workflows, such as build files and bots, can introduce additional maintenance
tasks in software projects. This suggests that workflow files, which are also
used to automate repetitive tasks in professional software production, may
generate extra workload for developers. However, the nature of such effort has
not been well studied. This paper presents a large-scale empirical
investigation towards characterizing the maintenance of GA workflows by
studying the evolution of workflow files in almost 200 mature GitHub projects
across ten programming languages. Our findings largely confirm the results of
previous studies on the maintenance of similar artifacts, while also revealing
GA-specific insights such as bug fixing and CI/CD improvement being among the
major drivers of GA maintenance. A direct implication is that practitioners
should be aware of proper resource planning and allocation for maintaining GA
workflows, thus exposing the ``hidden costs of automation.'' Our findings also
call for identifying and documenting best practices for such maintenance, and
for enhanced tool features supporting dependency tracking and better error
reporting of workflow specifications.


---

**[39. [2405.04865] Regime Learning for Differentiable Particle Filters](https://arxiv.org/pdf/2405.04865.pdf)** (2024-12-19)

*John-Joseph Brady, Yuhui Luo, Wenwu Wang, Victor Elvira, Yunpeng Li*

  Differentiable particle filters are an emerging class of models that combine
sequential Monte Carlo techniques with the flexibility of neural networks to
perform state space inference. This paper concerns the case where the system
may switch between a finite set of state-space models, i.e. regimes. No prior
approaches effectively learn both the individual regimes and the switching
process simultaneously. In this paper, we propose the neural network based
regime learning differentiable particle filter (RLPF) to address this problem.
We further design a training procedure for the RLPF and other related
algorithms. We demonstrate competitive performance compared to the previous
state-of-the-art algorithms on a pair of numerical experiments.


---

**[40. [2112.06780] Explanation Container in Case-Based Biomedical Question-Answering](https://arxiv.org/pdf/2112.06780.pdf)** (2021-12-23)

*Prateek Goel, Adam J. Johs, Manil Shrestha, Rosina O. Weber*

  The National Center for Advancing Translational Sciences(NCATS) Biomedical
Data Translator (Translator) aims to attenuate problems faced by translational
scientists. Translator is a multi-agent architecture consisting of six
autonomous relay agents (ARAs) and eight knowledge providers (KPs). In this
paper, we present the design of the Explanatory Agent (xARA), a case-based ARA
that answers biomedical queries by accessing multiple KPs, ranking results, and
explaining the ranking of results. The Explanatory Agent is designed with five
knowledge containers that include the four original knowledge containers and
one additional container for explanation - the Explanation Container. The
Explanation Container is case-based and designed with its own knowledge
containers.


---

**[41. [2202.01284] Dr.Jit: A Just-In-Time Compiler for Differentiable Rendering](https://arxiv.org/pdf/2202.01284.pdf)** (2022-05-02)

*Wenzel Jakob, Sébastien Speierer, Nicolas Roussel, Delio Vicini*

  Dr.Jit is a new just-in-time compiler for physically based rendering and its
derivative. Dr.Jit expedites research on these topics in two ways: first, it
traces high-level simulation code (e.g., written in Python) and aggressively
simplifies and specializes the resulting program representation, producing
data-parallel kernels with state-of-the-art performance on CPUs and GPUs.
  Second, it simplifies the development of differentiable rendering algorithms.
Efficient methods in this area turn the derivative of a simulation into a
simulation of the derivative. Dr.Jit provides fine-grained control over the
process of automatic differentiation to help with this transformation.
  Specialization is particularly helpful in the context of differentiation,
since large parts of the simulation ultimately do not influence the computed
gradients. Dr.Jit tracks data dependencies globally to find and remove
redundant computation.


---

**[42. [2412.13437] Deploying Foundation Model Powered Agent Services: A Survey](https://arxiv.org/pdf/2412.13437.pdf)** (2024-12-19)

*Wenchao Xu, Jinyu Chen, Peirong Zheng, Xiaoquan Yi, Tianyi Tian, Wenhui Zhu, Quan Wan, Haozhao Wang, Yunfeng Fan, Qinliang Su, Xuemin Shen*

  Foundation model (FM) powered agent services are regarded as a promising
solution to develop intelligent and personalized applications for advancing
toward Artificial General Intelligence (AGI). To achieve high reliability and
scalability in deploying these agent services, it is essential to
collaboratively optimize computational and communication resources, thereby
ensuring effective resource allocation and seamless service delivery. In
pursuit of this vision, this paper proposes a unified framework aimed at
providing a comprehensive survey on deploying FM-based agent services across
heterogeneous devices, with the emphasis on the integration of model and
resource optimization to establish a robust infrastructure for these services.
Particularly, this paper begins with exploring various low-level optimization
strategies during inference and studies approaches that enhance system
scalability, such as parallelism techniques and resource scaling methods. The
paper then discusses several prominent FMs and investigates research efforts
focused on inference acceleration, including techniques such as model
compression and token reduction. Moreover, the paper also investigates critical
components for constructing agent services and highlights notable intelligent
applications. Finally, the paper presents potential research directions for
developing real-time agent services with high Quality of Service (QoS).


---

**[43. [2305.07878] Automatic Differentiation in Prolog](https://arxiv.org/pdf/2305.07878.pdf)** (2023-05-16)

*Tom Schrijvers, Birthe van den Berg, Fabrizio Riguzzi*

  Automatic differentiation (AD) is a range of algorithms to compute the
numeric value of a function's (partial) derivative, where the function is
typically given as a computer program or abstract syntax tree. AD has become
immensely popular as part of many learning algorithms, notably for neural
networks. This paper uses Prolog to systematically derive gradient-based
forward- and reverse-mode AD variants from a simple executable specification:
evaluation of the symbolic derivative. Along the way we demonstrate that
several Prolog features (DCGs, co-routines) contribute to the succinct
formulation of the algorithm. We also discuss two applications in probabilistic
programming that are enabled by our Prolog algorithms. The first is parameter
learning for the Sum-Product Loop Language and the second consists of both
parameter learning and variational inference for probabilistic logic
programming.


---

**[44. [2112.02852] Target Entropy Annealing for Discrete Soft Actor-Critic](https://arxiv.org/pdf/2112.02852.pdf)** (2021-12-07)

*Yaosheng Xu, Dailin Hu, Litian Liang, Stephen McAleer, Pieter Abbeel, Roy Fox*

  Soft Actor-Critic (SAC) is considered the state-of-the-art algorithm in
continuous action space settings. It uses the maximum entropy framework for
efficiency and stability, and applies a heuristic temperature Lagrange term to
tune the temperature $\alpha$, which determines how "soft" the policy should
be. It is counter-intuitive that empirical evidence shows SAC does not perform
well in discrete domains. In this paper we investigate the possible
explanations for this phenomenon and propose Target Entropy Scheduled SAC
(TES-SAC), an annealing method for the target entropy parameter applied on SAC.
Target entropy is a constant in the temperature Lagrange term and represents
the target policy entropy in discrete SAC. We compare our method on Atari 2600
games with different constant target entropy SAC, and analyze on how our
scheduling affects SAC.


---

**[45. [2406.05804] A Review of Prominent Paradigms for LLM-Based Agents: Tool Use
  (Including RAG), Planning, and Feedback Learning](https://arxiv.org/pdf/2406.05804.pdf)** (2024-12-03)

*Xinzhe Li*

  Tool use, planning, and feedback learning are currently three prominent
paradigms for developing Large Language Model (LLM)-based agents across various
tasks. Although numerous frameworks have been devised for each paradigm, their
intricate workflows and inconsistent taxonomy create challenges in
understanding and reviewing the frameworks across different paradigms. This
survey introduces a unified taxonomy to systematically review and discuss these
frameworks. Specifically, 1) the taxonomy defines environments/tasks, common
LLM-profiled roles or LMPRs (policy models, evaluators, and dynamic models),
and universally applicable workflows found in prior work, and 2) it enables a
comparison of key perspectives on the implementations of LMPRs and workflow
designs across different agent paradigms and frameworks. 3) Finally, we
identify three limitations in existing workflow designs and systematically
discuss the future work. Resources have been made publicly available at in our
GitHub repository https://github.com/xinzhel/LLM-Agent-Survey.


---

**[46. [2411.01643] EcoAct: Economic Agent Determines When to Register What Action](https://arxiv.org/pdf/2411.01643.pdf)** (2024-11-05)

*Shaokun Zhang, Jieyu Zhang, Dujian Ding, Mirian Hipolito Garcia, Ankur Mallick, Daniel Madrigal, Menglin Xia, Victor Rühle, Qingyun Wu, Chi Wang*

  Recent advancements have enabled Large Language Models (LLMs) to function as
agents that can perform actions using external tools. This requires
registering, i.e., integrating tool information into the LLM context prior to
taking actions. Current methods indiscriminately incorporate all candidate
tools into the agent's context and retain them across multiple reasoning steps.
This process remains opaque to LLM agents and is not integrated into their
reasoning procedures, leading to inefficiencies due to increased context length
from irrelevant tools. To address this, we introduce EcoAct, a tool using
algorithm that allows LLMs to selectively register tools as needed, optimizing
context use. By integrating the tool registration process into the reasoning
procedure, EcoAct reduces computational costs by over 50% in multiple steps
reasoning tasks while maintaining performance, as demonstrated through
extensive experiments. Moreover, it can be plugged into any reasoning pipeline
with only minor modifications to the prompt, making it applicable to LLM agents
now and future.


---

**[47. [2401.05507] InfiAgent-DABench: Evaluating Agents on Data Analysis Tasks](https://arxiv.org/pdf/2401.05507.pdf)** (2024-03-12)

*Xueyu Hu, Ziyu Zhao, Shuang Wei, Ziwei Chai, Qianli Ma, Guoyin Wang, Xuwu Wang, Jing Su, Jingjing Xu, Ming Zhu, Yao Cheng, Jianbo Yuan, Jiwei Li, Kun Kuang, Yang Yang, Hongxia Yang, Fei Wu*

  In this paper, we introduce InfiAgent-DABench, the first benchmark
specifically designed to evaluate LLM-based agents on data analysis tasks.
These tasks require agents to end-to-end solving complex tasks by interacting
with an execution environment. This benchmark contains DAEval, a dataset
consisting of 257 data analysis questions derived from 52 CSV files, and an
agent framework which incorporates LLMs to serve as data analysis agents for
both serving and evaluation. Since data analysis questions are often open-ended
and hard to evaluate without human supervision, we adopt a format-prompting
technique to convert each question into a closed-form format so that they can
be automatically evaluated. Our extensive benchmarking of 34 LLMs uncovers the
current challenges encountered in data analysis tasks. In addition, building on
top of our agent framework, we develop a specialized agent, DAAgent, which
surpasses GPT-3.5 by 3.9% on DABench. Evaluation datasets and toolkits for
InfiAgent-DABench are released at https://github.com/InfiAgent/InfiAgent .


---

**[48. [2210.00211] Boosting Exploration in Actor-Critic Algorithms by Incentivizing
  Plausible Novel States](https://arxiv.org/pdf/2210.00211.pdf)** (2022-10-04)

*Chayan Banerjee, Zhiyong Chen, Nasimul Noman*

  Actor-critic (AC) algorithms are a class of model-free deep reinforcement
learning algorithms, which have proven their efficacy in diverse domains,
especially in solving continuous control problems. Improvement of exploration
(action entropy) and exploitation (expected return) using more efficient
samples is a critical issue in AC algorithms. A basic strategy of a learning
algorithm is to facilitate indiscriminately exploring all of the environment
state space, as well as to encourage exploring rarely visited states rather
than frequently visited one. Under this strategy, we propose a new method to
boost exploration through an intrinsic reward, based on measurement of a
state's novelty and the associated benefit of exploring the state (with regards
to policy optimization), altogether called plausible novelty. With incentivized
exploration of plausible novel states, an AC algorithm is able to improve its
sample efficiency and hence training performance. The new method is verified by
extensive simulations of continuous control tasks of MuJoCo environments on a
variety of prominent off-policy AC algorithms.


---

**[49. [2412.14684] Bel Esprit: Multi-Agent Framework for Building AI Model Pipelines](https://arxiv.org/pdf/2412.14684.pdf)** (2024-12-20)

*Yunsu Kim, AhmedElmogtaba Abdelaziz, Thiago Castro Ferreira, Mohamed Al-Badrashiny, Hassan Sawaf*

  As the demand for artificial intelligence (AI) grows to address complex
real-world tasks, single models are often insufficient, requiring the
integration of multiple models into pipelines. This paper introduces Bel
Esprit, a conversational agent designed to construct AI model pipelines based
on user-defined requirements. Bel Esprit employs a multi-agent framework where
subagents collaborate to clarify requirements, build, validate, and populate
pipelines with appropriate models. We demonstrate the effectiveness of this
framework in generating pipelines from ambiguous user queries, using both
human-curated and synthetic data. A detailed error analysis highlights ongoing
challenges in pipeline construction. Bel Esprit is available for a free trial
at https://belesprit.aixplain.com.


---

**[50. [2207.09714] Differentiable Agent-based Epidemiology](https://arxiv.org/pdf/2207.09714.pdf)** (2023-05-23)

*Ayush Chopra, Alexander Rodríguez, Jayakumar Subramanian, Arnau Quera-Bofarull, Balaji Krishnamurthy, B. Aditya Prakash, Ramesh Raskar*

  Mechanistic simulators are an indispensable tool for epidemiology to explore
the behavior of complex, dynamic infections under varying conditions and
navigate uncertain environments. Agent-based models (ABMs) are an increasingly
popular simulation paradigm that can represent the heterogeneity of contact
interactions with granular detail and agency of individual behavior. However,
conventional ABM frameworks are not differentiable and present challenges in
scalability; due to which it is non-trivial to connect them to auxiliary data
sources. In this paper, we introduce GradABM: a scalable, differentiable design
for agent-based modeling that is amenable to gradient-based learning with
automatic differentiation. GradABM can quickly simulate million-size
populations in few seconds on commodity hardware, integrate with deep neural
networks and ingest heterogeneous data sources. This provides an array of
practical benefits for calibration, forecasting, and evaluating policy
interventions. We demonstrate the efficacy of GradABM via extensive experiments
with real COVID-19 and influenza datasets.


---

**[51. [2210.06835] Multi-agent Dynamic Algorithm Configuration](https://arxiv.org/pdf/2210.06835.pdf)** (2022-10-14)

*Ke Xue, Jiacheng Xu, Lei Yuan, Miqing Li, Chao Qian, Zongzhang Zhang, Yang Yu*

  Automated algorithm configuration relieves users from tedious,
trial-and-error tuning tasks. A popular algorithm configuration tuning paradigm
is dynamic algorithm configuration (DAC), in which an agent learns dynamic
configuration policies across instances by reinforcement learning (RL).
However, in many complex algorithms, there may exist different types of
configuration hyperparameters, and such heterogeneity may bring difficulties
for classic DAC which uses a single-agent RL policy. In this paper, we aim to
address this issue and propose multi-agent DAC (MA-DAC), with one agent working
for one type of configuration hyperparameter. MA-DAC formulates the dynamic
configuration of a complex algorithm with multiple types of hyperparameters as
a contextual multi-agent Markov decision process and solves it by a cooperative
multi-agent RL (MARL) algorithm. To instantiate, we apply MA-DAC to a
well-known optimization algorithm for multi-objective optimization problems.
Experimental results show the effectiveness of MA-DAC in not only achieving
superior performance compared with other configuration tuning approaches based
on heuristic rules, multi-armed bandits, and single-agent RL, but also being
capable of generalizing to different problem classes. Furthermore, we release
the environments in this paper as a benchmark for testing MARL algorithms, with
the hope of facilitating the application of MARL.


---

**[52. [2501.14751] Optimizing LPB Algorithms using Simulated Annealing](https://arxiv.org/pdf/2501.14751.pdf)** (2025-01-30)

*Dana Rasul Hamad, Tarik A. Rashid*

  Learner Performance-based Behavior using Simulated Annealing (LPBSA) is an
improvement of the Learner Performance-based Behavior (LPB) algorithm. LPBSA,
like LPB, has been proven to deal with single and complex problems. Simulated
Annealing (SA) has been utilized as a powerful technique to optimize LPB. LPBSA
has provided results that outperformed popular algorithms, like the Genetic
Algorithm (GA), Particle Swarm Optimization (PSO), and even LPB. This study
outlines the improved algorithm's working procedure by providing a main
population and dividing it into Good and Bad populations and then applying
crossover and mutation operators. When some individuals are born in the
crossover stage, they have to go through the mutation process. Between these
two steps, we have applied SA using the Metropolis Acceptance Criterion (MAC)
to accept only the best and most useful individuals to be used in the next
iteration. Finally, the outcomes demonstrate that the population is enhanced,
leading to improved efficiency and validating the performance of LPBSA.


---

**[53. [2401.13011] CCA: Collaborative Competitive Agents for Image Editing](https://arxiv.org/pdf/2401.13011.pdf)** (2025-02-18)

*Tiankai Hang, Shuyang Gu, Dong Chen, Xin Geng, Baining Guo*

  This paper presents a novel generative model, Collaborative Competitive
Agents (CCA), which leverages the capabilities of multiple Large Language
Models (LLMs) based agents to execute complex tasks. Drawing inspiration from
Generative Adversarial Networks (GANs), the CCA system employs two equal-status
generator agents and a discriminator agent. The generators independently
process user instructions and generate results, while the discriminator
evaluates the outputs, and provides feedback for the generator agents to
further reflect and improve the generation results. Unlike the previous
generative model, our system can obtain the intermediate steps of generation.
This allows each generator agent to learn from other successful executions due
to its transparency, enabling a collaborative competition that enhances the
quality and robustness of the system's results. The primary focus of this study
is image editing, demonstrating the CCA's ability to handle intricate
instructions robustly. The paper's main contributions include the introduction
of a multi-agent-based generative model with controllable intermediate steps
and iterative optimization, a detailed examination of agent relationships, and
comprehensive experiments on image editing. Code is available at
\href{https://github.com/TiankaiHang/CCA}{https://github.com/TiankaiHang/CCA}.


---

**[54. [2207.12051] Flowsheet synthesis through hierarchical reinforcement learning and
  graph neural networks](https://arxiv.org/pdf/2207.12051.pdf)** (2024-01-17)

*Laura Stops, Roel Leenhouts, Qinghe Gao, Artur M. Schweidtmann*

  Process synthesis experiences a disruptive transformation accelerated by
digitization and artificial intelligence. We propose a reinforcement learning
algorithm for chemical process design based on a state-of-the-art actor-critic
logic. Our proposed algorithm represents chemical processes as graphs and uses
graph convolutional neural networks to learn from process graphs. In
particular, the graph neural networks are implemented within the agent
architecture to process the states and make decisions. Moreover, we implement a
hierarchical and hybrid decision-making process to generate flowsheets, where
unit operations are placed iteratively as discrete decisions and corresponding
design variables are selected as continuous decisions. We demonstrate the
potential of our method to design economically viable flowsheets in an
illustrative case study comprising equilibrium reactions, azeotropic
separation, and recycles. The results show quick learning in discrete,
continuous, and hybrid action spaces. Due to the flexible architecture of the
proposed reinforcement learning agent, the method is predestined to include
large action-state spaces and an interface to process simulators in future
research.


---

**[55. [2502.07928] Distributed Approach to Haskell Based Applications Refactoring with LLMs
  Based Multi-Agent Systems](https://arxiv.org/pdf/2502.07928.pdf)** (2025-02-13)

*Shahbaz Siddeeq, Zeeshan Rasheed, Malik Abdul Sami, Mahade Hasan, Muhammad Waseem, Jussi Rasku, Mika Saari, Kai-Kristian Kemell, Pekka Abrahamsson*

  We present a large language models (LLMs) based multi-agent system to
automate the refactoring of Haskell codebases. The multi-agent system consists
of specialized agents performing tasks such as context analysis, refactoring,
validation, and testing. Refactoring improvements are using metrics such as
cyclomatic complexity, run-time, and memory allocation. Experimental
evaluations conducted on Haskell codebases demonstrate improvements in code
quality. Cyclomatic complexity was reduced by 13.64% and 47.06% in the
respective codebases. Memory allocation improved by 4.17% and 41.73%, while
runtime efficiency increased by up to 50%. These metrics highlight the systems
ability to optimize Haskells functional paradigms while maintaining correctness
and scalability. Results show reductions in complexity and performance
enhancements across codebases. The integration of LLMs based multi-agent system
enables precise task execution and inter-agent collaboration, addressing the
challenges of refactoring in functional programming. This approach aims to
address the challenges of refactoring functional programming languages through
distributed and modular systems.


---

**[56. [2407.15318] Modified Bat Algorithm: A Newly Proposed Approach for Solving Complex
  and Real-World Problems](https://arxiv.org/pdf/2407.15318.pdf)** (2024-07-23)

*Shahla U. Umar, Tarik A. Rashid, Aram M. Ahmed, Bryar A. Hassan, Mohammed Rashad Baker*

  Bat Algorithm (BA) is a nature-inspired metaheuristic search algorithm
designed to efficiently explore complex problem spaces and find near-optimal
solutions. The algorithm is inspired by the echolocation behavior of bats,
which acts as a signal system to estimate the distance and hunt prey. Although
the BA has proven effective for various optimization problems, it exhibits
limited exploration ability and susceptibility to local optima. The algorithm
updates velocities and positions based on the current global best solution,
causing all agents to converge towards a specific location, potentially leading
to local optima issues in optimization problems. On this premise, this paper
proposes the Modified Bat Algorithm (MBA) as an enhancement to address the
local optima limitation observed in the original BA. MBA incorporates the
frequency and velocity of the current best solution, enhancing convergence
speed to the optimal solution and preventing local optima entrapment. While the
original BA faces diversity issues, both the original BA and MBA are
introduced. To assess MBAs performance, three sets of test functions (classical
benchmark functions, CEC2005, and CEC2019) are employed, with results compared
to those of the original BA, Particle Swarm Optimization (PSO), Genetic
Algorithm (GA), and Dragonfly Algorithm (DA). The outcomes demonstrate the MBAs
significant superiority over other algorithms. Additionally, MBA successfully
addresses a real-world assignment problem (call center problem), traditionally
solved using linear programming methods, with satisfactory results.


---

**[57. [2010.08925] Implementing Agent-Based Systems via Computability Logic CL2](https://arxiv.org/pdf/2010.08925.pdf)** (2021-08-31)

*Keehang Kwon*

  Computability logic(CoL) is a powerful computational model. In this paper, we
show that CoL naturally supports multi-agent programming models where resources
(coffee for example) are involved. To be specific, we discuss an implementation
of the Starbucks based on CoL (CL2 to be exact).


---

**[58. [2205.01927] Probabilistic Symmetry for Multi-Agent Dynamics](https://arxiv.org/pdf/2205.01927.pdf)** (2023-05-22)

*Sophia Sun, Robin Walters, Jinxi Li, Rose Yu*

  Learning multi-agent dynamics is a core AI problem with broad applications in
robotics and autonomous driving. While most existing works focus on
deterministic prediction, producing probabilistic forecasts to quantify
uncertainty and assess risks is critical for downstream decision-making tasks
such as motion planning and collision avoidance. Multi-agent dynamics often
contains internal symmetry. By leveraging symmetry, specifically rotation
equivariance, we can improve not only the prediction accuracy but also
uncertainty calibration. We introduce Energy Score, a proper scoring rule, to
evaluate probabilistic predictions. We propose a novel deep dynamics model,
Probabilistic Equivariant Continuous COnvolution (PECCO) for probabilistic
prediction of multi-agent trajectories. PECCO extends equivariant continuous
convolution to model the joint velocity distribution of multiple agents. It
uses dynamics integration to propagate the uncertainty from velocity to
position. On both synthetic and real-world datasets, PECCO shows significant
improvements in accuracy and calibration compared to non-equivariant baselines.


---

**[59. [2202.11932] Collective Conditioned Reflex: A Bio-Inspired Fast Emergency Reaction
  Mechanism for Designing Safe Multi-Robot Systems](https://arxiv.org/pdf/2202.11932.pdf)** (2022-08-19)

*Bowei He, Zhenting Zhao, Wenhao Luo, Rui Liu*

  A multi-robot system (MRS) is a group of coordinated robots designed to
cooperate with each other and accomplish given tasks. Due to the uncertainties
in operating environments, the system may encounter emergencies, such as
unobserved obstacles, moving vehicles, and extreme weather. Animal groups such
as bee colonies initiate collective emergency reaction behaviors such as
bypassing obstacles and avoiding predators, similar to muscle-conditioned
reflex which organizes local muscles to avoid hazards in the first response
without delaying passage through the brain. Inspired by this, we develop a
similar collective conditioned reflex mechanism for multi-robot systems to
respond to emergencies. In this study, Collective Conditioned Reflex (CCR), a
bio-inspired emergency reaction mechanism, is developed based on animal
collective behavior analysis and multi-agent reinforcement learning (MARL). The
algorithm uses a physical model to determine if the robots are experiencing an
emergency; then, rewards for robots involved in the emergency are augmented
with corresponding heuristic rewards, which evaluate emergency magnitudes and
consequences and decide local robots' participation. CCR is validated on three
typical emergency scenarios: \textit{turbulence, strong wind, and hidden
obstacle}. Simulation results demonstrate that CCR improves robot teams'
emergency reaction capability with faster reaction speed and safer trajectory
adjustment compared with baseline methods.


---

**[60. [2504.03260] Gradient Field-Based Dynamic Window Approach for Collision Avoidance in
  Complex Environments](https://arxiv.org/pdf/2504.03260.pdf)** (2025-04-07)

*Ze Zhang, Yifan Xue, Nadia Figueroa, Knut Åkesson*

  For safe and flexible navigation in multi-robot systems, this paper presents
an enhanced and predictive sampling-based trajectory planning approach in
complex environments, the Gradient Field-based Dynamic Window Approach
(GF-DWA). Building upon the dynamic window approach, the proposed method
utilizes gradient information of obstacle distances as a new cost term to
anticipate potential collisions. This enhancement enables the robot to improve
awareness of obstacles, including those with non-convex shapes. The gradient
field is derived from the Gaussian process distance field, which generates both
the distance field and gradient field by leveraging Gaussian process regression
to model the spatial structure of the environment. Through several obstacle
avoidance and fleet collision avoidance scenarios, the proposed GF-DWA is shown
to outperform other popular trajectory planning and control methods in terms of
safety and flexibility, especially in complex environments with non-convex
obstacles.


---

**[61. [2408.16517] Adaptive Variational Continual Learning via Task-Heuristic Modelling](https://arxiv.org/pdf/2408.16517.pdf)** (2024-08-30)

*Fan Yang*

  Variational continual learning (VCL) is a turn-key learning algorithm that
has state-of-the-art performance among the best continual learning models. In
our work, we explore an extension of the generalized variational continual
learning (GVCL) model, named AutoVCL, which combines task heuristics for
informed learning and model optimization. We demonstrate that our model
outperforms the standard GVCL with fixed hyperparameters, benefiting from the
automatic adjustment of the hyperparameter based on the difficulty and
similarity of the incoming task compared to the previous tasks.


---

**[62. [2405.15460] TD3 Based Collision Free Motion Planning for Robot Navigation](https://arxiv.org/pdf/2405.15460.pdf)** (2024-05-27)

*Hao Liu, Yi Shen, Chang Zhou, Yuelin Zou, Zijun Gao, Qi Wang*

  This paper addresses the challenge of collision-free motion planning in
automated navigation within complex environments. Utilizing advancements in
Deep Reinforcement Learning (DRL) and sensor technologies like LiDAR, we
propose the TD3-DWA algorithm, an innovative fusion of the traditional Dynamic
Window Approach (DWA) with the Twin Delayed Deep Deterministic Policy Gradient
(TD3). This hybrid algorithm enhances the efficiency of robotic path planning
by optimizing the sampling interval parameters of DWA to effectively navigate
around both static and dynamic obstacles. The performance of the TD3-DWA
algorithm is validated through various simulation experiments, demonstrating
its potential to significantly improve the reliability and safety of autonomous
navigation systems.


---

**[63. [2502.14345] FlowAgent: Achieving Compliance and Flexibility for Workflow Agents](https://arxiv.org/pdf/2502.14345.pdf)** (2025-02-21)

*Yuchen Shi, Siqi Cai, Zihan Xu, Yuei Qin, Gang Li, Hang Shao, Jiawei Chen, Deqing Yang, Ke Li, Xing Sun*

  The integration of workflows with large language models (LLMs) enables
LLM-based agents to execute predefined procedures, enhancing automation in
real-world applications. Traditional rule-based methods tend to limit the
inherent flexibility of LLMs, as their predefined execution paths restrict the
models' action space, particularly when the unexpected, out-of-workflow (OOW)
queries are encountered. Conversely, prompt-based methods allow LLMs to fully
control the flow, which can lead to diminished enforcement of procedural
compliance. To address these challenges, we introduce FlowAgent, a novel agent
framework designed to maintain both compliance and flexibility. We propose the
Procedure Description Language (PDL), which combines the adaptability of
natural language with the precision of code to formulate workflows. Building on
PDL, we develop a comprehensive framework that empowers LLMs to manage OOW
queries effectively, while keeping the execution path under the supervision of
a set of controllers. Additionally, we present a new evaluation methodology to
rigorously assess an LLM agent's ability to handle OOW scenarios, going beyond
routine flow compliance tested in existing benchmarks. Experiments on three
datasets demonstrate that FlowAgent not only adheres to workflows but also
effectively manages OOW queries, highlighting its dual strengths in compliance
and flexibility. The code is available at
https://github.com/Lightblues/FlowAgent.


---

**[64. [2309.04710] Jade: A Differentiable Physics Engine for Articulated Rigid Bodies with
  Intersection-Free Frictional Contact](https://arxiv.org/pdf/2309.04710.pdf)** (2023-09-12)

*Gang Yang, Siyuan Luo, Lin Shao*

  We present Jade, a differentiable physics engine for articulated rigid
bodies. Jade models contacts as the Linear Complementarity Problem (LCP).
Compared to existing differentiable simulations, Jade offers features including
intersection-free collision simulation and stable LCP solutions for multiple
frictional contacts. We use continuous collision detection to detect the time
of impact and adopt the backtracking strategy to prevent intersection between
bodies with complex geometry shapes. We derive the gradient calculation to
ensure the whole simulation process is differentiable under the backtracking
mechanism. We modify the popular Dantzig algorithm to get valid solutions under
multiple frictional contacts. We conduct extensive experiments to demonstrate
the effectiveness of our differentiable physics simulation over a variety of
contact-rich tasks.


---

**[65. [2212.01454] Agent Miner: An Algorithm for Discovering Agent Systems from Event Data](https://arxiv.org/pdf/2212.01454.pdf)** (2023-07-24)

*Andrei Tour, Artem Polyvyanyy, Anna Kalenkova, Arik Senderovich*

  Process discovery studies ways to use event data generated by business
processes and recorded by IT systems to construct models that describe the
processes. Existing discovery algorithms are predominantly concerned with
constructing process models that represent the control flow of the processes.
Agent system mining argues that business processes often emerge from
interactions of autonomous agents and uses event data to construct models of
the agents and their interactions. This paper presents and evaluates Agent
Miner, an algorithm for discovering models of agents and their interactions
from event data composing the system that has executed the processes which
generated the input data. The conducted evaluation using our open-source
implementation of Agent Miner and publicly available industrial datasets
confirms that our algorithm can provide insights into the process participants
and their interaction patterns and often discovers models that describe the
business processes more faithfully than process models discovered using
conventional process discovery algorithms.


---

**[66. [2410.04078] TeachTune: Reviewing Pedagogical Agents Against Diverse Student Profiles
  with Simulated Students](https://arxiv.org/pdf/2410.04078.pdf)** (2025-01-31)

*Hyoungwook Jin, Minju Yoo, Jeongeon Park, Yokyung Lee, Xu Wang, Juho Kim*

  Large language models (LLMs) can empower teachers to build pedagogical
conversational agents (PCAs) customized for their students. As students have
different prior knowledge and motivation levels, teachers must review the
adaptivity of their PCAs to diverse students. Existing chatbot reviewing
methods (e.g., direct chat and benchmarks) are either manually intensive for
multiple iterations or limited to testing only single-turn interactions. We
present TeachTune, where teachers can create simulated students and review PCAs
by observing automated chats between PCAs and simulated students. Our technical
pipeline instructs an LLM-based student to simulate prescribed knowledge levels
and traits, helping teachers explore diverse conversation patterns. Our
pipeline could produce simulated students whose behaviors correlate highly to
their input knowledge and motivation levels within 5% and 10% accuracy gaps.
Thirty science teachers designed PCAs in a between-subjects study, and using
TeachTune resulted in a lower task load and higher student profile coverage
over a baseline.


---

**[67. [2311.17541] TaskWeaver: A Code-First Agent Framework](https://arxiv.org/pdf/2311.17541.pdf)** (2024-06-21)

*Bo Qiao, Liqun Li, Xu Zhang, Shilin He, Yu Kang, Chaoyun Zhang, Fangkai Yang, Hang Dong, Jue Zhang, Lu Wang, Minghua Ma, Pu Zhao, Si Qin, Xiaoting Qin, Chao Du, Yong Xu, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang*

  Large Language Models (LLMs) have shown impressive abilities in natural
language understanding and generation, leading to their widespread use in
applications such as chatbots and virtual assistants. However, existing LLM
frameworks face limitations in handling domain-specific data analytics tasks
with rich data structures. Moreover, they struggle with flexibility to meet
diverse user requirements. To address these issues, TaskWeaver is proposed as a
code-first framework for building LLM-powered autonomous agents. It converts
user requests into executable code and treats user-defined plugins as callable
functions. TaskWeaver provides support for rich data structures, flexible
plugin usage, and dynamic plugin selection, and leverages LLM coding
capabilities for complex logic. It also incorporates domain-specific knowledge
through examples and ensures the secure execution of generated code. TaskWeaver
offers a powerful and flexible framework for creating intelligent
conversational agents that can handle complex tasks and adapt to
domain-specific scenarios. The code is open sourced at
https://github.com/microsoft/TaskWeaver/.


---

**[68. [2503.17671] ComfyGPT: A Self-Optimizing Multi-Agent System for Comprehensive ComfyUI
  Workflow Generation](https://arxiv.org/pdf/2503.17671.pdf)** (2025-03-25)

*Oucheng Huang, Yuhang Ma, Zeng Zhao, Mingrui Wu, Jiayi Ji, Rongsheng Zhang, Zhipeng Hu, Xiaoshuai Sun, Rongrong Ji*

  ComfyUI provides a widely-adopted, workflow-based interface that enables
users to customize various image generation tasks through an intuitive
node-based architecture. However, the intricate connections between nodes and
diverse modules often present a steep learning curve for users. In this paper,
we introduce ComfyGPT, the first self-optimizing multi-agent system designed to
generate ComfyUI workflows based on task descriptions automatically. ComfyGPT
comprises four specialized agents: ReformatAgent, FlowAgent, RefineAgent, and
ExecuteAgent. The core innovation of ComfyGPT lies in two key aspects. First,
it focuses on generating individual node links rather than entire workflows,
significantly improving generation precision. Second, we proposed FlowAgent, a
LLM-based workflow generation agent that uses both supervised fine-tuning (SFT)
and reinforcement learning (RL) to improve workflow generation accuracy.
Moreover, we introduce FlowDataset, a large-scale dataset containing 13,571
workflow-description pairs, and FlowBench, a comprehensive benchmark for
evaluating workflow generation systems. We also propose four novel evaluation
metrics: Format Validation (FV), Pass Accuracy (PA), Pass Instruct Alignment
(PIA), and Pass Node Diversity (PND). Experimental results demonstrate that
ComfyGPT significantly outperforms existing LLM-based methods in workflow
generation.


---

**[69. [2403.03031] Learning to Use Tools via Cooperative and Interactive Agents](https://arxiv.org/pdf/2403.03031.pdf)** (2024-06-25)

*Zhengliang Shi, Shen Gao, Xiuyi Chen, Yue Feng, Lingyong Yan, Haibo Shi, Dawei Yin, Pengjie Ren, Suzan Verberne, Zhaochun Ren*

  Tool learning empowers large language models (LLMs) as agents to use external
tools and extend their utility. Existing methods employ one single LLM-based
agent to iteratively select and execute tools, thereafter incorporating
execution results into the next action prediction. Despite their progress,
these methods suffer from performance degradation when addressing practical
tasks due to: (1) the pre-defined pipeline with restricted flexibility to
calibrate incorrect actions, and (2) the struggle to adapt a general LLM-based
agent to perform a variety of specialized actions. To mitigate these problems,
we propose ConAgents, a Cooperative and interactive Agents framework, which
coordinates three specialized agents for tool selection, tool execution, and
action calibration separately. ConAgents introduces two communication protocols
to enable the flexible cooperation of agents. To effectively generalize the
ConAgents into open-source models, we also propose specialized action
distillation, enhancing their ability to perform specialized actions in our
framework. Our extensive experiments on three datasets show that the LLMs, when
equipped with the ConAgents, outperform baselines with substantial improvement
(i.e., up to 14% higher success rate).


---

**[70. [2410.17186] DyPNIPP: Predicting Environment Dynamics for RL-based Robust Informative
  Path Planning](https://arxiv.org/pdf/2410.17186.pdf)** (2024-10-23)

*Srujan Deolasee, Siva Kailas, Wenhao Luo, Katia Sycara, Woojun Kim*

  Informative path planning (IPP) is an important planning paradigm for various
real-world robotic applications such as environment monitoring. IPP involves
planning a path that can learn an accurate belief of the quantity of interest,
while adhering to planning constraints. Traditional IPP methods typically
require high computation time during execution, giving rise to reinforcement
learning (RL) based IPP methods. However, the existing RL-based methods do not
consider spatio-temporal environments which involve their own challenges due to
variations in environment characteristics. In this paper, we propose DyPNIPP, a
robust RL-based IPP framework, designed to operate effectively across
spatio-temporal environments with varying dynamics. To achieve this, DyPNIPP
incorporates domain randomization to train the agent across diverse
environments and introduces a dynamics prediction model to capture and adapt
the agent actions to specific environment dynamics. Our extensive experiments
in a wildfire environment demonstrate that DyPNIPP outperforms existing
RL-based IPP algorithms by significantly improving robustness and performing
across diverse environment conditions.


---

**[71. [2011.11134] Differentiable Computational Geometry for 2D and 3D machine learning](https://arxiv.org/pdf/2011.11134.pdf)** (2020-11-24)

*Yuanxin Zhong*

  With the growth of machine learning algorithms with geometry primitives, a
high-efficiency library with differentiable geometric operators are desired. We
present an optimized Differentiable Geometry Algorithm Library (DGAL) loaded
with implementations of differentiable operators for geometric primitives like
lines and polygons. The library is a header-only templated C++ library with GPU
support. We discuss the internal design of the library and benchmark its
performance on some tasks with other implementations.


---

**[72. [2305.11831] Regularization of Soft Actor-Critic Algorithms with Automatic
  Temperature Adjustment](https://arxiv.org/pdf/2305.11831.pdf)** (2023-05-24)

*Ben You*

  This work presents a comprehensive analysis to regularize the Soft
Actor-Critic (SAC) algorithm with automatic temperature adjustment. The the
policy evaluation, the policy improvement and the temperature adjustment are
reformulated, addressing certain modification and enhancing the clarity of the
original theory in a more explicit manner.


---

**[73. [2410.17809] An Intelligent Agentic System for Complex Image Restoration Problems](https://arxiv.org/pdf/2410.17809.pdf)** (2025-02-18)

*Kaiwen Zhu, Jinjin Gu, Zhiyuan You, Yu Qiao, Chao Dong*

  Real-world image restoration (IR) is inherently complex and often requires
combining multiple specialized models to address diverse degradations. Inspired
by human problem-solving, we propose AgenticIR, an agentic system that mimics
the human approach to image processing by following five key stages:
Perception, Scheduling, Execution, Reflection, and Rescheduling. AgenticIR
leverages large language models (LLMs) and vision-language models (VLMs) that
interact via text generation to dynamically operate a toolbox of IR models. We
fine-tune VLMs for image quality analysis and employ LLMs for reasoning,
guiding the system step by step. To compensate for LLMs' lack of specific IR
knowledge and experience, we introduce a self-exploration method, allowing the
LLM to observe and summarize restoration results into referenceable documents.
Experiments demonstrate AgenticIR's potential in handling complex IR tasks,
representing a promising path toward achieving general intelligence in visual
processing.


---

**[74. [2203.07092] The Multi-Agent Pickup and Delivery Problem: MAPF, MARL and Its
  Warehouse Applications](https://arxiv.org/pdf/2203.07092.pdf)** (2022-03-15)

*Tim Tsz-Kit Lau, Biswa Sengupta*

  We study two state-of-the-art solutions to the multi-agent pickup and
delivery (MAPD) problem based on different principles -- multi-agent
path-finding (MAPF) and multi-agent reinforcement learning (MARL).
Specifically, a recent MAPF algorithm called conflict-based search (CBS) and a
current MARL algorithm called shared experience actor-critic (SEAC) are
studied. While the performance of these algorithms is measured using quite
different metrics in their separate lines of work, we aim to benchmark these
two methods comprehensively in a simulated warehouse automation environment.


---

**[75. [2210.13066] DaXBench: Benchmarking Deformable Object Manipulation with
  Differentiable Physics](https://arxiv.org/pdf/2210.13066.pdf)** (2023-03-13)

*Siwei Chen, Yiqing Xu, Cunjun Yu, Linfeng Li, Xiao Ma, Zhongwen Xu, David Hsu*

  Deformable Object Manipulation (DOM) is of significant importance to both
daily and industrial applications. Recent successes in differentiable physics
simulators allow learning algorithms to train a policy with analytic gradients
through environment dynamics, which significantly facilitates the development
of DOM algorithms. However, existing DOM benchmarks are either
single-object-based or non-differentiable. This leaves the questions of 1) how
a task-specific algorithm performs on other tasks and 2) how a
differentiable-physics-based algorithm compares with the non-differentiable
ones in general. In this work, we present DaXBench, a differentiable DOM
benchmark with a wide object and task coverage. DaXBench includes 9 challenging
high-fidelity simulated tasks, covering rope, cloth, and liquid manipulation
with various difficulty levels. To better understand the performance of general
algorithms on different DOM tasks, we conduct comprehensive experiments over
representative DOM methods, ranging from planning to imitation learning and
reinforcement learning. In addition, we provide careful empirical studies of
existing decision-making algorithms based on differentiable physics, and
discuss their limitations, as well as potential future directions.


---

**[76. [2401.08999] Continuous Time Continuous Space Homeostatic Reinforcement Learning
  (CTCS-HRRL) : Towards Biological Self-Autonomous Agent](https://arxiv.org/pdf/2401.08999.pdf)** (2024-01-18)

*Hugo Laurencon, Yesoda Bhargava, Riddhi Zantye, Charbel-Raphaël Ségerie, Johann Lussange, Veeky Baths, Boris Gutkin*

  Homeostasis is a biological process by which living beings maintain their
internal balance. Previous research suggests that homeostasis is a learned
behaviour. Recently introduced Homeostatic Regulated Reinforcement Learning
(HRRL) framework attempts to explain this learned homeostatic behavior by
linking Drive Reduction Theory and Reinforcement Learning. This linkage has
been proven in the discrete time-space, but not in the continuous time-space.
In this work, we advance the HRRL framework to a continuous time-space
environment and validate the CTCS-HRRL (Continuous Time Continuous Space HRRL)
framework. We achieve this by designing a model that mimics the homeostatic
mechanisms in a real-world biological agent. This model uses the
Hamilton-Jacobian Bellman Equation, and function approximation based on neural
networks and Reinforcement Learning. Through a simulation-based experiment we
demonstrate the efficacy of this model and uncover the evidence linked to the
agent's ability to dynamically choose policies that favor homeostasis in a
continuously changing internal-state milieu. Results of our experiments
demonstrate that agent learns homeostatic behaviour in a CTCS environment,
making CTCS-HRRL a promising framework for modellng animal dynamics and
decision-making.


---

**[77. [2501.07834] Flow: Modularized Agentic Workflow Automation](https://arxiv.org/pdf/2501.07834.pdf)** (2025-02-25)

*Boye Niu, Yiliao Song, Kai Lian, Yifan Shen, Yu Yao, Kun Zhang, Tongliang Liu*

  Multi-agent frameworks powered by large language models (LLMs) have
demonstrated great success in automated planning and task execution. However,
the effective adjustment of agentic workflows during execution has not been
well studied. An effective workflow adjustment is crucial in real-world
scenarios, as the initial plan must adjust to unforeseen challenges and
changing conditions in real time to ensure the efficient execution of complex
tasks. In this paper, we define workflows as an activity-on-vertex (AOV) graph,
which allows continuous workflow refinement by LLM agents through dynamic
subtask allocation adjustment based on historical performance and previous
AOVs. To further enhance framework performance, we emphasize modularity in
workflow design based on evaluating parallelism and dependency complexity. With
this design, our proposed multi-agent framework achieves efficient concurrent
execution of subtasks, effective goal achievement, and enhanced error
tolerance. Empirical results across various practical tasks demonstrate
significant improvements in the efficiency of multi-agent frameworks through
dynamic workflow refinement and modularization. The code is available at:
https://github.com/tmllab/2025_ICLR_FLOW.


---

**[78. [2009.06419] Federated Generalized Bayesian Learning via Distributed Stein
  Variational Gradient Descent](https://arxiv.org/pdf/2009.06419.pdf)** (2021-03-31)

*Rahif Kassab, Osvaldo Simeone*

  This paper introduces Distributed Stein Variational Gradient Descent (DSVGD),
a non-parametric generalized Bayesian inference framework for federated
learning. DSVGD maintains a number of non-random and interacting particles at a
central server to represent the current iterate of the model global posterior.
The particles are iteratively downloaded and updated by one of the agents with
the end goal of minimizing the global free energy. By varying the number of
particles, DSVGD enables a flexible trade-off between per-iteration
communication load and number of communication rounds. DSVGD is shown to
compare favorably to benchmark frequentist and Bayesian federated learning
strategies, also scheduling a single device per iteration, in terms of accuracy
and scalability with respect to the number of agents, while also providing
well-calibrated, and hence trustworthy, predictions.


---

**[79. [2409.07918] Tidal MerzA: Combining affective modelling and autonomous code
  generation through Reinforcement Learning](https://arxiv.org/pdf/2409.07918.pdf)** (2024-09-13)

*Elizabeth Wilson, György Fazekas, Geraint Wiggins*

  This paper presents Tidal-MerzA, a novel system designed for collaborative
performances between humans and a machine agent in the context of live coding,
specifically focusing on the generation of musical patterns. Tidal-MerzA fuses
two foundational models: ALCAA (Affective Live Coding Autonomous Agent) and
Tidal Fuzz, a computational framework. By integrating affective modelling with
computational generation, this system leverages reinforcement learning
techniques to dynamically adapt music composition parameters within the
TidalCycles framework, ensuring both affective qualities to the patterns and
syntactical correctness. The development of Tidal-MerzA introduces two distinct
agents: one focusing on the generation of mini-notation strings for musical
expression, and another on the alignment of music with targeted affective
states through reinforcement learning. This approach enhances the adaptability
and creative potential of live coding practices and allows exploration of
human-machine creative interactions. Tidal-MerzA advances the field of
computational music generation, presenting a novel methodology for
incorporating artificial intelligence into artistic practices.


---

**[80. [2009.07783] Generative Language-Grounded Policy in Vision-and-Language Navigation
  with Bayes' Rule](https://arxiv.org/pdf/2009.07783.pdf)** (2020-10-09)

*Shuhei Kurita, Kyunghyun Cho*

  Vision-and-language navigation (VLN) is a task in which an agent is embodied
in a realistic 3D environment and follows an instruction to reach the goal
node. While most of the previous studies have built and investigated a
discriminative approach, we notice that there are in fact two possible
approaches to building such a VLN agent: discriminative \textit{and}
generative. In this paper, we design and investigate a generative
language-grounded policy which uses a language model to compute the
distribution over all possible instructions i.e. all possible sequences of
vocabulary tokens given action and the transition history. In experiments, we
show that the proposed generative approach outperforms the discriminative
approach in the Room-2-Room (R2R) and Room-4-Room (R4R) datasets, especially in
the unseen environments. We further show that the combination of the generative
and discriminative policies achieves close to the state-of-the art results in
the R2R dataset, demonstrating that the generative and discriminative policies
capture the different aspects of VLN.


---

**[81. [2404.10740] N-Agent Ad Hoc Teamwork](https://arxiv.org/pdf/2404.10740.pdf)** (2024-10-07)

*Caroline Wang, Arrasy Rahman, Ishan Durugkar, Elad Liebman, Peter Stone*

  Current approaches to learning cooperative multi-agent behaviors assume
relatively restrictive settings. In standard fully cooperative multi-agent
reinforcement learning, the learning algorithm controls $\textit{all}$ agents
in the scenario, while in ad hoc teamwork, the learning algorithm usually
assumes control over only a $\textit{single}$ agent in the scenario. However,
many cooperative settings in the real world are much less restrictive. For
example, in an autonomous driving scenario, a company might train its cars with
the same learning algorithm, yet once on the road, these cars must cooperate
with cars from another company. Towards expanding the class of scenarios that
cooperative learning methods may optimally address, we introduce $N$-agent ad
hoc teamwork (NAHT), where a set of autonomous agents must interact and
cooperate with dynamically varying numbers and types of teammates. This paper
formalizes the problem, and proposes the Policy Optimization with Agent
Modelling (POAM) algorithm. POAM is a policy gradient, multi-agent
reinforcement learning approach to the NAHT problem, that enables adaptation to
diverse teammate behaviors by learning representations of teammate behaviors.
Empirical evaluation on tasks from the multi-agent particle environment and
StarCraft II shows that POAM improves cooperative task returns compared to
baseline approaches, and enables out-of-distribution generalization to unseen
teammates.


---

**[82. [2312.02231] Quality Diversity in the Amorphous Fortress (QD-AF): Evolving for
  Complexity in 0-Player Games](https://arxiv.org/pdf/2312.02231.pdf)** (2023-12-06)

*Sam Earle, M Charity, Dipika Rajesh, Mayu Wilson, Julian Togelius*

  We explore the generation of diverse environments using the Amorphous
Fortress (AF) simulation framework. AF defines a set of Finite State Machine
(FSM) nodes and edges that can be recombined to control the behavior of agents
in the `fortress' grid-world. The behaviors and conditions of the agents within
the framework are designed to capture the common building blocks of multi-agent
artificial life and reinforcement learning environments. Using quality
diversity evolutionary search, we generate diverse sets of environments. These
environments exhibit certain types of complexity according to measures of
agents' FSM architectures and activations, and collective behaviors. Our
approach, Quality Diversity in Amorphous Fortress (QD-AF) generates families of
0-player games akin to simplistic ecological models, and we identify the
emergence of both competitive and co-operative multi-agent and multi-species
survival dynamics. We argue that these generated worlds can collectively serve
as training and testing grounds for learning algorithms.


---

**[83. [2408.08571] AgentSimulator: An Agent-based Approach for Data-driven Business Process
  Simulation](https://arxiv.org/pdf/2408.08571.pdf)** (2024-08-19)

*Lukas Kirchdorfer, Robert Blümel, Timotheus Kampik, Han van der Aa, Heiner Stuckenschmidt*

  Business process simulation (BPS) is a versatile technique for estimating
process performance across various scenarios. Traditionally, BPS approaches
employ a control-flow-first perspective by enriching a process model with
simulation parameters. Although such approaches can mimic the behavior of
centrally orchestrated processes, such as those supported by workflow systems,
current control-flow-first approaches cannot faithfully capture the dynamics of
real-world processes that involve distinct resource behavior and decentralized
decision-making. Recognizing this issue, this paper introduces AgentSimulator,
a resource-first BPS approach that discovers a multi-agent system from an event
log, modeling distinct resource behaviors and interaction patterns to simulate
the underlying process. Our experiments show that AgentSimulator achieves
state-of-the-art simulation accuracy with significantly lower computation times
than existing approaches while providing high interpretability and adaptability
to different types of process-execution scenarios.


---

**[84. [2111.05318] A Differentiable Recipe for Learning Visual Non-Prehensile Planar
  Manipulation](https://arxiv.org/pdf/2111.05318.pdf)** (2021-11-10)

*Bernardo Aceituno, Alberto Rodriguez, Shubham Tulsiani, Abhinav Gupta, Mustafa Mukadam*

  Specifying tasks with videos is a powerful technique towards acquiring novel
and general robot skills. However, reasoning over mechanics and dexterous
interactions can make it challenging to scale learning contact-rich
manipulation. In this work, we focus on the problem of visual non-prehensile
planar manipulation: given a video of an object in planar motion, find
contact-aware robot actions that reproduce the same object motion. We propose a
novel architecture, Differentiable Learning for Manipulation (\ours), that
combines video decoding neural models with priors from contact mechanics by
leveraging differentiable optimization and finite difference based simulation.
Through extensive simulated experiments, we investigate the interplay between
traditional model-based techniques and modern deep learning approaches. We find
that our modular and fully differentiable architecture performs better than
learning-only methods on unseen objects and motions.
\url{https://github.com/baceituno/dlm}.


---

**[85. [2111.10003] Differentiable Wavetable Synthesis](https://arxiv.org/pdf/2111.10003.pdf)** (2022-02-15)

*Siyuan Shan, Lamtharn Hantrakul, Jitong Chen, Matt Avent, David Trevelyan*

  Differentiable Wavetable Synthesis (DWTS) is a technique for neural audio
synthesis which learns a dictionary of one-period waveforms i.e. wavetables,
through end-to-end training. We achieve high-fidelity audio synthesis with as
little as 10 to 20 wavetables and demonstrate how a data-driven dictionary of
waveforms opens up unprecedented one-shot learning paradigms on short audio
clips. Notably, we show audio manipulations, such as high quality
pitch-shifting, using only a few seconds of input audio. Lastly, we investigate
performance gains from using learned wavetables for realtime and interactive
audio synthesis.


---

**[86. [2409.18444] Cost-Aware Dynamic Cloud Workflow Scheduling using Self-Attention and
  Evolutionary Reinforcement Learning](https://arxiv.org/pdf/2409.18444.pdf)** (2024-12-31)

*Ya Shen, Gang Chen, Hui Ma, Mengjie Zhang*

  The Cost-aware Dynamic Multi-Workflow Scheduling (CDMWS) in the cloud is a
kind of cloud workflow management problem, which aims to assign virtual machine
(VM) instances to execute tasks in workflows so as to minimize the total costs,
including both the penalties for violating Service Level Agreement (SLA) and
the VM rental fees. Powered by deep neural networks, Reinforcement Learning
(RL) methods can construct effective scheduling policies for solving CDMWS
problems. Traditional policy networks in RL often use basic feedforward
architectures to separately determine the suitability of assigning any VM
instances, without considering all VMs simultaneously to learn their global
information. This paper proposes a novel self-attention policy network for
cloud workflow scheduling (SPN-CWS) that captures global information from all
VMs. We also develop an Evolution Strategy-based RL (ERL) system to train
SPN-CWS reliably and effectively. The trained SPN-CWS can effectively process
all candidate VM instances simultaneously to identify the most suitable VM
instance to execute every workflow task. Comprehensive experiments show that
our method can noticeably outperform several state-of-the-art algorithms on
multiple benchmark CDMWS problems.


---

**[87. [2405.03133] Lory: Fully Differentiable Mixture-of-Experts for Autoregressive
  Language Model Pre-training](https://arxiv.org/pdf/2405.03133.pdf)** (2024-08-20)

*Zexuan Zhong, Mengzhou Xia, Danqi Chen, Mike Lewis*

  Mixture-of-experts (MoE) models facilitate efficient scaling; however,
training the router network introduces the challenge of optimizing a
non-differentiable, discrete objective. Recently, a fully-differentiable MoE
architecture, SMEAR, was proposed (Muqeeth et al., 2023), which softly merges
experts in the parameter space; nevertheless, its effectiveness was only
demonstrated in downstream fine-tuning on classification tasks. In this paper,
we present Lory, the first approach that scales such architectures to
autoregressive language model pre-training. Lory introduces two key techniques:
(1) a causal segment routing strategy that achieves high efficiency for expert
merging operations while preserving the autoregressive nature of language
models; (2) a similarity-based data batching method that encourages expert
specialization by grouping similar documents in training instances. We
pre-train a series of Lory models on 150B tokens from scratch, with up to 32
experts and 30B (1.5B active) parameters. Experimental results show significant
performance gains over parameter-matched dense models on both perplexity
(+13.9%) and a variety of downstream tasks (+1.5%-11.1%). Despite segment-level
routing, Lory models achieve competitive performance compared to
state-of-the-art MoE models with token-level routing. We further demonstrate
that the trained experts in Lory capture domain-level specialization without
supervision. Our work highlights the potential of fully-differentiable MoE
architectures for language model pre-training and advocates future research in
this area.


---

**[88. [2403.03636] SheetAgent: Towards A Generalist Agent for Spreadsheet Reasoning and
  Manipulation via Large Language Models](https://arxiv.org/pdf/2403.03636.pdf)** (2025-03-04)

*Yibin Chen, Yifu Yuan, Zeyu Zhang, Yan Zheng, Jinyi Liu, Fei Ni, Jianye Hao, Hangyu Mao, Fuzheng Zhang*

  Spreadsheets are ubiquitous across the World Wide Web, playing a critical
role in enhancing work efficiency across various domains. Large language model
(LLM) has been recently attempted for automatic spreadsheet manipulation but
has not yet been investigated in complicated and realistic tasks where
reasoning challenges exist (e.g., long horizon manipulation with multi-step
reasoning and ambiguous requirements). To bridge the gap with the real-world
requirements, we introduce SheetRM, a benchmark featuring long-horizon and
multi-category tasks with reasoning-dependent manipulation caused by real-life
challenges. To mitigate the above challenges, we further propose SheetAgent, a
novel autonomous agent that utilizes the power of LLMs. SheetAgent consists of
three collaborative modules: Planner, Informer, and Retriever, achieving both
advanced reasoning and accurate manipulation over spreadsheets without human
interaction through iterative task reasoning and reflection. Extensive
experiments demonstrate that SheetAgent delivers 20--40\% pass rate
improvements on multiple benchmarks over baselines, achieving enhanced
precision in spreadsheet manipulation and demonstrating superior table
reasoning abilities. More details and visualizations are available at the
project website: https://sheetagent.github.io/. The datasets and source code
are available at https://anonymous.4open.science/r/SheetAgent.


---

**[89. [2103.09815] TeachMyAgent: a Benchmark for Automatic Curriculum Learning in Deep RL](https://arxiv.org/pdf/2103.09815.pdf)** (2021-06-10)

*Clément Romac, Rémy Portelas, Katja Hofmann, Pierre-Yves Oudeyer*

  Training autonomous agents able to generalize to multiple tasks is a key
target of Deep Reinforcement Learning (DRL) research. In parallel to improving
DRL algorithms themselves, Automatic Curriculum Learning (ACL) study how
teacher algorithms can train DRL agents more efficiently by adapting task
selection to their evolving abilities. While multiple standard benchmarks exist
to compare DRL agents, there is currently no such thing for ACL algorithms.
Thus, comparing existing approaches is difficult, as too many experimental
parameters differ from paper to paper. In this work, we identify several key
challenges faced by ACL algorithms. Based on these, we present TeachMyAgent
(TA), a benchmark of current ACL algorithms leveraging procedural task
generation. It includes 1) challenge-specific unit-tests using variants of a
procedural Box2D bipedal walker environment, and 2) a new procedural Parkour
environment combining most ACL challenges, making it ideal for global
performance assessment. We then use TeachMyAgent to conduct a comparative study
of representative existing approaches, showcasing the competitiveness of some
ACL algorithms that do not use expert knowledge. We also show that the Parkour
environment remains an open problem. We open-source our environments, all
studied ACL algorithms (collected from open-source code or re-implemented), and
DRL students in a Python package available at
https://github.com/flowersteam/TeachMyAgent.


---

**[90. [2009.05161] Multi-Goal Multi-Agent Path Finding via Decoupled and Integrated Goal
  Vertex Ordering](https://arxiv.org/pdf/2009.05161.pdf)** (2020-09-14)

*Pavel Surynek*

  We introduce multi-goal multi agent path finding (MAPF$^{MG}$) which
generalizes the standard discrete multi-agent path finding (MAPF) problem.
While the task in MAPF is to navigate agents in an undirected graph from their
starting vertices to one individual goal vertex per agent, MAPF$^{MG}$ assigns
each agent multiple goal vertices and the task is to visit each of them at
least once. Solving MAPF$^{MG}$ not only requires finding collision free paths
for individual agents but also determining the order of visiting agent's goal
vertices so that common objectives like the sum-of-costs are optimized. We
suggest two novel algorithms using different paradigms to address MAPF$^{MG}$:
a heuristic search-based search algorithm called Hamiltonian-CBS (HCBS) and a
compilation-based algorithm built using the SMT paradigm, called
SMT-Hamiltonian-CBS (SMT-HCBS). Experimental comparison suggests limitations of
compilation-based approach.


---

**[91. [2108.09996] MS-DARTS: Mean-Shift Based Differentiable Architecture Search](https://arxiv.org/pdf/2108.09996.pdf)** (2022-03-10)

*Jun-Wei Hsieh, Ming-Ching Chang, Ping-Yang Chen, Santanu Santra, Cheng-Han Chou, Chih-Sheng Huang*

  Differentiable Architecture Search (DARTS) is an effective continuous
relaxation-based network architecture search (NAS) method with low search cost.
It has attracted significant attentions in Auto-ML research and becomes one of
the most useful paradigms in NAS. Although DARTS can produce superior
efficiency over traditional NAS approaches with better control of complex
parameters, oftentimes it suffers from stabilization issues in producing
deteriorating architectures when discretizing the continuous architecture. We
observed considerable loss of validity causing dramatic decline in performance
at this final discretization step of DARTS. To address this issue, we propose a
Mean-Shift based DARTS (MS-DARTS) to improve stability based on sampling and
perturbation. Our approach can improve bot the stability and accuracy of DARTS,
by smoothing the loss landscape and sampling architecture parameters within a
suitable bandwidth. We investigate the convergence of our mean-shift approach,
together with the effects of bandwidth selection that affects stability and
accuracy. Evaluations performed on CIFAR-10, CIFAR-100, and ImageNet show that
MS-DARTS archives higher performance over other state-of-the-art NAS methods
with reduced search cost.


---

**[92. [2501.17167] QualityFlow: An Agentic Workflow for Program Synthesis Controlled by LLM
  Quality Checks](https://arxiv.org/pdf/2501.17167.pdf)** (2025-03-26)

*Yaojie Hu, Qiang Zhou, Qihong Chen, Xiaopeng Li, Linbo Liu, Dejiao Zhang, Amit Kachroo, Talha Oz, Omer Tripp*

  We introduce QualityFlow, a dynamic agentic workflow for program synthesis.
Given the English description of a programming problem and a set of unit tests,
the model's goal is to synthesize the correct program that solves the problem
and passes the tests. QualityFlow includes large language model (LLM) agents
resembling a software development team, including code generation, testing, and
self-debugging. We propose the LLM Quality Checker, which explicitly "imagines"
whether the synthesized programs' execution would conform to the unit tests.
The Quality Checks dynamically control the workflow, including actions to
submit the final answer, clarify the problem statement, and revert previous
workflow steps. Our experiments show that the Quality Checker can precisely
accept any correct program, mitigate faulty synthesized tests, and prevent
potential workflow deviation. QualityFlow establishes the state-of-the-art
results on four program synthesis benchmarks: MBPP, HumanEval, and stricter
evaluations from MBPP-EvalPlus and HumanEval-EvalPlus.


---

**[93. [2306.02910] Action-Evolution Petri Nets: a Framework for Modeling and Solving
  Dynamic Task Assignment Problems](https://arxiv.org/pdf/2306.02910.pdf)** (2023-06-12)

*Riccardo Lo Bianco, Remco Dijkman, Wim Nuijten, Willem van Jaarsveld*

  Dynamic task assignment involves assigning arriving tasks to a limited number
of resources in order to minimize the overall cost of the assignments. To
achieve optimal task assignment, it is necessary to model the assignment
problem first. While there exist separate formalisms, specifically Markov
Decision Processes and (Colored) Petri Nets, to model, execute, and solve
different aspects of the problem, there is no integrated modeling technique. To
address this gap, this paper proposes Action-Evolution Petri Nets (A-E PN) as a
framework for modeling and solving dynamic task assignment problems. A-E PN
provides a unified modeling technique that can represent all elements of
dynamic task assignment problems. Moreover, A-E PN models are executable, which
means they can be used to learn close-to-optimal assignment policies through
Reinforcement Learning (RL) without additional modeling effort. To evaluate the
framework, we define a taxonomy of archetypical assignment problems. We show
for three cases that A-E PN can be used to learn close-to-optimal assignment
policies. Our results suggest that A-E PN can be used to model and solve a
broad range of dynamic task assignment problems.


---

**[94. [2111.02354] Smooth Imitation Learning via Smooth Costs and Smooth Policies](https://arxiv.org/pdf/2111.02354.pdf)** (2021-11-04)

*Sapana Chaudhary, Balaraman Ravindran*

  Imitation learning (IL) is a popular approach in the continuous control
setting as among other reasons it circumvents the problems of reward
mis-specification and exploration in reinforcement learning (RL). In IL from
demonstrations, an important challenge is to obtain agent policies that are
smooth with respect to the inputs. Learning through imitation a policy that is
smooth as a function of a large state-action ($s$-$a$) space (typical of high
dimensional continuous control environments) can be challenging. We take a
first step towards tackling this issue by using smoothness inducing
regularizers on \textit{both} the policy and the cost models of adversarial
imitation learning. Our regularizers work by ensuring that the cost function
changes in a controlled manner as a function of $s$-$a$ space; and the agent
policy is well behaved with respect to the state space. We call our new smooth
IL algorithm \textit{Smooth Policy and Cost Imitation Learning} (SPaCIL,
pronounced 'Special'). We introduce a novel metric to quantify the smoothness
of the learned policies. We demonstrate SPaCIL's superior performance on
continuous control tasks from MuJoCo. The algorithm not just outperforms the
state-of-the-art IL algorithm on our proposed smoothness metric, but, enjoys
added benefits of faster learning and substantially higher average return.


---

**[95. [2404.15109] Compete and Compose: Learning Independent Mechanisms for Modular World
  Models](https://arxiv.org/pdf/2404.15109.pdf)** (2024-04-24)

*Anson Lei, Frederik Nolte, Bernhard Schölkopf, Ingmar Posner*

  We present COmpetitive Mechanisms for Efficient Transfer (COMET), a modular
world model which leverages reusable, independent mechanisms across different
environments. COMET is trained on multiple environments with varying dynamics
via a two-step process: competition and composition. This enables the model to
recognise and learn transferable mechanisms. Specifically, in the competition
phase, COMET is trained with a winner-takes-all gradient allocation,
encouraging the emergence of independent mechanisms. These are then re-used in
the composition phase, where COMET learns to re-compose learnt mechanisms in
ways that capture the dynamics of intervened environments. In so doing, COMET
explicitly reuses prior knowledge, enabling efficient and interpretable
adaptation. We evaluate COMET on environments with image-based observations. In
contrast to competitive baselines, we demonstrate that COMET captures
recognisable mechanisms without supervision. Moreover, we show that COMET is
able to adapt to new environments with varying numbers of objects with improved
sample efficiency compared to more conventional finetuning approaches.


---

**[96. [2312.01990] SARA-RT: Scaling up Robotics Transformers with Self-Adaptive Robust
  Attention](https://arxiv.org/pdf/2312.01990.pdf)** (2023-12-05)

*Isabel Leal, Krzysztof Choromanski, Deepali Jain, Avinava Dubey, Jake Varley, Michael Ryoo, Yao Lu, Frederick Liu, Vikas Sindhwani, Quan Vuong, Tamas Sarlos, Ken Oslund, Karol Hausman, Kanishka Rao*

  We present Self-Adaptive Robust Attention for Robotics Transformers
(SARA-RT): a new paradigm for addressing the emerging challenge of scaling up
Robotics Transformers (RT) for on-robot deployment. SARA-RT relies on the new
method of fine-tuning proposed by us, called up-training. It converts
pre-trained or already fine-tuned Transformer-based robotic policies of
quadratic time complexity (including massive billion-parameter
vision-language-action models or VLAs), into their efficient linear-attention
counterparts maintaining high quality. We demonstrate the effectiveness of
SARA-RT by speeding up: (a) the class of recently introduced RT-2 models, the
first VLA robotic policies pre-trained on internet-scale data, as well as (b)
Point Cloud Transformer (PCT) robotic policies operating on large point clouds.
We complement our results with the rigorous mathematical analysis providing
deeper insight into the phenomenon of SARA.


---

**[97. [2404.19542] One-Stage Open-Vocabulary Temporal Action Detection Leveraging Temporal
  Multi-scale and Action Label Features](https://arxiv.org/pdf/2404.19542.pdf)** (2024-05-01)

*Trung Thanh Nguyen, Yasutomo Kawanishi, Takahiro Komamizu, Ichiro Ide*

  Open-vocabulary Temporal Action Detection (Open-vocab TAD) is an advanced
video analysis approach that expands Closed-vocabulary Temporal Action
Detection (Closed-vocab TAD) capabilities. Closed-vocab TAD is typically
confined to localizing and classifying actions based on a predefined set of
categories. In contrast, Open-vocab TAD goes further and is not limited to
these predefined categories. This is particularly useful in real-world
scenarios where the variety of actions in videos can be vast and not always
predictable. The prevalent methods in Open-vocab TAD typically employ a 2-stage
approach, which involves generating action proposals and then identifying those
actions. However, errors made during the first stage can adversely affect the
subsequent action identification accuracy. Additionally, existing studies face
challenges in handling actions of different durations owing to the use of fixed
temporal processing methods. Therefore, we propose a 1-stage approach
consisting of two primary modules: Multi-scale Video Analysis (MVA) and
Video-Text Alignment (VTA). The MVA module captures actions at varying temporal
resolutions, overcoming the challenge of detecting actions with diverse
durations. The VTA module leverages the synergy between visual and textual
modalities to precisely align video segments with corresponding action labels,
a critical step for accurate action identification in Open-vocab scenarios.
Evaluations on widely recognized datasets THUMOS14 and ActivityNet-1.3, showed
that the proposed method achieved superior results compared to the other
methods in both Open-vocab and Closed-vocab settings. This serves as a strong
demonstration of the effectiveness of the proposed method in the TAD task.


---

**[98. [2412.00573] Opus: A Large Work Model for Complex Workflow Generation](https://arxiv.org/pdf/2412.00573.pdf)** (2024-12-09)

*Théo Fagnoni, Bellinda Mesbah, Mahsun Altin, Phillip Kingston*

  This paper introduces Opus, a novel framework for generating and optimizing
Workflows tailored to complex Business Process Outsourcing (BPO) use cases,
focusing on cost reduction and quality enhancement while adhering to
established industry processes and operational constraints. Our approach
generates executable Workflows from Intention, defined as the alignment of
Client Input, Client Output, and Process Context. These Workflows are
represented as Directed Acyclic Graphs (DAGs), with nodes as Tasks consisting
of sequences of executable Instructions, including tools and human expert
reviews. We adopt a two-phase methodology: Workflow Generation and Workflow
Optimization. In the Generation phase, Workflows are generated using a Large
Work Model (LWM) informed by a Work Knowledge Graph (WKG) that encodes
domain-specific procedural and operational knowledge. In the Optimization
phase, Workflows are transformed into Workflow Graphs (WFGs), where optimal
Workflows are determined through path optimization. Our experiments demonstrate
that state-of-the-art Large Language Models (LLMs) face challenges in reliably
retrieving detailed process data as well as generating industry-compliant
workflows. The key contributions of this paper include integrating a Work
Knowledge Graph (WKG) into a Large Work Model (LWM) to enable the generation of
context-aware, semantically aligned, structured and auditable Workflows. It
further introduces a two-phase approach that combines Workflow Generation from
Intention with graph-based Workflow Optimization. Finally, we present Opus
Alpha 1 Large and Opus Alpha 1 Small that outperform state-of-the-art LLMs by
38% and 29% respectively in Workflow Generation for a Medical Coding use case.


---

**[99. [2408.05006] COAST: Enhancing the Code Debugging Ability of LLMs through
  Communicative Agent Based Data Synthesis](https://arxiv.org/pdf/2408.05006.pdf)** (2025-02-13)

*Weiqing Yang, Hanbin Wang, Zhenghao Liu, Xinze Li, Yukun Yan, Shuo Wang, Yu Gu, Minghe Yu, Zhiyuan Liu, Ge Yu*

  Code debugging is a vital stage of software development, essential for
ensuring the reliability and performance of Large Language Models (LLMs) in the
code generation task. Human debugging typically follows a multi-stage process,
which includes Bug Localization, Bug Identification, Code Repair, and Code
Recognition. However, existing code debugging benchmarks predominantly focus on
the Code Repair stage, which offers only a limited perspective on evaluating
the debugging capabilities of LLMs. In this paper, we introduce DEBUGEVAL, a
comprehensive benchmark for evaluating the debugging abilities of LLMs by
emulating the multi-stage human debugging process. Through evaluating on
DEBUGEVAL, we observe that 7B-scale models consistently underperform compared
to their larger counterparts, highlighting their limitations in comprehending
code semantics. In this case, we propose the COmmunicative Agent-based data
SynThesis (COAST) framework, which employs a multi-agent system to generate
high-quality training data for supervised fine-tuning (SFT). Experimental
results demonstrate that COAST-generated data outperform human-curated and
GPT-4-generated data, enabling 7B-scale LLMs to achieve debugging performance
comparable to GPT-3.5. All data and codes are available at
https://github.com/NEUIR/COAST.


---

**[100. [2306.16884] Policy Space Diversity for Non-Transitive Games](https://arxiv.org/pdf/2306.16884.pdf)** (2023-11-09)

*Jian Yao, Weiming Liu, Haobo Fu, Yaodong Yang, Stephen McAleer, Qiang Fu, Wei Yang*

  Policy-Space Response Oracles (PSRO) is an influential algorithm framework
for approximating a Nash Equilibrium (NE) in multi-agent non-transitive games.
Many previous studies have been trying to promote policy diversity in PSRO. A
major weakness in existing diversity metrics is that a more diverse (according
to their diversity metrics) population does not necessarily mean (as we proved
in the paper) a better approximation to a NE. To alleviate this problem, we
propose a new diversity metric, the improvement of which guarantees a better
approximation to a NE. Meanwhile, we develop a practical and well-justified
method to optimize our diversity metric using only state-action samples. By
incorporating our diversity regularization into the best response solving in
PSRO, we obtain a new PSRO variant, Policy Space Diversity PSRO (PSD-PSRO). We
present the convergence property of PSD-PSRO. Empirically, extensive
experiments on various games demonstrate that PSD-PSRO is more effective in
producing significantly less exploitable policies than state-of-the-art PSRO
variants.


---

**[101. [2311.00334] MetisFL: An Embarrassingly Parallelized Controller for Scalable &
  Efficient Federated Learning Workflows](https://arxiv.org/pdf/2311.00334.pdf)** (2023-11-14)

*Dimitris Stripelis, Chrysovalantis Anastasiou, Patrick Toral, Armaghan Asghar, Jose Luis Ambite*

  A Federated Learning (FL) system typically consists of two core processing
entities: the federation controller and the learners. The controller is
responsible for managing the execution of FL workflows across learners and the
learners for training and evaluating federated models over their private
datasets. While executing an FL workflow, the FL system has no control over the
computational resources or data of the participating learners. Still, it is
responsible for other operations, such as model aggregation, task dispatching,
and scheduling. These computationally heavy operations generally need to be
handled by the federation controller. Even though many FL systems have been
recently proposed to facilitate the development of FL workflows, most of these
systems overlook the scalability of the controller. To meet this need, we
designed and developed a novel FL system called MetisFL, where the federation
controller is the first-class citizen. MetisFL re-engineers all the operations
conducted by the federation controller to accelerate the training of
large-scale FL workflows. By quantitatively comparing MetisFL against other
state-of-the-art FL systems, we empirically demonstrate that MetisFL leads to a
10-fold wall-clock time execution boost across a wide range of challenging FL
workflows with increasing model sizes and federation sites.


---

**[102. [2104.00660] Recognizing and Splitting Conditional Sentences for Automation of
  Business Processes Management](https://arxiv.org/pdf/2104.00660.pdf)** (2021-04-02)

*Ngoc Phuoc An Vo, Irene Manotas, Octavian Popescu, Algimantas Cerniauskas, Vadim Sheinin*

  Business Process Management (BPM) is the discipline which is responsible for
management of discovering, analyzing, redesigning, monitoring, and controlling
business processes. One of the most crucial tasks of BPM is discovering and
modelling business processes from text documents. In this paper, we present our
system that resolves an end-to-end problem consisting of 1) recognizing
conditional sentences from technical documents, 2) finding boundaries to
extract conditional and resultant clauses from each conditional sentence, and
3) categorizing resultant clause as Action or Consequence which later helps to
generate new steps in our business process model automatically. We created a
new dataset and three models solve this problem. Our best model achieved very
promising results of 83.82, 87.84, and 85.75 for Precision, Recall, and F1,
respectively, for extracting Condition, Action, and Consequence clauses using
Exact Match metric.


---

**[103. [2206.00649] Differentiable programming for functional connectomics](https://arxiv.org/pdf/2206.00649.pdf)** (2022-06-02)

*Department of Bioengineering, Stanford University  Rastko Ciric, Stanford Data Science, Stanford University  Armin W. Thomas, Department of Radiology, Université de Lausanne  Oscar Esteban, Department of Psychology, Stanford University  Russell A. Poldrack*

  Mapping the functional connectome has the potential to uncover key insights
into brain organisation. However, existing workflows for functional
connectomics are limited in their adaptability to new data, and principled
workflow design is a challenging combinatorial problem. We introduce a new
analytic paradigm and software toolbox that implements common operations used
in functional connectomics as fully differentiable processing blocks. Under
this paradigm, workflow configurations exist as reparameterisations of a
differentiable functional that interpolates them. The differentiable program
that we envision occupies a niche midway between traditional pipelines and
end-to-end neural networks, combining the glass-box tractability and domain
knowledge of the former with the amenability to optimisation of the latter. In
this preliminary work, we provide a proof of concept for differentiable
connectomics, demonstrating the capacity of our processing blocks both to
recapitulate canonical knowledge in neuroscience and to make new discoveries in
an unsupervised setting. Our differentiable modules are competitive with
state-of-the-art methods in problem domains including functional parcellation,
denoising, and covariance modelling. Taken together, our results and software
demonstrate the promise of differentiable programming for functional
connectomics.


---

**[104. [2203.14122] A Runtime Environment for Contract Automata](https://arxiv.org/pdf/2203.14122.pdf)** (2023-03-16)

*Davide Basile, Maurice H. ter Beek*

  Contract automata have been introduced for specifying applications through
behavioural contracts and for synthesising their orchestrations as finite state
automata. This paper addresses the realisation of applications from contract
automata specifications. We present CARE, a new runtime environment to
coordinate services implementing contracts that guarantees the adherence of the
implementation to its contract. We discuss how CARE can be adopted to realise
contract-based applications, its formal guarantees, and we identify the
responsibilities of the involved business actors. Experiments show the benefits
of adopting CARE with respect to manual implementations.


---

**[105. [2201.09305] An Analysis and Comparison of ACT-R and Soar](https://arxiv.org/pdf/2201.09305.pdf)** (2022-01-25)

*John E. Laird*

  This is a detailed analysis and comparison of the ACT-R and Soar cognitive
architectures, including their overall structure, their representations of
agent data and metadata, and their associated processing. It focuses on working
memory, procedural memory, and long-term declarative memory. I emphasize the
commonalities, which are many, but also highlight the differences. I identify
the processes and distinct classes of information used by these architectures,
including agent data, metadata, and meta-process data, and explore the roles
that metadata play in decision making, memory retrievals, and learning.


---

**[106. [2302.09639] An overview of differentiable particle filters for data-adaptive
  sequential Bayesian inference](https://arxiv.org/pdf/2302.09639.pdf)** (2023-12-15)

*Xiongjie Chen, Yunpeng Li*

  By approximating posterior distributions with weighted samples, particle
filters (PFs) provide an efficient mechanism for solving non-linear sequential
state estimation problems. While the effectiveness of particle filters has been
recognised in various applications, their performance relies on the knowledge
of dynamic models and measurement models, as well as the construction of
effective proposal distributions. An emerging trend involves constructing
components of particle filters using neural networks and optimising them by
gradient descent, and such data-adaptive particle filtering approaches are
often called differentiable particle filters. Due to the expressiveness of
neural networks, differentiable particle filters are a promising computational
tool for performing inference on sequential data in complex, high-dimensional
tasks, such as vision-based robot localisation. In this paper, we review recent
advances in differentiable particle filters and their applications. We place
special emphasis on different design choices for key components of
differentiable particle filters, including dynamic models, measurement models,
proposal distributions, optimisation objectives, and differentiable resampling
techniques.


---

**[107. [2202.02872] Differentiable Economics for Randomized Affine Maximizer Auctions](https://arxiv.org/pdf/2202.02872.pdf)** (2022-02-08)

*Michael Curry, Tuomas Sandholm, John Dickerson*

  A recent approach to automated mechanism design, differentiable economics,
represents auctions by rich function approximators and optimizes their
performance by gradient descent. The ideal auction architecture for
differentiable economics would be perfectly strategyproof, support multiple
bidders and items, and be rich enough to represent the optimal (i.e.
revenue-maximizing) mechanism. So far, such an architecture does not exist.
There are single-bidder approaches (MenuNet, RochetNet) which are always
strategyproof and can represent optimal mechanisms. RegretNet is multi-bidder
and can approximate any mechanism, but is only approximately strategyproof. We
present an architecture that supports multiple bidders and is perfectly
strategyproof, but cannot necessarily represent the optimal mechanism. This
architecture is the classic affine maximizer auction (AMA), modified to offer
lotteries. By using the gradient-based optimization tools of differentiable
economics, we can now train lottery AMAs, competing with or outperforming prior
approaches in revenue.


---

**[108. [2107.04619] Diverse Video Generation using a Gaussian Process Trigger](https://arxiv.org/pdf/2107.04619.pdf)** (2021-07-13)

*Gaurav Shrivastava, Abhinav Shrivastava*

  Generating future frames given a few context (or past) frames is a
challenging task. It requires modeling the temporal coherence of videos and
multi-modality in terms of diversity in the potential future states. Current
variational approaches for video generation tend to marginalize over
multi-modal future outcomes. Instead, we propose to explicitly model the
multi-modality in the future outcomes and leverage it to sample diverse
futures. Our approach, Diverse Video Generator, uses a Gaussian Process (GP) to
learn priors on future states given the past and maintains a probability
distribution over possible futures given a particular sample. In addition, we
leverage the changes in this distribution over time to control the sampling of
diverse future states by estimating the end of ongoing sequences. That is, we
use the variance of GP over the output function space to trigger a change in an
action sequence. We achieve state-of-the-art results on diverse future frame
generation in terms of reconstruction quality and diversity of the generated
sequences.


---

**[109. [2308.10806] DFWLayer: Differentiable Frank-Wolfe Optimization Layer](https://arxiv.org/pdf/2308.10806.pdf)** (2024-04-01)

*Zixuan Liu, Liu Liu, Xueqian Wang, Peilin Zhao*

  Differentiable optimization has received a significant amount of attention
due to its foundational role in the domain of machine learning based on neural
networks. This paper proposes a differentiable layer, named Differentiable
Frank-Wolfe Layer (DFWLayer), by rolling out the Frank-Wolfe method, a
well-known optimization algorithm which can solve constrained optimization
problems without projections and Hessian matrix computations, thus leading to
an efficient way of dealing with large-scale convex optimization problems with
norm constraints. Experimental results demonstrate that the DFWLayer not only
attains competitive accuracy in solutions and gradients but also consistently
adheres to constraints.


---

**[110. [2206.04199] Deep Surrogate Assisted Generation of Environments](https://arxiv.org/pdf/2206.04199.pdf)** (2022-10-13)

*Varun Bhatt, Bryon Tjanaka, Matthew C. Fontaine, Stefanos Nikolaidis*

  Recent progress in reinforcement learning (RL) has started producing
generally capable agents that can solve a distribution of complex environments.
These agents are typically tested on fixed, human-authored environments. On the
other hand, quality diversity (QD) optimization has been proven to be an
effective component of environment generation algorithms, which can generate
collections of high-quality environments that are diverse in the resulting
agent behaviors. However, these algorithms require potentially expensive
simulations of agents on newly generated environments. We propose Deep
Surrogate Assisted Generation of Environments (DSAGE), a sample-efficient QD
environment generation algorithm that maintains a deep surrogate model for
predicting agent behaviors in new environments. Results in two benchmark
domains show that DSAGE significantly outperforms existing QD environment
generation algorithms in discovering collections of environments that elicit
diverse behaviors of a state-of-the-art RL agent and a planning agent. Our
source code and videos are available at https://dsagepaper.github.io/.


---

**[111. [2404.13244] Intelligent Agents for Auction-based Federated Learning: A Survey](https://arxiv.org/pdf/2404.13244.pdf)** (2024-04-23)

*Xiaoli Tang, Han Yu, Xiaoxiao Li, Sarit Kraus*

  Auction-based federated learning (AFL) is an important emerging category of
FL incentive mechanism design, due to its ability to fairly and efficiently
motivate high-quality data owners to join data consumers' (i.e., servers') FL
training tasks. To enhance the efficiency in AFL decision support for
stakeholders (i.e., data consumers, data owners, and the auctioneer),
intelligent agent-based techniques have emerged. However, due to the highly
interdisciplinary nature of this field and the lack of a comprehensive survey
providing an accessible perspective, it is a challenge for researchers to enter
and contribute to this field. This paper bridges this important gap by
providing a first-of-its-kind survey on the Intelligent Agents for AFL (IA-AFL)
literature. We propose a unique multi-tiered taxonomy that organises existing
IA-AFL works according to 1) the stakeholders served, 2) the auction mechanism
adopted, and 3) the goals of the agents, to provide readers with a
multi-perspective view into this field. In addition, we analyse the limitations
of existing approaches, summarise the commonly adopted performance evaluation
metrics, and discuss promising future directions leading towards effective and
efficient stakeholder-oriented decision support in IA-AFL ecosystems.


---

**[112. [2407.20287] Variational Inference Using Material Point Method](https://arxiv.org/pdf/2407.20287.pdf)** (2024-07-31)

*Yongchao Huang*

  A new gradient-based particle sampling method, MPM-ParVI, based on material
point method (MPM), is proposed for variational inference. MPM-ParVI simulates
the deformation of a deformable body (e.g. a solid or fluid) under external
effects driven by the target density; transient or steady configuration of the
deformable body approximates the target density. The continuum material is
modelled as an interacting particle system (IPS) using MPM, each particle
carries full physical properties, interacts and evolves following conservation
dynamics. This easy-to-implement ParVI method offers deterministic sampling and
inference for a class of probabilistic models such as those encountered in
Bayesian inference (e.g. intractable densities) and generative modelling (e.g.
score-based).


---

**[113. [2410.06151] Quality Diversity Imitation Learning](https://arxiv.org/pdf/2410.06151.pdf)** (2024-10-10)

*Zhenglin Wan, Xingrui Yu, David Mark Bossens, Yueming Lyu, Qing Guo, Flint Xiaofeng Fan, Ivor Tsang*

  Imitation learning (IL) has shown great potential in various applications,
such as robot control. However, traditional IL methods are usually designed to
learn only one specific type of behavior since demonstrations typically
correspond to a single expert. In this work, we introduce the first generic
framework for Quality Diversity Imitation Learning (QD-IL), which enables the
agent to learn a broad range of skills from limited demonstrations. Our
framework integrates the principles of quality diversity with adversarial
imitation learning (AIL) methods, and can potentially improve any inverse
reinforcement learning (IRL) method. Empirically, our framework significantly
improves the QD performance of GAIL and VAIL on the challenging continuous
control tasks derived from Mujoco environments. Moreover, our method even
achieves 2x expert performance in the most challenging Humanoid environment.


---

**[114. [1912.08168] Differentiable programming and its applications to dynamical systems](https://arxiv.org/pdf/1912.08168.pdf)** (2020-05-05)

*Adrián Hernández, José M. Amigó*

  Differentiable programming is the combination of classical neural networks
modules with algorithmic ones in an end-to-end differentiable model. These new
models, that use automatic differentiation to calculate gradients, have new
learning capabilities (reasoning, attention and memory). In this tutorial,
aimed at researchers in nonlinear systems with prior knowledge of deep
learning, we present this new programming paradigm, describe some of its new
features such as attention mechanisms, and highlight the benefits they bring.
Then, we analyse the uses and limitations of traditional deep learning models
in the modeling and prediction of dynamical systems. Here, a dynamical system
is meant to be a set of state variables that evolve in time under general
internal and external interactions. Finally, we review the advantages and
applications of differentiable programming to dynamical systems.


---

**[115. [2007.05352] Multi-Emitter MAP-Elites: Improving quality, diversity and convergence
  speed with heterogeneous sets of emitters](https://arxiv.org/pdf/2007.05352.pdf)** (2021-07-07)

*Antoine Cully*

  Quality-Diversity (QD) optimisation is a new family of learning algorithms
that aims at generating collections of diverse and high-performing solutions.
Among those algorithms, the recently introduced Covariance Matrix Adaptation
MAP-Elites (CMA-ME) algorithm proposes the concept of emitters, which uses a
predefined heuristic to drive the algorithm's exploration. This algorithm was
shown to outperform MAP-Elites, a popular QD algorithm that has demonstrated
promising results in numerous applications. In this paper, we introduce
Multi-Emitter MAP-Elites (ME-MAP-Elites), an algorithm that directly extends
CMA-ME and improves its quality, diversity and data efficiency. It leverages
the diversity of a heterogeneous set of emitters, in which each emitter type
improves the optimisation process in different ways. A bandit algorithm
dynamically finds the best selection of emitters depending on the current
situation. We evaluate the performance of ME-MAP-Elites on six tasks, ranging
from standard optimisation problems (in 100 dimensions) to complex locomotion
tasks in robotics. Our comparisons against CMA-ME and MAP-Elites show that
ME-MAP-Elites is faster at providing collections of solutions that are
significantly more diverse and higher performing. Moreover, in cases where no
fruitful synergy can be found between the different emitters, ME-MAP-Elites is
equivalent to the best of the compared algorithms.


---

**[116. [2406.04843] Variational Flow Matching for Graph Generation](https://arxiv.org/pdf/2406.04843.pdf)** (2024-06-10)

*Floor Eijkelboom, Grigory Bartosh, Christian Andersson Naesseth, Max Welling, Jan-Willem van de Meent*

  We present a formulation of flow matching as variational inference, which we
refer to as variational flow matching (VFM). Based on this formulation we
develop CatFlow, a flow matching method for categorical data. CatFlow is easy
to implement, computationally efficient, and achieves strong results on graph
generation tasks. In VFM, the objective is to approximate the posterior
probability path, which is a distribution over possible end points of a
trajectory. We show that VFM admits both the CatFlow objective and the original
flow matching objective as special cases. We also relate VFM to score-based
models, in which the dynamics are stochastic rather than deterministic, and
derive a bound on the model likelihood based on a reweighted VFM objective. We
evaluate CatFlow on one abstract graph generation task and two molecular
generation tasks. In all cases, CatFlow exceeds or matches performance of the
current state-of-the-art models.


---

**[117. [2503.23138] EncGPT: A Multi-Agent Workflow for Dynamic Encryption Algorithms](https://arxiv.org/pdf/2503.23138.pdf)** (2025-04-01)

*Donghe Li, Zuchen Li, Ye Yang, Li Sun, Dou An, Qingyu Yang*

  Communication encryption is crucial in computer technology, but existing
algorithms struggle with balancing cost and security. We propose EncGPT, a
multi-agent framework using large language models (LLM). It includes rule,
encryption, and decryption agents that generate encryption rules and apply them
dynamically. This approach addresses gaps in LLM-based multi-agent systems for
communication security. We tested GPT-4o's rule generation and implemented a
substitution encryption workflow with homomorphism preservation, achieving an
average execution time of 15.99 seconds.


---

**[118. [2202.05650] Bernstein Flows for Flexible Posteriors in Variational Bayes](https://arxiv.org/pdf/2202.05650.pdf)** (2024-02-26)

*Oliver Dürr, Stephan Hörling, Daniel Dold, Ivonne Kovylov, Beate Sick*

  Variational inference (VI) is a technique to approximate difficult to compute
posteriors by optimization. In contrast to MCMC, VI scales to many
observations. In the case of complex posteriors, however, state-of-the-art VI
approaches often yield unsatisfactory posterior approximations. This paper
presents Bernstein flow variational inference (BF-VI), a robust and easy-to-use
method, flexible enough to approximate complex multivariate posteriors. BF-VI
combines ideas from normalizing flows and Bernstein polynomial-based
transformation models. In benchmark experiments, we compare BF-VI solutions
with exact posteriors, MCMC solutions, and state-of-the-art VI methods
including normalizing flow based VI. We show for low-dimensional models that
BF-VI accurately approximates the true posterior; in higher-dimensional models,
BF-VI outperforms other VI methods. Further, we develop with BF-VI a Bayesian
model for the semi-structured Melanoma challenge data, combining a CNN model
part for image data with an interpretable model part for tabular data, and
demonstrate for the first time how the use of VI in semi-structured models.


---

**[119. [2012.11903] Modelling Human Routines: Conceptualising Social Practice Theory for
  Agent-Based Simulation](https://arxiv.org/pdf/2012.11903.pdf)** (2020-12-23)

*Rijk Mercuur, Virginia Dignum, Catholijn M. Jonker*

  Our routines play an important role in a wide range of social challenges such
as climate change, disease outbreaks and coordinating staff and patients in a
hospital. To use agent-based simulations (ABS) to understand the role of
routines in social challenges we need an agent framework that integrates
routines. This paper provides the domain-independent Social Practice Agent
(SoPrA) framework that satisfies requirements from the literature to simulate
our routines. By choosing the appropriate concepts from the literature on agent
theory, social psychology and social practice theory we ensure SoPrA correctly
depicts current evidence on routines. By creating a consistent, modular and
parsimonious framework suitable for multiple domains we enhance the usability
of SoPrA. SoPrA provides ABS researchers with a conceptual, formal and
computational framework to simulate routines and gain new insights into social
systems.


---

**[120. [2502.00022] A Dynamic and High-Precision Method for Scenario-Based HRA Synthetic
  Data Collection in Multi-Agent Collaborative Environments Driven by LLMs](https://arxiv.org/pdf/2502.00022.pdf)** (2025-02-04)

*Xingyu Xiao, Peng Chen, Qianqian Jia, Jiejuan Tong, Jingang Liang, Haitao Wang*

  HRA (Human Reliability Analysis) data is crucial for advancing HRA
methodologies. however, existing data collection methods lack the necessary
granularity, and most approaches fail to capture dynamic features.
Additionally, many methods require expert knowledge as input, making them
time-consuming and labor-intensive. To address these challenges, we propose a
new paradigm for the automated collection of HRA data. Our approach focuses on
key indicators behind human error, specifically measuring workload in
collaborative settings. This study introduces a novel, scenario-driven method
for workload estimation, leveraging fine-tuned large language models (LLMs). By
training LLMs on real-world operational data from high-temperature gas-cooled
reactors (HTGRs), we simulate human behavior and cognitive load in real time
across various collaborative scenarios. The method dynamically adapts to
changes in operator workload, providing more accurate, flexible, and scalable
workload estimates. The results demonstrate that the proposed WELLA (Workload
Estimation with LLMs and Agents) outperforms existing commercial LLM-based
methods in terms of prediction accuracy.


---

**[121. [2306.10915] Practical Equivariances via Relational Conditional Neural Processes](https://arxiv.org/pdf/2306.10915.pdf)** (2023-11-07)

*Daolang Huang, Manuel Haussmann, Ulpu Remes, ST John, Grégoire Clarté, Kevin Sebastian Luck, Samuel Kaski, Luigi Acerbi*

  Conditional Neural Processes (CNPs) are a class of metalearning models
popular for combining the runtime efficiency of amortized inference with
reliable uncertainty quantification. Many relevant machine learning tasks, such
as in spatio-temporal modeling, Bayesian Optimization and continuous control,
inherently contain equivariances -- for example to translation -- which the
model can exploit for maximal performance. However, prior attempts to include
equivariances in CNPs do not scale effectively beyond two input dimensions. In
this work, we propose Relational Conditional Neural Processes (RCNPs), an
effective approach to incorporate equivariances into any neural process model.
Our proposed method extends the applicability and impact of equivariant neural
processes to higher dimensions. We empirically demonstrate the competitive
performance of RCNPs on a large array of tasks naturally containing
equivariances.


---

**[122. [2403.07041] Ant Colony Sampling with GFlowNets for Combinatorial Optimization](https://arxiv.org/pdf/2403.07041.pdf)** (2025-03-03)

*Minsu Kim, Sanghyeok Choi, Hyeonah Kim, Jiwoo Son, Jinkyoo Park, Yoshua Bengio*

  We present the Generative Flow Ant Colony Sampler (GFACS), a novel
meta-heuristic method that hierarchically combines amortized inference and
parallel stochastic search. Our method first leverages Generative Flow Networks
(GFlowNets) to amortize a \emph{multi-modal} prior distribution over
combinatorial solution space that encompasses both high-reward and diversified
solutions. This prior is iteratively updated via parallel stochastic search in
the spirit of Ant Colony Optimization (ACO), leading to the posterior
distribution that generates near-optimal solutions. Extensive experiments
across seven combinatorial optimization problems demonstrate GFACS's promising
performances.


---

**[123. [2502.00406] ALU: Agentic LLM Unlearning](https://arxiv.org/pdf/2502.00406.pdf)** (2025-02-04)

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

**[124. [2305.10952] Actor-Critic Methods using Physics-Informed Neural Networks: Control of
  a 1D PDE Model for Fluid-Cooled Battery Packs](https://arxiv.org/pdf/2305.10952.pdf)** (2023-05-19)

*Amartya Mukherjee, Jun Liu*

  This paper proposes an actor-critic algorithm for controlling the temperature
of a battery pack using a cooling fluid. This is modeled by a coupled 1D
partial differential equation (PDE) with a controlled advection term that
determines the speed of the cooling fluid. The Hamilton-Jacobi-Bellman (HJB)
equation is a PDE that evaluates the optimality of the value function and
determines an optimal controller. We propose an algorithm that treats the value
network as a Physics-Informed Neural Network (PINN) to solve for the
continuous-time HJB equation rather than a discrete-time Bellman optimality
equation, and we derive an optimal controller for the environment that we
exploit to achieve optimal control. Our experiments show that a hybrid-policy
method that updates the value network using the HJB equation and updates the
policy network identically to PPO achieves the best results in the control of
this PDE system.


---

**[125. [2102.07850] Differentiable Particle Filtering via Entropy-Regularized Optimal
  Transport](https://arxiv.org/pdf/2102.07850.pdf)** (2021-07-01)

*Adrien Corenflos, James Thornton, George Deligiannidis, Arnaud Doucet*

  Particle Filtering (PF) methods are an established class of procedures for
performing inference in non-linear state-space models. Resampling is a key
ingredient of PF, necessary to obtain low variance likelihood and states
estimates. However, traditional resampling methods result in PF-based loss
functions being non-differentiable with respect to model and PF parameters. In
a variational inference context, resampling also yields high variance gradient
estimates of the PF-based evidence lower bound. By leveraging optimal transport
ideas, we introduce a principled differentiable particle filter and provide
convergence results. We demonstrate this novel method on a variety of
applications.


---

**[126. [2409.01392] ComfyBench: Benchmarking LLM-based Agents in ComfyUI for Autonomously
  Designing Collaborative AI Systems](https://arxiv.org/pdf/2409.01392.pdf)** (2024-11-28)

*Xiangyuan Xue, Zeyu Lu, Di Huang, Zidong Wang, Wanli Ouyang, Lei Bai*

  Much previous AI research has focused on developing monolithic models to
maximize their intelligence, with the primary goal of enhancing performance on
specific tasks. In contrast, this work attempts to study using LLM-based agents
to design collaborative AI systems autonomously. To explore this problem, we
first introduce ComfyBench to evaluate agents's ability to design collaborative
AI systems in ComfyUI. ComfyBench is a comprehensive benchmark comprising 200
diverse tasks covering various instruction-following generation challenges,
along with detailed annotations for 3,205 nodes and 20 workflows. Based on
ComfyBench, we further develop ComfyAgent, a novel framework that empowers
LLM-based agents to autonomously design collaborative AI systems by generating
workflows. ComfyAgent is based on two core concepts. First, it represents
workflows with code, which can be reversibly converted into workflows and
executed as collaborative systems by the interpreter. Second, it constructs a
multi-agent system that cooperates to learn from existing workflows and
generate new workflows for a given task. While experimental results demonstrate
that ComfyAgent achieves a comparable resolve rate to o1-preview and
significantly surpasses other agents on ComfyBench, ComfyAgent has resolved
only 15\% of creative tasks. LLM-based agents still have a long way to go in
autonomously designing collaborative AI systems. Progress with ComfyBench is
paving the way for more intelligent and autonomous collaborative AI systems.


---

**[127. [2302.02714] Differentiable Programming of Chemical Reaction Networks](https://arxiv.org/pdf/2302.02714.pdf)** (2023-02-07)

*Alexander Mordvintsev, Ettore Randazzo, Eyvind Niklasson*

  We present a differentiable formulation of abstract chemical reaction
networks (CRNs) that can be trained to solve a variety of computational tasks.
Chemical reaction networks are one of the most fundamental computational
substrates used by nature. We study well-mixed single-chamber systems, as well
as systems with multiple chambers separated by membranes, under mass-action
kinetics. We demonstrate that differentiable optimisation, combined with proper
regularisation, can discover non-trivial sparse reaction networks that can
implement various sorts of oscillators and other chemical computing devices.


---

**[128. [2406.06910] Agent-SiMT: Agent-assisted Simultaneous Machine Translation with Large
  Language Models](https://arxiv.org/pdf/2406.06910.pdf)** (2024-06-13)

*Shoutao Guo, Shaolei Zhang, Zhengrui Ma, Min Zhang, Yang Feng*

  Simultaneous Machine Translation (SiMT) generates target translations while
reading the source sentence. It relies on a policy to determine the optimal
timing for reading sentences and generating translations. Existing SiMT methods
generally adopt the traditional Transformer architecture, which concurrently
determines the policy and generates translations. While they excel at
determining policies, their translation performance is suboptimal. Conversely,
Large Language Models (LLMs), trained on extensive corpora, possess superior
generation capabilities, but it is difficult for them to acquire translation
policy through the training methods of SiMT. Therefore, we introduce
Agent-SiMT, a framework combining the strengths of LLMs and traditional SiMT
methods. Agent-SiMT contains the policy-decision agent and the translation
agent. The policy-decision agent is managed by a SiMT model, which determines
the translation policy using partial source sentence and translation. The
translation agent, leveraging an LLM, generates translation based on the
partial source sentence. The two agents collaborate to accomplish SiMT.
Experiments demonstrate that Agent-SiMT attains state-of-the-art performance.


---

**[129. [2208.14446] You Only Search Once: On Lightweight Differentiable Architecture Search
  for Resource-Constrained Embedded Platforms](https://arxiv.org/pdf/2208.14446.pdf)** (2022-09-01)

*Xiangzhong Luo, Di Liu, Hao Kong, Shuo Huai, Hui Chen, Weichen Liu*

  Benefiting from the search efficiency, differentiable neural architecture
search (NAS) has evolved as the most dominant alternative to automatically
design competitive deep neural networks (DNNs). We note that DNNs must be
executed under strictly hard performance constraints in real-world scenarios,
for example, the runtime latency on autonomous vehicles. However, to obtain the
architecture that meets the given performance constraint, previous
hardware-aware differentiable NAS methods have to repeat a plethora of search
runs to manually tune the hyper-parameters by trial and error, and thus the
total design cost increases proportionally. To resolve this, we introduce a
lightweight hardware-aware differentiable NAS framework dubbed LightNAS,
striving to find the required architecture that satisfies various performance
constraints through a one-time search (i.e., \underline{\textit{you only search
once}}). Extensive experiments are conducted to show the superiority of
LightNAS over previous state-of-the-art methods.


---

**[130. [2404.04285] MIMIR: A Streamlined Platform for Personalized Agent Tuning in Domain
  Expertise](https://arxiv.org/pdf/2404.04285.pdf)** (2024-04-09)

*Chunyuan Deng, Xiangru Tang, Yilun Zhao, Hanming Wang, Haoran Wang, Wangchunshu Zhou, Arman Cohan, Mark Gerstein*

  Recently, large language models (LLMs) have evolved into interactive agents,
proficient in planning, tool use, and task execution across a wide variety of
tasks. However, without specific agent tuning, open-source models like LLaMA
currently struggle to match the efficiency of GPT- 4, particularly given the
scarcity of agent-tuning datasets for fine-tuning. In response, we introduce
\textsc{Mimir}: a streamlined platform offering a customizable pipeline that
enables users to leverage both private knowledge and publicly available,
legally compliant datasets at scale for \textbf{personalized agent tuning}.
Additionally, \textsc{Mimir} supports the generation of general
instruction-tuning datasets from the same input. This dual capability ensures
that language agents developed through the platform possess both specific agent
abilities and general competencies. \textsc{Mimir} integrates these features
into a cohesive end-to-end platform, facilitating everything from the uploading
of personalized files to one-click agent fine-tuning.


---

**[131. [2207.03945] High Performance Simulation for Scalable Multi-Agent Reinforcement
  Learning](https://arxiv.org/pdf/2207.03945.pdf)** (2022-07-11)

*Jordan Langham-Lopez, Sebastian M. Schmon, Patrick Cannon*

  Multi-agent reinforcement learning experiments and open-source training
environments are typically limited in scale, supporting tens or sometimes up to
hundreds of interacting agents. In this paper we demonstrate the use of Vogue,
a high performance agent based model (ABM) framework. Vogue serves as a
multi-agent training environment, supporting thousands to tens of thousands of
interacting agents while maintaining high training throughput by running both
the environment and reinforcement learning (RL) agents on the GPU. High
performance multi-agent environments at this scale have the potential to enable
the learning of robust and flexible policies for use in ABMs and simulations of
complex systems. We demonstrate training performance with two newly developed,
large scale multi-agent training environments. Moreover, we show that these
environments can train shared RL policies on time-scales of minutes and hours.


---

**[132. [2404.12611] Rethinking Clothes Changing Person ReID: Conflicts, Synthesis, and
  Optimization](https://arxiv.org/pdf/2404.12611.pdf)** (2024-04-22)

*Junjie Li, Guanshuo Wang, Fufu Yu, Yichao Yan, Qiong Jia, Shouhong Ding, Xingdong Sheng, Yunhui Liu, Xiaokang Yang*

  Clothes-changing person re-identification (CC-ReID) aims to retrieve images
of the same person wearing different outfits. Mainstream researches focus on
designing advanced model structures and strategies to capture identity
information independent of clothing. However, the same-clothes discrimination
as the standard ReID learning objective in CC-ReID is persistently ignored in
previous researches. In this study, we dive into the relationship between
standard and clothes-changing~(CC) learning objectives, and bring the inner
conflicts between these two objectives to the fore. We try to magnify the
proportion of CC training pairs by supplementing high-fidelity clothes-varying
synthesis, produced by our proposed Clothes-Changing Diffusion model. By
incorporating the synthetic images into CC-ReID model training, we observe a
significant improvement under CC protocol. However, such improvement sacrifices
the performance under the standard protocol, caused by the inner conflict
between standard and CC. For conflict mitigation, we decouple these objectives
and re-formulate CC-ReID learning as a multi-objective optimization (MOO)
problem. By effectively regularizing the gradient curvature across multiple
objectives and introducing preference restrictions, our MOO solution surpasses
the single-task training paradigm. Our framework is model-agnostic, and
demonstrates superior performance under both CC and standard ReID protocols.


---

**[133. [2404.05769] Dynamic Quality-Diversity Search](https://arxiv.org/pdf/2404.05769.pdf)** (2024-04-10)

*Roberto Gallotta, Antonios Liapis, Georgios N. Yannakakis*

  Evolutionary search via the quality-diversity (QD) paradigm can discover
highly performing solutions in different behavioural niches, showing
considerable potential in complex real-world scenarios such as evolutionary
robotics. Yet most QD methods only tackle static tasks that are fixed over
time, which is rarely the case in the real world. Unlike noisy environments,
where the fitness of an individual changes slightly at every evaluation,
dynamic environments simulate tasks where external factors at unknown and
irregular intervals alter the performance of the individual with a severity
that is unknown a priori. Literature on optimisation in dynamic environments is
extensive, yet such environments have not been explored in the context of QD
search. This paper introduces a novel and generalisable Dynamic QD methodology
that aims to keep the archive of past solutions updated in the case of
environment changes. Secondly, we present a novel characterisation of dynamic
environments that can be easily applied to well-known benchmarks, with minor
interventions to move them from a static task to a dynamic one. Our Dynamic QD
intervention is applied on MAP-Elites and CMA-ME, two powerful QD algorithms,
and we test the dynamic variants on different dynamic tasks.


---

**[134. [2306.00751] Differentiable Tree Operations Promote Compositional Generalization](https://arxiv.org/pdf/2306.00751.pdf)** (2023-06-02)

*Paul Soulos, Edward Hu, Kate McCurdy, Yunmo Chen, Roland Fernandez, Paul Smolensky, Jianfeng Gao*

  In the context of structure-to-structure transformation tasks, learning
sequences of discrete symbolic operations poses significant challenges due to
their non-differentiability. To facilitate the learning of these symbolic
sequences, we introduce a differentiable tree interpreter that compiles
high-level symbolic tree operations into subsymbolic matrix operations on
tensors. We present a novel Differentiable Tree Machine (DTM) architecture that
integrates our interpreter with an external memory and an agent that learns to
sequentially select tree operations to execute the target transformation in an
end-to-end manner. With respect to out-of-distribution compositional
generalization on synthetic semantic parsing and language generation tasks, DTM
achieves 100% while existing baselines such as Transformer, Tree Transformer,
LSTM, and Tree2Tree LSTM achieve less than 30%. DTM remains highly
interpretable in addition to its perfect performance.


---

**[135. [2309.02671] RLSynC: Offline-Online Reinforcement Learning for Synthon Completion](https://arxiv.org/pdf/2309.02671.pdf)** (2024-04-01)

*Frazier N. Baker, Ziqi Chen, Daniel Adu-Ampratwum, Xia Ning*

  Retrosynthesis is the process of determining the set of reactant molecules
that can react to form a desired product. Semi-template-based retrosynthesis
methods, which imitate the reverse logic of synthesis reactions, first predict
the reaction centers in the products, and then complete the resulting synthons
back into reactants. We develop a new offline-online reinforcement learning
method RLSynC for synthon completion in semi-template-based methods. RLSynC
assigns one agent to each synthon, all of which complete the synthons by
conducting actions step by step in a synchronized fashion. RLSynC learns the
policy from both offline training episodes and online interactions, which
allows RLSynC to explore new reaction spaces. RLSynC uses a standalone forward
synthesis model to evaluate the likelihood of the predicted reactants in
synthesizing a product, and thus guides the action search. Our results
demonstrate that RLSynC can outperform state-of-the-art synthon completion
methods with improvements as high as 14.9%, highlighting its potential in
synthesis planning.


---

**[136. [2503.07675] DynTaskMAS: A Dynamic Task Graph-driven Framework for Asynchronous and
  Parallel LLM-based Multi-Agent Systems](https://arxiv.org/pdf/2503.07675.pdf)** (2025-03-12)

*Junwei Yu, Yepeng Ding, Hiroyuki Sato*

  The emergence of Large Language Models (LLMs) in Multi-Agent Systems (MAS)
has opened new possibilities for artificial intelligence, yet current
implementations face significant challenges in resource management, task
coordination, and system efficiency. While existing frameworks demonstrate the
potential of LLM-based agents in collaborative problem-solving, they often lack
sophisticated mechanisms for parallel execution and dynamic task management.
This paper introduces DynTaskMAS, a novel framework that orchestrates
asynchronous and parallel operations in LLM-based MAS through dynamic task
graphs. The framework features four key innovations: (1) a Dynamic Task Graph
Generator that intelligently decomposes complex tasks while maintaining logical
dependencies, (2) an Asynchronous Parallel Execution Engine that optimizes
resource utilization through efficient task scheduling, (3) a Semantic-Aware
Context Management System that enables efficient information sharing among
agents, and (4) an Adaptive Workflow Manager that dynamically optimizes system
performance. Experimental evaluations demonstrate that DynTaskMAS achieves
significant improvements over traditional approaches: a 21-33% reduction in
execution time across task complexities (with higher gains for more complex
tasks), a 35.4% improvement in resource utilization (from 65% to 88%), and
near-linear throughput scaling up to 16 concurrent agents (3.47X improvement
for 4X agents). Our framework establishes a foundation for building scalable,
high-performance LLM-based multi-agent systems capable of handling complex,
dynamic tasks efficiently.


---

**[137. [2503.14269] DARS: Dynamic Action Re-Sampling to Enhance Coding Agent Performance by
  Adaptive Tree Traversal](https://arxiv.org/pdf/2503.14269.pdf)** (2025-03-19)

*Vaibhav Aggarwal, Ojasv Kamal, Abhinav Japesh, Zhijing Jin, Bernhard Schölkopf*

  Large Language Models (LLMs) have revolutionized various domains, including
natural language processing, data analysis, and software development, by
enabling automation. In software engineering, LLM-powered coding agents have
garnered significant attention due to their potential to automate complex
development tasks, assist in debugging, and enhance productivity. However,
existing approaches often struggle with sub-optimal decision-making, requiring
either extensive manual intervention or inefficient compute scaling strategies.
To improve coding agent performance, we present Dynamic Action Re-Sampling
(DARS), a novel inference time compute scaling approach for coding agents, that
is faster and more effective at recovering from sub-optimal decisions compared
to baselines. While traditional agents either follow linear trajectories or
rely on random sampling for scaling compute, our approach DARS works by
branching out a trajectory at certain key decision points by taking an
alternative action given the history of the trajectory and execution feedback
of the previous attempt from that point. We evaluate our approach on SWE-Bench
Lite benchmark, demonstrating that this scaling strategy achieves a pass@k
score of 55% with Claude 3.5 Sonnet V2. Our framework achieves a pass@1 rate of
47%, outperforming state-of-the-art (SOTA) open-source frameworks.


---

**[138. [2409.10680] Multi-agent Path Finding in Continuous Environment](https://arxiv.org/pdf/2409.10680.pdf)** (2024-09-18)

*Kristýna Janovská, Pavel Surynek*

  We address a variant of multi-agent path finding in continuous environment
(CE-MAPF), where agents move along sets of smooth curves. Collisions between
agents are resolved via avoidance in the space domain. A new Continuous
Environment Conflict-Based Search (CE-CBS) algorithm is proposed in this work.
CE-CBS combines conflict-based search (CBS) for the high-level search framework
with RRT* for low-level path planning. The CE-CBS algorithm is tested under
various settings on diverse CE-MAPF instances. Experimental results show that
CE-CBS is competitive w.r.t. to other algorithms that consider continuous
aspect in MAPF such as MAPF with continuous time.


---

**[139. [2312.15161] Networks of Classical Conditioning Gates and Their Learning](https://arxiv.org/pdf/2312.15161.pdf)** (2023-12-27)

*Shun-ichi Azuma, Dai Takakura, Ryo Ariizumi, Toru Asai*

  Chemical AI is chemically synthesized artificial intelligence that has the
ability of learning in addition to information processing. A research project
on chemical AI, called the Molecular Cybernetics Project, was launched in Japan
in 2021 with the goal of creating a molecular machine that can learn a type of
conditioned reflex through the process called classical conditioning. If the
project succeeds in developing such a molecular machine, the next step would be
to configure a network of such machines to realize more complex functions. With
this motivation, this paper develops a method for learning a desired function
in the network of nodes each of which can implement classical conditioning.
First, we present a model of classical conditioning, which is called here a
classical conditioning gate. We then propose a learning algorithm for the
network of classical conditioning gates.


---

**[140. [2006.02536] Phasic dopamine release identification using ensemble of AlexNet](https://arxiv.org/pdf/2006.02536.pdf)** (2020-06-05)

*Luca Patarnello, Marco Celin, Loris Nanni*

  Dopamine (DA) is an organic chemical that influences several parts of
behaviour and physical functions. Fast-scan cyclic voltammetry (FSCV) is a
technique used for in vivo phasic dopamine release measurements. The analysis
of such measurements, though, requires notable effort. In this paper, we
present the use of convolutional neural networks (CNNs) for the identification
of phasic dopamine releases.


---

**[141. [2110.12352] DiffSRL: Learning Dynamical State Representation for Deformable Object
  Manipulation with Differentiable Simulator](https://arxiv.org/pdf/2110.12352.pdf)** (2022-07-27)

*Sirui Chen, Yunhao Liu, Jialong Li, Shang Wen Yao, Tingxiang Fan, Jia Pan*

  Dynamic state representation learning is an important task in robot learning.
Latent space that can capture dynamics related information has wide application
in areas such as accelerating model free reinforcement learning, closing the
simulation to reality gap, as well as reducing the motion planning complexity.
However, current dynamic state representation learning methods scale poorly on
complex dynamic systems such as deformable objects, and cannot directly embed
well defined simulation function into the training pipeline. We propose
DiffSRL, a dynamic state representation learning pipeline utilizing
differentiable simulation that can embed complex dynamics models as part of the
end-to-end training. We also integrate differentiable dynamic constraints as
part of the pipeline which provide incentives for the latent state to be aware
of dynamical constraints. We further establish a state representation learning
benchmark on a soft-body simulation system, PlasticineLab, and our model
demonstrates superior performance in terms of capturing long-term dynamics as
well as reward prediction.


---

**[142. [2310.04353] An In-Context Learning Agent for Formal Theorem-Proving](https://arxiv.org/pdf/2310.04353.pdf)** (2024-08-09)

*Amitayush Thakur, George Tsoukalas, Yeming Wen, Jimmy Xin, Swarat Chaudhuri*

  We present an in-context learning agent for formal theorem-proving in
environments like Lean and Coq. Current state-of-the-art models for the problem
are finetuned on environment-specific proof data. By contrast, our approach,
called COPRA, repeatedly asks a high-capacity, general-purpose large language
model (GPT-4) to propose tactic applications from within a stateful
backtracking search. Proposed tactics are executed in the underlying proof
environment. Feedback from the execution is used to build the prompt for the
next model query, along with selected information from the search history and
lemmas retrieved from an external database. We evaluate our implementation of
COPRA on the miniF2F benchmark for Lean and a set of Coq tasks from the
CompCert project. On these benchmarks, COPRA significantly outperforms few-shot
invocations of GPT-4. It also compares favorably against finetuning-based
approaches, outperforming ReProver, a state-of-the-art finetuned approach for
Lean, in terms of the pass@1 metric. Our code and data are available at
https://github.com/trishullab/copra.


---

**[143. [2210.10125] Proximal Learning With Opponent-Learning Awareness](https://arxiv.org/pdf/2210.10125.pdf)** (2022-10-20)

*Stephen Zhao, Chris Lu, Roger Baker Grosse, Jakob Nicolaus Foerster*

  Learning With Opponent-Learning Awareness (LOLA) (Foerster et al. [2018a]) is
a multi-agent reinforcement learning algorithm that typically learns
reciprocity-based cooperation in partially competitive environments. However,
LOLA often fails to learn such behaviour on more complex policy spaces
parameterized by neural networks, partly because the update rule is sensitive
to the policy parameterization. This problem is especially pronounced in the
opponent modeling setting, where the opponent's policy is unknown and must be
inferred from observations; in such settings, LOLA is ill-specified because
behaviorally equivalent opponent policies can result in non-equivalent updates.
To address this shortcoming, we reinterpret LOLA as approximating a proximal
operator, and then derive a new algorithm, proximal LOLA (POLA), which uses the
proximal formulation directly. Unlike LOLA, the POLA updates are
parameterization invariant, in the sense that when the proximal objective has a
unique optimum, behaviorally equivalent policies result in behaviorally
equivalent updates. We then present practical approximations to the ideal POLA
update, which we evaluate in several partially competitive environments with
function approximation and opponent modeling. This empirically demonstrates
that POLA achieves reciprocity-based cooperation more reliably than LOLA.


---

**[144. [2304.03274] DiffMimic: Efficient Motion Mimicking with Differentiable Physics](https://arxiv.org/pdf/2304.03274.pdf)** (2023-04-27)

*Jiawei Ren, Cunjun Yu, Siwei Chen, Xiao Ma, Liang Pan, Ziwei Liu*

  Motion mimicking is a foundational task in physics-based character animation.
However, most existing motion mimicking methods are built upon reinforcement
learning (RL) and suffer from heavy reward engineering, high variance, and slow
convergence with hard explorations. Specifically, they usually take tens of
hours or even days of training to mimic a simple motion sequence, resulting in
poor scalability. In this work, we leverage differentiable physics simulators
(DPS) and propose an efficient motion mimicking method dubbed DiffMimic. Our
key insight is that DPS casts a complex policy learning task to a much simpler
state matching problem. In particular, DPS learns a stable policy by analytical
gradients with ground-truth physical priors hence leading to significantly
faster and stabler convergence than RL-based methods. Moreover, to escape from
local optima, we utilize a Demonstration Replay mechanism to enable stable
gradient backpropagation in a long horizon. Extensive experiments on standard
benchmarks show that DiffMimic has a better sample efficiency and time
efficiency than existing methods (e.g., DeepMimic). Notably, DiffMimic allows a
physically simulated character to learn Backflip after 10 minutes of training
and be able to cycle it after 3 hours of training, while the existing approach
may require about a day of training to cycle Backflip. More importantly, we
hope DiffMimic can benefit more differentiable animation systems with
techniques like differentiable clothes simulation in future research.


---

**[145. [2405.02957] Agent Hospital: A Simulacrum of Hospital with Evolvable Medical Agents](https://arxiv.org/pdf/2405.02957.pdf)** (2025-01-20)

*Junkai Li, Yunghwei Lai, Weitao Li, Jingyi Ren, Meng Zhang, Xinhui Kang, Siyu Wang, Peng Li, Ya-Qin Zhang, Weizhi Ma, Yang Liu*

  The recent rapid development of large language models (LLMs) has sparked a
new wave of technological revolution in medical artificial intelligence (AI).
While LLMs are designed to understand and generate text like a human,
autonomous agents that utilize LLMs as their "brain" have exhibited
capabilities beyond text processing such as planning, reflection, and using
tools by enabling their "bodies" to interact with the environment. We introduce
a simulacrum of hospital called Agent Hospital that simulates the entire
process of treating illness, in which all patients, nurses, and doctors are
LLM-powered autonomous agents. Within the simulacrum, doctor agents are able to
evolve by treating a large number of patient agents without the need to label
training data manually. After treating tens of thousands of patient agents in
the simulacrum (human doctors may take several years in the real world), the
evolved doctor agents outperform state-of-the-art medical agent methods on the
MedQA benchmark comprising US Medical Licensing Examination (USMLE) test
questions. Our methods of simulacrum construction and agent evolution have the
potential in benefiting a broad range of applications beyond medical AI.


---

**[146. [2110.14237] Learning Graph Cellular Automata](https://arxiv.org/pdf/2110.14237.pdf)** (2021-10-28)

*Daniele Grattarola, Lorenzo Livi, Cesare Alippi*

  Cellular automata (CA) are a class of computational models that exhibit rich
dynamics emerging from the local interaction of cells arranged in a regular
lattice. In this work we focus on a generalised version of typical CA, called
graph cellular automata (GCA), in which the lattice structure is replaced by an
arbitrary graph. In particular, we extend previous work that used convolutional
neural networks to learn the transition rule of conventional CA and we use
graph neural networks to learn a variety of transition rules for GCA. First, we
present a general-purpose architecture for learning GCA, and we show that it
can represent any arbitrary GCA with finite and discrete state space. Then, we
test our approach on three different tasks: 1) learning the transition rule of
a GCA on a Voronoi tessellation; 2) imitating the behaviour of a group of
flocking agents; 3) learning a rule that converges to a desired target state.


---

**[147. [2010.10210] Quality of service based radar resource management using deep
  reinforcement learning](https://arxiv.org/pdf/2010.10210.pdf)** (2021-07-26)

*Sebastian Durst, Stefan Brüggenwirth*

  An intelligent radar resource management is an essential milestone in the
development of a cognitive radar system. The quality of service based resource
allocation model (Q-RAM) is a framework allowing for intelligent decision
making but classical solutions seem insufficient for real-time application in a
modern radar system. In this paper, we present a solution for the Q-RAM radar
resource management problem using deep reinforcement learning considerably
improving on runtime performance.


---

**[148. [2302.09511] Dynamic Private Task Assignment under Differential Privacy](https://arxiv.org/pdf/2302.09511.pdf)** (2023-02-21)

*Leilei Du, Peng Cheng, Libin Zheng, Wei Xi, Xuemin Lin, Wenjie Zhang, Jing Fang*

  Data collection is indispensable for spatial crowdsourcing services, such as
resource allocation, policymaking, and scientific explorations. However,
privacy issues make it challenging for users to share their information unless
receiving sufficient compensation. Differential Privacy (DP) is a promising
mechanism to release helpful information while protecting individuals' privacy.
However, most DP mechanisms only consider a fixed compensation for each user's
privacy loss. In this paper, we design a task assignment scheme that allows
workers to dynamically improve their utility with dynamic distance privacy
leakage. Specifically, we propose two solutions to improve the total utility of
task assignment results, namely Private Utility Conflict-Elimination (PUCE)
approach and Private Game Theory (PGT) approach, respectively. We prove that
PUCE achieves higher utility than the state-of-the-art works. We demonstrate
the efficiency and effectiveness of our PUCE and PGT approaches on both real
and synthetic data sets compared with the recent distance-based approach,
Private Distance Conflict-Elimination (PDCE). PUCE is always better than PDCE
slightly. PGT is 50% to 63% faster than PDCE and can improve 16% utility on
average when worker range is large enough.


---

**[149. [2205.09123] A2C is a special case of PPO](https://arxiv.org/pdf/2205.09123.pdf)** (2022-05-20)

*Shengyi Huang, Anssi Kanervisto, Antonin Raffin, Weixun Wang, Santiago Ontañón, Rousslan Fernand Julien Dossa*

  Advantage Actor-critic (A2C) and Proximal Policy Optimization (PPO) are
popular deep reinforcement learning algorithms used for game AI in recent
years. A common understanding is that A2C and PPO are separate algorithms
because PPO's clipped objective appears significantly different than A2C's
objective. In this paper, however, we show A2C is a special case of PPO. We
present theoretical justifications and pseudocode analysis to demonstrate why.
To validate our claim, we conduct an empirical experiment using
\texttt{Stable-baselines3}, showing A2C and PPO produce the \textit{exact} same
models when other settings are controlled.


---
