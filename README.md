# ReinforcementLearningByPython
---
Reading Log of a book, "Reinforcement Learning by Python".

## Table of contents
---
<!-- TOC -->

- [ReinforcementLearningByPython](#reinforcementlearningbypython)
    - [Table of contents](#table-of-contents)
    - [Introduction](#introduction)
    - [Author](#author)
    - [Day 1: How does Reinforcement Learning work in Machine Learning?](#day-1-how-does-reinforcement-learning-work-in-machine-learning)
        - [Marcov Decision Process](#marcov-decision-process)
        - [Sum of Rewards](#sum-of-rewards)
    - [Day 2: Programming based on Environment](#day-2-programming-based-on-environment)
        - [Definition and calculation of Value: Bellman Equation](#definition-and-calculation-of-value-bellman-equation)
        - [Learning Value evaluation by Dynamic Programming: Value Iteration](#learning-value-evaluation-by-dynamic-programming-value-iteration)
        - [Learning Policy by Dynamic Programming: Policy Iteration](#learning-policy-by-dynamic-programming-policy-iteration)
        - [Difference between "Model based" and "Model free"](#difference-between-model-based-and-model-free)
    - [Day 3: Programming based on Experience](#day-3-programming-based-on-experience)
        - [Prerequisites](#prerequisites)
        - [Reference about Continuing Task](#reference-about-continuing-task)
        - [A balance of Experience Accumulation and Exploitation](#a-balance-of-experience-accumulation-and-exploitation)
        - [Plan Modification based on Experience or Prediction: Monte Carlo vs Temporal Difference](#plan-modification-based-on-experience-or-prediction-monte-carlo-vs-temporal-difference)
        - [Temporal Difference Error](#temporal-difference-error)
        - [Modification based on Experience](#modification-based-on-experience)
        - [Modification by Monte Carlo method](#modification-by-monte-carlo-method)
        - [Multi-step Learning](#multi-step-learning)
        - [TD($\lambda$) method](#td\lambda-method)
        - [Value based vs Policy based](#value-based-vs-policy-based)
    - [Day4: Applying Neural Network to Reinforcement learning](#day4-applying-neural-network-to-reinforcement-learning)
        - [1 layer Neural Network](#1-layer-neural-network)
        - [Activation function](#activation-function)
        - [Propagation](#propagation)
        - [Backpropagation](#backpropagation)
        - [Regularization](#regularization)
        - [Advantage of applying Neural Network](#advantage-of-applying-neural-network)
        - [Experience Reply](#experience-reply)
        - [Value Function Approximation](#value-function-approximation)
        - [Fixed Target Q-Network](#fixed-target-q-network)
        - [Reward Clipping](#reward-clipping)
        - [Deep Q-Network](#deep-q-network)
        - [Rainbow](#rainbow)
            - [1. Double DQN](#1-double-dqn)
            - [2. Prioritized Reply](#2-prioritized-reply)
            - [3. Dueling Network](#3-dueling-network)
            - [4. Multi-step Learning](#4-multi-step-learning)
            - [5. Distributional RL](#5-distributional-rl)
            - [6. Noisy Nets](#6-noisy-nets)
        - [Policy Gradient](#policy-gradient)
            - [Expectation of policy value](#expectation-of-policy-value)
            - [Maximizing expectation](#maximizing-expectation)
            - [Evaluating action value](#evaluating-action-value)
        - [Advantage Actor Critic(A2C)](#advantage-actor-critica2c)
            - [Deep Deterministic Policy Gradient(DDPG)](#deep-deterministic-policy-gradientddpg)
            - [Technique to stabilize execution result](#technique-to-stabilize-execution-result)
        - [Value evaluation or Policy?](#value-evaluation-or-policy)

<!-- /TOC -->

## Introduction
---
We can learn Reinforcement Learning and become be able to apply to our practice. For the purposes, this book has the following differences from any other books.  

1. Python sample codes which are designed practically.  
2. This book explains about "Weakness" of Reinforcement Learning and presents how to overcome it.  
3. This book organizes a system of research to deepen our learning after reading.  

I extracted some importance points which I impressed and wrote them as memo in this article.  

## Author
---
* [Takahiro Kubo](https://www.wantedly.com/users/245795)  

## Day 1: How does Reinforcement Learning work in Machine Learning?
---
* Environment(Task) which Agent can get Reward depending on Action is given and the parameters of model are adjusted to output Action which leads to Reward at each state.  

* "Action" and change of "State" depending on Action are defined in "Environment".  

* Model is a function which gets "State" and output "Action".  

* The different point from Supervised Learning is what Reinforcement Learning optimizes with whole of Rewards.  

* Period from start to end of Environment is called "Episode". The purpose of Reinforcement Learning is to optimize Rewards which can be got during 1 Period.  

* The model learns the following 2 things:  
    1. How to evaluate Action (This is one of the advantages).  
    2. How to select Action depending on evaluation.  

* The disadvantage is what we can not control the behavior of model.  

### Marcov Decision Process

* It is assumed that state of transition target depends on only the state and action at last minute.  

* $s$: State
* $a$: Action
* $T$: State transition probability (Input: State and Action, Output: Next State and Transition probability)
* $R$: Immediate Reward (Input: State, Next State and Action, Output: Reward)  

* $\pi$: Policy: Learning target model (Input: State, Output: Action)  

### Sum of Rewards

* $G_t$: Sum of Rewards at time $t$ (end of time: $T$)
$$
    G_t = r_{t+1}+r_{t+2}+r_{t+3} \cdots +r_T
$$

* Agent needs to know Sum of Rewards in selecting Action. (Estimation)
* Estimation is uncertain, so each Reward is discounted as follow.  
$$
    G_t = r_{t+1}+\gamma r_{t+2}+\gamma^2 r_{t+3} \cdots +\gamma^{T-t-1} r_T = \sum_{k=0}^{T}\gamma^k r_{t+k+1}
$$
* Discount factor: $\gamma$ is from 0 to 1.  
* Discounted current value: $\gamma^k r_{t+k+1}$  
* Estimated $G_t$ is called "Expected Rewards" or "Value"  
* Calculation of Value is called "Value Approximation"  
* Setting of default Reward affects Action of Agent. If it is set as negative value, it encourages Agent to reach Goal early.  

## Day 2: Programming based on Environment
---
* Method to learn based on Transition function and Reward function is called "Model based Learning".  

### Definition and calculation of Value: Bellman Equation

* Value $G_t$ has the following 2 problems:  
    1. The value of $R$(Immediate Reward) need to be found.  
    2. The Agent can absolutely get the Reward.  

* How to solve the 1st problem:  
    * Recursive definition of Value $G_t$:  
    $$
        G_t = r_{t+1} + \gamma G_{t+1}
    $$  
    * Only the latest Immediate Reward $r_{t+1}$ need to be found. The calculation of future $G_{t+1}$ can be left out.(assigned temporal appropriate value)  

* How to solve the 2nd problem:  
    * Action probability times Immediate Reward.  
    * Definition of Action (1): Agent actions based on Policy $\pi$  
    * Definition of Action (2): Agent always selects Action which maximize "Value".  
    * Action probability based on Policy: $\pi(a|s)$  
    * Transition probability: $T(s'|s,a)$  
    * Value by Action based on Policy: $V_\pi (s)$  
    $$
        V_\pi (s_t) = E_\pi [r_{t+1}+\gamma V_\pi (s_{t+1})]
    $$  
    $$
        V_\pi (s) = \sum_{a}\pi(a|s) \sum_{s'}T(s'|s,a)(R(s,s')+\gamma V_\pi(s'))
    $$  
    * Case of selecting Action which maximizes "Value":  
    $$
        V(s) = max \sum_{s'} T(s'|s,a)(R(s,s')+\gamma V(s'))
    $$  
    * Case of what Reward is decided depending on only State:  
    $$
        V(s) = R(s) + \gamma max \sum_{s'} T(s'|s,a)V(s')
    $$  

* Reinforcement Learning had 2 patterns of direction as follow:  
    1. Selecting Action based on Policy  
    2. Selecting Action which maximizes "Value"  

### Learning Value evaluation by Dynamic Programming: Value Iteration

* Basic thinking  
Agent calculates Value at each State and get Action to transit the highest value state.  

* Value Iteration  
Method to calculate Value at each State by Dynamic Programming.  
$$
    V_{i+1}(s) = max\{ \sum_{s'}T(s'|s,a)(R(s)+\gamma V_i(s')) \}
$$  

* How to judge  
"Get close to accurate value" is judged by checking the difference $|V_{i+1}(s)-V_i(s)|$ is getting lower than threshold.  

### Learning Policy by Dynamic Programming: Policy Iteration

* Basic thinking  
Agent acts depending on Policy. Policy outputs Action probability and can calculate Value(Expected value) based on Action probability.  

* Policy iteration process  
    1. Calculating Value based on Policy  
    2. Updating Policy in maximizing Value  
    3. Improving accuracy of both Value evaluation and Policy  

* Calculation performance  
Policy iteration is faster than Value iteration because Policy iteration doesn't need to calculate Value at all of States. Policy iteration needs to calculate Value every time Policy is updated.  

### Difference between "Model based" and "Model free"

* Model based method can get the optimal policy based on only Environment without Agent's movement because Transition function and Reward function are obvious. This method is effective when moving Agent is costly or Environment has a lot of noise.  

* Model free method can plan by moving Agent in fact. In this case, Transition function and Reward function doesn't need to be obvious.  

* Typically, Model free method is more popular than Model based method because there is few case which Transition function and Reward function are obvious or they can be modeled well.  

## Day 3: Programming based on Experience
---
### Prerequisites
Informations of Environment, in other wards, Transition function and Reward function is not obvious.  

### Reference about Continuing Task
* Unified Notation for Episodic and Continuing Tasks
[Reinforcement Learning: An Introduction](http://incompleteideas.net/book/bookdraft2017nov5.pdf)  

### A balance of Experience Accumulation and Exploitation
* Exploration-exploitation trade-off
How much Agent acts for Exploration and how much Agent acts for Rewards?  

* Epsilon-Greedy method
This method is to make a balance of trade-off. Agent acts for Exploration based on Epsilon probability and acts for Exploitation based on Greedy Action. For example, if a value of Epsilon is 0.2, the Agent acts for Exploration with 20% probability and acts for Exploitation with 80% probability.  

### Plan Modification based on Experience or Prediction: Monte Carlo vs Temporal Difference
* Experience means Sum of Rewards during the episode.  
* Problem of Experience based method  
Agent can not modify his Action until the episode ends. This means that Agent has to continue the Action even though the Action is not optimal.  
* Problem of Prediction based method  
Action is modified based on estimated sum of Rewards, so the accuracy may be not good.  
* Accuracy-Speed of modification trade-off  

### Temporal Difference Error
* $V(s)$: Estimated Value before Action  
* $V(s')$: Real Value after Action  
* $\gamma$: Discount rate  
$$
    r + \gamma V(s') - V(s)
$$  

### Modification based on Experience
* Process to reduce Temporal Difference Error  
* $\alpha$: Coefficient to make a balance between Estimated value and Real value  
* $\alpha$ is called "Learning Rate"  
$$
    V(s_t) \leftarrow V(s_t) + \alpha(r_{t+1} + \gamma V(s_{t+1}) - V(s_t))
$$  
* Q-Learning  
Value of Action at State ($Q(s, a)$) is called "Q value" Customarily. Transition Value is calculated based on Value based thinking. Agent acts $a$ maximizing the Value. "$V$" is used as a symbol which mean "Value of State".  

### Modification by Monte Carlo method
* Episode will end at time $T$  
$$
    V(s_t) \leftarrow V(s_t) + \alpha((r_{t+1}+\gamma r_{t+2}+\gamma^2 r_{t+3}+\cdots+\gamma^{T-1} r_T) - V(s_t))
$$  

* Every-Visit  
How to calculate a discounted current value $G$ from each time.  

* First-Visit  
Setting the origin of calculating a discounted current value $G$ to the time when each state and action appeared for the first time.  

### Multi-step Learning
* Duration until modification is set as longer than 1 and shorter than $T$.  
* This setting is usually 2 steps or 3 steps.  

### TD($\lambda$) method
* Calculating a real value at each step  
    1 step: $G_t^1=r_{t+1}+\gamma V(s_{t+1})$  
    2 step: $G_t^2=r_{t+1}+\gamma r_{t+2} + \gamma^2 V(s_{t+2})$  
    $\vdots$  
    episode end: $G_t^T=r_{t+1}+\gamma r_{t+2}+ \cdots + \gamma^{T-1} r_T$  

* Calculating sum of real values at each step  
* The real value at each step times coefficient $\lambda$(0~1)  
    $$
        G_t^\lambda = (1-\lambda) \sum_{n=1}^{T} \lambda^{n-1}G_t^{(n)}
    $$  

### Value based vs Policy based
* Which Experience is used for, a update of "Value evaluation" or "Policy"?  
* Policy based method: SARSA(State-Action-Reward-State-Action)  
* Actor-Critic method  
Method to combine "Value based" with "Policy based". Actor is in charge of Policy ans Critic is in charge of Value evaluation. 

## Day4: Applying Neural Network to Reinforcement learning
---
Method to implement an value evaluation and a policy as a function with a parameter.  

### 1 layer Neural Network

\[
    \left[
        \begin{array}{c}
            y_1 \\
            y_2 \\
            y_3 \\
            y_4 \\
        \end{array}
    \right] = \left[
        \begin{array}{cc}
            a_{11} & a_{12} \\
            a_{21} & a_{22} \\
            a_{31} & a_{32} \\
            a_{41} & a_{42} \\
        \end{array}
    \right] \left[
        \begin{array}{c}
            x_1 \\
            x_2 \\
        \end{array}
    \right] + \left[
        \begin{array}{c}
            b_1 \\
            b_2 \\
            b_3 \\
            b_4 \\
        \end{array}
    \right]
\]  

This is a simple 1 layer Neural Network example. Neural Netwark is what an input valuable times a weight plus a bias sequentially. An output at an layer will be an input at the next layer. When there is a lot of layers, this is called Deep Neural Network.  

### Activation function

A function is inserted between an input and an output. The function is called "Activation function".  

### Propagation

A passing process like output, activation function, input to next layer is called "Propagation". When the propagation process is executed, multiple data is processed all together. This unit is called "Batch".  

### Backpropagation

Neural Network is learned by adjusting a weight and a bias at each layer. "How is the output wrong?" is passed from the output side.  

### Regularization

This is a method to limit a weight used at each layer. If it was not limited, a prediction result which is accurate for only training data.  

### Advantage of applying Neural Network

Data which is similar with "State" a human observe in fact can be used for training of Agent.  
This method has a disadvantage that it takes long time to train too. For applying Neural Network to Reinforcement learning, a design to overcome the disadvantage should be executed.  

### Experience Reply

A history of action is stored temporarily and training data is sampled from them. This purpose is to stabilize the training. Data which has different time step at different episode can be used.  

### Value Function Approximation

A function to evaluate value is called Value function. Training(estimating) the value function is called Value Function Approximation. Selecting action is executed based on an output from value function (value based method).  

### Fixed Target Q-Network

This is a method to calculate a value at a transition state by a network with a fixed parameter during fixed interval.  

### Reward Clipping

This is to integrate rewards during all of games. For example, success reward is 1 and failure reward is -1.  

### Deep Q-Network

This is a method to use CNN(Convolutional Neural Netwark) and to stabilize training by Experience Reply, Fixed Target Q-Netwark and Reward Clipping.  

### Rainbow
[Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)  

This method has the following 6 techniques.  

#### 1. Double DQN  

This is used for improving an estimation accuracy of a value. This method suggests that a network of action value and the other one of action choice are devided.  

#### 2. Prioritized Reply  

This is used for improving a training efficiency. A rate of random sampling and the other one of sampling based on TD error are adjusted with a parameter.  

#### 3. Dueling Network

This is used for improving an estimation accuracy of a value. This method is to calculate a value of state and the other one of action separately.  

#### 4. Multi-step Learning

This is used for improving an estimation accuracy of a value. This method is to modify by rewards during n steps and values at a state in the future n steps.  

For example, a formula which n=3 is as follow.  

\[
    \delta = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \gamma^3 max Q(s_{t+3}, a) - Q(s_t, a_t)
\]  

#### 5. Distributional RL

This is used for improving an estimation accuracy of a value. Rewards is used as distribution. The average and the variance change depending on the state or action.  

#### 6. Noisy Nets

This is used for improving searching efficiency. It has the network learn how the agent acts randomly.  

\[
    y = (W + \sigma^w \odot \epsilon^w)x + (b + \sigma^b \odot \epsilon^b)
\]  

$\epsilon$: random noise  
$\sigma$: quantity of random noise  

### Policy Gradient

Policy can be expressed as a function with a parameter.  
Input: state  
Output: action or action probability  

#### Expectation of policy value

This expectation is calculated with transition probability to a state following policy, action probability and action value.  

\[
    J(\theta) \propto \sum_{s\in S} d^{\pi_\theta} (s) \sum_{a \in A} \pi_\theta (a|s) Q^{\pi_\theta} (s,a)
\]  

$\pi_\theta$: function with parameter $\theta$ as Policy.  
$d^{\pi_\theta} (s)$: transition probability to state $s$ following the policy  
$\pi_\theta (a|s)$: action probability which takes action $a$ at the state $s$  
$Q^{\pi_\theta} (s,a)$: action value  

#### Maximizing expectation

Expectation $J(\theta)$ is maximized by Gradient method. Normal Gradient method is for minimization, so it can be used as maximization by timing minus. Gradient of expectation $\nabla J(\theta)$ is expressed as follow.  

\[
    \nabla J(\theta) \propto \sum_{s\in S} d^{\pi_\theta} (s) \sum_{a \in A} \nabla \pi_\theta (a|s) Q^{\pi_\theta} (s,a)
\]  

\[
    \nabla \pi_\theta (a|s) = \pi_\theta (a|s) \frac{\nabla \pi_\theta (a|s)}{\pi_\theta (a|s)} = \pi_\theta (a|s) \nabla log \pi_\theta (a|s)
\]  

#### Evaluating action value

Action value at a state $Q(s, a)$ depends on a difference of the state rather than a difference of the action. So, the action value should be evaluated by subtracting the state value as follow.  

\[
    A(s, a) = Q_w (s, a) - V_v (s)
\]  

This pure action value $A(s, a)$ is called "Advantage".  

### Advantage Actor Critic(A2C)

Deep Neural Network(DNN) can be applied to a policy function.  

#### Deep Deterministic Policy Gradient(DDPG)

An action is not sampled from a distribution of action probability but output directly. This method uses not Advantage but TD error and uses Fixed Target Q-Network for learning.  

#### Technique to stabilize execution result

This method is to impose a constraint which an action distribution changes from the previous one gradually.  

\[
    E_t [KL[\pi_{\theta_{old}}(\cdot|s_t), \pi_{\theta}(\cdot|s_t)]] \le \delta
\]  

$\pi_{\theta_{old}}(\cdot|s_t)$: action distribution before update  
$\pi_{\theta}(\cdot|s_t)$: action distribution after update  
$KL$: Kullback-Leibler Distance, an index to measure between distributions  

\[
    maximize E_t [\frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}A_t]
\]  

$E_t$ is update by maximizing Advantage depending on a weight $\frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$. This method is "Trust Region Policy Optimization(TRPO)".  

### Value evaluation or Policy?

Policy based method can choose an action probabilistically and overcome an disadvantage of Value based method. The disadvantage is to choose an biased action if it has some same value actions. 