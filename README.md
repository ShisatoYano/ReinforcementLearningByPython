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
    G_t = r_{t+1}+\gamma r_{t+2}+\gamma^2 r_{t+3} \cdots +\gamma^{T-t-1} r_T = \sum_{k=0}^{k=T}\gamma^k r_{t+k+1}
$$
* Discount factor: $\gamma$ is from 0 to 1.  
* Discounted current value: $\gamma^k r_{t+k+1}$  
* Estimated $G_t$ is called "Expected Rewards" or "Value"  
* Calculation of Value is called "Value Approximation"  
* Setting of default Reward affects Action of Agent. If it is set as negative value, it encourages Agent to reach Goal early.  
