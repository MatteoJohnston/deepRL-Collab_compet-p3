## Problem

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to moves toward (or away from) the net, and jumping.
From this brief environment's description, we can quickly gather that main difference from the previous project is that we need model the interaction between agents. Hence the introduction to the so called multi-agent actor-critic models. 

## Goal
The goal of this project is to train two RL agents to play tennis. As in real tennis, the goal of each player is to keep the ball in play. And, when you have two equally matched opponents, you tend to see fairly long exchanges where the players hit the ball back and forth over the net.

##### &nbsp;

## The Environment
We'll work with an environment that is similar, but not identical to the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment on the Unity ML-Agents GitHub page.

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to moves toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved when the average (over 100 episodes) of those **scores** is at least +0.5.

##### &nbsp;

## Approach
Here are the high-level steps taken in building an agent that solves this environment.

1. Performance benchmarking: take a random action
2. "Know thyself": which algorithm to use
3. Methodology
4. Results and conclusions 
5. Further improvements

##### &nbsp;

## Benchmark: take a random action
As part of project, we tried to solve the environment just by taking a random action (randomly distributed). Although naive this is "de-facto" an initial benchmarking exercise. 
See the results <img src="Random action.PNG" align="bottom-left" alt="" title="Random action" />
It is clear that we need to do a little more in order to solve the problem. In this particular the maximum score per episode was 0!

## "Know thyself": which algorithm to use
To get started, there are a few high-level architecture decisions we need to make. First, we need to determine which types of algorithms are most suitable for the Tennis environment.


Before delving into the core of the exercise there are two two main key features that is worth mentioning:

1. **Multiple agents** &mdash; The Tennis environment has 2 different agents
2. **Continuous action space** &mdash; The action space is now _continuous_, which allows each agent to execute more complex and precise movements. Even though each tennis agent can only move forward, backward, or jump, there's an unlimited range of possible action values that control these movements. Whereas, the agent in the Navigation project was limited to four _discrete_ actions: left, right, forward, backward.

As articulated in our previous project, it is better to use a **policy-based method**. Policy-based methods are **well suited for continuous spaces**, hence they will be very useful in this context. Furthermore, differently from the value-based methods, they can learn also **stochastic policies** rather than just deterministic. Finally they can directly learn the optimal policy ![pi star](https://github.com/MatteoJohnston/deepRL-Continous_control-p2/pi_star.png) without having to maintain a separate value function estimate. Intuitively we can see how this can be a main advantage of the method both from a theoretical standpoint as well computational. Within the value-based methods, the agent uses its experience with the environment to maintain an estimate of the optimal action-value function, from which an optimal policy is derived. The computational cost for maintaining this estimate of the optimal action-value function can soon become expensive.


## Methodology: Multi-Agent Deep Deterministic Policy Gradient (MADDPG)
We built our MADDPG algorithm using the pre-existing body or work and research in project number 2. See the suggesting readings: [_Benchmarking Deep Reinforcement Learning for Continuous Control_][benchmarking-paper], [_Continuous Control with Deep Reinforcement Learning_][ddpg-paper]).

Our code base has been built upon the udacity repository, please find it here: [Udacity DRL `ddpg-bipedal` notebook][ddpg-repo], which further altered as a result of our own research. 

[benchmarking-paper]: https://arxiv.org/pdf/1604.06778.pdf
[ddpg-paper]: https://arxiv.org/pdf/1509.02971.pdf
[ddpg-repo]: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-bipedal/DDPG.ipynb

To make this algorithm suitable for the problem to solve, we followed the concepts discussed in [this paper](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf), _Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments_, by Lowe and Wu. In particular, we implemented their variation of the actor-critic method.

The image below shows a general graphical representation of multi-agent actor-critic methods:
 <img src="MADDPG network.PNG" align="bottom-left" alt="" title="MADDPG Network" />


The algorithm used lives under the umbrella of the so called "actor-critic methods" which, in a nutshell, are a "generalized policy iteration" alternating between a policy evaluation and a policy improvement step.
There are two closely related processes:
- actor improvement which aims at improving the current policy: the main task of the agent (actor) is to learn how to act by directly estimating the optimal policy and maximizing rewards;
- critic evaluation which evaluates the current policy: via a value-based approach, the agent (critic) learns how to estimate the value (i.e., the future cumulative reward) of different state-action pairs. 
The power behind "Actor-critic methods" is that they combine these two approaches in order to accelerate the learning process. Actor-critic agents are generally also more stable than value-based agents, while requiring fewer training samples than policy-based agents.

Our implementation however further differs because of the **decentralized actor with centralized critic** feature. In other words, whereas traditional actor-critic methods have a separate critic for each agent, this approach utilizes a single critic that receives as input the actions and state observations from all agents. This extra information makes training easier and allows for centralized training with decentralized execution. Each agent still takes actions based on its own unique observations of the environment.


## Actor-Critic models

You can find actor-critic logic implemented here as part the `Agent()` class in `maddpg_agent.py` of the source code <img src="actor_critic network.png" align="bottom-left" alt="" title="actor critic network" />. 
Please find the source code [here](https://github.com/MatteoJohnston/deepRL-Continous_control-p2/blob/master/ddpg_agent.py#L51).

The gradient of the actor is now defined as:
 <img src="Gradient of actors.PNG" align="bottom-left" alt="" title="gradient of actors" />
 
Note: We're again using local and target networks to improve stability. This is where one set of parameters w is used to select the best action, and another set of parameters w' is used to evaluate that action. In this project, local and target networks are implemented separately for both the actor and the critic.
 

## Network architecture 


You can find both the `Actor()` and the `Critic()` class in `model.py`. Please find the source code [here](https://github.com/MatteoJohnston/deepRL-Collab_compet-p3/blob/master/model.py#L1). 
Our architecture, which is quite standard, have 2 networks with the following structures and hyper parameters (hidden layers and number of units per hidden layers):

- Actor: 300 -> 300
- Critic: 300 -> 300

Although we tested smaller and bigger networks we realized that just augmenting the networks size is not normally enough or what what worse it is difficult to tell with only few hours. Unfortunately it took really long time to get AWS to work so we were left with not much time to experiment.

## Exploration vs Exploitation
A major challenge of learning in continuous action spaces is exploration. An advantage of off-policies algorithms such as DDPG is that we can treat the problem of exploration independently from the learning algorithm. As suggested from the Deep Mind paper ([_Continuous Control with Deep Reinforcement Learning_][ddpg-paper]) and from Udacity lessons, a suitable random process to use is the Ornstein-Uhlenbeck process which adds a certain amount of noise to the action values at each timestep. This noise is correlated to previous noise, and therefore tends to stay in the same direction for longer durations without canceling itself out. This allows the arm to maintain velocity and explore the action space with more continuity. Therefore an exploration policy µ is constructed by adding noise sampled from a noise process N to our actor policy

µ(st) = µ(st|θµt) + N

where N can be chosen to suit the environment.

You can find the Ornstein-Uhlenbeck process implemented `OUNoise()` class in `maddpg_agent.py` of the source code. Please find the source code [here](https://github.com/MatteoJohnston/deepRL-Collab_compet-p3/blob/master/maddpg_agent.py#L166).


The Ornstein-Uhlenbeck process itself has three hyper parameters that determine the noise characteristics and magnitude:

mu: the long-running mean --> 0
theta: the speed of mean reversion --> 0.15
sigma: the volatility parameter --> 0.2

We haven't experimented with the parameters. 

After many experiments we opted for adding couple of extra parameters:
epsilon: the long-running mean --> 5
epsilon_decay: the speed of mean reversion --> 1e-4

This decay mechanism ensures that more noise is introduced earlier in the training process (i.e., higher exploration), and the noise decreases over time as the agent gains more experience (i.e., higher exploitation).

For more information please read here.
[ddpg-blog]: http://reinforce.io/blog/introduction-to-tensorforce/

Again even in this case we didn't have too much to experiment on those parameters so we just used the ones above. However we found that adding the epsilon and epsilon decay hyper parameters massively improved the performance of our agents.

Please find their respective implementations: [epsilon](https://github.com/MatteoJohnston/deepRL-Collab_compet-p3/blob/master/maddpg_agent.py#L90) and [epsilon decay](https://github.com/MatteoJohnston/deepRL-Collab_compet-p3/blob/master/maddpg_agent.py#L150)  

As you can notice epsilon has been set at 5 and with a decay of 1e-4. This is to jump-start the initial episodes whereby the model learn too slowly. We found the sweet spot to be around 5 and in particular beyond 5.5 becomes too aggressive, hence the maximum score remains very low.

To speed up the learning we had to increase 

tau --> 1e-2

the soft update parameters. the aim it has been to encourage an aggressive exploratory behavior at early stages which happens to speed up the learning process. However just by changing the epsilon to 5.5 or to 4 can lead to different results. The model however still converges but it normally takes more episodes.

##Learning Interval
The greater the number of parameters the more difficult is to precisely pin point which ones really make the difference. We didn't have a lot of time to experiment with the learning intervals. Hence we left them both at 1.

LEARN_EVERY = 1        # learning time step interval
LEARN_NUM = 1          # number of learning passes


##Learning stability: gradient clipping and batch normalization

We implemented gradient clipping set it at 1, therefore placing an upper limit on the size of the parameter updates, and preventing them from growing exponentially. Gradient clipping has been explained during the coursework and documented in the papers quoted as well. It was also present in the base code we used. We didn't try to change this hyper parameter. You can find its implementation [here](https://github.com/MatteoJohnston/deepRL-Collab_compet-p3/blob/master/maddpg_agent.py#L72). 

Along with this, we implemented batch normalization achieving higher model stability after a certain number of episodes and rapidity. We added it both for the [actor](https://github.com/MatteoJohnston/deepRL-Collab_compet-p3/blob/master/model.py#L29) and for the [critic](https://github.com/MatteoJohnston/deepRL-Collab_compet-p3/blob/master/model.py#L68).

In principle we could have applied to every other layer beyond the first one but it would have slowed the learning time. Both those features are essential for solving this challenging environment.

##Experience Replay

Learning from past experiences is an essential part reinforcement learning. As with DQN in the previous project, DDPG also utilizes a replay buffer to gather experiences from each agent. Each experience is stored in a replay buffer as the agent interacts with the environment. In this project, there is one central replay buffer utilized by all 20 agents, therefore allowing agents to learn from each others' experiences.

The replay buffer contains a collection of experience tuples with the state, action, reward, and next state (s, a, r, s'). Each agent samples from this buffer as part of the learning step. Experiences are sampled randomly, so that the data is uncorrelated. This prevents action values from oscillating or diverging catastrophically, since a naive algorithm could otherwise become biased by correlations between sequential experience tuples.

Please find its implementation [here](https://github.com/MatteoJohnston/deepRL-Collab_compet-p3/blob/master/maddpg_agent.py#L195).


## Results 

Please see our results. In fairness we didn't have a lot of time to experiment further we deem our solution to be satisfactory.
<img src="Final results.PNG" align="bottom-left" alt="" title="Final results" />



## Further Enhancements

- **Add *prioritized* experience replay** &mdash; Rather than selecting experience tuples randomly, prioritized replay selects experiences based on a priority value that is correlated with the magnitude of error.



