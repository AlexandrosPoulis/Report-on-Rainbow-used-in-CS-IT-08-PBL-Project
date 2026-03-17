[Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)
---
- Rainbow is a combination of aspects from the following algorithms:
	- DQN(Deep Q-Networks)
	- DDQN(Double DQN)
	- A3C
	- Distributional Q-learning
	- Noisy DQN
	- Prioritized DDQN
	- Dueling DDQN
	
- Background information on Deep reinforcement learning and DQN
	- The base of Rainbow, DQN is a combination of deep networks with reinforcement learning through a convolutional neural network to approximate values for a state $S$<sub>t</sub>(that being the input to the neural network). There is also a replay memory buffer that holds the last million transitions, those being information about the action taken in that step ($S$<sub>t</sub>, A<sub>t</sub>, $R$<sub>t+1</sub>, $γ$<sub>t+1</sub>, $S$<sub>t+1</sub>). The timestep $t$ is a random timestep taken from that buffer, in this case with the last million transitions. The algorithm utilizes a stochastic gradient descent for the minimization of the loss:  
		![Pasted image 20260317174149.png](https://github.com/AlexandrosPoulis/Report-on-Rainbow-used-in-CS-IT-08-PBL-Project/blob/main/Images-used/Pasted%20image%2020260317174149.png)
		 - $t$ is the timestep picked randomly from the replay memory buffer
		 - $S$<sub>t</sub> is the state for the timestep t
		 - $A$<sub>t</sub> or a is the action taken for the timestep t ($a'$ is the $A$<sub>t+1</sub>)
		 - $γ$<sub>t</sub> is the discount value for the timestep t
		 - $R$<sub>t</sub> is the reward for the timestep t
		 - $p$<sub>θ</sub>($S$<sub>t</sub>, $A$<sub>t</sub>) is the state-action pair for the timestep t
		 - "The gradient of the loss is back-propagated only into the parameters $θ of the online network"(Not sure how to write this without doing a one for one copy paste)
		 
- Rainbow is essentially a combination of the following extensions made to DQN
	- Double Q-learning
		- Q-learning has a negative effect from overestimation bias. That is due to the maximization step shown in the Equation (1) above and can negatively affect learning.
		- The way to reduce that is using the following loss function:
			![Pasted image 20260317180354.png](https://github.com/AlexandrosPoulis/Report-on-Rainbow-used-in-CS-IT-08-PBL-Project/blob/main/Images-used/Pasted%20image%2020260317180354.png)
	- Prioritized replay
		- This is basically a way to chose from the replay buffer not randomly but focus more on transitions that we can learn a more from.
		- To achieve this prioritized experience replay was used and it samples transitions with probability $p$<sub>t</sub> relative to the last encountered absolute TD error with $ω$ being a hyper-parameter that determines the distribution. The latest additions to the buffer will have the maximum priority
				![Pasted image 20260317181036.png](https://github.com/AlexandrosPoulis/Report-on-Rainbow-used-in-CS-IT-08-PBL-Project/blob/main/Images-used/Pasted%20image%2020260317181036.png)
	- Dueling networks
		- They were designed for value based RL. It includes 2 streams of computations, value and advantage streams and through the following factorization of action values they are encoded and merged
			![Pasted image 20260317181552.png](https://github.com/AlexandrosPoulis/Report-on-Rainbow-used-in-CS-IT-08-PBL-Project/blob/main/Images-used/Pasted%20image%2020260317181552.png)
		- $ξ$, $η$ and $ψ$ are the parameters of $f$<sub>ξ</sub>
		- $f$<sub>ξ</sub> is the shared encoder of $v$<sub>η</sub> and $α$<sub>ψ</sub> 
		- $v$<sub>η</sub> is the value stream
		- $α$<sub>ψ</sub> is the advantage stream
		- $θ$ is the concatenation of $ξ$, $η$, $ψ$
	- Multi-step learning
		- Instead of the traditional single reward and action used for the bootstrapping where focus is only on the very next action, multi-step learning is taking into account the reward for the next n-steps, for example if action $a$ is good for the next step but action $b$ has worst reward for the next immediate step, $a$ would be picked in a traditional setting. In this case if action $b$ was followed by other high reward actions where $a$ had worst rewards for the following steps $b$ could be picked depending on those values.
		- This reward is given by the following function
					![Pasted image 20260317183026.png](https://github.com/AlexandrosPoulis/Report-on-Rainbow-used-in-CS-IT-08-PBL-Project/blob/main/Images-used/Pasted%20image%2020260317183026.png)
		 - The loss function in the case that multi-step learning is applied to the traditional DQN is the following
				 ![Pasted image 20260317183223.png](https://github.com/AlexandrosPoulis/Report-on-Rainbow-used-in-CS-IT-08-PBL-Project/blob/main/Images-used/Pasted%20image%2020260317183223.png)
	- Distributional RL([A small post with some basic explanations](https://www.shadecoder.com/topics/distributional-rl-a-comprehensive-guide-for-2025))
		- Instead of just trying to predict the expected return, Distributional RL focuses on the distributions(range) of returns. $Z$ is a vector defined by $z^i$ atoms: 
						![Pasted image 20260317185251.png](https://github.com/AlexandrosPoulis/Report-on-Rainbow-used-in-CS-IT-08-PBL-Project/blob/main/Images-used/Pasted%20image%2020260317185251.png)
		 The approximation of the distribution $d$<sub>t</sub> = ($z$, $p$<sub>θ</sub>($S$<sub>t</sub>, $A$<sub>t</sub>)) has a probability mass of $p$<sub>θ</sub>($S$<sub>t</sub>, $A$<sub>t</sub>) for every value $i$ of the vector with the goal to find a $θ$ that matches the actual distribution of returns. To that end return distributions must satisfy a variant of Bellman's equation, "for a given State $S$<sub>t</sub> and action $A$<sub>t</sub>, the distribution of returns under optimal policy $π^*$ should match a target distribution defined by taking the distribution for the next state $S$<sub>t+1</sub> and action $a^*$<sub>t+1</sub> = $π^*$($S$<sub>t+1</sub>), contracting it towards zero according to the discount, and shifting it by the rewards" (This is word for word how they describe the process). Then the last steps are constructing a new support for the target distribution and minimizing the Killbeck-Leibler divergence (also called also called **relative entropy** and **I-divergence**) between distribution $d$<sub>t</sub> and target $d'$:
						![Pasted image 20260317193335.png](https://github.com/AlexandrosPoulis/Report-on-Rainbow-used-in-CS-IT-08-PBL-Project/blob/main/Images-used/Pasted%20image%2020260317193335.png)
		 - Killbeck-Leibler divergence is in essence the distance of an approximation distribution from the true probability distribution
		 - $Φ$ is a [L2-projection](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/software/featflow2/tutorial/tutorial_l2proj.html) of the target distribution onto $z$
		 -  ![Pasted image 20260317193536.png](https://github.com/AlexandrosPoulis/Report-on-Rainbow-used-in-CS-IT-08-PBL-Project/blob/main/Images-used/Pasted%20image%2020260317193536.png) is the greedy action
		 - ![Pasted image 20260317193617.png|250](https://github.com/AlexandrosPoulis/Report-on-Rainbow-used-in-CS-IT-08-PBL-Project/blob/main/Images-used/Pasted%20image%2020260317193617.png) is the mean action values for state $S$<sub>t+1</sub>
	- Noisy Nets
		- The main goal of this component is to incorporate noise to drive exploration. That is done by the addition of a noisy linear layer to the network 
				![Pasted image 20260317194211.png](https://github.com/AlexandrosPoulis/Report-on-Rainbow-used-in-CS-IT-08-PBL-Project/blob/main/Images-used/Pasted%20image%2020260317194211.png)
		- $ε^b$ and $ε^w$ being random values
		- ![Pasted image 20260317194317.png](https://github.com/AlexandrosPoulis/Report-on-Rainbow-used-in-CS-IT-08-PBL-Project/blob/main/Images-used/Pasted%20image%2020260317194317.png) "denotes the element-wise product"
		- The above replaces the standard linear $y = b + Wx$ and over time the noised is ignored by the network and that allows state-conditional exploration 
		
- The implementation of the above in Rainbow is done by the following:
	- Since they are using the multi-step variant the target distribution is the following:
					![Pasted image 20260317194928.png](https://github.com/AlexandrosPoulis/Report-on-Rainbow-used-in-CS-IT-08-PBL-Project/blob/main/Images-used/Pasted%20image%2020260317194928.png)
	 and the loss is:
							![Pasted image 20260317194954.png](https://github.com/AlexandrosPoulis/Report-on-Rainbow-used-in-CS-IT-08-PBL-Project/blob/main/Images-used/Pasted%20image%2020260317194954.png)
	 - Then the loss is combined with double Q-learning and the action is evaluated on the target network, that action being the one selected with the use of the bootstrap.
	 - Next for the prioritized replay they deemed that through experimentation the prioritized transitions were made using the KL loss(Kullbeck-Leibler) and the algorithm thus tries to minimize the following:
						![Pasted image 20260317195803.png](https://github.com/AlexandrosPoulis/Report-on-Rainbow-used-in-CS-IT-08-PBL-Project/blob/main/Images-used/Pasted%20image%2020260317195803.png)
	- As for the network used they did indeed use a dueling network architecture adapted for use with return distributions. To explain the process in few words, the value and advantage streams are aggregated and passed through a SoftMax layer and later used to estimate the return's distributions:
					![Pasted image 20260317200228.png](https://github.com/AlexandrosPoulis/Report-on-Rainbow-used-in-CS-IT-08-PBL-Project/blob/main/Images-used/Pasted%20image%2020260317200228.png)
	- Finally they replace all linear layers with the noisy ones mentioned above and the noise chosen for those layers is Gaussian noise
	
- Hyper-parameter tuning for the paper:
	- In the DQN family of algorithms no learning updates were performed in the first 200k frames. That is to ensure "sufficiently uncorrelated updates". If we take into account prioritized replay that number can go lower to around 80k frames
	- DQP has a starting exploration value $ε$ of 1, dropping to as low as 0.01. When using Noisy Nets they started with a greedy value for $ε$ = 0, with the hyperparameter $σ$<sub>0</sub>, used in initializing the weights, being at a value of 0.5 for the "noisy stream" (if generating noise on a cpu instead of a gpu the value $σ$<sub>0</sub> can be lowered to 0.1).
	- For the learning rate DQN uses a rate of $a$ = 0.00025. In Ranbow variants they used a learning rate of $a/4$ and $1.5 \times 10^{-4}$ for $ε$ hyper-parameter for the Adam optimizer used in the paper (different from the exploration $ε$).
	- As for replay prioritization the recommended proportional variant was used with priority exponent $ω$ of 0.5 and linearly increased exponent $β$ from 0.4 to 1.
	- For multi-step learning the hyperparameter $n$ was very sensitive and a value of 3 was chosen as it performed a but better than the next best value of 5.
	- Finally they did mention that the above hyperparameters where used for all 57 games they tested and it performed well in all games, also mentioning that and I quote "the Rainbow agent really is a single agent setup that performs well across all the games"
					  ![Pasted image 20260316232734.png](https://github.com/AlexandrosPoulis/Report-on-Rainbow-used-in-CS-IT-08-PBL-Project/blob/main/Images-used/Pasted%20image%2020260316232734.png)
				  
- Analysis of results
	- They compare first Rainbow to other agents(A3C, DQN, DDQN, Prioritized DDQN, Dueling DDQN, Distributional DQN, and Noisy DQN) having the algorithm favorably compare to them and then they perform ablation studies on Rainbow.
	- According to them Rainbow achieves a medians score of 223% in no-ops regime and in human starts regime they measured a median of 153%
			![Pasted image 20260316234235.png](https://github.com/AlexandrosPoulis/Report-on-Rainbow-used-in-CS-IT-08-PBL-Project/blob/main/Images-used/Pasted%20image%2020260316234235.png)
	- In the ablation study, prioritized replay and multi-learning were the most crucial components, resulting in a large drop in median performance if one was removed, with early performance being dropped for any of the 2 removed components, with multi-learning also hurting the final performance. Following these 2 was Distributional Q-learning with no apparent difference for the first 40 million frames BUT after that point the performance started lagging behind when not used. For median performance including Noisy Nets seemed to be best with most games having a large performance drop when not included but also some games had a small increase in that same case. Dueling network seemed like it didn't really change much but in some games, specifically with above-human performances it performed better and it performed worse in sub-human performances. Lastly double Q-learning showed small differences in median performance but the researchers noted that if support of distributions is expanded Q-learning might have a bigger importance

Terms:
- bootstrap: Unlike Monte Carlos bootstrapping updates the estimated values of states and actions based on existing value estimates and not needing to wait for the final outcome. Simple explanation "current knowledge to improve future predictions" taken from [here](https://zilliz.com/ai-faq/what-is-bootstrapping-in-rl)
- [ablation studies](https://en.wikipedia.org/wiki/Ablation_(artificial_intelligence)): removal of a component from an AI system with the goal of measuring the impact each component has to the system as  a whole.
- For the above-human and sub-human performances, they didn't explain it at all but my interpretation is that, above-human means that the algorithm had a performance that was better than a human one and the sub-human means the exact opposite.
- no-ops: is the act of starting the level with some random amount of actions where the algorithm does nothing potentially resulting in enemies to move or the state of the level in general to change and the purpose of that is to try to remove the option for the algorithm to just remember the level and do the same actions in the same order every time it sees it since if any enemies have moved doing the same action will result in failed runs, more info [here](https://medium.com/data-science/paper-repro-deep-neuroevolution-756871e00a66)
- human starts: is the act of starting the level not from the start but from a state reached through human gameplay and in that gameplay you record the states and train the algorithm using those, this is done to  again prevent memorization since that state will likely be a unique one not reached through the training and to try to make the algorithm handle a wide range of  situations even ones it might not had seen with its training alone, more info [here](https://medium.com/data-science/paper-repro-deep-neuroevolution-756871e00a66)

---
