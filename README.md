# Classic Tabular Q-Learning with Self-play
X and O trained simultaneously by playing against each other, 1 million times

### The training algorithm:


```
For each episode:
  #X's choice
  X chooses a_t from state s_x-turn using e-greedy, producing x_reward, s_o-turn'

  #update O
  if not first turn: 
    Q(s_o-turn,a) = Q(s_o-turn,a) + lr * (o_reward + gamma*max_a'Q(s_o-turn',a') - Q(s_o-turn,a))

  #O's choice
  O chooses a_t from state s_o-turn' using e-greedy, producing o_reward, s_x-turn'

  #update X
  Q(s_x-turn,a) = Q(s_x-turn,a) + lr * (x_reward + gamma*max_a'Q(s_x-turn',a') - Q(s_x-turn,a))
```

### Experiments:
* Reward both X and O +1 for winning
* In addition, reward O +0.5 for a tie
<img src="stats/curves.png" alt="Image description" width="150%" height="150%">


### Epsilon Decay Schedule
<img src="stats/epsilon.png" alt="Image description" width="49%" height="49%">
