# experiment 1

In the first experiment I try to find an efficient way for solving the 1 of the multiple objectives in the rubik's cube.
This objective is coloring 1 predefined face of the cube all the same color.

In this experiment we compare the average reward of 2 different state sizes.
In the first run we let the agent observe the whole rubiks cube to make its decision.
While in the second run, the agent could only see the area he was trying to color.

This causes the state space to dramaticly decrease, but also gives the agent an uncomplete representation of the state.

We ran both experiments for 2000 runs, with the only differance being some changes in the network to let it support the smaller size.



### different reward funtions
The experiments are performed using 2 different reward schemes, the first being :
if the goal face is colored : reward = 100
if step_size > max : reward is amount of same color blocks in goal face
else reward = 0


And the second reward scheme is :
if the goal face is colored : reward = 100
else reward = 0


