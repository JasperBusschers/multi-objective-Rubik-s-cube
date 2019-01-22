# multi-objective-Rubik-s-cube
In this repo I will be using the Rubik's cube environment to experiment with multi objective optimisation for the course multi agent learning seminar.
The general problem of solving a rubix cube can be seen as a single objective optimisation problem, in this formulation we try to find a solution in a statespace of 4x2x10^19 different possible states.

In my experiment I wish to split this objective into 6 objectives, each being colloring 1 side correctly.
For each of these objectives I train a RL agent and record their actions for many random starting states.

Then I train an new agent using generative adversarial imitation learning using a discriminator for each objective.

### OBJECTIVES:
   - implement environment and agent : DONE
   - implement network and learning loop : DONE
   - figure out reward scheme that solves 1 face of the cube : TODO
   - train agents to solve each side of the cube : TODO
   - record all agents actions : TODO
   - implement GAIL for single objective : TODO
   - train agent using 2 discriminators in order to color 2 faces correct : TODO
   - record the new joint policy :TODO
   - train a new agent to imitate the joint objective and also another objective : TODO
   - evaluate results : TODO


### alternative objective
reproducing : https://arxiv.org/abs/1805.07470 with smaller objectives
approach makes use of autodidactical iterations and monte carlo three search

https://github.com/RobotVeraDS/deepcubes c++ to python

define neural network f(s) -> (v,p)

with V being value of that state and P the probability for each action.

for each training input
   - start from a solved cube
   - scramble m times
   - give generated state to f(s)
   - perform dept 1 BFS
   - evaluate value of each child
   - update value to the maximum value of the children, and policy to the action leading to that value

   

### experiment reference
   - experiment 1 : evaluate mean reward using different size state space, and naive vs complex reward function
   - experiment 2 : testing even smaller objectives



### notes to self
- maybe try Q networks without padding, or fully connencted.
- For gail dataset, generate every possible goal path for length 2 or 3
