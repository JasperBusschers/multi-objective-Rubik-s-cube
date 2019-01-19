# multi-objective-Rubik-s-cube
In this repo I will be using the Rubik's cube environment to experiment with multi objective optimisation for the course multi agent learning seminar.
The general problem of solving a rubix cube can be seen as a single objective optimisation problem, in this formulation we try to find a solution in a statespace of 4x2x10^19 different possible states.

In my experiment I wish to split this objective into 6 objectives, each being colloring 1 side correctly.
For each of these objectives I train a RL agent and record their actions for many random starting states.

Then I train an new agent using generative adversarial imitation learning using a discriminator for each objective.

