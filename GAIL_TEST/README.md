# implementation of gail for rubiks cube environment
credits for implementation go to https://github.com/nikhilbarhate99/Deterministic-GAIL-PyTorch
with minor modifications to support a different environment.

### data
data was collected, starting from a solved cube and randomly performing n actions on it.

The reverse action is then stored allong with the state as solution.


# experiments
currently experimenting with different step sizes n. To see if I can limit the state space so that a good solution can be found
