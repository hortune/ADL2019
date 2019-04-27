# How to train your model
- `python main.py --train_dqn`
- `python main.py --train_pg`

For the doubleDQN and noisyDQN, please `mv agent_double.py agent_dqn.py` or `mv agent_noisy.py agent_dqn.py` first.

# How to plot the figures in your report
By running above code, they will save history in json file. Put the history path to the list of `plot.py` and open visdom then run it.
