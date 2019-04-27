import visdom
import numpy as np
import json

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n



datassss = ['0.97','0.98','0.99','0.999']

for gamma in datassss:
    data = json.load(open("dqn_hist/dqn_{}".format(gamma),"r"))
    data = np.array(data).T
    X = data[0][99:]
    Y = moving_average(data[1], 100)
    viz = visdom.Visdom(port=8097, username="hortune", password="enutroh", env="ADL HW3")


    viz.line(
        Y=Y,
        X=X,
        opts=dict(  markers=False,
                    title="DQN @ Assault",
                    ylabel="Reward (moving average 100)",
                    xlabel="steps",
                    showlegend= True,
                    width= 700,
                    height= 350),
        win= 'alahua8888888',
        update= 'append',
        name= gamma,
    )
