import numpy as np
import visdom
datas = []
for idx, data in enumerate(open("log.csv")):
    if idx == 0:
        continue
    datas.append(list(map(float,data.split(','))))

viz = visdom.Visdom(port=8097, username="hortune", password="enutroh")

datas = list(zip(*datas))
x = np.arange(1,11)
viz.line(
    Y=np.column_stack(datas),
    X=np.column_stack((x, x, x, x)),
    opts=dict(  markers=False,
                title="ELMo model train / dev accuracy and loss",
                ylabel="Loss",
                xlabel="Epoch",
                legend= ['train CrossEntropy Loss', 
                         'train Accuracy',
                         'test CrossEntropy Loss',
                         'test Accuracy']),
)
