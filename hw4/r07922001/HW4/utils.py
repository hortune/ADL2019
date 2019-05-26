import numpy as np
from scipy.misc import imread, imresize, imrotate

def load_data():
    root_path = '../selected_cartoonset100k/'
    with open(root_path + 'cartoon_attr.txt') as fd:
        n = int(fd.readline())
        fd.readline()
        filename = []
        attr = []
        images = []
        for line in fd:
            if line.strip():
                line = line.split()
                filename.append(line[0])
                images.append(imread(root_path+'images/'+line[0]))
                attr.append(list(map(int, line[1:])))
    return np.array(images), filename, np.array(attr)

def load_test_data(root_path):
    attr = []
    with open(root_path) as fd:
        n = int(fd.readline())
        fd.readline()
        for i in range(n):
            line = fd.readline()
            if line.strip():
                line = line.split()
                attr.append(list(map(int, line)))
    return np.array(attr)
if __name__ == "__main__":
    images, filenames, attrs = load_data()
    from IPython import embed
    embed()
