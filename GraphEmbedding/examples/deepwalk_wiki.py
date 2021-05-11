
import numpy as np

from ge.classify import read_node_label, Classifier
from ge import DeepWalk
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE


def evaluate_embeddings(embeddings):
    X, Y = read_node_label('../data/wiki/wiki_labels.txt')
    tr_frac = 0.8
    print("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)


def plot_embeddings(embeddings,):
    X, Y = read_node_label('../data/wiki/wiki_labels.txt')

    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # G = nx.read_edgelist('../data/wiki/Wiki_edgelist.txt',
    #                      create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    # nx.draw(G, node_size=10, font_size=10, font_color="blue", font_weight="bold")
    # plt.show()
    import pandas as pd
    df = pd.DataFrame()
    df['source'] = [str(i) for i in [0,1,2,3,4,4,6,7,7,9]]
    df['target'] = [str(i) for i in [1,4,4,4,6,7,5,8,9,8]]

    G = nx.from_pandas_edgelist(df,create_using=nx.Graph())

    model = DeepWalk(G, walk_length=50, num_walks=180, workers=1)
    model.train(window_size=10, iter=3,embed_size=2)
    # model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)
    # model.train(window_size=5, iter=3,embed_size=128)
    embeddings = model.get_embeddings()
    #print(embeddings)
    x,y = [],[]
    print(sorted(embeddings.items(), key=lambda x: x[0]))
    for k,i in embeddings.items():
        x.append(i[0])
        y.append(i[1])
    plt.scatter(x,y)
    plt.show()

    # evaluate_embeddings(embeddings)
    # plot_embeddings(embeddings)
