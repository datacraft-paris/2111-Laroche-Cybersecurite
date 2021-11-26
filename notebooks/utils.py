import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import shap

from sklearn.linear_model import LassoCV
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.tree import DecisionTreeRegressor



def anomaly_score_interpretation_surrogate(
    X, y_pred, names=None, seed=None, algo='dt'):
    if algo == 'lasso':
        lasso = LassoCV(random_state=seed).fit(X, y_pred)
        w = lasso.coef_
    else:
        dt = DecisionTreeRegressor(random_state=seed).fit(X, y_pred)
        w = dt.feature_importances_
    fig = plt.figure()
    plt.imshow(w[np.newaxis,:], cmap='coolwarm')
    ax = plt.gca()
    ax.set_yticks([])
    ax.set_xticks(np.arange(len(w)))
    if names is not None:
        ax.set_xticklabels(names, rotation=45, ha='right')
    for i in range(len(w)):
        ax.text(i, 0, '{0:.2f}'.format(w[i]), ha='center', va='center')
    fig.set_size_inches(8, 2)
    plt.show()

def anomaly_score_interpretation(
    X, model, names=None):
    explainer = shap.Explainer(model, feature_names=names)
    shap_val = explainer(X)
    shap.plots.beeswarm(shap_val)

def evaluate_algo(y_true, y_pred, w=9, h=4, names=None):
    fig = plt.figure()
    axes = fig.subplots(1, 2)
    if y_pred.ndim == 1:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        axes[0].plot(fpr, tpr, label='AUC={0:.3f}'.format(auc(fpr, tpr)))
        pre, rec, _ = precision_recall_curve(y_true, y_pred)
        axes[1].plot(rec, pre, label='AUC={0:.3f}'.format(auc(rec, pre)))
    else:
        for i in range(y_pred.shape[1]):
            fpr, tpr, _ = roc_curve(y_true, y_pred[:,i])
            if names is None:
                lab = 'AUC={0:.3f}'.format(auc(fpr, tpr))
            else:
                lab = '{0} (AUC={1:.3f})'.format(names[i], auc(fpr, tpr))
            axes[0].plot(fpr, tpr, label=lab)
            pre, rec, _ = precision_recall_curve(y_true, y_pred[:,i])
            if names is None:
                lab = 'AUC={0:.3f}'.format(auc(rec, pre))
            else:
                lab = '{0} (AUC={1:.3f})'.format(names[i], auc(rec, pre))
            axes[1].plot(rec, pre, label=lab)
    axes[0].set_title('ROC curve')
    axes[0].set_xlabel('False positive rate')
    axes[0].set_ylabel('True positive rate')
    axes[0].legend()
    axes[1].set_title('PR curve')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].legend()
    fig.set_size_inches(w, h)
    plt.show()

def evaluate_algo_events(
    y_pred, counts_tot, counts_mal,
    w=9, h=4, names=None):
    y_test_events = np.concatenate([
        [0]*(counts_tot[i]-counts_mal[i])
        + [1]*counts_mal[i]
        for i in range(len(counts_tot))
    ])
    if y_pred.ndim == 1:
        y_pred_events = np.concatenate([
            [y_pred[i]]*counts_tot[i]
            for i in range(len(counts_tot))
        ])
    else:
        y_pred_events = np.stack([
            np.concatenate([
                [y_pred[i,j]]*counts_tot[i]
                for i in range(len(counts_tot))
            ]) for j in range(y_pred.shape[1])
        ], axis=1)
    evaluate_algo(y_test_events, y_pred_events, w=w, h=h, names=names)

def plot_examples(
    X, y_true, y_pred, cols, logscale=True, n_instances=10):
    n, d = X.shape
    idx = y_pred.argsort()

    tp = X[idx][y_true[idx]==1][-n_instances:,:]
    fp = X[idx][y_true[idx]==0][-n_instances:,:]
    tn = X[idx][y_true[idx]==0][:n_instances,:]
    fn = X[idx][y_true[idx]==1][:n_instances,:]

    fig = plt.figure()
    axes = fig.subplots(2, 2).flatten()
    titles = [
        'True positives', 'False positives',
        'False negatives', 'True negatives'
    ]
    colors = ['black', 'white']
    for i, x in enumerate([tp, fp, fn, tn]):
        if logscale:
            im = np.log(1+x)
        else:
            im = x
        axes[i].imshow(im, cmap='Greys')
        axes[i].set_title(titles[i])
        for j in range(d):
            for l in range(d):
                axes[i].text(l, j, '{0:.0f}'.format(x[j,l]),
                    ha='center', va='center',
                    color=colors[int(im[j,l]>.4*im.max())])
        if i//2 >= 1:
            axes[i].set_xticks(np.arange(d))
            axes[i].set_xticklabels(cols, rotation=45, ha='right')
        else:
            axes[i].set_xticklabels([])
    fig.set_size_inches(15, 15)
    fig.subplots_adjust(hspace=.08, wspace=.1)
    plt.show()

def plot_graph_examples(
    y_true, y_pred, idents, dir_path, n_instances=4):
    idx = y_pred.argsort()

    tp = np.array(idents['ident'])[idx][y_true[idx]==1][-n_instances:]
    fp = np.array(idents['ident'])[idx][y_true[idx]==0][-n_instances:]
    tn = np.array(idents['ident'])[idx][y_true[idx]==0][:n_instances]
    fn = np.array(idents['ident'])[idx][y_true[idx]==1][:n_instances]

    fig = plt.figure()
    if n_instances % 2 == 0:
        d = n_instances
    else:
        d = n_instances + 1
    axes = fig.subplots(d, 4)
    colors = ['green', 'red', 'red', 'green']
    shapes = ['v', 'o', 'v', 'o']
    for i, names in enumerate([tp, fp, fn, tn]):
        for j, n in enumerate(names):
            ax = axes[(i//2)*(d//2)+j//2, 2*i%4+j%2]
            user, day = n.split('_')
            fp = os.path.join(dir_path, user, '{0}.graphml'.format(day))
            g = nx.readwrite.graphml.read_graphml(fp)
            nx.draw_networkx(
                g, ax=ax,
                node_color=colors[i],
                node_shape=shapes[i])
            ax.set_title(n)
    fig.set_size_inches(16, 16)
    plt.show()
