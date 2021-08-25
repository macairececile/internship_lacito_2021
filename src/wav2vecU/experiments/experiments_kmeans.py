from math import sqrt, log

import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import argparse
import sys
from sklearn import mixture
import faiss
import fairseq
import pandas as pd
import os
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def get_parser():
    parser = argparse.ArgumentParser(description="experiments on kmeans")
    # fmt: off
    parser.add_argument('--feat_path', help='location of tsv files')
    parser.add_argument('--n', type=int, help='number of components')
    return parser


def apply_pca(feats):
    d = feats.shape[-1]
    print("Computing PCA")
    pca = faiss.PCAMatrix(d, 512)
    pca.train(feats)
    x = pca.apply_py(feats)
    return x


def bic_aic_criterion(data):
    data = apply_pca(data)
    print(data.shape)
    # n_components = [400]
    n_components = np.arange(200, 250)
    aics = []
    bics = []
    for n in n_components:
        print(n)
        model = mixture.GaussianMixture(n, max_iter=1000, covariance_type='full', random_state=0).fit(data)
        aics.append(model.aic(data))
        bics.append(model.bic(data))
    criterions = pd.DataFrame({'BIC': bics, 'AICS': aics})
    criterions.to_csv('criterions_200_250.csv', index=False, sep='\t')


def bic_calculation_trial(feats):
    data = apply_pca(feats)
    BIC = []
    n = data.shape # 3100 #number of datapoints
    for k in range(1, 50):
        RSS = 0 #residual sum of squares
        d = 512
        kmeans = faiss.Kmeans(
            d,
            k,
            niter=50,
            verbose=True,
            spherical=True,
            max_points_per_centroid=feats.shape[0],
            gpu=True,
            nredo=3,
        )
        kmeans.train(data)
        for i in range(1, n):
            RSS = RSS + sqrt((data(i, 1) - C(kmeans.centroids(i), 1)) ^ 2 + (data(i, 2) - C(kmeans.centroids(i), 2)) ^ 2)
            BIC.append(n*log(RSS/n)+(k*512)*log(n))
    # [p, l] = min(BIC)


def plot_criterions(csv_file, n):
    n_components = np.arange(1, n)
    n_components = np.append(n_components, [250, 300, 350, 400, 450])
    data = pd.read_csv(csv_file, sep='\t')
    bic_results = data['BIC'].tolist()
    aic_results = data['AICS'].tolist()
    print(bic_results)
    plt.plot(n_components, aic_results, label='BIC')
    plt.plot(n_components, bic_results, label='AIC')
    plt.legend(loc='best')
    plt.xlabel('n_components')
    plt.show()


def plot_clusters(feats):
    d = feats.shape[-1]
    pca = faiss.PCAMatrix(d, 512)
    pca.train(feats)
    print("Applying PCA")
    x = pca.apply_py(feats)
    d = 512
    kmeans = faiss.Kmeans(
        d,
        300,
        niter=50,
        verbose=True,
        spherical=True,
        max_points_per_centroid=feats.shape[0],
        gpu=True,
        nredo=3,
    )
    kmeans.train(x)
    centroids = kmeans.centroids
    feats_select = x[: 10000, :]
    labels_ = kmeans.index.search(x=feats_select, k=1)[1].reshape(-1)
    # with open(clusters, 'r') as f:
    #     lines = f.readlines()
    #     clusters = list(map(int, lines[0].split(' ')))
    # labels = np.array(clusters)
    # print(labels)
    u_labels = np.unique(labels_)
    colors = np.arange(u_labels.size)
    for i in u_labels:
        plt.scatter(feats_select[labels_ == i, 0], feats_select[labels_ == i, 1], c=colors, cmap='plasma')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", color='r')
    plt.legend(labels=labels_)
    plt.show()


def main():
    parser = get_parser()
    args = parser.parse_args()
    # Getting the Centroids
    # centroids = np.load(osp.join(args.path, "centroids.npy"))
    # print("Loaded centroids", centroids.shape, file=sys.stderr)
    feats = np.load(args.feat_path + ".npy")
    # print(feats)
    # bic_aic_criterion(feats)
    plot_clusters(feats)
    # plot_criterions('criterions.csv', args.n)


if __name__ == "__main__":
    main()
