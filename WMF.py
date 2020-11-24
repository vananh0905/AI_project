import numpy as np
import pandas as pd
import os.path
import utils


class WeightedMF:
    """
    Do weighted matrix factorization by gradient descent
    Argument:
        - P, C: P, C matrices
        - dict_user, dict_item: mapping between name and index of users and items
        - depth: number of latent features. Default: 5
        - n_epoches: number of epoches to train. Default: 200
        - lr: learning rate. Default: 1e-6
        - rgl: regularization value (lambda). Default: 0.02
        - graph_inferred: if True, implicit feedback is filled by neighborhood users (useful for sparse P matrix). Default: False
        - early_stopping: if True, stop early in gradient descent when loss may not improve; otherwise, train to the last epoch. Default: True
        - verbose: if True, print out the information of process. Default: False
    Methods:
        - fit(): training model with gradient descent
        - predict(): return list of recommendations for a user given the user_id and number of items to recommend
        - save(), load(): save and load U, I if they have been calculated already
    """
    def __init__(self, P, C, dict_user, dict_item, optimizer, depth=5, lr=1e-6, rgl=0.02, batch_size=0, graph_inferred=False, early_stopping=True, verbose=False):
        self.P = P
        self.C = C
        self.n_users = P.shape[0]
        self.n_items = P.shape[1]
        self.depth = depth
        self.dict_user = dict_user
        self.dict_item = dict_item
        self.U = np.random.rand(self.n_users, self.depth)
        self.I = np.random.rand(self.depth, self.n_items)
        self.predict = np.zeros_like(P)
        self.rgl = rgl
        self.lr = lr
        self.graph_inferred = graph_inferred
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.optimizer = optimizer
        if optimizer == 'sdg':
            self.n_epoches = 20
        else:
            self.n_epoches = 200
        if batch_size == 0:
            self.batch_size = int(self.n_items * self.n_users / 100)

    def __gd(self):
        E_U = np.ones((1, self.depth), dtype=float)
        E_i = np.ones((self.depth, 1), dtype=float)
        dU = np.zeros_like(self.U)
        dI = np.zeros_like(self.I)
        for i in range(self.n_users):
            for j in range(self.n_items):
                dU[i] += 2 * self.C[i, j] * (np.dot(self.U[i, :], self.I[:, j]) - self.P[i, j]) \
                         * np.dot(E_U, self.I[:, j])
            dU[i] += 2 * self.rgl * self.U[i]
            # dU[i] = normalize.sum_normalize(dU[i])

        for j in range(self.n_items):
            for i in range(self.n_users):
                if self.P[i, j] == 0:
                    continue
                dI[:, j] += 2 * self.C[i, j] * (np.dot(self.U[i, :], self.I[:, j]) - self.P[i, j]) \
                            * np.dot(self.U[i, :], E_i)
            dI[:, j] += 2 * self.rgl * self.I[:, j]
            # dI[:, j] = normalize.sum_normalize

        self.U -= self.lr * dU
        self.I -= self.lr * dI

    def __sgd(self):
        E_U = np.ones((1, self.depth), dtype=float)
        E_i = np.ones((self.depth, 1), dtype=float)

        dU = np.zeros_like(self.U)
        dI = np.zeros_like(self.I)
        X = []
        Y = []
        mark = np.zeros_like(self.C)

        n_steps = int(self.n_items * self.n_users / self.batch_size)
        for i in range(n_steps):
            index = 0
            while index < self.batch_size:
                x = np.random.randint(self.n_users)
                y = np.random.randint(self.n_items)
                if mark[x, y] == 0:
                    mark[x, y] = 1
                    X.append(x)
                    Y.append(y)
                    index += 1
            for u in range(self.batch_size):
                i = X[u]
                j = Y[u]
                dU[i] += 2 * self.C[i, j] * (np.dot(self.U[i, :], self.I[:, j]) - self.P[i, j]) \
                             * np.dot(E_U, self.I[:, j])
                dU[i] += 2 * self.rgl * self.U[i]

                dI[:, j] += 2 * self.C[i, j] * (np.dot(self.U[i, :], self.I[:, j]) - self.P[i, j]) \
                                * np.dot(self.U[i, :], E_i)
                dI[:, j] += 2 * self.rgl * self.I[:, j]
            self.U -= self.lr * dU
            self.I -= self.lr * dI

    def fit(self):
        loss = 0
        for current_epoch in range(self.n_epoches):
            prev_loss = loss
            loss = 0
            if self.verbose:
                print("Processing epoch {}".format(current_epoch))
            for i in range(self.n_users):
                for j in range(self.n_items):
                    loss += self.C[i, j] * (np.dot(self.U[i, :], self.I[:, j]) - self.P[i, j]) ** 2
            loss = loss + self.rgl * (np.sum(self.U ** 2) + np.sum(self.I ** 2))
            print("Loss at epoch {}: {:.3f}".format(current_epoch+1, loss))
            if self.early_stopping and current_epoch > 1 and (prev_loss - loss < 0.001):
                return None
            else:
                if self.optimizer == 'gd':
                    self.__gd()
                else:
                    self.__sgd()

        # # Normalize 0~1
        # for i in range(self.n_users):
        #     self.U[i] = utils.softmax_normalize(self.U[i])
        # for i in range(self.depth):
        #     self.I[i] = utils.softmax_normalize(self.I[i])
        # self.predict = np.dot(self.U, self.I)

    def get_recommendations(self, user_index, n_rec_items):
        recommendations = np.argsort(self.predict[user_index])
        recommendations = recommendations[-n_rec_items:]
        # name_of_songs_rec = []
        # for i in range(n_rec_items):
        #     name_of_songs_rec.append(self.dict_item[recommendations[i]])
        # return name_of_songs_rec
        return recommendations

    # def evaluate(self, user_index):
    #     sum_of_deleted_feedback = 0
    #     true_positive = 0
    #     name_of_songs_res, recommendations = self.predict()
    #     R_train, R_full = self.load_train_file()
    #     for i in range(self.n_items):
    #         if R_train[user_index, i] != R_full[user_index, i]:
    #             sum_of_deleted_feedback += 1
    #             if i in recommendations:
    #                 true_positive += 1
    #     return float(true_positive/sum_of_deleted_feedback) * 100

    def save(self):
        np.savetxt('./data/U.txt', self.U, delimiter=' ', fmt='%.5f')
        np.savetxt('./data/I.txt', self.I, delimiter=' ', fmt='%.5f')
        np.savetxt('./data/predict.txt', self.predict, delimiter=' ', fmt='%.5f')

    def load(self):
        if os.path.isfile('./data/U.txt'):
            with open('./data/P.txt', 'r') as f:
                self.U = [[float(num) for num in line[:-1].split(' ')] for line in f]
                self.U = np.array(self.U)
        if os.path.isfile('./data/I.txt'):
            with open('./data/C.txt', 'r') as f:
                self.I = [[float(num) for num in line[:-1].split(' ')] for line in f]
                self.I = np.array(self.I)

