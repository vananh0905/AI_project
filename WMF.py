import numpy as np
import pandas as pd
import os.path
import normalize


class IFConverter:
    """
    Convert triplets data (user_id, song_id, play_count) into R, P, C matrices
    Arguments:
        - path: path to triplets data
        - alpha (optional): penalty for confidence. Default: 40
    Methods:
        - get_implicit_feedback(): read triplets and calculate R. Return None
        - convert(): convert R into P and C for weighted matrix factorization. Return None
        - save() and load(): save and load R, P, C if they have been calculated already. Return None
    """
    def __init__(self, path, alpha=40):
        self.path = path
        self.n_users = 0
        self.n_items = 0
        self.R = None
        self.P = None
        self.C = None
        self.alpha = alpha
        self.dict_user = {}
        self.dict_items = {}

    def get_implicit_feedback(self):
        """Return R matrix"""
        data = pd.read_csv(self.path)
        users = data['User'].unique()
        items = data['Song'].unique()

        # list user
        id = 0
        for user in users:
            self.dict_user[id] = user
            self.dict_user[user] = id
            id += 1
        # list items
        id = 0
        for item in items:
            self.dict_items[id] = item
            self.dict_items[item] = id
            id += 1

        # make R matrix
        self.n_users = int(len(self.dict_user) / 2)
        self.n_items = int(len(self.dict_items) / 2)
        self.R = np.zeros((self.n_users, self.n_items), dtype=float)
        for index, row in data.iterrows():
            self.R[self.dict_user[row['User']], self.dict_items[row['Song']]] = int(row['Play count'])

    def convert(self):
        """Convert R into P and C"""
        self.P = np.zeros_like(self.R)
        self.C = np.ones_like(self.R)

        for i in range(self.n_users):
            for j in range(self.n_items):
                if self.R[i, j] == 0:
                    continue
                self.P[i, j] = 1
                self.C[i, j] = 1 + self.alpha * self.R[i, j]

    def save(self):
        if self.R is not None:
            np.savetxt('./data/R.txt', self.R, delimiter=' ', fmt='%d')
        if self.P is not None:
            np.savetxt('./data/P.txt', self.P, delimiter=' ', fmt='%d')
        if self.C is not None:
            np.savetxt('./data/C.txt', self.C, delimiter=' ', fmt='%.3f')

    def load(self):
        if os.path.isfile('./data/R-train.txt'):
            with open('./data/R-train.txt', 'r') as f:
                self.R = [[float(num) for num in line[:-1].split(' ')] for line in f]
                self.R = np.array(self.R)
                self.n_users = self.R.shape[0]
                self.n_items = self.R.shape[1]
        if os.path.isfile('./data/P.txt'):
            with open('./data/P.txt', 'r') as f:
                self.P = [[float(num) for num in line[:-1].split(' ')] for line in f]
                self.P = np.array(self.P)
        if os.path.isfile('./data/C.txt'):
            with open('./data/C.txt', 'r') as f:
                self.C = [[float(num) for num in line[:-1].split(' ')] for line in f]
                self.C = np.array(self.C)


class WeightedMF:
    """
    Do weighted matrix factorization by gradient descent
    Argument:
        - P, C: P, C matrices
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
    def __init__(self, P, C, depth=5, n_epoches=200, lr=1e-6, rgl=0.02, graph_inferred=False, early_stopping=True, verbose=False):
        self.P = P
        self.C = C
        self.n_users = P.shape[0]
        self.n_items = P.shape[1]
        self.depth = depth
        self.U = np.random.rand(self.n_users, self.depth)
        self.I = np.random.rand(self.depth, self.n_items)
        self.predict = np.zeros_like(P)
        self.n_epoches = n_epoches
        self.rgl = rgl
        self.lr = lr
        self.graph_inferred = graph_inferred
        self.verbose = verbose
        self.early_stopping = early_stopping

    def fit(self):
        loss = 0
        E_U = np.ones((1, self.depth), dtype=float)
        E_i = np.ones((self.depth, 1), dtype=float)
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
            else:  # Gradient Descent
                dU = np.zeros_like(self.U)
                dI = np.zeros_like(self.I)
                for i in range(self.n_users):
                    for j in range(self.n_items):
                        dU[i] += 2 * self.C[i, j] * (np.dot(self.U[i, :], self.I[:, j]) - self.P[i, j]) * np.dot(E_U, self.I[:, j])
                    dU[i] += 2 * self.rgl * self.U[i]
                    # dU[i] = normalize.sum_normalize(dU[i])

                for j in range(self.n_items):
                    for i in range(self.n_users):
                        if self.P[i, j] == 0:
                            continue
                        dI[:, j] += 2 * self.C[i, j] * (np.dot(self.U[i, :], self.I[:, j]) - self.P[i, j]) * np.dot(self.U[i, :], E_i)
                    dI[:, j] += 2 * self.rgl * self.I[:, j]
                    # dI[:, j] = normalize.sum_normalize(dI[:, j])

                self.U -= self.lr * dU
                self.I -= self.lr * dI

        # Normalize 0~1
        for i in range(self.n_users):
            self.U[i] = normalize.softmax_normalize(self.U[i])
        for j in range(self.n_items):
            self.I[:, j] = normalize.softmax_normalize(self.I[:, j])
        self.predict = np.dot(self.U, self.I)

    def predict(self, user_index, n_rec_items):
        recommendations = np.argsort(self.predict[user_index])
        return recommendations[-n_rec_items]

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
