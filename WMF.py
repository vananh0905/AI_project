import numpy as np
import pandas as pd
import os.path


class IFConverter:
    def __init__(self, path, alpha=0.5):
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
        if self.R:
            np.savetxt('./data/R.txt', self.R, delimiter=' ', fmt='%d')
        if self.P:
            np.savetxt('./data/P.txt', self.P, delimiter=' ', fmt='%d')
        if self.C:
            np.savetxt('./data/C.txt', self.C, delimiter=' ', fmt='%.3f')

    def load(self):
        if os.path.isfile('./data/R.txt'):
            with open('./data/R.txt', 'r') as f:
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
    def __init__(self, P, C, depth=5, n_epoches=200, lr=1e-6, rgl=0.02, graph_inferred=False, early_stopping=True, verbose=False):
        self.P = P
        self.C = C
        self.n_users = P.shape[0]
        self.n_items = P.shape[1]
        self.depth = depth
        self.U = np.random.rand(self.n_users, self.depth)
        self.I = np.random.rand(self.depth, self.n_items)
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

                for j in range(self.n_items):
                    for i in range(self.n_users):
                        if self.P[i, j] == 0:
                            continue
                        dI[:, j] += 2 * self.C[i, j] * (np.dot(self.U[i, :], self.I[:, j]) - self.P[i, j]) * np.dot(self.U[i, :], E_i)
                    dI[:, j] += 2 * self.rgl * self.I[:, j]

                self.U -= self.lr * dU
                self.I -= self.lr * dI

    def predict(self):
        U_tmp = np.zeros_like(self.U)
        I_tmp = np.zeros_like(self.I)
        for i in range(self.n_users):
            sum_user_i = np.sum(np.exp(self.U[i]))
            for k in range(self.depth):
                U_tmp[i, k] = np.exp(self.U[i, k]) / sum_user_i
        for j in range(self.n_items):
            sum_item_j = np.sum(np.exp(self.I[:, j]))
            for k in range(self.depth):
                I_tmp[k, j] = np.exp(self.I[k, j]) / sum_item_j
        predict = np.dot(U_tmp, I_tmp)
        np.savetxt('./data/predict.txt', predict, delimiter=' ', fmt='%.5f')
