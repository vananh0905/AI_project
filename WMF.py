import numpy as np
import pandas as pd

dict_user = {}
dict_items = {}

def get_interaction():
    data = pd.read_csv('./TEST.csv')
    users = data['User'].unique()
    items = data['Song'].unique()

    # list user
    id = 0
    for user in users:
        dict_user[id] = user
        dict_user[user] = id
        id += 1
    # list items
    id = 0
    for item in items:
        dict_items[id] = item
        dict_items[item] = id
        id += 1

    # make R matrix
    n_users = int(len(dict_user) / 2)
    n_items = int(len(dict_items) / 2)
    R = np.zeros((n_users, n_items))
    for index, row in data.iterrows():
        R[dict_user[row['User']], dict_items[row['Song']]] = int(row['Play count'])
    #np.savetxt('R.txt', R, delimiter=' ', fmt='%d')
    return R

class IFConverter:
    def __init__(self, learning_rate=0.005):
        self.R = get_interaction()
        self.lr = learning_rate

    def get_data(self):
        P = np.zeros_like(self.R) * 1.0
        C = np.ones_like(self.R) * 1.0
        n_users = self.R.shape[0]
        n_items = self.R.shape[1]

        for i in range(n_users):
            for j in range(n_items):
                if self.R[i,j] == 0:
                    continue
                P[i, j] = 1
                C[i, j] = 1 + self.lr * self.R[i, j]
        return P, C


class WeightedMF:
    def __init__(self, P, C, depth, epoches=100, lr=0.005, rgl=0, verbose=False, graph_inferred=False,
                 early_stopping=True):
        self.P = P
        self.C = C
        self.n_users = P.shape[0]
        self.n_items = P.shape[1]
        self.depth = depth
        self.U = np.random.rand(self.n_users, self.depth)
        self.I = np.random.rand(self.depth, self.n_items)
        self.n_epoches = epoches
        self.lr = lr
        self.rgl = rgl
        self.verbose = verbose
        self.graph_inferred = graph_inferred
        self.early_stopping = early_stopping

    def fit(self):
        loss = 0
        for current_epoch in range(self.n_epoches):
            prev_loss = loss
            loss = 0
            if self.verbose:
                print("Processing epoch {}".format(current_epoch))
            for i in range(self.n_users):
                for j in range(self.n_items):
                    if self.P[i, j] == 0:
                        continue
                    loss += self.C[i, j] * (np.dot(self.U[i, :], self.I[:, j]) - self.P[i, j]) ** 2
            loss = loss + self.rgl * (np.sum(self.U ** 2) + np.sum(self.I ** 2))
            print("Loss at epoch {}: {}".format(current_epoch, loss))
            if self.early_stopping and current_epoch > 1 and (prev_loss - loss < 0.001):
                return None
            else:   # Gradient Descent
                dU = np.zeros_like(self.U)
                dI = np.zeros_like(self.I)
                for i in range(self.n_users):
                    for j in range(self.n_items):
                        if self.P[i, j] == 0:
                            continue
                        dU[i] += 2 * self.C[i, j] * (np.dot(self.U[i, :], self.I[:, j]) - self.P[i, j]) * \
                                 np.dot(np.ones((self.depth,), dtype=int), self.I[:, j])
                    dU[i] += 2 * self.rgl * self.U[i]

                for j in range(self.n_items):
                    for i in range(self.n_users):
                        if self.P[i, j] == 0:
                            continue
                        dI[:, j] += 2 * self.C[i, j] * (np.dot(self.U[i, :], self.I[:, j]) - self.P[i, j]) * \
                                    np.dot(self.U[i, :], np.ones((self.depth,), dtype=int))
                    dI[:, j] += 2 * self.rgl * self.I[:, j]

                self.U -= self.lr * dU
                self.I -= self.lr * dI

    def count(self):
        sum = 0
        for i in range(self.n_users):
            sum += np.sum(self.P[i])
        print(sum / (self.n_items*self.n_users) * 100)
    def predict(self):
        U_tmp = np.zeros_like(self.U) * 1.0
        I_tmp = np.zeros_like(self.I) * 1.0
        for i in range(self.n_users):
            U_tmp[i] = np.exp(self.U[i]) / (np.sum(np.exp(self.U[i])))
        for i in range(self.depth):
            I_tmp[i] = np.exp(self.I[i]) / (np.sum(np.exp(self.I[i])))
        #self.U = U_tmp
        #self.I = I_tmp
        predict = np.dot(U_tmp, I_tmp)
        np.savetxt('predict.txt', predict, delimiter=' ', fmt='%5.3f')

