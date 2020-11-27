import numpy as np
import os.path


class WeightedMF:
    """
    Do weighted matrix factorization by gradient descent
    Argument:
        - P, C: P, C matrices
        - dict_user, dict_item: mapping between name and index of users and items
        - optimizer: approach to train. There are 3 choices: use formula, use derivative with sdg, use derivative with gd
        - depth: number of latent features. Default: 5
        - n_epoches: number of epoches to train. Default: 200
        - lr: learning rate. Default: 1e-6
        - rgl: regularization value (lambda). Default: 0.02
        - graph_inferred: if True, implicit feedback is filled by neighborhood users (useful for sparse P matrix). Default: False
        - early_stopping: if True, stop early in gradient descent when loss may not improve; otherwise, train to the last epoch. Default: True
        - verbose: if True, print out the information of process. Default: False
        - U, I: hidden representation for each user, item
    Methods:
        - fit_derivative(): training model with derivative
        - fit_formula(): training model with formula
        - get_recommendations(): return list of recommendations for a user given the user_id and number of items to recommend
        - save(), load(): save and load U, I if they have been calculated already

    """
    def __init__(self, P, C, n_epoches=0, optimizer='sgd', depth=5, lr=1e-6, rgl=0.02, batch_size=0, graph_inferred=False, early_stopping=True, verbose=False):
        self.P = P
        self.C = C
        self.n_users = P.shape[0]
        self.n_items = P.shape[1]
        self.depth = depth
        self.U = np.random.rand(self.n_users, self.depth)
        self.I = np.random.rand(self.depth, self.n_items)
        self.predict = np.zeros_like(P)
        self.rgl = rgl
        self.lr = lr
        self.graph_inferred = graph_inferred
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.optimizer = optimizer
        if n_epoches == 0:
            if optimizer == 'sgd':
                self.n_epoches = 20
            elif optimizer == 'gd':
                self.n_epoches = 200
            elif optimizer == 'formula':
                self.n_epoches = 10
        else:
            self.n_epoches = n_epoches
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

        for j in range(self.n_items):
            for i in range(self.n_users):
                dI[:, j] += 2 * self.C[i, j] * (np.dot(self.U[i, :], self.I[:, j]) - self.P[i, j]) \
                            * np.dot(self.U[i, :], E_i)
            dI[:, j] += 2 * self.rgl * self.I[:, j]

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

    def __fit_formula(self):
        loss = 0
        for current_epoch in range(self.n_epoches):
            prev_loss = loss
            for i in range(self.n_users):
                C_u = np.zeros((self.n_items, self.n_items))
                E = np.zeros((self.depth, self.depth))
                for j in range(self.n_items):
                    C_u[j, j] = self.C[i, j]
                self.U[i] = np.dot(
                    np.dot(np.dot(np.linalg.inv(np.dot(np.dot(self.I, C_u), self.I.T) + self.rgl * E), self.I), C_u),
                    self.P[i].T).T
            for i in range(self.n_users):
                C_i = np.zeros((self.n_users, self.n_users))
                E = np.zeros((self.depth, self.depth))
                for j in range(self.n_users):
                    C_i[j, j] = self.C[i, j]
                self.I[:, i] = np.dot(
                    np.dot(np.dot(np.linalg.inv(np.dot(np.dot(self.U.T, C_i), self.U) + self.rgl * E), self.U.T), C_i),
                    self.P[:, i])
            loss = np.sqrt(np.sum((np.dot(self.U, self.I) - self.P) ** 2) / (self.n_users * self.n_items))
            if self.early_stopping and current_epoch > 1:
                if prev_loss - loss < 0.0001:
                    break
            if self.verbose:
                print("Training at {}th epoch".format(current_epoch + 1))
                print("Avg. loss: {:.5f}".format(loss))

    def __fit_derivative(self):
        loss = 0
        for current_epoch in range(self.n_epoches):
            prev_loss = loss
            loss = 0
            if self.verbose:
                print("Processing epoch {}".format(current_epoch+1))
            for i in range(self.n_users):
                for j in range(self.n_items):
                    loss += self.C[i, j] * (np.dot(self.U[i, :], self.I[:, j]) - self.P[i, j]) ** 2
            loss = loss + self.rgl * (np.sum(self.U ** 2) + np.sum(self.I ** 2))
            if self.verbose:
                print("Loss at epoch {}: {:.3f}".format(current_epoch+1, loss))
            if self.early_stopping and current_epoch > 1 and (prev_loss - loss < 0.001):
                return None
            else:
                if self.optimizer == 'gd':
                    self.__gd()
                else:
                    self.__sgd()

    def fit(self):
        if self.optimizer == 'formula':
            self.__fit_formula()
        else:
            self.__fit_derivative()
        self.predict = np.dot(self.U, self.I)

    def get_recommendations(self, user_index, n_rec_items):
        recommendations = np.argsort(self.predict[user_index])
        recommendations = recommendations[-n_rec_items:]
        # name_of_songs_rec = []
        # for i in range(n_rec_items):
        #     name_of_songs_rec.append(self.dict_item[recommendations[i]])
        # return name_of_songs_rec
        return recommendations

    def save(self):
        np.savetxt('./data/U.txt', self.U, delimiter=' ', fmt='%.5f')
        np.savetxt('./data/I.txt', self.I, delimiter=' ', fmt='%.5f')
        np.savetxt('./data/predict.txt', self.predict, delimiter=' ', fmt='%.5f')

    def load(self):
        if os.path.isfile('./data/U.txt'):
            with open('./data/U.txt', 'r') as f:
                self.U = [[float(num) for num in line[:-1].split(' ')] for line in f]
                self.U = np.array(self.U)
        if os.path.isfile('./data/I.txt'):
            with open('./data/I.txt', 'r') as f:
                self.I = [[float(num) for num in line[:-1].split(' ')] for line in f]
                self.I = np.array(self.I)

