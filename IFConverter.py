import pandas as pd
import numpy as np
import os


class IFConverter:
    """
    Convert triplets data (user_id, song_id, play_count) into R, P, C matrices
    Arguments:
        - path: path to triplets data
        - alpha (optional): penalty for confidence. Default: 40
    Methods:
        - get_implicit_feedback(): read triplets and calculate R. Return R
        - convert(): convert R into P and C for weighted matrix factorization. Return P, C respectively
        - save() and load(): save and load R, P, C if they have been calculated already. Return None
        - get_dictionary(): Get mapping of users, items and their indexes. Return dict_users and dict_items respectively
    """
    def __init__(self, alpha=40):
        self.n_users = 0
        self.n_items = 0
        self.R = None
        self.P = None
        self.C = None
        self.alpha = alpha
        self.dict_user = {}
        self.dict_items = {}

    def get_implicit_feedback(self, path):
        """Convert triplets into R matrix"""
        data = pd.read_csv(path)
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
        return self.R

    def convert(self):
        """Convert R into P and C"""
        self.P = np.zeros_like(self.R)
        self.C = np.ones_like(self.R)
        self.n_users = self.R.shape[0]
        self.n_items = self.R.shape[1]
        for i in range(self.n_users):
            for j in range(self.n_items):
                if self.R[i, j] == 0:
                    continue
                self.P[i, j] = 1
                self.C[i, j] = 1 + self.alpha * self.R[i, j]
        return self.P, self.C

    def save(self):
        if self.P is not None:
            np.savetxt('./data/P.txt', self.P, delimiter=' ', fmt='%d')
        if self.C is not None:
            np.savetxt('./data/C.txt', self.C, delimiter=' ', fmt='%.3f')

    def load(self):
        if os.path.isfile('./data/P.txt'):
            with open('./data/P.txt', 'r') as f:
                self.P = [[float(num) for num in line[:-1].split(' ')] for line in f]
                self.P = np.array(self.P)
        if os.path.isfile('./data/C.txt'):
            with open('./data/C.txt', 'r') as f:
                self.C = [[float(num) for num in line[:-1].split(' ')] for line in f]
                self.C = np.array(self.C)

    def get_dictionary(self):
        return self.dict_user, self.dict_items
