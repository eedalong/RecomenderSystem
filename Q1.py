import numpy as np
import os
from typing import Dict, List
class DataLoader:
    def __init__(self, folder_path="data/"):
        self.folder_path = folder_path

    def loadIdx(self)->(Dict,Dict):
        # load user idx
        user_idx = {}
        with open(os.path.join(self.folder_path, "users.txt")) as user_file:
            for index, user_id in enumerate(user_file):
                user_id = user_id[:-1]
                user_idx[user_id] = index


        # load movie idx
        movie_idx = {}
        with open(os.path.join(self.folder_path, "movie_titles.txt")) as movie_file:
            for index, movie_id in enumerate(movie_file):
                movie_id = movie_id.split(",")[0]
                movie_idx[movie_id] = index
        return user_idx, movie_idx

    def loadData(self, file_name)->(str, List):
        with open(os.path.join(self.folder_path, file_name)) as data_file:

            firstLine = next(data_file)
            data = firstLine.split(" ")
            user_id = data[0]
            res = [data[1:]]
            for line in data_file:
                data = line.split(" ")
                if data[0] != user_id:
                    yield user_id, res
                    res = [data[1:]]
                    user_id = data[0]
                else:
                    res.append(data[1:])

def testCase():
    """
    Random check
    :return:
    """
    dataLoader = DataLoader()
    user_idx, movie_idx = dataLoader.loadIdx()
    test_mat = np.load("test_mat.npy")
    train_mat = np.load("train_mat.npy")
    assert train_mat[user_idx["1581163"]][movie_idx["708"]] == 5
    assert train_mat[user_idx["1932106"]][movie_idx["2040"]] == 5
    assert test_mat[user_idx["2123534"]][movie_idx["8764"]] == 4

if __name__ == "__main__":

    dataLoader = DataLoader()
    user_idx, movie_idx = dataLoader.loadIdx()
    # load train matrix
    train_mat = np.zeros((len(user_idx), len(movie_idx)))
    test_mat = np.zeros((len(user_idx), len(movie_idx)))

    test_idx = 0
    with open(os.path.join("data", "netflix_train.txt")) as data_file:
        for line in data_file:
            data = line.split(" ")
            user_id = user_idx[data[0]]
            movie_id = movie_idx[data[1]]
            train_mat[user_id][movie_id] = int(data[2])
        np.save("train_mat.npy", train_mat)

    with open(os.path.join("data", "netflix_test.txt")) as data_file:
        for line in data_file:
            data = line.split(" ")
            user_id = user_idx[data[0]]
            movie_id = movie_idx[data[1]]
            test_mat[user_id][movie_id] = int(data[2])
        np.save("test_mat.npy", test_mat)

    # test if we did everything right
    testCase()










