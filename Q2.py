import numpy as np
import os
from Q1 import DataLoader
import time
class CollabrotiveFilter:
    def measure(self, prediction, real_value):
        self.loss += np.square(prediction-real_value)


    def __init__(self, folder=""):
        self.train_data = np.load(os.path.join(folder, "train_mat.npy"))
        self.test_data = np.load(os.path.join(folder, "test_mat.npy"))
        self.loss = 0.0
        self.user_idx, self.movie_idx = DataLoader().loadIdx()
        self.sims = np.zeros((len(self.user_idx), len(self.user_idx)))

    def PearsonTestCase(self):
        vec1 = np.array([4.0,0,0,5,1,0,0])
        vec2 = np.array([5.0,5,4,0,0,0,0])
        assert np.round(self.calPearson(vec1, vec2), 3) == 0.092

    def calPearson(self, vec1, vec2):
        epsilon = 1e-10
        mean_a = np.mean(vec1[vec1!=0])
        mean_b = np.mean(vec2[vec2!=0])
        vec1[vec1 == 0] = mean_a
        vec2[vec2 == 0] = mean_b
        vec1 -= np.mean(vec1)
        vec2 -= np.mean(vec2)
        return np.dot(vec1, vec2) / (np.sqrt(np.dot(vec1, vec1)) * np.sqrt(np.dot(vec2, vec2)) + epsilon)

    def PearsonSim(self, idx1, idx2):

        vec1 = np.copy(self.train_data[idx1])
        vec2 = np.copy(self.train_data[idx2])
        sim = self.calPearson(vec1, vec2)

        return sim

    def cosSimTestCase(self):
        self.train_data = np.array([[4.0, 0, 0, 5, 1, 0, 0],[5.0, 5, 4, 0, 0, 0, 0]])
        self.test_data = np.array([[0.0, 5, 4, 0, 0, 0, 0],[0.0, 0,0, 5, 1, 0, 0]])
        self.cosSim()
        print(self.sims)
        assert self.sims[0][0] == 0

    def cosSim(self):
        product = self.train_data.dot(self.train_data.transpose())
        square_sum = self.train_data * self.train_data
        square_sum = np.sum(square_sum, axis=1)
        square_sum = np.sqrt(square_sum)
        left = np.array([square_sum]).transpose()
        right = np.array([square_sum])
        square_sum = left.dot(right)
        self.sims = product / square_sum
        self.sims[range(self.sims.shape[0]), range(self.sims.shape[0])] = 0



    def calSim(self):
        for idx1 in range(len(self.user_idx)):
            for idx2 in range(idx1+1, len(self.user_idx)):
                sim = self.PearsonSim(idx1, idx2)
                self.sims[idx1][idx2] = sim
                self.sims[idx2][idx1] = sim


    def predict(self, user_id, movie_id):
        pass

    def test(self):
        start_time = time.time()
        self.cosSim()
        print(f"SIM CALCULATE FINISHED")
        ones = np.ones(self.test_data.shape)
        contribution = np.zeros(self.test_data.shape)

        predict_res = self.sims.dot(self.train_data)
        contribution[self.train_data !=0] = 1
        contirbution = self.sims.dot(contribution)
        predict_res = predict_res  / (contirbution + 1e-10)

        predict_res[self.test_data == 0] = 0
        ones[self.test_data == 0] = 0
        diff = predict_res - self.test_data
        diff *= diff
        square_error = np.sum(diff)
        rmse = square_error / np.sum(ones)
        end_time = time.time()
        print(f"check rmse = {rmse}, consuming time = {end_time-start_time}")

if __name__ == "__main__":
    cf = CollabrotiveFilter()
    cf.test()
