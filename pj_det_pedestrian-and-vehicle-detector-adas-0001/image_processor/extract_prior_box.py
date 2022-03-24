import numpy as np

prior = np.load('0.npy')[0]
prior_box = prior[0]
prior_variance = prior[1]
for i in range(0, int(prior_box.shape[0] / 4)):
    print(str(prior_box[i * 4 + 0]) + ", " + str(prior_box[i * 4 + 1]) + ", " + str(prior_box[i * 4 + 2]) + ", " + str(prior_box[i * 4 + 3]) + ",")
    # print(str(prior_variance[i * 4 + 0]) + ", " + str(prior_variance[i * 4 + 1]) + ", " + str(prior_variance[i * 4 + 2]) + ", " + str(prior_variance[i * 4 + 3]))
