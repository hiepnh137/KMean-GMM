import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn import datasets


class KMeans:
    def __init__(self, n_clusters, n_features):
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.X = None

    def init_parameters(self):
        #initialize p
        self.n_samples = self.X.shape[0]

        max_dims = np.array([np.max(self.X[:, i]) for i in range(self.n_features)])
        min_dims = np.array([np.min(self.X[:, i]) for i in range(self.n_features)])
        d = (max_dims - min_dims) * 1. / (self.n_clusters - 1)
        self.centroids = np.array([min_dims + d * i for i in range(self.n_clusters)])

        self.distances = np.zeros(shape= (self.n_samples, self.n_clusters))
        self.predict = np.zeros(shape=  (self.n_samples, 1))
        self.loss =  []

    def fit(self, X, n_epochs = 20, epsi = 1e-6, plot= True):
        self.X = X
        self.init_parameters()
        self.update_distance()
        epoch = 1
        while self.criterion(epoch= epoch, n_epochs= n_epochs, epsi= epsi, plot = plot):
            self.update_centroids()
            self.update_distance()
            self.compute_loss()
            epoch += 1
            print('Epoch {}:-----------------------------'.format(epoch + 1))
        self.plot()

    def criterion(self,epoch, n_epochs, epsi, plot):
        #check stopping criterion
        if epoch <= 2:
            return True
        if (epoch < n_epochs) and ((self.loss[-2] - self.loss[-1]) > epsi):
            return True
        if plot:
            self.plot()
        return False

    def update_distance(self):
        # update distance
        for i in range(self.n_clusters):
            self.distances[:, i] = [np.linalg.norm((self.X - self.centroids[i])[j]) for j in range(self.n_samples)]
        self.predict = np.argmin(self.distances, axis= 1)

    def update_centroids(self):
        #update centroids
        for i in range(self.n_clusters):
            cluster_point = (self.predict == i).reshape(-1, 1)
            self.centroids[i] = np.sum((self.X * cluster_point), axis= 0) / np.sum(cluster_point)

    def compute_loss(self):
        self.loss.append(np.sum(self.distances[np.arange(self.n_samples), self.predict]))

    def plot(self):
        #visualize result
        if self.n_features == 2:
            #plot clustered data
            figure1 = plt.figure(figsize=(10, 10))
            plt.scatter(x=self.X[:, 0], y=self.X[:, 1])
            ax = figure1.add_subplot(111)
            x, y = np.meshgrid(np.sort(self.X[:, 0]), np.sort(self.X[:, 1]))
            self.XY = np.array([x.flatten(), y.flatten()]).T
            for k in range(self.n_clusters):
                mn = multivariate_normal(mean=self.centroids[k], cov=[[3, 0],[0, 3]])
                ax.contour(np.sort(self.X[:, 0]), np.sort(self.X[:, 1]),
                           mn.pdf(self.XY).reshape(len(self.X), len(self.X)), colors='black')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title('Clustered data')

        #plot log likelihood per epoch
        figure2 = plt.figure(figsize=(10, 10))
        iteration = np.array(range(len(self.loss)))
        plt.plot(iteration, self.loss)
        plt.xlabel('Iterations')
        plt.title('Loss')
        plt.show()



def generate_data(n_samples= 3000):
    #randomly generate data
    x1 = np.random.multivariate_normal(mean=[0, 0], cov=[[10, 7], [7, 10]], size=n_samples // 3)
    x2 = np.random.multivariate_normal(mean=[5, -15], cov=[[5, 8], [8, 19]], size=n_samples // 3)
    x3 = np.random.multivariate_normal(mean=[17, 14], cov=[[27, 3], [3, 3]], size=n_samples // 3)
    x = np.concatenate((x1, x2, x3), axis=0)
    index = np.arange(0, n_samples // 3 * 3)
    np.random.shuffle(index)
    y = np.array([x[i] for i in index])  # shape= (n, d)
    return y

# KM = KMeans(n_clusters= 3, n_features= 2)
# X = generate_data(n_samples= 1000)

inKM = KMeans(n_clusters= 3, n_features= 4)
KM.fit(X)