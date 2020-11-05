import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn import datasets


class GaussianMixtureModel:
    def __init__(self, n_clusters, n_features):
        self.n_clusters = n_clusters
        self.n_features = n_features

    def init_parameters(self):
        #initialize parameters
        self.n_samples = self.X.shape[0]

        min_of_dims = np.array([np.min(self.X[:, i]) for i in range(self.n_features)])
        max_of_dims = np.array([np.max(self.X[:, i]) for i in range(self.n_features)])
        d = (max_of_dims - min_of_dims) * 1. / (self.n_clusters - 1)
        #initialize mean and covariance of each cluster
        self.mean = np.array([min_of_dims + d * i for i in range(self.n_clusters)])
        self.cov = np.zeros(shape= (self.n_clusters, self.n_features, self.n_features))
        for i in range(self.n_clusters):
            self.cov[i] += np.identity(self.n_features) * np.random.randint(5, 10)
        self.reg_cov = 1e-6 * np.identity(self.n_features)      #avoid that covariance equals to zero

        self.alpha = np.array([1. / self.n_clusters for i in range(self.n_clusters)])     #prior probability of latent variable
        self.T = np.zeros(shape=(self.n_samples, self.n_clusters))      #posterior probability of latent variable
        self.log_likelihood = []
        
    def fit(self, X, n_epochs= 20, epsi= 1e-3, plot= True):
        #training
        self.X = X
        self.init_parameters()
        epoch = 1
        while self.criterion(epoch= epoch, n_epochs= n_epochs, epsi= epsi, plot= plot):
            self.e_step()
            self.m_step()
            self.compute_log_likelihood()
            epoch += 1
            if (epoch / n_epochs * 100) % 10 == 0:
                print('Epoch {}------------------------------------------'.format(epoch))

    def e_step(self):
        #update posterior
        self.pdf = np.array([multivariate_normal(mean= self.mean[i], cov= self.cov[i] + self.reg_cov).pdf(self.X)
                             for i in range(self.n_clusters)]).T     #probabil of that each observation belongs to each cluster
        numerator = self.pdf * self.alpha
        denominator = np.sum(numerator, axis= 1).reshape(-1, 1)
        self.T = numerator / denominator

    def m_step(self):
        #update hyperparameters: alpha, mean, cov
        self.alpha = np.sum(self.T, axis= 0) / self.n_samples
        for cluster in range(self.n_clusters):
            t = np.reshape(self.T[:, cluster], (-1, 1))
            self.mean[cluster] = np.sum(self.X * t, axis= 0) / np.sum(t, axis= 0)
            self.cov[cluster]= ((self.X - self.mean[cluster]).T * self.T[:, cluster]).dot((self.X - self.mean[cluster])) / np.sum(t, axis= 0)

    def compute_log_likelihood(self):
        #compute and store log likelihood
        self.log_likelihood.append(np.sum(np.log(np.sum(self.pdf * self.alpha, axis= 1))))

    def criterion(self,epoch, n_epochs, epsi, plot):
        #check stopping criterion
        if epoch <= 2:
            return True
        if (epoch < n_epochs) and ((self.log_likelihood[-1] - self.log_likelihood[-2]) > epsi):
            return True

        if plot:
            self.plot()
        return False

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
                mn = multivariate_normal(mean=self.mean[k], cov=self.cov[k] + self.reg_cov)
                ax.contour(np.sort(self.X[:, 0]), np.sort(self.X[:, 1]),
                           mn.pdf(self.XY).reshape(len(self.X), len(self.X)), colors='black')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title('Clustered data')

        #plot log likelihood per epoch
        figure2 = plt.figure(figsize=(10, 10))
        iteration = np.array(range(len(self.log_likelihood)))
        plt.plot(iteration, self.log_likelihood)
        plt.xlabel('Iterations')
        plt.title('Log likelihood')
        plt.show()

    def predict(self, x):
        #predict new observations
        pdf = np.array([multivariate_normal(mean= self.mean[i], cov= self.cov[i] + self.reg_cov).pdf(x) for i in range(self.n_clusters)])
        return np.argmax(pdf, axis= 0)

    def get_mean(self):
        #get mean of each cluster
        return self.mean

    def get_cov(self):
        #get covariace of each cluster
        return self.cov


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


# X = datasets.load_iris()['data']
# gmm = GaussianMixtureModel(n_clusters= 3, n_features= 4)
X = generate_data(n_samples= 3000)
gmm = GaussianMixtureModel(n_clusters= 3, n_features= 2)

gmm.fit(X= X, n_epochs= 50, plot= True)
print('Predicting: ', gmm.predict(X))