# -*- coding:utf-8 -*-
# Hidden Markov Model
# By tostq <tostq216@163.com>
# Blog: blog.csdn.net/tostq
import numpy as np
from math import pi,sqrt,exp,pow,log
from numpy.linalg import det, inv
from abc import ABCMeta, abstractmethod
from sklearn import cluster
import time
class _BaseHMM():
    """
    Base HMM abstract class, need to override emission probability related abstract functions
    n_state : number of hidden states
    n_iter : number of iterations
    x_size : observation dimension
    start_prob : initial probability
    transmat_prob : state transition probability
    """
    __metaclass__ = ABCMeta  # Abstract class declaration

    def __init__(self, n_state=1, x_size=1, iter=20):
        self.n_state = n_state
        self.x_size = x_size
        self.start_prob = np.ones(n_state) * (1.0 / n_state)  # Initial state probability
        self.transmat_prob = np.ones((n_state, n_state)) * (1.0 / n_state)  # State transition probability matrix
        self.trained = False # Whether retraining is needed
        self.n_iter = iter  # Number of EM training iterations

    # Initialize emission parameters
    @abstractmethod
    def _init(self,X):
        pass

    # Abstract function: return emission probability
    @abstractmethod
    def emit_prob(self, x):  # Calculate emission probability P(X|Z) for x under state k
        return np.array([0])

    # Abstract function
    @abstractmethod
    def generate_x(self, z): # Generate observation x given hidden state p(x|z)
        return np.array([0])

    # Abstract function: update emission probability
    @abstractmethod
    def emit_prob_updated(self, X, post_state):
        pass

    # Generate sequence using HMM
    def generate_seq(self, seq_length):
        X = np.zeros((seq_length, self.x_size))
        Z = np.zeros(seq_length)
        Z_pre = np.random.choice(self.n_state, 1, p=self.start_prob)  # Sample initial state
        X[0] = self.generate_x(Z_pre) # Sample first value of sequence
        Z[0] = Z_pre

        for i in range(seq_length):
            if i == 0: continue
            # P(Zn+1)=P(Zn+1|Zn)P(Zn)
            Z_next = np.random.choice(self.n_state, 1, p=self.transmat_prob[Z_pre,:][0])
            Z_pre = Z_next
            # P(Xn+1|Zn+1)
            X[i] = self.generate_x(Z_pre)
            Z[i] = Z_pre

        return X,Z

    # Estimate the probability of sequence X
    def X_prob(self, X, Z_seq=np.array([])):
        # State sequence preprocessing
        # Check if hidden state is known
        X_length = len(X)
        if Z_seq.any():
            Z = np.zeros((X_length, self.n_state))
            for i in range(X_length):
                Z[i][int(Z_seq[i])] = 1
        else:
            Z = np.ones((X_length, self.n_state))
        # Forward-backward factors
        _, c = self.forward(X, Z)  # P(x,z)
        # Sequence probability estimation
        prob_X = np.sum(np.log(c))  # P(X)
        return prob_X

    # Predict the probability of the next observation given current sequence X
    def predict(self, X, x_next, Z_seq=np.array([]), istrain=True):
        if self.trained == False or istrain == False:  # Need to retrain on this sequence
            self.train(X)

        X_length = len(X)
        if Z_seq.any():
            Z = np.zeros((X_length, self.n_state))
            for i in range(X_length):
                Z[i][int(Z_seq[i])] = 1
        else:
            Z = np.ones((X_length, self.n_state))
        # Forward-backward factors
        alpha, _ = self.forward(X, Z)  # P(x,z)
        t = self.emit_prob(np.array([x_next]))
        prob_x_next = self.emit_prob(np.array([x_next]))*np.dot(alpha[X_length - 1],self.transmat_prob)

        return prob_x_next

    def decode(self, X, istrain=True):
        """
        Use Viterbi algorithm to infer hidden state sequence given observations
        :param X: observation sequence
        :param istrain: whether to retrain on this sequence
        :return: hidden state sequence
        """
        if self.trained == False or istrain == False:  # Need to retrain on this sequence
            self.train(X)

        X_length = len(X)  # Sequence length
        state = np.zeros(X_length)  # Hidden state

        pre_state = np.zeros((X_length, self.n_state))  # Most likely previous state for each current state
        max_pro_state = np.zeros((X_length, self.n_state))  # Max probability for each state at each position

        _,c=self.forward(X,np.ones((X_length, self.n_state)))
        max_pro_state[0] = self.emit_prob(X[0]) * self.start_prob * (1/c[0]) # Initial probability

        # Forward process
        for i in range(X_length):
            if i == 0: continue
            for k in range(self.n_state):
                prob_state = self.emit_prob(X[i])[k] * self.transmat_prob[:,k] * max_pro_state[i-1]
                max_pro_state[i][k] = np.max(prob_state)* (1/c[i])
                pre_state[i][k] = np.argmax(prob_state)

        # Backward process
        state[X_length - 1] = np.argmax(max_pro_state[X_length - 1,:])
        for i in reversed(range(X_length)):
            if i == X_length - 1: continue
            state[i] = pre_state[i + 1][int(state[i + 1])]

        return  state

    # Training for multiple sequences
    def train_batch(self, X, Z_seq=list()):
        # For multiple sequences, the simplest way is to concatenate them, only need to adjust initial state probability
        # Input X: list(array)
        # Input Z: list(array), default empty (unknown hidden state)
        self.trained = True
        X_num = len(X) # Number of sequences
        self._init(self.expand_list(X)) # Initialize emission probability

        # State sequence preprocessing, convert single state to 1-to-k form
        # Check if hidden state is known
        if Z_seq==list():
            Z = []  # Initialize state sequence list
            for n in range(X_num):
                Z.append(list(np.ones((len(X[n]), self.n_state))))
        else:
            Z = []
            for n in range(X_num):
                Z.append(np.zeros((len(X[n]),self.n_state)))
                for i in range(len(Z[n])):
                    Z[n][i][int(Z_seq[n][i])] = 1

        for e in range(self.n_iter):  # EM step iteration
            # Update initial probability process
            #  E-step
            time_st = time.time()
            print("iter: ", e)
            b_post_state = []  # Batch accumulation: posterior probability of state, list(array)
            b_post_adj_state = np.zeros((self.n_state, self.n_state)) # Batch accumulation: joint posterior probability of adjacent states, array
            b_start_prob = np.zeros(self.n_state) # Batch accumulation of initial probability
            for n in range(X_num): # For each sequence
                X_length = len(X[n])
                alpha, c = self.forward(X[n], Z[n])  # P(x,z)
                beta = self.backward(X[n], Z[n], c)  # P(x|z)

                post_state = alpha * beta / np.sum(alpha * beta) # Normalize!
                b_post_state.append(post_state)
                post_adj_state = np.zeros((self.n_state, self.n_state))  # Joint posterior probability of adjacent states
                for i in range(X_length):
                    if i == 0: continue
                    if c[i]==0: continue
                    post_adj_state += (1 / c[i]) * np.outer(alpha[i - 1],
                                                            beta[i] * self.emit_prob(X[n][i])) * self.transmat_prob

                if np.sum(post_adj_state)!=0:
                    post_adj_state = post_adj_state/np.sum(post_adj_state)  # Normalize!
                b_post_adj_state += post_adj_state  # Batch accumulation: posterior probability of state
                b_start_prob += b_post_state[n][0] # Batch accumulation of initial probability

            # M-step, estimate parameters, avoid all-zero initial probability
            b_start_prob += 0.001*np.ones(self.n_state)
            self.start_prob = b_start_prob / np.sum(b_start_prob)
            b_post_adj_state += 0.001
            for k in range(self.n_state):
                if np.sum(b_post_adj_state[k])==0: continue
                self.transmat_prob[k] = b_post_adj_state[k] / np.sum(b_post_adj_state[k])

            self.emit_prob_updated(self.expand_list(X), self.expand_list(b_post_state))
            print(time.time() - time_st)

    def expand_list(self, X):
        # Expand list(array) type data to array type
        C = []
        for i in range(len(X)):
            C += list(X[i])
        return np.array(C)

    # Training for a single long sequence
    def train(self, X, Z_seq=np.array([])):
        # Input X: array
        # Input Z: array, default empty (unknown hidden state)
        self.trained = True
        X_length = len(X)
        self._init(X)

        # State sequence preprocessing
        # Check if hidden state is known
        if Z_seq.any():
            Z = np.zeros((X_length, self.n_state))
            for i in range(X_length):
                Z[i][int(Z_seq[i])] = 1
        else:
            Z = np.ones((X_length, self.n_state))

        for e in range(self.n_iter):  # EM step iteration
            # Intermediate parameters
            print(e, " iter")
            # E-step
            # Forward-backward factors
            alpha, c = self.forward(X, Z)  # P(x,z)
            beta = self.backward(X, Z, c)  # P(x|z)

            post_state = alpha * beta
            post_adj_state = np.zeros((self.n_state, self.n_state))  # Joint posterior probability of adjacent states
            for i in range(X_length):
                if i == 0: continue
                if c[i]==0: continue
                post_adj_state += (1 / c[i])*np.outer(alpha[i - 1],beta[i]*self.emit_prob(X[i]))*self.transmat_prob

            # M-step, estimate parameters
            self.start_prob = post_state[0] / np.sum(post_state[0])
            for k in range(self.n_state):
                self.transmat_prob[k] = post_adj_state[k] / np.sum(post_adj_state[k])

            self.emit_prob_updated(X, post_state)

    # Forward factor
    def forward(self, X, Z):
        X_length = len(X)
        alpha = np.zeros((X_length, self.n_state))  # P(x,z)
        alpha[0] = self.emit_prob(X[0]) * self.start_prob * Z[0] # Initial value
        # Normalization factor
        c = np.zeros(X_length)
        c[0] = np.sum(alpha[0])
        alpha[0] = alpha[0] / c[0]
        # Recursive propagation
        for i in range(X_length):
            if i == 0: continue
            alpha[i] = self.emit_prob(X[i]) * np.dot(alpha[i - 1], self.transmat_prob) * Z[i]
            c[i] = np.sum(alpha[i])
            if c[i]==0: continue
            alpha[i] = alpha[i] / c[i]

        return alpha, c

    # Backward factor
    def backward(self, X, Z, c):
        X_length = len(X)
        beta = np.zeros((X_length, self.n_state))  # P(x|z)
        beta[X_length - 1] = np.ones((self.n_state))
        # Recursive propagation
        for i in reversed(range(X_length)):
            if i == X_length - 1: continue
            beta[i] = np.dot(beta[i + 1] * self.emit_prob(X[i + 1]), self.transmat_prob.T) * Z[i]
            if c[i+1]==0: continue
            beta[i] = beta[i] / c[i + 1]

        return beta

# 2D Gaussian distribution function
def gauss2D(x, mean, cov):
    # x, mean, cov are all numpy.array
    z = -np.dot(np.dot((x-mean).T,inv(cov)),(x-mean))/2.0
    temp = pow(sqrt(2.0*pi),len(x))*sqrt(det(cov))
    return (1.0/temp)*exp(z)

class GaussianHMM(_BaseHMM):
    """
    HMM with Gaussian emission probability
    Args:
    emit_means: mean of Gaussian emission probability
    emit_covars: covariance of Gaussian emission probability
    """
    def __init__(self, n_state=1, x_size=1, iter=20):
        _BaseHMM.__init__(self, n_state=n_state, x_size=x_size, iter=iter)
        self.emit_means = np.zeros((n_state, x_size))      # Mean of Gaussian emission probability
        self.emit_covars = np.zeros((n_state, x_size, x_size)) # Covariance of Gaussian emission probability
        for i in range(n_state): self.emit_covars[i] = np.eye(x_size)  # Initialize as mean 0, variance 1 Gaussian

    def _init(self,X):
        # Use KMeans to determine initial state
        mean_kmeans = cluster.KMeans(n_clusters=self.n_state)
        mean_kmeans.fit(X)
        self.emit_means = mean_kmeans.cluster_centers_
        for i in range(self.n_state):
            self.emit_covars[i] = np.cov(X.T) + 0.01 * np.eye(len(X[0]))

    def emit_prob(self, x): # Calculate emission probability for x under state k
        prob = np.zeros((self.n_state))
        for i in range(self.n_state):
            prob[i]=gauss2D(x,self.emit_means[i],self.emit_covars[i])
        return prob

    def generate_x(self, z): # Generate x given state p(x|z)
        return np.random.multivariate_normal(self.emit_means[z][0],self.emit_covars[z][0],1)

    def emit_prob_updated(self, X, post_state): # Update emission probability
        for k in range(self.n_state):
            for j in range(self.x_size):
                self.emit_means[k][j] = np.sum(post_state[:,k] *X[:,j]) / np.sum(post_state[:,k])

            X_cov = np.dot((X-self.emit_means[k]).T, (post_state[:,k]*(X-self.emit_means[k]).T).T)
            self.emit_covars[k] = X_cov / np.sum(post_state[:,k])
            if det(self.emit_covars[k]) == 0: # Handle singular matrix
                self.emit_covars[k] = self.emit_covars[k] + 0.01*np.eye(len(X[0]))


class DiscreteHMM(_BaseHMM):
    """
    HMM with discrete emission probability
    Args:
    emit_prob : discrete probability distribution
    x_num: number of observation types
    x_size is 1 by default
    """
    def __init__(self, n_state=1, x_num=1, iter=20):
        _BaseHMM.__init__(self, n_state=n_state, x_size=1, iter=iter)
        self.emission_prob = np.ones((n_state, x_num)) * (1.0/x_num)  # Initialize emission probability mean
        self.x_num = x_num

    def _init(self, X):
        self.emission_prob = np.random.random(size=(self.n_state,self.x_num))
        for k in range(self.n_state):
            self.emission_prob[k] = self.emission_prob[k]/np.sum(self.emission_prob[k])

    def emit_prob(self, x): # Calculate emission probability for x under state k
        prob = np.zeros(self.n_state)
        # for i in range(self.n_state): prob[i]=self.emission_prob[i][int(x[0])]
        for i in range(self.n_state): prob[i]=self.emission_prob[i][int(x)]
        return prob

    def generate_x(self, z): # Generate x given state p(x|z)
        return np.random.choice(self.x_num, 1, p=self.emission_prob[z][0])

    def emit_prob_updated(self, X, post_state): # Update emission probability
        self.emission_prob = np.zeros((self.n_state, self.x_num))
        X_length = len(X)
        for n in range(X_length):
            self.emission_prob[:,int(X[n])] += post_state[n]

        self.emission_prob+= 0.1/self.x_num
        for k in range(self.n_state):
            if np.sum(post_state[:,k])==0: continue
            self.emission_prob[k] = self.emission_prob[k]/np.sum(post_state[:,k])