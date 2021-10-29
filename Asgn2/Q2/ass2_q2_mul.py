import numpy as np
import math
import pandas as pd
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from scipy import spatial
from itertools import combinations
import pickle
import time
import seaborn as sns
import matplotlib.pyplot as plt
import sys

training_data = pd.read_csv(sys.argv[1])
testing_data = pd.read_csv(sys.argv[2])

def get_data(data, pos_class, neg_class):
  positive_X = pd.DataFrame(data.loc[data["7"] == pos_class])
  negative_X = pd.DataFrame(data.loc[data["7"] == neg_class])
  positive_X["7"] = 1
  negative_X["7"] = -1
  X = pd.concat([positive_X, negative_X])
  Y = X["7"].to_numpy()
  X = X.drop("7", axis = 1)
  X /= 255
  X = X.to_numpy()
  return X, Y


part_num = sys.argv[3]


# load the test data
test_classes = testing_data["7"].to_numpy()
test_data = testing_data.drop("7", axis = 1)
test_data /= 255
test_data = test_data.to_numpy()


if part_num == "a":

  def train_gaussian_svm(X, Y, C, gamma):
    
    # trains a gaussian svm given data of two classes
    m, n = X.shape
    Y = Y.reshape(-1, 1) * 1
    Kxz = np.zeros((m, m))

    perp_dist = spatial.distance.pdist(X, 'sqeuclidean')
    Kxz = np.exp(-1 * gamma * spatial.distance.squareform(perp_dist))

    H = np.outer(Y, Y) * Kxz

    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((m, 1)))

    g1 = -1 * np.eye(m)
    g2 = C * np.eye(m)
    G = cvxopt_matrix(np.vstack((g1, g2)))

    h1 = np.zeros(m)
    h2 = C * np.ones(m)
    h = cvxopt_matrix(np.hstack((h1, h2)))

    A = cvxopt_matrix(Y.reshape(1, -1)*1.)
    b = cvxopt_matrix(np.zeros(1))

    cvxopt_solvers.options["abstol"] = 1e-10
    cvxopt_solvers.options["reltol"] = 1e-10
    cvxopt_solvers.options["feastol"] = 1e-10
    cvxopt_solvers.options["show_progress"] = False

    # solving the convex optimisation problem
    solution = cvxopt_solvers.qp(P, q, G, h, A, b)

    # calculating the optimal SVM with a gaussian kernel
    alphas = np.array(solution['x'])
    S = (alphas > 1e-3)

    supp_vec_indices = np.where(S == True)[0]
    S = S.flatten()

    perp_dist = spatial.distance.pdist(X[supp_vec_indices], 'sqeuclidean')
    Kxz = np.exp(-1 * gamma * spatial.distance.squareform(perp_dist))

    w = np.dot(Kxz.T, (alphas[S] * Y[S]))
    b = np.mean(Y[S] - w)

    return b, supp_vec_indices, alphas


  def gaussian_preds(train_data, train_classes, X, b, supp_vec_indices, alphas, gamma):
    # predicts the class of given Xs using gaussian kernel with SVM (w, b)

    xz = spatial.distance.cdist(train_data[supp_vec_indices], X, 'sqeuclidean')
    Kxz = np.exp(-1 * gamma * xz)
    w = np.dot(Kxz.T, (alphas[supp_vec_indices] * train_classes[supp_vec_indices]))
    pred_vals = w + b
    return pred_vals


  def accuracy(actual_classes, preds):
    # calculates the accuracy given the true classification labels
    # and the predicted labels

    correct_preds = 0
    for i in range(len(actual_classes)):
      if preds[i] == actual_classes[i]:
        correct_preds += 1
    return correct_preds*100/len(actual_classes)


  # uncomment the following section to train the models

  
  class_pairs = combinations(range(10), 2)

  print("Training the 10C2 gaussian SVMs")
  g_svms = []
  start = time.time()
  for k in list(class_pairs):
    
    # load the training data
    X, Y = get_data(training_data, k[0], k[1])
    g_svms.append(train_gaussian_svm(X, Y, 1, 0.05))
  end = time.time()
  training_time = end-start

  # save the model as a pockle file
  f = open('gaussian_svm.pickle', 'wb')
  pickle.dump(g_svms, f)

  """
  # load the svm model data
  g_svms = pickle.load(open('gaussian_svm.pickle', 'rb'))
  """

  # generating predictions for the test data
  class_pairs = list(combinations(range(10), 2))

  svm_values = np.zeros((len(test_data), 10))


  for i in range(len(class_pairs)):
  
    k = class_pairs[i]
  
    train_data, train_classes = get_data(training_data, k[0], k[1])
    train_classes = train_classes.reshape(-1, 1) * 1
    
    b, supp_vec_indices, alphas = g_svms[i]
    test_pred_vals = gaussian_preds(train_data, train_classes, test_data, b, supp_vec_indices, alphas, 0.05)

    class0 = np.where(test_pred_vals >= 0)[0]
    class1 = np.where(test_pred_vals < 0)[0]
    
    svm_values[class0, k[0]] += 1/(1 + np.exp(-np.abs(test_pred_vals[class0]))).flatten()
    svm_values[class1, k[1]] += 1/(1 + np.exp(-np.abs(test_pred_vals[class1]))).flatten()

  test_preds = [np.where(x == max(x)) for x in svm_values[:, :]]

  # calculating the accuracy of the gaussian models on the test data
  test_acc = accuracy(test_classes, test_preds)

  f = open('my_test_preds.pickle', 'wb')
  pickle.dump(test_preds, f)

  print(f"The training time is : {training_time}")

  print(f"The test accuracy is : {test_acc}")
  sys.exit()


if part_num == "b":

  from libsvm.svmutil import *

  training_data = pd.read_csv(sys.argv[1])
  testing_data = pd.read_csv(sys.argv[2])
  train_classes = training_data["7"].to_numpy()
  train_data = training_data.drop("7", axis = 1)
  train_data /= 255
  train_data = train_data.to_numpy()
  test_classes = testing_data["7"].to_numpy()
  test_data = testing_data.drop("7", axis = 1)
  test_data /= 255
  test_data = test_data.to_numpy()

  start = time.time()
  gaussian_svm = svm_train(train_classes.flatten(), train_data, '-c 1 -g 0.05 -t 2')
  end = time.time()
  training_time = end-start
  print(f"The gaussian SVM models are trained in : {training_time}")

  print("The accuracy of the libsvm gaussian models on the test data is :")
  gaussian_test_preds, gaussian_test_accuracy, gaussian_test_val = svm_predict(test_classes.flatten(), test_data, gaussian_svm)

  f = open('libsvm_test_preds.pickle', 'wb')
  pickle.dump(gaussian_test_preds, f)

  sys.exit()


if part_num == "c":

  my_test_preds = pickle.load(open('my_test_preds.pickle', 'rb'))
  libsvm_test_preds = pickle.load(open('libsvm_test_preds.pickle', 'rb'))

  my_test_preds = np.array(my_test_preds)

  confusion_matrix = np.zeros((10, 10), dtype = np.int)
  for i in range(len(my_test_preds)):
    confusion_matrix[int(test_classes[i])-1][int(my_test_preds[i])-1] += 1

  fig = plt.figure()
  ax = fig.add_subplot(111)
  sns.heatmap(confusion_matrix, annot = True, fmt = "d", ax = ax, cmap = 'viridis')

  ax.set_ylabel("Actual Label")
  ax.set_xlabel("Predicted Label")
  ax.set_title("my svm confusion matrix")

  fig.savefig("my_svm_conf_matrix.png")

  confusion_matrix = np.zeros((10, 10), dtype = np.int)
  for i in range(len(libsvm_test_preds)):
    confusion_matrix[int(test_classes[i])-1][int(libsvm_test_preds[i])-1] += 1

  fig = plt.figure()
  ax = fig.add_subplot(111)
  sns.heatmap(confusion_matrix, annot = True, fmt = "d", ax = ax, cmap = 'viridis')

  ax.set_ylabel("Actual Label")
  ax.set_xlabel("Predicted Label")
  ax.set_title("libsvm confusion matrix")

  fig.savefig("libsvm_conf_matrix.png")

  sys.exit()
