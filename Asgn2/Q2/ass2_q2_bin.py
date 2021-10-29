import numpy as np
import math
import pandas as pd
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from scipy import spatial
import sys
import time


print("d = 4 => (d+1) mod 10 = 5")

train_data = pd.read_csv(sys.argv[1])
test_data = pd.read_csv(sys.argv[2])

positive_train_data = pd.DataFrame(train_data.loc[train_data["7"] == 4])
negative_train_data = pd.DataFrame(train_data.loc[train_data["7"] == 5])
positive_train_data["7"] = 1
negative_train_data["7"] = -1
train_data = pd.concat([positive_train_data, negative_train_data])
train_classes = train_data["7"].to_numpy()
train_data = train_data.drop("7", axis = 1)
train_data /= 255
train_data = train_data.to_numpy()

positive_test_data = pd.DataFrame(test_data.loc[test_data["7"] == 4])
negative_test_data = pd.DataFrame(test_data.loc[test_data["7"] == 5])
positive_test_data["7"] = 1
negative_test_data["7"] = -1
test_data = pd.concat([positive_test_data, negative_test_data])
test_classes = test_data["7"].to_numpy()
test_data = test_data.drop("7", axis = 1)
test_data /= 255
test_data = test_data.to_numpy()

part_num = sys.argv[3]

if part_num == "a":

  # calculating parameters for the convex optimiser to optimise the SVM
  # with a linear kernel
  start = time.time()

  m, n = train_data.shape
  train_classes = train_classes.reshape(-1, 1) * 1
  XY = train_classes * train_data
  H = np.dot(XY, XY.T) * 1

  C = 1 # given in problem statement

  P = cvxopt_matrix(H)
  q = cvxopt_matrix(-np.ones((m, 1)))

  g1 = -1 * np.eye(m)
  g2 = C * np.eye(m)
  G = cvxopt_matrix(np.vstack((g1, g2)))

  h1 = np.zeros(m)
  h2 = C * np.ones(m)
  h = cvxopt_matrix(np.hstack((h1, h2)))

  A = cvxopt_matrix(train_classes.reshape(1, -1)*1.)
  b = cvxopt_matrix(np.zeros(1))

  cvxopt_solvers.options["abstol"] = 1e-10
  cvxopt_solvers.options["reltol"] = 1e-10
  cvxopt_solvers.options["feastol"] = 1e-10
  cvxopt_solvers.options["show_progress"] = True

  # solving the convex optimisation problem
  solution = cvxopt_solvers.qp(P, q, G, h, A, b)

  # calculating the optimal SVM with a linear kernel
  alphas = np.array(solution['x'])
  S = (alphas > 1e-3)

  supp_vec_indices = np.where(S == True)[0]
  S = S.flatten()

  w = ((train_classes * alphas).T @ train_data).reshape(-1, 1)
  b = np.mean(train_classes[S] - np.dot(train_data[S], w))

  end = time.time()

  training_time = end-start

  print(f"The training time is : {training_time}")

  def accuracy(actual_classes, preds):
    # calculates the accuracy given the true classification labels
    # and the predicted labels

    correct_preds = 0

    for i in range(len(actual_classes)):

      if preds[i] == actual_classes[i]:
        correct_preds += 1

    return correct_preds*100/len(actual_classes)

  def linear_preds(X, w, b):
    # predicts the class of given Xs using linear kernel with SVM (w, b)

    predictions = [1 if margin >= 0 else -1 for margin in (np.dot(X, w) + b)]

    return predictions


  # calculating predictions on the train_data and test_data
  linear_train_preds = linear_preds(train_data, w, b)
  linear_test_preds = linear_preds(test_data, w, b)

  # calculating accuracy of the classifier on the training and test data
  linear_train_accuracy = accuracy(train_classes, linear_train_preds)
  print(f"The training accuracy is : {linear_train_accuracy}")

  linear_test_accuracy = accuracy(test_classes, linear_test_preds)
  print(f"The accuracy on the test set is : {linear_test_accuracy}")
  sys.exit()


if part_num == "b":

  # calculating parameters for the convex optimiser to optimise the SVM
  # with a gaussian kernel
  start = time.time()

  m, n = train_data.shape
  train_classes = train_classes.reshape(-1, 1) * 1
  Kxz = np.zeros((m, m))
  gamma = 0.05  # given in problem statement

  perp_dist = spatial.distance.pdist(train_data, 'sqeuclidean')
  Kxz = np.exp(-1 * gamma * spatial.distance.squareform(perp_dist))

  H = np.outer(train_classes, train_classes) * Kxz

  C = 1 # given in problem statement

  P = cvxopt_matrix(H)
  q = cvxopt_matrix(-np.ones((m, 1)))

  g1 = -1 * np.eye(m)
  g2 = C * np.eye(m)
  G = cvxopt_matrix(np.vstack((g1, g2)))

  h1 = np.zeros(m)
  h2 = C * np.ones(m)
  h = cvxopt_matrix(np.hstack((h1, h2)))

  A = cvxopt_matrix(train_classes.reshape(1, -1)*1.)
  b = cvxopt_matrix(np.zeros(1))

  cvxopt_solvers.options["abstol"] = 1e-10
  cvxopt_solvers.options["reltol"] = 1e-10
  cvxopt_solvers.options["feastol"] = 1e-10
  cvxopt_solvers.options["show_progress"] = True

  # solving the convex optimisation problem
  solution = cvxopt_solvers.qp(P, q, G, h, A, b)

  # calculating the optimal SVM with a gaussian kernel
  alphas = np.array(solution['x'])
  S = (alphas > 1e-3)

  supp_vec_indices = np.where(S == True)[0]
  S = S.flatten()

  perp_dist = spatial.distance.pdist(train_data[supp_vec_indices], 'sqeuclidean')
  Kxz = np.exp(-1 * gamma * spatial.distance.squareform(perp_dist))

  w = np.dot(Kxz.T, (alphas[S] * train_classes[S]))
  b = np.mean(train_classes[S] - w)

  end = time.time()

  training_time = end - start
  print(f"The training time is : {training_time}")

  def gaussian_preds(train_data, train_classes, X, b, supp_vec_indices, alphas, gamma):
    # predicts the class of given Xs using gaussian kernel with SVM (w, b)

    xz = spatial.distance.cdist(train_data[supp_vec_indices], X, 'sqeuclidean')
    Kxz = np.exp(-1 * gamma * xz)
    w = np.dot(Kxz.T, (alphas[supp_vec_indices] * train_classes[supp_vec_indices]))
    pred_vals = w + b
    print(pred_vals.shape)
    predictions = [1 if val >= 0 else -1 for val in pred_vals]
    return predictions


  def accuracy(actual_classes, preds):
    # calculates the accuracy given the true classification labels
    # and the predicted labels

    correct_preds = 0
    for i in range(len(actual_classes)):
      if preds[i] == actual_classes[i]:
        correct_preds += 1
    return correct_preds*100/len(actual_classes)


  # calculating predictions on the train_data and test_data
  gaussian_train_preds = gaussian_preds(train_data, train_classes, train_data, b, supp_vec_indices, alphas, 0.05)
  gaussian_test_preds = gaussian_preds(train_data, train_classes, test_data, b, supp_vec_indices, alphas, 0.05)

  # calculating accuracy of the classifier on the training and test data
  gaussian_train_accuracy = accuracy(train_classes, gaussian_train_preds)
  print(f"The training accuracy is : {gaussian_train_accuracy}")

  gaussian_test_accuracy = accuracy(test_classes, gaussian_test_preds)
  print(f"The accuracy on the test set is : {gaussian_test_accuracy}")
  sys.exit()


if part_num == "c":

  from libsvm.svmutil import *

  print("Training a SVM with a linear kernel")
  start = time.time()
  linear_svm = svm_train(train_classes.flatten(), train_data, '-c 1 -t 0')
  end = time.time()
  training_time = end-start

  print(f"The training time is : {training_time}")

  print("The accuracy on the training data is:")
  linear_train_preds, linear_train_accuracy,  linear_train_val= svm_predict(train_classes.flatten(), train_data, linear_svm)

  print("The accuracy on the test data is:")
  linear_test_preds, linear_test_accuracy,  linear_test_val= svm_predict(test_classes.flatten(), test_data, linear_svm)

  print("Training a SVM with a gaussian kernel")
  start = time.time()
  gaussian_svm = svm_train(train_classes.flatten(), train_data, '-c 1 -g 0.05 -t 2')
  end = time.time()
  training_time = end-start

  print(f"The training time is : {training_time}")

  print("The accuracy on the training data is:")
  gaussian_train_preds, gaussian_train_accuracy, gaussian_train_val = svm_predict(train_classes.flatten(), train_data, gaussian_svm)

  print("The accuracy on the test data is:")
  gaussian_test_preds, gaussian_test_accuracy, gaussian_test_val = svm_predict(test_classes.flatten(), test_data, gaussian_svm)

  sys.exit()