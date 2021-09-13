# importing useful libraries
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import sys
import time

# Taking input from user and setting constants
LEARNING_RATE = float(input("learning_rate : "))
STOPPING_CRITERIA = float(input("stopping_criteria : "))
BATCH_SIZE = int(input("batch_size : "))
THETA_INIT = np.zeros((3,1))

# loading test data
test_data = pd.read_csv("q2/q2test.csv")
test_data["X_0"] = 1

x_test = test_data[["X_0", "X_1", "X_2"]].to_numpy()

y_test = test_data["Y"].to_numpy()
y_test = np.reshape(y_test, (x_test.shape[0], 1))



# Sampling 1 million data points with 
# X0 = 1, X1 ~ N(3, 4), X2 ~ N(-1, 4)
# theta = [3,1,2] and noise (epsilon) ~ N(0, 2)
theta0 = np.array([[3],[1],[2]])

X_0 = np.ones((1000000, 1))
X_1 = np.random.normal(3,2,1000000).reshape(1000000, 1)
X_2 = np.random.normal(-1, 2, 1000000).reshape(1000000, 1)

noise = np.random.normal(0, np.sqrt(2), 1000000).reshape(1000000, 1)

X_data = np.append(X_0, X_1, axis = 1)
X_data = np.append(X_data, X_2, axis = 1)

Y_data = np.dot(X_data, theta0) + noise



# shuffle the data
shuffling = np.append(X_data, Y_data, axis = 1)
np.random.shuffle(shuffling)

X_data = shuffling[:, 0:3]
Y_data = shuffling[:, 3:4]



# helper function
# returns the predicted values given the x values and parameters
def hypothesis(x, theta):
  return np.dot(x, theta)

# helper function
# returns the average mean squared error per data point given the predictions and y values
def average_error(y_pred, y, m):
  return np.sum(np.square(y-y_pred))/(2*m)



def stochastic_gradient_descent(X, Y, learning_rate, stopping_criteria, theta_init, batch_size):
  
  
  # starting timer to track runtime 
  start = time.time()
  
  # initialising parameters
  theta = theta_init
  theta_list = [theta]
  
  # initialising variables
  J_list = [average_error(np.dot(X[0:batch_size], theta), Y[0:batch_size], batch_size)]
  J_final = 0
  J_init = 0
  del_J = 0
  epoch_J = 0
  epoch = 0
  batch_num = 0
  num_batches = X.shape[0]/batch_size
  sample_size = int(min(num_batches, 1000))
  epoch_J_list = [average_error(np.dot(X[-sample_size*batch_size:], theta), Y[-sample_size*batch_size:], sample_size*batch_size)*sample_size]
  
  while(True):
    
    # at the end of every epoch we check if the change in average error 
    # per data point is less than the stopping criteria
    if(batch_num == num_batches):
      epoch = epoch + 1
      epoch_J = np.sum(J_list[-sample_size:])
      del_J = abs(epoch_J - epoch_J_list[-1])
      epoch_J_list.append(epoch_J)
      sys.stdout.write(f"Epoch: {epoch}, epoch_del_J: {del_J/sample_size}   \r")
      sys.stdout.flush()
      if(del_J/(sample_size) <= stopping_criteria):
        break
      else:
        batch_num = 0
        epoch_J = 0
    
    # slicing the data to get the minibatch for current iteration
    k = int(batch_num*batch_size)
    mb_x = X_data[k:k+batch_size]
    mb_y = Y_data[k:k+batch_size]
    
    # updating parameters and variables
    J_init = average_error(np.dot(mb_x, theta), mb_y, batch_size)
    v = np.dot(mb_x.T, np.dot(mb_x, theta) - mb_y)
    theta = theta - (learning_rate*v)/batch_size
    theta_list.append(theta)
    J_final = average_error(np.dot(mb_x, theta), mb_y, batch_size)
    J_list.append(J_final)
    if(len(J_list) > 2000):
      J_list = J_list[-1000:]
    
    batch_num = batch_num + 1
    
  end = time.time()
  sys.stdout.write("\n")
  sys.stdout.flush()
  sys.stdout.write(f"{end-start}                                    \r")
  sys.stdout.flush()
  sys.stdout.write("\n")
  sys.stdout.flush()
  
  # returns
  # theta : the final parameter vector
  # theta_list : list of parameter vectors after at each iteration
  # J_list : list of values of J(ϑ) at each iteration for the minibatch
  return (theta, theta_list, J_list)



theta, theta_list, J_list = stochastic_gradient_descent(X_data, Y_data, LEARNING_RATE, STOPPING_CRITERIA, THETA_INIT, BATCH_SIZE)

print("final theta : ", theta)

# calculate the differnce in average error between learned and original hypothesis
e1 = average_error(np.dot(x_test, theta0), y_test, 1000000)
print("test error of the original hypothesis : ", e1)
e2 = average_error(np.dot(x_test, theta), y_test, 1000000)
print("test error of the trained hypothesis : ", e2)
print("difference in test error : ", abs(e2-e1))


# manipulating data to plot
theta0_list = []
theta1_list = []
theta2_list = []
for i in range(0, len(theta_list), 50):
  theta0_list.append(theta_list[i][0][0])
  theta1_list.append(theta_list[i][1][0])
  theta2_list.append(theta_list[i][2][0])
theta0_list = np.array(theta0_list)
theta1_list = np.array(theta1_list)
theta2_list = np.array(theta2_list)

# plotting the movement of ϑ
fig = plt.figure()
ax = plt.axes(projection = "3d")
ax.scatter3D(theta0_list, theta1_list, theta2_list, color = "green", label = "theta")
plt.legend()
ax.set_xlabel("theta0")
ax.set_ylabel("theta1")
ax.set_zlabel("theta2")
plt.show()
# plt.savefig(f"batch_{BATCH_SIZE}_theta_movement.png")
plt.close()