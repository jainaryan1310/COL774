import sys
import numpy as np
import json
import math
import string
import matplotlib.pyplot as plt
import seaborn as sns


n = len(sys.argv)

if n != 4:
	print("Please check the instruction the correct format is:\n ./run.sh 1 <path_of_train_data> <path_of_test_data> <part_num>")
	sys.exit()


train_data = open(sys.argv[1], "r")
X_train = train_data.readlines()
test_data = open(sys.argv[2], "r")
X_test = test_data.readlines()

part_num = sys.argv[3]

if part_num == "a":
  """              PART A                   """

  # extracting useful components of the data
  test_data = []
  for x in X_test:
    x = json.loads(x)
    rating = x["overall"]
    text = x["reviewText"]
    test_data.append((text,rating))
    
  train_data = []
  for x in X_train:
    x = json.loads(x)
    rating = x["overall"]
    text = x["reviewText"]
    train_data.append((text,rating))


  # dictionary[0] is the global dictionary
  dictionary = {0:{}, 1:{}, 2:{}, 3:{}, 4:{}, 5:{}}

  # n[0] is the size of the global dictionary
  n = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}

  class_labels = {1:0, 2:0, 3:0, 4:0, 5:0}

  # building the dictionaries
  for x in train_data:
    text = x[0]
    rating = x[1]
    class_labels[rating] += 1
    words = text.split()
    n[0] += len(words)
    n[rating] += len(words)
    
    for word in words:
      if word in dictionary[0]:
        dictionary[0][word] += 1
      else:
        dictionary[0][word] = 1
        
      if word in dictionary[rating]:
        dictionary[rating][word] += 1
      else:
        dictionary[rating][word] = 1


  # calculating phi
  phi = {}
  for rating in range(1,6):
    phi[rating] = class_labels[rating]/len(X_train)


  def naive_bayes(dictionary, n, alpha):
    """
    This function is used to train the theta paramters
    """
    vocab_size = len(dictionary[0])
    theta = {}
    for rating in range(1, 6):
      theta[rating] = {}
    
    for word in dictionary[0].keys():
      for rating in range(1, 6):
        if word in dictionary[rating]:
          theta[rating][word] = ((dictionary[rating][word] + alpha)/(n[rating] + alpha * vocab_size))
        else:
          theta[rating][word] = (alpha/(n[rating] + alpha*vocab_size))
    
    return theta


  def predict(theta, x_text, phi):
    """
    This function is used to predict the class using the learned theta parameters.
    """

    class_scores = {1:0, 2:0, 3:0, 4:0, 5:0}
    words = x_text.split()
    
    for word in words:
      for rating in range(1, 6):
        if word in theta[rating]:
          class_scores[rating] += math.log(theta[rating][word])
    
    for rating in range(1, 6):
      class_scores[rating] += math.log(phi[rating])
      
    max_prob = class_scores[1]
    pred = 1
    
    for rating in range(2, 6):
      if class_scores[rating] > max_prob:
        max_prob = class_scores[rating]
        pred = rating
    
    return pred


  def accuracy(data, theta, phi):
    """
    This function is used to calculate the accuracy for a set of predictions given
    the correct classification labels
    """
    correct_preds = 0
    predictions = []
    actual_ratings = []
    
    for x in data:
      actual_rating = x[1]
      pred_rating = predict(theta, x[0], phi)
      
      actual_ratings.append(actual_rating)
      predictions.append(pred_rating)
      
      if actual_rating - pred_rating == 0:
        correct_preds += 1
        
    acc = correct_preds*100/len(data)
    return acc, predictions, actual_ratings


  # training theta parameters
  theta = naive_bayes(dictionary, n, 1)

  # calculating the training accuracy
  train_acc, train_preds, train_data_actual = accuracy(train_data, theta, phi)
  print(f"The training accuracy is : {train_acc}")

  # calculating the accuracy on the test_set
  test_acc, test_data_preds, test_data_actual = accuracy(test_data, theta, phi)
  print(f"The accuracy on the test set is : {test_acc}")
  sys.exit()


if part_num == "b":
  """              PART B                 """
  import random

  # extracting useful components of the data
  test_data = []
  for x in X_test:
    x = json.loads(x)
    rating = x["overall"]
    text = x["reviewText"]
    test_data.append((text,rating))

  train_data = []
  for x in X_train:
    x = json.loads(x)
    rating = x["overall"]
    text = x["reviewText"]
    train_data.append((text,rating))

  # randomly generating random predictions for the test_data
  def random_pred(num_predictions, num_classes):
    preds = [random.randint(1, 1+num_classes) for i in range(num_predictions)]
    return preds

  correct_preds = 0
  random_preds = random_pred(len(test_data), 5)

  for i in range(len(test_data)):
    actual_rating = test_data[i][1]
    pred_rating = random_preds[i]
    
    if actual_rating - pred_rating == 0:
      correct_preds += 1

  random_preds_acc = correct_preds*100/len(test_data)
  print(f"The accuracy for randomly generated predictions is : {random_preds_acc}")


  class_labels = {1:0, 2:0, 3:0, 4:0, 5:0}

  for x in train_data:
    rating = x[1]
    class_labels[rating] += 1


  # calculating phi
  phi = {}
  for rating in range(1,6):
    phi[rating] = class_labels[rating]/len(X_train)


  # predicting the maximum occurring class always
  correct_preds = 0
  max_class = max(phi, key = lambda x : phi[x])
  max_preds = [max_class for i in range(len(train_data))]

  for i in range(len(train_data)):
    actual_rating = train_data[i][1]
    pred_rating = max_preds[i]
    
    if actual_rating - pred_rating == 0:
      correct_preds += 1

  max_preds_acc = correct_preds*100/len(train_data)
  print(f"The accuracy for predicting the most occurring class always is : {max_preds_acc}")
  sys.exit()


if part_num == "c":


  # extracting useful components of the data
  test_data = []
  for x in X_test:
    x = json.loads(x)
    rating = x["overall"]
    text = x["reviewText"]
    test_data.append((text,rating))
    
  train_data = []
  for x in X_train:
    x = json.loads(x)
    rating = x["overall"]
    text = x["reviewText"]
    train_data.append((text,rating))


  # dictionary[0] is the global dictionary
  dictionary = {0:{}, 1:{}, 2:{}, 3:{}, 4:{}, 5:{}}

  # n[0] is the size of the global dictionary
  n = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}

  class_labels = {1:0, 2:0, 3:0, 4:0, 5:0}

  # building the dictionaries
  for x in train_data:
    text = x[0]
    rating = x[1]
    class_labels[rating] += 1
    words = text.split()
    n[0] += len(words)
    n[rating] += len(words)
    
    for word in words:
      if word in dictionary[0]:
        dictionary[0][word] += 1
      else:
        dictionary[0][word] = 1
        
      if word in dictionary[rating]:
        dictionary[rating][word] += 1
      else:
        dictionary[rating][word] = 1


  # calculating phi
  phi = {}
  for rating in range(1,6):
    phi[rating] = class_labels[rating]/len(X_train)


  def naive_bayes(dictionary, n, alpha):
    """
    This function is used to train the theta paramters
    """
    vocab_size = len(dictionary[0])
    theta = {}
    for rating in range(1, 6):
      theta[rating] = {}
    
    for word in dictionary[0].keys():
      for rating in range(1, 6):
        if word in dictionary[rating]:
          theta[rating][word] = ((dictionary[rating][word] + alpha)/(n[rating] + alpha * vocab_size))
        else:
          theta[rating][word] = (alpha/(n[rating] + alpha*vocab_size))
    
    return theta


  def predict(theta, x_text, phi):
    """
    This function is used to predict the class using the learned theta parameters.
    """

    class_scores = {1:0, 2:0, 3:0, 4:0, 5:0}
    words = x_text.split()
    
    for word in words:
      for rating in range(1, 6):
        if word in theta[rating]:
          class_scores[rating] += math.log(theta[rating][word])
    
    for rating in range(1, 6):
      class_scores[rating] += math.log(phi[rating])
      
    max_prob = class_scores[1]
    pred = 1
    
    for rating in range(2, 6):
      if class_scores[rating] > max_prob:
        max_prob = class_scores[rating]
        pred = rating
    
    return pred



  def accuracy(data, theta, phi):
    """
    This function is used to calculate the accuracy for a set of predictions given
    the correct classification labels
    """
    correct_preds = 0
    predictions = []
    actual_ratings = []
    
    for x in data:
      actual_rating = x[1]
      pred_rating = predict(theta, x[0], phi)
      
      actual_ratings.append(actual_rating)
      predictions.append(pred_rating)
      
      if actual_rating - pred_rating == 0:
        correct_preds += 1
        
    acc = correct_preds*100/len(data)
    return acc, predictions, actual_ratings


  # training theta parameters
  theta = naive_bayes(dictionary, n, 1)

  # calculating the accuracy and predictions on the test_set
  test_acc, test_data_preds, test_data_actual = accuracy(test_data, theta, phi)
  print(f"The accuracy on the test set is : {test_acc}") 


  # calculating the confusion matrix
  confusion_matrix = np.zeros((5, 5), dtype = np.int)
  for i in range(len(test_data_actual)):
    confusion_matrix[int(test_data_actual[i])-1][test_data_preds[i]-1] += 1


  # drawing the confusion matrix
  fig = plt.figure()
  ax = fig.add_subplot(111)
  sns.heatmap(confusion_matrix, annot = True, fmt = "d", ax = ax, cmap = 'viridis')

  ax.set_ylabel("Actual Label")
  ax.set_xlabel("Predicted Label")
  ax.set_title("naive_bayes test_data confusion matrix")
  ax.xaxis.set_ticklabels(['1','2', '3', '4', '5'])
  ax.yaxis.set_ticklabels(['1','2', '3', '4', '5'])

  fig.savefig("nb_test_conf.png")
  sys.exit()


from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

if part_num == "d":

  stemmer = PorterStemmer()
  stop_words = set(stopwords.words('english'))


  # extracting useful components of the data
  test_data = []
  for x in X_test:
    x = json.loads(x)
    rating = x["overall"]
    text = x["reviewText"]
    test_data.append((text,rating))
    
  train_data = []
  for x in X_train:
    x = json.loads(x)
    rating = x["overall"]
    text = x["reviewText"]
    train_data.append((text,rating))


  # dictionary[0] is the global dictionary
  dictionary = {0:{}, 1:{}, 2:{}, 3:{}, 4:{}, 5:{}}

  # n[0] is the size of the global dictionary
  n = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}

  class_labels = {1:0, 2:0, 3:0, 4:0, 5:0}

  for x in train_data:
    
    text = x[0]
    rating = x[1]
    class_labels[rating] += 1
    
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    words = text.split()
    
    for word in words:
      
      if word in stop_words:
        continue
        
      word = stemmer.stem(word)
      
      n[0] += 1
      n[rating] += 1
      
      if word in dictionary[0]:
        dictionary[0][word] += 1
      else:
        dictionary[0][word] = 1
        
      if word in dictionary[rating]:
        dictionary[rating][word] += 1
      else:
        dictionary[rating][word] = 1

  # calculating phi
  phi = {}
  for rating in range(1,6):
    phi[rating] = class_labels[rating]/len(X_train)


  def naive_bayes(dictionary, n, alpha):
    """
    This function is used to train the theta paramters
    """
    vocab_size = len(dictionary[0])
    theta = {}
    for rating in range(1, 6):
      theta[rating] = {}
    
    for word in dictionary[0].keys():
      for rating in range(1, 6):
        if word in dictionary[rating]:
          theta[rating][word] = ((dictionary[rating][word] + alpha)/(n[rating] + alpha * vocab_size))
        else:
          theta[rating][word] = (alpha/(n[rating] + alpha*vocab_size))
    
    return theta


  def predict(theta, x_text, phi):
    """
    This function is used to predict the class using the learned theta parameters.
    """
    class_scores = {1:0, 2:0, 3:0, 4:0, 5:0}
    x_text = x_text.translate(str.maketrans('', '', string.punctuation)).lower()
    words = x_text.split()
    
    for word in words:
      
      if word in stop_words:
        continue
        
      word = stemmer.stem(word)
      
      for rating in range(1, 6):
        if word in theta[rating]:
          class_scores[rating] += math.log(theta[rating][word])
    
    for rating in range(1, 6):
      class_scores[rating] += math.log(phi[rating])
      
    max_prob = class_scores[1]
    pred = 1
    
    for rating in range(2, 6):
      if class_scores[rating] > max_prob:
        max_prob = class_scores[rating]
        pred = rating
    
    return pred


  def accuracy(data, theta, phi):
    """
    This function is used to calculate the accuracy for a set of predictions given
    the correct classification labels
    """
    correct_preds = 0
    predictions = []
    actual_ratings = []
    
    for x in data:
      actual_rating = x[1]
      pred_rating = predict(theta, x[0], phi)
      
      actual_ratings.append(actual_rating)
      predictions.append(pred_rating)
      
      if actual_rating - pred_rating == 0:
        correct_preds += 1
        
    acc = correct_preds*100/len(data)
    return acc, predictions, actual_ratings

  # training theta parameters
  theta = naive_bayes(dictionary, n, 1)

  # calculating the training accuracy
  train_acc, train_preds, train_data_actual = accuracy(train_data, theta, phi)
  print(f"The training accuracy is : {train_acc}")

  # calculating the accuracy on the test_set
  test_acc, test_data_preds, test_data_actual = accuracy(test_data, theta, phi)
  print(f"The accuracy on the test set is : {test_acc}")
  sys.exit()


if part_num == "e":

  stemmer = PorterStemmer()
  stop_words = set(stopwords.words('english'))


  # extracting useful components of the data
  test_data = []
  for x in X_test:
    x = json.loads(x)
    rating = x["overall"]
    text = x["reviewText"]
    test_data.append((text,rating))
    
  train_data = []
  for x in X_train:
    x = json.loads(x)
    rating = x["overall"]
    text = x["reviewText"]
    train_data.append((text,rating))


  ################################## using Bi-Grams ####################################

  # dictionary[0] is the global dictionary
  dictionary = {0:{}, 1:{}, 2:{}, 3:{}, 4:{}, 5:{}}

  # n[0] is the size of the global dictionary
  n = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}

  class_labels = {1:0, 2:0, 3:0, 4:0, 5:0}

  for x in train_data:
    
    text = x[0]
    rating = x[1]
    class_labels[rating] += 1
    
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    words = text.split()
    
    i = 0
    while i < len(words)-1:
    
      first_word = words[i]
      i += 1
      second_word = words[i]
      i += 1
    
      term = ""
      
      if first_word in stop_words:
        first_word = ""
      else:
        first_word = stemmer.stem(first_word)
        
      if second_word in stop_words:
        second_word = ""
      else:
        second_word = stemmer.stem(second_word)
        
        
      if first_word == "" or second_word == "":
        term = first_word + second_word
      else:
        term = first_word + " " + second_word
      
      if term == "":
        continue
        
      n[0] += 1
      n[rating] += 1
        
      if term in dictionary[0]:
        dictionary[0][term] += 1
      else:
        dictionary[0][term] = 1
        
      if term in dictionary[rating]:
        dictionary[rating][term] += 1
      else:
        dictionary[rating][term] = 1


  phi = {}
  for rating in range(1,6):
    phi[rating] = class_labels[rating]/len(X_train)

  def naive_bayes(dictionary, n, alpha):

    vocab_size = len(dictionary[0])
    theta = {}
    for rating in range(1, 6):
      theta[rating] = {}
    
    for word in dictionary[0].keys():
      for rating in range(1, 6):
        if word in dictionary[rating]:
          theta[rating][word] = ((dictionary[rating][word] + alpha)/(n[rating] + alpha * vocab_size))
        else:
          theta[rating][word] = (alpha/(n[rating] + alpha*vocab_size))
    
    return theta

  def predict(theta, x_text, phi):

    class_scores = {1:0, 2:0, 3:0, 4:0, 5:0}
    x_text = x_text.translate(str.maketrans('', '', string.punctuation)).lower()
    words = x_text.split()
    
    for word in words:
      
      if word in stop_words:
        continue
        
      word = stemmer.stem(word)
      
      for rating in range(1, 6):
        if word in theta[rating]:
          class_scores[rating] += math.log(theta[rating][word])
    
    for rating in range(1, 6):
      class_scores[rating] += math.log(phi[rating])
      
    max_prob = class_scores[1]
    pred = 1
    
    for rating in range(2, 6):
      if class_scores[rating] > max_prob:
        max_prob = class_scores[rating]
        pred = rating
    
    return pred

  def accuracy(data, theta, phi):
    
    correct_preds = 0
    predictions = []
    actual_ratings = []
    
    for x in data:
      actual_rating = x[1]
      pred_rating = predict(theta, x[0], phi)
      
      actual_ratings.append(actual_rating)
      predictions.append(pred_rating)
      
      if actual_rating - pred_rating == 0:
        correct_preds += 1
        
    acc = correct_preds*100/len(data)
    return acc, predictions, actual_ratings

  print("Using Bi-Grams")

  # training theta parameters
  theta = naive_bayes(dictionary, n, 1)

  # calculating the training accuracy
  train_acc, train_preds, train_data_actual = accuracy(train_data, theta, phi)
  print(f"The training accuracy is : {train_acc}")

  # calculating the accuracy on the test_set
  test_acc, test_data_preds, test_data_actual = accuracy(test_data, theta, phi)
  print(f"The accuracy on the test set is : {test_acc}")


  ################################## using Tri-Grams ####################################

  # dictionary[0] is the global dictionary
  dictionary = {0:{}, 1:{}, 2:{}, 3:{}, 4:{}, 5:{}}

  # n[0] is the size of the global dictionary
  n = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}

  class_labels = {1:0, 2:0, 3:0, 4:0, 5:0}

  for x in train_data:
    
    text = x[0]
    rating = x[1]
    class_labels[rating] += 1
    
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    words = text.split()
    
    i = 0
    while i < len(words)-2:
    
      first_word = words[i]
      i += 1
      second_word = words[i]
      i += 1
      third_word = words[i]
      i += 1
      
      term = ""
      
      if first_word in stop_words:
        first_word = ""
      else:
        first_word = stemmer.stem(first_word)
        
      if second_word in stop_words:
        second_word = ""
      else:
        second_word = stemmer.stem(second_word)
        
      if third_word in stop_words:
        third_word = ""
      else:
        third_word = stemmer.stem(third_word)
      
      term = first_word + " " + second_word + " " + third_word
      
      term = term.strip()
      
      if term == "":
        continue
        
      n[0] += 1
      n[rating] += 1
        
      if term in dictionary[0]:
        dictionary[0][term] += 1
      else:
        dictionary[0][term] = 1
        
      if term in dictionary[rating]:
        dictionary[rating][term] += 1
      else:
        dictionary[rating][term] = 1

  phi = {}
  for rating in range(1,6):
    phi[rating] = class_labels[rating]/len(X_train)

  def naive_bayes(dictionary, n, alpha):
    vocab_size = len(dictionary[0])
    theta = {}
    for rating in range(1, 6):
      theta[rating] = {}
    
    for word in dictionary[0].keys():
      for rating in range(1, 6):
        if word in dictionary[rating]:
          theta[rating][word] = ((dictionary[rating][word] + alpha)/(n[rating] + alpha * vocab_size))
        else:
          theta[rating][word] = (alpha/(n[rating] + alpha*vocab_size))
    
    return theta

  def predict(theta, x_text, phi):
    class_scores = {1:0, 2:0, 3:0, 4:0, 5:0}
    x_text = x_text.translate(str.maketrans('', '', string.punctuation)).lower()
    words = x_text.split()
    
    i = 0
    while i < len(words)-2:
    
      first_word = words[i]
      i += 1
      second_word = words[i]
      i += 1
      third_word = words[i]
      i += 1
      
      term = ""
      
      if first_word in stop_words:
        first_word = ""
      else:
        first_word = stemmer.stem(first_word)
        
      if second_word in stop_words:
        second_word = ""
      else:
        second_word = stemmer.stem(second_word)
        
      if third_word in stop_words:
        third_word = ""
      else:
        third_word = stemmer.stem(third_word)
      
      term = first_word + " " + second_word + " " + third_word
      
      term = term.strip()
      
      if term == "":
        continue
      
      for rating in range(1, 6):
        if term in theta[rating]:
          class_scores[rating] += math.log(theta[rating][term])
    
    for rating in range(1, 6):
      class_scores[rating] += math.log(phi[rating])
      
    max_prob = class_scores[1]
    pred = 1
    
    for rating in range(2, 6):
      if class_scores[rating] > max_prob:
        max_prob = class_scores[rating]
        pred = rating
    
    return pred

  def accuracy(data, theta, phi):
    
    correct_preds = 0
    predictions = []
    actual_ratings = []
    
    for x in data:
      actual_rating = x[1]
      pred_rating = predict(theta, x[0], phi)
      
      actual_ratings.append(actual_rating)
      predictions.append(pred_rating)
      
      if actual_rating - pred_rating == 0:
        correct_preds += 1
        
    acc = correct_preds*100/len(data)
    return acc, predictions, actual_ratings

  print("Using Tri-Grams")

  # training theta parameters
  theta = naive_bayes(dictionary, n, 1)

  # calculating the training accuracy
  train_acc, train_preds, train_data_actual = accuracy(train_data, theta, phi)
  print(f"The training accuracy is : {train_acc}")

  # calculating the accuracy on the test_set
  test_acc, test_data_preds, test_data_actual = accuracy(test_data, theta, phi)
  print(f"The accuracy on the test set is : {test_acc}")


  ################################## removing words of low occurrence ####################################

  # dictionary[0] is the global dictionary
  dictionary = {0:{}, 1:{}, 2:{}, 3:{}, 4:{}, 5:{}}

  # n[0] is the size of the global dictionary
  n = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}

  class_labels = {1:0, 2:0, 3:0, 4:0, 5:0}

  for x in train_data:
    
    text = x[0]
    rating = x[1]
    class_labels[rating] += 1
    
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    words = text.split()
    
    for word in words:
      
      if word in stop_words:
        continue
        
      word = stemmer.stem(word)
      
      n[0] += 1
      n[rating] += 1
      
      if word in dictionary[0]:
        dictionary[0][word] += 1
      else:
        dictionary[0][word] = 1
        
      if word in dictionary[rating]:
        dictionary[rating][word] += 1
      else:
        dictionary[rating][word] = 1

  phi = {}
  for rating in range(1,6):
    phi[rating] = class_labels[rating]/len(X_train)

  new_dictionary = {0:{}, 1:{}, 2:{}, 3:{}, 4:{}, 5:{}}
  for i in range(0, 6):
    for key in dictionary[i]:
      if dictionary[i][key] >= 50:
        new_dictionary[i][key] = dictionary[i][key]
      else:
        n[0] -= dictionary[0][key]
        n[i] -= dictionary[i][key]

  def naive_bayes(dictionary, n, alpha):
    vocab_size = len(dictionary[0])
    theta = {}
    for rating in range(1, 6):
      theta[rating] = {}
    
    for word in dictionary[0].keys():
      for rating in range(1, 6):
        if word in dictionary[rating]:
          if dictionary[rating][word] > 500:
            theta[rating][word] = 1
          else:
            theta[rating][word] = ((dictionary[rating][word] + alpha)/(n[rating] + alpha * vocab_size))
        else:
          theta[rating][word] = (alpha/(n[rating] + alpha*vocab_size))
    
    return theta

  def predict(theta, x_text, phi):
    class_scores = {1:0, 2:0, 3:0, 4:0, 5:0}
    x_text = x_text.translate(str.maketrans('', '', string.punctuation)).lower()
    words = x_text.split()
    
    for word in words:
      
      if word in stop_words:
        continue
        
      word = stemmer.stem(word)
      
      for rating in range(1, 6):
        if word in theta[rating]:
          class_scores[rating] += math.log(theta[rating][word])
    
    for rating in range(1, 6):
      class_scores[rating] += math.log(phi[rating])
      
    max_prob = class_scores[1]
    pred = 1
    
    for rating in range(2, 6):
      if class_scores[rating] > max_prob:
        max_prob = class_scores[rating]
        pred = rating
    
    return pred

  def accuracy(data, theta, phi):
    
    correct_preds = 0
    predictions = []
    actual_ratings = []
    
    for x in data:
      actual_rating = x[1]
      pred_rating = predict(theta, x[0], phi)
      
      actual_ratings.append(actual_rating)
      predictions.append(pred_rating)
      
      if actual_rating - pred_rating == 0:
        correct_preds += 1
        
    acc = correct_preds*100/len(data)
    return acc, predictions, actual_ratings


  print("Removing words with very low occurrences")

  # training theta parameters
  theta = naive_bayes(dictionary, n, 1)

  # calculating the training accuracy
  train_acc, train_preds, train_data_actual = accuracy(train_data, theta, phi)
  print(f"The training accuracy is : {train_acc}")

  # calculating the accuracy on the test_set
  test_acc, test_data_preds, test_data_actual = accuracy(test_data, theta, phi)
  print(f"The accuracy on the test set is : {test_acc}")
  sys.exit()


if part_num == "g":

  """
  Method 1 : Words in the reviewText that are also present in the summary are counted 10 times
  """

  test_data = []
  for x in X_test:
    x = json.loads(x)
    rating = x["overall"]
    text = x["reviewText"]
    summary = x["summary"]
    test_data.append((text,rating, summary))
    
  train_data = []
  for x in X_train:
    x = json.loads(x)
    rating = x["overall"]
    text = x["reviewText"]
    summary = x["summary"]
    train_data.append((text,rating, summary))

  # dictionary[0] is the global dictionary
  dictionary = {0:{}, 1:{}, 2:{}, 3:{}, 4:{}, 5:{}}

  # n[0] is the size of the global dictionary
  n = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}

  class_labels = {1:0, 2:0, 3:0, 4:0, 5:0}

  val = 0

  for x in train_data:
    text = x[0]
    rating = x[1]
    summary = x[2]
    class_labels[rating] += 1

    words = text.split()
    
    summary_words = summary.split()
    
    for word in words:
      if word in summary_words:
        val = 10
      else:
        val = 1
        
      n[0] += val
      n[rating] += val
        
      if word in dictionary[0]:
        dictionary[0][word] += val
      else:
        dictionary[0][word] = val
        
      if word in dictionary[rating]:
        dictionary[rating][word] += val
      else:
        dictionary[rating][word] = val


  phi = {}
  for rating in range(1,6):
    phi[rating] = class_labels[rating]/len(X_train)


  def naive_bayes(dictionary, n, alpha):
    vocab_size = len(dictionary[0])
    theta = {}
    for rating in range(1, 6):
      theta[rating] = {}
    
    for word in dictionary[0].keys():
      for rating in range(1, 6):
        if word in dictionary[rating]:
          theta[rating][word] = ((dictionary[rating][word] + alpha)/(n[rating] + alpha * vocab_size))
        else:
          theta[rating][word] = (alpha/(n[rating] + alpha*vocab_size))
    
    return theta

  theta = naive_bayes(dictionary, n, 1)


  def predict(theta, x_text, phi):
    class_scores = {1:0, 2:0, 3:0, 4:0, 5:0}
    words = x_text.split()
    
    for word in words:
      for rating in range(1, 6):
        if word in theta[rating]:
          class_scores[rating] += math.log(theta[rating][word])
    
    for rating in range(1, 6):
      class_scores[rating] += math.log(phi[rating])
      
    max_prob = class_scores[1]
    pred = 1
    
    for rating in range(2, 6):
      if class_scores[rating] > max_prob:
        max_prob = class_scores[rating]
        pred = rating
    
    return pred


  def accuracy(data, theta, phi):
    
    correct_preds = 0
    predictions = []
    actual_ratings = []
    
    for x in data:
      actual_rating = x[1]
      pred_rating = predict(theta, x[0], phi)
      
      actual_ratings.append(actual_rating)
      predictions.append(pred_rating)
      
      if actual_rating - pred_rating == 0:
        correct_preds += 1
        
    acc = correct_preds*100/len(data)
    return acc, predictions, actual_ratings


  print("Method 1 : Words in the reviewText that are also present in the summary are counted 10 times")

  # training theta parameters
  theta = naive_bayes(dictionary, n, 1)

  # calculating the training accuracy
  train_acc, train_preds, train_data_actual = accuracy(train_data, theta, phi)
  print(f"The training accuracy is : {train_acc}")

  # calculating the accuracy on the test_set
  test_acc, test_data_preds, test_data_actual = accuracy(test_data, theta, phi)
  print(f"The accuracy on the test set is : {test_acc}")

  """
  Method 2 : Adding the summary 10 times at the end of the reviewText
  """

  test_data = []
  for x in X_test:
    x = json.loads(x)
    rating = x["overall"]
    text = x["reviewText"]
    summary = x["summary"]*10
    test_data.append((text + summary, rating))
    
  train_data = []
  for x in X_train:
    x = json.loads(x)
    rating = x["overall"]
    text = x["reviewText"]
    summary = x["summary"]*10
    train_data.append((text + summary, rating))

  # dictionary[0] is the global dictionary
  dictionary = {0:{}, 1:{}, 2:{}, 3:{}, 4:{}, 5:{}}

  # n[0] is the size of the global dictionary
  n = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}

  class_labels = {1:0, 2:0, 3:0, 4:0, 5:0}

  val = 1

  for x in train_data:
    text = x[0]
    rating = x[1]
    class_labels[rating] += 1

    words = text.split())
    
    summary_words = summary.split()
    
    for word in words:
        
      n[0] += val
      n[rating] += val
        
      if word in dictionary[0]:
        dictionary[0][word] += val
      else:
        dictionary[0][word] = val
        
      if word in dictionary[rating]:
        dictionary[rating][word] += val
      else:
        dictionary[rating][word] = val

  phi = {}
  for rating in range(1,6):
    phi[rating] = class_labels[rating]/len(X_train)

  def naive_bayes(dictionary, n, alpha):
    vocab_size = len(dictionary[0])
    theta = {}
    for rating in range(1, 6):
      theta[rating] = {}
    
    for word in dictionary[0].keys():
      for rating in range(1, 6):
        if word in dictionary[rating]:
          theta[rating][word] = ((dictionary[rating][word] + alpha)/(n[rating] + alpha * vocab_size))
        else:
          theta[rating][word] = (alpha/(n[rating] + alpha*vocab_size))
    
    return theta

  theta = naive_bayes(dictionary, n, 1)

  def predict(theta, x_text, phi):
    class_scores = {1:0, 2:0, 3:0, 4:0, 5:0}
    words = x_text.split()
    
    for word in words:
      for rating in range(1, 6):
        if word in theta[rating]:
          class_scores[rating] += math.log(theta[rating][word])
    
    for rating in range(1, 6):
      class_scores[rating] += math.log(phi[rating])
      
    max_prob = class_scores[1]
    pred = 1
    
    for rating in range(2, 6):
      if class_scores[rating] > max_prob:
        max_prob = class_scores[rating]
        pred = rating
    
    return pred

  def accuracy(data, theta, phi):
    
    correct_preds = 0
    predictions = []
    actual_ratings = []
    
    for x in data:
      actual_rating = x[1]
      pred_rating = predict(theta, x[0], phi)
      
      actual_ratings.append(actual_rating)
      predictions.append(pred_rating)
      
      if actual_rating - pred_rating == 0:
        correct_preds += 1
        
    acc = correct_preds*100/len(data)
    return acc, predictions, actual_ratings


  print("Method 2 : Adding the summary 10 times at the end of the reviewText")

  # training theta parameters
  theta = naive_bayes(dictionary, n, 1)

  # calculating the training accuracy
  train_acc, train_preds, train_data_actual = accuracy(train_data, theta, phi)
  print(f"The training accuracy is : {train_acc}")

  # calculating the accuracy on the test_set
  test_acc, test_data_preds, test_data_actual = accuracy(test_data, theta, phi)
  print(f"The accuracy on the test set is : {test_acc}")
  sys.exit()