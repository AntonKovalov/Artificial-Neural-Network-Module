# Artificial-Neural-Network-Module
This class allows to create ANN with a single hidden layer

class ANN(learning_rate = ) - is a ready artificial that allows for simple predictions. Due to lack of opimization algorithms in the code it does not perform effectively on large data sets. 

**MNIST handwritten digit recognition:**  25 epochs were needed to achieve 94% accuracy on the test set, however the learning time is around 7 minutes, which is not satisfactory for sch a simple problem.

____________________________________________________________________________________________________________________

1) create an instance of *ANN()* with the *learning_rate* parameter included

2) start the learning process with calling *train(X,Y,batches,epochs)* method<br />
  X - training matrix of features
  Y - training vector of values
  batches - under development now (recommend to use batches = 1)
  epochs - the amount of itterations over the training set to complete learning

3) prediction step requires calling the *predict(x_test,y_test)* method
