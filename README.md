# Artificial-Neural-Network-Module
This class allows to create ANN with a single hidden layer

class ANN(learning_rate = ) - is a ready artificial neaural network that allows for simple predictions. Due to lack of opimization algorithms in the code it does not perform effectively on large data sets. 


--> Checked on:<br />
**MNIST handwritten digit recognition**<br />
**Iris classification**

____________________________________________________________________________________________________________________

1) create an instance of *ANN(learning_rate=)* with the *learning_rate* parameter

2) compile your ANN calling *compile_(dim=[...])* and providing dimension array as an argument

3) start the learning process with calling *train(X,Y,batches,epochs)* method<br />
  X - training matrix of features<br />
  Y - training vector of values<br />
  batches - creates a subset of training set to learn faster<br />
  epochs - the amount of itterations over the training set to complete learning

4) prediction step requires calling the *predict(x_test,y_test)* method
