import numpy as np
import pandas as pd
import time

#Artificial Neural Network Class
class ANN():
	def __init__(self,learning_rate):
		#self.dim stores ANN layer node dimension
		print("Define the dimesions of your ANN:")
		self.dim = []

		self.lr = learning_rate

		self.l = []
		self.W = []
		self.b = []


	def compile_dynamically(self):
		self.dim.append(int(raw_input("Input dimension = ")))
		self.dim.append(int(raw_input("Hidden Layer dimension = ")))
		self.dim.append(int(raw_input("Output dimension = ")))

		for i in range(0,len(self.dim)):
			self.l.append(np.zeros(self.dim[i]))
			if i >= 1:
				self.W.append(np.random.randn(self.dim[i-1],self.dim[i]))
				self.b.append(np.ones(self.dim[i]))

	def compile_(self,dim):
		for i in dim:
			self.dim.append(i)

		for i in range(0,len(self.dim)):
			self.l.append(np.zeros(self.dim[i]))
			if i >= 1:
				self.W.append(np.random.randn(self.dim[i-1],self.dim[i]))
				self.b.append(np.ones(self.dim[i]))

	def sigmoid(self,Z):

		return 1./(1+np.exp(-Z))


	def forward_prop(self,l,W,b):
		l[1] = self.sigmoid(np.dot(l[0],W[0]) + b[0])
		l[2] = self.sigmoid(np.dot(l[1],W[1]) + b[1])

		return l

	def cost_f(self,Y,A):
		cost = 0
		for i in range(len(A)):
			cost = cost + 0.5*(Y[i] - A[i])**2

		return cost


	def backward_prop(self,Y,l,W,b,lr):
		db = [np.zeros(6),np.zeros(3)]
		db[1] = -(Y-l[2])*l[2]*(1-l[2])
		db[0] = l[1]*(1-l[1])*np.dot(db[1],W[1].T)

		dW = [np.zeros((4,6)),np.zeros((6,3))]
		dW[1] = np.dot(l[1].T, -(Y-l[2])*l[2]*(1-l[2]))
		dW[0] = np.dot(l[0].T, l[1]*(1-l[1])*np.dot(db[1],W[1].T))

		b[0] = b[0]-lr*db[0]
		b[1] = b[1]-lr*db[1]

		W[0] = W[0]-lr*dW[0]
		W[1] = W[1]-lr*dW[1]

		return W, b


	def train(self,X,Y,batches,epochs):
		print("------------Learning has started--------------")
		for i in range(epochs):
			start_time = time.time()
			indices = np.random.randint(low=0,high=len(X),size=batches)
			for j in indices:
				self.l[0] = np.array([X[j]])
				self.l = self.forward_prop(self.l,self.W,self.b)

				cost = self.cost_f(Y[j], self.l[2][0])
				self.W, self.b = self.backward_prop(Y[j],self.l,self.W,self.b,self.lr)


			print("Epoch = %d finished in %s seconds"%(i+1,time.time() - start_time))

	def predict(self,X,Y):
		pred = np.zeros((len(Y),len(Y[0])))

		for i in range(0, len(X)):
			pred[i] = self.sigmoid(np.dot(self.sigmoid(np.dot(X[i],self.W[0]) + self.b[0]),self.W[1]) + self.b[1])

			ind = np.where(pred[i] == np.amax(pred[i]))
			ind = ind[0][0]
			pred[i] = np.zeros(len(Y[0]))
			pred[i][ind] = 1

		return pred
