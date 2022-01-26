import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class perceptron():
	"""docstring for perceptron"""
	def __init__(self, lr, epoch, layer_dim, hidden_layers_activation):
		self.lr = lr
		self.epoch = epoch
		self.deep_size = len(layer_dim) 
		self.layer_dim = layer_dim
		self.hidden_layers_activation = hidden_layers_activation

	def init_param(self, X):
		n = np.shape(X)[0]
		self.m = np.shape(X)[1]
		self.param = {}
		self.A = []
		self.A.append(X)
		for i in range(self.deep_size):
			self.param[f'W{i}'] = np.random.rand(self.layer_dim[i], n) * 0.01
			self.param[f'B{i}'] = np.zeros((self.layer_dim[i], 1))
			self.A.append(np.zeros((self.layer_dim[i], self.m)))
			n = self.layer_dim[i]
		self.activation_func = {'relu':self.relu, 'tanh':self.tanh, 'sigmoide':self.sigmoide}
		self.d_activation_func = {'d_relu':self.d_relu, 'd_tanh':self.d_tanh, 'd_sigmoide':self.d_sigmoide}


	def d_sigmoide(self, A):
		return A - A**2 

	def d_relu(self, A):
		return 1 - A**2

	def d_tanh(self, A):
		return np.int64(A > 0)

	def relu(self, z):
		# print(np.shape(z))
		return np.maximum(0,z)

	def sigmoide(self, z):
		# print(np.shape(z))
		return 1 / (1 + np.exp(-z)) - 0.000000000000001

	def tanh(self, z):
		return np.tanh(z)

	def model(self, i):
		# print(i)
		# print(np.shape(self.param[f'W{i}']))
		# print(np.shape(self.A[i]))
		return np.dot(self.param[f'W{i}'], self.A[i]) + self.param[f'B{i}']
	
	def cost_function(self, X, Y):
		# print(self.A[self.deep_size])

		cost = (-1/self.m) * np.sum(Y * np.log(self.A[self.deep_size]) + (1 - Y) * np.log(1 - self.A[self.deep_size]))
		if cost == np.nan:
			exit()
		print(cost)
		return cost
		

	def update_param(self, dZ, i):
		dW = 1/self.m * np.dot(dZ, self.A[i-1].T)
		db = 1/self.m * np.sum(dZ, axis=1, keepdims=True)


		# print(i)
		# print(f'shape W avant update :{np.shape(self.param[f"W{i-1}"])}')
		# print(f'shape B avant update :{np.shape(self.param[f"W{i-1}"])}')
		# print(f'W avant update \n:{self.param[f"W{i-1}"][0]}')

		self.param[f'W{i-1}'] = self.param[f'W{i-1}'] - self.lr * dW
		self.param[f'B{i-1}'] = self.param[f'B{i-1}']  - self.lr * db

		# print(f'W apres update :\n{self.param[f"W{i-1}"][0]}')

		# print(f'shape W apres update :{np.shape(self.param[f"W{i-1}"])}')
		# print(f'shape B apres update :{np.shape(self.param[f"W{i-1}"])}')



	def backward_prop(self, Y):
		# print(f'{self.A[self.deep_size]}\n')
		dA_output = (1 - Y) / (1 - self.A[self.deep_size]) - Y / self.A[self.deep_size]
		dZ_output =  dA_output * self.d_activation_func[f'd_{self.hidden_layers_activation}'](self.A[self.deep_size])
		W_p = self.param[f'W{self.deep_size -1}']
		self.update_param(dZ_output, self.deep_size)
		dZ_p = dZ_output
		for i in range(self.deep_size-1, 0, -1):
			dZ = np.dot(W_p.T,dZ_p) * np.int64(self.A[i] > 0)
			W_p = self.param[f'W{i-1}']
			self.update_param(dZ,i)
			dZ_p = dZ

	def forward_prop(self, strr):
		for i in range(self.deep_size):
			Z = self.model(i)
			if i == self.deep_size-1:
				self.A[i+1] = self.activation_func['sigmoide'](Z)
			else:
				self.A[i+1] = self.activation_func[self.hidden_layers_activation](Z)
			if strr == 'print_shape':
				print(np.shape(self.A[i+1]))
			else:
				continue


	def fit(self, X, Y):

		self.init_param(X)
		cost_history = []
		for i in range(self.epoch):
			# print('------------coucou')
			self.forward_prop('jj')
			cost_history.append(self.cost_function(X, Y))
			self.backward_prop(Y)

		# plt.figure(figsize=(10, 6))
		# plt.plot(cost_history)
		# plt.xlabel("Iterations (per hundreds)")
		# plt.ylabel("Loss")
		# plt.title(f"Loss curve for the learning rate = {self.lr}")
		# plt.show()

	def predict(self, X):
		self.A[0] = X
		self.forward_prop('')
		self.A[self.deep_size] = np.int64(self.A[self.deep_size] > 0.5)
		return self.A[self.deep_size].T

		# print(f'last W :{self.param[f"W{self.deep_size-1}"]}')
		# print(f'Y pred :{self.A[self.deep_size]}\n')



df = pd.read_csv('data.csv')
df = df.fillna(df.mean())
df_train, df_test = np.split(df.sample(frac=1), [int(0.8*len(df))])
Y_train = df_train['M'].replace({'M': 1, 'B': 0})
Y_test = df_test['M'].replace({'M': 1, 'B': 0})
df_train.drop(columns=['M','842302'], inplace =True)
df_test.drop(columns=['M','842302'], inplace =True)
for name in df_train.columns:
		df_train[name] = (df_train[name] - df_train[name].mean()) / df_train[name].std()
		df_test[name] = (df_test[name] - df_test[name].mean()) / df_test[name].std()

mp = perceptron(0.01, 10000, [30,30,1], 'relu')
mp.fit(df_train.to_numpy().T, Y_train.to_numpy().T)
Y_pred = mp.predict(df_test.to_numpy().T)
print(f'Y_test : {Y_test.to_numpy().T}\n')
print(f'Y_pred : {Y_pred.T}\n')
print('precision:')
print((Y_pred.T[0] == Y_test.to_numpy()).mean())






