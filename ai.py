# http://neuralnetworksanddeeplearning.com/
import numpy as np

class Brain:
	def __init__(self, *layers_count):
		self.ls = []
		prev_count = 0
		for count in layers_count:
			self.ls.append(Layer(prev_count, count).randomize())
			prev_count = count
	
	def forward(self, X):
		# feedforward
		#activation = x
		#activations = [x] # list to store all the activations, layer by layer
		#zs = [] # list to store all the z vectors, layer by layer
		#for b, w in zip(self.biases, self.weights):
		#    z = np.dot(w, activation)+b
		#    zs.append(z)
		#   activation = sigmoid(z)
		#    activations.append(activation)
		ls = self.ls
		ls[0].activation = X#Sigmoid(X)
		for i in range(1, len(ls)):
			l_prev = ls[i - 1]
			l = ls[i]
			
			l.raw_output = np.dot(l_prev.activation, l.weights) + l.b
			#l.raw_output = np.dot(l.activation, l.weights.T) + l.b
			l.activation = ActivationFunction(l.raw_output)

		return ls[-1].a

	def train(self, X, targets, eta):
		self.forward(X)

		ls = self.ls
		ls[-1].delta = cost_derivativ(ls[-1].a, targets) * dActivationFunction(ls[-1].z)

		for i in range(2, len(self.ls)):
			l = ls[-i]
			next_l = ls[-i+1]
			l.delta = np.dot(next_l.delta, next_l.w.T) * dActivationFunction(l.z)

		for i in range(1, len(self.ls)):
			l = ls[-i]
			l_prior = ls[-i-1]
			#avgdelta = np.average(l.delta, axis=0)
			avgdelta = np.sum(l.delta, axis=0)
			l.bias = l.bias - eta  * avgdelta
			#avgdelta = np.average(np.dot(l_prior.a.T, l.delta), axis=0)
			#avgdelta = np.average(np.dot(l.a.T, l.delta), axis=0)
			avgdelta = np.sum(np.dot(l.a.T, l.delta), axis=0)
			l.weights= l.weights - eta  * avgdelta

			if i == 1:
				print(np.average(l.delta, axis=0))


class Layer():
	"""a hidden layer"""
	def __init__(self, input_count, self_count):
		self.weights = np.zeros((input_count, self_count))
		self.bias = np.zeros((1, self_count))
		self.raw_output = np.zeros((1, self_count))
		self.activation = np.zeros((1, self_count))
		self.delta = None
	
	def randomize(self):
		self.weights = (np.random.random_sample(self.weights.shape) - 0.5) #+ 0.5)# * 2.0
		#self.weights = np.ones(self.weights.shape)
		self.bias = np.random.random_sample(self.bias.shape) - 0.5 #np.zeros(self.bias.shape) + 1.0# + (np.random.random_sample(self.bias.shape)) * 0.3
		#self.bias = np.ones(self.bias.shape)
		return self

	def load(self, weights, bias):
		self.weights = weights
		self.bias = bias
		return self

	@property
	def w(self):
		return self.weights

	@property
	def b(self):
		return self.bias

	@property
	def z(self):
		return self.raw_output

	@property
	def a(self):
		return self.activation

def ActivationFunction(x):
	return ReLU(x)
	return LeakyReLU(x)
	return SoftPluss(x)
	return Sigmoid(x)

def dActivationFunction(x):
	return dReLU(x)
	return dLeakyReLU(x)
	return dSoftPluss(x)
	return dSigmoid(x)

def ReLU(x, out=None):
	return np.maximum(x, 0, out=out)
def dReLU(x):
	return (1 * (x >= 0))

def LeakyReLU(x):
	return x * (x >= 0) + x * (0.01 * (x < 0))
def dLeakyReLU(x):
	return (1 * (x >= 0) + 0.01 * (x < 0))

def SoftPluss(x, k=2):
	return np.log(1 + np.exp(k * x))
def dSoftPluss(x, k=2):
	return 1 / (1 + np.exp(-(k * x)))

def Sigmoid(x):
	return 1 / (1 + np.exp(-x))
def dSigmoid(x):
	s = Sigmoid(x)
	return s * (1 - s)
def revSig(y):
	return np.log(y / (1 - y))

def cost_derivativ(output_activations, y):
	return (output_activations - y)

def linearFunc(x):
	a = 2.4
	b = 7.0
	return x * a + b

brain = Brain(1, 1)
for i in range(10000):
	#X = np.array([[1,0.2],[1,0.2],[1,0.2],[1,0.2],[1,0.2],[1,0.2],[1,0.2]])
	#Y = np.array([[0.5, 0.3],[0.5, 0.3],[0.5, 0.3],[0.5, 0.3],[0.5, 0.3],[0.5, 0.3],[0.5, 0.3]])# * 8
	X = (np.random.random_sample((16, 1)) * 20)
	#X = np.array([np.linspace(0, np.pi, 32)]).T / 2.0 / np.pi
	#Y = np.sin(X * 2.0 * np.pi)
	Y = ActivationFunction(linearFunc(X))
	#print(Y)
	print(f"Training nr. {i}")
	#brain.train(X, Y, 0.0005)
	brain.train(X, Y, 0.000005)
	#brain.train(X, Y, 0.23)

x = np.array([np.linspace(0, 10, 10)]).T
print(x)
print(linearFunc(x))
#x = np.array([[1,0.2],[1,0.2],[1,0.2],[1,0.2],[1,0.2],[1,0.2],[1,0.2]])

print(brain.forward(x))
#print(revSig(brain.forward(x)))


for l in brain.ls:
	print()
	print("Bias")
	print(l.bias)
	print("Weights")
	print(l.weights)


















input_count = 1 #3 * 4 * 2
layer1_count = 10
layer2_count = 10
output_count = 1

#X = np.array([[1,0.2],[1,0.2],[1,0.2],[1,0.2],[1,0.2],[1,0.2],[1,0.2]])
#Y = np.array([[0.5],[0.5],[0.5],[0.5],[0.5],[0.5],[0.5]])
#Y = np.array([[0.5, 0.3],[0.5, 0.3],[0.5, 0.3],[0.5, 0.3],[0.5, 0.3],[0.5, 0.3],[0.5, 0.3]])
#Y = np.array([[0.5, 0.3],[0.5, 0.3],[0.5, 0.3],[0.5, 0.3],[0.5, 0.3],[0.5, 0.3],[0.5, 0.3]])# * 8

l = [
	Layer(0, input_count).randomize(),
	Layer(input_count, layer1_count).randomize(),
	Layer(layer1_count, layer2_count).randomize(),
	Layer(layer2_count, output_count).randomize()
]

#l[0].forward(X)
#print(y1)
#l[1].forward(l[0].a)
#print(y2)
#l[2].forward(l[1].a)
#print(l[2].a)


#print(np.sum(l[2].errorforinputlayer(Y), axis=0))
#print(np.sum(l[1].outputerror(l[2]), axis=0))
#print(np.sum(l[0].outputerror(l[1]), axis=0))

#l[2].learn(0.001, l[1].a)
#l[1].learn(0.001, l[0].a)
#l[0].learn(0.001, X)


#l[0].forward(X)
#print(y1)
#l[1].forward(l[0].a)
#print(y2)
#l[2].forward(l[1].a)
#print(l[2].a)
def sigmoid(z):
    """The sigmoid function."""
    return z
    return 1.0/(1.0+np.exp(-z))

def train(inputs, targets):
	forward(inputs)

	siginputs = sigmoid(inputs)

	print(l[2].errorforoutputlayer(targets))
	l[1].outputerror(l[2])
	l[0].outputerror(l[1])

	l[2].learn(0.00005, l[1].a)
	l[1].learn(0.00005, l[0].a)
	l[0].learn(0.00005, siginputs)

def forward(inputs):
	siginputs = sigmoid(inputs)
	l[0].forward(siginputs)
	l[1].forward(l[0].a)
	l[2].forward(l[1].a)
	return l[2].a

#print(f"Before training:")
#print(forward(np.array([[np.pi]])))

for i in range(0):
	X = (np.random.random_sample((8, 1))) * np.pi
	#X = np.array([np.linspace(-np.pi, np.pi, 16)]).T
	#print(X)
	Y = np.sin(X)
	train(X,Y)
	print(f"After training: {i}")
	print(forward(np.array([[np.pi]])))

#x = np.array([np.linspace(0, np.pi, 50)]).T
#print("Result:")
#print(l[0].forward(np.array([[1,0.2]])).a)
#print(l[0].forward(sigmoid(np.array([[np.pi]]))).a)
#print(l[1].forward(l[0].a).a)
#print(l[2].forward(l[1].a).a)
#print()
"""print("Biases:")
print(l[0].b)
print()
print(l[1].b)
print()
print(l[2].b)
print()
print("Weights:")
print(l[0].w)
print()
print(l[1].w)
print()
print(l[2].w)"""
#print(np.sin(np.array([[3.14]])))