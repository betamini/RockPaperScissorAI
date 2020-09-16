# http://neuralnetworksanddeeplearning.com/
import numpy as np

class Brain:
	def __init__(self, *layers_count):
		self.ls = []
		for count, next_count in zip(layers_count[:-1], layers_count[1:]):
			self.ls.append(Layer(count, next_count).randomize())
	
	def thing(self, X, targets):
		ls = self.ls
		ls[-1].delta = cost_derivativ(ls[-1].a, targets) * ReLUder(ls[-1].z)

		for i in range(2, len(self.ls)):
			ls[-i].delta = np.dot(ls[-i+1].delta, ls[-i+1].w.T) * ReLUder(ls[-i].z)


class Layer():
	"""a hidden layer"""
	def __init__(self, input_count, self_count):
		self.weights = np.zeros((input_count, self_count))
		self.bias = np.zeros((1, self_count))
		self.raw_output = np.zeros((1, self_count))
		self.activation = np.zeros((1, self_count))
		self.delta = None
	
	def randomize(self):
		self.weights = (np.random.random_sample(self.weights.shape) - 0.5 + 0.5)# * 2.0
		#self.weights = np.ones(self.weights.shape)
		self.bias = np.random.random_sample(self.bias.shape) #np.zeros(self.bias.shape) + 1.0# + (np.random.random_sample(self.bias.shape)) * 0.3
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
	
	def forward(self, input_):
		self.raw_output = np.zeros((input_.shape[0], self.weights.shape[1]))
		self.activation = np.zeros((input_.shape[0], self.weights.shape[1]))
		np.dot(input_, self.weights, out=self.raw_output)
		ReLU(self.raw_output, out=self.activation)
		return self

	def errorforoutputlayer(self, targets):
		self.cost = cost_derivativ(self.a, targets)
		self.output_error = np.multiply(self.cost, ReLUder(self.z))
		return self.output_error

	#def errorforinput(self, raw_input):
	#	self.input_error = np.multiply(np.dot(self.w.T, self.output_error), (1 * (raw_input > 0)))
	#	return self.input_error

	def outputerror(self, layer_after):
		#temp = np.dot(layer_after.delta, layer_after.w.T)
		temp = np.dot(layer_after.w.T, layer_after.delta)
		#print((1 * (self.z > 0)))
		self.delta = np.multiply(temp, ReLUder(self.z))
		return self.output_error

	def learn(self, eta, layer_before_activation):
		error = np.average(self.output_error, axis=0)
		self.bias = self.bias - (eta * error)
		errorweight = np.average(np.dot(layer_before_activation.T, self.output_error), axis=0)
		self.weights = self.weights - (eta * errorweight)
		#print()
		#print()
		#print()
		#print(error)
		#print()
		#print(errorweight)
		return self

def ReLU(x, out=None):
	return np.maximum(x, 0, out=out)

def ReLUder(x):
	#k = 10
	#return 1 / (1 + np.exp(-(k * x)))
	return (1 * (x > 0))
	#return (1 * (x > 0) + 0.01 * (x < 0))

def cost_derivativ(output_activations, y):
	return (output_activations - y)

X = np.array([[1,0.2],[1,0.2],[1,0.2],[1,0.2],[1,0.2],[1,0.2],[1,0.2]])
Y = np.array([[0.5, 0.3],[0.5, 0.3],[0.5, 0.3],[0.5, 0.3],[0.5, 0.3],[0.5, 0.3],[0.5, 0.3]])# * 8


brain = Brain(2, 10, 10, 2)
brain.thing(X, Y)


input_count = 1 #3 * 4 * 2
layer1_count = 10
layer2_count = 10
output_count = 1

#X = np.array([[1,0.2],[1,0.2],[1,0.2],[1,0.2],[1,0.2],[1,0.2],[1,0.2]])
#Y = np.array([[0.5],[0.5],[0.5],[0.5],[0.5],[0.5],[0.5]])
#Y = np.array([[0.5, 0.3],[0.5, 0.3],[0.5, 0.3],[0.5, 0.3],[0.5, 0.3],[0.5, 0.3],[0.5, 0.3]])
#Y = np.array([[0.5, 0.3],[0.5, 0.3],[0.5, 0.3],[0.5, 0.3],[0.5, 0.3],[0.5, 0.3],[0.5, 0.3]])# * 8

l = [
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
    return 1.0/(1.0+np.exp(-z))
    return z

def train(inputs, targets):
	forward(inputs)

	siginputs = sigmoid(inputs)

	print(l[2].errorforoutputlayer(targets))
	l[1].outputerror(l[2])
	l[0].outputerror(l[1])

	l[2].learn(0.002, l[1].a)
	l[1].learn(0.002, l[0].a)
	l[0].learn(0.002, siginputs)

def forward(inputs):
	siginputs = sigmoid(inputs)
	l[0].forward(siginputs)
	l[1].forward(l[0].a)
	l[2].forward(l[1].a)
	return l[2].a

print(f"Before training:")
print(forward(np.array([[np.pi]])))

for i in range(200):
	X = (np.random.random_sample((8, 1))) * np.pi
	#X = np.array([np.linspace(-np.pi, np.pi, 16)]).T
	#print(X)
	Y = np.sin(X)
	train(X,Y)
	print(f"After training: {i}")
	print(forward(np.array([[np.pi]])))

#x = np.array([np.linspace(0, np.pi, 50)]).T
print("Result:")
#print(l[0].forward(np.array([[1,0.2]])).a)
print(l[0].forward(sigmoid(np.array([[np.pi]]))).a)
print(l[1].forward(l[0].a).a)
print(l[2].forward(l[1].a).a)
print()
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