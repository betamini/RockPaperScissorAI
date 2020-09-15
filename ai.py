import numpy as np

class brain:
	def __init__(self, input_count, layer1_count, layer2_count, output_count):
		inputNodes = [Node("Prev"), Node("PrevPrev")]
		outputNodes = [Node("Rock"), Node("Paper"), Node("Scissor")]
		nodes = [Node("Rock"), Node("Paper"), Node("Scissor")]

		#w1 = np.array([np.rand])
		

class Layer():
	"""a hidden layer"""
	def __init__(self, input_count, self_count):
		self.weights = np.zeros((input_count, self_count))
		self.bias = np.zeros((1, self_count))
	
	def randomize(self):
		self.weights = np.random.rand(*self.weights.shape)
		#self.weights = np.ones(self.weights.shape)
		self.bias = np.random.rand(*self.bias.shape)
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
	
	def forward(self, input_):
		temp = np.dot(input_, self.weights)
		return temp + self.bias

input_count = 2 #3 * 4 * 2
layer1_count = 3
layer2_count = 3
output_count = 2

X = np.array([[1,0.2],[1,0.2],[1,0.2],[1,0.2],[1,0.2],[1,0.2],[1,0.2]])
#X = np.array([1,0.2])

l1 = Layer(input_count, layer1_count).randomize()
l2 = Layer(layer1_count, layer2_count).randomize()
l3 = Layer(layer2_count, output_count).randomize()

y1 = l1.forward(X)
#print(y1)
y2 = l2.forward(y1)
#print(y2)
y3 = l3.forward(y2)
print(y3)