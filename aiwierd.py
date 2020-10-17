class brain:
	def __init__(self):
		inputNodes = [Node("Prev"), Node("PrevPrev")]
		outputNodes = [Node("Rock"), Node("Paper"), Node("Scissor")]
		nodes = [Node("Rock"), Node("Paper"), Node("Scissor")]


class Edge(object):
	"""A conection between nodes"""
	def __init__(self, nodeA, nodeB):
		self.nodeA = nodeA
		nodeA.edges.append(self)
		self.nodeB = nodeB
		nodeB.edges.append(self)

class Node(object):
	"""A node"""
	def __init__(self, name):
		self.name = name
		self.strength = 0.5 # 0 - 1
		self.outputs = []
		self.weights = [] # 0 - 1

	def addnode(self, node):
		self.outputs.append(node)
		self.weights.append(node.strength)


