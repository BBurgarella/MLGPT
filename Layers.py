import numpy as np

"""
let's first create a class that matches what the dense layer in 
Keras do
"""
class dense:

    def __init__(self, input_size, output_size, activation=None, seed = None, **kwargs):
        
        # first, initialize the seed
        if seed is not None:
            np.random.seed(seed)

        # initialize the weights using a random distribution
        # Glorot, X., & Bengio, Y. (2010, March). Understanding the difficulty of training deep feedforward neural networks. 
        # In Proceedings of the thirteenth international conference on artificial intelligence and statistics (pp. 249-256). 
        # JMLR Workshop and Conference Proceedings.

        self.weights = np.random.randn(input_size, output_size)
        # Glorot initialization:
        self.weights = self.weights * np.sqrt(2. / (input_size + output_size))
        # Biases initialized to zero
        self.biaises = np.zeros(output_size)
        self.activation = activation

    def forward(self, input):

        Layer_Output =  np.dot(input,self.weights) + self.biaises

        if self.activation is not None:
            Layer_Output = self.activation(Layer_Output)

        return