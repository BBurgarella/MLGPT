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

        # Storage of variables between forward and backward:
        self.PreviousLayerUnactivated = None
        self.PreviousInput = None

    def forward(self, input):

        """
        input => Batch_size x N_in
        weights => N_in x N_Out
        Output = Batch_size x N_Out
        Layer => dot(input(Batch_size x N_in), weights(N_in x N_Out)) => Batch_size x N_Out
        
        """

        # fun fact I leared today, this formulation:
        # Layer_Output =  np.dot(input,self.weights) + self.biaises
        # can also be written as:
        Layer_Output =  (input @ self.weights) + self.biaises

        # Here we need to store the non activated output for traning
        self.PreviousInput = input
        self.PreviousLayerUnactivated = Layer_Output

        if self.activation is not None:
            Layer_Output = self.activation(Layer_Output)

        return Layer_Output
    
    def backward(self, GradNext, learningRate):
        
        """Evaluates the backward propagation for this layer

            I need to understand this better I think

        """

        pass
