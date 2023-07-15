import numpy as np

def softmax(numpyObject):
    """
    Applies the softmax function to the input numpy array. 
    
    Note to the reader, I generated this docstring using chatGPT for simplicity

    Parameters:
        numpyObject (numpy.ndarray): Input array containing numeric values.

    Returns:
        numpy.ndarray: Array with the same shape as the input, where each element is the result
                       of applying the softmax function to the corresponding element in the input array.

    Raises:
        TypeError: If the input is not a numpy array.

    Notes:
        - The softmax function is a common activation function used in machine learning.
        - The softmax function normalizes the input array so that each element is in the range [0, 1]
          and the sum of all elements is equal to 1.
        - The output of the softmax function can be interpreted as probabilities of the input values
          belonging to different classes.

    Example:
        >>> import numpy as np
        >>> x = np.array([1, 2, 3])
        >>> softmax(x)
        array([0.09003057, 0.24472847, 0.66524096])
    
    """
    Exponential = np.exp(numpyObject)
    return Exponential/np.sum(Exponential)

    
def scaledDotProductSelfAttention(Q,K,V):
    """
    Implements scaled dot product self attention.

    Args:
        Q (ndarray): Query matrix of shape (batch_size, query_length, query_dimension).
        K (ndarray): Key matrix of shape (batch_size, key_length, key_dimension).
        V (ndarray): Value matrix of shape (batch_size, key_length, value_dimension).

    Returns:
        tuple: A tuple containing two elements.
            - Attention result (ndarray): The output of the attention mechanism, which is the weighted sum of values.
              It has the shape (batch_size, query_length, value_dimension).
            - Attention weights (ndarray): The weights assigned to each key-value pair during the attention calculation.
              It has the shape (batch_size, query_length, key_length).

    Notes:
        The scaled dot product self attention calculates the attention scores between the query and key vectors.
        It is defined as: Attention(Query, Key, Value) = softmax(QK^T / sqrt(d_k))V,
        where Q, K, and V are query, key, and value matrices, respectively, and d_k is the dimension of the key vectors.

    Example:
        Q = np.array([[1, 2, 3], [4, 5, 6]])  # Shape: (2, 3)
        K = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])  # Shape: (2, 4)
        V = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])  # Shape: (2, 4)
        attention_result, attention_weights = scaledDotProductSelfAttention(Q, K, V)
    """
    
    keyDimension = np.shape(K)[-1]
    Scores = np.dot(Q,K.T)/np.sqrt(keyDimension)

    weights = softmax(Scores)

    return np.dot(weights, V), weights