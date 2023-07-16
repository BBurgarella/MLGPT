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
    # the numpyObject - np.max(numpyObject) is here to avoid infinty
    # as this function is invariant if offset by a constant
    
    # I also noted while debugging that the softmax layer in Kera actually is applied row by row
    # and not on the whole matrix, this makes sense if you think about how data is passed into machine learning models
    Exponential = np.exp(numpyObject - np.max(numpyObject, axis=-1, keepdims=True))
    return Exponential/np.sum(Exponential, axis=-1, keepdims=True)

    
def scaledDotProductSelfAttention(Q,K,V, mask=None):
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
    Scores = np.dot(Q,K.T)/np.sqrt(float(keyDimension))

    weights = softmax(Scores)

    # in the video, they used a tf.where and put the value to -np.inf
    # I think, that this is better and maybe more efficient, I may be wrong though
    if mask is not None:
        weights = weights * mask

    return np.dot(weights, V), weights


if __name__=="__main__":
    
        Queries = np.array([[0.29917916,0.32222616,0.44944202,0.40087818],
                            [0.42742966,0.75929284,0.94481094,0.16952069],
                            [0.66734589,0.18528671,0.72916473,0.01845809]])
        
        Keys = np.array([[0.36584701,0.43562185,0.4480778 ,0.90873129],
                         [0.31221214,0.07409408,0.83602278,0.04485871],
                         [0.87830161,0.9997026 ,0.95931107,0.09707838]])
        
        Values = np.array([[0.35764971,0.1772979 ,0.60785344,0.87649764],
                           [0.07441695,0.98663028,0.18366487,0.38199649],
                           [0.10817892,0.45987829,0.05264802,0.53601417]])

        ExpectedOutput = np.array([[0.18201628,0.5167383,0.27626652,0.6057456 ],
                                   [0.1683729 ,0.52207  ,0.24163088,0.5893614 ],
                                   [0.16761667,0.5406556,0.24742973,0.5844119 ]])
        
        ExpectedWeights = np.array([[0.33489865,0.28760365,0.3774977 ],
                                    [0.277405  ,0.26688248,0.4557126 ],
                                    [0.27928528,0.30317473,0.41753995]])

        Output, weights = scaledDotProductSelfAttention(Queries, Keys, Values)

        diffOut = np.sum(ExpectedOutput - Output)
        diffWeights = np.sum(ExpectedWeights - weights)

        print(np.sum(ExpectedOutput - Output))
        print(np.sum(ExpectedWeights - weights))