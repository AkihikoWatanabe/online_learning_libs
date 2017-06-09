# coding=utf-8

class Predictor():
    """ This class predicts confidence (label) of each features based on given weight class.
    """
    
    def __init__(self, mode="confidence"):
        """
        Params:
            mode(str): Set mode for prediction:
                "confidence": returns confidence score for each prediction (real value)
                "classify": returns {1, -1} labels according to confidence score for each prediction
        """
        self.MODE = mode

    def predict(self, x_list, weight):
        """
        Params:
            x_list(list): List of feature vectors. Each vector is represented by np.ndarray
            weight(Weight): weight class to use prediction
        Returns:
            y_list(list): List of result labels (or confidence score) on the prediction
        """
        if self.mode == "confidence":
            return [weight.w.dot(x) for x in x_list]
        elif self.mode == "classify":
            return [1.0 if weight.w.dot(x)>=0.0 else -1.0 for x in x_list]
