# online_learning_libs
This is a python implementation of online learning methods using Iterative Parameter Mixture.
This implementation is now supporting:
```
    - Perceptron
    - PA-I, PA-II
    - CW
    - AROW
    - SCW-I, SCW-II
```

Note that, AROW and SCW is implemented based on approximation algorithm, i.e. covariance matrix is updated by diag(x) instead of x^{T}x.

# Requirements
* python 2.7
* tqdm
* numpy
* scipy
* joblib

# Example
## Training
```python
from updater import Updater
from weight import Weight # if you want to use CW, AROW or SCW, use DistWeight instead of Weight

# number of maximum epochs
epochs = 100

# number of maximum number of features
max_feature_num = 5

# aggressive parameter for PA, CW and SCW
C = 0.01

# confidence parameter for CW and SCW
eta = 0.1

# regularization parameter for AROW
r = 0.1

# number of parallerization
parallel_num = 6

# make training data
# x_train represents feature vector using scipy.sparse.csr_matrix
# y_train represents labels (1 or -1) corresponding to each feature_vector
# e.g. discretized feature vector and labels:
# 	x_train -> scipy.sparse.csr_matrix( [[0.0, 0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0, 0.0], ...] )
# 	y_train -> [1.0, -1.0, ...]
# Also we can use non-discretized (real valued) feature vectors
x_train, y_train = make_data()

weight = Weight(max_feature_num)

# you can choose method for updater as follows:
#	- Perceptrion
#	- PA-I, PA-II (default)
#	- CW
#	- AROW
#	- SCW-I, SCW-II
updater = Updater(C=C, r=r, eta=eta, process_num=parallel_num, method="PA-II")

for _ in xrange(epochs):
	# update weight and get list of loss
	loss_list = updater.update(x_train, y_train, weight)
	# you can use averaged loss value to check the learning
	loss = sum(loss_list) / float(len(loss_list))
	# dump weight parameter
	weight.dump_weight("./models/pa")
```

## Testing
```python
from weight import Weight # if you want to use CW, AROW and SCW, use DistWeight instead of Weight
from predictor import Predictor

# make test data
# x_test represents feature vector using scipy.sparse.csr_matrix 
# y_test represents labels (1 or -1) corresponding to each feature_vector
# e.g. discretized feature vector and labels:
# 	x_test -> scipy.sparse.csr_matrix( [[0.0, 0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0, 0.0], ...] )
# 	y_test -> [1.0, -1.0, ...]
# Also we can use non-discretized (real valued) feature vectors
x_test, y_test = make_data()

# load trained weight parameters from model file
# second argument means number of epochs for weight that you want to load
weight = Weight()
weight.load_weight("./models/pa", 30) 

# you can set options as follows:
#	"confidence": predictor returns confidence score for each prediction (real value)
#       "classify": predictor returns {1, -1} labels according to confidence score for each prediction
predictor = Predictor("classify")

# get result on the prediction for x_test
y_pred = predictor.predict(x_test, weight)
```
