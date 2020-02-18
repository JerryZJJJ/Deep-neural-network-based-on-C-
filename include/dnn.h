#pragma once
#define NDEBUG
#include <cassert>
#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <cstddef>
#include <ctime>
#include <cmath>
#include <Eigen/Dense>

/*	优化措施：
	1.编译时加上 -fopen 参数使用 openmp 开启多线程，避免 hyper－thread ------ ×
	2.矩阵点乘的trick: B.noalias() = dot(A.transpose(),A) ------ ×
	3.使用 Intel MKL 加速 ----- √
	4.VS下选Release，g++下打O3优化 ------ √
	5.sse或avx优化 ------ ?
*/

/*
Input Arguments:
n_x -- size of the input layer
n_h -- size of the hidden layer
n_y -- size of the output layer

Output Arguments:
parameters -- python dictionary containing your parameters:
W1 -- weight matrix of shape (n_h, n_x)
b1 -- bias vector of shape (n_h, 1)
W2 -- weight matrix of shape (n_y, n_h)
b2 -- bias vector of shape (n_y, 1)
*/
void initialize_parameters(const int n_x, const int n_h, const int n_y, std::map<std::string, Eigen::MatrixXd>& parameters)
{
	parameters["W1"] = Eigen::MatrixXd::Zero(n_h, n_x);
	parameters["b1"] = Eigen::MatrixXd::Zero(n_h, 1);
	parameters["W2"] = Eigen::MatrixXd::Zero(n_y, n_h);
	parameters["b2"] = Eigen::MatrixXd::Zero(n_y, 1);
	static std::default_random_engine e(time(0));
	static std::normal_distribution<double> n(0, 1);
	parameters["W1"] = parameters["W1"].unaryExpr([](double dummy) {return n(e) * 0.01; });//乘0.01是为了减小初始权重，否则可能导致梯度爆炸
	parameters["W2"] = parameters["W2"].unaryExpr([](double dummy) {return n(e) * 0.01; });
	assert(parameters["W1"].rows() == n_h);
	assert(parameters["W1"].cols() == n_x);
	assert(parameters["b1"].rows() == n_h);
	assert(parameters["b1"].cols() == 1);
	assert(parameters["W2"].rows() == n_y);
	assert(parameters["W2"].cols() == n_h);
	assert(parameters["b2"].rows() == n_y);
	assert(parameters["b2"].cols() == 1);
}

/*
Implements the sigmoid activation in OpenCV

Input Arguments :
Z -- input OpenCV Mat of any shape

Ouput Arguments :
A -- output of sigmoid(Z), same shape as Z
cache -- ouput Z as well, useful during backpropagation
*/
void sigmoid(const Eigen::MatrixXd& Z, Eigen::MatrixXd& A, std::vector<Eigen::MatrixXd>& cache)
{
	A = 1 / (1 + (-Z).array().exp());
	assert(A.size() == Z.size());
	cache.push_back(Z);
}

/*
Implement the RELU function.

Input Arguments:
Z -- Output of the linear layer, of any shape

Output Arguments:
A -- Post-activation parameter, of the same shape as Z
cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
*/
void relu(const Eigen::MatrixXd& Z, Eigen::MatrixXd& A, std::vector<Eigen::MatrixXd>& cache)
{
	A = Z.array().max(0);
	assert(A.rows() == Z.rows());
	assert(A.cols() == Z.cols());
	cache.push_back(Z);
}
void tanh(const Eigen::MatrixXd& Z, Eigen::MatrixXd& A, std::vector<Eigen::MatrixXd>& cache){
	A = (Z.array().exp()-(-Z).array().exp()) / (Z.array().exp() + (-Z).array().exp());
	assert(A.size() == Z.size());
	cache.push_back(Z);
}

void softmax(const Eigen::MatrixXd& Z, Eigen::MatrixXd& A, std::vector<Eigen::MatrixXd>& cache)
{
	// // Eigen::matrixXd Deno(1,1,1)
	Eigen::Vector3d C(1,1,1);
	Eigen::MatrixXd D = (Eigen::MatrixXd)(Z.array().exp().transpose()) * C;
	A = (Eigen::MatrixXd)(Z.array().exp()) * D.asDiagonal().inverse();

	assert(A.rows() == Z.rows());
	assert(A.cols() == Z.cols());
	cache.push_back(Z);
}
/*
Implement the backward propagation for a single RELU unit.

Input Arguments:
dA -- post-activation gradient, of any shape
cache -- 'Z' where we store for computing backward propagation efficiently

Output Arguments:
dZ -- Gradient of the cost with respect to Z
*/
void relu_backward(const Eigen::MatrixXd& dA, const Eigen::MatrixXd& cache, Eigen::MatrixXd& dZ)
{
	dZ = dA;
	assert(dZ.rows() == cache.rows());
	assert(dZ.cols() == cache.cols());
	for (int i = 0; i < cache.rows(); ++i) {
		for (int j = 0; j < cache.cols(); ++j) {
			if (cache(i, j) <= 0) {
				dZ(i, j) = 0;
			}
		}
	}
}

/*
Implement the backward propagation for a single SIGMOID unit.

Input Arguments:
dA -- post-activation gradient, of any shape
cache -- 'Z' where we store for computing backward propagation efficiently

Output Arguments:
dZ -- Gradient of the cost with respect to Z
*/
void sigmoid_backward(const Eigen::MatrixXd& dA, const Eigen::MatrixXd& cache, Eigen::MatrixXd& dZ)
{
	Eigen::MatrixXd s = 1 / (1 + (-cache).array().exp());
	dZ = dA.array() * s.array() * (1 - s.array());//对应元素相乘
	assert(dZ.rows() == cache.rows());
	assert(dZ.cols() == cache.cols());
}

void tanh_backward(const Eigen::MatrixXd& dA, const Eigen::MatrixXd& cache, Eigen::MatrixXd& dZ)
{
	Eigen::MatrixXd s = 2 / (cache.array().exp() + (-cache).array().exp());
	dZ = dA.array() * s.array() * s.array();//对应元素相乘
	assert(dZ.rows() == cache.rows());
	assert(dZ.cols() == cache.cols());
}
/*
Implement the linear part of a layer's forward propagation.

Input Arguments:
A -- activations from previous layer (or input data): (size of previous layer, number of examples)
W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
b -- bias vector, numpy array of shape (size of the current layer, 1)

Output Arguments:
Z -- the input of the activation function, also called pre-activation parameter
cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
*/
void linear_forward(const Eigen::MatrixXd& A, const Eigen::MatrixXd& W, const Eigen::VectorXd& b, Eigen::MatrixXd& Z, std::vector<Eigen::MatrixXd>& cache)
{
	Z = (W * A).colwise() + b;
	assert(Z.rows() == W.rows());
	assert(Z.cols() == A.cols());
	cache.push_back(A);
	cache.push_back(W);
	cache.push_back(b); // VectorXd隐式转换为MatrixXd
}

/*
Implement the forward propagation for the LINEAR->ACTIVATION layer

Input Arguments:
A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
b -- bias vector, numpy array of shape (size of the current layer, 1)
activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

Output Arguments:
A -- the output of the activation function, also called the post-activation value
cache -- a python dictionary containing "linear_cache" and "activation_cache";
		stored for computing the backward pass efficiently
*/
void linear_activation_forward(const Eigen::MatrixXd& A_prev, const Eigen::MatrixXd& W, const Eigen::VectorXd& b, const std::string activation, Eigen::MatrixXd& A, std::vector<std::vector<Eigen::MatrixXd>>& cache)
{
	Eigen::MatrixXd Z;
	std::vector<Eigen::MatrixXd> linear_cache;
	std::vector<Eigen::MatrixXd> activation_cache;
	if (activation == "sigmoid") {
		linear_forward(A_prev, W, b, Z, linear_cache);
		sigmoid(Z, A, activation_cache);
	}
	else if (activation == "relu") {
		linear_forward(A_prev, W, b, Z, linear_cache);
		relu(Z, A, activation_cache);
	}else if (activation == "softmax") {
		linear_forward(A_prev, W, b, Z, linear_cache);
		softmax(Z, A, activation_cache);
	}else if (activation == "tanh"){
		linear_forward(A_prev, W, b, Z, linear_cache);
		tanh(Z, A, activation_cache);
	}

	assert(A.rows() == W.rows());
	assert(A.cols() == A_prev.cols());
	cache.push_back(linear_cache);
	cache.push_back(activation_cache);
}

/*
Implement the cost function defined by equation (7).

Input Arguments:
AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

Output Arguments:
cost -- cross-entropy cost
*/
double compute_cost(const Eigen::MatrixXd& AL, const Eigen::MatrixXd& Y)
{
	int m = Y.cols();
	Eigen::MatrixXd cost = (1.0 / m) * (-(Y * (Eigen::MatrixXd)AL.array().log().transpose()) - (Eigen::MatrixXd)(1 - Y.array()) * (Eigen::MatrixXd)((1 - AL.array()).log().transpose()));
	assert(cost.rows() == 1);
	assert(cost.cols() == 1);
	return cost(0, 0);
}

double compute_cost_softmax(const Eigen::MatrixXd& AL, const Eigen::MatrixXd& Y)
{
	int m = Y.cols();
	Eigen::MatrixXd cost = (1.0 / m) * (Eigen::MatrixXd)(-(Y.array() * AL.array().log()));
	return cost.sum();
}

/*
Implement the linear portion of backward propagation for a single layer (layer l)

	Arguments:
	dZ -- Gradient of the cost with respect to the linear output (of current layer l)
	cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

	Returns:
	dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
	dW -- Gradient of the cost with respect to W (current layer l), same shape as W
	db -- Gradient of the cost with respect to b (current layer l), same shape as b
*/
void linear_backward(const Eigen::MatrixXd& dZ, std::vector<Eigen::MatrixXd>& cache, Eigen::MatrixXd& dA_prev, Eigen::MatrixXd& dW, Eigen::VectorXd& db)
{
	Eigen::MatrixXd A_prev = cache[0];
	Eigen::MatrixXd W = cache[1];
	Eigen::VectorXd b = cache[2]; //MatrixXd隐式转换为VectorXd
	int m = A_prev.cols();

	dW = (1.0 / m) * dZ * A_prev.transpose();
	db = (1.0 / m) * dZ.rowwise().sum();
	dA_prev = W.transpose() * dZ;

	assert(dA_prev.rows() == A_prev.rows());
	assert(dA_prev.cols() == A_prev.cols());
	assert(dW.rows() == W.rows());
	assert(dW.cols() == W.cols());
	assert(db.rows() == b.rows());
	assert(db.cols() == b.cols());
}

/*
Implement the backward propagation for the LINEAR->ACTIVATION layer.

	Arguments:
	dA -- post-activation gradient for current layer l
	cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
	activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

	Returns:
	dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
	dW -- Gradient of the cost with respect to W (current layer l), same shape as W
	db -- Gradient of the cost with respect to b (current layer l), same shape as b
*/
void linear_activation_backward(const Eigen::MatrixXd& dA, std::vector<std::vector<Eigen::MatrixXd>>& cache, std::string activation, Eigen::MatrixXd& dA_prev, Eigen::MatrixXd& dW, Eigen::VectorXd& db)
{
	std::vector<Eigen::MatrixXd> linear_cache = cache[0];
	std::vector<Eigen::MatrixXd> activation_cache = cache[1];
	Eigen::MatrixXd dZ;

	if (activation == "relu") {
		relu_backward(dA, activation_cache[0], dZ);
		linear_backward(dZ, linear_cache, dA_prev, dW, db);
	}
	else if (activation == "sigmoid") {
		sigmoid_backward(dA, activation_cache[0], dZ);
		linear_backward(dZ, linear_cache, dA_prev, dW, db);
	}
	else if (activation == "tanh"){
		tanh_backward(dA, activation_cache[0], dZ);
		linear_backward(dZ, linear_cache, dA_prev, dW, db);
	}
}

/*
Update parameters using gradient descent

	Arguments:
	parameters -- python dictionary containing your parameters
	grads -- python dictionary containing your gradients, output of L_model_backward

	Returns:
	parameters -- python dictionary containing your updated parameters
				  parameters["W" + str(l)] = ...
				  parameters["b" + str(l)] = ...
*/
void update_parameters(std::map<std::string, Eigen::MatrixXd>& parameters, std::map<std::string, Eigen::MatrixXd> grads, double learning_rate)
{
	std::size_t L = parameters.size() / 2;
	for (int i = 0; i < L; ++i) {
		parameters["W" + std::to_string(i + 1)] = parameters["W" + std::to_string(i + 1)] - learning_rate * grads["dW" + std::to_string(i + 1)];
		parameters["b" + std::to_string(i + 1)] = parameters["b" + std::to_string(i + 1)] - learning_rate * grads["db" + std::to_string(i + 1)];
	}
}

/*
Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

Input Arguments:
X -- data, numpy array of shape (input size, number of examples)
parameters -- output of initialize_parameters_deep()

Ouput Arguments:
AL -- last post-activation value
caches -- list of caches containing:
			every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
			the cache of linear_sigmoid_forward() (there is one, indexed L-1)
*/
void L_model_forward(const Eigen::MatrixXd& X, std::map<std::string, Eigen::MatrixXd>& parameters, Eigen::MatrixXd& AL, std::vector<std::vector<std::vector<Eigen::MatrixXd>>>& caches)
{
	Eigen::MatrixXd A = X;

	std::size_t L = parameters.size() / 2;
	//Implement[LINEAR->RELU] * (L - 1).Add "cache" to the "caches" list.
	for (int i = 1; i < L; ++i) {
		std::vector<std::vector<Eigen::MatrixXd>> cache;
		Eigen::MatrixXd A_prev = A;
		linear_activation_forward(A_prev, parameters["W" + std::to_string(i)], parameters["b" + std::to_string(i)], "relu", A, cache);
		caches.push_back(cache);
	}
	std::vector<std::vector<Eigen::MatrixXd>> cache;
	linear_activation_forward(A, parameters["W" + std::to_string(L)], parameters["b" + std::to_string(L)], "sigmoid", AL, cache);
	caches.push_back(cache);

	assert(AL.rows() == 1);
	assert(AL.cols() == X.cols());
}

void L_model_forward_softmax(const Eigen::MatrixXd& X, std::map<std::string, Eigen::MatrixXd>& parameters, Eigen::MatrixXd& AL, std::vector<std::vector<std::vector<Eigen::MatrixXd>>>& caches)
{
	Eigen::MatrixXd A = X;

	std::size_t L = parameters.size() / 2;
	//Implement[LINEAR->RELU] * (L - 1).Add "cache" to the "caches" list.
	for (int i = 1; i < L; ++i) {
		std::vector<std::vector<Eigen::MatrixXd>> cache;
		Eigen::MatrixXd A_prev = A;
		linear_activation_forward(A_prev, parameters["W" + std::to_string(i)], parameters["b" + std::to_string(i)], "tanh", A, cache);
		caches.push_back(cache);
	}
	std::vector<std::vector<Eigen::MatrixXd>> cache;
	linear_activation_forward(A, parameters["W" + std::to_string(L)], parameters["b" + std::to_string(L)], "softmax", AL, cache);
	caches.push_back(cache);

	// assert(AL.rows() == 1);
	assert(AL.cols() == X.cols());
}


/*
This function is used to predict the results of a  L-layer neural network.

	Arguments:
	X -- data set of examples you would like to label
	parameters -- parameters of the trained model

	Returns:
	p -- predictions for the given dataset X
*/
void predict(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y, std::map<std::string, Eigen::MatrixXd>& parameters, Eigen::MatrixXd& p)
{
	int m = X.cols();
	std::size_t n = parameters.size() / 2;
	p.setZero(1, m);
	Eigen::MatrixXd probas;
	std::vector<std::vector<std::vector<Eigen::MatrixXd>>> caches;
	L_model_forward(X, parameters, probas, caches);
	for (int i = 0; i < probas.cols(); ++i) {
		if (probas(0, i) > 0.5) {
			p(0, i) = 1;
		}
		else {
			p(0, i) = 0;
		}
	}
	double accuracy = 0;
	for (int i = 0; i < m; ++i) {
		if (p(0, i) == y(0, i)) {
			accuracy += 1;
		}
	}
	accuracy /= m;
	std::cout << "Accuracy: " << accuracy << std::endl;
}

void predict_softmax(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y, std::map<std::string, Eigen::MatrixXd>& parameters, Eigen::MatrixXd& p)
{
	int m = X.cols();
	std::size_t n = parameters.size() / 2;
	p.setZero(3, m);
	Eigen::MatrixXd probas;
	std::vector<std::vector<std::vector<Eigen::MatrixXd>>> caches;
	L_model_forward_softmax(X, parameters, probas, caches);
	for (int i = 0; i < probas.cols(); ++i) {
		if (probas(0,i) > probas(1,i) && probas(0,i) > probas(2,i)){
			p(0,i) = 1;
		}else if (probas(1,i) > probas(0,i) && probas(1,i) > probas(2,i)){
			p(1,i) = 1;
		}else{
			p(2,i) = 1;
		}
	}

	double accuracy = 0;
	for (int i = 0; i < m; ++i) {
		if (p(0, i) == y(0, i) && p(1, i) == y(1, i)) {
			accuracy += 1;
		}
	}
	accuracy /= m;
	std::cout << "Accuracy: " << accuracy << std::endl;
	double macro_F1 =0 ;
	double micro_F1 =0;
	int FP_all = 0;
	int FN_all = 0;
	int TP_all = 0;
	int TN_all = 0;
	for (int k =0;k<3;++k){
		int FP = 0;
		int FN = 0;
		int TP = 0;
		int TN = 0;
		for(int i =0 ;i<m;++i){
			if (p(k,i) == 1 && y(k,i) == 1){
				TP += 1;
			}else if(p(k,i) == 1 && y(k,i) == 0){
				FP += 1;
			}else if(p(k,i) == 0 && y(k,i) == 1){
				FN += 1;
			}else{
				TN += 1;
			}
		}
		FP_all += FP; 
		FN_all += FN;
		TP_all += TP; 
		TN_all += TN; 
		double Recall = (double) TP/ (TP+FN);
		double Precision = (double) TP/ (TP+FP);
		macro_F1 += 2*Recall*Precision / (Recall + Precision)/3;
		printf("Recall %.2F, Precision %.2f",Recall,Precision);
	}
	double Recall_all = (double) TP_all/ (TP_all+FN_all);
	double Precision_all = (double) TP_all/ (TP_all+FP_all);
	micro_F1 = 2*Recall_all*Precision_all/(Recall_all + Precision_all);
	std::cout<<"macro_F1: "<<macro_F1<<std::endl;
	std::cout<<"micro_F1: "<<micro_F1<<std::endl;
	printf("%.2f",micro_F1);


}



/*
Input Arguments:
layer_dims -- python array (list) containing the dimensions of each layer in our network

Output Arguments:
parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
bl -- bias vector of shape (layer_dims[l], 1)
*/
void initialize_parameters_deep(const std::vector<int>& layer_dims, std::map<std::string, Eigen::MatrixXd>& parameters)
{
	int L = layer_dims.size();
	for (int i = 1; i < L; ++i) {
		parameters["W" + std::to_string(i)] = Eigen::MatrixXd::Zero(layer_dims[i], layer_dims[i - 1]);
		parameters["b" + std::to_string(i)] = Eigen::MatrixXd::Zero(layer_dims[i], 1);
		static std::default_random_engine e(time(0));
		static std::normal_distribution<double> n(0, 1);
		parameters["W" + std::to_string(i)] = parameters["W" + std::to_string(i)].unaryExpr([i, layer_dims](double dummy) {return n(e) / std::sqrt(layer_dims[i - 1]); });
		assert(parameters["W" + std::to_string(i)].rows() == layer_dims[i]);
		assert(parameters["W" + std::to_string(i)].cols() == layer_dims[i - 1]);
		assert(parameters["b" + std::to_string(i)].rows() == layer_dims[i]);
		assert(parameters["b" + std::to_string(i)].cols() == 1);
	}
}

/*
Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

	Arguments:
	AL -- probability vector, output of the forward propagation (L_model_forward())
	Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
	caches -- list of caches containing:
				every cache of linear_activation_forward() with "relu" (there are (L-1) or them, indexes from 0 to L-2)
				the cache of linear_activation_forward() with "sigmoid" (there is one, index L-1)

	Returns:
	grads -- A dictionary with the gradients
			 grads["dA" + str(l)] = ...
			 grads["dW" + str(l)] = ...
			 grads["db" + str(l)] = ...
*/
void L_model_backward(const Eigen::MatrixXd& AL, Eigen::MatrixXd& Y, std::vector<std::vector<std::vector<Eigen::MatrixXd>>>& caches,
	std::map<std::string, Eigen::MatrixXd>& grads)
{
	std::size_t L = caches.size();
	int m = AL.cols();
	Y.resize(AL.rows(), AL.cols());
	Eigen::MatrixXd dAL_temp = -((Y.array() / AL.array()) - ((1 - Y.array()) / (1 - AL.array())));
	std::vector<std::vector<Eigen::MatrixXd>> current_cache = caches[L - 1];
	Eigen::MatrixXd dAL, dWL;
	Eigen::VectorXd dbL;
	linear_activation_backward(dAL_temp, current_cache, "sigmoid", dAL, dWL, dbL);
	grads["dA" + std::to_string(L)] = dAL;
	grads["dW" + std::to_string(L)] = dWL;
	grads["db" + std::to_string(L)] = dbL;
	for (int i = L - 2; i >= 0; --i) {
		std::vector<std::vector<Eigen::MatrixXd>> current_cache = caches[i];
		Eigen::MatrixXd dA_prev_temp, dW_temp;
		Eigen::VectorXd db_temp;
		linear_activation_backward(grads["dA" + std::to_string(i + 2)], current_cache, "relu", dA_prev_temp, dW_temp, db_temp);
		grads["dA" + std::to_string(i + 1)] = dA_prev_temp;
		grads["dW" + std::to_string(i + 1)] = dW_temp;
		grads["db" + std::to_string(i + 1)] = db_temp;
	}
}

void L_model_backward_softmax(const Eigen::MatrixXd& AL, Eigen::MatrixXd& Y, std::vector<std::vector<std::vector<Eigen::MatrixXd>>>& caches,
	std::map<std::string, Eigen::MatrixXd>& grads)
{
	std::size_t L = caches.size();
	int m = AL.cols();
	Y.resize(AL.rows(), AL.cols());
	
	// dZL = AL - Y;
	Eigen::MatrixXd dZL = AL.array() - Y.array();

	std::vector<std::vector<Eigen::MatrixXd>> current_cache = caches[L - 1];  //存的ZL
	std::vector<Eigen::MatrixXd> linear_cache = current_cache[0];	//存的A_L-1, WL, dL
	Eigen::MatrixXd dAL, dWL;
	Eigen::VectorXd dbL;
	linear_backward(dZL, linear_cache, dAL, dWL, dbL);
	grads["dA" + std::to_string(L)] = dAL;
	grads["dW" + std::to_string(L)] = dWL;
	grads["db" + std::to_string(L)] = dbL;

	for (int i = L - 2; i >= 0; --i) {
		std::vector<std::vector<Eigen::MatrixXd>> current_cache = caches[i];
		Eigen::MatrixXd dA_prev_temp, dW_temp;
		Eigen::VectorXd db_temp;
		linear_activation_backward(grads["dA" + std::to_string(i + 2)], current_cache, "tanh", dA_prev_temp, dW_temp, db_temp);
		grads["dA" + std::to_string(i + 1)] = dA_prev_temp;
		grads["dW" + std::to_string(i + 1)] = dW_temp;
		grads["db" + std::to_string(i + 1)] = db_temp;
	}
}