#include <iostream>
#include <boost/random.hpp>
// #include <opencv2/core/mat.hpp>
#include <Eigen/Dense>
//使用Intel MKL优化Eigen
#define EIGEN_USE_MKL_ALL

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <random>
#include <ctime>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include "dnn.h"

using namespace std;
using namespace Eigen;
using namespace cv;

void test_initialize_parameters()
{
	cout << "======Test Initialize_parameters======" << endl;
	map<string, MatrixXd> parameters;
	initialize_parameters(5, 2, 1, parameters);
	cout << "W1 = \n" << parameters["W1"] << endl;
	cout << "b1 = \n" << parameters["b1"] << endl;
	cout << "W2 = \n" << parameters["W2"] << endl;
	cout << "b2 = \n" << parameters["b2"] << endl;
	cout << "======================================" << endl;
}

void test_sigmoid()
{
	cout << "======Test Sigmoid======" << endl;
	MatrixXd Z;
	Z.setZero(3, 3);
	MatrixXd A;
	vector< MatrixXd> cache;
	sigmoid(Z, A, cache);
	cout << "cache = \n" << cache[0] << endl;
	cout << "A = \n" << A << endl;
	cout << "========================" << endl;
}

void test_relu()
{
	cout << "======Test Relu======" << endl;
	MatrixXd Z = MatrixXd::Random(3, 3);
	MatrixXd A;
	vector< MatrixXd> cache;
	relu(Z, A, cache);
	cout << "cache = \n" << cache[0] << endl;
	cout << "A = \n" << A << endl;
	cout << "=====================" << endl;
}

void test_tanh()
{
	cout << "======Test Tanh======" << endl;
	MatrixXd Z = MatrixXd::Random(3, 3);
	MatrixXd A;
	vector< MatrixXd> cache;
	tanh(Z, A, cache);
	cout << "cache = \n" << cache[0] << endl;
	cout << "A = \n" << A << endl;
	cout << "=====================" << endl;
}

void test_relu_backward()
{
	cout << "======Test Relu_backward======" << endl;
	MatrixXd dA = MatrixXd::Ones(3, 3);
	MatrixXd cache = MatrixXd::Random(3, 3);
	MatrixXd dZ;
	relu_backward(dA, cache, dZ);
	cout << "dA = \n" << dA << endl;
	cout << "cache = \n" << cache << endl;
	cout << "dZ = \n" << dZ << endl;
	cout << "==============================" << endl;
}

void test_sigmoid_backward()
{
	cout << "======Test Sigmoid_backward======" << endl;
	MatrixXd dA(3, 3), cache(3, 3), dZ;
	dA << 0, 1, 2, 3, 4, 5, 6, 7, 8;
	cache << -3, -2, -1, 0, 1, 2, 3, 4, 5;
	sigmoid_backward(dA, cache, dZ);
	cout << "dA = \n" << dA << endl;
	cout << "cache = \n" << cache << endl;
	/* Expected dZ
	[[0.         0.10499359 0.39322387]
	[0.75       0.78644773 0.52496793]
	[0.2	7105996 0.12363894 0.05318445]]
	*/
	cout << "dZ = \n" << dZ << endl;
	cout << "=================================" << endl;
}

void test_tanh_backward()
{
	cout << "======Test Tanh_backward======" << endl;
	MatrixXd dA(3, 3), cache(3, 3), dZ;
	dA << 0, 1, 2, 3, 4, 5, 6, 7, 8;
	cache << -3, -2, -1, 0, 1, 2, 3, 4, 5;
	tanh_backward(dA, cache, dZ);
	cout << "dA = \n" << dA << endl;
	cout << "cache = \n" << cache << endl;
	/* Expected dZ
	[[0.         0.10499359 0.39322387]
	[0.75       0.78644773 0.52496793]
	[0.2	7105996 0.12363894 0.05318445]]
	*/
	cout << "dZ = \n" << dZ << endl;
	cout << "=================================" << endl;
}

void test_linear_forward()
{
	cout << "======Test Linear_forward======" << endl;
	MatrixXd W(3, 4), A(4, 2), b(3, 1), Z;
	vector< MatrixXd> cache;
	W << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11;
	A << -4, -3, -2, -1, 0, 1, 2, 3;
	b << 1, 2, 3;
	linear_forward(A, W, b, Z, cache);
	/* Expected Z
	[[  5  11]
	[-10  12]
	[-25  13]]
	*/
	cout << "A = \n" << A << endl;
	cout << "W = \n" << W << endl;
	cout << "b = \n" << b << endl;
	cout << "Z = \n" << Z << endl;
	for (auto x : cache) {
		cout << "---------------" << endl;
		cout << x << endl;
		cout << "---------------" << endl;
	}
	cout << "===============================" << endl;
}

void test_linear_activation_forward()
{
	cout << "======Test Linear_forward======" << endl;
	MatrixXd W(3, 4), A_prev(4, 2), b(3, 1), A;
	vector<vector< MatrixXd>> cache;
	W << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11;
	A_prev << -4, -3, -2, -1, 0, 1, 2, 3;
	b << 1, 2, 3;
	linear_activation_forward(A_prev, W, b, "relu", A, cache);
	/* Expected A -- sigmoid
	[[9.93307149e-01 9.99983299e-01]
	 [4.53978687e-05 9.99993856e-01]
	 [1.38879439e-11 9.99997740e-01]]

	 Expected A -- relu
	 [[ 5 11]
	 [ 0 12]
	 [ 0 13]]
	*/
	cout << "A_prev = \n" << A_prev << endl;
	cout << "W = \n" << W << endl;
	cout << "b = \n" << b << endl;
	cout << "A = \n" << A << endl;
	for (auto& x : cache) {
		for (auto& y : x) {
			cout << "-----------------" << endl;
			cout << y << endl;
			cout << "-----------------" << endl;
		}
	}
	cout << "===============================" << endl;
}

void test_compute_cost()
{
	cout << "======Test Compute_cost======" << endl;
	MatrixXd AL(1, 9), Y(1, 9);
	AL << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9;
	Y << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9;
	double cost = compute_cost(AL, Y);
	/* Expected cost: 0.5457633236163908
	*/
	cout << "AL = \n" << AL << endl;
	cout << "Y = \n" << Y << endl;
	cout << "cost = " << cost << endl;
	cout << "=============================" << endl;
}

void test_compute_cost_softmax()
{
	cout << "======Test Compute_cost======" << endl;
	MatrixXd AL(3, 4), Y(3, 4);
	AL << 0.1, 0.2, 0.3,0.4, 
		0.4, 0.5, 0.6,0.5, 
		0.7, 0.8, 0.9,0.6;
	Y << 0,1,0,1,
		0,0,1,0,
		1,0,0,0;
	double cost = compute_cost_softmax(AL, Y);
	/* Expected cost: 0.5457633236163908
	*/
	cout << "AL = \n" << AL << endl;
	cout << "Y = \n" << Y << endl;
	cout << "cost = " << cost << endl;
	cout << "=============================" << endl;
}


void test_linear_backward()
{
	cout << "======Test Linear_backward======" << endl;
	MatrixXd dZ(4, 3), A_prev(3, 3), W(4, 3), b(4, 1);
	MatrixXd dA_prev, dW;
	VectorXd db;
	dZ << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
	A_prev << 1, 2, 3, 4, 5, 6, 7, 8, 9;
	W << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
	b << 1, 2, 3, 4;
	vector< MatrixXd> cache;
	cache.push_back(A_prev);
	cache.push_back(W);
	cache.push_back(b);
	linear_backward(dZ, cache, dA_prev, dW, db);
	/* Expected dA_prev,dW,db:
	[[166 188 210]
	 [188 214 240]
	 [210 240 270]]

	[[ 4.66666667 10.66666667 16.66666667]
	 [10.66666667 25.66666667 40.66666667]
	 [16.66666667 40.66666667 64.66666667]
	 [22.66666667 55.66666667 88.66666667]]

	[[ 2.]
	 [ 5.]
	 [ 8.]
	 [11.]]
	*/
	cout << "dA_prev = \n" << dA_prev << endl;
	cout << "dW = \n" << dW << endl;
	cout << "db = " << db << endl;
	cout << "=============================" << endl;
}

void test_linear_activation_backward()
{
	cout << "======Test Linear_activation_backward======" << endl;
	MatrixXd dA(4, 3), A_prev(3, 3), W(4, 3), b(4, 1);
	MatrixXd dA_prev, dW;
	VectorXd db;
	dA << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
	A_prev << 1, 2, 3, 4, 5, 6, 7, 8, 9;
	W << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
	b << 1, 2, 3, 4;
	vector<MatrixXd> linear_cache;
	linear_cache.push_back(A_prev);
	linear_cache.push_back(W);
	linear_cache.push_back(b);
	vector<MatrixXd> activation_cache;
	MatrixXd acca(4, 3);
	acca << -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6;
	activation_cache.push_back(acca);
	vector<vector< MatrixXd>> cache;
	cache.push_back(linear_cache);
	cache.push_back(activation_cache);
	linear_activation_backward(dA, cache, "relu", dA_prev, dW, db);
	/* Expected dA_prev,dW,db --- sigmoid:

	[[13.08680077 10.57849109  9.27764066]
	 [15.06633377 12.50995348 11.34935869]
	 [17.04586676 14.44141586 13.42107671]]

	[[0.16129627 0.33879972 0.51630317]
	 [2.29536456 5.19839857 8.10143257]
	 [1.4253169  4.04813906 6.67096121]
	 [0.13722621 0.41658001 0.69593381]]

	[[0.05916782]
	 [0.967678  ]
	 [0.87427405]
	 [0.09311793]]

	 Expected dA_prev,dW,db --- relu:

	 [[149 166 183]
	 [166 185 204]
	 [183 204 225]]
	[[ 0.          0.          0.        ]
	 [ 0.          0.          0.        ]
	 [16.66666667 40.66666667 64.66666667]
	 [22.66666667 55.66666667 88.66666667]]
	[[ 0.]
	 [ 0.]
	 [ 8.]
	 [11.]]
	*/
	cout << "dA_prev = \n" << dA_prev << endl;
	cout << "dW = \n" << dW << endl;
	cout << "db = " << db << endl;
	cout << "=============================" << endl;
}

void test_update_parameters()
{
	cout << "======Test update_parameters======" << endl;
	map<string, MatrixXd> parameters;
	map<string, MatrixXd> grads;
	MatrixXd W1(2, 5), b1(2, 1), W2(1, 2), b2(1, 1);
	MatrixXd dW1(2, 5), db1(2, 1), dW2(1, 2), db2(1, 1);
	W1 << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
	b1 << 0, 0;
	W2 << 0, 1;
	b2 << 0;
	dW1 = W1;
	db1 = b1;
	dW2 = W2;
	db2 = b2;
	parameters["W1"] = W1;
	parameters["b1"] = b1;
	parameters["W2"] = W2;
	parameters["b2"] = b2;
	grads["dW1"] = dW1;
	grads["db1"] = db1;
	grads["dW2"] = dW2;
	grads["db2"] = db2;
	update_parameters(parameters, grads, 0.01);
	/* Expected parameters:
	'W1':
	[[0.  , 0.99, 1.98, 2.97, 3.96],
	[4.95, 5.94, 6.93, 7.92, 8.91]])
	'b1': [[0.],[0.]]
	'W2': [[0.  , 0.99]]
	'b2': [[0.]]
	*/
	cout << "parameters[W1] = \n" << parameters["W1"] << endl;
	cout << "parameters[b1] = \n" << parameters["b1"] << endl;
	cout << "parameters[W2] = \n" << parameters["W2"] << endl;
	cout << "parameters[b2] = \n" << parameters["b2"] << endl;
	cout << "=============================" << endl;
}

void two_layer_model(MatrixXd& X, MatrixXd& Y, vector<int>& layers_dims, double learning_rate, int num_iterations, bool print_cost, map<string, MatrixXd>& parameters)
{
	int m = X.cols();
	int n_x = layers_dims[0];
	int n_h = layers_dims[1];
	int n_y = layers_dims[2];

	initialize_parameters(n_x, n_h, n_y, parameters);

	for (int i = 0; i < num_iterations; ++i) {
		MatrixXd W1 = parameters["W1"];
		MatrixXd b1 = parameters["b1"];
		MatrixXd W2 = parameters["W2"];
		MatrixXd b2 = parameters["b2"];
		MatrixXd A1, A2;
		vector<vector<MatrixXd>> cache1, cache2;
		linear_activation_forward(X, W1, b1, "relu", A1, cache1);
		linear_activation_forward(A1, W2, b2, "sigmoid", A2, cache2);
		double cost = compute_cost(A2, Y);

		MatrixXd dA2;
		dA2 = -((Y.array() / A2.array()) - ((1 - Y.array()) / (1 - A2.array())));
		MatrixXd dA1, dW2;
		VectorXd db2;
		linear_activation_backward(dA2, cache2, "sigmoid", dA1, dW2, db2);
		MatrixXd dA0, dW1;
		VectorXd db1;
		linear_activation_backward(dA1, cache1, "relu", dA0, dW1, db1);

		map<string, MatrixXd> grads;
		grads["dW1"] = dW1;
		grads["db1"] = db1;
		grads["dW2"] = dW2;
		grads["db2"] = db2;

		update_parameters(parameters, grads, learning_rate);

		if (print_cost && i % 100 == 0) {
			cout << "Cost after iteration " << i << ": " << cost << endl;
		}
		//cout << "Cost after iteration " << i << ": " << cost << endl;
	}
}

void L_layer_model(MatrixXd& X, MatrixXd& Y, vector<int>& layers_dims, double learning_rate, int num_iterations, bool print_cost, map<string, MatrixXd>& parameters)
{
	initialize_parameters_deep(layers_dims, parameters);
	for (int i = 0; i < num_iterations; ++i) {
		MatrixXd AL;
		vector<vector<vector<MatrixXd>>> caches;
		L_model_forward(X, parameters, AL, caches);
		double cost = compute_cost(AL, Y);

		map<string, MatrixXd> grads;
		L_model_backward(AL, Y, caches, grads);
		update_parameters(parameters, grads, learning_rate);
		if (print_cost && i % 100 == 0) {
			cout << "Cost after iteration " << i << ": " << cost << endl;
		}
		//cout << "Cost after iteration " << i << ": " << cost << endl;
	}
}

void L_layer_model_softmax(MatrixXd& X, MatrixXd& Y, vector<int>& layers_dims, double learning_rate, int num_iterations, bool print_cost, map<string, MatrixXd>& parameters)
{
	initialize_parameters_deep(layers_dims, parameters);
	for (int i = 0; i < num_iterations; ++i) {
		MatrixXd AL;
		vector<vector<vector<MatrixXd>>> caches;
		L_model_forward_softmax(X, parameters, AL, caches); 
		double cost = compute_cost_softmax(AL, Y);

		map<string, MatrixXd> grads;
		L_model_backward_softmax(AL, Y, caches, grads);
		update_parameters(parameters, grads, learning_rate);
		if (print_cost && i % 100 == 0) {
			cout << "Cost after iteration " << i << ": " << cost << endl;
		}
		//cout << "Cost after iteration " << i << ": " << cost << endl;
	}
}

template<typename T>
T load_csv(const std::string& path) {
	std::ifstream indata;
	indata.open(path);
	std::string line;
	std::vector<double> values;
	int rows = 0;
	while (std::getline(indata, line)) {
		std::stringstream lineStream(line);
		std::string cell;
		while (std::getline(lineStream, cell, ',')) {
			values.push_back(std::stod(cell));
		}
		++rows;
	}
	return Map<const Matrix<typename T::Scalar, T::RowsAtCompileTime, T::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size() / rows);
}

void save_txt(const Eigen::MatrixXd& m, const std::string& path) {
	std::ofstream file(path);
	if (file.is_open())
	{
		file << m.rows() << " " << m.cols() << endl;
		file << m;
	}
	file.close();
}

Eigen::MatrixXd load_txt(const std::string& path)
{
	std::ifstream in(path);
	std::string line;
	std::getline(in, line);
	std::stringstream ss(line);
	int row, col;
	ss >> row >> col;
	Eigen::MatrixXd m(row, col);
	int i = 0, j = 0;
	while (std::getline(in, line)) {
		std::stringstream ss(line);
		double d;
		while (ss >> d) {
			if (j == col) {
				++i;
				j = 0;
			}
			m(i, j) = d;
			++j;
		}
	}
	in.close();
	return m;
}

int main()
{
	// vector<double> values ={1,2,3,4,5,6,7,8,9};
	// MatrixXd train_x(3,3);
	// MatrixXd train_y(1,3);
	// for (int i=0;i<3;i++){
	// 	train_y << 1,2,3;
	// 	for (int j=0;j<3;j++){
	// 		train_x(i,j) = values[3*i+j];
	// 	}
	// }
	// cout<<train_y<<endl;
	// cout<<train_x;

	// vector<vector<vector<double>>> values2(4,vector<vector<double>>(5,vector<double>(6,0)));
	// values2[0][1].push_back(5);



	//函数测试
	// test_initialize_parameters();
	// test_sigmoid(); 
	// test_relu();
	// test_tanh();
	// test_tanh_backward();
	// test_relu_backward();
	// test_sigmoid_backward();
	// test_linear_forward();
	// test_linear_activation_forward();
	// test_compute_cost();
	// test_linear_backward();
	// test_linear_activation_backward();
	// test_update_parameters();
	// test_compute_cost_softmax();

    // const MatrixXd A = MatrixXd::Random(3, 2);
    // const MatrixXd B = MatrixXd::Random(2, 6);
	// const MatrixXd B2 = MatrixXd::Random(3, 2);
    // const VectorXd C = VectorXd::Random(3);
    // const MatrixXd D = (A*B).colwise()+C;
    // cout<<A<<endl;
	// cout<<B2<<endl;
	// const MatrixXd D2 = A.array() * B2.array();
	// cout<<D2;
	// cout<<D2.sum();
	// const MatrixXd D2 = (A*B2) 
	// MatrixXd A;
 	// MatrixXd Z = MatrixXd::Random(3, 2);
	// cout<<Z<<endl;
	// Vector3d C(1,1,1);
	// MatrixXd D = (Eigen::MatrixXd)(Z.array().exp().transpose()) * C;

	// A = (Eigen::MatrixXd)(Z.array().exp()) * D.asDiagonal().inverse();
	// cout<<Z<<endl;
	
	// cout <<"D"<<D<<endl;
	// cout <<"D2"<<D2<<endl;
	// cout <<"A"<<A<<endl;
	

    // MatrixXd A = MatrixXd::Random(8, 200);
    // MatrixXd B = MatrixXd::Random(3, 200);
	// for (int i = 0;i<B.cols();i++){
	// 	if (A(0,i)>0.3333){
	// 		B(0,i) = 1;
	// 		B(1,i) = 0;
	// 		B(2,i) = 0;
	// 	}else if(A(0,i)<-0.3333){
	// 		B(0,i) = 0;
	// 		B(1,i) = 1;
	// 		B(2,i) = 0;
	// 	}else{
	// 		B(0,i) = 0;
	// 		B(1,i) = 0;
	// 		B(2,i) = 1;
	// 	}
	// } 


	// MatrixXd train_x = load_csv<MatrixXd>("./BackPropagationNeuralNetwork/CatData/train_x_iris.csv");
	// MatrixXd train_y = load_csv<MatrixXd>("./BackPropagationNeuralNetwork/CatData/train_y_iris.csv");
	// MatrixXd test_x = load_csv<MatrixXd>("./BackPropagationNeuralNetwork/CatData/test_x_iris.csv");
	// MatrixXd test_y = load_csv<MatrixXd>("./BackPropagationNeuralNetwork/CatData/test_y_iris.csv");

	// vector<int> layers_dims = { 3, 5, 5, 3 };// 5 - layer model
	// map<string, MatrixXd> parameters;
	// L_layer_model_softmax(train_x, train_y, layers_dims, 0.0075, 2000, true, parameters);
	// MatrixXd prediction;
	// predict_softmax(train_x, train_y, parameters, prediction);
	// predict_softmax(test_x, test_y, parameters, prediction);

	int m = 10;
    Eigen::MatrixXd Ask1(m,1); 
    Eigen::MatrixXd Bid_const(m,2);
    MatrixXd E = MatrixXd::Random(m, 2);
    MatrixXd F = MatrixXd::Random(m,1);
	Bid_const = E;
	Ask1 = F;
    Eigen::MatrixXd beta;
    beta = (Bid_const.transpose() * Bid_const).inverse() * Bid_const.transpose() * Ask1;
	cout<<beta<<endl;
	Eigen::MatrixXd y_hat = Bid_const * beta;
    double y_mean = Ask1.mean();
    double ss_fit = ((Ask1 - y_hat).transpose()*(Ask1 - y_hat))(0,0);
	cout <<ss_fit<<endl;

    double ss_mean = (((Eigen::MatrixXd)((Ask1.array() - y_mean)*(Ask1.array() - y_mean))).sum());
    double r2 = (ss_mean - ss_fit)/ ss_mean;
	cout<<r2<<endl; 
  
	// MatrixXd prediction2;
    // MatrixXd E = MatrixXd::Random(8, 20);
    // MatrixXd F = MatrixXd::Random(3, 20);
	// predict_softmax(E, F, parameters, prediction2);
	// cout << E<<endl;
	// cout <<F<<endl;
	// cout <<prediction2<<endl; 
	// MatrixXd test_x = load_csv<MatrixXd>("./BackPropagationNeuralNetwork/CatData/test_x.csv");
	// MatrixXd test_y = load_csv<MatrixXd>("./BackPropagationNeuralNetwork/CatData/test_y.csv");
	// cout << train_x.rows() << "*" << train_x.cols() << endl; 
	// cout << train_y.rows() << "*" << train_y.cols() << endl; 
 
	// clock_t start, end;
	// start = clock();

	//使用三层的神经网络
	// int n_x = train_x.rows();
	// int n_h = 7;
	// int n_y = 1;
	// vector<int> layers_dims = { n_x, n_h, n_y };
	// map<string, MatrixXd> parameters;
	// two_layer_model(train_x, train_y, layers_dims, 0.0075, 1000, true, parameters);
	// MatrixXd prediction;
	// predict(train_x, train_y, parameters, prediction);

	//使用五层的神经网络
	// vector<int> layers_dims = { 12288, 20, 7, 5, 1 };// 5 - layer model
	// map<string, MatrixXd> parameters;
	// L_layer_model(train_x, train_y, layers_dims, 0.0075, 2300, true, parameters);
	// MatrixXd prediction; 
	// predict(train_x, train_y, parameters, prediction);
	// end = clock();
	// cout << "totile time=" << (float)(end - start) * 2000 / CLOCKS_PER_SEC << "ms" << endl;
	// predict(test_x, test_y, parameters, prediction);

	//测试单个图片
	//Mat img = imread("Images\\9.jpg", IMREAD_COLOR);
	//Mat resizeImg;
	//resize(img, resizeImg, Size(64, 64));
	//vector<Mat> channels;
	//split(resizeImg, channels);
	//Mat chan01, threeChannels;
	//cv::vconcat(channels[0], channels[1], chan01);
	//cv::vconcat(chan01, channels[2], threeChannels);
	//MatrixXd m;
	//cv2eigen(threeChannels, m);
	//MatrixXd mt = m.transpose();
	//Map<Matrix< double, Dynamic, 1 >> line(mt.data(), mt.size());
	//MatrixXd result(line);
	////cout << result.rows() << "*" << result.cols() << endl;
	//MatrixXd y(1, 1);
	//y << 0; //不是猫
	//MatrixXd pre;
	//predict(result, y, parameters, pre);
	//cout << "prediction = " << pre << endl;
	system("pause");
	return 0;
}