/**
 * @file NeuralNetwork.cpp
 * @author  Thibault Lucidarme
 * @version 1.0
 *
 * @section LICENSE
 *
 * This library is under BSD License; you can redistribute it and/or
 * modify it under the condition that you cite the author of the original
 * library in the documentation.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details at
 * http://www.gnu.org/copyleft/gpl.html
 *
 * To support the author, please visit: auvercraft.no-ip.org
 * "Auvercraft : Let your crafty mind hover."
 *
 * @section DESCRIPTION
 *
 * The NeuralNetwork class implements a basic neural net.
 */

#ifndef NNETWORK
#define NNETWORK

//standard libraries
#include <cmath>
#include <iostream>
#include <vector>
#include <ctime>
#include <fstream>
#include <iomanip>   // needed for setprecision
#include <algorithm> // needed for for_each


namespace p
{
class NNEntry
{
private:

int					_id;

std::string 		_name;

std::vector<double> _value;

std::vector<double> _targetValue;

public:

NNEntry(void);

NNEntry(std::vector<double>, std::vector<double>);

~NNEntry(void);

void SetName(const std::string );

void SetValue(const std::vector<double> );

void SetValue(const double val, const unsigned int index);

void SetTargetValue(const std::vector<double> );

void SetTargetValue(const double val, const unsigned int index);

std::vector<double> GetValue(void);

std::vector<double> GetTargetValue(void);

std::string GetName( void );
};



class NeuralNetwork
{
//private attributes
//----------------------------------------------------------------------

private:

//Network structure
unsigned int				_nbLayers;
std::vector< unsigned int > _nbNodes;

//Nodes
std::vector< std::vector<double> > _neurons;

//weights
/** _weigth[i][j][k] is the weight linking the jth node of the ith layer to the kth node of the (i+1)th layer**/
std::vector<std::vector<std::vector<double> > > _weight;

//Weights updates
/** _Dweight[i][j][k] is the update of weight _weight[i][j][ĸ] **/
std::vector<std::vector<std::vector<double> > > _Dweight;
std::vector<std::vector<std::vector<double> > > _cumulDweight;

//Network parameters
unsigned int _epoch;        //number of passes over training set (cf. p251)
unsigned int _maxEpochs;
double		 _learningRate;
float		 _accuracyRequired;
bool		 _batchLearning;
bool		 _incrementalLearning;

//Logs
std::ofstream _trainingLog;
std::ofstream _generalizationLog;
std::ofstream _validationLog;
std::ofstream _resultLog;
bool		  _enableTrainingLog;
bool		  _enableGeneralizationLog;
bool		  _enableValidationLog;

//precision measures
double _validationAccuracy;
double _validationError;
double _generalizationAccuracy;
double _generalizationError;

//sets
std::vector<NNEntry*> _generalizationSet;
std::vector<NNEntry*> _trainingSet;

//destructor functor
struct DeleteFunctor
{
	template <class T> void operator()(T* ptr) const
	{
		delete ptr;
		ptr = 0;
	}
};





//public methods
//----------------------------------------------------------------------

public:


/**
 * Constructor that sets the structure of the network.
 *
 * Initializes Network parameters to default values:
 * _maxEpochs = 1
 * _learningRate = 0.0001
 * _accuracyRequired = 0.90
 *
 * @param array of number of nodes per layer.
 * Do not include bias node in the count of nodes per layer.
 * Number of layers is given by size of input vector (HAS TO BE >= 2)
 *
 */
NeuralNetwork(const std::vector< unsigned int > nbNodesPerLayer);

/**
 * Default Destructor
 *
 * Closes the network's logs.
 *
 */

~NeuralNetwork(void);

/**
 * Load default parameter
 *
 */

void LoadDefault(void);

/**
 * Enable batch learning: each weight change is accumulated over all patterns
 * and the weights are updated once after the complete pass over the whole training set.
 *
 */

void SetBatchLearning(void);

/**
 * Enable incremental learning: weights are updated in the same fashion as
 * a gradient descent
 *
 */

void SetIncrementalLearning(void);

/**
 * Enable a log to keep relevant training data
 *
 * @param relative path of the training log
 *
 */

void SetTrainingLog(const std::string filename);

/**
 * Enable a log to keep relevant generalization data
 *
 * @param relative path of the generalization log
 *
 */

void SetGeneralizationLog(const std::string filename);

/**
 * Enable a log to keep relevant validation data
 *
 * @param relative path of the validation log
 *
 */

void SetValidationLog(const std::string filename);

/**
 * Set learning rate parameter
 *
 * @param learning rate
 *
 */

void SetLearningRate(const double lr);

/**
 * Set maximum number of epochs to be done in case desired accuracy is not reached
 *
 * @param maximum number of epoch
 *
 */

void SetMaxEpochs(const int max);

/**
 * Set average accuracy training agent should obtain before stopping
 *
 * @param desired accuracy
 *
 */

void SetDesiredAccuracy(const float a);

/**
 * Set list of entries used to train the network's weights
 *
 * NNEntry is an interface class: make object you want to learn from derive from NNEntry
 *
 * @param training set
 *
 */

void LoadTrainingSet(const std::vector<NNEntry*> ts);

/**
 * Set list of entries used to train the network's accuracy
 *
 * NNEntry is an interface class: make object you want to learn from derive from NNEntry
 *
 * @param generalization set
 *
 */

void LoadGeneralizationSet(const std::vector<NNEntry*> gs);

/**
 * Train Network until either desired accuracy or maximum number of epoch is reached
 * The training set provided is used to update the net's values
 * while the generalization set is used to evaluate the performances.
 *
 */

void TrainNetwork(void);

/**
 * Display training results on console and write them down in the result log
 *
 */

void OutputTrainingResult(void);

/**
 * Updates the value of each Node based on the weights and the previous layer.
 *
 * @param values to give to the input layer
 *
 */

void FeedForward(const std::vector<double>&);

/**
 * Prints node values on given stream
 *
 * @param mode verbose: if true, prints inputs, hidden nodes and outputs.
 * if false, prints only inputs and outputs. Default is false
 * @param output stream. Default is cout
 *
 */

void PrintInfo(bool v = false, std::ostream& out = std::cout);



//private methods
//----------------------------------------------------------------------

private:

/**
 * Calculates the error between the expected value (supervised learning)
 * and the actual node value
 *
 * @param expected value of the output layer for the element that has just been feeded forward
 *  and is currently being backpropagated.
 *
 */

void Backpropagate(std::vector<double>);

/**
 * Updates the weights based of the error given by the backpropagation
 *
 */

void UpdateWeights();

/**
 * Activation Function
 *
 * Useful examples are
 *  - sigmoid 1/(1+exp(-a*x))
 *  - tanh(a*x)
 *  - linear a*x
 *
 * @param value
 * @param function parameter (default= 1.0)
 *
 */

inline double ActivationFunction(double x, double a = 1.0);

/**
 * Differential of the activation Function
 *
 * Useful examples are
 *  - sigmoid -> a*exp(a*x)/pow( 1+exp(a*x), 2)
 *  - tanh -> a*pow(sech(a*x),2)
 *  - linear -> cst a
 *
 * @param value
 * @param function parameter (default= 1.0)
 *
 */

inline double DifferentialActivationFunc(double x, double a = 1.0);

/**
 * Inverse of the activation Function
 *
 * Useful examples are
 *  - sigmoid -> logit log2( x/(1-x) )/a
 *  - tanh -> sum( 1/n*((a*x)^n); n=1,3,5,7,9...)
 *  - linear -> x/a
 *
 * @param value
 * @param function parameter (default= 1.0)
 *
 */

inline double InverseActivationFunc(double x, double a = 1.0);

/**
 * Trains the whole training set once
 * The process is composed of two phases:
 * - Feedforward
 * - Backpropagate
 *
 */

void RunSingleTraining(std::vector<NNEntry*>*);

/**
 * Process the average accuracy of the given set of entries
 *
 * @param set of entries
 *
 */

double GetAverageAccuracy(std::vector<NNEntry*>*);

/**
 * Calculate the accuracy of a unique entry
 *
 * @param entry
 *
 */

double GetAccuracy(NNEntry*);

/**
 * Process the mean square error of the given set of entries
 *
 * @param set of entries
 *
 */

double GetMSE(std::vector<NNEntry*>*);
};  //end class NeuralNetwork
} //end namespace

#endif
