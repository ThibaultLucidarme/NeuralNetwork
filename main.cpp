#include "NeuralNetwork.hpp"
#include <ctime>

#define NB_SAMPLE_TRAINING 4
#define NB_SAMPLE_GENE 50


template<class T>
void ClearPool(std::vector<T*>& a)
{
    for(size_t i = 0; i < a.size(); i++)
    {
    	delete a[i];
    	a[i] = 0;
	}

    a.clear();
}

/**
	Quick example learning the XOR function
	results are presented in the log files:
	-training.log presents the content of the NN during the learning process
	-Result.log presents the overall performance of the NN
	-generalization.log presents the evolution of the accuracy per epoch for each element of the generalization set
	valgrin.txt represents the output of a valgrind run (debug purposes)
	
	No cross validation is performed, 
	no test for overfitting is performed (none needed when learning xor),
	no adaptive learning rate
	no boosting is performed
**/

int main(int argc, char**argv)
{
	//create data (XOR)
	
	std::vector<double> v11,v10,v01,v00;
	v11.push_back(1.0);
	v11.push_back(1.0);
	v10.push_back(0.0);
	v10.push_back(1.0);
	v01.push_back(1.0);
	v01.push_back(0.0);
	v00.push_back(0.0);
	v00.push_back(0.0);
	
	//create data expected values
	
	std::vector<double> tv11,tv10,tv01,tv00;
	tv11.push_back(0.0);
	tv10.push_back(1.0);
	tv01.push_back(1.0);
	tv00.push_back(0.0);
	
	
	//create training set
	/*
	srand(time(0));
	
	std::vector<p::NNEntry*> training_data;
	training_data.clear();
	for (int i=0; i<NB_SAMPLE_TRAINING; i++)
	{
		int tmp = floor((rand() % 400)/100);
		switch(tmp)
		{
			case 0: training_data.push_back( new p::NNEntry( v11, tv11) ); break;
			case 1: training_data.push_back( new p::NNEntry( v01, tv01) ); break;
			case 2: training_data.push_back( new p::NNEntry( v10, tv10) ); break;
			case 3: training_data.push_back( new p::NNEntry( v00, tv00) ); break;
		}
		
	}
	*/	
	std::vector<p::NNEntry*> training_data;
	training_data.clear();
	training_data.push_back( new p::NNEntry( v11, tv11) ); 
		training_data.push_back( new p::NNEntry( v01, tv01) );
		training_data.push_back( new p::NNEntry( v10, tv10) ); 
		 training_data.push_back( new p::NNEntry( v00, tv00) ); 
	
	
	//create generalization set
	
	std::vector<p::NNEntry*> generalization_data;
	generalization_data.clear();
	for (int i=0; i<NB_SAMPLE_GENE; i++)
	{
		int tmp = floor((rand() % 400)/100);
		switch(tmp)
		{
			case 0: generalization_data.push_back( new p::NNEntry( v11, tv11) ); break;
			case 1: generalization_data.push_back( new p::NNEntry( v01, tv01) ); break;
			case 2: generalization_data.push_back( new p::NNEntry( v10, tv10) ); break;
			case 3: generalization_data.push_back( new p::NNEntry( v00, tv00) ); break;
		}
		
	}


	
	//create neural network
	
	std::vector<unsigned int> par;
	par.push_back(2);//2 inputs
	par.push_back(3);//1 hidden layer of 5 nodes
	par.push_back(1);//1 output
	p::NeuralNetwork nn(par);
	nn.LoadDefault();
	nn.LoadTrainingSet( training_data );
	nn.LoadGeneralizationSet( generalization_data );
	nn.SetBatchLearning();
	nn.SetMaxEpochs(1000);
	nn.SetLearningRate(0.01);
	
	nn.TrainNetwork();
	
	ClearPool<p::NNEntry>( training_data );
	ClearPool<p::NNEntry>( generalization_data );
	
	return EXIT_SUCCESS;	
}

