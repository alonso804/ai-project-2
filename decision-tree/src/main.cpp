#include "Iris.h"
#include "DecisionTree.h"
#include "CSVReader.h"
#include <fstream>

float accuracy(vector<float> a, vector<float> b){
	int countError = 0;
	for (int i = 0; i < a.size(); i++) {
		if (a[i] != b[i]) {
			countError += 1;
			//cout << "Error" << endl;
		}
	}

	return 100 - (float)(countError*100) / b.size();
}

int main(int argc, char *argv[]) {
	/*
	auto features = CSVReader<Iris>::read("iris.csv");

	for (auto f : features) {
		cout << f << endl;
	}
	*/

	auto decisionTree = DecisionTree("gender_classification.csv",3,5);
	//decisionTree.print();
	auto test =  decisionTree.testingData();
	auto predict = decisionTree.predict(test);
	auto real_results = decisionTree.getRealResults(test);

	auto errores = decisionTree.KFoldError(10);

	for(auto it:errores){
		cout << it << endl;
	}

	cout <<"Accuracy: " << accuracy(predict, real_results) << endl;
	//decisionTree.generatePDF();
	return 0;
}
