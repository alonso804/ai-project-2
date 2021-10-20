#include "Iris.h"
#include "DecisionTree.h"
#include "CSVReader.h"

int main(int argc, char *argv[]) {
	/*
	auto features = CSVReader<Iris>::read("iris.csv");

	for (auto f : features) {
		cout << f << endl;
	}
	*/

	auto decisionTree = DecisionTree("gender_classification.csv",3,5);
	//decisionTree.print();
	auto data = CSVReader::read("gender_classification.csv");
	auto test =  decisionTree.set_X(data);
	auto res =  decisionTree.set_Y(data);
	auto predict = decisionTree.predict(test);

	int countError = 0;
	for (int i = 0; i < test.size(); i++) {
		if (predict[i] != res[i]) {
			countError += 1;
			//cout << "Error" << endl;
		}
	}
	cout << "Hay " << countError << " errores" << endl;
	cout <<"Accuracy: " << 100 - (float)(countError*100) / res.size() << endl;
	decisionTree.generatePDF();
	return 0;
}
