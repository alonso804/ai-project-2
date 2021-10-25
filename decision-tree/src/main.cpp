#include "DecisionTree.h"
#include "CSVReader.h"

int main(int argc, char *argv[]) {
	auto decisionTree = DecisionTree("gender_classification.csv", 3, 3);
	decisionTree.generatePDF();
	decisionTree.print();
	auto test =  decisionTree.testingData();
	auto predict = decisionTree.predict(test);

	auto real_results = decisionTree.getRealResults(test);

	int countError = 0;
	for (int i = 0; i < test.size(); i++) {
		if (predict[i] != real_results[i]) {
			countError += 1;
		}
	}
	cout << "Hay " << countError << " errores" << endl;
	cout <<"Accuracy: " << 100 - (float)(countError*100) / real_results.size() << endl;
	return 0;
}
