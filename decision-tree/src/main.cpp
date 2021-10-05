#include "Iris.h"
#include "CSVReader.h"

int main(int argc, char *argv[]) {
	auto features = CSVReader<Iris>::read("iris.csv");

	for (auto f : features) {
		cout << f << endl;
	}
	
	return 0;
}
