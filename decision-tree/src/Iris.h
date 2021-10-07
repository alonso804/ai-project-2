#ifndef Iris_H
#define Iris_H

#include "Headers.h"

struct Iris {
	vector<double> features;
	int y;

	Iris() {}

	Iris(string A, string B, string C, string D, string CLASS) {
		this->features = {stof(A), stof(B), stof(C), stof(D)};

		if (CLASS == "Iris-setosa") {
			this->y = 1;
		} else {
			this->y = -1;
		}
	}

	Iris(vector<string> data) {
		this->features = {stof(data[0]), stof(data[1]), stof(data[2]), stof(data[3])};

		if (data[4] == "Iris-setosa") {
			this->y = 1;
		} else {
			this->y = -1;
		}
	}

	friend ostream& operator<<(ostream& os, const Iris& iris);
};

ostream& operator<<(ostream& os, const Iris& iris) {
	for (const auto& i : iris.features) {
		os << i << " ";
	}

	os << iris.y;

	return os;
}

#endif //Iris_H