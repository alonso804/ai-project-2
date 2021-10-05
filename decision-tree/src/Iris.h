#ifndef Iris_H
#define Iris_H

#include "Headers.h"

struct Iris {
	double A;
	double B;
	double C;
	double D;
	int CLASS;

	Iris() {}

	Iris(string A, string B, string C, string D, string CLASS) {
		this->A = stof(A);
		this->B = stof(B);
		this->C = stof(C);
		this->D = stof(D);

		if (CLASS == "Iris-setosa") {
			this->CLASS = 1;
		} else {
			this->CLASS = -1;
		}
	}

	Iris(vector<string> data) {
		this->A = stof(data[0]);
		this->B = stof(data[1]);
		this->C = stof(data[2]);
		this->D = stof(data[3]);

		if (data[4] == "Iris-setosa") {
			this->CLASS = 1;
		} else {
			this->CLASS = -1;
		}
	}

	friend ostream& operator<<(ostream& os, const Iris& iris);
};

ostream& operator<<(ostream& os, const Iris& iris) {
		os << iris.A << " " << iris.B << " " << iris.C << " " << iris.D << " " << iris.CLASS;

		return os;
}

#endif //Iris_H
