#ifndef CSVReader_H
#define CSVReader_H

#include "Headers.h"

template <typename Dataset>
class CSVReader {
public:
	static vector<Dataset> read(string fileName, char delimiter = ',') {
		fstream file;
		string line;
		vector<Dataset> dataset;

		file.open(fileName, ios::in);

		if (file.is_open()) {
			getline(file, line, '\n');

			while (getline(file, line, '\n')) {
				vector<string> row;
				stringstream s(line);
				string word;
 
 				while (getline(s, word, delimiter)) {
					row.push_back(word);
				}
 
				dataset.push_back(Dataset(row));
			}

			file.close();
		} else {
			cerr << "Can't open " << fileName << endl;
		}

		return dataset;
	}
};

#endif //CSVReader_H
