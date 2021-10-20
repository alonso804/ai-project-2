#ifndef CSVReader_H
#define CSVReader_H

#include "Headers.h"


class CSVReader {
public:
	static vector<vector<float>> read(string fileName, char delimiter = ',') {
		fstream file;
		string line;
		vector<vector<float>> dataset;

		file.open(fileName, ios::in);
//        getline(file, line, '\n');

		if (file.is_open()) {
			while (getline(file, line, '\n')) {
				vector<float> row;
				stringstream s(line);
				string word;
 
 				while (getline(s, word, delimiter)) {
					if (word.size() == 4 || word.size() == 5 || word.size() == 12){
						row.push_back(1.0);
					}
					else if (word.size() == 6 || word.size() == 7 || word.size() == 16){
						row.push_back(-1.0);
					} else {
						row.push_back(stof(word));
					}
				}
 
				dataset.push_back(row);
			}

			file.close();
		} else {
			cerr << "Can't open " << fileName << endl;
		}

		return dataset;
	}
};

#endif //CSVReader_H
