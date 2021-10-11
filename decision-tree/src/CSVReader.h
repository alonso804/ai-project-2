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
        getline(file, line, '\n');

		if (file.is_open()) {
			while (getline(file, line, '\n')) {
				vector<float> row;
				stringstream s(line);
				string word;
 
 				while (getline(s, word, delimiter)) {
					if (word.length() == 11){
						word = "1";
					}
					else if (word.length() == 15){
						word = "-1";
					}
					row.push_back(stof(word));
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
