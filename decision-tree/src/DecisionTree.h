#ifndef DecisionTree_H
#define DecisionTree_H

#include "CSVReader.h"
#include "Node.h"

template <typename T>
class DecisionTree {
  shared_ptr<Node<T>> root;

public:
  DecisionTree(string fileName) {
	  auto data = CSVReader<T>::read(fileName);
    this->root = make_shared<Node<T>>(data);
    this->treeCreate();
  }

	void split(int index,float value, vector<T> dataset, vector<T> &left, vector<T> &right) {
    for (auto it = begin(dataset);it != end(dataset);it++){
      if (*it[index] < value){
        left.push_back(*it);
      }
      else {
        right.push_back(*it);
      }
    }
  }

  void try_splits(){
    for (auto it = begin(dataset);it != end(dataset);it++){
      vector<T> left;
      vector<T> right;
    }
  }

  void treeCreate() {
    auto outputsBefore = (this->root->data)[(this->root->data).size() - 1];
    // cout << outputsBefore << endl;
  }

  ~DecisionTree() {}
};

#endif //DecisionTree_H
