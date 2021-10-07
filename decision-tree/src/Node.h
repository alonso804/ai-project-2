#ifndef Node_H
#define Node_H

#include <memory>

using namespace std;

template <typename Dataset>
class DecisionTree;

template <typename Dataset>
class Node {
  vector<Dataset> data;
  shared_ptr<Node<Dataset>> left;
  shared_ptr<Node<Dataset>> right;

public:
  Node(vector<Dataset> data) {
    this->data = data;
  }

  template <class>
  friend class DecisionTree;
};

#endif //Node_H