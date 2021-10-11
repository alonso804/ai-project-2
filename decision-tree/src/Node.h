#ifndef Node_H
#define Node_H

#include <memory>

using namespace std;

struct Node{
  //decision Node
  int feature_index;
  Node* left;
  Node* right;
  float gini_gain;

  //leaf Node
  float value;

  Node(int feature_index,float value, Node *left, Node *right, float gini_gain){
    this->feature_index = feature_index;
    this-> value = value;
    this->left = left;
    this->right = right;
    this-> gini_gain;
  }

  Node(float value){
    this->value = value;
  }

};

#endif //Node_H