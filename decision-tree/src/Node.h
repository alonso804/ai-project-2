#ifndef Node_H
#define Node_H

#include <memory>

using namespace std;

struct Node{
  //decision Node
  int feature_index;
  Node* left = nullptr;
  Node* right = nullptr;
  float gini_gain;
  int idx;

  //leaf Node
  float value;

  Node(int feature_index,float value, Node *left, Node *right, float gini_gain, int idx){
    this->feature_index = feature_index;
    this-> value = value;
    this->left = left;
    this->right = right;
    this-> gini_gain = gini_gain;
    this->idx = idx;
  }

  Node(float value, int leafIdx){
    this->value = value;
    this->idx = leafIdx;
  }

  void print(){
    if (!this->right && !this->left){
        if(this->value == 1) cout << "Male";
        if(this->value == -1) cout << "Female";
    }
    else{
      cout << "X_"<<this-> feature_index+1<<" <= "<<value<<", gini: "<<this->gini_gain;
    }
  }

};



#endif //Node_H
