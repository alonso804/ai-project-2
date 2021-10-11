#ifndef DecisionTree_H
#define DecisionTree_H

#include "CSVReader.h"
#include "Node.h"

struct best_split{
  int feature_index;
  float value;
  vector<vector<float>> data_left;
  vector<vector<float>> data_right;
  float gini_gain;

  void set(int feature_index,float value, vector<vector<float>> data_left, 
  vector<vector<float>> data_right, float gini_gain){
    this->feature_index = feature_index;
    this->value = value;
    this->data_left = data_left;
    this->data_right = data_right;
    this->gini_gain = gini_gain;
  }
};


class DecisionTree {
  int min_samples_split;
  int max_depth;  
  vector<vector<float>> data;
  int n_features;
  int n_rows;
  Node *root;

public:
  DecisionTree(string fileName,int min_samples_split, int max_depth) {
	  this->data = CSVReader::read(fileName);
    this->min_samples_split = min_samples_split;
    n_rows = 100;
    n_features = this->data[0].size() - 1;
    this->max_depth = max_depth;
    this->root = build_tree(this->data);
  }

	void split(int index,float value, vector<vector<float>> &dataset, vector<vector<float>> &left, vector<vector<float>> &right) {
    for (auto it = begin(dataset);it != end(dataset);it++){
      if ((*it)[index] <= value){
        left.push_back(*it);
      }
      else {
        right.push_back(*it);
      }
    }
  }


  vector<float> set_Y(vector<vector<float>> &data){
    vector<float> Y;
    for (auto it = begin(data); it!= end(data);it++)
      Y.push_back((*it)[(*it).size()]);

    return Y;
  }

  vector<vector<float>> set_X(vector<vector<float>> data){
    vector<vector<float>> X;
    vector<float> helper;

    for (auto it = begin(data); it!= end(data);it++){
        helper = *it;
        helper.pop_back();
        X.push_back(helper);
    }

    return X;
  }

  Node *build_tree(vector<vector<float>> data, int curr_depth = 0) {
    //auto outputsBefore = (this->root->data)[(this->root->data).size() - 1];
    // cout << outputsBefore << endl;

    auto Y = set_Y(data);
    auto X = set_X(data);
    int num_samples = Y.size();
    int num_features = X[0].size();

    if (num_samples >= this->min_samples_split && curr_depth <= this->max_depth){
      //find the best split
      auto best_split = get_best_split(data,num_samples,num_features);

      if (best_split.gini_gain > 0){
        auto left_subtree = build_tree(best_split.data_left,curr_depth+1);
        auto right_subtree = build_tree(best_split.data_right,curr_depth+1);

        auto node = new Node(best_split.feature_index,best_split.value,
        left_subtree,right_subtree,best_split.gini_gain);
        return node;

      }
    }
    float leaf_value = calculate_leaf_value(Y);
    auto node = new Node(leaf_value);
    return node;

  }


  vector<float> get_column(vector<vector<float>> &data, int index){
      vector<float> column;

      for(auto it = begin(data); it!= end(data); it++){
          column.push_back((*it)[index]);
      }

      return column;
  }

  best_split get_best_split(vector<vector<float>> &data,int num_samples,int num_features){
    float max_gain = INT_MIN;
    best_split best;

    //recorremos todas las columnas
    for (int i = 0; i<num_features; i++){
      auto feature_values = get_column(data,i);

      //recorremos todas las filas en una columna
      set<float> s( feature_values.begin(), feature_values.end());
      vector<float> unique_values;
      unique_values.assign(s.begin(),s.end());

      for (auto it = begin(unique_values); it != end(unique_values); it++){
        //spliteamos por cada valor 
        vector<vector<float>> data_right, data_left;
        split(i,*it,data,data_left,data_right);

        if (data_left.size() > 0 && data_right.size() > 0){
          vector<float> Y =  set_Y(data);
          vector<float> left_Y = get_column(data_left,num_features);
          vector<float> right_Y = get_column(data_right,num_features);
          float curr_gain = gini_gain(Y, left_Y,right_Y);

          if (curr_gain > max_gain){
            best.set(i,*it,data_left,data_left,curr_gain);
            max_gain = curr_gain;
          }
        }
      }
    }
    return best;
  }

  float gini_gain(vector<float> &Y,vector<float> &left_Y,vector<float> &right_Y){
    float weight_l = left_Y.size() / Y.size();
    float weight_r = right_Y.size() / Y.size();
    return  gini_impurity(Y) - (weight_l*gini_impurity(left_Y) +weight_r*gini_impurity(right_Y)); 
  }

  float gini_impurity(vector<float> &Y){
    float gini = 0;
    set<float> s( Y.begin(), Y.end());
    vector<float> uniques;
    uniques.assign(s.begin(),s.end());
    for (auto it = begin(uniques); it!= end(uniques);it++){
        auto per_class = count(Y.begin(), Y.end(), *it)/Y.size();
        gini+= per_class*per_class;
    }
    return 1 - gini;
  }

  float calculate_leaf_value(vector<float> Y){
    sort(Y.begin(),Y.end());
    float max_value_repeated;
    int max_times_repeated = 0;
    int current = 0;
    for (auto it = begin(Y); it!= end(Y)-1; it++){
      if (*it == *(it+1)){
        current++;
        if (current > max_times_repeated){
          max_times_repeated = current;
          max_value_repeated = *it;
        }
      }
    }
    return max_value_repeated;
  }

  void traverse_tree(Node *node, int indent = 1){
    if (node){
      node->print();
      for (int i = 0;i<indent;i++)
        cout<<"\t";
      cout<<"left:";
      traverse_tree(node->left,indent+indent);
      for (int i = 0;i<indent;i++)
        cout<<"\t";
      cout<<"right:";
      traverse_tree(node->right,indent+indent);
    }
  }

  void print(){
    traverse_tree(this->root);
  }
  ~DecisionTree() {}
};

#endif //DecisionTree_H
