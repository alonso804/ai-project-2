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
  Node *root = nullptr;
  int idx;
  int leafIdx;

public:
  DecisionTree(string fileName,int min_samples_split, int max_depth) {
    this->idx = 0;
    this->leafIdx = -1;
	  this->data = CSVReader::read(fileName);
    this->min_samples_split = min_samples_split;
    n_features = this->data[0].size() - 1;
    this->max_depth = max_depth;
    shuffle(this->data);
    this->root = build_tree(trainingData(this->data));

  }

    #include <algorithm>    // std::shuffle
#include <array>        // std::array
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock

  void shuffle(vector<vector<float>> &data){
      unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
      std::shuffle(data.begin(), data.end(), std::default_random_engine(seed));
    }

vector<vector<float>> trainingData(vector<vector<float>> &data){
    vector<vector<float>> result;

    for(int i=0; i <= 0.08*data.size(); i++){
      result.push_back(this->data[i]);
    }

    return result;
  }

  vector<vector<float>> testingData(){
    vector<vector<float>> result;

    for(int i=0.08*this->data.size(); i < this->data.size(); i++){
      result.push_back(this->data[i]);
    }

    return result;
  }

  vector<float> KFoldError(int folds){
    vector<float> error;
    for(int i=0; i<folds; i++){
      vector<vector<float>> result;
      int ind = this->data.size()/folds;
      for(int j= i*(ind); j < ind + (ind * i); j++){
        result.push_back(this->data[j]);
      }

      auto pred = predict(result);
      auto real_results = getRealResults(result);

      error.push_back(calcError(pred, real_results));
    }

    return error;

  }

  float calcError(vector<float> a, vector<float> b){
	int countError = 0;
	for (int i = 0; i < a.size(); i++) {
		if (a[i] != b[i]) {
			countError += 1;
		}
	}

	return (float) countError*100 / b.size();
}

  vector<float> getRealResults(vector<vector<float>> &data){
    vector<float> Y;
    for (auto it = begin(data); it!= end(data);it++)
      Y.push_back((*it)[(*it).size()-1]);

    return Y;
  }

  vector<float> set_Y(vector<vector<float>> &data){
    vector<float> Y;
    for (auto it = begin(data); it!= end(data);it++)
      Y.push_back((*it)[(*it).size()-1]);

    return Y;
  }

  vector<vector<float>> set_X(vector<vector<float>> &data){
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
    //cout<<Y<<endl;
    int num_samples = Y.size();
    int num_features = X[0].size();

    if (num_samples >= this->min_samples_split && curr_depth <= this->max_depth){
      //find the best split
      auto best_split = get_best_split(data,num_samples,num_features);

      if (best_split.gini_gain > 0){
        auto left_subtree = build_tree(best_split.data_left,curr_depth+1);
        auto right_subtree = build_tree(best_split.data_right,curr_depth+1);

        auto node = new Node(best_split.feature_index,best_split.value,
        left_subtree,right_subtree,best_split.gini_gain, idx++);
        return node;

      }
    }
    float leaf_value = calculate_leaf_value(Y);
    auto node = new Node(leaf_value, leafIdx--);
    return node;

  }


  vector<float> get_column(vector<vector<float>> &data, int index){
      vector<float> column;

      for(auto it = begin(data); it!= end(data); it++){
          column.push_back((*it)[index]);
      }

      return column;
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
            best.set(i,*it,data_left,data_right,curr_gain);
            max_gain = curr_gain;
          }
        }
      }
    }
    return best;
  }

  float gini_gain(vector<float> &Y,vector<float> &left_Y,vector<float> &right_Y){
    float weight_l = float(left_Y.size()) / Y.size();
    float weight_r = float(right_Y.size()) / Y.size();
    return gini_impurity(Y) - (weight_l*gini_impurity(left_Y) +weight_r*gini_impurity(right_Y));
  }

  float gini_impurity(vector<float> &Y){
    float gini = 0;
    set<float> s( Y.begin(), Y.end());
    vector<float> uniques;
    uniques.assign(s.begin(),s.end());
    for (auto it = begin(uniques); it!= end(uniques);it++){
        auto per_class = float(count(Y.begin(), Y.end(), *it))/Y.size();
        gini+= per_class*per_class;
    }
    return 1 - gini;
  }

    float calculate_leaf_value(vector<float> Y){
        sort(Y.begin(),Y.end());
        float max_val = Y[0];
        map<float, int> frec;

        for(auto it : Y){
            frec[it]++;
        }
        for(int i = 0; i<Y.size() ; i++){
            if(frec[Y[i]] < frec[Y[i+1]])
                max_val = Y[i+1];
        }
        return max_val;
    }

  void traverse_tree(Node *node, int cont = 0){
      if (node == nullptr) return;
      else {
          traverse_tree(node->right, cont+1);
          for (int i = 0; i < cont; i++) {
              cout << "|" << "                       ";
          }
      }
      node->print();
      cout << "--|" << endl;
      traverse_tree(node->left, cont + 1);
  }

  vector<float> predict(vector<vector<float>> &X){
    vector<float> predictions;
    for (auto &it:X){
      predictions.push_back(make_prediction(it,this->root));
    }
    return predictions;
  }

  float make_prediction(vector<float> &row, Node * node){
      if (node->value ==1 || node->value == -1){
          return node->value;
      }
      float feature_value = row[node->feature_index];
      if (feature_value<= node->value) return make_prediction(row,node->left);
      else return make_prediction(row,node->right);
  }

  void print(){
      cout << "∧: No" << endl;
      cout << "∨: Yes" << endl;
      traverse_tree(this->root);
  }

	void printNodesConnections(fstream &file, Node*& node) {
    if (node) {
        if(node->left){
          file << "\"" << node->idx << "\"->";
          file << "\"" << node->left->idx << "\";\n";
          printNodesConnections(file, node->left);
        }

        if(node->right) {
          file << "\"" << node->idx << "\"->";
          file << "\"" << node->right->idx << "\";\n";
          printNodesConnections(file, node->right);
        }
    }
	}

	void printAllNodes(fstream &file, Node*& node){
    if (node) {
      if (node->idx >= 0) {
        file << "\"" << node->idx << "\" [\n";
        file << "\tlabel = \"" << "X_"<< node->feature_index+1 <<"<= "<<node->value  << " \"\n]\n";
      } else {
        file << "\"" << node->idx << "\" [\n";
        if (node->value == 1) {
          file << "\tlabel = \"" << "Male" << " \"\n]\n";
        } else if (node->value == -1) {
          file << "\tlabel = \"" << "Female" << " \"\n]\n";
        }
      }

        if(node->left) {
            printAllNodes(file, node->left);
        }

        if(node->right) {
            printAllNodes(file, node->right);
        }
    }
	}

  void generatePDF() {
      fstream file("graph.vz", fstream::out | fstream::trunc);
      if (file.is_open()) {
          file << "digraph G\n";
          file << "{\n";
          printAllNodes(file, root);
          printNodesConnections(file, root);
          file << "}\n";
          file.close();
          system("dot -Tpdf graph.vz -o graph.pdf");
      }
  }

  ~DecisionTree() {}
};

#endif //DecisionTree_H
