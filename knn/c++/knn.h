#ifndef KNN_H
#define KNN_H

#include <iostream>
using std::cout;
using std::endl;

#include <vector>
using std::vector;

#include <utility>
using std::make_pair;
using std::pair;

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/index/rtree.hpp>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

template<typename Point, size_t Features = 7>
class KNN {
	using point = bg::model::point<double, Features, bg::cs::cartesian>;
	using box = bg::model::box<point>;
	using value = pair<point, unsigned>;

  bgi::rtree<value, bgi::quadratic<16>> rtree;

public:
	KNN();

	void insert(const Point& newPoint) {
		point boostPoint(newPoint.x[0], newPoint.x[1], newPoint.x[2], newPoint.x[3], newPoint.x[4], newPoint.x[5], newPoint.x[6]);
		rtree.insert(make_pair(boostPoint, newPoint.y));
	}

	vector<point> nearest_neighbor(const size_t& k, const Point& reference) {
		vector<point> resultN;
		rtree.query(bgi::nearest(point(reference.x[0], reference.x[1], reference.x[2], reference.x[3], reference.x[4], reference.x[5], reference.x[6]), k), back_inserter(resultN));

		value to_print_out;
    for (size_t i = 0; i < resultN.size(); i++) {
			to_print_out = resultN[i];
			float x = to_print_out.first.get<0>();
			float y = to_print_out.first.get<1>();
			cout << "Select point: " << to_print_out.second << endl;
			cout << "x: " << x << ", y: " << y << endl;
    }

	 return resultN;
	}
};

#endif //KNN_H
