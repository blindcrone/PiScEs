#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <concepts>
#include <ranges>
#include <cmath>
#include <functional>
#include <execution>
#include <vector>
#include <iostream>
#include <cmath>
namespace py = pybind11;
namespace rv = std::ranges::views;
namespace ex = std::execution;

//A little variadic sum helper function for distance calculations
constexpr auto sum(const auto n, const auto... m){
	if constexpr (sizeof...(m) == 0)
		return n;
	else
		return n + sum(m...);
}

//A little generic square function. I guess I could just use pow(x,2) but this often optimizes better
constexpr auto sq(const auto n){ return n * n; }

typedef py::ssize_t idx;

//Standard euclidean distance measure in color space, adapted for numpy array access
constexpr auto dcolor(const auto& r, const idx x1, const idx y1, const idx x2, const idx y2){
	return sqrt( sum( sq(r(x1, y1, 0) - r(x2, y2, 0))
					, sq(r(x1, y1, 1) - r(x2, y2, 1))
					, sq(r(x1, y1, 2) - r(x2, y2, 2))));
}

//An alternate distance measure that's just sum-of-absolute-distances, turns out not to be faster or better
constexpr auto colord(const auto& r, const idx x1, const idx y1, const idx x2, const idx y2){
	using std::abs;
	return sum( abs(r(x1, y1, 0) - r(x2, y2, 0))
			  , abs(r(x1, y1, 1) - r(x2, y2, 1))
			  , abs(r(x1, y1, 2) - r(x2, y2, 2)));
}

/* The overall estimator draws a diagonal line through the image and finds contiguous blocks of color
 * Parameters:
 * 1. Minimum downscaling: PiScEs will return a value no lower than this
 * 2. Color threshold: Essentially the distance between colors that counts as "the same" for contiguity
 * 3. Frequency Cutoff: The ratio of the sorted contiguity metric that is used as the scale
 *                      (E.G. 2 would be the median, 4 would be the bottom of the top quartile, etc)
 */
auto PiScEs(const idx minSize, const int colorThreshold, const idx freqCutoff){
	using std::transform_reduce;
	using std::max;
	using std::min;
	return std::function([=](const py::array_t<uint8_t>& img){
		auto r = img.unchecked<3>();
		auto xdim = r.shape(0)
		   , ydim = r.shape(1);
		std::vector<idx> xst;
		//Contiguous color streak finder based on specified threshold
		for(idx n = 1; n < xdim; ++n){
			idx last = n;
			for(; n < min(ydim,xdim) && dcolor(r, n - 1, n - 1, n, n) < colorThreshold; ++n);
			if(auto streak = n - last + 1; streak > 1)
				xst.emplace_back(streak);
		}
		//Sort the streaks, smallest to biggest
		std::sort(xst.begin(), xst.end());
		return max(minSize, xst[xst.size() / freqCutoff - 1]);
	});
}
