#include <array>
#include <vector>
#include <cmath>
#include <memory>
#include <random>
#include <algorithm>

using namespace std;

typedef std::bernoulli_distribution bernT;
typedef std::weibull_distribution<float> weibT;

class cellOutcome {
	std::mt19937 gen;
	std::unique_ptr<bernT> bern;
	std::unique_ptr<weibT> weib;

public:
	cellOutcome(double bernVal, double wA, double wB) {
		std::random_device rd;
		gen = std::mt19937(rd());
		bern = std::make_unique<bernT>(bernVal);
		weib = std::make_unique<weibT>(wA, wB);
	}

	std::pair<bool, float> genVal() {
		return std::make_pair((*bern)(gen), (*weib)(gen));
	}
};


class cell {
	float tstop; // Time that the cell lifespan ends
	size_t parent; // Pointer to the cell's parent
	std::array<size_t, 2> children; // Pointer to the cell's children

public:
	explicit cell(float tstartIn, size_t parentIn) : tstart(tstartIn), parent(parentIn), latent(0) {
		tstop = std::numeric_limits<float>::quiet_NaN();
	}

	void setDivided(float tstopIn, std::array<size_t, 2> childrenIn) {
		// Check that tstop isn't already set
		// Check that children isn't already set
		// 

		tstop = tstopIn;
		children = childrenIn;
	}

	void setDead(float tstopIn) {
		tstop = tstopIn;
	}

	inline float getTstop() {
		return tstop;
	}

	std::uint8_t latent; // Latent variable if we are clustering the cells
	const float tstart; // Time for start of cell lifespan
};

class tree {
	std::vector<cell> tree;

public:
	void addUnrooted(float tstart) {
		tree.emplace_back(tstart, tree.size()-1);
	}

	void setDivide(size_t idx, float tdivide) {
		if (!isnan(tree[idx].getTstop()))
			throw runtime_error("Set cell to divide when it already had an end event.");

		tree.emplace_back(tdivide, idx);
		tree.emplace_back(tdivide, idx);

		tree[idx].setDivided(tdivide, {{tree.size()-1, tree.size()-2}});
	}

	void setDead(size_t idx, float tdead) {
		if (!isnan(tree[idx].getTstop()))
			throw runtime_error("Set cell to divide when it already had an end event.");

		tree[idx].setDead(tdead);
	}

	/**
	 * @brief      Get the number of cells at time T.
	 *
	 * @param[in]  t     The time point of interest.
	 *
	 * @return     The number of cells found.
	 */
	unsigned int cellsAtT(float t) {
		return std::count_if(tree.begin(), tree.end(), [&t](cell &x) {return x.tstart > t && t < x.getTstop();});
	}

	/**
	 * @brief      Find a cell that is born before t but does not yet have an outcome.
	 *
	 * @param[in]  t     The time point of interest.
	 *
	 * @return     Iterator to the cell found. Will point to end if none were found.
	 */
	std::vector<cell>::iterator findUnfinished(float t) {
		return std::find_if(tree.begin(), tree.end(), [&t](cell &x) {return x.tstart < t && isnan(x.getTstop());});
	}
};
    
