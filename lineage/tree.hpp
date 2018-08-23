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
	std::mt19937 gen; ///< Random number generator
	std::unique_ptr<bernT> bern; ///< Bernoulli distribution for deciding cell outcome
	std::unique_ptr<weibT> weib; ///< Weibull distribution for determining time of cell event

public:
	cellOutcome(double bernVal, double wA, double wB) {
		std::random_device rd;
		gen = std::mt19937(rd());
		bern = std::make_unique<bernT>(bernVal);
		weib = std::make_unique<weibT>(wA, wB);
	}

	std::pair<bool, float> operator() (){
		return std::make_pair((*bern)(gen), (*weib)(gen));
	}
};


/**
 * @brief      Class that implements a cell.
 */
class cell {
	float tstop; ///< Time that the cell lifespan ends
	size_t parent; ///< Index of cell's parent
	std::array<size_t, 2> children; ///< Indexes of cell's children

public:
	explicit cell(float tstartIn, size_t parentIn) : parent(parentIn), tstart(tstartIn) {
		tstop = std::numeric_limits<float>::quiet_NaN();
	}

	bool isAlive(float t) {
		if (isnan(tstop))
			throw runtime_error("Can't ask if a cell is alive if it doesn't have an end event.");

		return tstart > t && t < tstop;
	}

	void setDivided(float tstopIn, std::array<size_t, 2> childrenIn) {
		if (!isnan(tstop))
			throw runtime_error("Set cell to divide when it already had an end event.");

		tstop = tstopIn;
		children = childrenIn;
	}

	void setDead(float tstopIn) {
		if (!isnan(tstop))
			throw runtime_error("Set cell to die when it already had an end event.");

		tstop = tstopIn;
	}

	size_t getParent() {
		return parent;
	}

	float getTstop() {
		if (isnan(tstop))
			throw runtime_error("Stop time still undefined.");

		return tstop;
	}

	bool isUnfinished() {
		return isnan(tstop);
	}
	
	const float tstart; ///< Time for start of cell lifespan
};

/**
 * @brief      Implementation of a cell with a latent variable.
 */
class cellLatent : public cell {
public:
	std::uint8_t latent; ///< Latent variable if we are clustering the cells
};


/**
 * @brief      Implementation of a lineage tree.
 * \todo May need to refactor to make this available for using a latent variable.
 */
class tree {
	std::vector<cell> tree;

public:
	void addUnrooted(float tstart) {
		tree.emplace_back(tstart, tree.size()-1);
	}

	size_t getNcells() {
		return tree.size();
	}

	void setDivide(size_t idx, float tdivide) {
		tree.emplace_back(tdivide, idx);
		tree.emplace_back(tdivide, idx);

		tree[idx].setDivided(tdivide, {{tree.size()-1, tree.size()-2}});
	}

	void setDead(size_t idx, float tdead) {
		tree[idx].setDead(tdead);
	}

	/**
	 * @brief      Determines an end event for the current cell.
	 *
	 * @param[in]  idx      The cell of interest.
	 * @param      outcome  The distribution of outcomes.
	 */
	void setEnd(size_t idx, cellOutcome &outcome) {
		if (!tree[idx].isUnfinished())
			throw runtime_error("Set cell event when it already had an end event.");

		std::pair<bool, float> outt = outcome();

		if (outt.first) {
			setDivide(idx, tree[idx].tstart + outt.second);
		} else {
			setDead(idx, tree[idx].tstart + outt.second);
		}
	}

	/**
	 * @brief      Make sure all cells have end events up to time t. (No latent states.)
	 *
	 * @param[in]  t        The time point of interest.
	 * @param      outcome  The distribution of outcomes.
	 */
	void fillToT(float t, cellOutcome &outcome) {
		std::vector<cell>::iterator it;

		it = findUnfinished(t);

		while (it != tree.end()) {
			setEnd(it - tree.begin(), outcome);

			it = findUnfinished(t);
		}
	}

	/**
	 * @brief      Get the number of cells at time T.
	 *
	 * @param[in]  t     The time point of interest.
	 *
	 * @return     The number of cells found.
	 */
	unsigned int cellsAtT(float t) {
		return std::count_if(tree.begin(), tree.end(), [&t](cell &x) {return x.isAlive(t);});
	}

	/**
	 * @brief      Find a cell that is born before t but does not yet have an outcome.
	 *
	 * @param[in]  t     The time point of interest.
	 *
	 * @return     Iterator to the cell found. Will point to end if none were found.
	 */
	std::vector<cell>::iterator findUnfinished(float t) {
		return std::find_if(tree.begin(), tree.end(), [&t](cell &x) {return x.tstart < t && x.isUnfinished();});
	}
};
    
