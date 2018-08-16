#include <array>
#include <forward_list>
#include <algorithm>

using namespace std;

class cell {
	double tstop; // Time that the cell lifespan ends
	cell *parent; // Pointer to the cell's parent
	std::array<cell *, 2> children; // Pointer to the cells children

public:
	explicit cell(double tstartIn, cell *parentIn) : tstart(tstartIn), parent(parentIn), latent(0) {
		tstop = std::numeric_limits<double>::quiet_NaN();
	}

	std::array<cell *, 2> setDivide(double tstopIn) {
		// Check that tstop isn't already set
		// Check that children isn't already set
		// 
		

		tstop = tstopIn;

		return children;
	}

	void setDead(double tstopIn) {
		tstop = tstopIn;
	}

	double getTstop() {
		return tstop;
	}

	std::uint8_t latent; // Latent variable if we are clustering the cells
	const double tstart; // Time for start of cell lifespan
};

class tree {
	std::forward_list<cell> tree;

public:
	void addUnrooted(double tstart) {
		tree.emplace_front(tstart, nullptr);
	}

	/**
	 * @brief      Get the number of cells at time T.
	 *
	 * @param[in]  t     The time point of interest.
	 *
	 * @return     The number of cells found.
	 */
	unsigned int cellsAtT(double t) {
		// Provide the cell count at time T.
		return std::count_if(tree.begin(), tree.end(), [&t](cell &x) {return x.tstart > t && t < x.getTstop();});
	}

	size_t findUnfinished(double t) {
		// Find a cell that is born before t but does not yet have an outcome.

		return 0;
	}
};
    