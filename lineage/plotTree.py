""" In this file we plot! """

import numpy as np
from Bio.Phylo.BaseTree import Clade
import networkx as nx

cs = ['lightblue', 'orange', 'lightgreen', 'red', 'purple', 'olive', 'gray']
stateColors = ['blue', 'orange', 'green', 'red', 'purple', 'olive', 'gray']


def CladeRecursive(cell, a, censore, color):
    """ To plot the lineage while censored (from G1 or G2).
    If cell died in G1, the lifetime of the cell until dies is shown in red.
    If cell died in G2, the lifetime of the cell until dies is shown in blue.
    If none of the above, the cell continues to divide and is shown in black.
    a should be: a = [Clade(lineage1.full_lineage[0].obs[2]+lineage1.full_lineage[0].obs[3])]
    If you are interested, you can take a look at the source code for creating Clades manually:
    https://github.com/biopython/biopython/blob/fce4b11b4b8e414f1bf093a76e04a3260d782905/Bio/Phylo/BaseTree.py#L801
    """
    if color:
        if np.isfinite(cell.state):
            colorr = stateColors[cell.state]
        else:
            colorr = "black"
    else:
        colorr = "black"

    if cell.isLeaf() and censore:
        if np.isfinite(cell.obs[2]) and np.isfinite(cell.obs[3]):
            length = cell.obs[2] + cell.obs[3]
        elif np.isnan(cell.obs[2]):
            length = cell.obs[3]
        elif np.isnan(cell.obs[3]):
            length = cell.obs[2]
        my_clade = Clade(branch_length=length, width=1, color=colorr)
        my_clade.G1lw = 1.1
        my_clade.G2lw = 0.8
        my_clade.G1 = cell.obs[2] if np.isfinite(cell.obs[2]) else 1e-4
        my_clade.G2 = cell.obs[3] if np.isfinite(cell.obs[3]) else 1e-4
        return my_clade

    else:
        clades = []
        if cell.left is not None and cell.left.observed:
            clades.append(CladeRecursive(cell.left, a, censore, color))
        if cell.right is not None and cell.right.observed:
            clades.append(CladeRecursive(cell.right, a, censore, color))
        if np.isnan(cell.obs[3]):  # if the cell got stuck in G1
            lengths = cell.obs[2]
        elif np.isnan(cell.obs[2]):  # is a root parent and G1 is not observed
            lengths = cell.obs[3]
        else:
            lengths = cell.obs[2] + cell.obs[3]  # both are observed
        my_clade = Clade(branch_length=lengths, width=1, clades=clades, color=colorr)
        my_clade.G1lw = 1.2
        my_clade.G2lw = 0.5
        my_clade.G1 = cell.obs[2] if np.isfinite(cell.obs[2]) else 1e-4
        my_clade.G2 = cell.obs[3] if np.isfinite(cell.obs[3]) else 1e-4
        return my_clade


def plotLineage(lineage, axes, censore=True, color=True):
    """
    Makes lineage tree.
    """

    root = lineage.output_lineage[0]
    if np.isfinite(root.obs[4]):  # starts from G1
        if np.isfinite(root.obs[3]):
            length = root.obs[2] + root.obs[3]
        else:
            length = root.obs[2]
        assert np.isfinite(length)
    else:  # starts from G2
        length = root.obs[3]
        assert np.isfinite(length)
    a = [Clade(length)]

    # input the root cells in the lineage
    c = CladeRecursive(lineage.output_lineage[0], a, censore, color)

    return draw(c, axes=axes)


def plot_networkx(num_states, T, drug_name):
    """ This plots the Transition matrix for each condition. """
    G = nx.MultiDiGraph()
    num_states = T.shape[0]

    # node labels
    labels = {}
    for i in range(num_states):
        labels[i] = "state " + str(i + 1)

    # add nodes
    for i in range(num_states):
        G.add_node(i, pos=(-2, -2), label=labels[i], style='filled', fillcolor=cs[i])

    # add edges
    for i in range(num_states):
        for j in range(num_states):
            G.add_edge(i, j, penwidth=3 * T[i, j], minlen=1, label=str(np.round(T[i, j], 2)))

    # add graphviz layout options (see https://stackoverflow.com/a/39662097)
    G.graph['edge'] = {'arrowsize': '0.6', 'splines': 'curved'}
    G.graph['graph'] = {'scale': '1'}

    # adding attributes to edges in multigraphs is more complicated but see
    # https://stackoverflow.com/a/26694158
    for i in range(num_states):
        G[i][i][0]['color'] = 'black'

    A = nx.drawing.nx_agraph.to_agraph(G)
    A.layout('dot')
    A.draw('lineage/figures/cartoons/' + str(drug_name) + '.svg')


def draw(
    tree,
    label_func=str,
    do_show=True,
    show_confidence=True,
    # For power users
    axes=None,
    branch_labels=None,
    label_colors=None,
    *args,
    **kwargs
):
    """Plot the given tree using matplotlib (or pylab).
    The graphic is a rooted tree, drawn with roughly the same algorithm as
    draw_ascii.
    Additional keyword arguments passed into this function are used as pyplot
    options. The input format should be in the form of:
    pyplot_option_name=(tuple), pyplot_option_name=(tuple, dict), or
    pyplot_option_name=(dict).
    Example using the pyplot options 'axhspan' and 'axvline'::
        from Bio import Phylo, AlignIO
        from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
        constructor = DistanceTreeConstructor()
        aln = AlignIO.read(open('TreeConstruction/msa.phy'), 'phylip')
        calculator = DistanceCalculator('identity')
        dm = calculator.get_distance(aln)
        tree = constructor.upgma(dm)
        Phylo.draw(tree, axhspan=((0.25, 7.75), {'facecolor':'0.5'}),
        ... axvline={'x':0, 'ymin':0, 'ymax':1})
    Visual aspects of the plot can also be modified using pyplot's own functions
    and objects (via pylab or matplotlib). In particular, the pyplot.rcParams
    object can be used to scale the font size (rcParams["font.size"]) and line
    width (rcParams["lines.linewidth"]).
    :Parameters:
        label_func : callable
            A function to extract a label from a node. By default this is str(),
            but you can use a different function to select another string
            associated with each node. If this function returns None for a node,
            no label will be shown for that node.
        do_show : bool
            Whether to show() the plot automatically.
        show_confidence : bool
            Whether to display confidence values, if present on the tree.
        axes : matplotlib/pylab axes
            If a valid matplotlib.axes.Axes instance, the phylogram is plotted
            in that Axes. By default (None), a new figure is created.
        branch_labels : dict or callable
            A mapping of each clade to the label that will be shown along the
            branch leading to it. By default this is the confidence value(s) of
            the clade, taken from the ``confidence`` attribute, and can be
            easily toggled off with this function's ``show_confidence`` option.
            But if you would like to alter the formatting of confidence values,
            or label the branches with something other than confidence, then use
            this option.
        label_colors : dict or callable
            A function or a dictionary specifying the color of the tip label.
            If the tip label can't be found in the dict or label_colors is
            None, the label will be shown in black.
    """

    import matplotlib.pyplot as plt
    import matplotlib.collections as mpcollections

    # Arrays that store lines for the plot of clades
    horizontal_linecollections = []
    vertical_linecollections = []

    # Options for displaying branch labels / confidence
    def conf2str(conf):
        if int(conf) == conf:
            return str(int(conf))
        return str(conf)

    if not branch_labels:
        if show_confidence:

            def format_branch_label(clade):
                try:
                    confidences = clade.confidences
                    # phyloXML supports multiple confidences
                except AttributeError:
                    pass
                else:
                    return "/".join(conf2str(cnf.value) for cnf in confidences)
                if clade.confidence is not None:
                    return conf2str(clade.confidence)
                return None

        else:

            def format_branch_label(clade):
                return None

    elif isinstance(branch_labels, dict):

        def format_branch_label(clade):
            return branch_labels.get(clade)

    else:
        if not callable(branch_labels):
            raise TypeError(
                "branch_labels must be either a dict or a callable (function)"
            )
        format_branch_label = branch_labels

    # options for displaying label colors.
    if label_colors:
        if callable(label_colors):

            def get_label_color(label):
                return label_colors(label)

        else:
            # label_colors is presumed to be a dict
            def get_label_color(label):
                return label_colors.get(label, "black")

    else:

        def get_label_color(label):
            # if label_colors is not specified, use black
            return "black"

    # Layout

    def get_x_positions(tree):
        """Create a mapping of each clade to its horizontal position.
        Dict of {clade: x-coord}
        """
        depths = tree.depths()
        # If there are no branch lengths, assume unit branch lengths
        if not max(depths.values()):
            depths = tree.depths(unit_branch_lengths=True)
        return depths

    def get_y_positions(tree):
        """Create a mapping of each clade to its vertical position.
        Dict of {clade: y-coord}.
        Coordinates are negative, and integers for tips.
        """
        maxheight = tree.count_terminals()
        # Rows are defined by the tips
        heights = {
            tip: maxheight - i for i, tip in enumerate(reversed(tree.get_terminals()))
        }

        # Internal nodes: place at midpoint of children
        def calc_row(clade):
            for subclade in clade:
                if subclade not in heights:
                    calc_row(subclade)
            # Closure over heights
            heights[clade] = (
                heights[clade.clades[0]] + heights[clade.clades[-1]]
            ) / 2.0

        if tree.root.clades:
            calc_row(tree.root)
        return heights

    x_posns = get_x_positions(tree)
    y_posns = get_y_positions(tree)
    # The function draw_clade closes over the axes object
    if axes is None:
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)
    elif not isinstance(axes, plt.matplotlib.axes.Axes):
        raise ValueError("Invalid argument for axes: %s" % axes)

    def draw_clade_lines(
        use_linecollection=False,
        orientation="horizontal",
        y_here=0,
        x_start=0,
        x_here=0,
        y_bot=0,
        y_top=0,
        color="black",
        lw=".1",
    ):
        """Create a line with or without a line collection object.
        Graphical formatting of the lines representing clades in the plot can be
        customized by altering this function.
        """
        if not use_linecollection and orientation == "horizontal":
            axes.hlines(y_here, x_start, x_here, color=color, lw=lw)
        elif use_linecollection and orientation == "horizontal":
            horizontal_linecollections.append(
                mpcollections.LineCollection(
                    [[(x_start, y_here), (x_here, y_here)]], color=color, lw=lw
                )
            )
        elif not use_linecollection and orientation == "vertical":
            axes.vlines(x_here, y_bot, y_top, color=color)
        elif use_linecollection and orientation == "vertical":
            vertical_linecollections.append(
                mpcollections.LineCollection(
                    [[(x_here, y_bot), (x_here, y_top)]], color=color, lw=lw
                )
            )

    def draw_clade(clade, x_start, color, lw):
        """Recursively draw a tree, down from the given clade."""
        x_here = x_posns[clade]
        y_here = y_posns[clade]
        # phyloXML-only graphics annotations
        if hasattr(clade, "color") and clade.color is not None:
            color = clade.color.to_hex()
        if hasattr(clade, "width") and clade.width is not None:
            lw = clade.width * plt.rcParams["lines.linewidth"]
        # Draw a horizontal line from start to here
        draw_clade_lines(
            use_linecollection=True,
            orientation="horizontal",
            y_here=y_here,
            x_start=x_start,
            x_here=(x_start + clade.G1),
            color=color,
            lw=clade.G1lw,
        )
        draw_clade_lines(
            use_linecollection=True,
            orientation="horizontal",
            y_here=y_here,
            x_start=(x_start + clade.G1),
            x_here=x_here,
            color=color,
            lw=clade.G2lw,
        )
        # Add node/taxon labels
        label = label_func(clade)
        if label not in (None, clade.__class__.__name__):
            axes.text(
                x_here,
                y_here,
                " %s" % label,
                verticalalignment="center",
                color=get_label_color(label),
            )
        # Add label above the branch (optional)
        conf_label = format_branch_label(clade)
        if conf_label:
            axes.text(
                0.5 * (x_start + x_here),
                y_here,
                conf_label,
                fontsize="small",
                horizontalalignment="center",
            )
        if clade.clades:
            # Draw a vertical line connecting all children
            y_top = y_posns[clade.clades[0]]
            y_bot = y_posns[clade.clades[-1]]
            # Only apply widths to horizontal lines, like Archaeopteryx
            draw_clade_lines(
                use_linecollection=True,
                orientation="vertical",
                x_here=x_here,
                y_bot=y_bot,
                y_top=y_top,
                color=color,
                lw=lw,
            )
            # Draw descendents
            for child in clade:
                draw_clade(child, x_here, color, lw)

    draw_clade(tree.root, 0, "k", plt.rcParams["lines.linewidth"])

    # If line collections were used to create clade lines, here they are added
    # to the pyplot plot.
    for i in horizontal_linecollections:
        axes.add_collection(i)
    for i in vertical_linecollections:
        axes.add_collection(i)

    # Aesthetics

    try:
        name = tree.name
    except AttributeError:
        pass
    else:
        if name:
            axes.set_title(name)
    axes.set_xlabel("branch length")
    axes.set_ylabel("taxa")
    # Add margins around the tree to prevent overlapping the axes
    xmax = max(x_posns.values())
    axes.set_xlim(-0.05 * xmax, 1.25 * xmax)
    # Also invert the y-axis (origin at the top)
    # Add a small vertical margin, but avoid including 0 and N+1 on the y axis
    axes.set_ylim(max(y_posns.values()) + 0.8, 0.2)

    # Parse and process key word arguments as pyplot options
    for key, value in kwargs.items():
        try:
            # Check that the pyplot option input is iterable, as required
            list(value)
        except TypeError:
            raise ValueError(
                'Keyword argument "%s=%s" is not in the format '
                "pyplot_option_name=(tuple), pyplot_option_name=(tuple, dict),"
                " or pyplot_option_name=(dict) " % (key, value)
            ) from None
        if isinstance(value, dict):
            getattr(plt, str(key))(**dict(value))
        elif not (isinstance(value[0], tuple)):
            getattr(plt, str(key))(*value)
        elif isinstance(value[0], tuple):
            getattr(plt, str(key))(*value[0], **dict(value[1]))

    if do_show:
        plt.show()
