import numpy as np
import numpy.linalg as linalg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.axes import Axes

from nn import Network, Node, Link

nodewidth = nodeheight = 1
nodesize = np.array([nodewidth, nodeheight])


def draw_curve(ax: Axes, start, end, startdir=(1, 0), enddir=(1, 0), linewidth=1, color='black', alpha=1):
    pointnum = 20
    dist = linalg.norm(np.array(start) - np.array(end))*0.5
    startdir = np.array(startdir)*dist
    enddir = np.array(enddir)*dist
    t = np.linspace(0, 1, pointnum)
    t2 = t*t
    t3 = t2*t
    ht1 = t3*2 - t2*3 + 1
    ht2 = -t3*2 + t2*3
    ht3 = t3 - t2*2 + t
    ht4 = t3 - t2
    pts = np.hstack([(start[0]*ht1 + end[0]*ht2 + startdir[0]*ht3 + enddir[0]*ht4)[:, np.newaxis],
                     (start[1]*ht1 + end[1]*ht2 + startdir[1]*ht3 + enddir[1]*ht4)[:, np.newaxis]])
    line = Line2D(pts[:,0], pts[:,1])
    line.set_linewidth(linewidth)
    line.set_color(color)
    line.set_alpha(alpha)
    line.set_zorder(-1)
    return ax.add_line(line)


def draw_link(ax: Axes, link: Link, full_draw=True):
    absweight = abs(link.weight.data.item())
    linewidth = absweight*5
    color = plt.cm.rainbow(absweight)
    alpha = np.clip(absweight*2, 0, 1)
    if full_draw:
        start = link.srcnode.xy + nodesize / 2
        end = link.dstnode.xy + nodesize / 2
        graph = draw_curve(ax=ax, start=start, end=end, linewidth=linewidth, color=color, alpha=alpha)
        link.graph = graph
    elif link.graph is not None:
        line: Line2D = link.graph
        line.set_linewidth(linewidth)
        line.set_color(color)
        line.set_alpha(alpha)
        line.set_zorder(-1)


def draw_network(ax: Axes, nw: Network, full_draw=True) -> None:
    if not isinstance(nw, Network):
        return
    if full_draw:
        gaplayer = 5 + nodewidth
        gapnode = 0.5 + nodeheight
        for ilayer in range(len(nw.layers)):
            lnodes = nw.layers[ilayer]
            offsetx = 0
            offsety = -len(lnodes)/2
            for inode in range(len(lnodes)):
                node: Node = lnodes[inode]
                if node.xy is None:
                    node.xy = np.array([ilayer*gaplayer + offsetx, inode*gapnode + offsety])
                nodeui = mpatches.FancyBboxPatch(xy=node.xy, width=nodewidth, height=nodeheight, mutation_scale=0.5)
                ax.add_patch(nodeui)

    for layer in nw.layers:
        for node in layer:
            for link in node.outlinks:
                draw_link(ax, link, full_draw)

    if full_draw:
        ax.autoscale_view(scalex=True, scaley=True)
