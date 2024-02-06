import matplotlib
import matplotlib.pyplot as plt


def plot_estimated_MI(MI, estimated_MI, title):
    """
    Plot estimated mutual information values.
    """
    
    estimated_MI_mean = estimated_MI[:,0]
    estimated_MI_std  = estimated_MI[:,1]
    
    fig_normal, ax_normal = plt.subplots()

    fig_normal.set_figheight(11)
    fig_normal.set_figwidth(16)

    # Grid.
    ax_normal.grid(color='#000000', alpha=0.15, linestyle='-', linewidth=1, which='major')
    ax_normal.grid(color='#000000', alpha=0.1, linestyle='-', linewidth=0.5, which='minor')

    ax_normal.set_title(title)
    ax_normal.set_xlabel("$I(X,Y)$")
    ax_normal.set_ylabel("$\\hat I(X,Y)$")
    
    ax_normal.minorticks_on()

    ax_normal.plot(MI, MI, label="$I(X,Y)$", color='red')
    ax_normal.plot(MI, estimated_MI_mean, label="$\\hat I(X,Y)$")        
    ax_normal.fill_between(MI, estimated_MI_mean + estimated_MI_std, estimated_MI_mean - estimated_MI_std, alpha=0.2)

    ax_normal.legend(loc='upper left')

    ax_normal.set_xlim((0.0, None))
    ax_normal.set_ylim((0.0, None))

    plt.show();