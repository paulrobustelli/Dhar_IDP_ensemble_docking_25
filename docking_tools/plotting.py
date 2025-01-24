import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import ndarray


def plot_ensemble_ii(run_dict, weights, lig23= False, 
                 c_labels=['Hydrophobic contact', 'Aromatic stacking', 'Hydrogen Bonding', 'Charge-charge contact'], 
                 c_list=['forestgreen', 'black', 'indianred','cornflowerblue'], 
                 res_sequence = ['D121','','E123',
                '','Y125','','M127','','S129','',
                'E131','','Y133','','D135','','E137','','E139','']): 
    
    """ Takes a dictionary for a given run that has the keys ['hphob', 'aro', 'charge', 'hbond'], 
    each leading to an array that has the same (20,) with values corresponding to the average contact 
    probability for that contact type. Also takes a numpy array of cluster weights. """

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
     
    if lig23 : 
        c_types=['hphob', 'aro', 'hbond']
    else: 
        c_types=['hphob', 'aro', 'hbond', 'charge']
        
    # for each type of contact
    for j, ctype in enumerate(c_types): 
        con = run_dict[ctype].mean(1).T@weights
        ax.plot(con, color = c_list[j], label = c_labels[j], lw=2.5, alpha=[0.7, 0.7, 0.8, 0.8][j], )

    ax.set_ylim(0,0.7)
    ax.set_xticks(np.arange(20)-1, labels=res_sequence)
    ax.tick_params(labelsize=15)
    plt.setp(ax.get_xticklabels(), rotation=45)
    ax.set_ylabel("Probability", size=15, labelpad=15)
    ax.set_xlabel('Residue', size=15, labelpad=15)
    ax.legend()
 
    plt.tight_layout()
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    ##############################

def plot_clus_ii(run_dict, lig23= False, 
                 c_labels=['Hydrophobic contact', 'Aromatic stacking', 'Hydrogen Bonding', 'Charge-charge contact'], 
                 c_list=['forestgreen', 'black', 'indianred','cornflowerblue'], 
                 res_sequence = ['D121','','E123',
                '','Y125','','M127','','S129','',
                'E131','','Y133','','D135','','E137','','E139','']): 
    
    """ Takes a dictionary for a given run that has the keys ['hphob', 'aro', 'charge', 'hbond'], 
    each leading to an array that has the same (20,) with values corresponding to the average contact 
    probability for that contact type. """

    # Now, plotting... 
    fig, axes = plt.subplots(5, 4, figsize=(32, 24), sharex=True, sharey=True)
    
    # initializing correlation array (20 clusters, 4 contact types) 
    if lig23 == False: 
        c_types=['hphob', 'aro', 'hbond', 'charge']
    else: 
        c_types=['hphob', 'aro', 'hbond']

    
    # for each cluster 
    for i, ax in enumerate(axes.flat):
        # for each type of contact
        for j, ctype in enumerate(c_types): 
            con = run_dict[ctype][i].mean(0)
            ax.plot(con, color = c_list[j], label = f'{c_labels[j]}', lw=3.5, alpha=[0.7, 0.7, 0.8, 0.8][j])


            if i == 0 : 
                ax.legend(loc='upper left',prop={'size': 20})
        ax.set_ylim(0,0.7)
        ax.set_xticks(np.arange(20)-1, labels=res_sequence)
        ax.tick_params(labelsize=28)
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.set_title(f'Cluster {i}',size=28)
        if i == 0 or i%4 == 0 : ax.set_ylabel("Probability", size=28, labelpad=15)
        if  i >15: ax.set_xlabel('Residue', size=28, labelpad=15)

    plt.tight_layout()
    ##############################
    
def subplot_snsheat(data, ax=None, cbar=False, cbar_ax=None, cbar_s = None):
    if type(data) is ndarray:
        if ax is None:
            ax = plt.gca()
        plot = sns.heatmap(data, cmap="jet",vmin=0.0,vmax=0.75,ax=ax, cbar=cbar, cbar_ax=None if not cbar else cbar_ax, alpha=.8)
        cax = plot.figure.axes[-1]
        if cbar== False: 
            cax.tick_params(labelsize=20)
        else: 
            cax.tick_params(labelsize=16)
            cax.set_ylabel("Dual-Residue Contact Probability", fontsize=cbar_s)
        return plot
    else: pass

def plot_clus_dual(rundata, res_sequence=['D121', '', 'E123', '', 'Y125', '', 'M127', '', 'S129', 
                                    '', 'E131', '', 'Y133', '', 'D135', '', 'E137', '', 'E139', '']): 
    n_residues=len(res_sequence)
    fig, axes = plt.subplots(4, 5, sharex=True, sharey=True, figsize=(22, 17))
    cbar_ax = fig.add_axes([.91, .3, .03, .4])

    # Now, going through the clusters... 
    for i, ax in enumerate(axes.flat):
        cb = i==1
        subplot_snsheat(rundata[i], ax=ax, cbar=cb, cbar_ax=cbar_ax, cbar_s=25) 
        ax.set_title(f'Cluster {i}',size=20)
        ax.set_xticks(range(n_residues), res_sequence, rotation=45, size=17)
        ax.set_yticks(range(n_residues), res_sequence, rotation=45, size=17)
        ax.invert_yaxis()

def plot_ensemble_dual(run_data, weights, 
                  res_sequence=['D121', '', 'E123', '', 'Y125', '', 'M127', '', 'S129', '', 'E131', 
                                '', 'Y133', '', 'D135', '', 'E137', '', 'E139', '']): 

    # Initializing figure
    fig, ax = plt.subplots(figsize=(5, 5))
    cbar_ax = fig.add_axes([.91, .2, .05, .5])
    cb = 1

    # Now, going through the clusters... 
    con =  np.sum(run_data * weights[:, np.newaxis, np.newaxis], axis=0)
    subplot_snsheat(con, ax=ax, cbar=cb, cbar_ax=cbar_ax, cbar_s=14) 

    ax.set_xticks(range(20), res_sequence, rotation=45, size=18)
    ax.set_yticks(range(20), res_sequence, rotation=45, size=18)
    ax.invert_yaxis()
