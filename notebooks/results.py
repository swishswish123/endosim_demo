import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

import plotly.express as px
from plotly.subplots import make_subplots

'''
def main(df):
    sigma = 0.12
    pointer_12 = df[(df['sigma'] == sigma) & (df['method'] == 'pointer')]
    end_12 = df[(df['sigma'] == sigma) & (df['method'] == 'endoscope')]
    vision_12 = df[(df['sigma'] == sigma) & (df['method'] == 'vision')]

    data = np.array([
        [
            'pointer', 'tracking+registration', pointer_12['registration']],

        ['endoscope', 'tracking+registration', end_12['registration']],
        ['endoscope', 'hand-eye', end_12['hand_eye']],
        ['endoscope AR', 'AR', end_12['AR']],

        ['vision', '2D', vision_12['2D']],
        ['vision', '3D', vision_12['3D']],
        ['vision', '2D&3D', vision_12['2D&3D']],

    ])

    df = pd.DataFrame(data, columns=['method', 'errors', 'R'])

    df['R'] = df['R'].astype(float)

    df.pivot(index='method', columns='errors', values='R').plot.bar(rot=0, stacked=True, color=['blue', 'green', 'red'])

    labels = ['pointer', 'endoscope', 'vision']
'''

def plot_pxl_errors(endoscope_px,phantom_pxl,sigma='sigma_12', ylim=10):
    plt.title(f'sigma 0.{sigma[-2:]}')
    #plt.bar(data=automatic_px, x=automatic_px.loc[:].index, height='sigma_12', label='automatic')
    #plt.bar(data=phantom_pxl, x=phantom_pxl.loc[:].index,  width=0.2,  height='sigma_25', label='phantom',alpha=0.5,  hatch='/')
    #plt.bar(data=phantom_pxl, x=phantom_pxl.loc[:].index,  width=0.2, height='sigma_15', label='phantom',alpha=0.5,  hatch='.')
    plt.bar(data=phantom_pxl, x=phantom_pxl.loc[:].index, height=sigma, label='phantom',alpha=0.5)
    plt.text(0, phantom_pxl[sigma]+0.1, round(float(phantom_pxl[sigma]),1), verticalalignment='center', horizontalalignment='center')

    #plt.bar(data=endoscope_px, x=endoscope_px.loc[:].index, height='sigma_25', label='endoscope',alpha=0.5,  hatch='/')
    #plt.bar(data=endoscope_px, x=endoscope_px.loc[:].index, height='sigma_15', label='endoscope',alpha=0.5,  hatch='.')
    plt.bar(data=endoscope_px, x=endoscope_px.loc[:].index, height=sigma, label='endoscope',alpha=0.5)
    plt.text(1, endoscope_px[sigma]+0.1, round(float(endoscope_px[sigma]),1), verticalalignment='center', horizontalalignment='center')

    plt.xticks(rotation = 45)
    plt.tight_layout()
    plt.legend()
    plt.ylim(0,ylim)
    #plt.show()
    #plt.savefig(f'notebooks/results/all_pxl_errors{sigma}.pdf')
    #return fig


def plot_mm_errors(endoscope_mm, pointer, phantom_mm, sigma='sigma_12', ylim=10):
    total_pointer = pointer.loc['pointer_T_R', :][sigma]
    total_end = endoscope_mm.loc[['hand_eye', 'endo_T_R', 'tracking'], :][sigma].sum() # - endoscope_mm.loc['tracking'][sigma]

    #plt.figure()
    plt.title(f'sigma 0.{sigma[-2:]}')

    # pointer
    #pointer_height = total_pointer
    plt.bar(x=1.5, width=4, height=total_pointer, label='total pointer',color=(0.1, 0.1, 0.1, 0.1),  edgecolor='blue')
    plt.text(1.5, total_pointer+0.1, round(float(total_pointer),1), verticalalignment='center', horizontalalignment='center')
    plt.bar(data=pointer, x=pointer.loc[:].index, height=sigma,alpha=0.5)
    
    # phantom
    #plt.bar(data=automatic_mm, x=automatic_mm.loc[:].index, height=sigma, label='automatic')
    phantom_height = phantom_mm[sigma]
    plt.bar(x=4, width=1, height=phantom_mm[sigma], label='total phantom',color=(0.1, 0.1, 0.1, 0.1),  edgecolor='orange')
    plt.text(4, phantom_height+0.1, round(float(phantom_height),1), verticalalignment='center', horizontalalignment='center')
    plt.bar(data=phantom_mm, x=phantom_mm.loc[:].index, height=sigma,alpha=0.5)

    # end
    #end_height = total_end[sigma].sum()
    plt.bar(x=6, width=3, height=total_end, label='total endoscope',color=(0.1, 0.1, 0.1, 0.1),  edgecolor='green')
    plt.text(6, total_end+0.1, round(float(total_end),1), verticalalignment='center', horizontalalignment='center')
    plt.bar(data=endoscope_mm, x=endoscope_mm.loc[:].index, height=sigma,alpha=0.5)

    plt.xticks(rotation = 'vertical')
    plt.ylim(0,ylim)
    plt.tight_layout()
    plt.legend()
    
    #plt.show()
    #plt.savefig(f'notebooks/results/all_mm_errors{sigma}.pdf')

if __name__ == '__main__':
    results_path = '/Users/aure/Documents/i4health/project/endosim_demo/notebooks/results'
    automatic = pd.read_csv(f'{results_path}/automatic.csv',index_col=0)
    #automatic['group'] = 'auto'
    pointer = pd.read_csv(f'{results_path}/pointer.csv',index_col=0)
    #pointer['group'] = 'pointer'
    endoscope = pd.read_csv(f'{results_path}/endoscope.csv',index_col=0)
    #endoscope['group'] = 'endoscope'
    phantom = pd.read_csv(f'{results_path}/phantom.csv',index_col=0)
    #phantom['group'] = 'phantom'

    automatic_px = automatic[::2] 
    automatic_mm = automatic[1::2]
    endoscope_mm = endoscope.iloc[:-1]
    endoscope_px = endoscope.iloc[-1:]
    phantom_mm = phantom.iloc[:1]
    phantom_pxl = phantom.iloc[1:2]

    all_pxl = pd.concat([phantom_pxl, endoscope_px])
    all_mm = pd.concat([phantom_mm, endoscope_mm, pointer])

    #total_pointer = pointer.loc['']
    #fig = make_subplots(rows=1, cols=2)
    #fig = px.bar(data_frame=all_mm, x='group', y='sigma_12', color='error',text_auto=True, fascet_row='')
    #fig.show()
    ylim = all_pxl.max().max()+all_pxl.max().max()*0.1
    fig = plt.figure()
    plt.suptitle('pxl errors')
    plt.subplot(131)
    plot_pxl_errors(endoscope_px, phantom_pxl,sigma='sigma_12', ylim= ylim)
    plt.subplot(132)
    plot_pxl_errors(endoscope_px, phantom_pxl,sigma='sigma_15', ylim=ylim)
    plt.subplot(133)
    plot_pxl_errors(endoscope_px, phantom_pxl,sigma='sigma_25', ylim=ylim)
    fig.set_figwidth(12)

    plt.savefig(f'notebooks/results/all_pxl_errors.pdf')
    plt.show()
    
    total_end = endoscope_mm.loc[['hand_eye', 'endo_T_R', 'tracking'], :].sum().max()
    ylim = total_end+0.2*total_end
    fig = plt.figure()
    plt.suptitle('mm errors')
    plt.subplot(131)
    plot_mm_errors(endoscope_mm, pointer, phantom_mm,sigma='sigma_12',ylim=ylim)
    plt.subplot(132)
    plot_mm_errors(endoscope_mm, pointer, phantom_mm,sigma='sigma_15',ylim=ylim)
    plt.subplot(133)
    plot_mm_errors(endoscope_mm, pointer, phantom_mm,sigma='sigma_25',ylim=ylim)
    plt.tight_layout()
    #plt.show()
    fig.set_figwidth(12)
    plt.savefig(f'notebooks/results/all_mm_errors.pdf')
    plt.show()

    #for sigma_val in ['sigma_12', 'sigma_15', 'sigma_25']:
        #plot_mm_errors(endoscope_mm, pointer, phantom_mm,sigma='sigma_12')

