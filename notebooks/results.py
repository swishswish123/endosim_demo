import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 



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


if __name__ == '__main__':
    results_path = '/Users/aure/Documents/i4health/project/endosim_demo/notebooks/results'
    automatic = pd.read_csv(f'{results_path}/automatic.csv',index_col=0)
    pointer = pd.read_csv(f'{results_path}/pointer.csv', index_col=0)
    endoscope = pd.read_csv(f'{results_path}/endoscope.csv', index_col=0)

    automatic_px = automatic[::2]
    automatic_mm = automatic[1::2]
    endoscope_mm = endoscope.iloc[:-1]
    endoscope_px = endoscope.iloc[-1:]

    plt.figure()
    plt.title('px errors')
    plt.bar(data=automatic_px, x=automatic_px.loc[:].index, height='sigma_12', label='automatic')
    plt.bar(data=endoscope_px, x=endoscope_px.loc[:].index, height='sigma_12', label='endoscope')
    plt.legend()
    plt.show()

    plt.figure()
    plt.title('mm errors')
    plt.bar(data=pointer, x=pointer.loc[:].index, height='sigma_12', label='pointer')
    plt.bar(data=automatic_mm, x=automatic_mm.loc[:].index, height='sigma_12', label='automatic')
    plt.bar(data=endoscope_mm, x=endoscope_mm.loc[:].index, height='sigma_12', label='endoscope')

    plt.legend()
    plt.show()

