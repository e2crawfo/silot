import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

locs = [[0.5, 0.75], [0.25, 0.25], [0.1, 0.1]]
stds = [0.1, 0.15, 0.2]

fig_unit_size = 3


def grid_subplots(h, w, fig_unit_size):
    fig, axes = plt.subplots(h, w, figsize=(w * fig_unit_size, h * fig_unit_size))
    axes = np.array(axes).reshape(h, w)  # to fix the inconsistent way axes is return if h==1 or w==1
    return fig, axes


fig, axes = grid_subplots(len(stds), 2*len(locs), fig_unit_size)

max_pdf = 0.0

pdfs = np.zeros((len(stds), len(locs)), dtype=np.object)
n_samples = 48

for i, std in enumerate(stds):
    print(std)
    for j, (x, y) in enumerate(locs):
        dist_x = norm(loc=x, scale=std)
        dist_y = norm(loc=y, scale=std)

        x_activation = dist_x.pdf(np.linspace(0, 1, n_samples))
        y_activation = dist_y.pdf(np.linspace(0, 1, n_samples))
        print(x_activation / x_activation.max())

        pdf = y_activation[:, None] * x_activation[None, :]
        pdfs[i, j] = pdf

        max_pdf = np.maximum(max_pdf, pdf.max())


for i, std in enumerate(stds):
    for j, (x, y) in enumerate(locs):
        ax = axes[i, 2*j]
        ax.imshow(pdfs[i, j], vmin=0.0, vmax=max_pdf)
        ax.set_title('std={}, loc=(x={}, y={})'.format(std, x, y))

        ax = axes[i, 2*j+1]
        pdf = pdfs[i, j]
        for k in range(pdf.shape[0]):
            ax.plot(np.linspace(0, 1, n_samples), pdf[k])

        ax.set_ylabel('pdf')
        ax.set_xlabel('x')
        ax.set_ylim((-0.01, 1.1 * max_pdf))

plt.show()