import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def select(propagated_presence, discovered_presence, temperature):
    remaining_presence = []
    weights = []
    used_weights = []
    final_weights = []

    for i in range(propagated_presence.shape[1]):
        remaining_presence.append(discovered_presence)

        probs = discovered_presence / (tf.reduce_sum(discovered_presence, axis=-1, keepdims=True) + 1e-6)
        _weights = tfp.distributions.RelaxedOneHotCategorical(temperature, probs=probs).sample()
        emptiness = 1 - propagated_presence[:, i:i+1]
        _used_weights = emptiness * _weights
        _final_weights = _used_weights * discovered_presence
        discovered_presence = discovered_presence * (1-_used_weights)

        weights.append(_weights)
        used_weights.append(_used_weights)
        final_weights.append(_final_weights)

    remaining_presence = tf.stack(remaining_presence, axis=1)
    weights = tf.stack(weights, axis=1)
    used_weights = tf.stack(used_weights, axis=1)
    final_weights = tf.stack(final_weights, axis=1)

    return remaining_presence, weights, used_weights, final_weights


if __name__ == "__main__":

    B = 3
    N = 16
    M = 16
    temperature = 0.01
    propagated_presence = tfp.distributions.RelaxedBernoulli(probs=0.4 * np.ones((B, N)), temperature=0.1).sample()
    discovered_presence = tfp.distributions.RelaxedBernoulli(probs=0.4 * np.ones((B, M)), temperature=0.1).sample()
    remaining_presence, weights, used_weights, final_weights = select(propagated_presence, discovered_presence, temperature=temperature)

    sess = tf.Session()
    pp, rp, w, u, f = sess.run([propagated_presence, remaining_presence, weights, used_weights, final_weights])

    print(pp)
    print(rp)
    print(w)
    print(u)
    print(f)

    fig_unit_size = 3
    fig_height = N + 1
    n_images = 4
    fig_width = B * n_images
    alpha = 1.0
    fig, axes = plt.subplots(fig_height, fig_width, figsize=(fig_unit_size*fig_width, fig_unit_size*fig_height))

    # for ax in axes.flatten():
    #     ax.set_axis_off()

    for b in range(B):
        bg_color = 'k'

        ax = axes[0, b*n_images]
        ax.bar(np.arange(N), pp[b], align='center', alpha=alpha, color='b')
        ax.set_title("prop presence")
        ax.set_facecolor(bg_color)
        ax.set_ylim((0, 1))

        for i in np.arange(N):
            ax = axes[i+1, b*n_images+0]
            ax.bar(np.arange(M), rp[b, i], align='center', alpha=alpha, color='r')
            if i == 0:
                ax.set_title("remaining_presence")
            ax.set_facecolor(bg_color)
            ax.set_ylim((0, 1))

            rect = patches.Rectangle(
                (-0.02, 0), 0.02, pp[b, i], clip_on=False, transform=ax.transAxes, facecolor='b')
            ax.add_patch(rect)

            ax = axes[i+1, b*n_images+1]

            ax.bar(np.arange(M), w[b, i], align='center', alpha=alpha, color='g')
            if i == 0:
                ax.set_title("weights")
            ax.set_facecolor(bg_color)
            ax.set_ylim((0, 1))

            ax = axes[i+1, b*n_images+2]

            ax.bar(np.arange(M), u[b, i], align='center', alpha=alpha, color='c')
            if i == 0:
                ax.set_title("used weights")
            ax.set_facecolor(bg_color)
            ax.set_ylim((0, 1))

            ax = axes[i+1, b*n_images+3]

            ax.bar(np.arange(M), f[b, i], align='center', alpha=alpha, color='y')
            if i == 0:
                ax.set_title("final weights")
            ax.set_facecolor(bg_color)
            ax.set_ylim((0, 1))

    plt.subplots_adjust(left=0.05, right=.95, top=.95, bottom=0.05, wspace=0.1, hspace=0.15)

    plt.show()

    """
    When actually accepting the final values after the selection has been done, be sure to modulate by the
    remaining presence. This will ensure that...after everything has been selected, and we just have empty slots but no demand,
    things don't actually get placed in multiple slots because of the modulation by remaining presence.

    It will be used_weights * remaining_presence.

    I always though of this in terms of softmax using the presence values, but what we more end up doing is taking a uniform
    distribution over the present objects. Just a smooth version of that.

    """
