import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_activity(states, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(states[:200, :60].T, aspect="auto", cmap="coolwarm", interpolation="nearest")
    ax.set(xlabel="Input digit time step", ylabel="Neuron index (first 60)", title="Teacher-forced reservoir activity (rate states, not spikes)")
    fig.colorbar(im, ax=ax, label="State"); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
