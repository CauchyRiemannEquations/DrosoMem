import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_memory(recall, path):
    target = recall['prompt'] + recall['target']; pred = recall['prompt'] + recall['prediction']
    error = recall['first_error_digit_index']
    start = max(0, (error or 0) - 28); end = min(len(target), start + 58)
    fig, ax = plt.subplots(figsize=(12, 2.9)); ax.axis('off')
    ax.text(0, .95, f"Pi Memory Score: {recall['pi_memory_score']} generated digits" + (" (horizon reached)" if recall['censored'] else ""), fontsize=16, weight='bold')
    for y, label, seq in [(.63, 'Target', target), (.35, 'Recall', pred)]:
        ax.text(0, y, label, fontsize=12)
        for k, char in enumerate(seq[start:end]):
            idx = start + k
            color = '#dc2626' if label == 'Recall' and idx == error else ('#777777' if idx < recall['prompt_length'] else '#112233')
            ax.text(.105 + k * .015, y, char, fontfamily='monospace', fontsize=12, color=color)
    ax.text(0, .07, f"Showing digit indices {start}–{end-1} (leading 3 = index 0); prompt excluded from score. First error: {error}", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)

def plot_training(history, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([h['epoch'] for h in history], [h['accuracy'] for h in history])
    ax.set(xlabel="Epoch", ylabel="Next-digit accuracy", title="Training prefix only", ylim=(0,1.03))
    ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
