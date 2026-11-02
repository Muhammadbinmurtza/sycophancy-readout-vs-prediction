"""Paper figures (static PDF + PNG). Run from paper/figs with the project venv:
    ../../sycophancy-direction/.venv/bin/python make_figs.py
Palette: validated categorical slots (light mode); low-contrast slots always carry markers and direct labels.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = ".."  # repository root; run from figs/
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#22252a", "#6b6f76", "#e3e4e6"
MODELS = [("qwen1.5b", "Qwen2.5-1.5B", f"{R}/pilot/e8_qwen1.5b_claude.json"),
          ("qwen3_4b", "Qwen3-4B", f"{R}/gpu_out/qwen3_4b/e8_qwen3_4b_claude.json"),
          ("llama3b", "Llama-3.2-3B", f"{R}/gpu_out/llama3b/e8_llama3b_claude.json"),
          ("gemma4b", "Gemma-3-4B", f"{R}/gpu_out/gemma4b/e8_gemma4b_claude.json")]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    os.makedirs("out", exist_ok=True)
    fig.savefig(f"out/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"out/{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_e8():
    """Per-layer resid AUROC on real episodes: simulated->real transfer vs in-domain, answer vs pre-reply."""
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.6), sharey=True)
    for j, (tag, name, path) in enumerate(MODELS):
        pl = json.load(open(path))["per_layer"]
        for i, ro in enumerate(("ans", "pre")):
            ax = axes[i, j]
            ks = sorted([k for k in pl if k.startswith(ro + "_L")], key=lambda k: int(k.split("L")[1]))
            n = len(ks) - 1
            x = [int(k.split("L")[1]) / n for k in ks]
            ax.axhline(0.5, color=MUTED, lw=0.7, ls=(0, (3, 2)))
            ax.plot(x, [pl[k]["indomain"] for k in ks], color=ORANGE, lw=1.6, marker="s", ms=2.2, label="in-domain (real)")
            ax.plot(x, [pl[k]["transfer"] for k in ks], color=BLUE, lw=1.6, marker="o", ms=2.2, label="trained on simulated")
            ax.set_ylim(0.25, 1.0)
            ax.set_xticks([0, 0.5, 1])
            style(ax)
            if i == 0:
                ax.set_title(name, fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(("Answer read-out" if ro == "ans" else "Pre-reply") + "\nAUROC")
            if i == 1:
                ax.set_xlabel("relative layer depth")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    save(fig, "fig_e8_layers")


def fig_e10():
    """E10: out-of-fold r per layer (two-way FE residual P(cave)) for the in-domain probe and the
    simulated-dialogue probe, with the max-over-layers null threshold."""
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.5), sharey=True)
    for ax, (tag, name, _) in zip(axes, MODELS):
        a = json.load(open(f"{R}/gpu_out/e10/e10_{tag}_p200.json"))
        g = json.load(open(f"{R}/gpu_out/e10/e10_genadi_{tag}.json"))
        fe = a["r_per_layer"]["fe"]
        L = len(fe) - 1
        xs = [l / L for l in range(1, L + 1)]
        ax.axhline(0, color=MUTED, lw=0.7)
        thr = a["primary"]["null_max_95"]
        ax.axhspan(-thr, thr, color=GRID, lw=0, label="null, 95% max over layers")
        ax.plot(xs, fe[1:], color=ORANGE, lw=1.6, marker="s", ms=2.2, label="in-domain probe")
        gk = sorted([k for k in g["r"] if k.startswith("resid_L")], key=lambda k: int(k.split("L")[1]))
        ax.plot([int(k.split("L")[1]) / L for k in gk], [g["r"][k] for k in gk], color=BLUE, lw=1.6, marker="o",
                ms=2.2, label="trained on simulated (resid)")
        ax.axhline(g["top16_r"], color=BLUE, lw=1.2, ls=(0, (4, 2)), label="trained on simulated (top-16 heads)")
        b = a["primary"]["best_layer"]
        ax.annotate(f"r = {fe[b]:.2f}", (b / L, fe[b]), xytext=(4, 4), textcoords="offset points", color=INK, fontsize=8)
        ax.set_title(name, fontsize=9, color=INK)
        ax.set_xlabel("relative layer depth")
        ax.set_xticks([0, 0.5, 1])
        style(ax)
    axes[0].set_ylabel("out-of-fold r with\nresidual P(cave)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    save(fig, "fig_e10")


def fig_factorial():
    """Top-16 head score decomposition on simulated dialogues (units of within-kind SD)."""
    paths = {"qwen1.5b": f"{R}/pilot/factorial_qwen1.5b.json"}
    paths.update({t: f"{R}/gpu_out/{t}/factorial_{t}.json" for t in ("qwen3_4b", "llama3b", "gemma4b")})
    effects = [("changed", "answer changed"), ("final_false", "final answer false"), ("first_true", "first answer true")]
    colors = [BLUE, ORANGE, AQUA]
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    w = 0.26
    names = []
    for i, (tag, name, _) in enumerate(MODELS):
        if not os.path.exists(paths[tag]):
            continue
        d = json.load(open(paths[tag]))
        top = d["top_heads_mean"]
        names.append((i, name))
        for k, ((key, lab), c) in enumerate(zip(effects, colors)):
            v = top[key]
            ax.bar(i + (k - 1) * w, v, w * 0.92, color=c, label=lab if i == 0 else None)
            ax.text(i + (k - 1) * w, v + 0.04, f"{v:.1f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks([i for i, _ in names])
    ax.set_xticklabels([n for _, n in names])
    ax.set_ylabel("effect on probe score (SD)")
    style(ax)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.18))
    save(fig, "fig_factorial")


if __name__ == "__main__":
    fig_e8()
    fig_e10()
    try:
        fig_factorial()
    except (KeyError, FileNotFoundError) as e:
        print("factorial figure skipped:", e)
