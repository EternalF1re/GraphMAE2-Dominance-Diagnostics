"""Approved publication styles shared by the two new figures; no inference."""
import os
from pathlib import Path
import tempfile

GRAPH_UPDATES=(250,711,1422,2133,3555,5688,8532)


def publication_destination(repository_root, destination):
    original = Path(destination)
    if original.is_symlink(): raise ValueError("Plot output cannot be a symlink")
    root, target = Path(repository_root).resolve(), original.resolve()
    if target.suffix.lower() != ".pdf": raise ValueError("Plot output must be a new PDF")
    if target.is_relative_to(root):
        parts = target.relative_to(root).parts
        allowed = parts[0] == "reproduced" or (len(parts) >= 4 and parts[0] == "figures" and parts[2] == "reproduced")
        if not allowed: raise ValueError("Plot output would overlap protected source or frozen evidence")
    if target.exists() or target.with_suffix(".png").exists():
        raise FileExistsError("Use a new plot destination; existing files are not replaced")
    return target


def publish_figure(draw, payload, repository_root, destination):
    target = publication_destination(repository_root, destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="render_", dir=target.parent) as temporary:
        folder = Path(temporary)
        configure_matplotlib(folder)
        draw(payload, folder, 220)
        for suffix in [".pdf", ".png"]:
            generated = list(folder.glob("*"+suffix))
            if len(generated) != 1: raise ValueError("Expected exactly one rendered artifact")
            with target.with_suffix(suffix).open("xb") as stream: stream.write(generated[0].read_bytes())
    return target

def configure_matplotlib(root):
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8.5, "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans", "text.usetex": False, "axes.unicode_minus": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .7})
    return matplotlib

def clean_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D9DDE0", linestyle="--", linewidth=.55)
    ax.tick_params(width=.6, length=3)

def validate_layout(fig, axes):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    overlap_count = 0
    for ax in axes:
        boxes = [t.get_window_extent(renderer) for t in ax.get_xticklabels() if t.get_visible() and t.get_text()]
        overlap_count += sum(a.overlaps(b) for a, b in zip(boxes, boxes[1:]))
        annotations = [t.get_window_extent(renderer) for t in ax.texts if t.get_visible() and t.get_text()]
        overlap_count += sum(a.overlaps(b) for i, a in enumerate(annotations) for b in annotations[i + 1:])
        for box in annotations:
            if not (fig.bbox.x0 <= box.x0 <= box.x1 <= fig.bbox.x1 and fig.bbox.y0 <= box.y0 <= box.y1 <= fig.bbox.y1):
                raise ValueError("Annotation extends outside figure")
    if overlap_count: raise ValueError("Overlapping tick/ratio annotation bounding boxes")
    if fig._suptitle is not None and fig._suptitle.get_text(): raise ValueError("Unrequested overall title")
    return dict(overlapping_tick_or_ratio_labels=int(overlap_count), annotations_within_figure=True, overall_title=False)

def graph_figure(trajectory, folder, dpi):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8., 3.65), sharey=True, layout="constrained")
    settings = [("G1", "#BF443F", "o", r"Original weight ($\lambda=10$)"),
                ("G3", "#007D78", "s", r"Reduced weight ($\lambda=0.3$)")]
    low = min(r["mean_accuracy_pct"] - r["sd_accuracy_pp"] for r in trajectory if r["group"] in ["G1", "G3"])
    high = max(r["mean_accuracy_pct"] + r["sd_accuracy_pp"] for r in trajectory if r["group"] in ["G1", "G3"])
    for ax, protocol in zip(axes, "AB"):
        for group, color, marker, label in settings:
            points = [r for r in trajectory if r["protocol"] == protocol and r["group"] == group]
            ax.errorbar([r["global_update"] for r in points], [r["mean_accuracy_pct"] for r in points],
                yerr=[r["sd_accuracy_pp"] for r in points], color=color, marker=marker,
                markersize=4.3, markerfacecolor="white", markeredgewidth=1.1,
                linewidth=1.35, elinewidth=.85, capsize=2.5, label=label)
        title = "(a)  Global gradient clipping" if protocol == "A" else "(b)  Group-wise gradient clipping"
        ax.set_title(title, loc="left", fontweight="semibold", pad=9)
        ax.set_xlabel("Optimizer updates", labelpad=5)
        ax.set_xticks(GRAPH_UPDATES, [str(x) for x in GRAPH_UPDATES], rotation=70, ha="right")
        ax.set_xlim(0, 9000); ax.set_ylim(low - .35, high + .35)
        clean_axis(ax)
    axes[0].set_ylabel("Validation accuracy (%)")
    axes[1].tick_params(axis="y", labelleft=True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    geometry = validate_layout(fig, axes)
    fig.savefig(folder / "Fig_GraphMAE2_Trajectory_Main.pdf", bbox_inches="tight")
    fig.savefig(folder / "Fig_GraphMAE2_Trajectory_Main.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return geometry

def mask_figure(groups, folder, dpi):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8., 3.65), layout="constrained")
    for ax, key, sd, scale, color, label, panel in [
        (axes[0], "auc_mean", "auc_sd", 100, "#2C6C9E", "Validation-accuracy trajectory AUC (%)", "A"),
        (axes[1], "epoch500_mean_pct", "epoch500_sd_pp", 1, "#B35D26", "Epoch-500 validation accuracy (%)", "B")]:
        values = [scale * r[key] for r in groups]; errors = [scale * r[sd] for r in groups]
        x = list(range(5))
        ax.errorbar(x, values, yerr=errors, color=color, marker="o", markerfacecolor="white",
                    markeredgewidth=1.1, markersize=4.8, linewidth=1.35, elinewidth=.9, capsize=3)
        minimum, maximum = min(v - e for v, e in zip(values, errors)), max(v + e for v, e in zip(values, errors))
        span = maximum - minimum
        ax.set_ylim(minimum - .13 * span, maximum + .27 * span)
        ax.set_xlim(-.55, 4.55)
        ax.set_xticks(x, ["0.001", "0.125", "0.25", "0.5", "1.0"])
        ax.set_xlabel(r"Auxiliary loss weight ($\alpha$)"); ax.set_ylabel(label)
        ax.set_title(panel, loc="left", fontweight="semibold", pad=9)
        clean_axis(ax)
    geometry = validate_layout(fig, axes)
    fig.savefig(folder / "Fig_MaskGAE_Dose_Response.pdf", bbox_inches="tight")
    fig.savefig(folder / "Fig_MaskGAE_Dose_Response.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return geometry
