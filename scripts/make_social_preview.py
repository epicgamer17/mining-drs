import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

W, H = 1280, 640
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

bg = np.array([11, 15, 26]) / 255.0
ax.set_facecolor(bg)

left = 60
right = W - 60


def color_to_rgb(c):
    return np.array(c) / 255.0


accent = color_to_rgb((253, 177, 48))
accent2 = color_to_rgb((90, 200, 250))
gray = color_to_rgb((150, 158, 175))

ax.text(
    left,
    H - 120,
    "Mining-DRS",
    fontsize=88,
    fontweight="bold",
    color="white",
    ha="left",
    va="center",
    family="DejaVu Sans",
)
ax.text(
    left,
    H - 210,
    "Discrete Rate Simulation",
    fontsize=42,
    color=accent,
    ha="left",
    va="center",
    family="DejaVu Sans",
)

ax.text(
    left,
    H - 280,
    "Object-oriented framework for simulating continuous material-flow\n"
    "supply chains: fleets, stockpiles, and processing plants.",
    fontsize=24,
    color=gray,
    ha="left",
    va="center",
    family="DejaVu Sans",
    linespacing=1.5,
)

tags = ["Mining", "Simulation", "Supply Chain", "Monte Carlo", "Python"]
x = left
for i, tag in enumerate(tags):
    tw = len(tag) * 15 + 34
    th = 46
    if x + tw > right - 10:
        break
    tag_bg = np.array([25, 34, 54]) / 255.0
    ax.add_patch(
        plt.Rectangle(
            (x, 120),
            tw,
            th,
            facecolor=tag_bg,
            edgecolor=accent,
            linewidth=1.5,
            zorder=2,
        )
    )
    ax.text(
        x + tw / 2,
        120 + th / 2,
        tag,
        fontsize=19,
        color="white",
        ha="center",
        va="center",
        family="DejaVu Sans",
        zorder=3,
    )
    x += tw + 18

xs = np.linspace(left, right, 300)
y1 = 300 + 130 * np.sin(xs / 60)
y2 = 300 + 130 * np.sin(xs / 60 + 1.2) - 60
ax.plot(xs, y1, color=accent2, linewidth=3, alpha=0.9)
ax.plot(xs, y2, color=accent, linewidth=3, alpha=0.7)
ax.fill_between(xs, y1, y2, color=accent2, alpha=0.12)

fig.savefig("docs/assets/social-preview.png", dpi=100, facecolor=bg)
print("saved docs/assets/social-preview.png")
