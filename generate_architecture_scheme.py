import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

BASE_DIR = r"C:\Users\metla\.gemini\antigravity-ide\scratch\digital_agronomist_robotrek"
CHARTS_DIR = os.path.join(BASE_DIR, "output", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
ax.set_xlim(0, 14.5)
ax.set_ylim(0, 6)
ax.axis('off')

# Layer blocks
blocks = [
    {"name": "Входной слой\nInput (6 фичей)", "sub": "Area, Item, Year,\nRainfall, Pesticides, Temp\n(нормализованные)", "color": "#1565C0", "x": 0.4, "w": 2.0},
    {"name": "Dense Блок 1\n(256 нейронов)", "sub": "Linear(6 -> 256)\nBatchNorm1d + ReLU\nDropout (p = 0.02)", "color": "#1B5E20", "x": 2.7, "w": 2.1},
    {"name": "Dense Блок 2\n(256 нейронов)", "sub": "Linear(256 -> 256)\nBatchNorm1d + ReLU\nDropout (p = 0.02)", "color": "#2E7D32", "x": 5.1, "w": 2.1},
    {"name": "Dense Блок 3\n(128 нейронов)", "sub": "Linear(256 -> 128)\nBatchNorm1d + ReLU\nDropout (p = 0.02)", "color": "#388E3C", "x": 7.5, "w": 2.1},
    {"name": "Dense Блок 4\n(64 нейрона)", "sub": "Linear(128 -> 64)\nBatchNorm1d + ReLU\n(компрессия фичей)", "color": "#43A047", "x": 9.9, "w": 2.1},
    {"name": "Выходной слой\nDense (1 нейрон)", "sub": "Linear(64 -> 1)\nЛинейная регрессия\n(Требование NNTrack)", "color": "#C62828", "x": 12.3, "w": 1.8},
]

y_pos = 1.45
h_box = 3.3

for b in blocks:
    # Outer container
    rect = patches.FancyBboxPatch(
        (b["x"], y_pos), b["w"], h_box,
        boxstyle="round,pad=0.12",
        edgecolor=b["color"],
        facecolor=b["color"],
        alpha=0.10,
        linewidth=2.2
    )
    ax.add_patch(rect)
    
    # Title bar
    header_rect = patches.FancyBboxPatch(
        (b["x"], y_pos + h_box - 1.15), b["w"], 1.15,
        boxstyle="round,pad=0.08",
        edgecolor="none",
        facecolor=b["color"],
        alpha=0.92
    )
    ax.add_patch(header_rect)
    
    # Header text
    ax.text(b["x"] + b["w"]/2, y_pos + h_box - 0.58, b["name"],
            ha='center', va='center', color='white', fontsize=8.8, fontweight='bold')
    
    # Body text
    ax.text(b["x"] + b["w"]/2, y_pos + 0.95, b["sub"],
            ha='center', va='center', color='#1A237E', fontsize=8.3, fontweight='medium')

# Arrows
for i in range(len(blocks) - 1):
    x_start = blocks[i]["x"] + blocks[i]["w"] + 0.04
    x_end = blocks[i+1]["x"] - 0.04
    y_mid = y_pos + h_box / 2
    ax.annotate(
        "", xy=(x_end, y_mid), xytext=(x_start, y_mid),
        arrowprops=dict(arrowstyle="-|>", color="#37474F", lw=2.2, mutation_scale=16)
    )

# Title & Subtitle
ax.text(7.25, 5.55, "Архитектура финальной нейронной сети в среде NNTrack",
        ha='center', va='center', fontsize=14, fontweight='bold', color='#0D47A1')
ax.text(7.25, 5.12, "Полносвязная глубокая сеть: Dense(256)-Dense(256)-Dense(128)-Dense(64)-Dense(1) с BatchNorm и регуляризацией",
        ha='center', va='center', fontsize=10.5, fontstyle='italic', color='#455A64')

# Metrics box at bottom
ax.text(7.25, 0.58, "Метрики на валидации (20%): R² = 0.9654 (38.62/40 баллов) | MAE = 0.0968 | RMSE = 0.1573 | Экспорт: TorchScript (.pt)",
        ha='center', va='center', fontsize=10.5, fontweight='bold', color='#1B5E20',
        bbox=dict(boxstyle='round,pad=0.55', facecolor='#E8F5E9', edgecolor='#4CAF50', lw=1.8))

plt.tight_layout()
arch_chart = os.path.join(CHARTS_DIR, "nntrack_architecture_scheme.png")
plt.savefig(arch_chart)
plt.close()
print(f"Saved updated architecture scheme to {arch_chart}")
