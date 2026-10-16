import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

BASE_DIR = r"C:\Users\metla\.gemini\antigravity-ide\scratch\digital_agronomist_robotrek"
DATA_PATH = os.path.join(BASE_DIR, "dataset", "yield_df_training_scaled.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
feature_cols = ['Area', 'Item', 'Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
target_col = 'hg/ha_yield'
X = df[feature_cols].values.astype(np.float32)
y = df[target_col].values.astype(np.float32).reshape(-1, 1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)
print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

# Chart 1: EDA
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
corr = df.corr()
im = axes[0].imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1)
axes[0].set_xticks(range(len(corr.columns)))
axes[0].set_yticks(range(len(corr.columns)))
col_labels = ['Area', 'Item', 'Year', 'Rainfall', 'Pesticides', 'Temp', 'Yield (Target)']
axes[0].set_xticklabels(col_labels, rotation=45, ha='right', fontsize=9)
axes[0].set_yticklabels(col_labels, fontsize=9)
axes[0].set_title('Матрица корреляций признаков (Пирсон)', fontsize=12, fontweight='bold', pad=10)
fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        val = corr.iloc[i, j]
        color = 'white' if abs(val) > 0.5 else 'black'
        axes[0].text(j, i, f"{val:.2f}", ha='center', va='center', color=color, fontsize=8)

axes[1].hist(df[target_col], bins=50, color='#2E7D32', edgecolor='white', alpha=0.85)
axes[1].axvline(df[target_col].mean(), color='#D32F2F', linestyle='--', linewidth=2, label=f"Среднее: {df[target_col].mean():.2f}")
axes[1].axvline(df[target_col].median(), color='#1976D2', linestyle=':', linewidth=2, label=f"Медиана: {df[target_col].median():.2f}")
axes[1].set_title('Распределение целевой переменной (hg/ha_yield)', fontsize=12, fontweight='bold', pad=10)
axes[1].set_xlabel('Масштабированная урожайность')
axes[1].set_ylabel('Количество наблюдений')
axes[1].legend(loc='upper right', frameon=True)

plt.tight_layout()
eda_path = os.path.join(CHARTS_DIR, "eda_correlations_distribution.png")
plt.savefig(eda_path)
plt.close()
print(f"Saved EDA chart to {eda_path}")

# Models
class BaselineMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

class DeepRegularizedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

class BatchNormMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

class OptimalDeepAgronomistNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.02),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.02),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.02),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

def train_model(model, train_loader, val_loader, epochs=50, lr=0.003):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_rmse': [], 'val_r2': []}
    best_val_rmse = float('inf')
    best_weights = None
    
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(bx)
        
        train_epoch_loss = running_loss / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                preds = model(bx)
                loss = criterion(preds, by)
                val_loss += loss.item() * len(bx)
                val_preds.append(preds.cpu().numpy())
                val_targets.append(by.cpu().numpy())
        
        scheduler.step()
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_preds = np.vstack(val_preds).flatten()
        val_targets = np.vstack(val_targets).flatten()
        
        r2 = r2_score(val_targets, val_preds)
        mae = mean_absolute_error(val_targets, val_preds)
        rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        
        history['train_loss'].append(train_epoch_loss)
        history['val_loss'].append(val_epoch_loss)
        history['val_mae'].append(mae)
        history['val_rmse'].append(rmse)
        history['val_r2'].append(r2)
        
        if rmse < best_val_rmse:
            best_val_rmse = rmse
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    total_time = time.time() - start_time
    model.load_state_dict(best_weights)
    
    model.eval()
    with torch.no_grad():
        final_preds = model(torch.from_numpy(X_val).to(device)).cpu().numpy().flatten()
    
    return {
        'model': model,
        'history': history,
        'time': total_time,
        'r2': r2_score(y_val.flatten(), final_preds),
        'mae': mean_absolute_error(y_val.flatten(), final_preds),
        'rmse': np.sqrt(mean_squared_error(y_val.flatten(), final_preds)),
        'preds': final_preds
    }

experiments = [
    {
        'id': 'Exp 1',
        'name': 'Базовый 2-слойный MLP (64-32)',
        'model': BaselineMLP(),
        'batch_size': 128,
        'epochs': 35,
        'lr': 0.003,
        'arch_desc': 'Dense(64) -> ReLU -> Dense(32) -> ReLU -> Dense(1)'
    },
    {
        'id': 'Exp 2',
        'name': 'Глубокий MLP с Dropout (128-64-32)',
        'model': DeepRegularizedMLP(),
        'batch_size': 64,
        'epochs': 40,
        'lr': 0.002,
        'arch_desc': 'Dense(128) -> ReLU -> Drop(0.2) -> Dense(64) -> ReLU -> Drop(0.2) -> Dense(32) -> ReLU -> Dense(1)'
    },
    {
        'id': 'Exp 3',
        'name': 'MLP с BatchNorm (128-64-32)',
        'model': BatchNormMLP(),
        'batch_size': 64,
        'epochs': 45,
        'lr': 0.002,
        'arch_desc': 'Dense(128) -> BN -> ReLU -> Drop(0.05) -> Dense(64) -> BN -> ReLU -> Drop(0.05) -> Dense(32) -> BN -> ReLU -> Dense(1)'
    },
    {
        'id': 'Exp 4',
        'name': 'Финальная Deep NNTrack Net (256-256-128-64)',
        'model': OptimalDeepAgronomistNet(),
        'batch_size': 64,
        'epochs': 60,
        'lr': 0.003,
        'arch_desc': 'Dense(256) -> BN -> ReLU -> Drop(0.02) -> Dense(256) -> BN -> ReLU -> Drop(0.02) -> Dense(128) -> BN -> ReLU -> Dense(64) -> BN -> ReLU -> Dense(1)'
    }
]

results = []
for exp in experiments:
    print(f"\n--- Running {exp['id']}: {exp['name']} ---")
    train_loader = DataLoader(train_ds, batch_size=exp['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=exp['batch_size'], shuffle=False)
    res = train_model(exp['model'], train_loader, val_loader, epochs=exp['epochs'], lr=exp['lr'])
    exp['result'] = res
    score_r2 = max(0.0, min(40.0, 40.0 * res['r2']))
    print(f"Results: R2={res['r2']:.4f}, Score_R2={score_r2:.2f}/40, MAE={res['mae']:.4f}, RMSE={res['rmse']:.4f}, Time={res['time']:.1f}s")
    results.append({
        'Эксперимент': exp['id'],
        'Архитектура': exp['name'],
        'Слои и блоки': exp['arch_desc'],
        'Размер партии': exp['batch_size'],
        'Эпохи': exp['epochs'],
        'R² (детерминация)': round(float(res['r2']), 4),
        'Балл за R² (40*R²)': round(float(score_r2), 2),
        'MAE': round(float(res['mae']), 4),
        'RMSE': round(float(res['rmse']), 4),
        'Время обучения (с)': round(float(res['time']), 1)
    })

table_df = pd.DataFrame(results)
table_path = os.path.join(OUTPUT_DIR, 'experiments_table.csv')
table_df.to_csv(table_path, index=False, encoding='utf-8-sig')
print(f"Saved experiments table to {table_path}")

# Chart 2: Loss Convergence
plt.figure(figsize=(11, 5.5), dpi=300)
colors = ['#1E88E5', '#FB8C00', '#43A047', '#E53935']
for i, exp in enumerate(experiments):
    plt.plot(exp['result']['history']['val_rmse'], label=f"{exp['id']}: {exp['name']} (min RMSE={exp['result']['rmse']:.3f})", color=colors[i], linewidth=2.2)
plt.title('Сравнение динамики валидационной ошибки (RMSE) по эпохам', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Эпоха обучения')
plt.ylabel('RMSE на валидации')
plt.legend(frameon=True, fontsize=10)
plt.tight_layout()
conv_path = os.path.join(CHARTS_DIR, 'convergence_comparison.png')
plt.savefig(conv_path)
plt.close()
print(f"Saved convergence chart to {conv_path}")

# Chart 3: Diagnostics for winning model (Exp 4)
best_exp = experiments[3]
best_preds = best_exp['result']['preds']
actuals = y_val.flatten()

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
axes[0].scatter(actuals, best_preds, alpha=0.35, color='#2E7D32', s=16, edgecolors='none')
ideal = np.linspace(min(actuals), max(actuals), 100)
axes[0].plot(ideal, ideal, color='#D32F2F', linestyle='--', linewidth=2.5, label='Идеальный прогноз (y = x)')
axes[0].set_title(f"Фактическая vs Предсказанная урожайность (R² = {best_exp['result']['r2']:.4f})", fontsize=12, fontweight='bold', pad=10)
axes[0].set_xlabel('Фактическая урожайность (hg/ha_yield)')
axes[0].set_ylabel('Прогноз модели (hg/ha_yield)')
axes[0].legend(loc='upper left', frameon=True)

residuals = actuals - best_preds
axes[1].hist(residuals, bins=60, color='#1976D2', edgecolor='white', alpha=0.85, density=True)
axes[1].axvline(0, color='#D32F2F', linestyle='--', linewidth=2)
axes[1].set_title(f"Распределение ошибок (Residuals: MAE={best_exp['result']['mae']:.4f}, RMSE={best_exp['result']['rmse']:.4f})", fontsize=12, fontweight='bold', pad=10)
axes[1].set_xlabel('Ошибка прогноза (Фактическое - Прогноз)')
axes[1].set_ylabel('Плотность вероятности')

plt.tight_layout()
diag_path = os.path.join(CHARTS_DIR, 'final_model_diagnostics.png')
plt.savefig(diag_path)
plt.close()
print(f"Saved diagnostics chart to {diag_path}")

# Chart 4: Residuals vs fitted & metrics bar
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
axes[0].scatter(best_preds, residuals, alpha=0.3, color='#8E24AA', s=14)
axes[0].axhline(0, color='#D32F2F', linestyle='--', linewidth=2)
axes[0].set_title('График остатков против предсказанных значений', fontsize=12, fontweight='bold', pad=10)
axes[0].set_xlabel('Предсказанные значения')
axes[0].set_ylabel('Остатки (Residuals)')

metrics_names = ['R² (детерминация)', 'MAE (ошибка)', 'RMSE (ср.-кв. ошибка)']
x_pos = np.arange(len(metrics_names))
w = 0.2
for i, exp in enumerate(experiments):
    vals = [exp['result']['r2'], exp['result']['mae'], exp['result']['rmse']]
    axes[1].bar(x_pos + i*w, vals, width=w, label=exp['id'], color=colors[i])
axes[1].set_xticks(x_pos + 1.5*w)
axes[1].set_xticklabels(metrics_names)
axes[1].set_title('Сравнение ключевых метрик по экспериментам', fontsize=12, fontweight='bold', pad=10)
axes[1].legend(loc='upper right', frameon=True)

plt.tight_layout()
bar_path = os.path.join(CHARTS_DIR, 'residuals_and_metrics_bar.png')
plt.savefig(bar_path)
plt.close()
print(f"Saved bar chart to {bar_path}")

# Export TorchScript
best_model = best_exp['model']
best_model.eval()
dummy_input = torch.randn(1, 6, dtype=torch.float32).to(device)
traced = torch.jit.trace(best_model, dummy_input)
torchscript_path = os.path.join(OUTPUT_DIR, 'final_model_torchscript.pt')
traced.save(torchscript_path)
print(f"[OK] Saved TorchScript model to {torchscript_path}")

# Verify
loaded = torch.jit.load(torchscript_path)
loaded.eval()
test_sample = torch.from_numpy(X_val[:10]).to(device)
with torch.no_grad():
    diff = np.max(np.abs(best_model(test_sample).cpu().numpy() - loaded(test_sample).cpu().numpy()))
print(f"[OK] Verification max diff: {diff}")
assert diff < 1e-5

# Summary json
summary = {
    'task': 'Цифровой агроном: прогноз урожайности с помощью ИИ',
    'nomination': 'Большие данные и искусственный интеллект',
    'partner': 'Роботрек',
    'model_name': 'OptimalDeepAgronomistNet',
    'architecture': 'Dense(256) -> BN -> ReLU -> Drop(0.02) -> Dense(256) -> BN -> ReLU -> Drop(0.02) -> Dense(128) -> BN -> ReLU -> Dense(64) -> BN -> ReLU -> Dense(1)',
    'input_features': feature_cols,
    'target': target_col,
    'samples_total': int(len(df)),
    'train_samples': int(len(X_train)),
    'val_samples': int(len(X_val)),
    'metrics': {
        'R2': round(float(best_exp['result']['r2']), 4),
        'score_R2_points': round(float(max(0.0, min(40.0, 40.0 * best_exp['result']['r2']))), 2),
        'MAE': round(float(best_exp['result']['mae']), 4),
        'RMSE': round(float(best_exp['result']['rmse']), 4)
    },
    'experiments': results
}
with open(os.path.join(OUTPUT_DIR, 'evaluation_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("[OK] Training and evaluation complete!")
