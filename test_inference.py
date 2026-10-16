"""
Скрипт проверки инференса экспортированной TorchScript модели
Всероссийский хакатон «ИТ-Прорыв» 2026 | Номинация «Большие данные и ИИ»
Кейс «Цифровой агроном: прогноз урожайности с помощью ИИ» (Партнер: «Роботрек»)
Команда: «АгроИнтеллект»
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Гарантия корректной кодировки в консоли Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

FEATURE_COLS = [
    'Area',
    'Item',
    'Year',
    'average_rain_fall_mm_per_year',
    'pesticides_tonnes',
    'avg_temp'
]
TARGET_COL = 'hg/ha_yield'

def evaluate(model_path, data_path, test_split=0.2, seed=42):
    print("=" * 68)
    print(" Всероссийский хакатон «ИТ-Прорыв» — Кейс «Цифровой агроном»")
    print(" Проверка экспортированной нейросетевой модели TorchScript")
    print("=" * 68)

    if not os.path.exists(model_path):
        print(f"[-] Ошибка: файл модели не найден: {model_path}")
        sys.exit(1)

    if not os.path.exists(data_path):
        print(f"[-] Ошибка: файл датасета не найден: {data_path}")
        sys.exit(1)

    print(f"[*] Загрузка TorchScript модели: {model_path}")
    try:
        model = torch.jit.load(model_path, map_location=torch.device('cpu'))
        model.eval()
        print("[+] Модель успешно загружена и переведена в режим eval()")
    except Exception as e:
        print(f"[-] Ошибка при загрузке модели: {e}")
        sys.exit(1)

    print(f"[*] Загрузка данных: {data_path}")
    df = pd.read_csv(data_path)
    print(f"[+] Размерность набора данных: {df.shape[0]} строк, {df.shape[1]} колонок")

    for col in FEATURE_COLS:
        if col not in df.columns:
            print(f"[-] Ошибка: отсутствует обязательный признак '{col}'")
            sys.exit(1)

    X = df[FEATURE_COLS].values.astype(np.float32)

    has_target = TARGET_COL in df.columns
    if has_target:
        y = df[TARGET_COL].values.astype(np.float32).reshape(-1, 1)
        if test_split > 0 and len(df) > 1000:
            print(f"[*] Выделение валидационной выборки: {int(test_split*100)}% (seed={seed})")
            _, X_eval, _, y_eval = train_test_split(X, y, test_size=test_split, random_state=seed)
        else:
            print("[*] Валидация на полном предоставленном наборе данных")
            X_eval, y_eval = X, y
    else:
        print("[!] Целевая колонка отсутствует (режим инференса на закрытом тесте без ответов)")
        X_eval = X
        y_eval = None

    print(f"[*] Выполнение инференса на {len(X_eval)} наблюдениях...")
    with torch.no_grad():
        inputs = torch.from_numpy(X_eval)
        predictions = model(inputs).numpy().flatten()

    print("[+] Инференс завершен успешно!")

    if y_eval is not None:
        y_true = y_eval.flatten()
        r2 = r2_score(y_true, predictions)
        mae = mean_absolute_error(y_true, predictions)
        rmse = np.sqrt(mean_squared_error(y_true, predictions))
        score_r2 = max(0.0, min(40.0, 40.0 * r2))

        print("-" * 68)
        print(" ИТОГОВЫЕ СОРЕВНОВАТЕЛЬНЫЕ МЕТРИКИ:")
        print(f"  -> Коэффициент детерминации (R^2):    {r2:.4f}")
        print(f"  -> Соревновательный балл за R^2:       {score_r2:.2f} / 40.00 баллов")
        print(f"  -> Средняя абсолютная ошибка (MAE):    {mae:.4f}")
        print(f"  -> Среднеквадратичная ошибка (RMSE):   {rmse:.4f}")
        print("-" * 68)

        print("\nПримеры сопоставления фактических и предсказанных значений (первые 5):")
        for i in range(min(5, len(y_true))):
            diff = abs(y_true[i] - predictions[i])
            print(f"  [{i+1}] Истина: {y_true[i]:8.4f} | Прогноз: {predictions[i]:8.4f} | Разница: {diff:8.4f}")
    else:
        print("\nПервые 5 предсказанных значений:")
        for i in range(min(5, len(predictions))):
            print(f"  [{i+1}] Прогноз: {predictions[i]:.4f}")

    print("=" * 68)
    print(" Проверка завершена успешно. Модель готова к соревновательной оценке.")
    print("=" * 68)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TorchScript Model Inference & Validation")
    parser.add_argument("--model", type=str, default="final_model_torchscript.pt", help="Путь к файлу модели (.pt)")
    parser.add_argument("--data", type=str, default="yield_df_training_scaled.csv", help="Путь к датасету (.csv)")
    parser.add_argument("--split", type=float, default=0.2, help="Доля валидационной выборки (0.2 по умолчанию)")
    parser.add_argument("--seed", type=int, default=42, help="Случайное зерно для воспроизводимости")
    args = parser.parse_args()

    evaluate(args.model, args.data, test_split=args.split, seed=args.seed)
