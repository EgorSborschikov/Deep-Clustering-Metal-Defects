import matplotlib.pyplot as plt
import json

def plot_results(results_path, save_path):

    # Загрузка результатов
    with open('metal_defects_results/results.json', 'r') as f:
        data = json.load(f)

    # Извлекаем пары (ключ_как_строка, значение) и сортируем по числовому значению
    items = []
    for key_str, metrics in data.items():
        eta_val = float(key_str.strip())  # Убираем пробелы и конвертируем в float
        items.append((eta_val, key_str, metrics))

    # Сортируем по числовому значению η
    items.sort(key=lambda x: x[0])

    # Теперь извлекаем данные в правильном порядке
    etas = [item[0] for item in items]
    nmi_vals = [item[2]["nmi"] for item in items]
    ari_vals = [item[2]["ari"] for item in items]
    entropy_vals = [item[2]["entropy"] for item in items]

    # Далее — ваш код визуализации (без изменений)
    plt.rcParams.update({'font.size': 12})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. NMI vs η
    axes[0, 0].plot(etas, nmi_vals, 'o-', color='tab:blue', linewidth=2, markersize=8, label='NMI')
    axes[0, 0].plot(etas, ari_vals, 's--', color='tab:orange', linewidth=2, markersize=6, label='ARI')
    axes[0, 0].set_xlabel('η (коэффициент энтропийной регуляризации)')
    axes[0, 0].set_ylabel('Метрика качества')
    axes[0, 0].set_title('Влияние η на качество кластеризации')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Энтропия vs η
    axes[0, 1].plot(etas, entropy_vals, '^-', color='tab:green', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('η')
    axes[0, 1].set_ylabel('H(γ) — средняя энтропия')
    axes[0, 1].set_title('Влияние η на неопределённость присваиваний')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Компромисс: NMI vs H(γ)
    scatter = axes[1, 0].scatter(entropy_vals, nmi_vals, c=etas, cmap='viridis', s=120, edgecolor='k')
    for i, eta in enumerate(etas):
        axes[1, 0].annotate(f'η={eta}', (entropy_vals[i], nmi_vals[i]),
                            textcoords="offset points", xytext=(0,8), ha='center', fontsize=10)
    axes[1, 0].set_xlabel('H(γ) — энтропия')
    axes[1, 0].set_ylabel('NMI — качество')
    axes[1, 0].set_title('Компромисс: качество ↔ мягкость')
    plt.colorbar(scatter, ax=axes[1, 0], label='η')

    # 4. Относительное изменение (относительно η=0.0)
    base_nmi = nmi_vals[0]
    base_entropy = entropy_vals[0]
    rel_nmi = [(nmi - base_nmi) / base_nmi * 100 for nmi in nmi_vals]
    rel_entropy = [(ent - base_entropy) / base_entropy * 100 for ent in entropy_vals]

    axes[1, 1].plot(etas, rel_nmi, 'o-', color='tab:red', linewidth=2, markersize=8, label='ΔNMI, %')
    axes[1, 1].plot(etas, rel_entropy, '^-', color='tab:purple', linewidth=2, markersize=8, label='ΔH(γ), %')
    axes[1, 1].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axes[1, 1].set_xlabel('η')
    axes[1, 1].set_ylabel('Относительное изменение, %')
    axes[1, 1].set_title('Изменение метрик относительно базовой модели (η=0.0)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('metal_defects_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()