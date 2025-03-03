import matplotlib.pyplot as plt
import numpy as np

# Data metrik
models = ['Naïve Bayes', 'BERT']
accuracy = [0.6878, 0.9688]
precision = [0.74, 0.97]
recall = [0.69, 0.97]
f1_score = [0.60, 0.97]

# Plotting
x = np.arange(len(models))  # Posisi label
width = 0.2  # Lebar bar

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, accuracy, width, label='Accuracy')
rects2 = ax.bar(x, precision, width, label='Precision')
rects3 = ax.bar(x + width, recall, width, label='Recall')
rects4 = ax.bar(x + 2*width, f1_score, width, label='F1-Score')

# Tambahkan label, judul, dan legenda
ax.set_xlabel('Model')
ax.set_ylabel('Score')
ax.set_title('Perbandingan Metrik Naïve Bayes dan BERT')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# Tambahkan nilai di atas setiap bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

plt.tight_layout()
plt.show()