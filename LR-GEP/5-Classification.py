import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, f1_score

df = pd.read_csv("LR-GEP_predicted_vs_true_aqi_dynamic.csv")

def classify_aqi(aqi):
    if aqi <= 50:
        return 'Class I'
    elif aqi <= 100:
        return 'Class II'
    elif aqi <= 150:
        return 'Class III'
    elif aqi <= 200:
        return 'Class IV'
    elif aqi <= 300:
        return 'Class V'
    else:
        return 'Class VI'

df['True_Class'] = df['True AQI'].apply(classify_aqi)
df['Predicted_Class'] = df['Predicted AQI (LR-GEP)'].apply(classify_aqi)

accuracy = accuracy_score(df['True_Class'], df['Predicted_Class'])
recall = recall_score(df['True_Class'], df['Predicted_Class'], average='macro')
f1 = f1_score(df['True_Class'], df['Predicted_Class'], average='macro')

print("Classification Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall (macro): {recall:.4f}")
print(f"F1 Score (macro): {f1:.4f}\n")
print("Classification Report:")
print(classification_report(df['True_Class'], df['Predicted_Class']))

plt.figure(figsize=(8, 6))
labels = ['Class I', 'Class II', 'Class III', 'Class IV', 'Class V', 'Class VI']
cm = confusion_matrix(df['True_Class'], df['Predicted_Class'], labels=labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=labels, yticklabels=labels)
plt.title("AQI Classification Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.tight_layout()
plt.savefig("AQI_classification_confusion_matrix.png", dpi=300)
plt.show()

df.to_csv("AQI_classification_results.csv", index=False, encoding='utf-8-sig')
print("Saved classification results: AQI_classification_results.csv")
print("Confusion matrix saved as: AQI_classification_confusion_matrix.png")