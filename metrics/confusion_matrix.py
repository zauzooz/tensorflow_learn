from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

labels = ['b', 'a', 'd', 'c', 'e']

# Example data
y_true = ['a', 'b', 'a', 'd', 'b', 'c', 'e']
y_pred = ['a', 'b', 'c', 'c', 'b', 'c', 'c']

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=labels)
print(cm)

# Visualize the confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
