from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def geraGrafico(y_test, y_prev, titulo):
    cm = confusion_matrix(y_test, y_prev)
    plot_cm(cm, titulo)

def plot_cm(conf_matrix, title):
  sns.set(font_scale=1.4,color_codes=True,palette="deep")
  sns.heatmap(conf_matrix,annot=True,annot_kws={"size":16},fmt="d",cmap="YlGnBu")
  plt.title(title)
  plt.xlabel("Predicted Value")
  plt.ylabel("True Value")
  plt.show()
