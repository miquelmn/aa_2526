import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from Perceptron import Perceptron

# Generació del conjunt de mostres
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=1.25,
                           random_state=0)

y[y == 0] = -1  # La nostra implementació esta pensada per tenir les classes 1 i -1.

# Entrenament
perceptron = Perceptron(eta=0.001, n_iter=20)
perceptron.fit(X, y)
y_prediction = perceptron.predict(X)

#  Mostram els resultats
plt.figure(1)
# Dibuixem el núvol de punts (el paràmetre c indica que pintem segons la classe)
plt.scatter(X[:, 0], X[:, 1], c=y)

# Dibuixem la recta. Usem l'equació punt-pendent
m = -perceptron.w_[1] / perceptron.w_[2]
origen = (0, -perceptron.w_[0] / perceptron.w_[2])
plt.axline(xy1=origen, slope=m)


# Extra: Dibuixam el nombre d'errors en cada iteracio de l'algorisme

plt.figure(2)
plt.plot(perceptron.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of miss classifications')
plt.show()

