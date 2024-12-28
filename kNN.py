# Sweetness and crunchiness data
# Corresponding to: apple, bacon, banana, carrot, celery, cheese
X = [[10, 9], [1, 4], [10, 1], [7, 10], [3, 10], [1, 1]]
y = ['fruit', 'protein', 'fruit', 'vegetable', 'vegetable', 'protein']

from sklearn.neighbors import KNeighborsClassifier

# Create a kNN classifier with 3 neighbors
classifier = KNeighborsClassifier(n_neighbors=3)

# Fit the model with the data
classifier.fit(X, y)

# Test the model with new data
tomato = [[6, 4]]
print(classifier.predict(tomato))

carrot = [[4, 9]]
print(classifier.predict(carrot))
