import numpy as np


class KnnClassifier:
    """
    K-Nearest Neighbors classifier.
    Follows the class structure from Lab 3.
    """

    def __init__(self, train_images, train_labels):
        """
        Constructor stores training data.
        train_images : numpy array of shape (num_train, num_features)
        train_labels : numpy array of shape (num_train,), values 1-5
        """
        self.train_images = train_images
        self.train_labels = train_labels

    def classify_image(self, test_image, num_neighbors=3, metric='l2'):
        """
        Classify a single test image using K-Nearest Neighbors.

        test_image   : 1D numpy array of shape (num_features,)
        num_neighbors: number of nearest neighbors to use (K)
        metric       : 'l2' for Euclidean distance, 'l1' for Manhattan distance

        Returns: predicted label (int, 1-5)
        """
        test_image = test_image.ravel()

        # compute distance from test_image to every training sample
        if metric == 'l2':
            # L2(X, Y) = sqrt(sum((Xi - Yi)^2))
            distances = np.sqrt(np.sum((self.train_images - test_image) ** 2, axis=1))
        elif metric == 'l1':
            # L1(X, Y) = sum(|Xi - Yi|)
            distances = np.sum(np.abs(self.train_images - test_image), axis=1)
        elif metric == 'chi2':
            # Chi-square distance — standard metric for histogram comparison:
            # chi2(h1, h2) = sum((h1_i - h2_i)^2 / (h1_i + h2_i + eps))
            # Designed for probability distributions; penalizes differences
            # in bins with low counts less than those with high counts.
            eps = 1e-10
            diff = self.train_images - test_image
            summ = self.train_images + test_image + eps
            distances = np.sum((diff ** 2) / summ, axis=1)
        else:
            raise ValueError(f"Unknown metric '{metric}'. Use 'l1', 'l2', or 'chi2'.")

        # get indices that would sort the distances (ascending)
        sorted_indices = np.argsort(distances)

        # take labels of the K nearest neighbors
        nearest_labels = self.train_labels[sorted_indices[:num_neighbors]]

        # majority vote: labels are 1-5, np.bincount counts occurrences at each index
        counts = np.bincount(nearest_labels)
        predicted_label = np.argmax(counts)

        return predicted_label

    def classify_batch(self, test_images, num_neighbors=3, metric='l2'):
        """
        Classify a batch of test images by calling classify_image for each one.

        test_images : numpy array of shape (num_test, num_features)
        Returns     : numpy array of predictions of shape (num_test,)
        """
        num_test = test_images.shape[0]
        predictions = np.zeros(num_test, dtype=np.int32)

        for i in range(num_test):
            if (i + 1) % 200 == 0:
                print(f'    KNN classified {i + 1}/{num_test}')
            predictions[i] = self.classify_image(
                test_images[i], num_neighbors, metric
            )

        return predictions


def confusion_matrix(y_true, y_pred):
    """
    Compute the confusion matrix.
    C[i][j] = number of examples from class i predicted as class j.
    Expects 1-indexed labels (1 to num_classes).

    From Lab 2 convention.
    """
    y_true = np.array(y_true, dtype=np.int32) - 1  # convert to 0-indexed
    y_pred = np.array(y_pred, dtype=np.int32) - 1

    num_classes = int(y_true.max()) + 1
    C = np.zeros((num_classes, num_classes), dtype=np.int32)

    for t, p in zip(y_true, y_pred):
        C[t][p] += 1

    return C


def compute_accuracy(y_true, y_pred):
    """
    Compute classification accuracy:
        accuracy = number of correct predictions / total predictions
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sum(y_true == y_pred) / len(y_true)
