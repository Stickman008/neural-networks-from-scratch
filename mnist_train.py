from tensorflow import keras

# Download MINST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Normalize image data
def preprocess_img(img):
    return img / 255.0

X_train = preprocess_img(X_train)
X_test = preprocess_img(X_test)

