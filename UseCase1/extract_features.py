import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Flatten

trainData_path = "Data/train.p"
testData_path = "Data/test.p"
testDataBright_path = "Data/test_bright.p"

(X_train, y_train) = pickle.load(open(trainData_path, "rb"))
(X_test, y_test) = pickle.load(open(testData_path, "rb"))
(X_test_bright, _) = pickle.load(open(testDataBright_path, "rb"))

X_train = X_train / 255.0
X_test = X_test / 255.0
X_test_bright = X_test_bright / 255.0

classifier_encoder_path = "Models/classifier_encoder"
AE_encoder_path = "Models/AE_encoder"

classifier_encoder = load_model(classifier_encoder_path)
AE_encoder = load_model(AE_encoder_path)

train_features_AE = Flatten()(AE_encoder.predict(X_train)).numpy()
train_features_classifier = Flatten()(classifier_encoder.predict(X_train)).numpy()
save_path = "./Data/train_features_ae.p"
pickle.dump(train_features_AE, open(save_path, "wb"))
save_path = "./Data/train_features_classifier.p"
pickle.dump(train_features_classifier, open(save_path, "wb"))

test_features_AE = Flatten()(AE_encoder.predict(X_test)).numpy()
test_features_classifier = Flatten()(classifier_encoder.predict(X_test)).numpy()
save_path = "./Data/test_features_ae.p"
pickle.dump(test_features_AE, open(save_path, "wb"))
save_path = "./Data/test_features_classifier.p"
pickle.dump(test_features_classifier, open(save_path, "wb"))

testBright_features_AE = Flatten()(AE_encoder.predict(X_test_bright)).numpy()
testBright_features_classifier = Flatten()(classifier_encoder.predict(X_test_bright)).numpy()
save_path = "./Data/test_bright_features_ae.p"
pickle.dump(testBright_features_AE, open(save_path, "wb"))
save_path = "./Data/test_bright_features_classifier.p"
pickle.dump(testBright_features_classifier, open(save_path, "wb"))
