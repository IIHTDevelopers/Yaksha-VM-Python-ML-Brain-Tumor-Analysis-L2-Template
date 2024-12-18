import unittest
from test.TestUtils import TestUtils
from main import load_data, split_data, create_cnn_model, train_model, evaluate_model


class BoundaryTest(unittest.TestCase):

    def setUp(self):
        self.test_obj = TestUtils()
        self.minimum_accuracy = 0.8
        self.maximum_loss = 0.5
        self.data_path = "brain_tumor_dataset"

    def test_accuracy_boundary(self):
        try:
            X, y = load_data(self.data_path)
            X = X / 255.0
            X = X.reshape(-1, 64, 64, 1)
            X_train, X_test, y_train, y_test = split_data(X, y)

            model = create_cnn_model()
            train_model(model, X_train, y_train, epochs=5, batch_size=16)
            _, accuracy = evaluate_model(model, X_test, y_test)

            is_valid = accuracy >= self.minimum_accuracy
            self.test_obj.yakshaAssert("TestAccuracyBoundary", is_valid, "boundary")
            print(f"Accuracy Boundary Test: {accuracy:.4f} → {'Passed' if is_valid else 'Failed'}")
        except Exception as e:
            print("TestAccuracyBoundary → Failed")


if __name__ == "__main__":
    unittest.main()
