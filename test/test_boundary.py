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
        """Test if the model's accuracy meets the minimum threshold."""
        try:
            # Load and preprocess data
            X, y = load_data(self.data_path)
            X = X / 255.0
            X = X.reshape(-1, 64, 64, 1)
            X_train, X_test, y_train, y_test = split_data(X, y)

            # Create and train the model
            model = create_cnn_model()
            train_model(model, X_train, y_train, epochs=5, batch_size=16)

            # Evaluate the model's performance
            _, accuracy = evaluate_model(model, X_test, y_test)

            # Validate accuracy boundary
            if accuracy >= self.minimum_accuracy:
                self.test_obj.yakshaAssert("TestAccuracyBoundary", True, "boundary")
                print(f"TestAccuracyBoundary: Passed (Accuracy: {accuracy:.4f})")
            else:
                self.test_obj.yakshaAssert("TestAccuracyBoundary", False, "boundary")
                print(f"TestAccuracyBoundary: Failed (Accuracy: {accuracy:.4f})")

        except FileNotFoundError as e:
            self.test_obj.yakshaAssert("TestAccuracyBoundary", False, "boundary")
            print(f"TestAccuracyBoundary: Failed (FileNotFoundError:) ")
        except ValueError as e:
            self.test_obj.yakshaAssert("TestAccuracyBoundary", False, "boundary")
            print(f"TestAccuracyBoundary: Failed (ValueError:)")
        except Exception as e:
            self.test_obj.yakshaAssert("TestAccuracyBoundary", False, "boundary")
            print(f"TestAccuracyBoundary: Failed (Exception:) ")


if __name__ == "__main__":
    unittest.main()
