import unittest
import predict_gender_ml as predict_gender


class TestGenderPrediction(unittest.TestCase):
    def test_predict_gender(self):
        result = predict_gender.predict("David")
        self.assertEqual(result.gender, "M")

        result = predict_gender.predict("Mary")
        self.assertEqual(result.gender, "F")

    def test_gender_probabilities(self):
        probs = predict_gender.predict("John").probability
        self.assertGreater(probs[0], 70)
        self.assertLess(probs[1], 30)

        probs = predict_gender.predict("Alice").probability
        self.assertLess(probs[0], 70)
        self.assertGreater(probs[1], 30)



if __name__ == "__main__":
    unittest.main()
