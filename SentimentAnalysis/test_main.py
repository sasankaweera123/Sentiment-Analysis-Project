import unittest
import main


class TestMain(unittest.TestCase):

    def test_check_bob(self):
        text = "This is a test"
        result = main.check_bob(text)
        self.assertEqual(result, 0.0)

    def test_create_data_set(self):
        result = main.create_data_set()
        self.assertEqual(result.shape, (280, 7))

    def test_calculate_average(self):
        avg_title_comment, avg_rating = main.calculate_average([10, 10, 12], 2)
        self.assertEqual(avg_title_comment, 5)
        self.assertEqual(avg_rating, 6)

    def test_convert_sentiment_to_rating(self):
        result = main.convert_sentiment_to_rating(0)
        self.assertEqual(result, 3)

    def create_word_cloud(self):
        result = main.create_word_cloud()
        self.assertEqual(result, None)


if __name__ == '__main__':
    unittest.main()
