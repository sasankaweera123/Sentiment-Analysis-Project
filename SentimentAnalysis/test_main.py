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
        avg_TC, avg_R = main.calculate_average(10, 10, 12, 2)
        self.assertEqual(avg_TC, 5)
        self.assertEqual(avg_R, 6)


if __name__ == '__main__':
    unittest.main()
