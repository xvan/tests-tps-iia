import unittest

from multiply.multiply_by_three import multiply_by_three

class TestMultiplyByThree(unittest.TestCase):

    def test_multiply_by_three(self):
        pass
        self.assertEqual(multiply_by_three(3), 9)

if __name__ == '__main__':
    unittest.main()
