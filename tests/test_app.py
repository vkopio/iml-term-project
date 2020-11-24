import unittest
from src.app import app


class TestMain(unittest.TestCase):
    def test_main(self):
        self.assertEqual('Hello, World!', app())


if __name__ == '__main__':
    unittest.main()
