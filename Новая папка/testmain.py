import unittest
import main
from numpy import array_equal


class TestCountRisk(unittest.TestCase):
    """ Тестирование подсчета матрицы рисков """

    def test_value_risk_matrix_1(self):
        PayMatrix = [
            [5, 4, 5, 2],
            [9, 9, 4, 3],
            [9, 6, 1, 8]
        ]

        RiskMatrix = [
            [4, 5, 0, 6],
            [0, 0, 1, 5],
            [0, 3, 4, 0]
        ]

        ResultMatrix = main.count_risk(PayMatrix)

        assert array_equal(ResultMatrix, RiskMatrix) == True

    def test_value_risk_matrix_2(self):
        PayMatrix = [
            [3, 3, 4, 9, 1],
            [5, 2, 2, 1, 1],
            [1, 2, 5, 4, 3],
            [7, 6, 9, 1, 1],
            [5, 2, 3, 8, 3]
        ]

        RiskMatrix = [
            [4, 3, 5, 0, 2],
            [2, 4, 7, 8, 2],
            [6, 4, 4, 5, 0],
            [0, 0, 0, 8, 2],
            [2, 4, 6, 1, 0]
        ]

        ResultMatrix = main.count_risk(PayMatrix)

        assert array_equal(ResultMatrix, RiskMatrix) == True


class TestProbability(unittest.TestCase):
    """ Тестирование решения, полученного из критерия, основанного на известных вероятностях условиях """

    def test_on_probaility_1(self):
        RiskMatrix = [
            [2, 2, 1, 9],
            [1, 8, 2, 6],
            [6, 2, 7, 5]
        ]

        right_result_list = [2.4, 4.6, 4.6]

        probability_list = [0.2, 0.4, 0.3, 0.1]

        result_list = main.with_probability(RiskMatrix, probability_list)

        assert array_equal(result_list, right_result_list) == True

    def test_on_probaility_2(self):
        RiskMatrix = [
            [2, 6, 6, 2, 1],
            [2, 9, 5, 4, 2],
            [0, 4, 9, 0, 4],
            [0, 2, 0, 7, 2],
            [6, 1, 4, 7, 0]
        ]

        right_result_list = [3.4, 4.4, 3.4, 2.2, 3.6]

        probability_list = [0.2, 0.2, 0.2, 0.2, 0.2]

        result_list = main.with_probability(RiskMatrix, probability_list)

        assert array_equal(result_list, right_result_list) == True


class TestWald(unittest.TestCase):
    """ Тестирование решения, полученного из критерия Вальда """

    def test_criterion_Wald_1(self):
        MatrixPay = [
            [8, 2, 6, 8],
            [3, 2, 4, 3],
            [4, 1, 6, 1]
        ]

        right_result_list = [2, 2, 1]

        result_list = main.Wald(MatrixPay)

        assert array_equal(result_list, right_result_list) == True

    def test_criterion_Wald_2(self):
        MatrixPay = [
            [5, 9, 1, 2, 2],
            [9, 3, 2, 5, 1],
            [1, 2, 5, 2, 8],
            [7, 3, 7, 3, 5],
            [7, 6, 3, 8, 3]
        ]

        right_result_list = [1, 1, 1, 3, 3]

        result_list = main.Wald(MatrixPay)

        assert array_equal(result_list, right_result_list) == True


class TestSavage(unittest.TestCase):
    """ Тестирование решения, полученного из критерия Сэвиджа """

    def test_criterion_Savage_1(self):
        RiskMatrix = [
            [2, 4, 1, 1],
            [0, 1, 7, 6],
            [0, 2, 9, 7]
        ]

        right_result_list = [4, 7, 9]

        result_list = main.Savage(RiskMatrix)

        assert array_equal(result_list, right_result_list) == True

    def test_criterion_Savage_2(self):
        RiskMatrix = [
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4],
            [2, 3, 4, 5, 6]
        ]

        right_result_list = [1, 2, 3, 4, 6]

        result_list = main.Savage(RiskMatrix)

        assert array_equal(result_list, right_result_list) == True


class TestHurwitzMatrix(unittest.TestCase):
    """ Тестирование решения, полученного из критерия Гурвица, основанном на выигрыше """

    def test_criterion_Hurwitz_pay_1(self):
        MatrixPay = [
            [1, 1, 5, 7],
            [9, 5, 4, 0],
            [4, 6, 1, 2]
        ]

        right_result_list = [4.0, 4.5, 3.5]

        result_list = main.Hurwitz_matrix(MatrixPay)

        assert array_equal(result_list, right_result_list) == True

    def test_criterion_Hurwitz_pay_2(self):
        MatrixPay = [
            [5, 9, 4, 2, 8],
            [7, 3, 1, 1, 1],
            [1, 2, 5, 4, 3],
            [2, 9, 7, 3, 5],
            [1, 2, 4, 4, 3]
        ]

        right_result_list = [5.5, 4.0, 3.0, 5.5, 2.5]

        result_list = main.Hurwitz_matrix(MatrixPay)

        assert array_equal(result_list, right_result_list) == True


class TestHurwitzRisk(unittest.TestCase):
    """ Тестирование решения, полученного из критерия Гурвица, основанном на риске """

    def test_criterion_Hurwitz_risk_1(self):
        RiskMatrix = [
            [3, 4, 7, 0],
            [5, 0, 1, 6],
            [1, 2, 8, 7]
        ]

        right_result_list = [3.5, 3.0, 4.5]

        result_list = main.Hurwitz_risk(RiskMatrix)

        assert array_equal(result_list, right_result_list) == True

    def test_criterion_Hurwitz_risk_2(self):
        RiskMatrix = [
            [5, 9, 4, 2, 8],
            [7, 3, 1, 1, 1],
            [1, 2, 5, 4, 3],
            [2, 9, 7, 3, 5],
            [1, 2, 4, 4, 3]
        ]

        right_result_list = [5.5, 4.0, 3.0, 5.5, 2.5]

        result_list = main.Hurwitz_risk(RiskMatrix)

        assert array_equal(result_list, right_result_list) == True


if __name__ == "__main__":
    unittest.main()
