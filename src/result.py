import numpy as np

class ResultQaoa:
    def __init__(self):
        self.__log = []
        self.__len_result = 0
        self.__len_params = None

    def __repr__(self):
        result_str = ""
        result_str += "ResultQaoa:\n"
        result_str += "    Length = " + str(self.get_len()) + "\n"
        result_str += "    Final Cost = " + str(self.get_final_cost()) + "\n"
        result_str += "    Final Params = " + str(self.get_final_params())
        return result_str

    def append(self, cost: float, params: np.ndarray):

        assert isinstance(cost, int) or isinstance(cost, float)
        assert isinstance(params, np.ndarray)        
        assert np.issubdtype(params.dtype, np.integer) or np.issubdtype(params.dtype, np.floating)
        
        len_params = len(params)
        assert len_params > 0
        assert len_params % 2 == 0
        if self.__len_params is None:
            self.__len_params = len_params
        else:
            assert self.__len_params == len_params

        self.__log.append(
            (cost, params)
        )
        self.__len_result += 1

    def get_len(self):
        assert self.__len_result == len( self.__log)
        return self.__len_result

    def get_cost_with_index(self, index: int):
        assert isinstance(index, int)
        assert index >= 0
        assert index < self.__len_result
        return self.__log[index][0]

    def get_params_with_index(self, index: int):
        assert isinstance(index, int)
        assert index >= 0
        assert index < self.__len_result
        return self.__log[index][1]

    def get_final_cost(self):
        assert self.__len_result > 0
        return self.__log[-1][0]

    def get_final_params(self):
        assert self.__len_result > 0
        return self.__log[-1][1]