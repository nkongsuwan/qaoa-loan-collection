import numpy as np

class ResultQaoa:
    def __init__(self):
        self.__log = []
        self.__len_result = 0
        self.__len_params = None
        self.__scipy_result = None

    def __repr__(self):
        result_str = ""
        result_str += "ResultQaoa:\n"
        result_str += "    Convergence      = " + str(self.is_success()) + "\n"
        result_str += "    Length           = " + str(self.get_len()) + "\n"
        result_str += "    Optimized Cost   = " + str(self.get_optimized_cost()) + "\n"
        result_str += "    Optimized Params = " + str(self.get_optimized_params())
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

    def add_scipy_result(self, scipy_result):
        self.__scipy_result = scipy_result

    def get_len(self):
        return self.__len_result

    def get_cost_with_index(self, index: int):
        return self.__log[index][0]

    def get_params_with_index(self, index: int):
        return self.__log[index][1]

    def get_optimized_cost(self):
        assert self.__scipy_result is not None
        return self.__scipy_result.fun

    def get_optimized_params(self):
        assert self.__scipy_result is not None
        return self.__scipy_result.x

    def is_success(self):
        assert self.__scipy_result is not None
        return self.__scipy_result.success

    def get_list_costs(self):
        return [element[0] for element in self.__log]
    
    def get_list_params(self):
        return [element[1] for element in self.__log]