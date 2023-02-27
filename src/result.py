import numpy as np


class ResultQaoa:

    def __init__(self):
        self.__list_costs = []
        self.__list_params = []
        self.__len_params = None
        self.__success = False


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

        self.__list_costs.append(cost)
        self.__list_params.append(params)


    def finalize(self, result):
        self.__success = result


    def get_len(self):
        return len(self.__list_costs)


    def get_cost_with_index(self, index: int):
        return self.__list_costs[index]


    def get_params_with_index(self, index: int):
        return self.__list_params[index]


    def get_optimized_cost(self):
        return np.min(self.__list_costs)


    def get_optimized_params(self):
        index = np.argmin(self.__list_costs)
        return self.__list_params[index]


    def is_success(self):
        return self.__success


    def get_list_costs(self):
        return self.__list_costs

    
    def get_list_params(self):
        return self.__list_params