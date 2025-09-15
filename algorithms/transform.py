import pandas as pd
import numpy as np

import datamanage


class Transform:

    def changeEdgeToMatrix(self, path):
        dm = datamanage.DataManage()
        node_dict, N, M = dm.generateMap(path)
        matrix = np.random.randint(0, 1, size=(N, M))

        df = pd.read_csv(path, index_col=False, header=None)
        arr = df.values
        index = 0
        for each in arr:

            edge_list = list(map(int, each[0].split(" ")))
            for edge in edge_list:

                matrix[node_dict[edge]][index] = 1
            index = index + 1
        return pd.DataFrame(matrix), N

