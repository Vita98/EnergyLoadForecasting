import pandas as pd
import os
import numpy as np





if __name__ == "__main__":
    houses = ["ukdale_def1", "ukdale_def2", "ukdale_def3", "ukdale_def4", "ukdale_def5"]
    #houses = ["ukdale_def1"]
    algorithms = ["arimastd","sarima", "sarimax"]
    #algorithms = ["sarimax"]

    

    for alg in algorithms:
        for house in houses:
            days7 = list()
            days30 = list()

            df = pd.read_csv("results/" + alg + "/" + house + "/total/accuracy.csv", header=None, usecols=[2, 4, 5])

            for i in range(0, len(df)):
                if df[5].iloc[i] == 7:
                    days7.append(df[2].iloc[i])
                else:
                    days30.append(df[2].iloc[i])

            print("Alg: " + alg + " House: "+ house)
            print("Avg RMSE 7 days: " + str(np.mean(days7)))
            print("Avg RMSE 30 days: " + str(np.mean(days30)), end= "\n\n")
