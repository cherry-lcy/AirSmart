import pandas as pd
import numpy as np
from ac_model import AC_Tester
import pickle

# variable list
place = "HKA"
path = "" # path you store the data

# load test data
test_seq = pd.read_csv(path+"TEST_"+place+"_DATASET_.csv")
test_seq = np.array(test_seq["Mean Temp"])

# test model
if __name__ == "__main__":
    # initialize tester
    tester = AC_Tester(
        model_path=path+"ac_model_test.pth",
        test_data=test_seq,
        target_temp=22,
        hist_len=10
    )

    # run testing (5 episodes, and visualize each episode)
    results = tester.run_test(num_episodes=5, render_interval=1)

    # save test result
    with open("test_results.pkl", "wb") as f:
        pickle.dump(results, f)