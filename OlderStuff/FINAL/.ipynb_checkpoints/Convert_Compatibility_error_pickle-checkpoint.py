## This utility can be used to convert pickle files to compatible versions. See error on activity document under XGBoost for more details.

import pickle, pandas as pd, numpy as np
from pathlib import Path

# Path to your NumPy-2 pickle
pkl = Path("./data/Images_Classified_NN_cl6.pkl")

with pkl.open("rb") as f:
    obj = pickle.load(f)

# Re-serialize using an older pickle protocol
out_path = pkl.with_name(pkl.stem + "_np126.pkl")

# pandas DataFrame case
if isinstance(obj, pd.DataFrame):
    with out_path.open("wb") as f:
        pickle.dump(obj, f, protocol=4)   # protocol 4 works fine for TF/NumPy 1.x
    print("✅ Converted DataFrame saved as:", out_path)

# pure NumPy array case
elif isinstance(obj, np.ndarray):
    with out_path.open("wb") as f:
        pickle.dump(obj, f, protocol=4)
    print("✅ Converted ndarray saved as:", out_path)

# other Python objects
else:
    with out_path.open("wb") as f:
        pickle.dump(obj, f, protocol=4)
    print("✅ Converted generic object saved as:", out_path)
