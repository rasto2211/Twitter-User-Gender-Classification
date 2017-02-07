import common
import pandas as pd


DIR = "data"
DATA_SETS = ["train", "test"]


df = common.load_data()
selected_rows = common.select_rows(df)

train, test = common.split_data(selected_rows)

for rows, data_set_name in zip([train, test], DATA_SETS):
    pd.DataFrame(rows, columns=["row_number"])\
        .to_csv("{}/{}_rows.txt".format(DIR, data_set_name),
                index=False)
