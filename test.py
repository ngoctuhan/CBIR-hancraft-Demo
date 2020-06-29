from utils.utils.utils import find_have_in_database

# print(find_have_in_database())

import pandas as pd #

FILE_RESULT = 'utils/result/result.csv'

FILE_WEIGHT = 'utils/result/weight.csv'

df = pd.read_csv(FILE_RESULT)
df.to_csv(FILE_RESULT)