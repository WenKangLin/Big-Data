import pandas as pd

traditional = pd.read_csv('timings.csv')
mapreduce   = pd.read_csv('mapreduce_timings.csv')

traditional.insert(0, 'Method', 'Traditional')
mapreduce.insert(0,   'Method', 'MapReduce')

combined = pd.concat([traditional, mapreduce], ignore_index=True)
combined = combined.sort_values('Section').reset_index(drop=True)
combined.to_csv('timings_comparison.csv', index=False)
print(combined.to_string(index=False))
