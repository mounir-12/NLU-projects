import pandas as pd

pred_stories = pd.read_csv('data/test-stories.csv')
df = pred_stories
df.insert(loc=0, column='id', value=range(1, len(df) + 1))
column_names = ["InputStoryid","sentence1","sentence2","sentence3","sentence4","sentence5"]

df_ending_1 = df.drop("RandomFifthSentenceQuiz2", axis=1)
df_ending_1.columns = column_names
df_ending_1['label'] = 0

df_ending_2 = df.drop("RandomFifthSentenceQuiz1", axis=1)
df_ending_2.columns = column_names
df_ending_2['label'] = 1

concat = pd.concat([df_ending_1, df_ending_2])

concat.to_csv("./data/predict.csv", index=False)
