import pandas

df = pandas.read_csv('/fs/clip-scratch/tasnim/hay2/DPR/DPR/dpr/downloads/data/retriever/dev.csv')
df2 = df.rename(columns={'"answers":"answer'})
df2.to_csv("'/fs/clip-scratch/tasnim/hay2/DPR/DPR/dpr/downloads/data/retriever/dev_.csv'", index=None)
