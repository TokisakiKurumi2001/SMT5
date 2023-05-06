import pandas as pd
import os

if __name__ == "__main__":
    mypath = "./translate_snli"
    files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    ls = []
    for file in files:
        df = pd.read_csv(f"{mypath}/{file}")
        df['idx'] = list(range(6000))
        ls.append(df)
    train_df = pd.concat(ls)
    train_df.to_csv('tsnli.train.csv', index=False)