import pandas as pd

def df_add_row(df, d):
    d = d.copy() # just to be safe
    for x,y in d.items():
        d[x] = [y]
    d = pd.DataFrame(d)
    return pd.concat([df, d], ignore_index=True)
