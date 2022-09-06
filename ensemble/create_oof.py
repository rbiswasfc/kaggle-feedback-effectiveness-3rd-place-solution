import pandas as pd
import os
import glob

oof_path = '../output/exp213-deb-l-prompt-mlm50-kaggle/oofs/'

oof_files = glob.glob(oof_path + "/*.csv")
print(oof_files)
oof_df = pd.DataFrame()
for oof_path in oof_files:
    oof_df_ = pd.read_csv(oof_path)
   # print(oof_df_.head())
   # print(oof_df_.shape)
    oof_df = pd.concat([oof_df, oof_df_])

oof_df = oof_df.sort_values(by='discourse_id').reset_index(drop=True)
print(oof_df.shape)
oof_df.to_csv('exp213_oof_deb_l_prompt_10folds.csv', index=False)
