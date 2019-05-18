import pandas as pd
df=pd.read_excel("/home/rahul/PycharmProjects/Datathon/final_data_v2.xlsx")
df.drop(["index","PRIZM5DA"], axis=1, inplace=True)
df_corr=df.corr()
list_of_var=['PRIZM5DA_20', 'HSCM001F', 'V0564', 'WSD2AR', 'V1218', 'V4229', 'V4809I', 'SV00041', 'V6476', 'ECYMARSING', 'ECYHTA3034', 'SV00066', 'HSSH037B', 'HSTA005', 'SV00030']
df_corr_top=df_corr.filter(list_of_var,axis=0)
#df_corr_top_T=df_corr_top.transpose()
cols=list(df_corr_top.columns.values)
rows=list(df_corr_top.index.values)
print(cols)
print(rows)
res_df=pd.DataFrame(columns=["Top 15 Variables","Other Variables","Correlation"])
for row in rows:
    for col in cols:
        if not row==col:
            if df_corr_top.at[row,col]>=0.65 or df_corr_top.at[row,col]<=-0.65:
                res_df=res_df.append({"Top 15 Variables":row,"Other Variables":col,"Correlation":df_corr_top.at[row,col]}, ignore_index = True)
            else:
                continue
        else:
            continue
print("res_df=",res_df)