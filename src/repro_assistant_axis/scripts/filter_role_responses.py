import pandas as pd 

def filter_and_prepare_roles(model_name):
    eval_df = pd.read_parquet(f"evaluations_{model_name}.parquet")
    good_df = eval_df[eval_df['score'].isin([2, 3])].copy()
    
    counts = good_df.groupby(['role', 'score']).size()
    kept = counts[counts >= 10].reset_index()[['role', 'score']]
    
    filtered_df = good_df.merge(kept, on=['role', 'score'])
    filtered_df.to_parquet(f"filtered_{model_name}_responses.parquet")
    
    print(f"{model_name}:")
    print(f"  Responses kept: {len(filtered_df)}")
    print(f"  Role-score combinations: {len(kept)}")
    print(f"  Unique roles: {filtered_df['role'].nunique()}")
    
    return filtered_df, kept