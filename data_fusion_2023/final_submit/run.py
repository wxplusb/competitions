from catboost import CatBoostClassifier
import pandas as pd
import sys
import numpy as np

from model import reliable_predict

from pt_model import pt_predict


def main():
    source_file, output_path = sys.argv[1:]
    
    bins_path = "./nn_bins.pickle"
    model_path = "./nn_weights.ckpt"

    pt_result = pt_predict(source_file, bins_path, model_path).sort_values('user_id').reset_index(drop=True)

    result = reliable_predict(source_file, bins_path, model_path)
    
    df = pd.read_csv(source_file, parse_dates=['transaction_dttm'])

    df['time'] = df.transaction_dttm.apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)

    model_cb = CatBoostClassifier().load_model("./model_cb.cbm")
    columns = model_cb.get_feature_importance(prettified=True)['Feature Id'].values

    preds = []
    for i in range(20):
        df_sample = df.groupby('user_id').sample(n=300, random_state=34+i, replace=True)
    
        mcc_f = df_sample.pivot_table(
            index='user_id', columns=['mcc_code'], values=['transaction_amt'], 
            aggfunc=['count'], fill_value=0
        )
        mcc_f.columns = [f'{i[0]}_mcc_{i[2]}' for i in mcc_f.columns]
        for col in mcc_f.columns:
            mcc_f[col] //= 20
            
        time_f = df_sample.groupby('user_id')['time'].agg(['mean', 'std', 'min', 'max', 'median'])
        time_f.columns = [f'time_{c}' for c in time_f.columns]
        
        X_test = pd.concat([result.set_index('user_id').target.rename('rel_pred'),time_f,mcc_f], axis=1)
        
        for col in columns:
            if col not in X_test.columns:
                X_test[col] = 0

        predicts = model_cb.predict_proba(X_test[columns])[:,1]
        preds.append(predicts)
        
    preds = np.mean(preds, axis=0)

    submission = pd.DataFrame({"user_id": X_test.index, "target": preds}).sort_values('user_id').reset_index(drop=True)

    submission['target'] = (submission['target'] + (pt_result['target']*0.6))/2

    submission.to_csv(output_path, index=False)
    


if __name__ == "__main__":
    main()