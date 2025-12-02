import pandas as pd

def rm_main(df):
    # Thresholds - lower = more likely to predict that class
    thresholds = {'DNA': 0.15, 'RNA': 0.20, 'DRNA': 0.10, 'nonDRNA': 0.50}


    
    def custom_predict(row):
        # Check minority classes first
        for cls in ['DRNA', 'DNA', 'RNA']:
            if row[f'confidence({cls})'] > thresholds[cls]:
                return cls
        return 'nonDRNA'
    
    df['prediction(class)'] = df.apply(custom_predict, axis=1)
    
    return df