import joblib
import pandas as pd


def batch_predict(data_path='prediction_data_100rows.csv', model_path='water_level_model.joblib', output_path='predictions_output.csv'):
    """
    Load saved model and make bulk predictions on a CSV of sensor data.
    
    Args:
        data_path: CSV with sensor readings (no water_level column)
        model_path: Path to saved model bundle
        output_path: Where to save predictions
    """
    bundle = joblib.load(model_path)
    model = bundle['model']
    feature_columns = bundle['feature_columns']
    
    # Load prediction data
    df = pd.read_csv(data_path)
    
    if len(df.columns) != len(feature_columns):
        raise ValueError(
            f'Expected {len(feature_columns)} columns, got {len(df.columns)}.'
        )
    
    # Ensure column order matches training
    df = df[feature_columns]
    
    # Predict all rows
    predictions = model.predict(df)
    
    # Create output with input + predictions
    result_df = df.copy()
    result_df['predicted_water_level'] = predictions
    
    result_df.to_csv(output_path, index=False)
    
    print(f'Batch predictions complete.')
    print(f'Processed: {len(df)} rows')
    print(f'Saved to: {output_path}')
    print(f'\nSample predictions (first 10):')
    print(result_df[['predicted_water_level']].head(10).to_string())


if __name__ == '__main__':
    batch_predict()
