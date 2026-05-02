import joblib
import pandas as pd


def interactive_predict():
    """
    Load saved model and interactively prompt user for sensor values.
    """
    bundle = joblib.load('water_level_model.joblib')
    model = bundle['model']
    feature_columns = bundle['feature_columns']
    
    print('=' * 60)
    print('WATER LEVEL PREDICTOR - Interactive Mode')
    print('=' * 60)
    print(f'Enter sensor readings for {len(feature_columns)} features.')
    print('Press Ctrl+C to exit.\n')
    
    while True:
        try:
            values = []
            for col in feature_columns:
                while True:
                    try:
                        val = float(input(f'{col}: '))
                        values.append(val)
                        break
                    except ValueError:
                        print('Invalid input. Please enter a number.')
            
            # Make prediction
            sample_df = pd.DataFrame([values], columns=feature_columns)
            prediction = model.predict(sample_df)[0]
            
            print(f'\nPredicted Water Level: {prediction:.4f} units\n')
            print('-' * 60 + '\n')
            
        except KeyboardInterrupt:
            print('\n\nExit. Goodbye!')
            break


if __name__ == '__main__':
    interactive_predict()
