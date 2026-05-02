import argparse
import joblib
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description='Predict water level from one sensor sample using saved model.'
    )
    parser.add_argument(
        '--values',
        type=float,
        nargs='+',
        required=True,
        help='Sensor values in this order: ir_value ir_strength us_value acc_x acc_y acc_z gyr_acc_x gyr_acc_y gyr_acc_z gyr_x gyr_y gyr_z'
    )
    parser.add_argument(
        '--model-path',
        default='water_level_model.joblib',
        help='Path to saved model bundle (default: water_level_model.joblib)'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    bundle = joblib.load(args.model_path)

    model = bundle['model']
    feature_columns = bundle['feature_columns']

    if len(args.values) != len(feature_columns):
        raise ValueError(
            f'Expected {len(feature_columns)} values, received {len(args.values)}.'
        )

    sample_df = pd.DataFrame([args.values], columns=feature_columns)
    prediction = model.predict(sample_df)[0]

    print('--- PREDICTION RESULT ---')
    print('Predicted Water Level:', round(float(prediction), 4), 'units')


if __name__ == '__main__':
    main()
