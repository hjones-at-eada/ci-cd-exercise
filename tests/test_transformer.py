from src.transform import Transformer, balance_dataset
import pandas as pd


def test_map_binary_column_to_int():
    transformer = Transformer()
    df = pd.DataFrame(
        {
            "Gender": ["Female", "Male", "Female", "Male"],
        }
    )

    expected_df = pd.DataFrame(
        {
            "Gender": [1, 0, 1, 0],
        }
    )

    transformed_df = transformer._map_binary_column_to_int(df)

    # Test the result against the expected DataFrame
    pd.testing.assert_frame_equal(transformed_df, expected_df)
