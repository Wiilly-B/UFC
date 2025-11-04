import pandas as pd
import re


def preprocess_column(column):
    # Convert to lowercase
    column = column.str.lower()

    # Remove spaces and non-alphanumeric characters
    column = column.apply(lambda x: re.sub(r'[^a-zA-Z0-9]', '', x))

    return column


def merge_csv_files(file1_path, file2_path, file3_path, output_file_path):
    try:
        # Read the first CSV file
        df1 = pd.read_csv(file1_path)

        # Read the second CSV file
        df2 = pd.read_csv(file2_path)

        # Read the third CSV file
        df3 = pd.read_csv(file3_path)

        # Store the original 'BOUT' and 'EVENT' columns
        df1['bout_original'] = df1['BOUT']
        df1['event_original'] = df1['EVENT']
        df2['bout_original'] = df2['BOUT']
        df2['event_original'] = df2['EVENT']
        df3['event_original'] = df3['EVENT']

        # Preprocess the 'BOUT' and 'EVENT' columns in df1 and df2
        df1['bout_preprocessed'] = preprocess_column(df1['BOUT'])
        df1['event_preprocessed'] = preprocess_column(df1['EVENT'])
        df2['bout_preprocessed'] = preprocess_column(df2['BOUT'])
        df2['event_preprocessed'] = preprocess_column(df2['EVENT'])

        # Preprocess the 'EVENT' column in df3
        df3['event_preprocessed'] = preprocess_column(df3['EVENT'])

        # Merge df2 into df1 based on the preprocessed 'BOUT' and 'EVENT' columns
        merged_df = pd.merge(df1, df2, on=['bout_preprocessed', 'event_preprocessed'], how='left')

        # Merge df3 into the merged DataFrame based on the preprocessed 'EVENT' column
        merged_df = pd.merge(merged_df, df3, on='event_preprocessed', how='left')

        # Drop the preprocessed 'BOUT' and 'EVENT' columns
        merged_df.drop(['bout_preprocessed', 'event_preprocessed'], axis=1, inplace=True)

        # Convert all column names to lowercase
        merged_df.columns = [col.lower() for col in merged_df.columns]

        # Remove columns with '_y' or '_original' in their names
        columns_to_drop = [col for col in merged_df.columns if '_y' in col or '_original' in col] + ["details",
                                                                                                     "referee"]
        merged_df.drop(columns_to_drop, axis=1, inplace=True)

        # Remove the '_x' suffix from column names
        merged_df.columns = [col.rstrip('_x') for col in merged_df.columns]

        # Drop the first column
        merged_df = merged_df.iloc[:, 1:]

        # Save the merged DataFrame to a new CSV file
        merged_df.to_csv(output_file_path, index=False)

        print("Files merged successfully!")
    except Exception as e:
        print(f"An error occurred during merge: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def process_merged_csv(merged_file_path, processed_file_path):
    try:
        # Read the merged CSV file
        df = pd.read_csv(merged_file_path)

        # Convert date format
        df['fight_date'] = pd.to_datetime(df['date'], format='%B %d, %Y').dt.strftime('%Y-%m-%d')

        # Process the 'round' column
        df['round'] = df['round'].str.replace('Round ', '', regex=False)
        df['round'] = pd.to_numeric(df['round'], errors='coerce').fillna(0).astype(int)

        # Create a new column 'last_round' with the maximum round for each bout and event
        df['last_round'] = df.groupby(['bout', 'event'])['round'].transform('max')

        # Create a new column 'id' for each unique bout and event
        df['id'] = df.groupby(['bout', 'event']).ngroup() + 1

        # Rename the 'kd' column to 'knockdown'
        df.rename(columns={'kd': 'knockdowns'}, inplace=True)

        # Process the 'sig.str.', 'total str.', 'td', 'head', 'body', 'leg', 'distance', 'clinch', and 'ground' columns
        columns_to_process = ['sig.str.', 'total str.', 'td', 'head', 'body', 'leg', 'distance', 'clinch', 'ground']
        for col in columns_to_process:
            if col in df.columns:
                col_index = df.columns.get_loc(col)
                new_columns = [f"{col.replace('.', '').replace(' ', '_')}_landed", f"{col.replace('.', '').replace(' ', '_')}_attempted"]
                df[new_columns] = df[col].str.split(' of ', expand=True)
                df.drop(columns=[col], inplace=True)
                df.insert(col_index, new_columns[0], df.pop(new_columns[0]))
                df.insert(col_index + 1, new_columns[1], df.pop(new_columns[1]))

        # Rename columns and convert percentages to decimals
        df.rename(columns={
            'sigstr_landed': 'significant_strikes_landed',
            'sigstr_attempted': 'significant_strikes_attempted',
            'sig.str. %': 'significant_strikes_rate',
            'total_str_landed': 'total_strikes_landed',
            'total_str_attempted': 'total_strikes_attempted',
            'td_landed': 'takedown_successful',
            'td_attempted': 'takedown_attempted',
            'td %': 'takedown_rate',
            'sub.att': 'submission_attempt',
            'rev.': 'reversals',
            'ctrl': 'control',
            'outcome': 'winner',
            'time format': 'scheduled_rounds',
            'method': 'result',
            'weightclass': 'weight_class'
        }, inplace=True)

        # Replace '---' with '0' in 'significant_strikes_rate' and 'takedown_rate' columns
        for col in ['significant_strikes_rate', 'takedown_rate']:
            if col in df.columns:
                df[col] = df[col].replace('---', 0)

        # Convert percentages to decimals
        for col in ['significant_strikes_rate', 'takedown_rate']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: float(x.strip('%')) / 100 if isinstance(x, str) else x)

        if 'control' in df.columns:
            df['control'] = df['control'].replace('--', 0)

        # Convert control time from 'min:seconds' to seconds
        if 'control' in df.columns:
            df['control'] = df['control'].apply(lambda x: sum(int(i) * 60 ** idx for idx, i in enumerate(reversed(str(x).split(':')))) if isinstance(x, str) else 0)

        # Process the 'outcome' column
        def process_outcome(row):
            try:
                if row['winner'] in ['W/L', 'L/W']:
                    bout_fighter = str(row['bout']).split('vs.')[0].strip().lower().replace(' ', '')
                    fighter = str(row['fighter']).lower().replace(' ', '')
                    if row['winner'] == 'W/L':
                        return 'W' if fighter == bout_fighter else 'L'
                    else:  # L/W
                        return 'L' if fighter == bout_fighter else 'W'
                return row['winner']
            except Exception as e:
                print(f"Error processing row: {row}")
                print(f"Error message: {str(e)}")
                return row['winner']

        df['winner'] = df.apply(process_outcome, axis=1)

        # Remove 'bout' and 'url' columns
        df.drop(columns=['bout', 'url'], inplace=True, errors='ignore')

        # Reorder columns
        column_order = [
            'fighter', 'knockdowns', 'significant_strikes_landed', 'significant_strikes_attempted',
            'significant_strikes_rate', 'total_strikes_landed', 'total_strikes_attempted',
            'takedown_successful', 'takedown_attempted', 'takedown_rate', 'submission_attempt',
            'reversals', 'head_landed', 'head_attempted', 'body_landed', 'body_attempted',
            'leg_landed', 'leg_attempted', 'distance_landed', 'distance_attempted',
            'clinch_landed', 'clinch_attempted', 'ground_landed', 'ground_attempted',
            'control', 'round', 'result', 'last_round', 'time', 'scheduled_rounds',
            'winner', 'weight_class', 'event', 'fight_date', 'location', 'id'
        ]

        # Reorder the columns and select only the ones in column_order
        df = df[column_order]

        # Save the processed DataFrame to a new CSV file
        df.to_csv(processed_file_path, index=False)

        print("CSV file processed successfully!")
    except FileNotFoundError:
        print("Merged file not found.")


if __name__ == "__main__":
    # Specify the paths to the input CSV files and the output file
    file1_path = '../../../data/raw/ufc_fight_stats.csv'
    file2_path = '../../../data/raw/ufc_fight_results.csv'
    file3_path = '../../../data/raw/ufc_event_details.csv'
    output_file_path = '../../../data/raw/ufc_fight_merged.csv'
    processed_file_path = '../../../data/processed/ufc_fight_processed.csv'

    # Call the function to merge the CSV files
    merge_csv_files(file1_path, file2_path, file3_path, output_file_path)

    # Call the function to process the merged CSV file
    process_merged_csv(output_file_path, processed_file_path)
