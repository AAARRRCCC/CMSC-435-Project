import subprocess
import os
import pandas as pd

# Configure which pfeature jobs to run - add or remove methods as needed
PFEATURE_JOBS = ['aac',
                 'dpc',
                 'pcp',
                 'ddr'           
                ]


def rm_main(pfeatureinputfilepath= "C:/Users/rbrad/Desktop/VCU Fall 2025/CMSC 435/CMSC-435-Project/sequences_test.txt"):

    

    # Run each pfeature job and collect results
    feature_dfs = []
    for job in PFEATURE_JOBS:
        temp_output = f"CSVTEMP_{job}.csv"

        print(f"\n{'='*60}")
        print(f"Starting feature extraction: {job.upper()}")
        print(f"{'='*60}")

        # Run pfeature with real-time output display
        # Use shell=True to allow output streaming
        result = subprocess.run(
            f"python pfeature_comp.py -i ../Dataset/Unfinished/pfeatureSequenced.fa -o {temp_output} -j {job}",
            shell=True,
            text=True
        )

        if result.returncode != 0:
            print(f"WARNING: {job.upper()} extraction had non-zero exit code: {result.returncode}")

        df_job = pd.read_csv(temp_output)
        feature_dfs.append(df_job)
        print(f"✓ {job.upper()} complete - extracted {len(df_job.columns)} features")

    # Combine all feature sets horizontally (concatenate columns)
    # First dataframe includes the sequence identifiers, others just add features
    combined_df = feature_dfs[0]
    for df in feature_dfs[1:]:
        # Drop any duplicate identifier columns and merge on index
        feature_cols = [col for col in df.columns if not col.startswith('seq_')]
        combined_df = pd.concat([combined_df, df[feature_cols]], axis=1)

    return combined_df

if __name__ == "__main__":
    input_path = "..Dataset/Unfinished/pfeatureSequenced.fa"
    rm_main(input_path)