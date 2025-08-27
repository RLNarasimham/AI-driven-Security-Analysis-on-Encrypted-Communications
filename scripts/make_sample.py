import pandas as pd
import os

# Path to one of the CICIDS2017 CSVs from the original dataset
# Adjust this path if your dataset folder is named differently
# SOURCE_FILE = "MachineLearningCSV/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

SOURCE_FILE = "../data/CICIDS2017_CSVs/MachineLearningCSV/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"


# Output folder for sample
OUTPUT_FILE = "../data/CICIDS2017_sample.csv"


def main():
    if not os.path.exists(SOURCE_FILE):
        print(f"❌ Source file not found: {SOURCE_FILE}")
        print("Please place the original CICIDS2017 CSVs in the 'MachineLearningCSV/' folder.")
        return

    os.makedirs("data", exist_ok=True)

    # Take a random sample of 10 rows with all columns preserved
    df = pd.read_csv(SOURCE_FILE)
    sample = df.sample(10, random_state=42)
    sample.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ Sample CSV created at {OUTPUT_FILE} with {len(sample)} rows and {len(sample.columns)} columns.")

if __name__ == "__main__":
    main()
