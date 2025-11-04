import pandas as pd

def main():
    print("Hello from cmsc-435-project!")

    df = pd.read_csv('C:/Users/rbrad/Desktop/VCU Fall 2025/CMSC 435/CMSC 435 Project/Dataset/rapidminer_results.csv')
    print(df.head())


if __name__ == "__main__":
    main()
