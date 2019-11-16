# Load libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings  # To ignore any warnings


def main():
    warnings.filterwarnings("ignore")
    dataset = pd.read_csv('loan_data_set.csv')


    sns.countplot(x='Loan_Status', data=dataset, palette='hls')
    sns.set(rc={'axes.facecolor': '#f8f9fa', 'figure.facecolor': '#f8f9fa'})
    plt.show()

    missing_values = dataset.isnull()

    sns.heatmap(data=missing_values, yticklabels=False, cbar=False, cmap='viridis')
    sns.set(rc={'axes.facecolor': '#f8f9fa', 'figure.facecolor': '#f8f9fa'})
    plt.show()

    sns.countplot(x='Loan_Status', data=dataset,hue='Education')
    sns.set(rc={'axes.facecolor': '#f8f9fa', 'figure.facecolor': '#f8f9fa'})
    plt.show()


if __name__ == '__main__':
    main()
