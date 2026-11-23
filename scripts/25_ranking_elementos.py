import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("Generando Ranking de Elementos...")
    df = pd.read_csv(os.path.join("..", "data", "dataset_limpio_100k.csv"))
    df['primer_elemento'] = df['elementos'].astype(str).str.split(',').str[0]
    top_elem = df['primer_elemento'].value_counts().head(15)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_elem.values, y=top_elem.index, palette="rocket")
    plt.title("Top 15 Most Frequent Elements in Dataset")
    plt.savefig(os.path.join("..", "final_figures", "fig25_top_elements.png"), dpi=300)
    print("✅ Guardado.")

if __name__ == "__main__":
    main()