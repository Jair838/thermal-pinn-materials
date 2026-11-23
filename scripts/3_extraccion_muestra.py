import pandas as pd
import os

def main():
    print("Extrayendo muestra rapida (esto tardara menos de 1 minuto)...")
    ruta_entrada = os.path.join("..", "data", "dataset_gigante.csv")
    ruta_salida = os.path.join("..", "data", "muestra_expres.csv")
    
    try:
        # Leemos 1 millón de filas
        df = pd.read_csv(ruta_entrada, nrows=1000000)
        
        # Mezclamos para tener variedad de materiales
        df = df.sample(frac=1).reset_index(drop=True)
        
        # Guardamos
        df.to_csv(ruta_salida, index=False)
        
        materiales = df['formula'].nunique()
        print(f"✅ ¡Listo! Archivo guardado con {len(df):,} filas.")
        print(f"✅ Materiales distintos encontrados: {materiales}")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    main()