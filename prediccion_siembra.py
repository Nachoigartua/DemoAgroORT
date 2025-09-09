import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#aca pongo datos random para entrenar a la ia (deberian ir datos chequeados climaticos etc)
data = {
    "Cultivo": ["soja", "soja", "maiz", "maiz", "trigo", "trigo", "soja", "maiz", "trigo"],
    "Mes": [10, 11, 9, 12, 5, 6, 12, 9, 7],
    "Lluvia_mm": [120, 140, 90, 110, 60, 70, 130, 100, 80],
    "Temp_prom": [20, 22, 25, 28, 15, 17, 21, 26, 16],
    "Rendimiento": [3.2, 3.5, 4.1, 3.0, 2.5, 2.8, 3.6, 4.0, 2.7]
}
df = pd.DataFrame(data) #clase de pandas para crear la data en una matriz
print("\n--- Dataset ---")
print(df)

# sklearn con la clase labelencoder convierne todos los strings a numeros, para que nuestro modelo de ml entienda
encoder = LabelEncoder() #clase que tiene metodos que convierten strings en numeros para que el modelo lo entienda, le da un numero q cada string, como si fuese un """id"""
df["Cultivo_cod"] = encoder.fit_transform(df["Cultivo"]) #transforma el cultivo (string) en un numero por primera vez
print("\n--- Dataset con codificación ---")
print(df)

X = df[["Cultivo_cod", "Mes", "Lluvia_mm", "Temp_prom"]] #variables que usamos para predecir
y = df["Rendimiento"] #lo que queremos predecir

# una vez separadas x de y , preparo un modelo para predecir, pasandole una x y una y para la prediccion(test), y otra x e y para que aprenda a predecir(train)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = LinearRegression() #regresion lineal es un modelo tipico de ML, una forma de predecir
modelo.fit(X_train, y_train) #metodo de sklearn que entrena al modelo con los datos historicos (harcodeados en nuestro caso)
score = modelo.score(X_test, y_test) #devuelve un numero que representa que tan bien predijo los datos
print(f"\nPrecisión del modelo: {score:.2f}")

# rutina para hacer predicciones con los datos q le pasas por parametro
def predecir_rendimiento(cultivo, mes, lluvia, temp):
    cultivo_cod = encoder.transform([cultivo])[0] #el transform te devuevle el numero q se le asigno al string con el fit.transform 
    rend_pred = modelo.predict([[cultivo_cod, mes, lluvia, temp]]) #en base a todo lo que aprendio el modelo con el .fit , predice
    print(f"Predicción de rendimiento para {cultivo} en mes {mes}: {rend_pred[0]:.2f} toneladas/ha")
    if rend_pred[0] > 3.0:
        print("✅ Buena fecha de siembra")
    else:
        print("⚠️ No es la mejor fecha")
    return rend_pred[0]

predecir_rendimiento("soja", 11, 130, 21)
