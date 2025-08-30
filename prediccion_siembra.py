import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import ttk, messagebox

# Datos de ejemplo
data = {
    "Cultivo": ["soja", "soja", "maiz", "maiz", "trigo", "trigo", "soja", "maiz", "trigo"],
    "Mes": [10, 11, 9, 12, 5, 6, 12, 9, 7],
    "Lluvia_mm": [120, 140, 90, 110, 60, 70, 130, 100, 80],
    "Temp_prom": [20, 22, 25, 28, 15, 17, 21, 26, 16],
    "Rendimiento": [3.2, 3.5, 4.1, 3.0, 2.5, 2.8, 3.6, 4.0, 2.7],
}

df = pd.DataFrame(data)

encoder = LabelEncoder()
df["Cultivo_cod"] = encoder.fit_transform(df["Cultivo"])

X = df[["Cultivo_cod", "Mes", "Lluvia_mm", "Temp_prom"]]
y = df["Rendimiento"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = LinearRegression()
modelo.fit(X_train, y_train)
score = modelo.score(X_test, y_test)
print(f"\nPrecisión del modelo: {score:.2f}")


def predecir_rendimiento(cultivo, mes, lluvia, temp):
    cultivo_cod = encoder.transform([cultivo])[0]
    rend_pred = modelo.predict([[cultivo_cod, mes, lluvia, temp]])[0]
    return rend_pred, rend_pred > 3.0


def mostrar_prediccion():
    cultivo = cultivo_var.get()
    try:
        mes = int(mes_var.get())
        lluvia = float(lluvia_var.get())
        temp = float(temp_var.get())
    except ValueError:
        messagebox.showerror("Error", "Ingrese valores numéricos válidos")
        return

    pred, buena_fecha = predecir_rendimiento(cultivo, mes, lluvia, temp)
    mensaje = f"Rendimiento estimado: {pred:.2f} toneladas/ha\n"
    mensaje += "✅ Buena fecha de siembra" if buena_fecha else "⚠️ No es la mejor fecha"
    result_label.config(text=mensaje)


root = tk.Tk()
root.title("Predicción de Siembra")

cultivo_var = tk.StringVar(value="soja")
mes_var = tk.StringVar()
lluvia_var = tk.StringVar()
temp_var = tk.StringVar()

ttk.Label(root, text="Cultivo").grid(row=0, column=0, sticky="e")
ttk.Combobox(root, textvariable=cultivo_var, values=["soja", "maiz", "trigo"], state="readonly").grid(row=0, column=1)

ttk.Label(root, text="Mes").grid(row=1, column=0, sticky="e")
ttk.Entry(root, textvariable=mes_var).grid(row=1, column=1)

ttk.Label(root, text="Lluvia (mm)").grid(row=2, column=0, sticky="e")
ttk.Entry(root, textvariable=lluvia_var).grid(row=2, column=1)

ttk.Label(root, text="Temp (°C)").grid(row=3, column=0, sticky="e")
ttk.Entry(root, textvariable=temp_var).grid(row=3, column=1)

ttk.Button(root, text="Predecir", command=mostrar_prediccion).grid(row=4, column=0, columnspan=2, pady=5)
result_label = ttk.Label(root, text="", font=("Arial", 12))
result_label.grid(row=5, column=0, columnspan=2)

root.mainloop()
