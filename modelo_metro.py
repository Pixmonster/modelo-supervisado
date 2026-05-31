import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


df = pd.read_csv("dataset_metro_madrid_modelo-supervisado.csv")

print()
print("Modelo de aprendizaje supervisado - Metro de Madrid")
print("-" * 52)
print(f"Se cargaron {len(df)} registros con {len(df.columns)} columnas.")
print()


le = LabelEncoder()

columnas_texto = ["Estacion", "Dia_Semana", "Clima"]

for col in columnas_texto:
    df[col] = le.fit_transform(df[col])

df["Nivel_Congestion"] = le.fit_transform(df["Nivel_Congestion"])

print("Las columnas de texto fueron convertidas a valores numericos.")
print("Esto es necesario para que el modelo pueda procesarlas.")
print()


X = df.drop(columns=["Nivel_Congestion"])
y = df["Nivel_Congestion"]

print("Variables usadas para entrenar el modelo:")
for col in X.columns:
    print(f"  - {col}")
print()
print("Variable a predecir: Nivel_Congestion (Alta, Media o Baja)")
print()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Division del dataset:")
print(f"  Entrenamiento : {len(X_train)} registros")
print(f"  Prueba        : {len(X_test)} registros")
print()


modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

print("El modelo fue entrenado usando Random Forest con 100 arboles.")
print()

y_pred = modelo.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("-" * 52)
print("Resultados del modelo")
print("-" * 52)
print()
print(f"Precision general (accuracy): {accuracy * 100:.2f}%")
print()
print("Detalle por categoria:")
print()
print(classification_report(y_test, y_pred,
      target_names=["Alta", "Baja", "Media"]))

print("Que tanto influye cada variable en la prediccion:")
print()
importancias = pd.Series(
    modelo.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

for var, valor in importancias.items():
    barra = "█" * int(valor * 50)
    print(f"  {var:<25} {barra} {valor:.4f}")

print()

cm = confusion_matrix(y_test, y_pred)
etiquetas = ["Alta", "Baja", "Media"]

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar(im)

ax.set(
    xticks=[0, 1, 2],
    yticks=[0, 1, 2],
    xticklabels=etiquetas,
    yticklabels=etiquetas,
    xlabel="Prediccion del modelo",
    ylabel="Valor real",
    title="Matriz de Confusion - Nivel de Congestion"
)

for i in range(3):
    for j in range(3):
        ax.text(j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=14, fontweight="bold")

plt.tight_layout()
plt.savefig("matriz_confusion.png", dpi=150)
print("La matriz de confusion se guardo como imagen en 'matriz_confusion.png'")
print()

ejemplo = pd.DataFrame([{
    "Estacion": 5,
    "Dia_Semana": 1,
    "Hora_Dia": 8,
    "Clima": 1,
    "Torniquetes_Activos": 8,
    "Retraso_Operativo_Min": 0,
    "Pasajeros_Hora": 1800,
    "Tiempo_Espera_Min": 6.0
}])

resultado = modelo.predict(ejemplo)[0]
probabilidades = modelo.predict_proba(ejemplo)[0]

niveles = {0: "Alta", 1: "Baja", 2: "Media"}
nivel_predicho = niveles[resultado]

print("-" * 52)
print("Prediccion de los datos")
print("-" * 52)
print()
print("Condiciones del caso:")
print("  Hora: 8am")
print("  Pasajeros por hora: 1800")
print("  Torniquetes activos: 8")
print("  Retraso: 0 minutos")
print()
print(f"El modelo predice un nivel de congestion: {nivel_predicho}")
print()
print("Probabilidades calculadas:")
print(f"  Alta  : {probabilidades[0]*100:.1f}%")
print(f"  Baja  : {probabilidades[1]*100:.1f}%")
print(f"  Media : {probabilidades[2]*100:.1f}%")
print()