
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import math

# Obtener la carpeta donde se encuentra api.py
CARPETA = Path(__file__).resolve().parent

# Cargar el modelo y las características
modelo = joblib.load(CARPETA / "modelo_cancer.pkl")
columnas = joblib.load(CARPETA / "columnas_modelo.pkl")

# Cargar el conjunto de datos original para consultar ejemplos reales
from sklearn.datasets import load_breast_cancer

datos_originales = load_breast_cancer(as_frame=True)

# DataFrame con las 30 características originales
df_datos = datos_originales.data.copy()

# Convertir las etiquetas originales a los nombres usados en el proyecto
# En este conjunto: 0 = maligno y 1 = benigno
df_datos["diagnostico_real"] = datos_originales.target.map({
    0: "Maligno",
    1: "Benigno"
})


# Crear la aplicación
app = FastAPI(
    title="API de predicción de cáncer",
    description="Proyecto académico de Machine Learning"
)

# Permitir solicitudes desde el HTML durante las pruebas locales
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Definir los datos que recibirá la API
class DatosPaciente(BaseModel):
    parametros: dict[str, float] = Field(
        description="Las 30 características del modelo"
    )


# Ruta principal
@app.get("/")
def inicio():
    return {
        "mensaje": "API funcionando correctamente",
        "total_parametros": len(columnas)
    }


# Consultar las características requeridas
@app.get("/parametros")
def obtener_parametros():
    return {"parametros": columnas}


# Realizar una predicción
@app.post("/predecir")
def predecir(datos: DatosPaciente):
    entrada = datos.parametros

    faltantes = set(columnas) - set(entrada.keys())
    adicionales = set(entrada.keys()) - set(columnas)

    if faltantes or adicionales:
        raise HTTPException(
            status_code=422,
            detail={
                "parametros_faltantes": sorted(faltantes),
                "parametros_adicionales": sorted(adicionales)
            }
        )

    if not all(
        math.isfinite(valor) for valor in entrada.values()
    ):
        raise HTTPException(
            status_code=422,
            detail="Los valores deben ser números finitos."
        )

    # Crear el DataFrame respetando el orden original
    X = pd.DataFrame(
        [[entrada[col] for col in columnas]],
        columns=columnas
    )

    # Obtener la clasificación
    prediccion = int(modelo.predict(X)[0])

    nombres = {
        0: "Maligno",
        1: "Benigno"
    }

    # Obtener las probabilidades estimadas
    probabilidades = modelo.predict_proba(X)[0]

    resultado_probabilidades = {
        nombres[int(clase)]: round(float(prob), 4)
        for clase, prob in zip(modelo.classes_, probabilidades)
    }

    return {
        "prediccion": prediccion,
        "resultado": nombres[prediccion],
        "probabilidades_estimadas": resultado_probabilidades,
        "advertencia": (
            "Resultado experimental. No constituye un diagnóstico médico."
        )
    }


    
# Consultar registros reales del conjunto de prueba

from fastapi import HTTPException

@app.get("/registros-prueba")
def obtener_registros_prueba():
    try:
        # Obtener los registros originales del conjunto de prueba
        registros = X_test.copy()

        # Agregar el diagnóstico real de cada registro
        registros["diagnostico_real"] = list(y_test)

        # Convertir los datos a JSON
        return {
            "total": len(registros),
            "registros": registros.head(20).to_dict(orient="records")
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al consultar los registros: {str(e)}"
        )

        
# Consultar ejemplos reales del conjunto de datos
@app.get("/registros-reales")
def obtener_registros_reales():
    return {
        "total": len(df_datos),
        "registros": df_datos.head(20).to_dict(orient="records")
    }

