# Crear un ejemplo con alto riesgo de diabetes
nuevos_datos_alto_riesgo = pd.DataFrame({
    'GenHlth': [5],          # Muy mala salud general
    'HighBP': [1],           # Alta presión arterial
    'BMI': [45],             # Alto índice de masa corporal
    'DiffWalk': [1],         # Dificultad para caminar
    'HighChol': [1],         # Alto colesterol
    'Age': [10],             # Mayor edad (edad aproximada en un rango alto)
    'HeartDiseaseorAttack': [1], # Historial de problemas cardíacos
    'PhysHlth': [20],        # Mala salud física en general
    'Income': [1],           # Bajo nivel de ingresos, que a veces se asocia con acceso limitado a servicios de salud
    'Diabetes_012': [0]      # Valor predeterminado si no está disponible
}, columns=top_features)

# Predecir la probabilidad de diabetes
probabilidad_diabetes_alto_riesgo = modelo_cargado.predict_proba(nuevos_datos_alto_riesgo)[:, 1]
print("Probabilidad de diabetes (alto riesgo):", probabilidad_diabetes_alto_riesgo)

# Probabilidad de diabetes (alto riesgo): [6.70603486e-05]


Luego asumiendo que esta persona tiene diabetes, podemos ver la probabilidad de tener otra enfermedad

#Ejemplo 3: Persona de edad avanzada con problemas moderados de salud
nuevos_datos_3 = pd.DataFrame({
    'age': [75],                  # Edad avanzada
    'bmi': [32.0],                # IMC alto
    'HbA1c_level': [7.5],         # Moderadamente alto nivel de HbA1c
    'blood_glucose_level': [160]  # Moderadamente alto nivel de glucosa
})

# Calcular la probabilidad
probabilidad_3 = rf_model.predict_proba(nuevos_datos_3)[:, 1]
print("Probabilidad de tener otra enfermedad (Ejemplo 3):", probabilidad_3)

# Probabilidad de tener otra enfermedad (Ejemplo 3): [0.68]


