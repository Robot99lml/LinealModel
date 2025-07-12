# Importación de librerías principales
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Machine Learning - Scikit-learn
from sklearn.model_selection import (train_test_split, cross_val_score, StratifiedKFold, 
                                   GridSearchCV, RandomizedSearchCV, cross_validate)
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler, 
                                 LabelEncoder, OneHotEncoder)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           classification_report, confusion_matrix, roc_auc_score)

# Algoritmos de Clasificación
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                            AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Algoritmos avanzados
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Balanceo de clases
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN

# Utilidades
from scipy import stats
from scipy.stats import entropy
import joblib
from tqdm import tqdm
import time
from collections import defaultdict

# Configuración de visualización
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("Todas las librerías importadas correctamente")
print(f"Versión de pandas: {pd.__version__}")
print(f"Versión de numpy: {np.__version__}")
print(f"Versión de scikit-learn: {__import__('sklearn').__version__}")

# Cargar el dataset
data_path = 'https://raw.githubusercontent.com/Robot99lml/LinealModel/refs/heads/main/Data/hhrr_dataset.csv'
df = pd.read_csv(data_path)

print(" INFORMACIÓN BÁSICA DEL DATASET")
print("=" * 50)
print(f" Dimensiones: {df.shape[0]:,} filas x {df.shape[1]} columnas")
print(f" Tamaño en memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n PRIMERAS 5 FILAS:")
print("=" * 30)
print(df.head())

print("\n TIPOS DE DATOS:")
print("=" * 25)
print(df.dtypes.value_counts())

print("\n DISTRIBUCIÓN DE LA VARIABLE OBJETIVO:")
print("=" * 45)
target_counts = df['Attrition'].value_counts()
target_pct = df['Attrition'].value_counts(normalize=True) * 100

print(f"• No deserción: {target_counts['No']:,} ({target_pct['No']:.1f}%)")
print(f"• Deserción: {target_counts['Yes']:,} ({target_pct['Yes']:.1f}%)")
print(f"• Ratio de desbalanceo: {target_counts['No'] / target_counts['Yes']:.1f}:1")

# Verificar valores nulos
print("\n VALORES NULOS:")
print("=" * 20)
null_counts = df.isnull().sum()
if null_counts.sum() == 0:
    print(" No hay valores nulos en el dataset")
else:
    print(null_counts[null_counts > 0])

print("\n ESTADÍSTICAS DESCRIPTIVAS DE VARIABLES NUMÉRICAS:")
print("=" * 55)
print(df.describe().round(2))

def calculate_entropy(target_variable):
    """
    Calcula la entropía de una variable objetivo binaria.
    
    Fórmula: H(X) = -Σ(p_i * log2(p_i))
    donde p_i es la probabilidad de cada clase
    """
    # Obtener las probabilidades de cada clase
    value_counts = target_variable.value_counts(normalize=True)
    
    # Calcular entropía
    entropy_value = entropy(value_counts, base=2)
    
    return entropy_value, value_counts

# Calcular entropía del dataset
target_entropy, class_probabilities = calculate_entropy(df['Attrition'])

print("ANÁLISIS DE ENTROPÍA DEL DATASET")
print("=" * 40)
print(f"Entropía de la variable objetivo: {target_entropy:.4f}")
print(f"Entropía máxima posible (perfectamente balanceado): {np.log2(2):.4f}")
print(f"Porcentaje de entropía actual: {(target_entropy / np.log2(2)) * 100:.1f}%")

print("\nDISTRIBUCIÓN DE PROBABILIDADES:")
print("=" * 35)
for class_name, probability in class_probabilities.items():
    print(f"• {class_name}: {probability:.4f} ({probability*100:.1f}%)")

print("\nINTERPRETACIÓN DE LA ENTROPÍA:")
print("=" * 35)
if target_entropy > 0.8:
    interpretation = "🟢 EXCELENTE: Los datos están bien balanceados para clasificación"
elif target_entropy > 0.6:
    interpretation = "🟡 BUENO: Ligero desbalanceo, pero manejable"
elif target_entropy > 0.4:
    interpretation = "🟠 MODERADO: Desbalanceo notable, se recomienda técnicas de balanceo"
else:
    interpretation = "🔴 CRÍTICO: Desbalanceo severo, requiere técnicas avanzadas de balanceo"

print(interpretation)

# Visualización de la entropía
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Gráfico de barras de distribución
class_counts = df['Attrition'].value_counts()
colors = ['#FF6B6B', '#4ECDC4']
ax1.bar(class_counts.index, class_counts.values, color=colors, alpha=0.8, edgecolor='black')
ax1.set_title('Distribución de Clases\n(Deserción de Empleados)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Número de Empleados')
ax1.set_xlabel('Attrición')

# Agregar valores en las barras
for i, v in enumerate(class_counts.values):
    ax1.text(i, v + 10, f'{v:,}\n({class_probabilities.iloc[i]*100:.1f}%)', 
             ha='center', va='bottom', fontweight='bold')

# Gráfico de entropía
entropy_data = [target_entropy, np.log2(2) - target_entropy]
entropy_labels = ['Entropía Actual', 'Entropía Restante']
colors2 = ['#FF9F43', '#E8E8E8']

ax2.pie(entropy_data, labels=entropy_labels, colors=colors2, autopct='%1.1f%%', 
        startangle=90, explode=(0.05, 0))
ax2.set_title(f'Análisis de Entropía\n(Valor: {target_entropy:.4f})', 
              fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

print(f"\nCONCLUSIÓN SOBRE LA CALIDAD DE LOS DATOS:")
print("=" * 50)
print(f"La entropía de {target_entropy:.4f} indica que tenemos un dataset con desbalanceo")
print(f"moderado. Esto es típico en problemas de deserción laboral donde la mayoría")
print(f"de empleados no desertan. Implementaremos técnicas de balanceo para optimizar")
print(f"el rendimiento de nuestros modelos de clasificación.")


# Identificar tipos de variables
categorical_vars = df.select_dtypes(include=['object']).columns.tolist()
numerical_vars = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Remover variables que no son útiles para el análisis
variables_to_remove = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
numerical_vars = [var for var in numerical_vars if var not in variables_to_remove]

print("CLASIFICACIÓN DE VARIABLES")
print("=" * 35)
print(f"Variables categóricas ({len(categorical_vars)}): {categorical_vars}")
print(f"Variables numéricas ({len(numerical_vars)}): {numerical_vars}")

# Análisis de variables categóricas
print("\nANÁLISIS DE VARIABLES CATEGÓRICAS VS ATTRITION")
print("=" * 55)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.ravel()

for i, var in enumerate(categorical_vars):
    if i < 6:  # Mostrar solo las primeras 6 variables categóricas
        # Crear tabla cruzada
        crosstab = pd.crosstab(df[var], df['Attrition'], normalize='index') * 100
        
        # Gráfico de barras apiladas
        crosstab.plot(kind='bar', ax=axes[i], color=['#4ECDC4', '#FF6B6B'], 
                      alpha=0.8, edgecolor='black')
        axes[i].set_title(f'{var} vs Attrition', fontweight='bold')
        axes[i].set_xlabel(var)
        axes[i].set_ylabel('Porcentaje (%)')
        axes[i].legend(title='Attrition')
        axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Análisis estadístico de variables categóricas
print("\nANÁLISIS ESTADÍSTICO DE VARIABLES CATEGÓRICAS")
print("=" * 50)

categorical_analysis = {}
for var in categorical_vars:
    # Chi-cuadrado test
    contingency_table = pd.crosstab(df[var], df['Attrition'])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    # Cramér's V (medida de asociación)
    n = contingency_table.sum().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
    
    categorical_analysis[var] = {
        'chi2': chi2,
        'p_value': p_value,
        'cramers_v': cramers_v,
        'significant': p_value < 0.05
    }
    
    print(f"\n• {var}:")
    print(f"  - Chi² = {chi2:.3f}, p-value = {p_value:.6f}")
    print(f"  - Cramér's V = {cramers_v:.3f}")
    print(f"  - Significativo: {'Sí' if p_value < 0.05 else 'No'}")

# Crear DataFrame con resultados
cat_results_df = pd.DataFrame(categorical_analysis).T
cat_results_df = cat_results_df.sort_values('cramers_v', ascending=False)

print(f"\nRANKING DE VARIABLES CATEGÓRICAS (por Cramér's V):")
print("=" * 55)
for i, (var, row) in enumerate(cat_results_df.iterrows(), 1):
    print(f"{i}. {var}: {row['cramers_v']:.3f} {'🔥' if row['cramers_v'] > 0.3 else '📊' if row['cramers_v'] > 0.1 else '📉'}")


# Análisis de variables numéricas
print("\n ANÁLISIS DE VARIABLES NUMÉRICAS VS ATTRITION")
print("=" * 50)

# Codificar Attrition para análisis numérico
df_numeric_analysis = df.copy()
df_numeric_analysis['Attrition_num'] = df_numeric_analysis['Attrition'].map({'No': 0, 'Yes': 1})

# Análisis estadístico de variables numéricas
numerical_analysis = {}
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.ravel()

for i, var in enumerate(numerical_vars[:16]):  # Primeras 16 variables numéricas
    # Test t de Student
    group_no = df[df['Attrition'] == 'No'][var]
    group_yes = df[df['Attrition'] == 'Yes'][var]
    
    t_stat, p_value = stats.ttest_ind(group_no, group_yes)
    
    # Correlación point-biserial
    correlation = stats.pointbiserialr(df_numeric_analysis['Attrition_num'], df_numeric_analysis[var])
    
    numerical_analysis[var] = {
        't_stat': t_stat,
        'p_value': p_value,
        'correlation': correlation.correlation,
        'correlation_p': correlation.pvalue,
        'significant': p_value < 0.05
    }        # Boxplot
    df.boxplot(column=var, by='Attrition', ax=axes[i])
    axes[i].set_title(f'{var}\nr={correlation.correlation:.3f}')
    axes[i].set_xlabel('')

plt.tight_layout()
plt.show()

# Crear DataFrame con resultados numéricos
num_results_df = pd.DataFrame(numerical_analysis).T
num_results_df = num_results_df.reindex(num_results_df['correlation'].abs().sort_values(ascending=False).index)

print(f"\nRANKING DE VARIABLES NUMÉRICAS (por correlación absoluta):")
print("=" * 65)
for i, (var, row) in enumerate(num_results_df.iterrows(), 1):
    corr_abs = abs(row['correlation'])
    emoji = '🔥' if corr_abs > 0.3 else '📊' if corr_abs > 0.15 else '📉'
    significance = 'Si' if row['significant'] else 'No'
    print(f"{i:2d}. {var:<25}: r={row['correlation']:6.3f} {emoji} (p={row['p_value']:.6f} {significance})")

# Matriz de correlación completa
print(f"\n MATRIZ DE CORRELACIÓN DE VARIABLES IMPORTANTES")
print("=" * 50)

# Seleccionar variables más importantes basadas en correlación con Attrition
important_vars = num_results_df.head(10).index.tolist()
important_vars.append('Attrition_num')

correlation_matrix = df_numeric_analysis[important_vars].corr()

# Visualizar matriz de correlación
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
            square=True, fmt='.2f', cbar_kws={"shrink": .8})
plt.title('Matriz de Correlación - Variables Más Importantes', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print(f"\n RESUMEN DE ANÁLISIS EDA:")
print("=" * 30)
print(f"• Variables categóricas más importantes: {cat_results_df.head(3).index.tolist()}")
print(f"• Variables numéricas más importantes: {num_results_df.head(3).index.tolist()}")
print(f"• Variables con correlación alta entre sí (>0.7): ", end="")

# Encontrar correlaciones altas
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.7:
            high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))

if high_corr_pairs:
    for pair in high_corr_pairs:
        print(f"({pair[0]}, {pair[1]})")
else:
    print("Ninguna")

def create_dataset_versions(df):
    """
    Crea 5 versiones diferentes del dataset con diferentes técnicas de preprocesamiento.
    """
    versions = {}
    
    # Preparar datos base
    df_work = df.copy()
    
    # Remover variables irrelevantes identificadas en EDA
    columns_to_remove = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
    df_work = df_work.drop(columns=columns_to_remove, errors='ignore')
    
    # Separar target
    y = df_work['Attrition'].map({'No': 0, 'Yes': 1})
    X = df_work.drop('Attrition', axis=1)
    
    # Identificar variables categóricas y numéricas
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Variables categóricas: {categorical_cols}")
    print(f"Variables numéricas: {numerical_cols}")
    
    # VERSION 1: Dataset Base (solo encoding básico)
    print("\n Creando Versión 1: Dataset Base...")
    X_v1 = X.copy()
    # Label encoding para variables categóricas
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_v1[col] = le.fit_transform(X_v1[col])
        le_dict[col] = le
    
    versions['v1'] = {
        'X': X_v1,
        'y': y,
        'description': 'Dataset base con Label Encoding',
        'features': X_v1.shape[1]
    }
    
    # VERSION 2: StandardScaler
    print("\n Creando Versión 2: StandardScaler...")
    X_v2 = X_v1.copy()
    scaler_v2 = StandardScaler()
    X_v2[numerical_cols] = scaler_v2.fit_transform(X_v2[numerical_cols])
    
    versions['v2'] = {
        'X': X_v2,
        'y': y,
        'description': 'Label Encoding + StandardScaler',
        'features': X_v2.shape[1],
        'scaler': scaler_v2
    }
    
    # VERSION 3: RobustScaler (mejor para outliers)
    print("\n Creando Versión 3: RobustScaler...")
    X_v3 = X_v1.copy()
    scaler_v3 = RobustScaler()
    X_v3[numerical_cols] = scaler_v3.fit_transform(X_v3[numerical_cols])
    
    versions['v3'] = {
        'X': X_v3,
        'y': y,
        'description': 'Label Encoding + RobustScaler',
        'features': X_v3.shape[1],
        'scaler': scaler_v3
    }
    
    # VERSION 4: Feature Selection (solo variables importantes)
    print("\n Creando Versión 4: Feature Selection...")
    
    # Seleccionar top variables categóricas y numéricas basado en análisis EDA
    top_categorical = ['OverTime', 'BusinessTravel', 'JobRole', 'MaritalStatus']
    top_numerical = ['Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany', 
                    'DistanceFromHome', 'JobLevel', 'StockOptionLevel', 'WorkLifeBalance']
    
    # Asegurar que las columnas existen
    selected_features = []
    for col in top_categorical + top_numerical:
        if col in X_v2.columns:
            selected_features.append(col)
    
    X_v4 = X_v2[selected_features].copy()
    
    versions['v4'] = {
        'X': X_v4,
        'y': y,
        'description': f'Feature Selection: {len(selected_features)} variables importantes',
        'features': X_v4.shape[1],
        'selected_features': selected_features
    }
    
    # VERSION 5: Feature Engineering + PCA
    print("\n Creando Versión 5: Feature Engineering + PCA...")
    X_v5 = X_v2.copy()

    # Feature Engineering: crear variables derivadas
    if 'Age' in X_v5.columns and 'TotalWorkingYears' in X_v5.columns:
        X_v5['Experience_to_Age_Ratio'] = X_v5['TotalWorkingYears'] / (X_v5['Age'] + 1)
    
    if 'MonthlyIncome' in X_v5.columns and 'Age' in X_v5.columns:
        X_v5['Income_per_Age'] = X_v5['MonthlyIncome'] / (X_v5['Age'] + 1)
        
    if 'YearsAtCompany' in X_v5.columns and 'TotalWorkingYears' in X_v5.columns:
        X_v5['Company_Experience_Ratio'] = X_v5['YearsAtCompany'] / (X_v5['TotalWorkingYears'] + 1)
    
    # Aplicar PCA para reducir dimensionalidad manteniendo 95% de varianza
    from sklearn.decomposition import PCA
    pca = PCA(n_components=0.95, random_state=42)
    X_v5_pca = pca.fit_transform(X_v5)
    X_v5_df = pd.DataFrame(X_v5_pca, columns=[f'PC{i+1}' for i in range(X_v5_pca.shape[1])])
    
    versions['v5'] = {
        'X': X_v5_df,
        'y': y,
        'description': f'Feature Engineering + PCA ({X_v5_pca.shape[1]} componentes)',
        'features': X_v5_df.shape[1],
        'pca': pca,
        'variance_explained': pca.explained_variance_ratio_.sum()
    }
    
    return versions

# Crear las versiones del dataset
print("CREANDO 5 VERSIONES DEL DATASET")
print("=" * 40)
dataset_versions = create_dataset_versions(df)

# Mostrar resumen de versiones
print("RESUMEN DE VERSIONES CREADAS:")
print("=" * 35)
for version_name, version_data in dataset_versions.items():
    print(f"{version_name.upper()}:")
    print(f"   Descripción: {version_data['description']}")
    print(f"   Características: {version_data['features']}")
    print(f"   Dimensiones: {version_data['X'].shape}")
    
    if 'variance_explained' in version_data:
        print(f"   Varianza explicada: {version_data['variance_explained']:.1%}")

print(f"Se crearon {len(dataset_versions)} versiones del dataset exitosamente!")

def apply_balancing_techniques(X, y):
    """
    Aplica diferentes técnicas de balanceo de clases y las evalúa.
    """
    balancing_techniques = {}
    
    print("APLICANDO TÉCNICAS DE BALANCEO")
    print("=" * 40)
    
    # Datos originales (sin balanceo)
    print("Dataset original:")
    unique, counts = np.unique(y, return_counts=True)
    print(f"   Clase 0 (No deserción): {counts[0]:,}")
    print(f"   Clase 1 (Deserción): {counts[1]:,}")
    print(f"   Ratio: {counts[0]/counts[1]:.1f}:1")
    
    balancing_techniques['original'] = {
        'X': X,
        'y': y,
        'description': 'Sin balanceo',
        'class_distribution': dict(zip(unique, counts))
    }
    
    # SMOTE
    print("Aplicando SMOTE...")
    smote = SMOTE(random_state=42, k_neighbors=3)
    try:
        X_smote, y_smote = smote.fit_resample(X, y)
        unique, counts = np.unique(y_smote, return_counts=True)
        print(f"   Resultado: {counts[0]:,} | {counts[1]:,}")
        
        balancing_techniques['smote'] = {
            'X': X_smote,
            'y': y_smote,
            'description': 'SMOTE (Synthetic Minority Oversampling)',
            'class_distribution': dict(zip(unique, counts))
        }
    except Exception as e:
        print(f"  Error en SMOTE: {e}")
    
    # ADASYN
    print("Aplicando ADASYN...")
    adasyn = ADASYN(random_state=42, n_neighbors=3)
    try:
        X_adasyn, y_adasyn = adasyn.fit_resample(X, y)
        unique, counts = np.unique(y_adasyn, return_counts=True)
        print(f"   Resultado: {counts[0]:,} | {counts[1]:,}")
        
        balancing_techniques['adasyn'] = {
            'X': X_adasyn,
            'y': y_adasyn,
            'description': 'ADASYN (Adaptive Synthetic Sampling)',
            'class_distribution': dict(zip(unique, counts))
        }
    except Exception as e:
        print(f"   ❌ Error en ADASYN: {e}")
    
    # BorderlineSMOTE
    print("Aplicando BorderlineSMOTE...")
    borderline_smote = BorderlineSMOTE(random_state=42, k_neighbors=3)
    try:
        X_borderline, y_borderline = borderline_smote.fit_resample(X, y)
        unique, counts = np.unique(y_borderline, return_counts=True)
        print(f"   Resultado: {counts[0]:,} | {counts[1]:,}")
        
        balancing_techniques['borderline_smote'] = {
            'X': X_borderline,
            'y': y_borderline,
            'description': 'BorderlineSMOTE (SMOTE para casos borderline)',
            'class_distribution': dict(zip(unique, counts))
        }
    except Exception as e:
        print(f"   Error en BorderlineSMOTE: {e}")
    
    # SMOTETomek
    print("Aplicando SMOTETomek...")
    smote_tomek = SMOTETomek(random_state=42)
    try:
        X_smote_tomek, y_smote_tomek = smote_tomek.fit_resample(X, y)
        unique, counts = np.unique(y_smote_tomek, return_counts=True)
        print(f"   Resultado: {counts[0]:,} | {counts[1]:,}")
        
        balancing_techniques['smote_tomek'] = {
            'X': X_smote_tomek,
            'y': y_smote_tomek,
            'description': 'SMOTETomek (SMOTE + Tomek Links)',
            'class_distribution': dict(zip(unique, counts))
        }
    except Exception as e:
        print(f"   Error en SMOTETomek: {e}")
    
    return balancing_techniques

def evaluate_balancing_quick(balancing_techniques, dataset_version='v2'):
    """
    Evaluación rápida de técnicas de balanceo usando Random Forest.
    """
    print(f"EVALUACIÓN RÁPIDA DE TÉCNICAS DE BALANCEO")
    print("=" * 50)
    
    results = {}
    
    for technique_name, technique_data in balancing_techniques.items():
        print(f"Evaluando: {technique_data['description']}")
        
        X = technique_data['X']
        y = technique_data['y']
        
        # Split de datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Modelo simple para evaluación
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[technique_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'description': technique_data['description']
        }
        
        print(f"   Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
    
    # Ordenar por F1-score
    sorted_results = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
    
    print(f"RANKING DE TÉCNICAS DE BALANCEO (por F1-Score):")
    print("=" * 55)
    for i, (technique, metrics) in enumerate(sorted_results, 1):
        print(f"{i}. {technique.upper()}: F1={metrics['f1']:.3f} " if i == 1 
              else f"{i}. {technique.upper()}: F1={metrics['f1']:.3f}")
    
    return results, sorted_results[0][0]  # Retorna resultados y mejor técnica

# Aplicar técnicas de balanceo en la versión 2 del dataset (StandardScaler)
print("SELECCIÓN DE VERSIÓN DEL DATASET PARA BALANCEO")
print("=" * 50)
print("Utilizaremos la Versión 2 (StandardScaler) porque:")
print("• Es un buen balance entre preprocesamiento y simplicidad")
print("• StandardScaler es compatible con técnicas de balanceo")
print("• Mantiene todas las variables para evaluación completa")

X_for_balancing = dataset_versions['v2']['X']
y_for_balancing = dataset_versions['v2']['y']

# Aplicar técnicas de balanceo
balancing_techniques = apply_balancing_techniques(X_for_balancing, y_for_balancing)

# Evaluación rápida
balance_evaluation, best_balancing = evaluate_balancing_quick(balancing_techniques)

print(f"TÉCNICA DE BALANCEO SELECCIONADA: {best_balancing.upper()}")
print(f"Justificación: Obtuvo el mejor F1-Score en la evaluación rápida")
print(f"Esta técnica será aplicada a todas las versiones del dataset")

def get_algorithms():
    """
    Define todos los algoritmos de clasificación con sus configuraciones base.
    """
    algorithms = {
        'random_forest': {
            'model': RandomForestClassifier(random_state=42, n_jobs=-1),
            'params_random': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'params_grid': {
                'n_estimators': [200, 300],
                'max_depth': [20, 30],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        },
        'xgboost': {
            'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'params_random': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.1, 0.2, 0.3],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'params_grid': {
                'n_estimators': [200, 300],
                'max_depth': [5, 7],
                'learning_rate': [0.1, 0.2]
            }
        },
        'lightgbm': {
            'model': lgb.LGBMClassifier(random_state=42, verbose=-1),
            'params_random': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 100],
                'feature_fraction': [0.8, 0.9, 1.0]
            },
            'params_grid': {
                'n_estimators': [200, 300],
                'max_depth': [5, 7],
                'learning_rate': [0.1, 0.2]
            }
        },
        'catboost': {
            'model': CatBoostClassifier(random_state=42, verbose=False),
            'params_random': {
                'iterations': [100, 200, 300],
                'depth': [4, 6, 8],
                'learning_rate': [0.01, 0.1, 0.2],
                'l2_leaf_reg': [1, 3, 5]
            },
            'params_grid': {
                'iterations': [200, 300],
                'depth': [6, 8],
                'learning_rate': [0.1, 0.2]
            }
        },
        'logistic_regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params_random': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'params_grid': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2']
            }
        },
        'svm': {
            'model': SVC(random_state=42, probability=True),
            'params_random': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'poly']
            },
            'params_grid': {
                'C': [1, 10],
                'gamma': ['scale', 0.01],
                'kernel': ['rbf']
            }
        },
        'knn': {
            'model': KNeighborsClassifier(),
            'params_random': {
                'n_neighbors': [3, 5, 7, 9, 11, 15],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            },
            'params_grid': {
                'n_neighbors': [5, 7, 9],
                'weights': ['uniform', 'distance']
            }
        },
        'naive_bayes': {
            'model': GaussianNB(),
            'params_random': {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
            },
            'params_grid': {
                'var_smoothing': [1e-9, 1e-8, 1e-7]
            }
        },
        'decision_tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params_random': {
                'max_depth': [None, 5, 10, 15, 20],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 5, 10],
                'criterion': ['gini', 'entropy']
            },
            'params_grid': {
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5, 10]
            }
        },
        'gradient_boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params_random': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'min_samples_split': [2, 5, 10]
            },
            'params_grid': {
                'n_estimators': [200, 300],
                'max_depth': [5, 7],
                'learning_rate': [0.1, 0.2]
            }
        },
        'extra_trees': {
            'model': ExtraTreesClassifier(random_state=42, n_jobs=-1),
            'params_random': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'params_grid': {
                'n_estimators': [200, 300],
                'max_depth': [20, 30],
                'min_samples_split': [2, 5]
            }
        },
        'mlp': {
            'model': MLPClassifier(random_state=42, max_iter=500),
            'params_random': {
                'hidden_layer_sizes': [(100,), (100, 50), (150, 100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            },
            'params_grid': {
                'hidden_layer_sizes': [(100,), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001]
            }
        }
    }
    
    return algorithms

# Configurar validación cruzada
def setup_cross_validation():
    """
    Configura la estrategia de validación cruzada.
    
    Utilizaré StratifiedKFold con 10 folds porque:
    - Mantiene la proporción de clases en cada fold
    - 10 folds es un estándar que balancea bias-variance
    - Es más robusto que hold-out para datasets medianos
    """
    cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    print("CONFIGURACIÓN DE VALIDACIÓN CRUZADA")
    print("=" * 40)
    print("Estrategia: StratifiedKFold")
    print("Número de folds: 10")
    print("Shuffle: Activado")
    print("Semilla: 42")
    print("Ventajas:")
    print("   • Mantiene proporción de clases")
    print("   • Reduce variabilidad de resultados")
    print("   • Estándar de la industria")
    
    return cv_strategy

# Sistema de tracking de experimentos
class ExperimentTracker:
    """
    Clase para trackear todos los experimentos realizados.
    """
    def __init__(self):
        self.experiments = []
        self.experiment_id = 0
    
    def log_experiment(self, algorithm, dataset_version, balancing_technique, 
                      search_type, params, metrics, execution_time):
        """
        Registra un experimento con todos sus detalles.
        """
        self.experiment_id += 1
        
        experiment = {
            'id': self.experiment_id,
            'algorithm': algorithm,
            'dataset_version': dataset_version,
            'balancing_technique': balancing_technique,
            'search_type': search_type,  # 'random' o 'grid'
            'best_params': params,
            'metrics': metrics,
            'execution_time': execution_time,
            'timestamp': time.time()
        }
        
        self.experiments.append(experiment)
        
        return self.experiment_id
    
    def get_summary_df(self):
        """
        Retorna un DataFrame con resumen de todos los experimentos.
        """
        if not self.experiments:
            return pd.DataFrame()
        
        summary_data = []
        for exp in self.experiments:
            row = {
                'ID': exp['id'],
                'Algorithm': exp['algorithm'],
                'Dataset': exp['dataset_version'],
                'Balancing': exp['balancing_technique'],
                'Search': exp['search_type'],
                'F1_Score': exp['metrics']['f1'],
                'Accuracy': exp['metrics']['accuracy'],
                'Precision': exp['metrics']['precision'],
                'Recall': exp['metrics']['recall'],
                'Time_seconds': exp['execution_time']
            }
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def get_best_experiments(self, metric='f1', top_n=10):
        """
        Retorna los mejores experimentos según una métrica.
        """
        summary_df = self.get_summary_df()
        if summary_df.empty:
            return summary_df
        
        metric_col = metric.title() + '_Score' if metric == 'f1' else metric.title()
        return summary_df.nlargest(top_n, metric_col)

# Inicializar componentes
print("INICIALIZANDO SISTEMA DE EXPERIMENTACIÓN")
print("=" * 45)

algorithms = get_algorithms()
#cv_strategy = setup_cross_validation()
tracker = ExperimentTracker()

print(f"Sistema inicializado:")
print(f"   Algoritmos configurados: {len(algorithms)}")
print(f"   Versiones de dataset: {len(dataset_versions)}")
print(f"   Técnicas de balanceo: {len(balancing_techniques)}")
#print(f"   Validación cruzada: {cv_strategy.n_splits}-fold")
#print(f"   Tracker de experimentos: Activo")

def run_hybrid_nested_experiments(max_experiments=1000, selected_balancing_technique='borderline_smote'):
    """
     EXPERIMENTACIÓN HÍBRIDA CON VALIDACIÓN CRUZADA ANIDADA Y TRAZABILIDAD COMPLETA
    
    METODOLOGÍA HÍBRIDA AVANZADA:
    1. Random Search → Grid Search Inteligente (automático por algoritmo)
    2. Validación Cruzada Anidada: CV externa (evaluación) + CV interna (selección)
    3. Trazabilidad completa: cada fold es analizable individualmente
    4. Búsqueda híbrida: cada modelo tiene su propio Random Search seguido de Grid inteligente
    
    Parámetros:
    - max_experiments: Número máximo de experimentos
    - selected_balancing_technique: Técnica de balanceo a usar
    
    Retorna:
    - tracker con funciones adicionales para análisis híbrido
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.base import clone
    from collections import defaultdict
    import numpy as np
    import time
    
    print("EXPERIMENTACIÓN HÍBRIDA: VALIDACIÓN CRUZADA ANIDADA + TRAZABILIDAD")
    print("=" * 75)
    print(f"Experimentos máximos: {max_experiments}")
    print(f"Balanceo: {selected_balancing_technique.upper()}")
    print(f"CV: Anidada (5 externa × 3 interna = 15 evaluaciones/experimento)")
    print(f"Búsqueda: Random → Grid Inteligente (híbrido por algoritmo)")
    print(f"Trazabilidad: Completa por fold con análisis de estabilidad")
    print()
    
    # Verificar técnica de balanceo
    if selected_balancing_technique not in balancing_techniques:
        print(f"Error: Técnica '{selected_balancing_technique}' no encontrada")
        print(f"Disponibles: {list(balancing_techniques.keys())}")
        return None
    
    # Configuración de validación cruzada anidada
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  # CV externa
    inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  # CV interna
    
    # Inicialización
    experiment_count = 0
    start_time = time.time()
    
    # Storage para resultados híbridos y trazabilidad
    nested_cv_results = {}
    fold_level_tracking = {}
    hybrid_search_results = defaultdict(dict)
    algorithm_random_results = defaultdict(list)
    
    print("CONFIGURACIÓN VALIDACIÓN CRUZADA ANIDADA:")
    print(f"   CV Externa: {outer_cv.n_splits} folds → Evaluación final del modelo")
    print(f"   CV Interna: {inner_cv.n_splits} folds → Selección hiperparámetros")
    print(f"   Total: {outer_cv.n_splits * inner_cv.n_splits} entrenamientos por experimento")
    print()
    
    # Distribución inteligente de experimentos
    n_algorithms = len(algorithms)
    n_datasets = len(dataset_versions)
    experiments_per_combo = max_experiments // (n_algorithms * n_datasets * 2)  # 2 fases por combo
    
    print(f"DISTRIBUCIÓN DE EXPERIMENTOS:")
    print(f"   Algoritmos: {n_algorithms}")
    print(f"   Datasets: {n_datasets}")
    print(f"   Fases por combo: 2 (Random + Grid)")
    print(f"   Experimentos por combinación: ~{experiments_per_combo}")
    print()
    
    # EXPERIMENTACIÓN HÍBRIDA POR ALGORITMO
    for algo_name, algo_config in algorithms.items():
        if experiment_count >= max_experiments:
            break
        
        print(f" ALGORITMO: {algo_name.upper()}")
        print("=" * 60)
        
        algo_start_time = time.time()
        algo_experiments = 0
        
        # EXPERIMENTACIÓN POR DATASET
        for dataset_name, dataset_info in dataset_versions.items():
            if experiment_count >= max_experiments:
                break
            
            print(f"   Dataset: {dataset_name}")
            
            # Preparar datos
            X_dataset = dataset_info['X']
            y_dataset = dataset_info['y']
            
            # Aplicar balanceo
            print(f"       Aplicando {selected_balancing_technique.upper()}... ", end="")
            try:
                if selected_balancing_technique == 'original':
                    X_balanced = X_dataset
                    y_balanced = y_dataset
                    print("Sin balanceo (datos originales)")
                else:
                    # Seleccionar balanceador
                    if selected_balancing_technique == 'smote':
                        balancer = SMOTE(random_state=42, k_neighbors=3)
                    elif selected_balancing_technique == 'adasyn':
                        balancer = ADASYN(random_state=42, n_neighbors=3)
                    elif selected_balancing_technique == 'borderline_smote':
                        balancer = BorderlineSMOTE(random_state=42, k_neighbors=3)
                    elif selected_balancing_technique == 'smote_tomek':
                        balancer = SMOTETomek(random_state=42)
                    else:
                        balancer = SMOTE(random_state=42, k_neighbors=3)
                    
                    X_balanced, y_balanced = balancer.fit_resample(X_dataset, y_dataset)
                    print(f" {X_balanced.shape[0]} muestras balanceadas")
                    
            except Exception as e:
                print(f" Error: {str(e)[:50]}")
                X_balanced = X_dataset
                y_balanced = y_dataset
            
            # FASE 1: RANDOM SEARCH CON VALIDACIÓN CRUZADA ANIDADA
            print(f"      🎲 Random Search + CV Anidada... ", end="")
            try:
                rs_start = time.time()
                
                # Validación cruzada anidada para Random Search
                outer_fold_results = {}
                random_search_metrics = []
                
                for outer_fold, (train_outer, test_outer) in enumerate(outer_cv.split(X_balanced, y_balanced)):
                    # División externa
                    X_train_outer = X_balanced.iloc[train_outer]
                    X_test_outer = X_balanced.iloc[test_outer]
                    y_train_outer = y_balanced.iloc[train_outer]
                    y_test_outer = y_balanced.iloc[test_outer]
                    
                    # Random Search en validación cruzada interna
                    random_search = RandomizedSearchCV(
                        algo_config['model'],
                        algo_config['params_random'],
                        n_iter=12,  # Balanceado para eficiencia
                        cv=inner_cv,
                        scoring='f1',
                        n_jobs=-1,
                        random_state=42 + outer_fold
                    )
                    
                    random_search.fit(X_train_outer, y_train_outer)
                    
                    # Evaluar en fold externo con mejor modelo encontrado
                    best_model = random_search.best_estimator_
                    y_pred_outer = best_model.predict(X_test_outer)
                    
                    # Métricas detalladas del fold externo
                    fold_metrics = {
                        'accuracy': accuracy_score(y_test_outer, y_pred_outer),
                        'precision': precision_score(y_test_outer, y_pred_outer, zero_division=0),
                        'recall': recall_score(y_test_outer, y_pred_outer, zero_division=0),
                        'f1': f1_score(y_test_outer, y_pred_outer, zero_division=0),
                        'best_params': random_search.best_params_,
                        'cv_score': random_search.best_score_,
                        'samples': len(y_test_outer),
                        'positives_true': int(sum(y_test_outer)),
                        'positives_pred': int(sum(y_pred_outer)),
                        'fold_index': outer_fold
                    }
                    print(fold_metrics)
                    outer_fold_results[f'outer_fold_{outer_fold}'] = fold_metrics
                    random_search_metrics.append(fold_metrics['f1'])
                    
                    # Guardar parámetros prometedores para Grid Search
                    algorithm_random_results[f"{algo_name}_{dataset_name}"].append({
                        'params': random_search.best_params_,
                        'f1_score': fold_metrics['f1'],
                        'cv_score': random_search.best_score_,
                        'outer_fold': outer_fold
                    })
                
                # Estadísticas agregadas Random Search
                rs_stats = {
                    'f1_mean': np.mean(random_search_metrics),
                    'f1_std': np.std(random_search_metrics),
                    'f1_min': np.min(random_search_metrics),
                    'f1_max': np.max(random_search_metrics),
                    'f1_cv': np.std(random_search_metrics) / (np.mean(random_search_metrics) + 1e-8),
                    'stability': 1.0 - (np.std(random_search_metrics) / (np.mean(random_search_metrics) + 1e-8))
                }
                
                rs_time = time.time() - rs_start
                
                # Registrar experimento Random Search
                rs_experiment_id = tracker.log_experiment(
                    algorithm=algo_name,
                    dataset_version=dataset_name,
                    balancing_technique=selected_balancing_technique,
                    search_type='random_nested_cv',
                    params=algorithm_random_results[f"{algo_name}_{dataset_name}"][-1]['params'],
                    metrics={
                        'accuracy': np.mean([fold['accuracy'] for fold in outer_fold_results.values()]),
                        'precision': np.mean([fold['precision'] for fold in outer_fold_results.values()]),
                        'recall': np.mean([fold['recall'] for fold in outer_fold_results.values()]),
                        'f1': rs_stats['f1_mean']
                    },
                    execution_time=rs_time
                )
                
                # Guardar trazabilidad detallada
                fold_level_tracking[rs_experiment_id] = {
                    'algorithm': algo_name,
                    'dataset': dataset_name,
                    'search_type': 'random_nested_cv',
                    'balancing_technique': selected_balancing_technique,
                    'outer_fold_results': outer_fold_results,
                    'aggregated_stats': rs_stats,
                    'execution_time': rs_time,
                    'cv_configuration': {
                        'outer_splits': outer_cv.n_splits,
                        'inner_splits': inner_cv.n_splits,
                        'total_evaluations': outer_cv.n_splits * inner_cv.n_splits
                    }
                }
                
                experiment_count += 1
                algo_experiments += 1
                
                print(f"({rs_time:.1f}s)")
                print(f"         F1: {rs_stats['f1_mean']:.3f} ± {rs_stats['f1_std']:.3f}")
                print(f"         Rango: [{rs_stats['f1_min']:.3f}, {rs_stats['f1_max']:.3f}]")
                print(f"         Estabilidad: {rs_stats['stability']:.3f}")
                
            except Exception as e:
                print(f"Error Random Search: {str(e)[:50]}")
            
            # FASE 2: GRID SEARCH INTELIGENTE (solo si hay resultados de Random Search)
            algo_key = f"{algo_name}_{dataset_name}"
            if algo_key in algorithm_random_results and algorithm_random_results[algo_key] and experiment_count < max_experiments:
                print(f"      Grid Search Inteligente... ", end="")
                
                try:
                    gs_start = time.time()
                    
                    # Generar grid inteligente basado en mejores Random Search
                    top_random = sorted(algorithm_random_results[algo_key], 
                                      key=lambda x: x['f1_score'], reverse=True)[:3]
                    
                    # Construir grid expandido inteligentemente
                    param_groups = defaultdict(list)
                    for result in top_random:
                        for param_name, value in result['params'].items():
                            param_groups[param_name].append(value)
                    
                    intelligent_grid = {}
                    for param_name, values in param_groups.items():
                        unique_values = list(set(values))
                        
                        if len(unique_values) == 1:
                            # Expandir alrededor del valor único
                            base_val = unique_values[0]
                            if isinstance(base_val, (int, float)):
                                if isinstance(base_val, int):
                                    expansion = max(1, int(base_val * 0.3))
                                    intelligent_grid[param_name] = [
                                        max(1, base_val - expansion),
                                        base_val,
                                        base_val + expansion
                                    ]
                                else:
                                    expansion = base_val * 0.3
                                    intelligent_grid[param_name] = [
                                        max(0.001, base_val - expansion),
                                        base_val,
                                        base_val + expansion
                                    ]
                            else:
                                intelligent_grid[param_name] = [base_val]
                        else:
                            # Usar valores únicos encontrados
                            intelligent_grid[param_name] = unique_values
                    
                    # Grid Search con validación cruzada anidada
                    outer_fold_results_gs = {}
                    grid_search_metrics = []
                    
                    for outer_fold, (train_outer, test_outer) in enumerate(outer_cv.split(X_balanced, y_balanced)):
                        X_train_outer = X_balanced.iloc[train_outer]
                        X_test_outer = X_balanced.iloc[test_outer]
                        y_train_outer = y_balanced.iloc[train_outer]
                        y_test_outer = y_balanced.iloc[test_outer]
                        
                        # Grid Search en validación cruzada interna
                        grid_search = GridSearchCV(
                            algo_config['model'],
                            intelligent_grid,
                            cv=inner_cv,
                            scoring='f1',
                            n_jobs=-1
                        )
                        
                        grid_search.fit(X_train_outer, y_train_outer)
                        
                        # Evaluar en fold externo
                        best_model_gs = grid_search.best_estimator_
                        y_pred_outer_gs = best_model_gs.predict(X_test_outer)
                        
                        # Métricas detalladas del fold externo
                        fold_metrics_gs = {
                            'accuracy': accuracy_score(y_test_outer, y_pred_outer_gs),
                            'precision': precision_score(y_test_outer, y_pred_outer_gs, zero_division=0),
                            'recall': recall_score(y_test_outer, y_pred_outer_gs, zero_division=0),
                            'f1': f1_score(y_test_outer, y_pred_outer_gs, zero_division=0),
                            'best_params': grid_search.best_params_,
                            'cv_score': grid_search.best_score_,
                            'samples': len(y_test_outer),
                            'positives_true': int(sum(y_test_outer)),
                            'positives_pred': int(sum(y_pred_outer_gs)),
                            'fold_index': outer_fold
                        }
                        
                        outer_fold_results_gs[f'outer_fold_{outer_fold}'] = fold_metrics_gs
                        grid_search_metrics.append(fold_metrics_gs['f1'])
                    
                    # Estadísticas agregadas Grid Search
                    gs_stats = {
                        'f1_mean': np.mean(grid_search_metrics),
                        'f1_std': np.std(grid_search_metrics),
                        'f1_min': np.min(grid_search_metrics),
                        'f1_max': np.max(grid_search_metrics),
                        'f1_cv': np.std(grid_search_metrics) / (np.mean(grid_search_metrics) + 1e-8),
                        'stability': 1.0 - (np.std(grid_search_metrics) / (np.mean(grid_search_metrics) + 1e-8))
                    }
                    
                    gs_time = time.time() - gs_start
                    
                    # Registrar experimento Grid Search
                    gs_experiment_id = tracker.log_experiment(
                        algorithm=algo_name,
                        dataset_version=dataset_name,
                        balancing_technique=selected_balancing_technique,
                        search_type='intelligent_grid_nested_cv',
                        params=outer_fold_results_gs['outer_fold_0']['best_params'],
                        metrics={
                            'accuracy': np.mean([fold['accuracy'] for fold in outer_fold_results_gs.values()]),
                            'precision': np.mean([fold['precision'] for fold in outer_fold_results_gs.values()]),
                            'recall': np.mean([fold['recall'] for fold in outer_fold_results_gs.values()]),
                            'f1': gs_stats['f1_mean']
                        },
                        execution_time=gs_time
                    )
                    
                    # Guardar trazabilidad detallada
                    fold_level_tracking[gs_experiment_id] = {
                        'algorithm': algo_name,
                        'dataset': dataset_name,
                        'search_type': 'intelligent_grid_nested_cv',
                        'balancing_technique': selected_balancing_technique,
                        'outer_fold_results': outer_fold_results_gs,
                        'aggregated_stats': gs_stats,
                        'execution_time': gs_time,
                        'intelligent_grid_used': intelligent_grid,
                        'based_on_random_results': len(top_random),
                        'cv_configuration': {
                            'outer_splits': outer_cv.n_splits,
                            'inner_splits': inner_cv.n_splits,
                            'total_evaluations': outer_cv.n_splits * inner_cv.n_splits
                        }
                    }
                    
                    # Calcular mejora sobre Random Search
                    rs_f1 = rs_stats['f1_mean']
                    gs_f1 = gs_stats['f1_mean']
                    improvement = ((gs_f1 - rs_f1) / rs_f1 * 100) if rs_f1 > 0 else 0
                    
                    # Guardar comparación híbrida
                    hybrid_search_results[algo_name][dataset_name] = {
                        'random_f1_mean': rs_f1,
                        'random_f1_std': rs_stats['f1_std'],
                        'random_stability': rs_stats['stability'],
                        'grid_f1_mean': gs_f1,
                        'grid_f1_std': gs_stats['f1_std'],
                        'grid_stability': gs_stats['stability'],
                        'improvement_pct': improvement,
                        'stability_improvement': gs_stats['stability'] - rs_stats['stability']
                    }
                    
                    experiment_count += 1
                    algo_experiments += 1
                    
                    print(f"({gs_time:.1f}s)")
                    print(f"         F1: {gs_stats['f1_mean']:.3f} ± {gs_stats['f1_std']:.3f}")
                    print(f"         Mejora: {improvement:+.2f}% vs Random Search")
                    print(f"         Estabilidad: {gs_stats['stability']:.3f} ({(gs_stats['stability'] - rs_stats['stability']):+.3f})")
                    print(f"         Grid usado: {len(intelligent_grid)} parámetros")
                    
                except Exception as e:
                    print(f"Error Grid Search: {str(e)[:50]}")
        
        algo_time = time.time() - algo_start_time
        print(f"   Total {algo_name}: {algo_time:.1f}s | Experimentos: {algo_experiments}")
        print()
    
    total_time = time.time() - start_time
    
    # RESUMEN FINAL DETALLADO
    print("EXPERIMENTACIÓN HÍBRIDA CON CV ANIDADA COMPLETADA")
    print("="*65)
    print(f"Experimentos realizados: {experiment_count}")
    print(f"Tiempo total: {total_time/60:.1f} minutos ({total_time/experiment_count:.1f}s/exp)")
    print(f"Validación: Cruzada Anidada (5×3 = 15 evaluaciones/experimento)")
    print(f"Trazabilidad: {len(fold_level_tracking)} experimentos con detalles por fold")
    print(f"Búsquedas híbridas: {len(hybrid_search_results)} algoritmos")
    print(f"Balanceo: {selected_balancing_technique.upper()}")
    
    # Resumen de mejoras híbridas
    if hybrid_search_results:
        print(f"RESUMEN MEJORAS HÍBRIDAS (Grid vs Random):")
        print("-" * 55)
        total_improvements = []
        stability_improvements = []
        
        for algo_name, datasets in hybrid_search_results.items():
            print(f"   {algo_name.upper()}:")
            for dataset_name, results in datasets.items():
                improvement = results['improvement_pct']
                stab_improvement = results['stability_improvement']
                total_improvements.append(improvement)
                stability_improvements.append(stab_improvement)
                
                print(f"      {dataset_name}: F1 {improvement:+.2f}%, Estabilidad {stab_improvement:+.3f}")
        
        if total_improvements:
            avg_improvement = np.mean(total_improvements)
            avg_stability_improvement = np.mean(stability_improvements)
            positive_improvements = sum(1 for x in total_improvements if x > 0)
            
            print(f"ESTADÍSTICAS GENERALES:")
            print(f"   Mejora F1 promedio: {avg_improvement:+.2f}%")
            print(f"   Mejora estabilidad promedio: {avg_stability_improvement:+.3f}")
            print(f"   Casos con mejora: {positive_improvements}/{len(total_improvements)} ({positive_improvements/len(total_improvements)*100:.1f}%)")
    
    # FUNCIONES DE ANÁLISIS AVANZADO
    def get_nested_cv_analysis(experiment_id):
        """Análisis detallado de validación cruzada anidada por experimento."""
        if experiment_id not in fold_level_tracking:
            print(f"Experimento {experiment_id} no encontrado en trazabilidad")
            print(f"IDs disponibles: {list(fold_level_tracking.keys())[:10]}...")
            return None
        
        data = fold_level_tracking[experiment_id]
        
        print(f"ANÁLISIS VALIDACIÓN CRUZADA ANIDADA - Experimento {experiment_id}")
        print("="*70)
        print(f"Algoritmo: {data['algorithm']}")
        print(f"Dataset: {data['dataset']}")
        print(f"Búsqueda: {data['search_type']}")
        print(f"Balanceo: {data['balancing_technique']}")
        print(f"Tiempo: {data['execution_time']:.2f}s")
        
        cv_config = data['cv_configuration']
        print(f"CONFIGURACIÓN VALIDACIÓN CRUZADA:")
        print(f"   CV Externa: {cv_config['outer_splits']} folds (evaluación final)")
        print(f"   CV Interna: {cv_config['inner_splits']} folds (selección hiperparámetros)")
        print(f"   Total evaluaciones: {cv_config['total_evaluations']}")
        
        print(f"RESULTADOS POR FOLD EXTERNO:")
        print("-"*45)
        outer_results = data['outer_fold_results']
        for fold_id, fold_data in outer_results.items():
            print(f"{fold_id}: F1={fold_data['f1']:.4f}, CV_score={fold_data['cv_score']:.4f}, N={fold_data['samples']}")
        
        print(f"ESTADÍSTICAS AGREGADAS:")
        print("-"*30)
        stats = data['aggregated_stats']
        print(f"F1 Score: {stats['f1_mean']:.4f} ± {stats['f1_std']:.4f}")
        print(f"Rango: [{stats['f1_min']:.4f}, {stats['f1_max']:.4f}]")
        print(f"Coef. Variación: {stats['f1_cv']:.4f}")
        print(f"Estabilidad: {stats['stability']:.4f} (1.0 = perfectamente estable)")
        
        if 'intelligent_grid_used' in data:
            print(f"GRID INTELIGENTE GENERADO:")
            print(f"   Basado en {data['based_on_random_results']} mejores resultados Random")
            for param, values in data['intelligent_grid_used'].items():
                print(f"   {param}: {values}")
        
        return data
    
    def get_hybrid_comparison():
        """Comparación completa Random vs Grid Search por algoritmo y dataset."""
        print("COMPARACIÓN BÚSQUEDA HÍBRIDA DETALLADA")
        print("="*60)
        
        if not hybrid_search_results:
            print("❌ No hay resultados híbridos disponibles")
            return None
        
        for algo_name, datasets in hybrid_search_results.items():
            print(f" {algo_name.upper()}:")
            print("-"*40)
            
            for dataset_name, results in datasets.items():
                print(f"    {dataset_name}:")
                print(f"       Random Search:")
                print(f"         F1: {results['random_f1_mean']:.4f} ± {results['random_f1_std']:.4f}")
                print(f"         Estabilidad: {results['random_stability']:.4f}")
                print(f"       Grid Search Inteligente:")
                print(f"         F1: {results['grid_f1_mean']:.4f} ± {results['grid_f1_std']:.4f}")
                print(f"         Estabilidad: {results['grid_stability']:.4f}")
                print(f"       Mejoras:")
                print(f"         F1: {results['improvement_pct']:+.2f}%")
                print(f"         Estabilidad: {results['stability_improvement']:+.4f}")
                print()
        
        return hybrid_search_results
    
    def get_stability_ranking():
        """ Ranking de experimentos por estabilidad y rendimiento."""
        stability_data = []
        
        for exp_id, data in fold_level_tracking.items():
            stats = data['aggregated_stats']
            stability_data.append({
                'exp_id': exp_id,
                'algorithm': data['algorithm'],
                'dataset': data['dataset'],
                'search_type': data['search_type'],
                'f1_mean': stats['f1_mean'],
                'f1_std': stats['f1_std'],
                'stability': stats['stability'],
                'cv_coefficient': stats['f1_cv']
            })
        
        # Ordenar por estabilidad (descendente) y luego por F1 (descendente)
        stability_data.sort(key=lambda x: (x['stability'], x['f1_mean']), reverse=True)
        
        print("\n RANKING POR ESTABILIDAD Y RENDIMIENTO (TOP 15)")
        print("="*75)
        print("Exp_ID | Algoritmo      | Dataset | Búsqueda    | F1_Mean | F1_Std | Estabilidad")
        print("-"*75)
        
        for i, item in enumerate(stability_data[:15], 1):
            print(f"{item['exp_id']:6d} | {item['algorithm'][:14]:14s} | {item['dataset'][:7]:7s} | "
                  f"{item['search_type'][:11]:11s} | {item['f1_mean']:.3f}   | {item['f1_std']:.3f}  | {item['stability']:.3f}")
        
        return stability_data
    
    def get_algorithm_comparison():
        """ Comparación de rendimiento promedio por algoritmo."""
        algo_performance = defaultdict(list)
        
        for data in fold_level_tracking.values():
            algo = data['algorithm']
            stats = data['aggregated_stats']
            algo_performance[algo].append({
                'f1_mean': stats['f1_mean'],
                'stability': stats['stability'],
                'search_type': data['search_type']
            })
        
        print("\n RENDIMIENTO PROMEDIO POR ALGORITMO")
        print("="*50)
        
        for algo, results in algo_performance.items():
            f1_scores = [r['f1_mean'] for r in results]
            stabilities = [r['stability'] for r in results]
            
            print(f"\n{algo.upper()}:")
            print(f"    F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
            print(f"    Estabilidad: {np.mean(stabilities):.4f} ± {np.std(stabilities):.4f}")
            print(f"    Experimentos: {len(results)}")
        
        return algo_performance
    
    # Agregar funciones al tracker
    tracker.get_nested_cv_analysis = get_nested_cv_analysis
    tracker.get_hybrid_comparison = get_hybrid_comparison
    tracker.get_stability_ranking = get_stability_ranking
    tracker.get_algorithm_comparison = get_algorithm_comparison
    
    # Agregar datos detallados
    tracker.fold_level_tracking = fold_level_tracking
    tracker.hybrid_search_results = hybrid_search_results
    tracker.nested_cv_results = nested_cv_results
    tracker.algorithm_random_results = algorithm_random_results
    
    return tracker

# 🚀 EJECUTAR EXPERIMENTACIÓN HÍBRIDA CON VALIDACIÓN CRUZADA ANIDADA
print(" Iniciando experimentación híbrida con trazabilidad completa por fold...")
print(" Metodología: Random Search → Grid Search Inteligente")
print(" Validación: Cruzada Anidada (10×10 = 100 evaluaciones por experimento)")
print(" Balanceo: Usando la mejor técnica identificada (BorderlineSMOTE)")
print()

# Ejecutar con número reducido para demostración (ajustar según necesidad)
tracker_hybrid = run_hybrid_nested_experiments(
    max_experiments=50,  # Ajustar según recursos disponibles
    selected_balancing_technique=best_balancing  # Usar la mejor técnica encontrada
)

print("\n Experimentación híbrida completada!")
print(" Funciones disponibles para análisis:")
print("   • tracker_hybrid.get_nested_cv_analysis(experiment_id)")
print("   • tracker_hybrid.get_hybrid_comparison()")
print("   • tracker_hybrid.get_stability_ranking()") 
print("   • tracker_hybrid.get_algorithm_comparison()")