import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    graficos = ['figura1_analisis_descriptivo.png', 'figura2_distribucion_proyecciones.png', 
                'figura3_comparativa_internacional.png', 'figura4_mapa_radiacion.png']
    return render_template('index.html', graficos=graficos)

# Configuración de márgenes y espaciado
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['axes.titlepad'] = 20
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# Configuración de estilo
plt.style.use('default')
sns.set_palette(["#523C39", "#588A50", "#F69845", "#E6D396"])

# Datos de generación solar de Colombia (1980-2024)
colombia_data = {
    'years': [1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2021, 2022, 2023, 2024],
    'generation': [0.5, 1.2, 2.5, 5.8, 12.4, 28.5, 85.5, 253.4, 1452.8, 1785.2, 2245.3, 2458.7, 2845.3]
}

# Crear DataFrame
df = pd.DataFrame(colombia_data)
df['growth_rate'] = df['generation'].pct_change() * 100

# 1. ESTADÍSTICAS DESCRIPTIVAS BÁSICAS
def calcular_estadisticas_basicas(datos):
    """Calcula estadísticas descriptivas básicas"""
    # Cálculo de moda manual
    valores_unicos, conteos = np.unique(datos, return_counts=True)
    max_conteo = np.max(conteos)
    
    if max_conteo > 1:
        modas = valores_unicos[conteos == max_conteo]
        moda_valor = modas[0] if len(modas) == 1 else f'Múltiples modas: {modas}'
    else:
        moda_valor = 'No hay moda'
    
    return {
        'media': np.mean(datos),
        'mediana': np.median(datos),
        'moda': moda_valor,
        'minimo': np.min(datos),
        'maximo': np.max(datos),
        'desviacion_estandar': np.std(datos),
        'varianza': np.var(datos),
        'rango': np.ptp(datos),
        'coef_variacion': (np.std(datos) / np.mean(datos)) * 100 if np.mean(datos) != 0 else float('inf')
    }

# 2. ANÁLISIS DE TENDENCIA Y CRECIMIENTO
def analizar_tendencia(datos, años):
    """Analiza la tendencia y crecimiento de los datos"""
    # Regresión lineal para tendencia
    slope, intercept, r_value, p_value, std_err = stats.linregress(años, datos)
    
    # Cálculo de tasas de crecimiento
    crecimiento_absoluto = datos.iloc[-1] - datos.iloc[0]
    crecimiento_relativo = (datos.iloc[-1] / datos.iloc[0] - 1) * 100 if datos.iloc[0] != 0 else float('inf')
    
    # Períodos de duplicación (regla del 70)
    if datos.iloc[0] != 0 and len(años) > 1:
        tasa_crecimiento_anual = (np.power(datos.iloc[-1]/datos.iloc[0], 1/(len(años)-1)) - 1) * 100
        periodo_duplicacion = 70 / tasa_crecimiento_anual if tasa_crecimiento_anual > 0 else float('inf')
    else:
        tasa_crecimiento_anual = float('inf')
        periodo_duplicacion = float('inf')
    
    return {
        'pendiente_tendencia': slope,
        'intercepto': intercept,
        'r_cuadrado': r_value**2,
        'p_value': p_value,
        'crecimiento_absoluto': crecimiento_absoluto,
        'crecimiento_relativo': crecimiento_relativo,
        'tasa_crecimiento_anual': tasa_crecimiento_anual,
        'periodo_duplicacion': periodo_duplicacion
    }

# 3. ANÁLISIS DE DISTRIBUCIÓN
def analizar_distribucion(datos):
    """Analiza la distribución de los datos"""
    # Test de normalidad
    stat_sw, p_sw = stats.shapiro(datos)
    stat_ks, p_ks = stats.kstest(datos, 'norm', args=(np.mean(datos), np.std(datos, ddof=1)))
    
    # Asimetría y curtosis
    asimetria = stats.skew(datos)
    curtosis = stats.kurtosis(datos)
    
    # Cuartiles and percentiles
    cuartiles = np.percentile(datos, [25, 50, 75])
    percentiles = np.percentile(datos, [10, 25, 50, 75, 90, 95])
    
    return {
        'shapiro_wilk_stat': stat_sw,
        'shapiro_wilk_p': p_sw,
        'kolmogorov_smirnov_stat': stat_ks,
        'kolmogorov_smirnov_p': p_ks,
        'asimetria': asimetria,
        'curtosis': curtosis,
        'cuartiles': dict(zip(['Q1', 'Q2', 'Q3'], cuartiles)),
        'percentiles': dict(zip(['P10', 'P25', 'P50', 'P75', 'P90', 'P95'], percentiles))
    }

# 4. ANÁLISIS DE VOLATILIDAD Y ESTABILIDAD
def analizar_volatilidad(datos):
    """Analiza la volatilidad y estabilidad de la serie"""
    if len(datos) > 1:
        retornos = np.diff(datos) / datos.iloc[:-1] * 100
        volatilidad = np.std(retornos) if len(retornos) > 0 else 0
        
        maximas_subidas = np.max(retornos) if len(retornos) > 0 else 0
        maximas_caidas = np.min(retornos) if len(retornos) > 0 else 0
        
        sharpe_ratio = np.mean(retornos) / volatilidad if volatilidad != 0 else 0
        
        retornos_negativos = retornos[retornos < 0]
        sortino_ratio = np.mean(retornos) / np.std(retornos_negativos) if len(retornos_negativos) > 0 and np.std(retornos_negativos) != 0 else 0
    else:
        volatilidad = 0
        maximas_subidas = 0
        maximas_caidas = 0
        sharpe_ratio = 0
        sortino_ratio = 0
    
    return {
        'volatilidad_anualizada': volatilidad,
        'maxima_subida': maximas_subidas,
        'maxima_caida': maximas_caidas,
        'ratio_sharpe': sharpe_ratio,
        'ratio_sortino': sortino_ratio
    }

# 5. PROYECCIONES Y PREDICCIONES
def hacer_proyecciones(datos, años, años_futuros=5):
    """Realiza proyecciones basadas en tendencia histórica"""
    datos_positivos = [max(x, 0.001) for x in datos]
    log_datos = np.log(datos_positivos)
    slope_log, intercept_log, _, _, _ = stats.linregress(años, log_datos)
    
    años_futuros_list = list(range(años.iloc[-1] + 1, años.iloc[-1] + años_futuros + 1))
    proyecciones_exp = np.exp(intercept_log + slope_log * np.array(años_futuros_list))
    
    slope_lin, intercept_lin, _, _, _ = stats.linregress(años, datos)
    proyecciones_lin = intercept_lin + slope_lin * np.array(años_futuros_list)
    
    return {
        'años_futuros': años_futuros_list,
        'proyeccion_exponencial': proyecciones_exp.tolist(),
        'proyeccion_lineal': proyecciones_lin.tolist(),
        'tasa_crecimiento_exponencial': slope_log * 100
    }

# EJECUTAR ANÁLISIS
print("=" * 60)
print("ANÁLISIS ESTADÍSTICO - ENERGÍA SOLAR COLOMBIA")
print("=" * 60)

# 1. Estadísticas básicas
estadisticas_basicas = calcular_estadisticas_basicas(df['generation'])
print("\n1. ESTADÍSTICAS DESCRIPTIVAS BÁSICAS:")
for key, value in estadisticas_basicas.items():
    if isinstance(value, (int, float)):
        print(f"{key.replace('_', ' ').title():25}: {value:10.2f}")
    else:
        print(f"{key.replace('_', ' ').title():25}: {value:>10}")

# 2. Análisis de tendencia
analisis_tendencia = analizar_tendencia(df['generation'], df['years'])
print(f"\n2. ANÁLISIS DE TENDENCIA:")
print(f"Pendiente de tendencia:      {analisis_tendencia['pendiente_tendencia']:10.2f}")
print(f"R²:                          {analisis_tendencia['r_cuadrado']:10.4f}")
print(f"Crecimiento absoluto:        {analisis_tendencia['crecimiento_absoluto']:10.2f} GWh")
print(f"Crecimiento relativo:        {analisis_tendencia['crecimiento_relativo']:10.2f}%")
print(f"Tasa crecimiento anual:      {analisis_tendencia['tasa_crecimiento_anual']:10.2f}%")
print(f"Período de duplicación:      {analisis_tendencia['periodo_duplicacion']:10.2f} años")

# 3. Análisis de distribución
analisis_distribucion = analizar_distribucion(df['generation'])
print(f"\n3. ANÁLISIS DE DISTRIBUCIÓN:")
print(f"Asimetría:                   {analisis_distribucion['asimetria']:10.4f}")
print(f"Curtosis:                    {analisis_distribucion['curtosis']:10.4f}")
print(f"Shapiro-Wilk p-value:        {analisis_distribucion['shapiro_wilk_p']:10.4f}")

print("\nPercentiles:")
for key, value in analisis_distribucion['percentiles'].items():
    print(f"{key:25}: {value:10.2f} GWh")

# 4. Análisis de volatilidad
analisis_volatilidad = analizar_volatilidad(df['generation'])
print(f"\n4. ANÁLISIS DE VOLATILIDAD:")
print(f"Volatilidad anualizada:      {analisis_volatilidad['volatilidad_anualizada']:10.2f}%")
print(f"Máxima subida:               {analisis_volatilidad['maxima_subida']:10.2f}%")
print(f"Máxima caída:                {analisis_volatilidad['maxima_caida']:10.2f}%")
print(f"Ratio de Sharpe:             {analisis_volatilidad['ratio_sharpe']:10.4f}")

# 5. Proyecciones
proyecciones = hacer_proyecciones(df['generation'], df['years'])
print(f"\n5. PROYECCIONES FUTURAS:")
for i, año in enumerate(proyecciones['años_futuros']):
    print(f"Año {año}:")
    print(f"  Proyección exponencial: {proyecciones['proyeccion_exponencial'][i]:10.2f} GWh")
    print(f"  Proyección lineal:      {proyecciones['proyeccion_lineal'][i]:10.2f} GWh")

# PRIMERA FIGURA: Análisis Descriptivo
fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))
fig1.suptitle('ANÁLISIS DESCRIPTIVO - GENERACIÓN SOLAR COLOMBIA\n(1980-2024)', 
              fontsize=18, fontweight='bold', y=0.98)

# 1. Serie temporal con tendencia (Primera figura)
axes1[0].plot(df['years'], df['generation'], 'o-', color='#588A50', linewidth=2, markersize=6, label='Datos reales')
z = np.polyfit(df['years'], df['generation'], 1)
p = np.poly1d(z)
axes1[0].plot(df['years'], p(df['years']), "--", color='#F69845', linewidth=2, label='Tendencia lineal')
axes1[0].set_title('EVOLUCIÓN TEMPORAL\nCon Tendencia Lineal', fontweight='bold', pad=20)
axes1[0].set_xlabel('Año')
axes1[0].set_ylabel('Generación (GWh)')
axes1[0].legend()
axes1[0].grid(True, alpha=0.3)
axes1[0].ticklabel_format(style='plain', axis='y')

# 2. Histograma con distribución (Primera figura)
axes1[1].hist(df['generation'], bins=6, color='#E6D396', edgecolor='#523C39', alpha=0.7)
axes1[1].axvline(estadisticas_basicas['media'], color='#F69845', linestyle='--', linewidth=2, label='Media')
axes1[1].axvline(estadisticas_basicas['mediana'], color='#588A50', linestyle='--', linewidth=2, label='Mediana')
axes1[1].set_title('DISTRIBUCIÓN DE FRECUENCIAS\nMedia y Mediana', fontweight='bold', pad=20)
axes1[1].set_xlabel('Generación (GWh)')
axes1[1].set_ylabel('Frecuencia')
axes1[1].legend()
axes1[1].grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('figura1_analisis_descriptivo.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
plt.show()
plt.close(fig1)

# SEGUNDA FIGURA: Análisis de Distribución y Proyecciones
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
fig2.suptitle('ANÁLISIS DE DISTRIBUCIÓN Y PROYECCIONES\nGENERACIÓN SOLAR COLOMBIA', 
              fontsize=18, fontweight='bold', y=0.98)

# 3. Boxplot (Segunda figura)
boxplot = axes2[0].boxplot(df['generation'], patch_artist=True,
                  boxprops=dict(facecolor='#E6D396', color='#523C39'),
                  medianprops=dict(color='#F69845', linewidth=2),
                  whiskerprops=dict(color='#523C39'),
                  capprops=dict(color='#523C39'),
                  flierprops=dict(marker='o', color='#588A50', alpha=0.7))
axes2[0].set_title('DIAGRAMA DE CAJA\nDistribución Estadística', fontweight='bold', pad=20)
axes2[0].set_ylabel('Generación (GWh)')
axes2[0].grid(True, alpha=0.3)
axes2[0].set_xticklabels(['Generación Solar'])

# 4. Proyecciones futuras (Segunda figura)
axes2[1].plot(df['years'], df['generation'], 'o-', color='#588A50', linewidth=2, markersize=6, label='Datos históricos')
axes2[1].plot(proyecciones['años_futuros'], proyecciones['proyeccion_exponencial'], 's--', 
               color='#F69845', linewidth=2, markersize=6, label='Proyección exponencial')
axes2[1].plot(proyecciones['años_futuros'], proyecciones['proyeccion_lineal'], '^--', 
               color='#523C39', linewidth=2, markersize=6, label='Proyección lineal')
axes2[1].set_title('PROYECCIONES FUTURAS\nEscenarios 2025-2029', fontweight='bold', pad=20)
axes2[1].set_xlabel('Año')
axes2[1].set_ylabel('Generación (GWh)')
axes2[1].legend()
axes2[1].grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('figura2_distribucion_proyecciones.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
plt.show()
plt.close(fig2)

# TERCERA FIGURA: Comparativa Internacional (FIGURA SEPARADA)
fig3 = plt.figure(figsize=(14, 8))
paises = ['China', 'EEUU', 'Alemania', 'Japón', 'India', 'Italia', 'Reino Unido', 'Australia', 'Corea del Sur', 'España', 'Colombia']
generacion = [392.5, 164.2, 59.3, 78.1, 95.4, 25.6, 13.8, 29.5, 18.9, 23.7, 2.85]

# Crear colores diferenciados
colores = ['#F69845' if pais == 'Colombia' else '#588A50' for pais in paises]

bars = plt.barh(paises, generacion, color=colores)
plt.xlabel('Generación Solar (TWh)', fontsize=12)
plt.title('COMPARATIVA INTERNACIONAL - GENERACIÓN SOLAR 2024', fontsize=16, fontweight='bold', pad=20)
plt.bar_label(bars, fmt='%.1f TWh', padding=5, fontsize=10)
plt.grid(True, alpha=0.3, axis='x')

# Destacar Colombia
plt.annotate('Colombia: Creciendo rápidamente', 
             xy=(2.85, paises.index('Colombia')), 
             xytext=(50, paises.index('Colombia')),
             arrowprops=dict(arrowstyle='->', color='#523C39'),
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#E6D396", alpha=0.8),
             fontsize=10)

plt.tight_layout()
plt.savefig('figura3_comparativa_internacional.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
plt.show()
plt.close(fig3)

# CUARTA FIGURA: Mapa de Radiación Solar (FIGURA SEPARADA)
fig4 = plt.figure(figsize=(12, 8))
regiones = ['Caribe', 'Andina', 'Pacífico', 'Orinoquía', 'Amazonía']
radiacion = [5.8, 5.2, 4.9, 5.5, 4.8]

# Crear gráfico de barras para radiación
bars = plt.bar(regiones, radiacion, 
               color=['#F69845' if x == max(radiacion) else '#E6D396' for x in radiacion],
               edgecolor='#523C39', linewidth=1.5)

plt.ylabel('Radiación Solar (kWh/m²/día)', fontsize=12)
plt.title('MAPA DE RADIACIÓN SOLAR POR REGIÓN - COLOMBIA', fontsize=16, fontweight='bold', pad=20)
plt.ylim(0, 7)
plt.grid(True, alpha=0.3, axis='y')

# Añadir valores en las barras
for bar, valor in zip(bars, radiacion):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{valor:.1f} kWh/m²/día', ha='center', va='bottom', fontweight='bold')

# Añadir información adicional
plt.text(0.02, 0.98, 'Región Caribe: Mayor radiación\nIdeal para proyectos solares', 
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#E6D396", alpha=0.8))

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figura4_mapa_radiacion.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
plt.show()
plt.close(fig4)

# ANÁLISIS ADICIONAL
print(f"\n6. ANÁLISIS COMPARATIVO:")
print(f"Generación 2024 Colombia:    {df[df['years'] == 2024]['generation'].values[0]:10.2f} GWh")

if len(df) >= 5:
    crecimiento_5años = (df['generation'].iloc[-1] - df['generation'].iloc[-5]) / df['generation'].iloc[-5] * 100
    print(f"Crecimiento últimos 5 años:  {crecimiento_5años:10.2f}%")

if df['generation'].iloc[0] > 0:
    elasticidad = (np.log(df['generation'].iloc[-1]) - np.log(df['generation'].iloc[0])) / (df['years'].iloc[-1] - df['years'].iloc[0])
    print(f"Elasticidad temporal:        {elasticidad:10.4f}")

print("\n" + "=" * 60)
print("ANÁLISIS COMPLETADO EXITOSAMENTE")
print("=" * 60)

if __name__ == '__main__':
    app.run(debug=True)