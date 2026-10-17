# -*- coding: utf-8 -*-
"""
Laboratorio 9 - Visualización Estática (CC3084)
Genera la infografía (PNG y PDF 11x17) y cada gráfica por separado
a partir de las series y resultados del Laboratorio 1.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings('ignore')

CarpetaBase = Path(__file__).resolve().parent
CarpetaDatos = CarpetaBase / 'Data'
CarpetaSalida = CarpetaBase / 'Salida'
CarpetaGraficas = CarpetaSalida / 'Graficas'

AzulMarino = '#1F3A5F'
AzulGris = '#6B7C93'
AzulClaro = '#E8EEF5'
Naranja = '#E4572E'
NaranjaClaro = '#F2A68F'
AzulRmse = '#A7BAD5'
AzulCelda = '#6F8DB9'
GrisPrueba = '#E1E5EA'
Fondo = '#F7F7F5'
TextoOscuro = '#2D3748'
Rejilla = '#DADDE2'

FechaCorte = '2021-04-01'
Semilla = 42
Simulaciones = 1000

NombresSeries = {
    'Total_Consistent': 'Total consistente',
    'Via_Aerea': 'Vía aérea',
    'Via_Terrestre': 'Vía terrestre',
    'Via_Maritima': 'Vía marítima',
    'Pais_El Salvador': 'El Salvador',
    'Pais_Guatemala': 'Guatemala',
    'Pais_Estados Unidos de América': 'Estados Unidos',
}
NombresModelos = {
    'Holt-Winters': 'Holt-Winters',
    'Prophet': 'Prophet',
    'Suav. Exponencial (SES)': 'Suav. exponencial',
    'Seasonal Naive': 'Seasonal Naive',
    'SARIMA': 'SARIMA',
}
Meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8.5,
    'axes.edgecolor': AzulGris,
    'axes.labelcolor': TextoOscuro,
    'xtick.color': TextoOscuro,
    'ytick.color': TextoOscuro,
    'axes.unicode_minus': True,
    'savefig.facecolor': Fondo,
})


def CargarDatos():
    Series = pd.read_csv(CarpetaDatos / 'series_de_tiempo_completas.csv',
                         index_col='Fecha', parse_dates=True).asfreq('MS')
    Metricas = pd.read_csv(CarpetaDatos / 'model_comparison_metrics.csv')
    Comparativas = pd.read_csv(CarpetaDatos / 'comparative_analysis_stats.csv', index_col='Serie')
    return Series, Metricas, Comparativas


def CalcularIndicadores(Series, Comparativas):
    Total = Series['Total_Consistent']
    Anual = Total.groupby(Total.index.year).sum()
    Base = Total['2015':'2019']
    Estacional = Base.groupby(Base.index.month).mean() / Base.mean() - 1
    return {
        'Total2019': Anual[2019],
        'Total2025': Anual[2025],
        'Recuperacion': Anual[2025] / Anual[2019],
        'Caida2020': 1 - Total['2020'].min() / Total['2019'].mean(),
        'Minimo2020': Total['2020'].min(),
        'Record': Total.max(),
        'FechaRecord': Total.idxmax(),
        'Estacional': Estacional,
        'AcumuladoSalvador': Series['Pais_El Salvador'].sum(),
        'Comparativas': Comparativas,
    }


def AjustarHoltWinters(Series):
    Total = Series['Total_Consistent'].interpolate(method='time').fillna(0).clip(lower=0)
    Entrenamiento = Total[:FechaCorte]
    Prueba = Total[FechaCorte:].iloc[1:]
    Modelo = ExponentialSmoothing(Entrenamiento, trend='add', seasonal='add',
                                  seasonal_periods=12,
                                  initialization_method='heuristic').fit()
    Prediccion = pd.Series(Modelo.forecast(len(Prueba)).values, index=Prueba.index)
    Trayectorias = Modelo.simulate(len(Prueba), repetitions=Simulaciones, error='add',
                                   anchor='end', random_state=Semilla)
    Inferior = np.clip(Trayectorias.quantile(0.10, axis=1).values, 0, None)
    Superior = Trayectorias.quantile(0.90, axis=1).values
    return Entrenamiento, Prueba, Prediccion, Inferior, Superior


def FormatoMil(Valor, _):
    return '0' if Valor == 0 else f'{Valor / 1000:.0f} mil'


def FormatoPorcentaje(Valor, _):
    if abs(Valor) < 1e-9:
        return '0'
    return f'{Valor:+.0%}'.replace('-', '−')


def LimpiarEjes(Ax, RejillaY=True):
    for Lado in ['top', 'right']:
        Ax.spines[Lado].set_visible(False)
    Ax.set_facecolor('none')
    Ax.tick_params(length=3, labelsize=8.5)
    if RejillaY:
        Ax.grid(axis='y', color=Rejilla, linewidth=0.6)
        Ax.set_axisbelow(True)


def DibujarHistorico(Ax, Series, Ind):
    Total = Series['Total_Consistent']
    Covid = Total['2020-03':'2020-12']
    Ax.axvspan(pd.Timestamp('2021-05-01'), pd.Timestamp('2026-06-01'), color=GrisPrueba, zorder=0)
    Ax.add_patch(Rectangle((mdates.date2num(pd.Timestamp('2022-01-01')), 0.965),
                           mdates.date2num(pd.Timestamp('2024-01-01')) - mdates.date2num(pd.Timestamp('2022-01-01')),
                           0.035, transform=Ax.get_xaxis_transform(), color=AzulGris, zorder=1))
    Ax.plot(Total.index, Total.values, color=AzulMarino, linewidth=1.6, zorder=3)
    Ax.plot(Covid.index, Covid.values, color=Naranja, linewidth=2.2, zorder=4)

    Ax.annotate(f'Récord: {Ind["Record"] / 1000:.0f} mil\n(dic {Ind["FechaRecord"].year})',
                xy=(Ind['FechaRecord'], Ind['Record']), xytext=(pd.Timestamp('2016-08-01'), 405000),
                color=AzulMarino, fontsize=8.5, fontweight='bold', va='bottom',
                arrowprops=dict(arrowstyle='-', color=AzulGris, linewidth=0.8))
    Ax.text(pd.Timestamp('2014-09-01'), 5000,
            'Mar–may 2020: cierre de fronteras y espacio\n'
            f'aéreo (COVID-19). Mínimo: {Ind["Minimo2020"]:,.0f} viajeros.\n'
            'Sep 2020: reabre el aeropuerto La Aurora.',
            color=Naranja, fontsize=8.5, fontweight='bold', ha='left', va='bottom', linespacing=1.15)
    Ax.text(pd.Timestamp('2022-01-01'), 0.93,
            '2022–2023: la base cambia la definición de «Viajero»\n(por eso se usa Turista + Excursionista)',
            transform=Ax.get_xaxis_transform(), color=TextoOscuro, fontsize=8.5, va='top', clip_on=False)
    Ax.text(pd.Timestamp('2021-06-01'), 12000, 'Conjunto de prueba (may 2021 – jun 2026)',
            color=AzulGris, fontsize=8.5, va='bottom')

    Ax.set_xlim(pd.Timestamp('2009-01-01'), pd.Timestamp('2026-06-01'))
    Ax.set_ylim(0, 540000)
    Ax.set_yticks([0, 100000, 200000, 300000, 400000])
    Ax.yaxis.set_major_formatter(FuncFormatter(FormatoMil))
    Ax.xaxis.set_major_locator(mdates.YearLocator(2))
    Ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    Ax.set_ylabel('Viajeros por mes', fontsize=8.5)
    LimpiarEjes(Ax)


def DibujarEstacionalidad(Ax, Ind):
    Valores = Ind['Estacional'].values
    Posiciones = np.arange(12)
    Colores = [AzulMarino if V >= 0 else AzulGris for V in Valores]
    Colores[11] = Naranja
    Ax.bar(Posiciones, Valores, width=0.62, color=Colores, zorder=3)
    Ax.axhline(0, color=TextoOscuro, linewidth=0.9, zorder=4)
    for Pos, Val in zip(Posiciones, Valores):
        Etiqueta = f'{Val:+.0%}' if round(Val * 100) != 0 else '0%'
        if Val >= 0 and round(Val * 100) != 0:
            Ax.text(Pos, Val + 0.012, Etiqueta, ha='center', va='bottom', fontsize=7.5, color=TextoOscuro)
        else:
            Ax.text(Pos, min(Val, 0) - 0.012, Etiqueta, ha='center', va='top', fontsize=7.5, color=TextoOscuro)

    Ax.text(2.5, 0.29, 'Semana Santa', ha='center', fontsize=8.5, fontweight='bold', color=AzulMarino)
    Ax.text(6.0, 0.215, 'Bono 14 y\nvacaciones', ha='center', fontsize=8.5, fontweight='bold',
            color=AzulMarino, linespacing=1.05)
    Ax.text(8.5, 0.07, 'Valle de\nsep–oct', ha='center', fontsize=8.5, fontweight='bold',
            color=AzulGris, linespacing=1.05)
    Ax.text(11.0, 0.395, 'Dic: aguinaldo\ny fiestas', ha='right', fontsize=8.5, fontweight='bold',
            color=Naranja, linespacing=1.05, multialignment='right')

    LimpiarEjes(Ax)
    Ax.set_xlim(-0.95, 11.95)
    Ax.set_ylim(-0.31, 0.46)
    Ax.set_xticks(Posiciones)
    Ax.set_xticklabels(Meses)
    Ax.tick_params(axis='x', length=0)
    Ax.set_yticks([-0.2, 0, 0.2, 0.4])
    Ax.yaxis.set_major_formatter(FuncFormatter(FormatoPorcentaje))
    Ax.set_ylabel('Frente a un mes típico', fontsize=8.5)
    Ax.spines['bottom'].set_visible(False)


def DibujarTabla(Ax, Comparativas):
    Columnas = [('Fuerza_Estacional', 'Estacional'), ('Fuerza_Tendencia', 'Tendencia'),
                ('Volatilidad_CV', 'Volatilidad'), ('Caida_Pandemia_pct', 'Caída 2020')]
    Escala = LinearSegmentedColormap.from_list('Escala', [AzulClaro, AzulCelda])
    Ancho, Alto, AnchoNombre, AltoEncabezado = 572, 236, 152, 29
    AnchoColumna = (Ancho - AnchoNombre) / len(Columnas)
    AltoFila = (Alto - AltoEncabezado) / len(NombresSeries)
    Borde = dict(edgecolor='#4A4A4A', linewidth=0.7)

    Ax.set_xlim(0, Ancho)
    Ax.set_ylim(Alto, 0)
    Ax.axis('off')
    Ax.add_patch(Rectangle((0, 0), AnchoNombre, AltoEncabezado, facecolor='white', **Borde))
    for J, (_, Titulo) in enumerate(Columnas):
        X = AnchoNombre + J * AnchoColumna
        Ax.add_patch(Rectangle((X, 0), AnchoColumna, AltoEncabezado, facecolor='white', **Borde))
        Ax.text(X + AnchoColumna / 2, AltoEncabezado / 2, Titulo, ha='center', va='center',
                fontsize=8.5, fontweight='bold', color=TextoOscuro)

    for I, (Clave, Nombre) in enumerate(NombresSeries.items()):
        Y = AltoEncabezado + I * AltoFila
        Ax.add_patch(Rectangle((0, Y), AnchoNombre, AltoFila, facecolor='white', **Borde))
        Ax.text(7, Y + AltoFila / 2, Nombre, va='center', fontsize=9, color=TextoOscuro)
        for J, (Campo, _) in enumerate(Columnas):
            Columna = Comparativas[Campo]
            Valor = Columna[Clave]
            EsMaximo = Valor == Columna.max()
            Nivel = (Valor - Columna.min()) / (Columna.max() - Columna.min())
            Color = Naranja if EsMaximo else Escala(Nivel)
            Texto = f'−{Valor:.1f}%' if Campo == 'Caida_Pandemia_pct' else f'{Valor:.2f}'
            X = AnchoNombre + J * AnchoColumna
            Ax.add_patch(Rectangle((X, Y), AnchoColumna, AltoFila, facecolor=Color, **Borde))
            Ax.text(X + AnchoColumna / 2, Y + AltoFila / 2, Texto, ha='center', va='center', fontsize=8.5,
                    color='white' if EsMaximo else TextoOscuro, fontweight='bold' if EsMaximo else 'normal')


def DibujarModelos(Ax, Metricas):
    Datos = (Metricas[Metricas['Serie'] == 'Total_Consistent']
             .sort_values('MAE').reset_index(drop=True))
    Alto = 0.36
    for I, Fila in Datos.iterrows():
        EsGanador = I == 0
        ColorMae = Naranja if EsGanador else AzulMarino
        ColorRmse = NaranjaClaro if EsGanador else AzulRmse
        Ax.barh(I - Alto / 2, Fila['MAE'], height=Alto, color=ColorMae)
        Ax.barh(I + Alto / 2, Fila['RMSE'], height=Alto, color=ColorRmse)
        Peso = 'bold' if EsGanador else 'normal'
        Ax.text(Fila['MAE'] + 4000, I - Alto / 2, f'{Fila["MAE"] / 1000:.0f} mil',
                va='center', fontsize=8.5, color=TextoOscuro, fontweight=Peso)
        Ax.text(Fila['RMSE'] + 4000, I + Alto / 2, f'{Fila["RMSE"] / 1000:.0f} mil',
                va='center', fontsize=8.5, color=TextoOscuro, fontweight=Peso)

    Ax.set_yticks(range(len(Datos)))
    Ax.set_yticklabels([NombresModelos[M] for M in Datos['Modelo']], fontsize=9)
    Ax.set_ylim(len(Datos) - 0.45, -0.6)
    Ax.set_xlim(0, 330000)
    Ax.set_xticks([])
    Ax.tick_params(axis='y', length=0)
    for Lado in ['top', 'right', 'bottom', 'left']:
        Ax.spines[Lado].set_visible(False)
    Ax.set_facecolor('none')
    Ax.text(0, -1.0, '■', color=AzulMarino, fontsize=8, va='center')
    Ax.text(9000, -1.0, 'MAE: error promedio por mes', color=TextoOscuro, fontsize=8.5, va='center')
    Ax.text(170000, -1.0, '■', color=AzulRmse, fontsize=8, va='center')
    Ax.text(179000, -1.0, 'RMSE: castiga errores grandes', color=TextoOscuro, fontsize=8.5, va='center')


def DibujarPronostico(Ax, Entrenamiento, Prueba, Prediccion, Inferior, Superior):
    Previo = Entrenamiento['2018-01-01':]
    Ax.fill_between(Prueba.index, Inferior, Superior, color=Naranja, alpha=0.2, linewidth=0, zorder=1)
    Ax.plot(Previo.index, Previo.values, color=AzulGris, linewidth=1.3, zorder=2)
    Ax.plot(Prueba.index, Prueba.values, color=AzulMarino, linewidth=1.9, zorder=4)
    Ax.plot(Prediccion.index, Prediccion.values, color=Naranja, linewidth=2, linestyle='--', zorder=3)
    Ax.axvline(pd.Timestamp('2021-05-01'), color=AzulGris, linewidth=0.8, linestyle=':', zorder=2)

    Pico = Prueba['2022':'2023'].idxmax()
    Ax.annotate('Dato real', xy=(Pico, Prueba[Pico]), xytext=(pd.Timestamp('2022-06-01'), 400000),
                color=AzulMarino, fontsize=8.5, fontweight='bold', ha='right', va='center',
                arrowprops=dict(arrowstyle='-', color=AzulMarino, linewidth=0.8))
    Ax.text(pd.Timestamp('2026-01-01'), 445000, 'Estimación 80 %\n(rango probable)', ha='right',
            va='center', fontsize=8.5, fontweight='bold', color=Naranja, linespacing=1.05)
    Ax.text(pd.Timestamp('2025-04-01'), 15000, 'Predicción\nHolt-Winters', ha='center', va='bottom',
            fontsize=8.5, fontweight='bold', color=Naranja, linespacing=1.05)
    Ax.text(pd.Timestamp('2018-02-01'), 140000, 'Datos de\nentrenamiento', fontsize=8.5,
            color=AzulGris, linespacing=1.05)

    Ax.set_xlim(pd.Timestamp('2018-01-01'), pd.Timestamp('2026-06-01'))
    Ax.set_ylim(0, 475000)
    Ax.set_yticks([0, 100000, 200000, 300000, 400000])
    Ax.yaxis.set_major_formatter(FuncFormatter(FormatoMil))
    Ax.xaxis.set_major_locator(mdates.YearLocator(1))
    Ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    LimpiarEjes(Ax)


def Fx(Px):
    return Px / 1294


def Fy(Py):
    return 1 - Py / 2000


def Caja(X0, Y0, X1, Y1):
    return [Fx(X0), Fy(Y1), Fx(X1) - Fx(X0), Fy(Y0) - Fy(Y1)]


def TextoRico(Fig, X, Y, Segmentos, AnchoMax, Tam=9.5, Interlineado=18, ColorBase=TextoOscuro):
    Renderer = Fig.canvas.get_renderer()
    EscalaPx = Fig.bbox.width / 1294

    def Medir(Texto, Peso):
        Temporal = Fig.text(0, 0, Texto, fontsize=Tam, fontweight=Peso)
        Ancho = Temporal.get_window_extent(Renderer).width / EscalaPx
        Temporal.remove()
        return Ancho

    Espacio = Medir('x x', 'normal') - Medir('xx', 'normal')
    Cursor, Linea = X, 0
    for Texto, Peso, Color in Segmentos:
        for Palabra in Texto.split(' '):
            if not Palabra:
                continue
            Ancho = Medir(Palabra, Peso)
            if Cursor + Ancho > X + AnchoMax and Cursor > X:
                Cursor, Linea = X, Linea + 1
            Fig.text(Fx(Cursor), Fy(Y + Linea * Interlineado), Palabra, fontsize=Tam, fontweight=Peso,
                     color=Color or ColorBase, va='center')
            Cursor += Ancho + Espacio


def EtiquetaBloque(Fig, X, Y, Texto):
    Fig.text(Fx(X), Fy(Y), Texto, fontsize=9.5, fontweight='bold', color=AzulGris, va='center')


def TituloBloque(Fig, X, Y, Texto, Tam=13.5):
    Fig.text(Fx(X), Fy(Y), Texto, fontsize=Tam, fontweight='bold', color=AzulMarino,
             va='top', linespacing=1.05)


def MejorModeloPorSerie(Metricas):
    Mejores = Metricas.loc[Metricas.groupby('Serie')['RMSE'].idxmin()].sort_values('Serie')
    Partes = []
    for Modelo, Grupo in Mejores.groupby('Modelo', sort=True):
        Partes.append(f'{Modelo}: ' + ', '.join(NombresSeries[S] for S in Grupo['Serie']) + '.')
    return ' '.join(Partes)


def ConstruirInfografia(Series, Metricas, Comparativas, Ind, Pronostico):
    Fig = plt.figure(figsize=(11, 17), facecolor=Fondo)
    Comp = Ind['Comparativas']
    Diciembre = Ind['Estacional'][12]
    Septiembre = Ind['Estacional'][9]

    Fig.patches.append(Rectangle((0, Fy(235)), 1, 1 - Fy(235), transform=Fig.transFigure,
                                 facecolor=AzulMarino, zorder=0))
    Fig.text(Fx(59), Fy(40), 'Del colapso a la recuperación: lo que 17 años de viajeros\n'
             'dicen del futuro del turismo en Guatemala', fontsize=26, fontweight='bold',
             color='white', va='top', linespacing=1.1)
    Fig.text(Fx(59), Fy(184), 'Radiografía del flujo migratorio 2009–2026 para decidir campañas, '
             'capacidad y mercados', fontsize=13, color='#D3DBE6', va='center')
    for I, Autor in enumerate(['Roberto Barreda #23354', 'Javier España #23361', 'Angel Esquit #23221']):
        Fig.text(Fx(1279), Fy(178 + I * 17.5), Autor, fontsize=9, color='#D3DBE6',
                 ha='right', va='center')

    Tarjetas = [
        (f'{Ind["Total2019"] / 1e6:.2f} M', AzulMarino,
         'viajeros en 2019,\nel mejor año antes de la pandemia'),
        (f'−{Ind["Caida2020"] * 100:.1f}%', Naranja,
         'caída del mes más bajo de 2020\nfrente al promedio de 2019'),
        (f'{Ind["Recuperacion"]:.0%}', AzulMarino,
         f'de la cifra de 2019 ya se\nrecuperó en 2025 ({Ind["Total2025"] / 1e6:.2f} M)'),
        (f'+{Diciembre:.0%}', AzulMarino,
         'sobre un mes típico:\ndiciembre es el mes más fuerte'),
    ]
    for I, (Cifra, Color, Descripcion) in enumerate(Tarjetas):
        X0 = 58 + I * 298.5
        Fig.patches.append(Rectangle((Fx(X0), Fy(412)), Fx(282), Fy(260) - Fy(412),
                                     transform=Fig.transFigure, facecolor=AzulClaro, zorder=0))
        Fig.text(Fx(X0 + 17), Fy(303), Cifra, fontsize=34, fontweight='bold', color=Color, va='center')
        Fig.text(Fx(X0 + 17), Fy(340), Descripcion, fontsize=9.5, color=TextoOscuro, va='top',
                 linespacing=1.3)

    EtiquetaBloque(Fig, 59, 442, 'BLOQUE 2 · PANORAMA HISTÓRICO 2009–2026')
    TituloBloque(Fig, 59, 462, 'Crecimiento sostenido hasta 2019, colapso en 2020 y recuperación '
                 f'hacia el {Ind["Recuperacion"]:.0%} del nivel previo', Tam=15)
    DibujarHistorico(Fig.add_axes(Caja(145, 525, 1150, 776)), Series, Ind)

    EtiquetaBloque(Fig, 59, 832, 'BLOQUE 3 · ESTACIONALIDAD')
    TituloBloque(Fig, 59, 852, f'El calendario se repite: diciembre sube {Diciembre:.0%} y\n'
                 f'septiembre cae {abs(Septiembre):.0%}')
    DibujarEstacionalidad(Fig.add_axes(Caja(128, 925, 632, 1180)), Ind)

    EtiquetaBloque(Fig, 661, 832, 'BLOQUE 3 · SERIES COMPARADAS')
    TituloBloque(Fig, 661, 852, 'Cuatro preguntas, cuatro ganadores entre las siete\nseries')
    DibujarTabla(Fig.add_axes(Caja(663, 916, 1235, 1152)), Comparativas)
    Fig.text(Fx(661), Fy(1170), 'En naranja, el máximo de cada columna. La vía marítima cayó menos por '
             'su nivel bajo e\nirregular, no por resistencia.', fontsize=8.5, color=AzulGris, va='top',
             linespacing=1.2)

    EtiquetaBloque(Fig, 59, 1227, 'BLOQUE 4 · MODELOS')
    TituloBloque(Fig, 59, 1247, 'Holt-Winters fue el modelo con menor error para el\ntotal de viajeros')
    DibujarModelos(Fig.add_axes(Caja(183, 1345, 632, 1585)), Metricas)

    EtiquetaBloque(Fig, 661, 1227, 'BLOQUE 4 · PRONÓSTICO (CONJUNTO DE PRUEBA)')
    TituloBloque(Fig, 661, 1247, 'La recuperación real superó la estimación central en\n'
                 f'los {len(Pronostico[1])} meses de prueba')
    DibujarPronostico(Fig.add_axes(Caja(720, 1320, 1233, 1565)), *Pronostico)

    TextoRico(Fig, 59, 1613, [('Mejor modelo por serie', 'bold', AzulMarino),
                              ('· ' + MejorModeloPorSerie(Metricas), 'normal', None),
                              ('Errores del gráfico en viajeros por mes; menor es mejor.', 'normal', AzulGris)],
              AnchoMax=1175, Tam=8.8, Interlineado=16)

    EtiquetaBloque(Fig, 59, 1657, 'BLOQUE 5 · HALLAZGOS Y RECOMENDACIONES PARA EL INGUAT')
    TituloBloque(Fig, 59, 1677, 'Cinco acciones que salen de los datos')
    Acciones = [
        ('Programar campañas y capacidad según el calendario.',
         f'Reforzar diciembre (+{Diciembre:.0%}), marzo–abril y julio; usar los valles de '
         'septiembre–octubre y febrero para promociones.'),
        ('Reducir la dependencia del canal aéreo.',
         f'Fue el más golpeado (−{Comp.loc["Via_Aerea", "Caida_Pandemia_pct"]:.1f}% en 2020) y casi todo '
         'entra por una sola frontera, La Aurora: preparar un plan de contingencia.'),
        ('Cuidar la base centroamericana.',
         f'El Salvador es el mayor emisor acumulado ({Ind["AcumuladoSalvador"] / 1e6:.1f} millones de '
         f'viajeros) y su tendencia es firme ({Comp.loc["Pais_El Salvador", "Fuerza_Tendencia"]:.2f}).'),
        ('Alinear la oferta con el calendario vacacional de Estados Unidos,',
         'el mercado más estacional '
         f'({Comp.loc["Pais_Estados Unidos de América", "Fuerza_Estacional"]:.2f}), con campañas '
         'dirigidas a sus meses pico.'),
        ('Reentrenar los modelos con datos posteriores a la pandemia',
         'y comunicar las proyecciones como rangos: todos se quedaron cortos frente a la recuperación real.'),
    ]
    for I, (Accion, Detalle) in enumerate(Acciones):
        Y = [1731, 1774, 1816, 1858, 1901][I]
        Fig.text(Fx(65), Fy(Y + 3), str(I + 1), fontsize=14, fontweight='bold', color=Naranja,
                 ha='center', va='center')
        TextoRico(Fig, 95, Y, [(Accion, 'bold', AzulMarino), (Detalle, 'normal', None)],
                  AnchoMax=1150, Tam=9.5, Interlineado=17.5)

    Fig.lines.append(plt.Line2D([Fx(59), Fx(1236)], [Fy(1935), Fy(1935)], transform=Fig.transFigure,
                                color=Rejilla, linewidth=0.6))
    TextoRico(Fig, 59, 1951, [('Fuente:', 'bold', AzulGris),
                              ('Base de Migración 2009–2026 INGUAT/IGM. Serie = Turista + Excursionista '
                               '(total consistente). Modelo: Holt-Winters aditivo entrenado con ene 2009–abr 2021 '
                               'y evaluado en may 2021–jun 2026; intervalo del 80 % simulado con '
                               f'{Simulaciones:,} trayectorias (límite inferior recortado en 0).', 'normal', AzulGris)],
              AnchoMax=1175, Tam=7.6, Interlineado=14)
    TextoRico(Fig, 59, 1980, [('Créditos:', 'bold', AzulGris),
                              ('Javier España, Ángel Esquit y Roberto Barreda · CC3084 Data Science · '
                               'Universidad del Valle de Guatemala · 2026', 'normal', AzulGris)],
              AnchoMax=1175, Tam=7.6, Interlineado=14)
    return Fig


def ExportarGraficas(Series, Metricas, Comparativas, Ind, Pronostico):
    Graficas = {
        'Bloque2_Historico': ((11, 3.2), lambda Ax: DibujarHistorico(Ax, Series, Ind)),
        'Bloque3_Estacionalidad': ((5.6, 3.2), lambda Ax: DibujarEstacionalidad(Ax, Ind)),
        'Bloque3_SeriesComparadas': ((5.6, 2.4), lambda Ax: DibujarTabla(Ax, Comparativas)),
        'Bloque4_Modelos': ((5.6, 3.2), lambda Ax: DibujarModelos(Ax, Metricas)),
        'Bloque4_Pronostico': ((5.6, 3.2), lambda Ax: DibujarPronostico(Ax, *Pronostico)),
    }
    for Nombre, (Tamano, Dibujar) in Graficas.items():
        Fig, Ax = plt.subplots(figsize=Tamano, facecolor=Fondo)
        Dibujar(Ax)
        Fig.savefig(CarpetaGraficas / f'{Nombre}.png', dpi=220, bbox_inches='tight')
        plt.close(Fig)


def Main():
    CarpetaGraficas.mkdir(parents=True, exist_ok=True)
    Series, Metricas, Comparativas = CargarDatos()
    Ind = CalcularIndicadores(Series, Comparativas)
    Pronostico = AjustarHoltWinters(Series)

    Fig = ConstruirInfografia(Series, Metricas, Comparativas, Ind, Pronostico)
    Fig.savefig(CarpetaSalida / 'Infografia.png', dpi=220)
    Fig.savefig(CarpetaSalida / 'Infografia.pdf')
    plt.close(Fig)
    ExportarGraficas(Series, Metricas, Comparativas, Ind, Pronostico)

    print(f'2019: {Ind["Total2019"]:,.0f} | 2025: {Ind["Total2025"]:,.0f} | '
          f'Recuperación: {Ind["Recuperacion"]:.1%} | Caída 2020: {Ind["Caida2020"]:.1%}')
    print('Estacionalidad 2015–2019:', (Ind['Estacional'] * 100).round(1).tolist())
    print('Meses de prueba sobre la predicción:', int((Pronostico[1] > Pronostico[2]).sum()),
          'de', len(Pronostico[1]))


if __name__ == '__main__':
    Main()
