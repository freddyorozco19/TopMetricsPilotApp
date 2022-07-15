# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 12:05:48 2022
@author: ACER
"""

import pandas as pd
import numpy as np
import streamlit as st
from mplsoccer.radar_chart2 import Radar
import matplotlib.font_manager as font_manager
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components
import io
from matplotlib import colors as mcolors

##################################################################################################################################
## Import resources
font_path = 'fonts/keymer-bold.otf'  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

font_path = 'fonts/BasierCircle-Italic.ttf'  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop1 = font_manager.FontProperties(fname=font_path)

#####################################################################################################################################################
            
def colorlist(color1, color2, num):
    """Generate list of num colors blending from color1 to color2"""
    result = [np.array(color1), np.array(color2)]
    while len(result) < num:
        temp = [result[0]]
        for i in range(len(result)-1):
            temp.append(np.sqrt((result[i]**2+result[i+1]**2)/2))
            temp.append(result[i+1])
        result = temp
    indices = np.linspace(0, len(result)-1, num).round().astype(int)
    return [result[i] for i in indices] 

#####################################################################################################################################################

hex_list = ['#222222', '#390E1A', '#FF0046']

#hex_list = ['#222222', '#273454', '#0054A5']

cmap = sns.cubehelix_palette(start=.25, rot=-.3, light=1, reverse=True, as_cmap=True)
cmap2 = sns.diverging_palette(250, 344, as_cmap=True, center="dark")
cmap3 = sns.color_palette("dark:#FF0046", as_cmap=True)


def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]


def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp


##################################################################################################################################

df1 = pd.read_excel('ColombiaApertura22_AllMetricsCalculated_DataCleaning.xlsx')
df2 = pd.read_excel('ArgentinaApertura22_AllMetricsCalculated_DataCleaning.xlsx')
df3 = pd.read_excel('BoliviaApertura22_AllMetricsCalculated_DataCleaning.xlsx')
df4 = pd.read_excel('ChileApertura22_AllMetricsCalculated_DataCleaning.xlsx')
df5 = pd.read_excel('EcuadorApertura22_AllMetricsCalculated_DataCleaning.xlsx')
df6 = pd.read_excel('ParaguayApertura22_AllMetricsCalculated_DataCleaning.xlsx')
df7 = pd.read_excel('PeruApertura22_AllMetricsCalculated_DataCleaning.xlsx')
df8 = pd.read_excel('VenezuelaApertura22_160622_AllMetricsCalculated_DataCleaning.xlsx')
df9 = pd.read_excel('UruguayApertura22_160622_AllMetricsCalculated_DataCleaning.xlsx')
df10 = pd.read_excel('Brasil22_F16_130722_AllMetricsCalculated_DataCleaning.xlsx')
#df2 = pd.read_excel('')

##################################################################################################################################
#Streamlit configuration
st.set_page_config(layout="wide")

#st.title(f"DIAGRAMA RADIAL")
st.markdown("""---""")

with st.sidebar:
    
    st.image("https://i.ibb.co/qjvrH5y/win.png", width=250) 
    
    #COUNTRY CHOISE
    countries = ["Todos los países", "Argentina", "Bolivia", "Brasil", "Chile", "Colombia", "Ecuador", "Paraguay", "Perú", "Uruguay", "Venezuela"]
    cousel = st.selectbox("Seleccionar país:", countries)
    
    if cousel == "Colombia":
        df = df1
    elif cousel == "Argentina":
        df = df2
    elif cousel == "Bolivia":
        df = df3
    elif cousel == "Chile":
        df = df4
    elif cousel == "Ecuador":
        df = df5
    elif cousel == "Paraguay":
        df = df6
    elif cousel == "Perú":
        df = df7
    elif cousel == "Uruguay":
        df = df8
    elif cousel == "Venezuela":
        df = df9
    elif cousel == "Brasil":
        df = df10
    elif cousel == "Todos los países":
        df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10], axis = 0)
    
    #LEAGUE CHOISE
    leagues = ["Todos los torneos", "Liga Bet Play - Apertura 2022", "Torneo Bet Play - Apertura 2022"]
    ligasel = st.selectbox("Seleccionar el campeonato:", leagues)
    
    
    #ALL METRICS
    dftra = df.transpose()            
    dftra = dftra.reset_index()            
    metrics = list(dftra['index'].drop_duplicates())
    metrics = metrics[9:]
    metsel = st.selectbox('Selecciona la métrica:', metrics)    
    
    #MINS FILTER
    maxmin = df['Minutes played'].max() + 5
    minsel = st.slider('Filtro de minutos (%):', 0, 100)
    minsel1 = (minsel*maxmin)/100
    df = df[df['Minutes played'] >= minsel1].reset_index()
    dfc = df
    #df5 = df5[df5['Minutes played'] >= minsel1].reset_index()
    
    #SELECT POSITION OPTION
    positions = list(df['Pos1'].drop_duplicates())
    auxpos = "ALL"
    positions.append(auxpos)
    possel = st.selectbox("Filtrar por posición:", positions)

    if possel == "ALL":
        df = dfc
    else:
        df = df[df['Pos1'] == possel]
        
    #AGE FILTER
    agesel = st.slider('Filtro de edad:', 15, 45, (15, 45), 1)   
    df = df[df['Age'] <= agesel[1]]
    df = df[df['Age'] >= agesel[0]]
             
    
    

space0, space1 = st.columns((1.16, 0.85))

with space0:
    
    fig, ax = plt.subplots(figsize = (12,12), dpi=600)
    fig.set_facecolor('#050E1E')
    ax.patch.set_facecolor('#050E1E')
    #ax.axis("off")
    
    plt.setp(ax.get_yticklabels(), fontproperties = prop, fontsize=18, color='#FFF')
    plt.setp(ax.get_xticklabels(), fontproperties = prop, fontsize=20, color=(1,1,1,1))

    df = df.sort_values(by=[metsel], ascending=True)

    Y1 = df[metsel].tail(10)
    Y2 = df['Total second assists'].tail(10)
    Y3 = df['Total third assists'].tail(10)
    Z = df['Player'].tail(10).str.upper()

    colors = colorlist((0.0196078431372549, 0.0549019607843137, 0.1176470588235294), (1, 0, 0.2784313725490196), 60)
    
    ax.barh(Z, Y1, edgecolor=(1,1,1,0.5), lw = 1, color=colors)
    #ax.barh(Z, Y2, left = Y1, facecolor='#1C2E46', edgecolor=(1,1,1,0.5), lw = 1)
    #ax.barh(Z, Y3, left = Y2+Y1, facecolor='#404C5B', edgecolor=(1,1,1,0.5), lw = 1)

    plt.xlabel(metsel, fontproperties = prop, color = 'w', fontsize=15, labelpad=20)
    #ax.set_xticks([0, 5, 10])

    #ax.set_xlim(0, 18)
    ax.tick_params(axis='y', which='major', pad=15)
    
    spines = ['top','bottom','left','right']
    for x in spines:
        if x in spines:
            ax.spines[x].set_visible(False)
            
    for patch in ax .patches:
        x,y = patch.get_xy()
        w, h = patch.get_width(), patch.get_height()
        if w > 0:
            aux = ax.text(x+w/2, y+h/2, str(round(w,2)),
                          ha='center', va='center', 
                          color='w', fontproperties=prop, fontsize=18, zorder=10
                          )
            aux.set_path_effects([path_effects.withStroke(linewidth=1, foreground="#000")])        
            
    
    st.pyplot(fig, bbox_inches="tight", dpi=600, format="png")   
    
with space1:
    
    df1 = df
    df1 = df1.sort_values(by=[metsel], ascending=False)
    dfaux1 = df1[["Player", "Team", "Pos1", "Age", "90s"]]
    
    df1 = df1[metsel]
    #df1 = df1.head(10)
    
    df1 = pd.concat([dfaux1, df1], axis=1)
    
    st.table(df1.head(10).style.set_precision(2))    