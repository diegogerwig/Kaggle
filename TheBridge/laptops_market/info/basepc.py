'''
He creado un modulo para trabajar con el y poder realizar la limpieza del train y del test de forma automatica

En este modulo encontramos algunas librerias, un objeto cleaner pensado para limpiar los datas.
Dentro de este objeto cleaner enocntramos atributos de clase ordinal y scaler para "numerizar"
variables categoricas. Con el train entrenamos estos atributos de clase y el test se transforma en base a ese entrenamiento.
El objeto contiene funciones para limpiar cada columna individualemnte y arrojar info como cardinalidar, dropear el target, guardar 
el .csv y alguno mas.

Tambien encontramos funciones que sirven para el objeto


La preparacion de este modulo se encuentra en el archivo "trabajo_data.ipynb" si bien aqui esta todo el resultado final
'''



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import re

train = pd.read_csv('./data/train.csv').drop(['laptop_ID'], axis= 1)
test = pd.read_csv('./data/test.csv').drop(['laptop_ID'], axis= 1)


class Cleaner(object):

    ordinal_company = OrdinalEncoder(handle_unknown= 'use_encoded_value', unknown_value= train['Company'].nunique()+1)
    ordinal_product = OrdinalEncoder(handle_unknown= 'use_encoded_value', unknown_value= train['Product'].nunique()+1)
    ordinal_typename = OrdinalEncoder(handle_unknown= 'use_encoded_value', unknown_value= train['TypeName'].nunique()+1)
    ordinal_screen = OrdinalEncoder(handle_unknown= 'use_encoded_value', unknown_value= train['ScreenResolution'].nunique()+1)
    ordinal_memory = OrdinalEncoder(handle_unknown= 'use_encoded_value', unknown_value= train['Memory'].nunique()+1)
    ordinal_gpu = OrdinalEncoder(handle_unknown= 'use_encoded_value', unknown_value= train['Gpu'].nunique()+1)
    ordinal_opsys = OrdinalEncoder(handle_unknown= 'use_encoded_value', unknown_value= train['OpSys'].nunique()+1)

    scaler = StandardScaler()

    def __init__(self, data: pd.DataFrame, test: bool = False, target:str= None):

        self.data = data
        self.data_md = self.data.copy()

        self.test = test
        self.target = target
    
    def show_cardinality(self):

        for feature in self.data.columns:

            cardinalidad = self.data[feature].nunique()
            porc_cardinalidad = cardinalidad / len(self.data)

            print(feature.upper())
            print('-'*10)
            print(f'Cardinalidad feature "{feature}" -> {cardinalidad}.')
            print(f'Cardinalidad tanto por 1 feature "{feature}" -> {porc_cardinalidad}.\n')
    
    def show_cardinality_md(self):

        for feature in self.data_md.columns:

            cardinalidad = self.data_md[feature].nunique()
            porc_cardinalidad = cardinalidad / len(self.data_md)

            print(feature.upper())
            print('-'*10)
            print(f'Cardinalidad feature "{feature}" -> {cardinalidad}.')
            print(f'Cardinalidad tanto por 1 feature "{feature}" -> {porc_cardinalidad}.\n')
    
    def drop_target(self):

        self.data_md.drop([self.target], axis= 1, inplace= True)
    
    def clean_company(self):

        if self.test == False:
            self.data_md['Company'] = Cleaner.ordinal_company.fit_transform(self.data_md[['Company']])
        else:
            self.data_md['Company'] = Cleaner.ordinal_company.transform(self.data_md[['Company']])

    def clean_product(self):

        if self.test == False:
            self.data_md['Product'] = Cleaner.ordinal_product.fit_transform(self.data_md[['Product']])
        else:
            self.data_md['Product'] = Cleaner.ordinal_product.transform(self.data_md[['Product']])
    
    def clean_typename(self):

        if self.test == False:
            self.data_md['TypeName'] = Cleaner.ordinal_typename.fit_transform(self.data_md[['TypeName']])
        else:
            self.data_md['TypeName'] = Cleaner.ordinal_typename.transform(self.data_md[['TypeName']])

    def clean_screenresolution(self):

        if self.test == False:
            self.data_md['ScreenResolution'] = self.data_md['ScreenResolution'].apply(resolution_converter)
            self.data_md['ScreenResolution'] = Cleaner.ordinal_screen.fit_transform(self.data_md[['ScreenResolution']])
        else:
            self.data_md['ScreenResolution'] = self.data_md['ScreenResolution'].apply(resolution_converter)
            self.data_md['ScreenResolution'] = Cleaner.ordinal_screen.transform(self.data_md[['ScreenResolution']])


    def clean_cpu(self):
        
        if self.test == False:
            self.data_md['Cpu'] = self.data_md['Cpu'].apply(cpu_converter)
        else:
            self.data_md['Cpu'] = self.data_md['Cpu'].apply(cpu_converter)
    
    def clean_ram(self):
        if self.test == False:
            self.data_md['Ram'] = self.data_md['Ram'].apply(ram_converter)
        else:
            self.data_md['Ram'] = self.data_md['Ram'].apply(ram_converter)
    
    def clean_memory(self):

        if self.test == False:
            self.data_md['Memory'] = Cleaner.ordinal_memory.fit_transform(self.data_md[['Memory']])
        else:
            self.data_md['Memory'] = Cleaner.ordinal_memory.transform(self.data_md[['Memory']])

    def clean_gpu(self):

        if self.test == False:
            self.data_md['Gpu'] = self.data_md['Gpu'].apply(gpu_converter)
            self.data_md['Gpu'] = Cleaner.ordinal_gpu.fit_transform(self.data_md[['Gpu']])
        else:
            self.data_md['Gpu'] = self.data_md['Gpu'].apply(gpu_converter)
            self.data_md['Gpu'] = Cleaner.ordinal_gpu.transform(self.data_md[['Gpu']])

    def clean_opsys(self):

        if self.test == False:
            self.data_md['OpSys'] = Cleaner.ordinal_opsys.fit_transform(self.data_md[['OpSys']])
        else:
            self.data_md['OpSys'] = Cleaner.ordinal_opsys.transform(self.data_md[['OpSys']])

    def clean_weight(self):

        if self.test == False:
            self.data_md['Weight'] = self.data_md['Weight'].apply(weight_converter)
        else:
            self.data_md['Weight'] = self.data_md['Weight'].apply(weight_converter)
    
    def escalar(self):

        if self.test == False:
            no_escalar = self.data_md[[self.target, 'id']]
            self.data_md = pd.DataFrame(Cleaner.scaler.fit_transform(self.data_md.drop([self.target, 'id'], axis= 1)), columns= self.data_md.drop([self.target, 'id'], axis= 1).columns, index= self.data_md.index)
            self.data_md[[self.target, 'id']] = no_escalar
        else:
            no_escalar = self.data_md['id']
            self.data_md = pd.DataFrame(Cleaner.scaler.transform(self.data_md.drop(['id'], axis= 1)), columns= self.data_md.drop(['id'], axis= 1).columns, index= self.data_md.index)
            self.data_md['id'] = no_escalar
            
    def clean_all(self):

        self.clean_ram()
        self.clean_company()
        self.clean_cpu()
        self.clean_gpu()
        self.clean_memory()
        self.clean_opsys()
        self.clean_product()
        self.clean_screenresolution()
        self.clean_typename()
        self.clean_weight()
        self.escalar()

    def get_cleaned_train(self) -> pd.DataFrame:

        return self.data_md
    
    def save_data(self, name:str):

        self.data_md.to_csv(f'./data/{name}.csv', index=False)



def resolution_converter(texto):

    patron = r'\d{3,4}x\d{3,4}'
    res = re.findall(patron, texto)

    return res[0]

def cpu_converter(texto):

    patron = r'\d{1,2}\.\d{1,2}GHz'
    res = re.findall(patron, texto)

    if res == []:

        patron = r'\dGHz'
        res = re.findall(patron, texto)
    
    res[0] = res[0].replace('GHz', '')

    return float(res[0])

def ram_converter(texto):

    patron = r'\d{1,2}'
    res = re.findall(patron, texto)

    return int(res[0])

def gpu_converter(texto):

    patrones = [r'AMD', r'Nvidia', r'Intel']

    for i in patrones:
        res = re.findall(i, texto)
        if res != []:
            break
    try:
        return res[0]
    except:
        return 'Others'
    
def weight_converter(texto):

    patron = r'\d{1,2}\.\d{1,2}'
    res = re.findall(patron, texto)

    if res == []:

        patron = r'\d'
        res = re.findall(patron, texto)

    return float(res[0])