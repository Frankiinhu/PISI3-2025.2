#!/usr/bin/env python
# Gera um dataset de exemplo para o dashboard (DATASET FINAL WRDP.csv)
import os
import csv
import random

os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'data'), exist_ok=True)
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'DATASET FINAL WRDP.csv'))

headers = ['Gênero', 'Idade', 'Diagnóstico', 'Temperatura (°C)', 'Umidade', 'Velocidade do Vento (km/h)']
diagnoses = ['H1', 'H2', 'H3', 'H4', 'H5']

with open(path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    for i in range(400):
        gender = random.choice([0, 1])
        age = random.randint(0, 90)
        diag = random.choices(diagnoses, weights=[30, 20, 20, 15, 15])[0]
        temp = round(random.uniform(15.0, 35.0), 1)
        hum = random.randint(20, 95)
        wind = round(random.uniform(0.0, 40.0), 1)
        writer.writerow([gender, age, diag, temp, hum, wind])

print(f"Dataset de exemplo criado em: {path}")
