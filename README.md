# Genal Activation Family 🧠

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20304195.svg)](https://doi.org/10.5281/zenodo.20304195)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0009--6495--4085-green)](https://orcid.org/0009-0009-6495-4085)

A learnable activation function that adapts its curvature to each task. Outperforms ReLU, GELU, and Swish on 16 diverse benchmarks.

**Author:** Genal Ediso Lombano Pineda — Independent Researcher, Venezuela 🇻🇪  
**Paper:** [zenodo.org/records/20304195](https://zenodo.org/records/20304195)  
**ORCID:** [0009-0009-6495-4085](https://orcid.org/0009-0009-6495-4085)  
**Email:** genallombano@gmail.com

---

## 📈 Resumen del Framework

**Genal Activation Family** es una suite matemática de funciones de activación adaptativas diseñada completamente de forma nativa e independiente en PyTorch. El núcleo de esta tecnología radica en su capacidad para aprender parámetros geométricos durante el proceso de entrenamiento, optimizando el flujo de gradientes según la complejidad de cada tarea.

### Variantes de la Familia
1. **GenalAdvanced (`main.py`)**: Utiliza una curvatura `k` aprendible e independiente por cada canal de la red neuronal. Es la variante estándar para arquitecturas de visión y procesamiento de señales.
2. **GenalShift (`main_shift.py`)**: Añade un parámetro de desplazamiento horizontal `beta` aprendible por canal, además de la curvatura `k`. Esta flexibilidad matemática permite una dinámica de activación superior en entornos de alta complejidad predictiva.

---

## 🚀 Rendimiento y Validación Empírica

El comportamiento estadístico y matemático del algoritmo ha sido evaluado rigurosamente mediante múltiples pruebas modulares independientes, demostrando una notable estabilidad de gradientes y la prevención del fenómeno de neuronas muertas:

* **Clasificación de Imágenes (CIFAR-10)**: Alcanza un **81.91% de precisión**, superando a la función tradicional ReLU en un **+7.18%**.
* **Problemas de Física Matemática (Navier-Stokes)**: Logró una pérdida de error (**Loss**) **44 veces menor** en comparación con ReLU durante las pruebas de simulación de fluidos dinámicos.
* **Diagnóstico Médico Predictivo**: Implementada con éxito en modelos de clasificación de datos clínicos, alcanzando métricas estables superiores al 95% de precisión (Parkinson: 97.44%, Oncología: 97.37%, Cardiología: 95.08%).

---

## 🛠️ Requisitos e Instalación

El framework está diseñado para entornos de desarrollo basados en Python 3 y está completamente disponible a través del índice global PyPI.

### Dependencias requeridas:
* `torch` (PyTorch)
* `torchvision`
* `matplotlib` (para la generación de curvas de convergencia)

### Instalación vía pip:
```bash
pip install genal-activation
import torch
import torch.nn as nn
from genal_activation import GenalAdvanced

class ModeloEjemplo(nn.Module):
    def __init__(self):
        super(ModeloEjemplo, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        # Inicializa la activación adaptativa pasando el número de canales
        self.act = GenalAdvanced(num_channels=64) 

    def forward(self, x):
        return self.act(self.conv(x))
Lombano Pineda, G. E. (2024). Genal Activation Family: A learnable activation function suite for PyTorch. Zenodo. DOI: 10.5281/zenodo.20304195

