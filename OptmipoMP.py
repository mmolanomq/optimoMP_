# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 11:54:54 2025

@author: Usuario
"""

import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import StringIO
import requests # Necesario para enviar datos a Google
import time

# ==============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==============================================================================
st.set_page_config(page_title="Diseño Avanzado de Micropilotes", layout="wide", page_icon="🏗️")

# Estilos CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] { background-color: #FFFFFF; border-bottom: 2px solid #FF4B4B; }
</style>
""", unsafe_allow_html=True)

# URL DE TU SCRIPT DE GOOGLE (LO OBTENDRÁS EN EL PASO 2)
# Por ahora déjalo vacío o pon una URL de prueba
GOOGLE_SCRIPT_URL = "AQUI_PEGARAS_TU_URL_DEL_PASO_2"

# ==============================================================================
# 0. SISTEMA DE REGISTRO
# ==============================================================================
if 'usuario_registrado' not in st.session_state:
    st.session_state['usuario_registrado'] = False
if 'datos_usuario' not in st.session_state:
    st.session_state['datos_usuario'] = {}

def enviar_a_google_sheets(datos):
    """Envía los datos al Webhook de Google Apps Script"""
    if GOOGLE_SCRIPT_URL == "AQUI_PEGARAS_TU_URL_DEL_PASO_2":
        return True # Modo simulación si no has configurado la URL
    
    try:
        response = requests.post(GOOGLE_SCRIPT_URL, json=datos)
        if response.status_code == 200:
            return True
        else:
            return False
    except Exception as e:
        st.error(f"Error de conexión: {e}")
        return False

def mostrar_registro():
    """Muestra la pantalla de bloqueo/registro"""
    col_main1, col_main2, col_main3 = st.columns([1, 2, 1])
    
    with col_main2:
        st.markdown("## 🔒 Acceso a Herramienta de Ingeniería")
        st.info("Para acceder a la calculadora de optimización y huella de carbono, por favor regístrese.")
        
        with st.form("formulario_registro", clear_on_submit=False):
            nombre = st.text_input("Nombre Completo")
            col1, col2 = st.columns(2)
            empresa = col1.text_input("Empresa / Universidad")
            cargo = col2.selectbox("Cargo", ["Ingeniero Geotecnista", "Ingeniero Estructural", "Constructor/Residente", "Estudiante", "Otro"])
            email = st.text_input("Correo Electrónico Corporativo")
            
            acepto = st.checkbox("Acepto los términos de uso.")
            
            submit = st.form_submit_button("🚀 INGRESAR AL SISTEMA", type="primary")
            
            if submit:
                if not nombre or not email or not empresa:
                    st.warning("Por favor complete los campos obligatorios.")
                elif not acepto:
                    st.warning("Debe aceptar los términos para continuar.")
                else:
                    # Preparar datos
                    datos = {
                        "nombre": nombre,
                        "empresa": empresa,
                        "cargo": cargo,
                        "email": email,
                        "fecha": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    # Guardar en sesión
                    st.session_state['datos_usuario'] = datos
                    st.session_state['usuario_registrado'] = True
                    
                    # Enviar a Google Sheets (Segundo plano)
                    with st.spinner("Registrando acceso..."):
                        exito = enviar_a_google_sheets(datos)
                        if exito:
                            st.success("¡Registro exitoso!")
                            time.sleep(1)
                            st.rerun() # Recarga la página para quitar el formulario
                        else:
                            st.error("Hubo un problema guardando sus datos, pero puede ingresar.")
                            time.sleep(2)
                            st.rerun()

# ==============================================================================
# APLICACIÓN PRINCIPAL
# ==============================================================================
def app_principal():
    # Barra lateral con info de sesión
    with st.sidebar:
        st.success(f"👤 **{st.session_state['datos_usuario'].get('nombre', 'Usuario')}**")
        st.caption(f"{st.session_state['datos_usuario'].get('cargo', '')}")
        if st.button("Cerrar Sesión"):
            st.session_state['usuario_registrado'] = False
            st.session_state['datos_usuario'] = {}
            st.rerun()
        st.markdown("---")

    st.title("🏗️ Sistema de Diseño de Micropilotes")
    st.markdown("Optimización de diseño y análisis geotécnico integrado.")

    # Crear pestañas principales
    tab_diseno, tab_geo = st.tabs(["📐 Diseño & Optimización", "🌍 Correlaciones Geotécnicas (SPT)"])

    # ... AQUI VA EL RESTO DE TU CODIGO DE LAS PESTAÑAS (Pestaña 1 y 2) ...
    # ... COPIA Y PEGA EL CONTENIDO DE "PESTAÑA 1" Y "PESTAÑA 2" DE TU CÓDIGO ANTERIOR AQUÍ ...
    # (Para no hacer la respuesta infinita, asumo que mantienes tu lógica interna igual)
    
    with tab_diseno:
        st.info("Aquí va tu módulo de diseño (copiar del código anterior)")
        # ... (Tu código de la pestaña 1)
        
    with tab_geo:
        st.info("Aquí va tu módulo de geotecnia (copiar del código anterior)")
        # ... (Tu código de la pestaña 2)


# ==============================================================================
# CONTROL DE FLUJO
# ==============================================================================
if st.session_state['usuario_registrado']:
    app_principal()
else:
    mostrar_registro()