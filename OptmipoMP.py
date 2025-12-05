import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass
import requests
import time
from io import BytesIO

# Manejo de libreria PDF opcional para evitar crash si no está instalada
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ==============================================================================
# 0. CONFIGURACIÓN GLOBAL
# ==============================================================================
st.set_page_config(page_title="GeoMicropile Suite", layout="wide", page_icon="🏗️")

# Estilos CSS Profesionales
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; background-color: #f8f9fa; border-radius: 5px 5px 0px 0px;
        padding: 10px; font-weight: 600;
    }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-top: 3px solid #0047AB; }
    h1, h2, h3 { color: #2C3E50; font-family: 'Helvetica', sans-serif; }
    .metric-box { border: 1px solid #e0e0e0; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MÓDULO 1: AUTENTICACIÓN Y DATOS (AuthModule)
# ==============================================================================
class AuthModule:
    GOOGLE_SCRIPT_URL = "" # Pega tu URL aquí

    @staticmethod
    def check_auth():
        if 'usuario_registrado' not in st.session_state:
            st.session_state['usuario_registrado'] = False
        if 'datos_usuario' not in st.session_state:
            st.session_state['datos_usuario'] = {}

    @staticmethod
    def login_form():
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.markdown("## 🔒 GeoMicropile Suite")
            st.info("Plataforma Modular de Diseño y Optimización.")
            with st.form("login"):
                nombre = st.text_input("Nombre Completo")
                empresa = st.text_input("Empresa")
                email = st.text_input("Correo Corporativo")
                cargo = st.selectbox("Cargo", ["Ingeniero Geotecnista", "Ingeniero Estructural", "Estudiante"])
                
                if st.form_submit_button("INGRESAR"):
                    if nombre and email:
                        st.session_state['usuario_registrado'] = True
                        st.session_state['datos_usuario'] = {'nombre': nombre, 'empresa': empresa}
                        # Enviar dato asincrono
                        AuthModule.send_to_sheets({'fecha': time.strftime("%Y-%m-%d"), 'nombre': nombre, 'email': email, 'cargo': cargo})
                        st.rerun()
                    else:
                        st.error("Datos incompletos.")

    @staticmethod
    def send_to_sheets(data):
        if not AuthModule.GOOGLE_SCRIPT_URL: return
        try: requests.post(AuthModule.GOOGLE_SCRIPT_URL, json=data, timeout=2)
        except: pass

# ==============================================================================
# MÓDULO 2: BASE DE DATOS Y MATERIALES (DatabaseModule)
# ==============================================================================
class DatabaseModule:
    @staticmethod
    def get_dywidag():
        """Retorna DataFrame con sistemas Dywidag actualizados."""
        data = {
            "Sistema": ["R32-280", "R32-360", "R38-500", "R38-550", "R51-660", "R51-800", "T76-1200", "T76-1600"],
            "D_ext_mm": [32, 32, 38, 38, 51, 51, 76, 76],
            "As_mm2": [410, 510, 750, 800, 970, 1150, 1610, 1990],
            "fy_MPa": [535, 550, 533, 560, 555, 556, 620, 600]
        }
        return pd.DataFrame(data)

    @staticmethod
    def get_soil_defaults():
        return pd.DataFrame([
            {"Espesor_m": 4.0, "Tipo": "Arcilla", "N_SPT": 5, "Su_kPa": 40, "Kh_kNm3": 8000, "Alpha_Manual": 0.0, "F_Exp": 1.1},
            {"Espesor_m": 8.0, "Tipo": "Arena", "N_SPT": 25, "Su_kPa": 0, "Kh_kNm3": 25000, "Alpha_Manual": 0.0, "F_Exp": 1.2},
            {"Espesor_m": 6.0, "Tipo": "Roca", "N_SPT": 50, "Su_kPa": 0, "Kh_kNm3": 90000, "Alpha_Manual": 400.0, "F_Exp": 1.0},
        ])

# ==============================================================================
# MÓDULO 3: GEOTECNIA (GeotechModule)
# ==============================================================================
@dataclass
class Layer:
    z_top: float; z_bot: float; tipo: str; alpha: float; kh: float; f_exp: float
    def contains(self, z): return self.z_top <= z <= self.z_bot

class GeotechModule:
    @staticmethod
    def process_input(df):
        """Procesa entradas y calcula correlaciones (Wolff, Kulhawy, FHWA)."""
        res = []
        z_acum = 0
        for _, r in df.iterrows():
            try:
                esp = float(r['Espesor_m']); tipo = r['Tipo']
                n = float(r['N_SPT']); su = float(r['Su_kPa'])
                kh = float(r['Kh_kNm3']); a_man = float(r['Alpha_Manual'])
                f_exp = float(r.get('F_Exp', 1.1))
            except: continue

            # Correlaciones
            alpha = 0
            if tipo == "Arena": alpha = min(3.8 * n, 250)
            elif tipo == "Arcilla":
                if su < 25: alpha = 20
                elif su < 50: alpha = 40
                elif su < 100: alpha = 70
                else: alpha = 100
            else: alpha = 300 # Roca
            
            alpha_final = a_man if a_man > 0 else alpha
            z_fin = z_acum + esp
            
            res.append({
                "z_ini": z_acum, "z_fin": z_fin, "Espesor_m": esp, "Tipo": tipo,
                "Alpha_Design": alpha_final, "N_SPT": n, "Kh_kNm3": kh, "F_Exp": f_exp
            })
            z_acum = z_fin
        return pd.DataFrame(res)

    @staticmethod
    def plot_stratigraphy(df_geo):
        """Genera gráfica compuesta: Perfil Visual + Parámetros."""
        fig, axs = plt.subplots(1, 3, figsize=(10, 6), sharey=True)
        z_max = df_geo['z_fin'].max() + 1
        
        # Arrays para step plot seguro
        z_plt, n_plt, a_plt = [], [], []
        for _, r in df_geo.iterrows():
            z_plt.extend([r['z_ini'], r['z_fin']])
            n_plt.extend([r['N_SPT'], r['N_SPT']])
            a_plt.extend([r['Alpha_Design'], r['Alpha_Design']])
            
        axs[0].plot(n_plt, z_plt, 'b'); axs[0].set_title("N-SPT")
        axs[0].invert_yaxis(); axs[0].grid(True, ls=':')
        
        axs[1].plot(a_plt, z_plt, 'r'); axs[1].set_title("Alpha Bond (kPa)")
        axs[1].grid(True, ls=':')
        
        colors = {"Arcilla": "#D7BDE2", "Arena": "#F9E79F", "Roca": "#AED6F1", "Relleno": "#D3D3D3"}
        for _, r in df_geo.iterrows():
            rect = patches.Rectangle((0, r['z_ini']), 1, r['Espesor_m'], fc=colors.get(r['Tipo'], 'white'), ec='k')
            axs[2].add_patch(rect)
            axs[2].text(0.5, (r['z_ini']+r['z_fin'])/2, r['Tipo'], ha='center', va='center', rotation=90)
        axs[2].set_xlim(0, 1); axs[2].set_title("Estratigrafía"); axs[2].axis('off')
        
        plt.ylim(z_max, 0)
        return fig

# ==============================================================================
# MÓDULO 4: MOTOR DE CÁLCULO (CalcEngine)
# ==============================================================================
class CalcEngine:
    @staticmethod
    def optimize(carga_ton, fs_req, w_c, df_geo):
        """
        ALGORITMO DE OPTIMIZACIÓN (Lógica Preservada).
        Itera sobre Diámetros, Cantidades y Longitudes.
        """
        # Constantes de Optimización
        DIAMETROS_COM = {0.100: 1.00, 0.115: 0.95, 0.150: 0.90, 0.200: 0.85}
        LISTA_D = sorted(list(DIAMETROS_COM.keys()))
        MIN_MICROS = 3; MAX_MICROS = 15
        RANGO_L = range(5, 36)
        COSTO_BASE = 100
        F_CO2_CEM = 0.90; F_CO2_PERF = 15.0; F_CO2_ACERO = 1.85
        DEN_ACERO = 7850.0; DEN_CEM = 3150.0; FY_KPA = 500000.0
        
        # Preparar capas para velocidad
        layers = df_geo.to_dict('records') 
        carga_kn = carga_ton * 9.81
        results = []
        
        for D in LISTA_D:
            for N in range(MIN_MICROS, MAX_MICROS + 1):
                Q_act_unit = carga_kn / N
                Q_req_geo = Q_act_unit * fs_req
                
                for L in RANGO_L:
                    # 1. Capacidad Geo (Iterar capas)
                    Q_cap = 0; z_curr = 0
                    for l in layers:
                        if z_curr >= L: break
                        z_bot = min(l['z_fin'], L)
                        th = z_bot - z_curr
                        if th > 0: Q_cap += (np.pi * D * th) * l['Alpha_Design']
                        z_curr = z_bot
                    
                    if Q_cap >= Q_req_geo:
                        # Solución factible encontrada
                        fs_real = Q_cap / Q_act_unit
                        
                        # 2. Volumen Grout
                        vol_grout = 0; z_curr = 0
                        for l in layers:
                            if z_curr >= L: break
                            z_bot = min(l['z_fin'], L)
                            th = z_bot - z_curr
                            if th > 0: vol_grout += (np.pi*(D/2)**2 * th) * l['F_Exp']
                            z_curr = z_bot
                        vol_grout *= N
                        
                        # 3. Metricas
                        costo = (L * N * COSTO_BASE) / DIAMETROS_COM[D]
                        peso_acero = (Q_act_unit/FY_KPA) * L * N * DEN_ACERO
                        peso_cem = vol_grout * (1.0 / (w_c/1000 + 1/DEN_CEM))
                        co2 = (peso_acero*F_CO2_ACERO + peso_cem*F_CO2_CEM + L*N*F_CO2_PERF)/1000
                        
                        results.append({
                            "D_mm": int(D*1000), "N": N, "L_m": L, "L_Tot": L*N,
                            "FS": fs_real, "Vol_Grout": vol_grout, "Costo_Idx": int(costo),
                            "CO2_ton": co2, "Q_act": Q_act_unit/9.81
                        })
                        break # Salir de loop L
        return pd.DataFrame(results)

    @staticmethod
    def winkler(L, D, EI, V, M, layers):
        """Método de Diferencias Finitas o Solución Analítica para deflexión."""
        # Usamos solución analítica simplificada asumiendo Kh promedio superior
        kh = layers[0].kh if layers and layers[0].kh > 0 else 5000
        beta = ((kh * D) / (4 * EI))**0.25
        z = np.linspace(0, L, 100)
        y_list, m_list, v_list = [], [], []
        
        for x in z:
            bz = beta * x
            if bz > 10: 
                y_list.append(0); m_list.append(0); v_list.append(0); continue
            
            exp = np.exp(-bz); sin = np.sin(bz); cos = np.cos(bz)
            A = exp*(cos+sin); B = exp*sin; C = exp*(cos-sin); D_f = exp*cos
            
            y = (2*V*beta/(kh*D))*D_f + (2*M*beta**2/(kh*D))*C
            m = (V/beta)*B + M*A
            v = V*C - 2*M*beta*D_f
            
            y_list.append(y); m_list.append(m); v_list.append(v)
            
        return z, np.array(y_list), np.array(m_list), np.array(v_list)

# ==============================================================================
# MÓDULO 5: INTERFAZ DE APLICACIÓN (AppInterface)
# ==============================================================================
class AppInterface:
    def run():
        AuthModule.check_auth()
        if not st.session_state['usuario_registrado']:
            AuthModule.login_form()
            return

        # --- SIDEBAR ---
        with st.sidebar:
            st.success(f"👤 **{st.session_state['datos_usuario'].get('nombre')}**")
            if st.button("Generar Reporte PDF"):
                st.info("Función de reporte lista para integrar con ReportModule.")
            if st.button("Cerrar Sesión"):
                st.session_state['usuario_registrado'] = False
                st.rerun()

        st.title("🏗️ GeoMicropile Suite")
        t1, t2, t3, t4 = st.tabs(["🌍 1. Geotecnia", "🚀 2. Optimizador", "📐 3. Diseño Detallado", "🧱 4. Zapatas/Caissons"])

        # --- TAB 1: GEOTECNIA ---
        with t1:
            c1, c2 = st.columns([1.3, 1])
            with c1:
                st.subheader("Entrada de Datos")
                edited_df = st.data_editor(
                    DatabaseModule.get_soil_defaults(),
                    column_config={
                        "Tipo": st.column_config.SelectboxColumn(options=["Arcilla", "Arena", "Roca", "Relleno"]),
                        "F_Exp": st.column_config.NumberColumn("Factor Expansión", min_value=1.0, max_value=2.0, step=0.1)
                    },
                    num_rows="dynamic", use_container_width=True
                )
                df_geo = GeotechModule.process_input(edited_df)
                st.session_state['df_geo'] = df_geo # Persistencia
                
                # Crear objetos Layer para calculos posteriores
                layers_objs = [Layer(r['z_ini'], r['z_fin'], r['Tipo'], r['Alpha_Design'], r['Kh_kNm3'], r['F_Exp']) for _, r in df_geo.iterrows()]
                st.session_state['layers_objs'] = layers_objs

                st.markdown("##### Parámetros Procesados")
                st.dataframe(df_geo[["z_ini", "z_fin", "Tipo", "Alpha_Design", "Kh_kNm3"]].style.format("{:.1f}"), use_container_width=True)

            with c2:
                if not df_geo.empty:
                    st.pyplot(GeotechModule.plot_stratigraphy(df_geo))

        # --- TAB 2: OPTIMIZACIÓN ---
        with t2:
            st.subheader("Algoritmo de Optimización Multi-Variable")
            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                Q_total = st.number_input("Carga Total Grupo (Ton)", 50.0, 5000.0, 150.0)
                FS_obj = st.number_input("FS Geotécnico Objetivo", 1.5, 3.0, 2.0)
            with col_opt2:
                wc_ratio = st.slider("Relación A/C Lechada", 0.4, 0.6, 0.5)
            
            if st.button("EJECUTAR OPTIMIZACIÓN"):
                if 'df_geo' in st.session_state and not st.session_state['df_geo'].empty:
                    with st.spinner("Iterando combinaciones..."):
                        res_opt = CalcEngine.optimize(Q_total, FS_obj, wc_ratio, st.session_state['df_geo'])
                        
                        if not res_opt.empty:
                            best = res_opt.sort_values("Costo_Idx").iloc[0]
                            st.session_state['opt_result'] = best.to_dict() # Guardar mejor opción
                            
                            st.success("✅ Diseño Óptimo Encontrado")
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Configuración", f"{int(best['N'])} x Ø{int(best['D_mm'])}mm")
                            m2.metric("Longitud", f"{int(best['L_m'])} m")
                            m3.metric("Huella CO2", f"{best['CO2_ton']:.2f} Ton")
                            
                            st.dataframe(res_opt.sort_values("Costo_Idx").head(10).style.background_gradient(subset=['Costo_Idx'], cmap='Greens_r'), use_container_width=True)
                        else:
                            st.error("No se encontraron soluciones factibles.")
                else:
                    st.warning("Defina el suelo en la Pestaña 1 primero.")

        # --- TAB 3: DISEÑO DETALLADO ---
        with t3:
            st.subheader("Verificación Detallada (Micropilote Individual)")
            
            # Cargar valores optimos o defaults
            opt = st.session_state.get('opt_result', {})
            def_L = float(opt.get('L_m', 12.0)); def_D = float(opt.get('D_mm', 200.0))/1000
            
            c_in, c_out = st.columns([1, 1.5])
            with c_in:
                L = st.number_input("Longitud (m)", 1.0, 50.0, def_L)
                D = st.number_input("Diámetro (m)", 0.1, 0.6, def_D)
                
                db = DatabaseModule.get_dywidag()
                sys = st.selectbox("Refuerzo (Dywidag)", db['Sistema'], index=2)
                row_s = db[db['Sistema'] == sys].iloc[0]
                st.caption(f"Area: {row_s['As_mm2']} mm2 | Fy: {row_s['fy_MPa']} MPa")
                
                fc = st.number_input("f'c Grout (MPa)", 20.0, 50.0, 30.0)
                Pu = st.number_input("Compresión (kN)", value=float(opt.get('Q_act', 50)*9.81) if opt else 500.0)
                Vu = st.number_input("Cortante (kN)", value=30.0)
                Mu = st.number_input("Momento (kNm)", value=15.0)

            with c_out:
                if 'layers_objs' in st.session_state:
                    layers = st.session_state['layers_objs']
                    
                    # 1. Capacidad Axial (Manual Integration for plotting)
                    z_ax = np.linspace(0, L, 100)
                    q_ult = []; curr = 0; perim = np.pi*D
                    for z in z_ax:
                        alpha = 0
                        if z > 0:
                            for l in layers:
                                if l.contains(z): alpha = l.alpha; break
                            if layers and z > layers[-1].z_bot: alpha = layers[-1].alpha
                        curr += alpha * perim * (L/100)
                        q_ult.append(curr)
                    q_adm = np.array(q_ult)/2.0 # FS=2 para visualización
                    
                    # 2. Estructural & Lateral
                    As = row_s['As_mm2']; fy = row_s['fy_MPa']
                    Ag = (np.pi*(D*1000/2)**2) - As
                    P_est = (0.4*fc*Ag + 0.47*fy*As)/1000
                    
                    I_bar = (np.pi*(row_s['D_ext_mm']/1000)**4)/64
                    EI = 200e6 * I_bar + (4700*np.sqrt(fc)*1000 * ((np.pi*D**4)/64 - I_bar))
                    
                    z_lat, y_lat, m_lat, v_lat = CalcEngine.winkler(L, D, EI, Vu, Mu, layers)
                    
                    # Resultados
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Q Geo Adm", f"{q_adm[-1]:.1f} kN")
                    k2.metric("P Est Comp", f"{P_est:.1f} kN")
                    k3.metric("Deflexión", f"{max(abs(y_lat))*1000:.1f} mm")
                    
                    # --- GRÁFICAS RECUPERADAS ---
                    fig_res, (ax_geo, ax_lat) = plt.subplots(1, 2, figsize=(12, 6))
                    
                    # Grafica Axial + Suelo
                    max_q = max(q_ult)*1.2
                    cols = {"Arcilla": "#D7BDE2", "Arena": "#F9E79F", "Roca": "#AED6F1"}
                    for l in layers:
                        rect = patches.Rectangle((0, l.z_top), max_q, l.z_bot-l.z_top, fc=cols.get(l.tipo,'white'), alpha=0.3)
                        ax_geo.add_patch(rect)
                        ax_geo.text(max_q*0.05, (l.z_top+l.z_bot)/2, f"{l.tipo}\n$\\alpha$={l.alpha:.0f}")
                    
                    ax_geo.plot(q_adm, z_ax, 'b-', label='Q Adm')
                    ax_geo.plot(q_ult, z_ax, 'k--', label='Q Ult')
                    ax_geo.axvline(Pu, c='r', ls=':', label='Pu')
                    # Dibujo Micropilote
                    rect_mp = patches.Rectangle((max_q*0.8, 0), max_q*0.05, L, fc='gray', ec='k')
                    ax_geo.add_patch(rect_mp)
                    ax_geo.invert_yaxis(); ax_geo.legend(); ax_geo.set_title("Capacidad Axial")
                    
                    # Grafica Lateral
                    ax_lat.plot(m_lat, z_lat, 'g-', label='Momento (kNm)')
                    ax_lat2 = ax_lat.twiny()
                    ax_lat2.plot(y_lat*1000, z_lat, 'm--', label='Deflexión (mm)')
                    ax_lat.invert_yaxis(); ax_lat.grid(True); ax_lat.legend(loc='upper left'); ax_lat2.legend(loc='upper right')
                    
                    st.pyplot(fig_res)
                    
                    # Ecuaciones
                    with st.expander("Ver Ecuaciones"):
                        st.latex(r"Q_{ult} = \sum \alpha_{bond} \cdot \pi D \cdot \Delta L")
                        st.latex(r"EI \frac{d^4y}{dz^4} + k_h D y = 0")

        # --- TAB 4: ZAPATAS (Placeholder para Modularidad) ---
        with t4:
            st.info("Módulo de Zapatas y Caissons listo para implementación futura.")
            st.metric("Capacidad Portante", "Pendiente")

# Ejecutar
if __name__ == "__main__":
    AppInterface.run()
