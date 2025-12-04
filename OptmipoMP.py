import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass
import requests
import time
from io import BytesIO

# --- IMPORTACIONES PARA PDF ---
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except ImportError:
    st.error("⚠️ Falta la librería 'reportlab'. Agrégala a requirements.txt")

# ==============================================================================
# CONFIGURACIÓN GLOBAL
# ==============================================================================
st.set_page_config(page_title="GeoStructure Pro", layout="wide", page_icon="🏗️")

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] { background-color: #FFFFFF; border-bottom: 2px solid #FF4B4B; }
    h1, h2, h3 { color: #2C3E50; }
</style>
""", unsafe_allow_html=True)

GOOGLE_SCRIPT_URL = ""  # Pega tu URL de Google Apps Script aquí

# ==============================================================================
# 1. SISTEMA DE REGISTRO Y ESTADO
# ==============================================================================
if 'usuario_registrado' not in st.session_state: st.session_state['usuario_registrado'] = False
if 'datos_usuario' not in st.session_state: st.session_state['datos_usuario'] = {}

# CORRECCIÓN DEL ERROR: Inicializar como diccionario vacío, no como None
if 'design_optimo' not in st.session_state: st.session_state['design_optimo'] = {} 

def enviar_a_google_sheets(datos):
    if not GOOGLE_SCRIPT_URL: return True
    try:
        requests.post(GOOGLE_SCRIPT_URL, json=datos, timeout=3)
        return True
    except: return False

def mostrar_registro():
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("## 🔒 GeoStructure Suite")
        with st.form("reg"):
            nombre = st.text_input("Nombre")
            empresa = st.text_input("Empresa")
            email = st.text_input("Email")
            if st.form_submit_button("INGRESAR"):
                if nombre and email:
                    st.session_state['usuario_registrado'] = True
                    st.session_state['datos_usuario'] = {'nombre': nombre, 'empresa': empresa}
                    enviar_a_google_sheets({'nombre': nombre, 'email': email, 'fecha': time.strftime("%Y-%m-%d")})
                    st.rerun()
                else: st.error("Complete los datos.")

# ==============================================================================
# 2. BASES DE DATOS Y CLASES CIENTÍFICAS
# ==============================================================================

def get_dywidag_db():
    """
    Catálogo Dywidag Systems International (DSI) - SOLO DYWIDAG.
    """
    data = {
        "Sistema": [
            "R25-210", "R32-280", "R32-360", "R38-500", "R38-550", 
            "R51-660", "R51-800", "T76-1200", "T76-1600", "T76-1900"
        ],
        "D_ext_mm": [
            25, 32, 32, 38, 38, 
            51, 51, 76, 76, 76
        ],
        "As_mm2": [
            260, 410, 510, 750, 800, 
            970, 1150, 1610, 1990, 2360
        ],
        "fy_MPa": [
            500, 535, 550, 533, 560, 
            555, 556, 620, 600, 635
        ]
    }
    return pd.DataFrame(data)

def procesar_geotecnia(df_input):
    results = []
    z_acum = 0
    for _, row in df_input.iterrows():
        try:
            esp = float(row.get('Espesor_m', 0)); tipo = row.get('Tipo', 'Arcilla')
            n_spt = float(row.get('N_SPT', 0)); su = float(row.get('Su_kPa', 0))
            kh = float(row.get('Kh_kNm3', 0)); a_manual = float(row.get('Alpha_Manual', 0))
        except: continue
        
        phi = 0; E_MPa = 0; alpha = 0
        if tipo == "Arena":
            phi = ( (np.sqrt(20*n_spt)+20) + (27.1+0.3*n_spt-0.00054*n_spt**2) ) / 2
            E_MPa = 1.0 * n_spt
            alpha = min(3.8 * n_spt, 250)
        elif tipo == "Arcilla":
            E_MPa = 0.3 * su
            if su < 25: alpha = 20
            elif su < 50: alpha = 40
            elif su < 100: alpha = 70
            else: alpha = 100
        else: # Roca
            E_MPa = 5000; alpha = 300
            
        a_final = a_manual if a_manual > 0 else alpha
        z_fin = z_acum + esp
        
        results.append({
            "z_ini": z_acum, "z_fin": z_fin, "Espesor_m": esp, "Tipo": tipo,
            "N_SPT": n_spt, "Su_kPa": su, "Kh_kNm3": kh, "Alpha_Design": a_final,
            "Phi_Deg": phi, "E_MPa": E_MPa, "Gamma_kN": 18.0
        })
        z_acum = z_fin
    return pd.DataFrame(results)

@dataclass
class SoilLayerObj:
    z_top: float; z_bot: float; alpha: float; kh: float; phi: float; su: float; gamma: float
    def contains(self, z): return self.z_top <= z <= self.z_bot

class GeotechEngine:
    def __init__(self, layers):
        self.layers = layers

    def get_prop(self, z, prop_name):
        if z < 0: return 0
        for l in self.layers:
            if l.contains(z): return getattr(l, prop_name)
        if self.layers and z > self.layers[-1].z_bot: return getattr(self.layers[-1], prop_name)
        return 0

    def calc_micropile_axial(self, L, D, fs):
        perim = np.pi * D
        z_arr = np.linspace(0, L, 100)
        q_ult = []; curr = 0
        for z in z_arr:
            alpha = self.get_prop(z, 'alpha')
            if z > 0: curr += alpha * perim * (L/100)
            q_ult.append(curr)
        return z_arr, np.array(q_ult), np.array(q_ult)/fs

    def calc_winkler(self, L, D, EI, V, M):
        kh_sup = self.get_prop(1.0, 'kh') 
        if kh_sup <= 0: kh_sup = 1000
        
        beta = ((kh_sup * D) / (4 * EI))**0.25
        z = np.linspace(0, L, 200)
        y, m_res, v_res = [], [], []
        
        for x in z:
            bz = beta * x
            if bz > 10: 
                y.append(0); m_res.append(0); v_res.append(0); continue
            
            exp = np.exp(-bz); sin = np.sin(bz); cos = np.cos(bz)
            A = exp*(cos+sin); B = exp*sin; C = exp*(cos-sin); D_fact = exp*cos
            
            y_val = (2*V*beta/(kh_sup*D))*D_fact + (2*M*beta**2/(kh_sup*D))*C
            m_val = (V/beta)*B + M*A
            v_val = V*C - 2*M*beta*D_fact
            
            y.append(y_val); m_res.append(m_val); v_res.append(v_val)
            
        return z, np.array(y), np.array(m_res), np.array(v_res)

    def calc_zapata(self, B, L, Df, carga_v):
        phi = self.get_prop(Df + B/2, 'phi')
        su = self.get_prop(Df + B/2, 'su')
        gamma = self.get_prop(Df, 'gamma')
        
        if phi < 1: phi = 1
        rad = np.radians(phi)
        Nq = np.exp(np.pi * np.tan(rad)) * (np.tan(np.radians(45) + rad/2))**2
        Nc = (Nq - 1) / np.tan(rad) if phi > 0 else 5.14
        Ny = 2 * (Nq + 1) * np.tan(rad)
        
        q = Df * gamma
        if su > 0: qu = 1.3 * su * Nc + q * Nq 
        else: qu = q * Nq + 0.4 * gamma * B * Ny
            
        return qu, qu/(carga_v/(B*L))

# ==============================================================================
# 3. INTERFAZ PRINCIPAL
# ==============================================================================
def app_principal():
    with st.sidebar:
        st.success(f"Ingeniero: **{st.session_state['datos_usuario'].get('nombre')}**")
        
        if st.button("📄 Generar Reporte PDF"):
            try:
                buffer = BytesIO()
                p = canvas.Canvas(buffer, pagesize=letter)
                p.drawString(100, 750, "REPORTE DE CÁLCULO GEOTÉCNICO - GeoStructure Pro")
                p.drawString(100, 730, f"Usuario: {st.session_state['datos_usuario'].get('nombre')}")
                p.drawString(100, 710, f"Empresa: {st.session_state['datos_usuario'].get('empresa')}")
                p.drawString(100, 680, "Resumen de Resultados:")
                
                # Obtener de forma segura (con valor por defecto {})
                opt = st.session_state.get('design_optimo') or {}
                if opt:
                    p.drawString(100, 660, f"Configuración Óptima: {opt.get('N')} x D={opt.get('D_mm')}mm")
                    p.drawString(100, 640, f"Longitud: {opt.get('L_m')} m | CO2 Estimado: {opt.get('CO2_Ton'):.1f} Ton")
                else:
                    p.drawString(100, 660, "No se ha ejecutado optimización aún.")
                    
                p.showPage()
                p.save()
                st.download_button("Descargar PDF", buffer, "Reporte_Calculo.pdf", "application/pdf")
            except Exception as e:
                st.error(f"Error generando PDF: {e}")

        if st.button("Salir"):
            st.session_state['usuario_registrado'] = False; st.rerun()

    st.title("🏗️ Sistema Avanzado de Diseño de Cimentaciones")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌍 1. Geotecnia", 
        "🚀 2. Optimización Micropilotes", 
        "📐 3. Diseño Detallado MP", 
        "🧱 4. Zapatas & Caissons"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: GEOTECNIA
    # --------------------------------------------------------------------------
    with tab1:
        c1, c2 = st.columns([1.5, 1])
        with c1:
            st.subheader("Caracterización del Subsuelo")
            df_template = pd.DataFrame([
                {"Espesor_m": 4.0, "Tipo": "Arcilla", "N_SPT": 5, "Su_kPa": 40, "Kh_kNm3": 8000, "Alpha_Manual": 0.0},
                {"Espesor_m": 8.0, "Tipo": "Arena", "N_SPT": 25, "Su_kPa": 0, "Kh_kNm3": 25000, "Alpha_Manual": 0.0},
                {"Espesor_m": 6.0, "Tipo": "Roca", "N_SPT": 50, "Su_kPa": 0, "Kh_kNm3": 90000, "Alpha_Manual": 400.0},
            ])
            edited_df = st.data_editor(
                df_template, 
                column_config={"Tipo": st.column_config.SelectboxColumn(options=["Arcilla", "Arena", "Roca", "Relleno"])},
                num_rows="dynamic", use_container_width=True
            )
            df_geo = procesar_geotecnia(edited_df)
            
            # Formato seguro
            format_cols = {col: "{:.1f}" for col in df_geo.columns if df_geo[col].dtype in ['float64', 'int64']}
            st.dataframe(df_geo.style.format(format_cols), use_container_width=True)
            
            layers_objs = []
            for _, r in df_geo.iterrows():
                layers_objs.append(SoilLayerObj(r['z_ini'], r['z_fin'], r['Alpha_Design'], r['Kh_kNm3'], r['Phi_Deg'], r.get('Su_kPa',0), r['Gamma_kN']))
            
            st.session_state['layers_objs'] = layers_objs

        with c2:
            if not df_geo.empty:
                fig, axs = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
                z_plot, n_plot, a_plot = [], [], []
                
                for _, r in df_geo.iterrows():
                    z_plot.extend([r['z_ini'], r['z_fin']])
                    n_plot.extend([r['N_SPT'], r['N_SPT']])
                    a_plot.extend([r['Alpha_Design'], r['Alpha_Design']])
                
                axs[0].plot(n_plot, z_plot, 'b'); axs[0].set_title("N-SPT"); axs[0].invert_yaxis(); axs[0].grid(True, ls=':')
                axs[1].plot(a_plot, z_plot, 'r'); axs[1].set_title("Alpha (kPa)"); axs[1].grid(True, ls=':')
                
                colors = {"Arcilla": "#D7BDE2", "Arena": "#F9E79F", "Roca": "#AED6F1"}
                for _, r in df_geo.iterrows():
                    rect = patches.Rectangle((0, r['z_ini']), 1, r['Espesor_m'], facecolor=colors.get(r['Tipo'], 'white'), ec='k')
                    axs[2].add_patch(rect)
                    axs[2].text(0.5, (r['z_ini']+r['z_fin'])/2, r['Tipo'], ha='center', va='center', rotation=90)
                axs[2].set_xlim(0, 1); axs[2].set_title("Perfil"); axs[2].axis('off')
                st.pyplot(fig)

    # --------------------------------------------------------------------------
    # TAB 2: OPTIMIZACIÓN
    # --------------------------------------------------------------------------
    with tab2:
        st.subheader("🚀 Búsqueda de Solución Óptima")
        c_opt1, c_opt2 = st.columns(2)
        with c_opt1:
            Carga_Total = st.number_input("Carga Total Grupo (Ton)", 50.0, 1000.0, 150.0)
            FS_target = st.number_input("FS Objetivo", 1.5, 3.0, 2.0)
        
        if st.button("Correr Algoritmo de Optimización"):
            if not st.session_state.get('layers_objs'):
                st.error("Defina la estratigrafía en Tab 1 primero.")
            else:
                Carga_kN = Carga_Total * 9.81
                # Solo Dywidag Comunes para optimizar rapido
                DIAMETROS_COM = {0.100: 1.0, 0.150: 0.9, 0.200: 0.85, 0.250: 0.8}
                resultados = []
                engine = GeotechEngine(st.session_state['layers_objs'])
                
                progress = st.progress(0)
                for i, D in enumerate(DIAMETROS_COM):
                    progress.progress((i+1)/4)
                    for N in range(3, 16):
                        Q_indiv_req = (Carga_kN / N) * FS_target
                        for L in range(6, 31):
                            _, _, q_adm_arr = engine.calc_micropile_axial(L, D, 1.0)
                            if len(q_adm_arr) > 0:
                                Q_ult_cap = q_adm_arr[-1]
                                if Q_ult_cap >= Q_indiv_req:
                                    vol_grout = (np.pi*(D/2)**2 * L * 1.2) * N
                                    kg_acero = (np.pi*(0.04/2)**2 * 7850 * L * N)
                                    costo = (L * N * 100) / DIAMETROS_COM[D]
                                    co2 = (kg_acero*1.85 + vol_grout*1000*0.9 + L*N*15)/1000
                                    
                                    resultados.append({
                                        "D_mm": int(D*1000), "N": N, "L_m": L,
                                        "Q_ult_geo": Q_ult_cap, "FS_Real": Q_ult_cap / (Carga_kN/N),
                                        "Costo_Idx": int(costo), "CO2_Ton": co2
                                    })
                                    break 
                progress.empty()
                
                if resultados:
                    df_opt = pd.DataFrame(resultados).sort_values("Costo_Idx")
                    best = df_opt.iloc[0]
                    st.session_state['design_optimo'] = best.to_dict()
                    
                    st.success("✅ Optimización Finalizada")
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Mejor Configuración", f"{best['N']} x Ø{best['D_mm']}mm")
                    k2.metric("Longitud", f"{best['L_m']} m")
                    k3.metric("Huella CO2", f"{best['CO2_Ton']:.1f} Ton")
                    st.dataframe(df_opt.head(5).style.background_gradient(subset=['Costo_Idx'], cmap='Greens_r'), use_container_width=True)
                else:
                    st.error("No se encontraron soluciones factibles.")

    # --------------------------------------------------------------------------
    # TAB 3: DISEÑO DETALLADO
    # --------------------------------------------------------------------------
    with tab3:
        st.subheader("📐 Verificación Detallada (Diseño Seleccionado)")
        
        # Recuperación SEGURA de datos (Fix del error AttributeError)
        opt = st.session_state.get('design_optimo') or {} 
        def_L = float(opt.get('L_m', 12.0)); def_D = float(opt.get('D_mm', 200.0))/1000
        
        c_in, c_out = st.columns([1, 1.5])
        with c_in:
            L = st.number_input("Longitud (m)", 1.0, 50.0, def_L)
            D = st.number_input("Diámetro (m)", 0.1, 0.6, def_D)
            
            db = get_dywidag_db()
            sys = st.selectbox("Sistema Refuerzo", db['Sistema'], index=3)
            row_s = db[db['Sistema'] == sys].iloc[0]
            st.caption(f"Area: {row_s['As_mm2']} mm2 | Fy: {row_s['fy_MPa']} MPa")
            
            fc = st.number_input("f'c Grout (MPa)", 20.0, 50.0, 30.0)
            P_u = st.number_input("Compresión (kN)", value=500.0)
            V_u = st.number_input("Cortante (kN)", value=30.0)
            M_u = st.number_input("Momento (kNm)", value=15.0)
            
        with c_out:
            if not st.session_state.get('layers_objs'):
                st.warning("Vaya a la Pestaña 1 y defina el suelo primero.")
            else:
                engine = GeotechEngine(st.session_state['layers_objs'])
                z_ax, q_ult, q_adm = engine.calc_micropile_axial(L, D, 2.0)
                
                As = row_s['As_mm2']; fy = row_s['fy_MPa']
                A_g = (np.pi*(D*1000/2)**2) - As
                P_est_comp = (0.40*fc*A_g + 0.47*fy*As)/1000
                
                I_b = (np.pi*(row_s['D_ext_mm']/1000)**4)/64
                EI = 200e6 * I_b + (4700*np.sqrt(fc)*1000 * ((np.pi*D**4)/64 - I_b))
                z_lat, y_lat, m_lat, v_lat = engine.calc_winkler(L, D, EI, V_u, M_u)
                
                k1, k2, k3 = st.columns(3)
                q_geo_val = q_adm[-1] if len(q_adm)>0 else 0
                k1.metric("Q Admisible Geo.", f"{q_geo_val:.1f} kN")
                k2.metric("P Est. Compresión", f"{P_est_comp:.1f} kN")
                k3.metric("Deflexión Máx", f"{max(abs(y_lat))*1000:.1f} mm")
                
                tab_g1, tab_g2 = st.tabs(["Axial", "Lateral (Winkler)"])
                with tab_g1:
                    fig_ax, ax = plt.subplots(figsize=(8,4))
                    ax.plot(q_adm, z_ax, label="Q Adm"); ax.plot(q_ult, z_ax, '--', label="Q Ult")
                    ax.axvline(P_u, c='r', ls=':', label="Pu")
                    ax.invert_yaxis(); ax.legend(); ax.grid(True, ls=':')
                    st.pyplot(fig_ax)
                with tab_g2:
                    fig_lat, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
                    ax1.plot(y_lat*1000, z_lat, 'm'); ax1.set_title("Deflexión (mm)"); ax1.invert_yaxis(); ax1.grid(True, ls=':')
                    ax2.plot(m_lat, z_lat, 'g'); ax2.set_title("Momento (kNm)"); ax2.grid(True, ls=':')
                    ax3.plot(v_lat, z_lat, 'b'); ax3.set_title("Cortante (kN)"); ax3.grid(True, ls=':')
                    st.pyplot(fig_lat)

    # --------------------------------------------------------------------------
    # TAB 4: ZAPATAS & CAISSONS
    # --------------------------------------------------------------------------
    with tab4:
        st.subheader("Diseño de Losa de Cabezal & Punzonamiento")
        
        c1, c2 = st.columns(2)
        with c1:
            Bx = st.number_input("Ancho X (m)", 1.0, 20.0, 5.0)
            By = st.number_input("Ancho Y (m)", 1.0, 20.0, 5.0)
            H = st.number_input("Espesor (m)", 0.3, 2.0, 0.6)
            q_sup = st.number_input("Sobrecarga (kPa)", 0.0, 100.0, 20.0)
        with c2:
            fcl = st.number_input("f'c Losa (MPa)", 21.0, 42.0, 28.0)
            
            Q_unit = 500 # Default
            if 'q_geo_val' in locals() and q_geo_val > 0:
                Q_unit = q_geo_val
            
            Q_tot = (q_sup + H*24)*Bx*By
            N = int(np.ceil(Q_tot/Q_unit)) if Q_unit > 0 else 0
            st.metric("Carga Total", f"{Q_tot:.1f} kN")
            st.metric("Micropilotes Req.", N)
            
            d_eff = H - 0.075
            bo = np.pi*(0.2 + d_eff)
            vc = 0.75 * 0.33 * np.sqrt(fcl) * bo * d_eff * 1000
            pu = Q_unit * 1.4 
            
            ratio = pu / vc if vc > 0 else 999
            
            st.write(f"**Carga Pu:** {pu:.1f} kN | **Capacidad phi*Vc:** {vc:.1f} kN")
            
            if ratio < 1.0:
                st.success(f"✅ PASA Punzonamiento (Ratio: {ratio:.2f})")
            else:
                st.error(f"❌ FALLA Punzonamiento (Ratio: {ratio:.2f}) - Aumente espesor")

# ==============================================================================
# CONTROL DE FLUJO
# ==============================================================================
if st.session_state['usuario_registrado']:
    app_principal()
else:
    mostrar_registro()
