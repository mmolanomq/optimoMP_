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
# 1. SISTEMA DE REGISTRO
# ==============================================================================
if 'usuario_registrado' not in st.session_state: st.session_state['usuario_registrado'] = False
if 'datos_usuario' not in st.session_state: st.session_state['datos_usuario'] = {}
if 'design_optimo' not in st.session_state: st.session_state['design_optimo'] = {}
if 'layers_objs' not in st.session_state: st.session_state['layers_objs'] = []

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
        st.info("Acceso para Ingenieros y Especialistas.")
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
# 2. MOTOR CIENTÍFICO (BASE DE DATOS Y CLASES)
# ==============================================================================

def get_dywidag_db():
    """
    Catálogo Dywidag Systems International (DSI) - SOLO DYWIDAG.
    """
    data = {
        "Sistema": [
            "R32-280", "R32-360", "R38-500", "R38-550", 
            "R51-660", "R51-800", "T76-1200", "T76-1600", "T76-1900"
        ],
        "D_ext_mm": [
            32, 32, 38, 38, 
            51, 51, 76, 76, 76
        ],
        "As_mm2": [
            410, 510, 750, 800, 
            970, 1150, 1610, 1990, 2360
        ],
        "fy_MPa": [
            535, 550, 533, 560, 
            555, 556, 620, 600, 635
        ]
    }
    return pd.DataFrame(data)

@dataclass
class SoilLayerObj:
    z_top: float; z_bot: float; tipo: str; alpha: float; kh: float; phi: float; su: float; gamma: float
    def contains(self, z): return self.z_top <= z <= self.z_bot

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
        # Correlaciones Básicas
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
            "Phi_Deg": phi, "E_MPa": E_MPa, "Gamma_kN": 18.0,
            "f_exp": 1.2 # Factor expansion por defecto para optimizacion
        })
        z_acum = z_fin
    return pd.DataFrame(results)

def calc_winkler(L, D, EI, V, M, layers):
    # Obtener Kh promedio superficial (o iterar por capas para mayor precision)
    # Aqui usamos un Kh constante representativo de la zona superior (critica para flexion)
    if not layers: return np.array([]), np.array([]), np.array([]), np.array([]), 0
    kh = layers[0].kh 
    if kh <= 0: kh = 1000
    
    beta = ((kh * D) / (4 * EI))**0.25
    z = np.linspace(0, L, 200)
    y, m_res, v_res = [], [], []
    
    for x in z:
        bz = beta * x
        if bz > 15: 
            y.append(0); m_res.append(0); v_res.append(0); continue
        
        exp = np.exp(-bz); sin = np.sin(bz); cos = np.cos(bz)
        A = exp*(cos+sin); B = exp*sin; C = exp*(cos-sin); D_fact = exp*cos
        
        y_val = (2*V*beta/(kh*D))*D_fact + (2*M*beta**2/(kh*D))*C
        m_val = (V/beta)*B + M*A
        v_val = V*C - 2*M*beta*D_fact
        
        y.append(y_val); m_res.append(m_val); v_res.append(v_val)
        
    return z, np.array(y), np.array(m_res), np.array(v_res), beta

# ==============================================================================
# 3. INTERFAZ PRINCIPAL
# ==============================================================================
def app_principal():
    with st.sidebar:
        st.success(f"Ingeniero: **{st.session_state['datos_usuario'].get('nombre')}**")
        
        if st.button("📄 Reporte PDF"):
            try:
                buffer = BytesIO()
                p = canvas.Canvas(buffer, pagesize=letter)
                p.drawString(100, 750, "REPORTE DE CÁLCULO - GeoStructure Pro")
                p.drawString(100, 730, f"Usuario: {st.session_state['datos_usuario'].get('nombre')}")
                opt = st.session_state.get('design_optimo') or {}
                if opt:
                    p.drawString(100, 680, f"Diseño Óptimo: {opt.get('N')} micros x D={opt.get('D_mm')}mm")
                    p.drawString(100, 660, f"Longitud: {opt.get('L_m')} m | CO2: {opt.get('CO2_ton'):.1f} Ton")
                p.showPage(); p.save()
                st.download_button("Descargar", buffer, "Reporte.pdf", "application/pdf")
            except Exception as e: st.error(f"Error PDF: {e}")

        if st.button("Salir"):
            st.session_state['usuario_registrado'] = False; st.rerun()

    st.title("🏗️ Sistema Avanzado de Diseño de Cimentaciones")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌍 1. Geotecnia", 
        "🚀 2. Optimización", 
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
            
            # Formato
            fmt_cols = {c: "{:.1f}" for c in df_geo.select_dtypes(include='number').columns}
            st.dataframe(df_geo.style.format(fmt_cols), use_container_width=True)
            
            # Guardar Objetos para uso global
            layers_objs = []
            for _, r in df_geo.iterrows():
                layers_objs.append(SoilLayerObj(r['z_ini'], r['z_fin'], r['Tipo'], r['Alpha_Design'], r['Kh_kNm3'], r['Phi_Deg'], r.get('Su_kPa',0), r['Gamma_kN']))
            st.session_state['layers_objs'] = layers_objs
            st.session_state['df_geo'] = df_geo

        with c2:
            if not df_geo.empty:
                # Graficas Perfil
                fig, axs = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
                z_plot, n_plot, a_plot = [], [], []
                for _, r in df_geo.iterrows():
                    z_plot.extend([r['z_ini'], r['z_fin']])
                    n_plot.extend([r['N_SPT'], r['N_SPT']])
                    a_plot.extend([r['Alpha_Design'], r['Alpha_Design']])
                
                axs[0].plot(n_plot, z_plot, 'b'); axs[0].set_title("N-SPT"); axs[0].invert_yaxis(); axs[0].grid(True, ls=':')
                axs[1].plot(a_plot, z_plot, 'r'); axs[1].set_title("Alpha (kPa)"); axs[1].grid(True, ls=':')
                
                cols = {"Arcilla": "#D7BDE2", "Arena": "#F9E79F", "Roca": "#AED6F1"}
                for _, r in df_geo.iterrows():
                    rect = patches.Rectangle((0, r['z_ini']), 1, r['Espesor_m'], facecolor=cols.get(r['Tipo'], 'white'), ec='k')
                    axs[2].add_patch(rect)
                    axs[2].text(0.5, (r['z_ini']+r['z_fin'])/2, r['Tipo'], ha='center', va='center', rotation=90)
                axs[2].set_xlim(0, 1); axs[2].set_title("Perfil"); axs[2].axis('off')
                st.pyplot(fig)
            
            with st.expander("Ver Ecuaciones de Correlación"):
                st.latex(r"\phi' = \frac{(\sqrt{20 N} + 20) + (27.1 + 0.3N - 0.00054N^2)}{2}")
                st.latex(r"\alpha_{bond} \approx \min(3.8 N, 250) \text{ (Arena)}")

    # --------------------------------------------------------------------------
    # TAB 2: OPTIMIZACIÓN (INTEGRADA Y ADAPTADA)
    # --------------------------------------------------------------------------
    with tab2:
        st.subheader("🚀 Algoritmo de Optimización")
        
        # --- INPUTS OPTIMIZACIÓN ---
        c_opt1, c_opt2 = st.columns(2)
        with c_opt1:
            CARGA_TON = st.number_input("Carga Total Grupo (Ton)", 50.0, 2000.0, 150.0)
            FS_REQ = st.number_input("FS Objetivo", 1.5, 3.0, 2.0)
        with c_opt2:
            RELACION_AGUA_CEMENTO = st.slider("Relación A/C", 0.4, 0.6, 0.5)
        
        if st.button("Ejecutar Optimización"):
            if 'df_geo' not in st.session_state or st.session_state['df_geo'].empty:
                st.error("Primero define la estratigrafía en la Pestaña 1.")
            else:
                # --- ADAPTADOR: DATOS DE TAB 1 A FORMATO OPTIMIZADOR ---
                ESTRATOS = []
                for _, r in st.session_state['df_geo'].iterrows():
                    ESTRATOS.append({
                        "z_fin": r['z_fin'],
                        "qs": r['Alpha_Design'],
                        "f_exp": r['f_exp'],
                        "espesor": r['Espesor_m'] # Usado para calculo visual si fuera necesario
                    })
                
                # --- TU ALGORITMO ORIGINAL (ADAPTADO) ---
                DIAMETROS_COM = {0.100: 1.00, 0.115: 0.95, 0.150: 0.90, 0.200: 0.85}
                LISTA_D = sorted(list(DIAMETROS_COM.keys()))
                MIN_MICROS = 3; MAX_MICROS = 15
                RANGO_L = range(5, 36)
                COSTO_PERF_BASE = 100
                FACTOR_CO2_CEMENTO = 0.90; FACTOR_CO2_PERF = 15.0; FACTOR_CO2_ACERO = 1.85
                DENSIDAD_ACERO = 7850.0; DENSIDAD_CEMENTO = 3150.0
                FY_ACERO_KPA = 500000.0

                CARGA_REQ_KN = CARGA_TON * 9.81
                resultados_raw = []

                with st.spinner('Optimizando...'):
                    for D in LISTA_D:
                        for N in range(MIN_MICROS, MAX_MICROS + 1):
                            Q_act_por_pilote_kn = CARGA_REQ_KN / N
                            Q_req_geotec_por_pilote = Q_act_por_pilote_kn * FS_REQ
                            
                            for L in RANGO_L:
                                # 1. Capacidad Geo
                                Q_ult = 0; z_curr = 0
                                for e in ESTRATOS:
                                    if z_curr >= L: break
                                    z_bot = min(e["z_fin"], L)
                                    thick = z_bot - z_curr
                                    if thick > 0: Q_ult += (np.pi * D * thick) * e["qs"]
                                    z_curr = z_bot
                                
                                if Q_ult >= Q_req_geotec_por_pilote:
                                    FS_calc = Q_ult / Q_act_por_pilote_kn
                                    
                                    # 2. Volúmenes
                                    v_exp_tot = 0; z_curr = 0
                                    for e in ESTRATOS:
                                        if z_curr >= L: break
                                        z_bot = min(e["z_fin"], L)
                                        thick = z_bot - z_curr
                                        if thick > 0: v_exp_tot += (np.pi*(D/2)**2 * thick) * e["f_exp"]
                                        z_curr = z_bot
                                    v_exp_tot *= N
                                    
                                    # 3. Costo & CO2
                                    costo_idx = (L * N * COSTO_PERF_BASE) / DIAMETROS_COM[D]
                                    area_acero_m2 = Q_act_por_pilote_kn / FY_ACERO_KPA
                                    peso_acero = area_acero_m2 * L * N * DENSIDAD_ACERO
                                    
                                    # Peso Cemento (Simplificado)
                                    peso_cem_m3 = 1.0 / (RELACION_AGUA_CEMENTO/1000.0 + 1.0/DENSIDAD_CEMENTO)
                                    peso_cem = v_exp_tot * peso_cem_m3
                                    
                                    co2 = (peso_acero*FACTOR_CO2_ACERO + peso_cem*FACTOR_CO2_CEMENTO + (L*N)*FACTOR_CO2_PERF)/1000
                                    
                                    resultados_raw.append({
                                        "D_mm": int(D*1000), "N": N, "L_m": L, "L_Tot_m": L*N,
                                        "FS": FS_calc, "Vol_Exp": v_exp_tot, "Costo_Idx": int(costo_idx),
                                        "CO2_ton": co2, "Q_adm_geo": Q_ult/FS_REQ/9.81, "Q_act": Q_act_por_pilote_kn/9.81
                                    })
                                    break 

                if resultados_raw:
                    df_res = pd.DataFrame(resultados_raw).sort_values("Costo_Idx")
                    best = df_res.iloc[0]
                    st.session_state['design_optimo'] = best.to_dict() # Guardar para Tab 3
                    
                    st.success("✅ Resultados Encontrados")
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Mejor Config.", f"{best['N']} x Ø{best['D_mm']}mm")
                    k2.metric("Longitud", f"{best['L_m']} m")
                    k3.metric("Huella CO2", f"{best['CO2_ton']:.1f} Ton")
                    
                    st.dataframe(df_res.head(8).style.background_gradient(subset=['Costo_Idx'], cmap='Greens_r'), use_container_width=True)
                else:
                    st.error("No hay soluciones factibles.")

    # --------------------------------------------------------------------------
    # TAB 3: DISEÑO DETALLADO
    # --------------------------------------------------------------------------
    with tab3:
        st.subheader("📐 Diseño Detallado & Verificaciones")
        
        opt = st.session_state.get('design_optimo', {})
        def_L = float(opt.get('L_m', 12.0)); def_D = float(opt.get('D_mm', 200.0))/1000
        
        c_in, c_out = st.columns([1, 1.5])
        with c_in:
            st.markdown("##### Geometría")
            L = st.number_input("Longitud (m)", 1.0, 50.0, def_L)
            D = st.number_input("Diámetro (m)", 0.1, 0.6, def_D)
            
            st.markdown("##### Estructura")
            db = get_dywidag_db()
            sys = st.selectbox("Refuerzo Principal", db['Sistema'], index=2)
            row_s = db[db['Sistema'] == sys].iloc[0]
            st.caption(f"Area: {row_s['As_mm2']} mm2 | Fy: {row_s['fy_MPa']} MPa")
            fc = st.number_input("f'c Grout (MPa)", 20.0, 50.0, 30.0)
            
            st.markdown("##### Cargas (por Micropilote)")
            P_u = st.number_input("Compresión Pu (kN)", value=500.0)
            V_u = st.number_input("Cortante Vu (kN)", value=30.0)
            M_u = st.number_input("Momento Mu (kNm)", value=15.0)
            
        with c_out:
            if not st.session_state.get('layers_objs'):
                st.warning("⚠️ Defina suelo en Tab 1.")
            else:
                # 1. Capacidad Axial
                layers = st.session_state['layers_objs']
                perim = np.pi * D; q_ult_list = []; q_curr = 0; z_ax = np.linspace(0, L, 100)
                
                # Integración manual para graficar
                for z in z_ax:
                    alpha = 0
                    if z > 0:
                        for l in layers:
                            if l.contains(z): alpha = l.alpha; break
                        if z > layers[-1].z_bot: alpha = layers[-1].alpha
                    q_curr += alpha * perim * (L/100)
                    q_ult_list.append(q_curr)
                
                q_ult_arr = np.array(q_ult_list); q_adm_arr = q_ult_arr / 2.0
                
                # 2. Estructural
                As = row_s['As_mm2']; fy = row_s['fy_MPa']
                A_g = (np.pi*(D*1000/2)**2) - As
                P_est = (0.40*fc*A_g + 0.47*fy*As)/1000
                
                # 3. Lateral
                I_b = (np.pi*(row_s['D_ext_mm']/1000)**4)/64
                EI = 200e6 * I_b + (4700*np.sqrt(fc)*1000 * ((np.pi*D**4)/64 - I_b))
                
                z_lat, y_lat, m_lat, v_lat, beta = calc_winkler(L, D, EI, V_u, M_u, layers)
                
                # --- RESULTADOS ---
                k1, k2, k3 = st.columns(3)
                q_geo_val = q_adm_arr[-1] if len(q_adm_arr)>0 else 0
                k1.metric("Q Admisible Geo", f"{q_geo_val:.1f} kN", delta="OK" if q_geo_val>P_u else "FALLA")
                k2.metric("P Estructural", f"{P_est:.1f} kN")
                k3.metric("Deflexión Máx", f"{max(abs(y_lat))*1000:.1f} mm")
                
                # --- GRÁFICAS DETALLADAS ---
                st.markdown("#### Comportamiento del Micropilote")
                
                # Grafica Axial en Contexto
                fig_ax, (ax_geo, ax_lat) = plt.subplots(1, 2, figsize=(10, 6))
                
                # Columna de Suelo y Carga Axial
                max_q = max(q_ult_arr)*1.2 if len(q_ult_arr)>0 else 100
                cols = {"Arcilla": "#D7BDE2", "Arena": "#F9E79F", "Roca": "#AED6F1"}
                for l in layers:
                    rect = patches.Rectangle((0, l.z_top), max_q, l.z_bot-l.z_top, fc=cols.get(l.tipo,'white'), alpha=0.3)
                    ax_geo.add_patch(rect)
                    ax_geo.text(max_q*0.05, (l.z_top+l.z_bot)/2, f"{l.tipo}\n$\\alpha$={l.alpha:.0f}", fontsize=8)
                
                ax_geo.plot(q_adm_arr, z_ax, 'b-', label='Q Adm')
                ax_geo.plot(q_ult_arr, z_ax, 'k--', label='Q Ult')
                ax_geo.axvline(P_u, c='r', ls=':', label='Pu')
                
                # Dibujo esquemático del micropilote
                rect_mp = patches.Rectangle((max_q*0.8, 0), max_q*0.05, L, fc='gray', ec='k')
                ax_geo.add_patch(rect_mp)
                
                ax_geo.invert_yaxis(); ax_geo.legend(); ax_geo.set_title("Capacidad Axial"); ax_geo.grid(True, ls=':')
                
                # Grafica Lateral
                ax_lat.plot(m_lat, z_lat, 'g-', label='Momento')
                ax_lat.set_xlabel("Momento (kNm)")
                ax_lat2 = ax_lat.twiny()
                ax_lat2.plot(y_lat*1000, z_lat, 'm--', label='Deflexión')
                ax_lat2.set_xlabel("Deflexión (mm)")
                
                ax_lat.invert_yaxis(); ax_lat.grid(True, ls=':'); ax_lat.set_title("Lateral")
                
                st.pyplot(fig_ax)
                
                # Ecuaciones
                with st.expander("Ver Ecuaciones Utilizadas"):
                    st.latex(r"Q_{all} = \frac{\sum \alpha \cdot \pi D \cdot \Delta L}{FS}")
                    st.latex(r"M(z), y(z) \rightarrow \text{Winkler Model (Beam on Elastic Foundation)}")

    # --------------------------------------------------------------------------
    # TAB 4: ZAPATAS & CAISSONS
    # --------------------------------------------------------------------------
    with tab4:
        st.subheader("Diseño de Cimentación Superficial/Semiprofunda")
        tipo = st.radio("Tipo:", ["Zapata", "Caisson"])
        
        c1, c2 = st.columns(2)
        with c1:
            B = st.number_input("Ancho B / Diámetro (m)", 1.0, 5.0, 1.5)
            L_zap = st.number_input("Largo L (m)", 1.0, 5.0, 1.5) if tipo == "Zapata" else B
            Df = st.number_input("Profundidad Df (m)", 0.5, 10.0, 1.5)
            Q_load = st.number_input("Carga Vertical (kN)", 100.0, 5000.0, 500.0)
            
        with c2:
            if not st.session_state.get('layers_objs'):
                st.warning("Defina suelo en Tab 1")
            else:
                # Calculo simple Terzaghi
                engine = GeotechEngine(st.session_state['layers_objs']) # Dummy init
                # Extraer props a profundidad Df
                phi = 0; su = 0; gamma = 18.0
                for l in st.session_state['layers_objs']:
                    if l.contains(Df + B/2):
                        phi = l.phi; su = l.su; break
                
                if phi < 1: phi = 1
                rad = np.radians(phi)
                Nq = np.exp(np.pi * np.tan(rad)) * (np.tan(np.radians(45) + rad/2))**2
                Nc = (Nq - 1) / np.tan(rad) if phi > 0 else 5.14
                Ny = 2 * (Nq + 1) * np.tan(rad)
                
                q = Df * gamma
                if su > 0: qu = 1.3 * su * Nc + q * Nq 
                else: qu = q * Nq + 0.4 * gamma * B * Ny
                
                Area = B * L_zap if tipo == "Zapata" else np.pi*(B/2)**2
                Q_ult_cap = qu * Area
                FS = Q_ult_cap / Q_load
                
                st.metric("Capacidad Última Total", f"{Q_ult_cap:.1f} kN")
                st.metric("Factor de Seguridad", f"{FS:.2f}", delta="OK" if FS>=3 else "BAJO")
                st.write(f"**Parámetros usados (z={Df+B/2:.1f}m):** Phi={phi:.1f}, Su={su:.1f}")

# ==============================================================================
# MAIN
# ==============================================================================
if st.session_state['usuario_registrado']:
    app_principal()
else:
    mostrar_registro()
      
