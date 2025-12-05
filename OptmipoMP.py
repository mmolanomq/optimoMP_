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
    pass # Se maneja el error en el botón

# ==============================================================================
# CONFIGURACIÓN GLOBAL
# ==============================================================================
st.set_page_config(page_title="MicroPile Pro Optimizer", layout="wide", page_icon="🏗️")

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] { background-color: #FFFFFF; border-bottom: 2px solid #0047AB; }
    h1, h2, h3 { color: #2C3E50; }
</style>
""", unsafe_allow_html=True)

GOOGLE_SCRIPT_URL = ""  # Pega tu URL aquí si deseas registro en la nube

# ==============================================================================
# 1. SISTEMA DE REGISTRO & ESTADO
# ==============================================================================
if 'usuario_registrado' not in st.session_state: st.session_state['usuario_registrado'] = False
if 'datos_usuario' not in st.session_state: st.session_state['datos_usuario'] = {}
# Inicialización segura de variables globales
if 'design_optimo' not in st.session_state: st.session_state['design_optimo'] = {}
if 'layers_objs' not in st.session_state: st.session_state['layers_objs'] = []
if 'df_geo' not in st.session_state: st.session_state['df_geo'] = pd.DataFrame()

def enviar_a_google_sheets(datos):
    if not GOOGLE_SCRIPT_URL: return True
    try:
        requests.post(GOOGLE_SCRIPT_URL, json=datos, timeout=3)
        return True
    except: return False

def mostrar_registro():
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("## 🔒 MicroPile Pro Optimizer")
        st.info("Herramienta especializada en diseño y optimización de micropilotes.")
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
                else: st.error("Complete los datos requeridos.")

# ==============================================================================
# 2. MOTOR CIENTÍFICO (DYWIDAG & GEOTECNIA)
# ==============================================================================

def get_dywidag_db():
    """Catálogo Dywidag (DSI) - Hollow Bars."""
    data = {
        "Sistema": ["R32-280", "R32-360", "R38-500", "R38-550", "R51-660", "R51-800", "T76-1200", "T76-1600"],
        "D_ext_mm": [32, 32, 38, 38, 51, 51, 76, 76],
        "As_mm2": [410, 510, 750, 800, 970, 1150, 1610, 1990],
        "fy_MPa": [535, 550, 533, 560, 555, 556, 620, 600]
    }
    return pd.DataFrame(data)

@dataclass
class SoilLayerObj:
    z_top: float; z_bot: float; tipo: str; alpha: float; kh: float
    def contains(self, z): return self.z_top <= z <= self.z_bot

def procesar_geotecnia(df_input):
    """Calcula Alpha Bond, Phi y E basado en inputs."""
    results = []
    z_acum = 0
    for _, row in df_input.iterrows():
        try:
            esp = float(row.get('Espesor_m', 0)); tipo = row.get('Tipo', 'Arcilla')
            n_spt = float(row.get('N_SPT', 0)); su = float(row.get('Su_kPa', 0))
            kh = float(row.get('Kh_kNm3', 0)); a_manual = float(row.get('Alpha_Manual', 0))
        except: continue
        
        phi = 0; E_MPa = 0; alpha = 0
        
        # Correlaciones (Wolff, Kulhawy, FHWA)
        if tipo == "Arena":
            phi = ((np.sqrt(20*n_spt)+20) + (27.1+0.3*n_spt-0.00054*n_spt**2))/2
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
            "Phi_Deg": phi, "E_MPa": E_MPa, "f_exp": 1.2 # Factor expansión default
        })
        z_acum = z_fin
    return pd.DataFrame(results)

def calc_micropile_axial(L, D, layers, fs):
    """Integra la capacidad por fuste (Bond)."""
    perim = np.pi * D
    z_arr = np.linspace(0, L, 100)
    q_ult = []; curr = 0
    
    for z in z_arr:
        alpha = 0
        if z > 0:
            for l in layers:
                if l.contains(z): alpha = l.alpha; break
            if layers and z > layers[-1].z_bot: alpha = layers[-1].alpha
            
        curr += alpha * perim * (L/100) # Integration step
        q_ult.append(curr)
        
    return z_arr, np.array(q_ult), np.array(q_ult)/fs

def calc_winkler(L, D, EI, V, M, layers):
    """Modelo de Viga sobre Fundación Elástica (Winkler)."""
    if not layers: return np.array([]), np.array([]), np.array([]), np.array([]), 0
    
    # Kh simplificado (promedio superior)
    kh = layers[0].kh if layers[0].kh > 0 else 5000
    
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
    # --- SIDEBAR ---
    with st.sidebar:
        st.success(f"Ingeniero: **{st.session_state['datos_usuario'].get('nombre')}**")
        
        if st.button("📄 Generar Reporte PDF"):
            try:
                buffer = BytesIO()
                p = canvas.Canvas(buffer, pagesize=letter)
                p.drawString(100, 750, "REPORTE DE DISEÑO - MICROPILE PRO")
                p.drawString(100, 730, f"Usuario: {st.session_state['datos_usuario'].get('nombre')}")
                p.drawString(100, 710, f"Empresa: {st.session_state['datos_usuario'].get('empresa')}")
                
                opt = st.session_state.get('design_optimo')
                if opt:
                    p.drawString(100, 680, "RESUMEN OPTIMIZACIÓN:")
                    p.drawString(100, 660, f"- Configuración: {opt.get('N')} micropilotes x D={opt.get('D_mm')}mm")
                    p.drawString(100, 640, f"- Longitud: {opt.get('L_m')} m")
                    p.drawString(100, 620, f"- Huella CO2: {opt.get('CO2_ton'):.2f} Ton")
                else:
                    p.drawString(100, 680, "Nota: No se ha ejecutado la optimización.")
                
                p.showPage()
                p.save()
                st.download_button("Descargar Reporte", buffer, "Reporte_Micropilotes.pdf", "application/pdf")
            except Exception as e:
                st.error("Instale 'reportlab' en requirements.txt para usar PDF.")

        if st.button("Cerrar Sesión"):
            st.session_state['usuario_registrado'] = False; st.rerun()

    st.title("🏗️ MicroPile Pro Optimizer")
    
    tab1, tab2, tab3 = st.tabs([
        "🌍 1. Información Geotécnica", 
        "🚀 2. Optimizador de Diseño", 
        "📐 3. Diseño Detallado & Planos"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: GEOTECNIA (ENTRADA Y VISUALIZACIÓN)
    # --------------------------------------------------------------------------
    with tab1:
        c1, c2 = st.columns([1.3, 1])
        with c1:
            st.subheader("1.1 Estratigrafía y Parámetros")
            st.info("Ingrese el perfil del suelo. El 'Alpha Bond' se calcula automáticamente a menos que ingrese un valor manual.")
            
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
            
            # Guardar en sesión
            st.session_state['df_geo'] = df_geo
            layers_objs = []
            for _, r in df_geo.iterrows():
                layers_objs.append(SoilLayerObj(r['z_ini'], r['z_fin'], r['Tipo'], r['Alpha_Design'], r['Kh_kNm3']))
            st.session_state['layers_objs'] = layers_objs
            
            st.markdown("#### Parámetros de Diseño Calculados")
            fmt = {c: "{:.1f}" for c in ["z_ini", "z_fin", "Alpha_Design", "Phi_Deg", "E_MPa"]}
            st.dataframe(df_geo.style.format(fmt), use_container_width=True)
            
            with st.expander("Ver Ecuaciones de Correlación Utilizadas"):
                st.latex(r"\phi' = \frac{(\sqrt{20 N} + 20) + (27.1 + 0.3N - 0.00054N^2)}{2}")
                st.latex(r"\alpha_{bond} \text{ (Arena)} \approx \min(3.8 N_{SPT}, 250)")
                st.latex(r"\alpha_{bond} \text{ (Arcilla)} \approx f(S_u) \text{ (FHWA Tabla 5-3)}")

        with c2:
            st.subheader("1.2 Perfil Estratigráfico")
            if not df_geo.empty:
                fig, axs = plt.subplots(1, 3, figsize=(10, 6), sharey=True)
                z_max = df_geo['z_fin'].max() + 1
                
                # Preparar datos para "step plot" correcto
                z_plot, n_plot, a_plot = [], [], []
                for _, r in df_geo.iterrows():
                    z_plot.extend([r['z_ini'], r['z_fin']])
                    n_plot.extend([r['N_SPT'], r['N_SPT']])
                    a_plot.extend([r['Alpha_Design'], r['Alpha_Design']])
                
                # Plot 1: N-SPT
                axs[0].plot(n_plot, z_plot, 'b', linewidth=2); axs[0].set_title("N-SPT")
                axs[0].grid(True, ls=':'); axs[0].invert_yaxis(); axs[0].set_ylabel("Profundidad (m)")
                
                # Plot 2: Alpha
                axs[1].plot(a_plot, z_plot, 'r', linewidth=2); axs[1].set_title("Alpha Bond (kPa)")
                axs[1].grid(True, ls=':')
                
                # Plot 3: Visual
                cols = {"Arcilla": "#D7BDE2", "Arena": "#F9E79F", "Roca": "#AED6F1", "Relleno": "#D3D3D3"}
                for _, r in df_geo.iterrows():
                    rect = patches.Rectangle((0, r['z_ini']), 1, r['Espesor_m'], facecolor=cols.get(r['Tipo'], 'white'), ec='k')
                    axs[2].add_patch(rect)
                    axs[2].text(0.5, (r['z_ini']+r['z_fin'])/2, r['Tipo'], ha='center', va='center', rotation=90, fontsize=8)
                axs[2].set_xlim(0, 1); axs[2].set_title("Perfil"); axs[2].axis('off')
                
                plt.ylim(z_max, 0)
                st.pyplot(fig)

    # --------------------------------------------------------------------------
    # TAB 2: OPTIMIZACIÓN (LÓGICA ANTERIOR)
    # --------------------------------------------------------------------------
    with tab2:
        st.subheader("🚀 Optimizador Multi-Variable")
        st.info("Algoritmo iterativo para encontrar el diseño más eficiente en Costo y CO2.")
        
        c_opt1, c_opt2 = st.columns(2)
        with c_opt1:
            CARGA_TON = st.number_input("Carga Total Grupo (Ton)", 50.0, 5000.0, 200.0)
            FS_REQ = st.number_input("FS Geotécnico Objetivo", 1.5, 3.0, 2.0)
        with c_opt2:
            RELACION_AGUA_CEMENTO = st.slider("Relación A/C Lechada", 0.4, 0.6, 0.5)
        
        if st.button("EJECUTAR OPTIMIZACIÓN"):
            if 'df_geo' not in st.session_state or st.session_state['df_geo'].empty:
                st.error("⚠️ Faltan datos de suelo. Vaya a la Pestaña 1.")
            else:
                # --- PREPARACIÓN DE DATOS ---
                ESTRATOS = []
                for _, r in st.session_state['df_geo'].iterrows():
                    ESTRATOS.append({
                        "z_fin": r['z_fin'], "qs": r['Alpha_Design'], "f_exp": r['f_exp']
                    })
                
                # --- PARÁMETROS FIJOS (TU LÓGICA) ---
                DIAMETROS_COM = {0.100: 1.00, 0.115: 0.95, 0.150: 0.90, 0.200: 0.85}
                LISTA_D = sorted(list(DIAMETROS_COM.keys()))
                MIN_MICROS = 3; MAX_MICROS = 20
                RANGO_L = range(6, 31) # 6m a 30m
                COSTO_PERF_BASE = 100
                F_CO2_CEM = 0.9; F_CO2_PERF = 15.0; F_CO2_ACE = 1.85
                DENS_ACERO = 7850.0; DENS_CEM = 3150.0; FY_KPA = 500000.0
                
                CARGA_KN = CARGA_TON * 9.81
                resultados_raw = []
                
                bar = st.progress(0)
                
                # --- BUCLE DE OPTIMIZACIÓN ---
                for idx, D in enumerate(LISTA_D):
                    bar.progress((idx+1)/len(LISTA_D))
                    for N in range(MIN_MICROS, MAX_MICROS + 1):
                        Q_act_pilote = CARGA_KN / N
                        Q_req_geo = Q_act_pilote * FS_REQ
                        
                        for L in RANGO_L:
                            # 1. Calc Capacidad Geo
                            Q_ult = 0; z_curr = 0
                            for e in ESTRATOS:
                                if z_curr >= L: break
                                z_bot = min(e["z_fin"], L)
                                thick = z_bot - z_curr
                                if thick > 0: Q_ult += (np.pi * D * thick) * e["qs"]
                                z_curr = z_bot
                            
                            if Q_ult >= Q_req_geo:
                                # Diseño viable encontrado para este N y D (L mínima)
                                FS_calc = Q_ult / Q_act_pilote
                                
                                # 2. Volúmenes
                                v_exp_tot = 0; z_curr = 0
                                for e in ESTRATOS:
                                    if z_curr >= L: break
                                    z_bot = min(e["z_fin"], L)
                                    thick = z_bot - z_curr
                                    if thick > 0: v_exp_tot += (np.pi*(D/2)**2 * thick) * e["f_exp"]
                                    z_curr = z_bot
                                v_exp_tot *= N
                                
                                # 3. Metricas
                                costo = (L * N * COSTO_PERF_BASE) / DIAMETROS_COM[D]
                                area_acero = Q_act_pilote / FY_KPA
                                peso_acero = area_acero * L * N * DENS_ACERO
                                peso_cem = v_exp_tot * (1.0 / (RELACION_AGUA_CEMENTO/1000 + 1/DENS_CEM))
                                co2 = (peso_acero*F_CO2_ACE + peso_cem*F_CO2_CEM + (L*N)*F_CO2_PERF)/1000
                                
                                resultados_raw.append({
                                    "D_mm": int(D*1000), "N": N, "L_m": L, "L_Tot_m": L*N,
                                    "FS": FS_calc, "Vol_Exp": v_exp_tot, "Costo_Idx": int(costo),
                                    "CO2_ton": co2, "Q_adm_geo": Q_ult/FS_REQ/9.81, "Q_act": Q_act_pilote/9.81
                                })
                                break # Salir del loop de L, ya encontramos la minima
                
                bar.empty()
                
                if resultados_raw:
                    df_res = pd.DataFrame(resultados_raw).sort_values("Costo_Idx")
                    best = df_res.iloc[0]
                    st.session_state['design_optimo'] = best.to_dict() # Guardar
                    
                    st.success("✅ Diseño Optimizado Encontrado")
                    
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Configuración", f"{best['N']} x Ø{best['D_mm']}mm")
                    k2.metric("Longitud", f"{best['L_m']} m")
                    k3.metric("Huella CO2", f"{best['CO2_ton']:.1f} Ton")
                    k4.metric("Índice Costo", f"{best['Costo_Idx']}")
                    
                    st.dataframe(df_res.head(10).style.background_gradient(subset=['Costo_Idx'], cmap='Greens_r'), use_container_width=True)
                    
                    # Gráfica de Dispersión (Espacio de soluciones)
                    fig_sc, ax_sc = plt.subplots(figsize=(8,4))
                    sc = ax_sc.scatter(df_res['Costo_Idx'], df_res['CO2_ton'], c=df_res['L_m'], cmap='viridis', alpha=0.7)
                    plt.colorbar(sc, label='Longitud (m)')
                    ax_sc.set_xlabel('Índice Costo'); ax_sc.set_ylabel('Huella CO2')
                    ax_sc.set_title("Mapa de Soluciones Factibles")
                    ax_sc.grid(True, ls=':')
                    st.pyplot(fig_sc)
                else:
                    st.error("No se encontraron soluciones. Intente aumentar el rango de diámetros o reducir la carga.")

    # --------------------------------------------------------------------------
    # TAB 3: DISEÑO DETALLADO
    # --------------------------------------------------------------------------
    with tab3:
        st.subheader("📐 Diseño Detallado & Verificaciones")
        
        # Recuperar optimización de forma segura
        opt = st.session_state.get('design_optimo', {})
        def_L = float(opt.get('L_m', 12.0)); def_D = float(opt.get('D_mm', 200.0))/1000
        
        c_in, c_out = st.columns([1, 1.5])
        with c_in:
            st.markdown("##### Configuración del Micropilote")
            L = st.number_input("Longitud (m)", 1.0, 50.0, def_L)
            D = st.number_input("Diámetro (m)", 0.1, 0.6, def_D)
            
            db = get_dywidag_db()
            sys = st.selectbox("Refuerzo (Dywidag)", db['Sistema'], index=2)
            row_s = db[db['Sistema'] == sys].iloc[0]
            st.info(f"Acero: {row_s['As_mm2']} mm² | Fy: {row_s['fy_MPa']} MPa")
            
            fc = st.number_input("f'c Grout (MPa)", 20.0, 60.0, 30.0)
            
            st.markdown("##### Cargas por Micropilote")
            P_u = st.number_input("Compresión Pu (kN)", value=float(opt.get('Q_act', 50)*9.81) if opt else 500.0)
            V_u = st.number_input("Cortante Vu (kN)", value=30.0)
            M_u = st.number_input("Momento Mu (kNm)", value=15.0)
            
        with c_out:
            if not st.session_state.get('layers_objs'):
                st.warning("⚠️ Vaya a la Pestaña 1 y defina el suelo.")
            else:
                layers = st.session_state['layers_objs']
                
                # 1. CÁLCULO AXIAL DETALLADO
                z_ax, q_ult, q_adm = calc_micropile_axial(L, D, layers, 2.0)
                
                # 2. CÁLCULO ESTRUCTURAL
                As = row_s['As_mm2']; fy = row_s['fy_MPa']
                A_g = (np.pi*(D*1000/2)**2) - As
                P_est = (0.40*fc*A_g + 0.47*fy*As)/1000
                
                # 3. CÁLCULO LATERAL (WINKLER)
                I_b = (np.pi*(row_s['D_ext_mm']/1000)**4)/64
                EI = 200e6 * I_b + (4700*np.sqrt(fc)*1000 * ((np.pi*D**4)/64 - I_b))
                z_lat, y_lat, m_lat, v_lat, beta = calc_winkler(L, D, EI, V_u, M_u, layers)
                
                # --- RESULTADOS ---
                k1, k2, k3 = st.columns(3)
                q_geo_val = q_adm[-1] if len(q_adm)>0 else 0
                k1.metric("Q Admisible Geo", f"{q_geo_val:.1f} kN", delta="OK" if q_geo_val>P_u else "FALLA")
                k2.metric("P Estructural", f"{P_est:.1f} kN")
                k3.metric("Deflexión Máx", f"{max(abs(y_lat))*1000:.1f} mm")
                
                # --- GRÁFICAS INTEGRADAS ---
                fig_res, (ax_geo, ax_def, ax_mom) = plt.subplots(1, 3, figsize=(12, 6), sharey=True)
                
                # 1. Capacidad vs Profundidad + Suelo
                max_q = max(q_ult)*1.2 if len(q_ult)>0 else 100
                cols_map = {"Arcilla": "#D7BDE2", "Arena": "#F9E79F", "Roca": "#AED6F1"}
                for l in layers:
                    rect = patches.Rectangle((0, l.z_top), max_q, l.z_bot-l.z_top, fc=cols_map.get(l.tipo,'white'), alpha=0.3)
                    ax_geo.add_patch(rect)
                    ax_geo.text(max_q*0.05, (l.z_top+l.z_bot)/2, f"{l.tipo}\n$\\alpha$={l.alpha:.0f}", fontsize=8)
                
                ax_geo.plot(q_adm, z_ax, 'b-', label='Q Adm')
                ax_geo.plot(q_ult, z_ax, 'k--', label='Q Ult')
                ax_geo.axvline(P_u, c='r', ls=':', label='Pu')
                
                # Dibujo esquemático del micropilote
                rect_mp = patches.Rectangle((max_q*0.8, 0), max_q*0.05, L, fc='gray', ec='k')
                ax_geo.add_patch(rect_mp)
                
                ax_geo.invert_yaxis(); ax_geo.legend(); ax_geo.set_title("Capacidad Axial"); ax_geo.grid(True, ls=':')
                ax_geo.set_ylabel("Profundidad (m)")
                
                # 2. Deflexión
                ax_def.plot(y_lat*1000, z_lat, 'm'); ax_def.set_title("Deflexión (mm)"); ax_def.grid(True, ls=':')
                
                # 3. Momento / Cortante
                ax_mom.plot(m_lat, z_lat, 'g', label='Momento'); ax_mom.set_title("Momento (kNm)"); ax_mom.grid(True, ls=':')
                ax_mom2 = ax_mom.twiny()
                ax_mom2.plot(v_lat, z_lat, 'orange', ls='--', label='Cortante')
                
                st.pyplot(fig_res)
                
                # ECUACIONES DETALLADAS
                st.markdown("---")
                with st.expander("📝 Ver Ecuaciones de Diseño Detalladas"):
                    c_eq1, c_eq2 = st.columns(2)
                    with c_eq1:
                        st.markdown("**1. Capacidad Geotécnica (Bond)**")
                        st.latex(r"Q_{ult} = \sum (\alpha_{bond} \cdot \pi \cdot D_b \cdot \Delta L)")
                        st.latex(r"Q_{adm} = Q_{ult} / FS")
                        
                        st.markdown("**2. Capacidad Estructural (Compresión)**")
                        st.latex(r"P_{c,all} = 0.40 f'_c A_{grout} + 0.47 f_y A_{steel}")
                    
                    with c_eq2:
                        st.markdown("**3. Deflexión Lateral (Winkler)**")
                        st.latex(r"EI \frac{d^4y}{dz^4} + k_h D y = 0")
                        st.latex(r"\beta = \sqrt[4]{\frac{k_h D}{4 EI}}")
                        st.write(f"Rigidez Flexionante Calculada $EI$: {EI/1e9:.2f} GN·m²")

# ==============================================================================
# CONTROL DE FLUJO
# ==============================================================================
if st.session_state['usuario_registrado']:
    app_principal()
else:
    mostrar_registro()
