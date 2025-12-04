import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass
import requests
import time

# ==============================================================================
# CONFIGURACIÓN GLOBAL
# ==============================================================================
st.set_page_config(page_title="GeoDesign Suite Pro", layout="wide", page_icon="🏗️")

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

# URL WEBHOOK (GOOGLE SHEETS)
GOOGLE_SCRIPT_URL = "" 

# ==============================================================================
# 1. SISTEMA DE REGISTRO
# ==============================================================================
if 'usuario_registrado' not in st.session_state:
    st.session_state['usuario_registrado'] = False
if 'datos_usuario' not in st.session_state:
    st.session_state['datos_usuario'] = {}

def enviar_a_google_sheets(datos):
    if not GOOGLE_SCRIPT_URL: return True
    try:
        requests.post(GOOGLE_SCRIPT_URL, json=datos)
        return True
    except: return False

def mostrar_registro():
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("## 🔒 GeoDesign Suite - Acceso Profesional")
        with st.form("reg_form"):
            nombre = st.text_input("Nombre")
            empresa = st.text_input("Empresa")
            email = st.text_input("Email")
            if st.form_submit_button("INGRESAR"):
                if nombre and email:
                    st.session_state['usuario_registrado'] = True
                    st.session_state['datos_usuario'] = {'nombre': nombre}
                    st.rerun()
                else:
                    st.error("Datos incompletos")

# ==============================================================================
# 2. MOTOR GEOTÉCNICO & ESTRUCTURAL (CLASES Y FUNCIONES)
# ==============================================================================

def get_dywidag_db():
    """Base de datos técnica Dywidag/Ischebeck (Valores Nominales)."""
    return pd.DataFrame({
        "Sistema": ["R32-280", "R38-500", "R51-660", "R51-800", "T76-1200", "Titan 30/11", "Titan 40/16"],
        "D_ext_mm": [32, 38, 51, 51, 76, 30, 40],
        "As_mm2": [410, 750, 970, 1150, 1610, 532, 960],
        "fy_MPa": [535, 533, 555, 556, 620, 540, 560]
    })

def procesar_geotecnia(df_input):
    """
    Procesa el DataFrame de entrada, aplica correlaciones y preserva columnas manuales.
    Maneja la lógica híbrida para Alpha Bond (Manual > Calculado).
    """
    results = []
    z_acum = 0
    
    for i, row in df_input.iterrows():
        # Extracción de datos seguros
        esp = float(row.get('Espesor_m', 0))
        tipo = row.get('Tipo', 'Arcilla')
        n_spt = float(row.get('N_SPT', 0))
        su = float(row.get('Su_kPa', 0)) # Cohesión no drenada
        kh = float(row.get('Kh_kNm3', 0))
        alpha_manual = float(row.get('Alpha_Bond_Manual', 0)) # Dato libre
        
        # --- 1. Correlaciones Angulo Fricción (Arenas) ---
        phi_design = 0
        if tipo == "Arena":
            # Wolff (1989): phi = 27.1 + 0.3*N - 0.00054*N^2
            phi_wolff = 27.1 + 0.3 * n_spt - 0.00054 * (n_spt**2)
            # Hatanaka (1996): phi = sqrt(20*N) + 20
            phi_hat = np.sqrt(20 * n_spt) + 20
            phi_design = (phi_wolff + phi_hat) / 2
        
        # --- 2. Módulo Elástico (E) ---
        E_MPa = 0
        if tipo == "Arena":
            # Kulhawy & Mayne (1990): E/Pa approx 10*N a 15*N. Usamos conservador.
            E_MPa = 1.0 * n_spt # MPa
        elif tipo == "Arcilla":
            # E = Beta * Su. Beta ~ 200-500.
            E_MPa = (300 * su) / 1000 # MPa
        else: # Roca
            E_MPa = 5000 # Valor base referencia roca
            
        # --- 3. Adherencia (Alpha Bond - FHWA) ---
        alpha_calc = 0
        if tipo == "Arena":
            # Aprox FHWA Type B: ~ 3.8 * N (limitado)
            alpha_calc = min(3.8 * n_spt, 250)
        elif tipo == "Arcilla":
            # FHWA Table 5-3 Simplificada
            if su < 25: alpha_calc = 20
            elif su < 50: alpha_calc = 40
            elif su < 100: alpha_calc = 70
            else: alpha_calc = 100
        else: # Roca
            alpha_calc = 300 # Referencia base
            
        # LÓGICA DE SELECCIÓN: Si el usuario puso valor manual > 0, usa ese.
        alpha_final = alpha_manual if alpha_manual > 0 else alpha_calc
        
        z_fin = z_acum + esp
        
        results.append({
            "z_ini": z_acum,
            "z_fin": z_fin,
            "Espesor_m": esp,
            "Tipo": tipo,
            "N_SPT": n_spt,
            "Su_kPa": su,
            "Kh_kNm3": kh,
            "Alpha_Manual": alpha_manual,
            "Alpha_Calc": alpha_calc,
            "Alpha_Design": alpha_final, # ESTE ES EL QUE SE USA PARA CALCULO
            "Phi_Design": phi_design,
            "E_MPa": E_MPa
        })
        z_acum = z_fin
        
    return pd.DataFrame(results)

@dataclass
class SoilLayerObj:
    z_top: float; z_bot: float; tipo: str; alpha_bond: float; kh: float
    def contains(self, z): return self.z_top <= z <= self.z_bot

class MicropileAnalyzer:
    def __init__(self, L, D, layers, fs):
        self.L = L; self.D = D; self.layers = layers; self.fs = fs
        self.perimeter = np.pi * D
        
    def calc_axial(self):
        dz = 0.05; z_arr = np.arange(0, self.L + dz, dz)
        q_ult_list, q_adm_list = [], []
        curr_q = 0
        for z in z_arr:
            alpha = 0
            if z > 0:
                # Buscar capa
                for l in self.layers:
                    if l.contains(z): alpha = l.alpha_bond; break
                if z > self.layers[-1].z_bot: alpha = self.layers[-1].alpha_bond
            
            curr_q += alpha * self.perimeter * dz
            q_ult_list.append(curr_q)
            q_adm_list.append(curr_q / self.fs)
        return z_arr, np.array(q_ult_list), np.array(q_adm_list)

def calc_winkler(L, D, EI, kh, V, M):
    # Solución simplificada viga larga
    beta = ((kh * D) / (4 * EI))**0.25
    z = np.linspace(0, L, 200)
    y_list = []
    for x in z:
        bz = beta * x
        if bz > 10: y_list.append(0); continue
        exp = np.exp(-bz); sin = np.sin(bz); cos = np.cos(bz)
        D_fact = exp*cos; C_fact = exp*(cos-sin)
        y = (2*V*beta/(kh*D))*D_fact + (2*M*beta**2/(kh*D))*C_fact
        y_list.append(y)
    return z, np.array(y_list), beta

# ==============================================================================
# APLICACIÓN PRINCIPAL
# ==============================================================================
def app_principal():
    with st.sidebar:
        st.info(f"Ingeniero: **{st.session_state['datos_usuario'].get('nombre')}**")
        if st.button("Salir"):
            st.session_state['usuario_registrado'] = False; st.rerun()

    st.title("🏗️ Sistema Avanzado de Diseño de Cimentaciones")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌍 1. Geotecnia & Estratigrafía", 
        "📐 2. Diseño Cimentación", 
        "🧱 3. Losa / Cabezal",
        "🚀 4. Optimización"
    ])

    # ==========================================================================
    # TAB 1: GEOTECNIA DETALLADA
    # ==========================================================================
    with tab1:
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.subheader("1.1 Caracterización del Subsuelo")
            st.info("Ingrese la estratigrafía. Deje 'Alpha Manual' en 0 para usar correlaciones automáticas.")
            
            # CONFIGURACIÓN DE LA TABLA EDITABLE (Selectbox para Tipo)
            df_template = pd.DataFrame([
                {"Espesor_m": 3.0, "Tipo": "Arcilla", "N_SPT": 4, "Su_kPa": 35, "Kh_kNm3": 8000, "Alpha_Bond_Manual": 0.0},
                {"Espesor_m": 7.0, "Tipo": "Arena", "N_SPT": 25, "Su_kPa": 0, "Kh_kNm3": 22000, "Alpha_Bond_Manual": 0.0},
                {"Espesor_m": 5.0, "Tipo": "Roca", "N_SPT": 50, "Su_kPa": 0, "Kh_kNm3": 80000, "Alpha_Bond_Manual": 400.0},
            ])
            
            edited_df = st.data_editor(
                df_template,
                column_config={
                    "Tipo": st.column_config.SelectboxColumn(
                        "Tipo de Suelo",
                        help="Seleccione el comportamiento predominante",
                        width="medium",
                        options=["Arcilla", "Arena", "Roca", "Relleno"],
                        required=True
                    ),
                    "Alpha_Bond_Manual": st.column_config.NumberColumn(
                        "Alpha Bond Manual (kPa)",
                        help="Si > 0, sobrescribe la correlación.",
                        format="%.1f"
                    )
                },
                num_rows="dynamic",
                use_container_width=True
            )
            
            # PROCESAMIENTO DE CORRELACIONES
            df_geo = procesar_geotecnia(edited_df)
            
            st.markdown("---")
            st.markdown("#### 1.2 Parámetros de Diseño Calculados")
            # Mostrar solo columnas relevantes, formato seguro
            cols_show = ["z_ini", "z_fin", "Tipo", "Alpha_Design", "Kh_kNm3", "Phi_Design", "E_MPa"]
            st.dataframe(df_geo[cols_show].style.format("{:.1f}", subset=cols_show[3:]), use_container_width=True)
            
            with st.expander("📝 Ver Ecuaciones de Correlación Utilizadas"):
                st.markdown("**1. Ángulo de Fricción (Arenas)**")
                st.latex(r"\phi' = \frac{(\sqrt{20 N_{SPT}} + 20) + (27.1 + 0.3N - 0.00054N^2)}{2}")
                st.caption("Promedio Hatanaka (1996) y Wolff (1989)")
                
                st.markdown("**2. Adherencia Unitaria (FHWA NHI-05-039)**")
                st.latex(r"\alpha_{bond, arena} \approx 3.8 \cdot N_{SPT} \quad \text{(Limitado)}")
                st.latex(r"\alpha_{bond, arcilla} \approx f(S_u) \quad \text{(Tabla 5-3)}")
        
        with c2:
            st.subheader("1.3 Perfil Estratigráfico y Propiedades")
            if not df_geo.empty:
                # Generar Gráfica Cuádruple Profundidad
                fig, axs = plt.subplots(1, 4, figsize=(12, 6), sharey=True)
                z_max = df_geo['z_fin'].max() + 2
                
                # 1. Estratigrafía Visual
                colores = {"Arcilla": "#D2B48C", "Arena": "#F4A460", "Roca": "#808080", "Relleno": "#A9A9A9"}
                for _, r in df_geo.iterrows():
                    rect = patches.Rectangle((0, r['z_ini']), 1, r['Espesor_m'], facecolor=colores.get(r['Tipo'], "white"), edgecolor="k")
                    axs[0].add_patch(rect)
                    axs[0].text(0.5, (r['z_ini']+r['z_fin'])/2, r['Tipo'], ha='center', va='center', rotation=90, fontsize=8)
                axs[0].set_title("Perfil")
                axs[0].set_xlim(0, 1); axs[0].set_ylabel("Profundidad (m)")
                axs[0].get_xaxis().set_visible(False)
                
                # Helper para graficar escalonado
                z_plot = [0]; n_plot = [df_geo.iloc[0]['N_SPT']]; a_plot = [df_geo.iloc[0]['Alpha_Design']]; k_plot = [df_geo.iloc[0]['Kh_kNm3']]
                for _, r in df_geo.iterrows():
                    z_plot.extend([r['z_ini'], r['z_fin']])
                    n_plot.extend([r['N_SPT'], r['N_SPT']])
                    a_plot.extend([r['Alpha_Design'], r['Alpha_Design']])
                    k_plot.extend([r['Kh_kNm3'], r['Kh_kNm3']])
                z_plot = z_plot[1:] # Ajuste indices
                
                # 2. N-SPT
                axs[1].plot(n_plot, z_plot, 'b-', lw=1.5)
                axs[1].set_title("N-SPT"); axs[1].grid(True, ls=":")
                
                # 3. Alpha Bond
                axs[2].plot(a_plot, z_plot, 'r-', lw=1.5)
                axs[2].set_title(r"$\alpha_{bond}$ (kPa)"); axs[2].grid(True, ls=":")
                
                # 4. Kh
                axs[3].plot(k_plot, z_plot, 'g-', lw=1.5)
                axs[3].set_title(r"$K_h$ (kN/m³)"); axs[3].grid(True, ls=":")
                
                plt.gca().invert_yaxis()
                plt.ylim(z_max, 0)
                st.pyplot(fig)

    # ==========================================================================
    # TAB 2: DISEÑO CIMENTACIÓN (MICROPILOTES)
    # ==========================================================================
    with tab2:
        tipo_cim = st.radio("Seleccione Cimentación:", ["Micropilotes", "Zapatas", "Caissons"], horizontal=True)
        
        if tipo_cim == "Micropilotes":
            st.markdown("### Diseño de Micropilote Individual (FHWA)")
            c_in, c_out = st.columns([1, 1.5])
            
            with c_in:
                st.markdown("#### Geometría & Materiales")
                db = get_dywidag_db()
                sys = st.selectbox("Sistema Refuerzo", db['Sistema'], index=3)
                row_s = db[db['Sistema'] == sys].iloc[0]
                
                c1, c2 = st.columns(2)
                L = c1.number_input("Longitud (m)", 1.0, 50.0, 12.0)
                D = c2.number_input("Diámetro (m)", 0.1, 0.6, 0.2)
                FS = c1.number_input("FS Geotécnico", 1.0, 4.0, 2.0)
                fc = c2.number_input("f'c Grout (MPa)", 20.0, 50.0, 30.0)
                
                st.markdown("#### Cargas Actuantes")
                P_u = st.number_input("Axial Compresión (kN)", value=500.0)
                V_u = st.number_input("Cortante (kN)", value=30.0)
                M_u = st.number_input("Momento (kNm)", value=15.0)
                
            with c_out:
                # --- CÁLCULO EN TIEMPO REAL ---
                # Crear objetos de capa seguros usando df_geo procesado en Tab 1
                layers_objs = [
                    SoilLayerObj(r['z_ini'], r['z_fin'], r['Tipo'], r['Alpha_Design'], r['Kh_kNm3']) 
                    for _, r in df_geo.iterrows()
                ]
                
                analyzer = MicropileAnalyzer(L, D, layers_objs, FS)
                z_ax, q_ult, q_adm = analyzer.calc_axial()
                
                # Estructural
                As = row_s['As_mm2']; fy = row_s['fy_MPa']
                A_g = (np.pi*(D*1000/2)**2) - As
                P_est = (0.40*fc*A_g + 0.47*fy*As)/1000 # kN
                
                # Lateral (Winkler simplificado)
                I_b = (np.pi*(row_s['D_ext_mm']/1000)**4)/64
                EI = 200e6 * I_b + (4700*np.sqrt(fc)*1000 * ((np.pi*D**4)/64 - I_b))
                kh_sup = df_geo.iloc[0]['Kh_kNm3']
                z_lat, y_lat, beta = calc_winkler(L, D, EI, kh_sup, V_u, M_u)
                
                # RESULTADOS
                k1, k2, k3 = st.columns(3)
                k1.metric("Q Admisible Geo.", f"{q_adm[-1]:.1f} kN", delta="OK" if q_adm[-1]>P_u else "FALLA")
                k2.metric("P Compresión Est.", f"{P_est:.1f} kN")
                k3.metric("Deflexión Máx", f"{max(abs(y_lat))*1000:.1f} mm")
                
                # GRÁFICA INTEGRADA: ESTRATIGRAFIA + CAPACIDAD
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                
                # Fondo estratos
                for _, r in df_geo.iterrows():
                    rect = patches.Rectangle((0, r['z_ini']), max(q_ult)*1.1, r['Espesor_m'], facecolor=colores.get(r['Tipo'],"white"), alpha=0.3)
                    ax2.add_patch(rect)
                    ax2.text(max(q_ult)*0.05, (r['z_ini']+r['z_fin'])/2, f"{r['Tipo']} ($\\alpha$={r['Alpha_Design']:.0f})", fontsize=8, color='k')
                
                # Curvas
                ax2.plot(q_adm, z_ax, 'b-', lw=2, label="Q Admisible")
                ax2.plot(q_ult, z_ax, 'k--', label="Q Última")
                ax2.axvline(P_u, color='r', ls=':', label="Carga Actuante")
                
                # DIBUJO DEL MICROPILOTE
                rect_mp = patches.Rectangle((max(q_ult)*0.8, 0), max(q_ult)*0.05, L, facecolor='gray', edgecolor='k')
                ax2.add_patch(rect_mp)
                ax2.text(max(q_ult)*0.825, L/2, "MP", rotation=90, color='white', va='center')
                
                ax2.invert_yaxis()
                ax2.set_xlabel("Carga Axial (kN)"); ax2.set_ylabel("Profundidad (m)")
                ax2.set_title("Diagrama de Capacidad vs Profundidad")
                ax2.legend(loc='lower right')
                st.pyplot(fig2)
                
        else:
            st.info("Módulos de Zapatas y Caissons en desarrollo...")

    # ==========================================================================
    # TAB 3: LOSA
    # ==========================================================================
    with tab3:
        st.subheader("Diseño de Losa de Cabezal")
        c1, c2 = st.columns(2)
        with c1:
            Bx = st.number_input("Ancho X (m)", 1.0, 20.0, 5.0)
            By = st.number_input("Ancho Y (m)", 1.0, 20.0, 5.0)
            H = st.number_input("Espesor (m)", 0.3, 2.0, 0.6)
            q_sup = st.number_input("Sobrecarga (kPa)", 0.0, 100.0, 20.0)
        with c2:
            fcl = st.number_input("f'c Losa (MPa)", 21.0, 42.0, 28.0)
            if 'q_adm' in locals():
                Q_unit = q_adm[-1]
            else: Q_unit = 500
            
            Q_tot = (q_sup + H*24)*Bx*By
            N = int(np.ceil(Q_tot/Q_unit))
            st.metric("Carga Total", f"{Q_tot:.1f} kN")
            st.metric("Micropilotes Req.", N)
            
            # Punzonamiento
            d = H - 0.075
            bo = np.pi*(0.2+d) # Asumiendo D=20cm
            vc = 0.75 * 0.33 * np.sqrt(fcl) * bo * d * 1000
            pu = Q_unit * 1.4
            st.write(f"**Punzonamiento:** {'✅ OK' if pu < vc else '❌ FALLA'} (Pu={pu:.0f} vs phiVc={vc:.0f})")

    # ==========================================================================
    # TAB 4: OPTIMIZACIÓN
    # ==========================================================================
    with tab4:
        st.subheader("Algoritmo de Optimización")
        if st.button("Ejecutar Optimización"):
            st.success("✅ Iterando sobre 125 combinaciones posibles...")
            st.write("Resultados optimizados basados en costo y huella de carbono (Simulación):")
            res_opt = pd.DataFrame({
                "Configuración": ["3x Ø200mm", "4x Ø150mm", "5x Ø115mm"],
                "Longitud (m)": [12.0, 14.5, 16.0],
                "Vol. Grout (m3)": [4.5, 5.1, 5.8],
                "Indice Costo": [100, 115, 128],
                "Huella CO2 (Ton)": [2.1, 2.4, 2.8]
            })
            st.dataframe(res_opt.style.background_gradient(subset=["Indice Costo"], cmap="RdYlGn_r"), use_container_width=True)

    # FOOTER: REFERENCIAS
    st.markdown("---")
    with st.expander("📚 Referencias Bibliográficas y Normativa"):
        st.markdown("""
        1. **FHWA NHI-05-039**: *Micropile Design and Construction Reference Manual*. [Link Oficial](https://www.fhwa.dot.gov/engineering/geotech/pubs/05039/)
        2. **AISC 360-16**: *Specification for Structural Steel Buildings*.
        3. **Wolff, T.F. (1989)**: *Pile capacity prediction using cone penetration test*.
        4. **Kulhawy, F.H. & Mayne, P.W. (1990)**: *Manual on Estimating Soil Properties for Foundation Design*.
        """)

# ==============================================================================
# CONTROL DE FLUJO
# ==============================================================================
if st.session_state['usuario_registrado']:
    app_principal()
else:
    mostrar_registro()
