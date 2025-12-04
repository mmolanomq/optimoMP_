import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass
import requests
import time

# ==============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==============================================================================
st.set_page_config(page_title="Suite de Diseño Geotécnico", layout="wide", page_icon="🏗️")

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

# URL para registro (Google Apps Script) - Pegar tu URL aquí
GOOGLE_SCRIPT_URL = "" 

# ==============================================================================
# 0. SISTEMA DE REGISTRO
# ==============================================================================
if 'usuario_registrado' not in st.session_state:
    st.session_state['usuario_registrado'] = False
if 'datos_usuario' not in st.session_state:
    st.session_state['datos_usuario'] = {}

def mostrar_registro():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("## 🔒 Acceso a Herramienta de Ingeniería")
        st.info("Plataforma de Diseño y Optimización de Cimentaciones.")
        with st.form("formulario_registro"):
            nombre = st.text_input("Nombre Completo")
            c1, c2 = st.columns(2)
            empresa = c1.text_input("Empresa")
            email = c2.text_input("Email")
            cargo = st.selectbox("Cargo", ["Ingeniero Geotecnista", "Ingeniero Estructural", "Estudiante", "Otro"])
            acepto = st.checkbox("Acepto términos de uso académico/profesional.")
            if st.form_submit_button("🚀 INGRESAR"):
                if nombre and email and acepto:
                    st.session_state['datos_usuario'] = {"nombre": nombre, "empresa": empresa}
                    st.session_state['usuario_registrado'] = True
                    st.rerun()
                else:
                    st.warning("Complete los datos requeridos.")

# ==============================================================================
# FUNCIONES AUXILIARES (CORRELACIONES Y BASE DE DATOS)
# ==============================================================================

def get_dywidag_db():
    """Base de datos corregida Dywidag (R51-800 Area=1150mm2)."""
    data = {
        "Sistema": ["R32-280", "R38-500", "R51-660", "R51-800", "T76-1200", "Titan 30/11", "Titan 40/16"],
        "D_ext_mm": [32, 38, 51, 51, 76, 30, 40],
        "As_mm2": [410, 750, 970, 1150, 1610, 532, 960],
        "fy_MPa": [535, 533, 555, 556, 620, 540, 560]
    }
    return pd.DataFrame(data)

def calcular_correlaciones(df):
    """Genera correlaciones geotécnicas basadas en inputs."""
    # Referencias: 
    # [1] Wolff (1989) - Phi vs N60
    # [2] Hatanaka & Uchida (1996) - Phi vs N60
    # [3] Kulhawy & Mayne (1990) - E vs N60
    # [4] FHWA NHI-05-039 - Alpha Bond vs N/Su
    
    results = []
    z_acum = 0
    
    for i, row in df.iterrows():
        n_spt = row['N_SPT']
        tipo = row['Tipo'] # Arena / Arcilla
        su = row.get('Su_kPa', 0)
        
        # 1. Angulo de Friccion (Arenas)
        phi_1, phi_2 = 0, 0
        if tipo == "Arena":
            phi_1 = 27.1 + 0.3 * n_spt - 0.00054 * (n_spt**2) # Wolff
            phi_2 = np.sqrt(20 * n_spt) + 20 # Hatanaka approx
            phi_prom = (phi_1 + phi_2)/2
        else:
            phi_prom = 0 # Arcilla saturada phi=0 (Total stress)
            
        # 2. Modulo Elasticidad (E) - MPa
        # Arena: E = alpha * N (alpha ~ 1.0 a 3.0 MPa para N60)
        # Arcilla: E = beta * Su (beta ~ 200 to 500)
        E_calc = 0
        if tipo == "Arena":
            E_calc = 1.5 * n_spt # Valor medio conservador
        else:
            E_calc = (500 * su) / 1000 # MPa (Beta=500 muy rigido, 200 blando)
            
        # 3. Adherencia Unitari (Alpha Bond) - FHWA Type A/B
        alpha_bond = 0
        if tipo == "Arena":
            # Aprox k ~ 3.5 a 5 * N (kPa) limitado a ~250
            alpha_bond = min(3.5 * n_spt, 250)
        else:
            # Arcilla: Alpha Method (Sladen 1992 modif FHWA)
            # Simplificado FHWA Tabla 5-3
            if su < 50: alpha_bond = 40
            elif su < 100: alpha_bond = 80
            else: alpha_bond = 120
        
        z_fin = z_acum + row['Espesor_m']
        results.append({
            "z_ini": z_acum, "z_fin": z_fin, "z_mid": (z_acum+z_fin)/2,
            "Tipo": tipo, "N_SPT": n_spt, "Su_kPa": su,
            "Phi_Wolff": phi_1, "Phi_Hatanaka": phi_2, "Phi_Design": phi_prom,
            "E_MPa": E_calc, "Alpha_Bond_kPa": alpha_bond
        })
        z_acum = z_fin
        
    return pd.DataFrame(results)

# ==============================================================================
# CLASES DE CÁLCULO (MICROPILE CORE)
# ==============================================================================
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
            # Buscar propiedad en la capa correspondiente
            alpha = 0
            if z > 0: # Evitar z=0
                for l in self.layers:
                    if l.contains(z): alpha = l.alpha_bond; break
                if z > self.layers[-1].z_bot: alpha = self.layers[-1].alpha_bond
            
            curr_q += alpha * self.perimeter * dz
            q_ult_list.append(curr_q)
            q_adm_list.append(curr_q / self.fs)
            
        return np.array(z_arr), np.array(q_ult_list), np.array(q_adm_list)

def calc_winkler(L, D, EI, kh, V, M, L_casing=0, EI_cas=0):
    # EI Variable si hay casing
    z_nodes = np.linspace(0, L, 200)
    y, m_res, v_res = [], [], []
    
    # Simplificación: Usamos Beta promedio o Beta por tramos. 
    # Para visualización rápida usamos Beta del tramo superior (crítico)
    EI_top = EI_cas if L_casing > 0 else EI
    beta = ((kh * D) / (4 * EI_top))**0.25
    
    for z in z_nodes:
        # Si z > L_casing, la rigidez cambia, pero la solución analítica simple asume EI cte.
        # Aquí mostramos la solución basada en la rigidez SUPERIOR (donde ocurre el Mmax).
        bz = beta * z
        if bz > 10: 
            y.append(0); m_res.append(0); v_res.append(0); continue
            
        exp = np.exp(-bz); sin = np.sin(bz); cos = np.cos(bz)
        A = exp*(cos+sin); B = exp*sin; C = exp*(cos-sin); D_fact = exp*cos
        
        y_val = (2*V*beta/(kh*D))*D_fact + (2*M*beta**2/(kh*D))*C
        m_val = (V/beta)*B + M*A
        v_val = V*C - 2*M*beta*D_fact
        
        y.append(y_val); m_res.append(m_val); v_res.append(v_val)
        
    return z_nodes, np.array(y), np.array(m_res), np.array(v_res), beta

# ==============================================================================
# APLICACIÓN PRINCIPAL
# ==============================================================================
def app_principal():
    with st.sidebar:
        st.success(f"Ingeniero: **{st.session_state['datos_usuario'].get('nombre')}**")
        if st.button("Cerrar Sesión"):
            st.session_state['usuario_registrado'] = False; st.rerun()

    st.title("🏗️ Sistema Avanzado de Diseño de Cimentaciones")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌍 1. Caracterización Geotécnica", 
        "📐 2. Diseño Estructural & Geotécnico", 
        "🧱 3. Diseño de Losa/Cabezal",
        "🚀 4. Optimización (Algoritmo)"
    ])

    # ==========================================================================
    # TAB 1: CARACTERIZACIÓN GEOTÉCNICA
    # ==========================================================================
    with tab1:
        c1, c2 = st.columns([1.5, 1])
        with c1:
            st.subheader("Entrada de Parámetros de Suelo")
            st.info("Defina el perfil estratigráfico. Los parámetros se correlacionan automáticamente.")
            
            # Data Editor Inicial
            default_data = pd.DataFrame([
                {"Espesor_m": 4.0, "Tipo": "Arcilla", "N_SPT": 5, "Su_kPa": 40, "Kh_kNm3": 8000},
                {"Espesor_m": 6.0, "Tipo": "Arena", "N_SPT": 20, "Su_kPa": 0, "Kh_kNm3": 25000},
                {"Espesor_m": 5.0, "Tipo": "Roca", "N_SPT": 50, "Su_kPa": 0, "Kh_kNm3": 100000},
            ])
            edited_df = st.data_editor(default_data, num_rows="dynamic")
            
            # Calcular Correlaciones
            df_geo = calcular_correlaciones(edited_df)
            st.markdown("##### 📊 Parámetros Correlacionados")
            
            # --- CORRECCIÓN AQUI: Formato especifico por columnas para evitar error con texto ---
            st.dataframe(
                df_geo[["z_ini", "z_fin", "Tipo", "Phi_Design", "E_MPa", "Alpha_Bond_kPa"]].style.format({
                    "z_ini": "{:.1f}", "z_fin": "{:.1f}", 
                    "Phi_Design": "{:.1f}", "E_MPa": "{:.1f}", "Alpha_Bond_kPa": "{:.1f}"
                }), 
                use_container_width=True
            )
            
            # Tabla de Referencia
            with st.expander("📚 Ver Tabla de Referencia FHWA (Valores Típicos)"):
                st.markdown("""
                **FHWA NHI-05-039 Table 5-3 (Resumen): Adherencia Grout-Suelo ($\alpha_{bond}$)**
                * *Suelo Granular (Arenas):*
                    * Media Densa (N=10-30): 80 - 150 kPa
                    * Densa (N=30-50): 150 - 250 kPa
                * *Suelo Cohesivo (Arcillas):*
                    * Blanda (Su < 50): 30 - 60 kPa
                    * Rigida (Su > 100): 80 - 120 kPa
                * *Roca:*
                    * Lutita/Pizarra: 200 - 500 kPa
                    * Arenisca/Caliza: 500 - 1000 kPa
                
                [🔗 Referencia: FHWA Manual](https://www.fhwa.dot.gov/engineering/geotech/pubs/05039/)
                """)

        with c2:
            st.subheader("Perfiles de Diseño")
            if not df_geo.empty:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6), sharey=True)
                
                # Gráfica N-SPT
                z_plot = []; n_plot = []; alpha_plot = []
                for _, r in df_geo.iterrows():
                    z_plot.extend([r['z_ini'], r['z_fin']])
                    n_plot.extend([r['N_SPT'], r['N_SPT']])
                    alpha_plot.extend([r['Alpha_Bond_kPa'], r['Alpha_Bond_kPa']])
                
                ax1.plot(n_plot, z_plot, 'b-', lw=2)
                ax1.set_title("N-SPT"); ax1.set_xlabel("Golpes"); ax1.invert_yaxis(); ax1.grid(True, ls=":")
                
                ax2.plot(alpha_plot, z_plot, 'r-', lw=2)
                ax2.set_title("Adherencia (Alpha Bond)"); ax2.set_xlabel("kPa"); ax2.grid(True, ls=":")
                
                st.pyplot(fig)

    # ==========================================================================
    # TAB 2: DISEÑO ESTRUCTURAL & GEOTÉCNICO (SINGLE PILE)
    # ==========================================================================
    with tab2:
        tipo_cim = st.selectbox("Tipo de Cimentación", ["Micropilotes", "Zapatas (Próximamente)", "Caissons (Próximamente)"])
        
        if tipo_cim == "Micropilotes":
            col_in, col_out = st.columns([1, 1.5])
            
            with col_in:
                st.subheader("1. Configuración")
                db_sys = get_dywidag_db()
                sys_sel = st.selectbox("Sistema Refuerzo", db_sys['Sistema'], index=3)
                row_sys = db_sys[db_sys['Sistema'] == sys_sel].iloc[0]
                
                L_tot = st.number_input("Longitud (m)", 5.0, 40.0, 15.0)
                D_perf = st.number_input("Diámetro Perforación (m)", 0.1, 0.4, 0.2)
                FS_geo = st.number_input("FS Geotécnico", 1.5, 3.0, 2.0)
                
                st.markdown("**Cargas Actuantes**")
                P_comp = st.number_input("Compresión (kN)", 0.0, 5000.0, 500.0)
                V_lat = st.number_input("Cortante (kN)", 0.0, 500.0, 30.0)
                M_cab = st.number_input("Momento (kNm)", 0.0, 500.0, 15.0)
                
                st.markdown("**Casing (Tubería)**")
                use_casing = st.checkbox("Incluir Casing")
                L_cas = 0; D_cas_mm = 0; t_cas_mm = 0; fy_cas = 0
                if use_casing:
                    L_cas = st.number_input("Longitud Casing (m)", 1.0, L_tot, 3.0)
                    D_cas_mm = st.number_input("Ø Ext (mm)", value=178.0)
                    t_cas_mm = st.number_input("Espesor (mm)", value=12.7)
                    fy_cas = st.number_input("Fy Casing (MPa)", value=240.0)

            with col_out:
                # --- CÁLCULOS ---
                # 1. Geotecnia
                layers_calc = [SoilLayerObj(r['z_ini'], r['z_fin'], r['Tipo'], r['Alpha_Bond_kPa'], r['Kh_kNm3']) for _, r in df_geo.iterrows()]
                analyzer = MicropileAnalyzer(L_tot, D_perf, layers_calc, FS_geo)
                z_arr, q_ult, q_adm = analyzer.calc_axial()
                
                # 2. Estructural
                As_bar = row_sys['As_mm2']; fy_bar = row_sys['fy_MPa']
                # Capacidades Nominales (AISC/FHWA)
                # Compresión
                A_grout = (np.pi*(D_perf/2)**2) - (As_bar*1e-6)
                P_comp_adm = 0.40 * 30000 * A_grout + 0.47 * (fy_bar*1000) * (As_bar*1e-6) # fc=30MPa asumido
                
                # Lateral
                I_bar = (np.pi*(row_sys['D_ext_mm']/1000)**4)/64
                EI_base = 200e6 * I_bar + (4700*np.sqrt(30)*1000 * ((np.pi*D_perf**4)/64 - I_bar)) # Grout contribution
                
                EI_casing_val = 0; Mn_tot = 0
                
                # Momento Resistente Barra
                Z_bar = ((np.sqrt(4*As_bar/np.pi)/1000)**3)/6
                Mn_bar = Z_bar * fy_bar * 1000
                Mn_tot = 0.9 * Mn_bar
                
                if use_casing:
                    D_ext_m = D_cas_mm/1000; D_int_m = D_ext_m - 2*(t_cas_mm/1000)
                    I_casing_val = np.pi*(D_ext_m**4 - D_int_m**4)/64
                    EI_casing_val = 200e6 * I_casing_val
                    
                    Z_cas = (D_ext_m**3 - D_int_m**3)/6
                    Mn_cas = Z_cas * fy_cas * 1000
                    Mn_tot += 0.9 * Mn_cas

                EI_design = EI_base + EI_casing_val if use_casing else EI_base
                
                # Winkler
                kh_surf = df_geo.iloc[0]['Kh_kNm3']
                z_lat, y_lat, m_lat, v_lat, beta = calc_winkler(L_tot, D_perf, EI_base, kh_surf, V_lat, M_cab, L_cas, EI_design)

                # --- RESULTADOS VISUALES ---
                st.subheader("Resultados de Diseño")
                
                # KPIs
                k1, k2, k3 = st.columns(3)
                delta_geo = "OK" if q_adm[-1] >= P_comp else "FALLA"
                k1.metric("Capacidad Geo. Adm.", f"{q_adm[-1]:.1f} kN", delta=delta_geo)
                k2.metric("Capacidad Est. Comp.", f"{P_comp_adm:.1f} kN")
                k3.metric("Momento Resistente", f"{Mn_tot:.1f} kNm", delta="OK" if Mn_tot > abs(m_lat).max() else "FALLA")
                
                # GRÁFICA CAPACIDAD VS PROFUNDIDAD
                st.markdown("##### 📉 Curvas de Capacidad vs Profundidad")
                fig_cap, ax_cap = plt.subplots(figsize=(10, 5))
                ax_cap.plot(q_adm, z_arr, 'b-', label='Q Admisible', lw=2)
                ax_cap.plot(q_ult, z_arr, 'k--', label='Q Última', alpha=0.6)
                ax_cap.axvline(P_comp, color='r', linestyle=':', label='Carga Actuante')
                
                ax_cap.fill_betweenx(z_arr, 0, q_adm, color='blue', alpha=0.1)
                ax_cap.set_xlabel("Carga Axial (kN)"); ax_cap.set_ylabel("Profundidad (m)")
                ax_cap.invert_yaxis(); ax_cap.legend(); ax_cap.grid(True, ls=':')
                st.pyplot(fig_cap)
                
                # DETALLE ECUACIONES
                with st.expander("Ver Ecuaciones y Referencias Normativas"):
                    st.markdown("### Referencias: FHWA NHI-05-039 & AISC 360-16")
                    c_eq1, c_eq2 = st.columns(2)
                    with c_eq1:
                        st.markdown("**Geotecnia (Bond)**")
                        st.latex(r"Q_{ult} = \sum (\alpha_{bond} \cdot \pi \cdot D_b \cdot \Delta L)")
                        st.latex(r"Q_{all} = Q_{ult} / FS")
                        st.markdown("[🔗 FHWA Manual Link](https://www.fhwa.dot.gov/engineering/geotech/pubs/05039/)")
                    with c_eq2:
                        st.markdown("**Estructural (LRFD/ASD)**")
                        st.latex(r"P_{c,all} = 0.4 f'_c A_g + 0.47 f_y A_s")
                        st.latex(r"M_n = F_y \cdot Z_{plastic}")
                        st.markdown("Winkler (Deflexión):")
                        st.latex(r"y(z) = \frac{2V\beta}{k_h D} D_{\beta z} + \frac{2M\beta^2}{k_h D} C_{\beta z}")

    # ==========================================================================
    # TAB 3: DISEÑO DE LOSA
    # ==========================================================================
    with tab3:
        st.subheader("Diseño de Cabezal / Losa de Repartición")
        col_losa1, col_losa2 = st.columns(2)
        with col_losa1:
            B_x = st.number_input("Ancho X (m)", 1.0, 20.0, 5.0)
            B_y = st.number_input("Largo Y (m)", 1.0, 20.0, 5.0)
            H_losa = st.number_input("Espesor (m)", 0.3, 2.0, 0.6)
            q_app = st.number_input("Sobrecarga (kPa)", 0.0, 100.0, 20.0)
            fc_losa = st.number_input("f'c Losa (MPa)", 21.0, 40.0, 28.0)
        
        with col_losa2:
            # Recuperar capacidad del Tab 2
            if 'q_adm' in locals():
                Q_micro_unit = q_adm[-1]
            else:
                Q_micro_unit = 500.0 # Default
                
            W_pp = H_losa * 24
            Q_tot = (q_app + W_pp) * B_x * B_y
            N_req = int(np.ceil(Q_tot / Q_micro_unit))
            
            st.metric("Carga Total Losa", f"{Q_tot:.1f} kN")
            st.metric("Micropilotes Req.", N_req)
            
            # Punzonamiento
            st.markdown("**Verificación Punzonamiento (ACI 318)**")
            Pu_crit = Q_micro_unit * 1.4 # Factorado aprox
            d = H_losa - 0.075
            b0 = np.pi * (0.2 + d) # Asumiendo D=20cm
            phi_Vc = 0.75 * 0.33 * np.sqrt(fc_losa) * b0 * d * 1000
            
            if Pu_crit < phi_Vc:
                st.success(f"✅ Pasa Cortante: Vu={Pu_crit:.0f} < phiVc={phi_Vc:.0f} kN")
            else:
                st.error(f"❌ Falla Cortante: Aumente espesor")

    # ==========================================================================
    # TAB 4: OPTIMIZACIÓN (EL ALGORITMO SOLICITADO)
    # ==========================================================================
    with tab4:
        st.subheader("🚀 Algoritmo de Optimización Costo/CO2")
        st.info("Este módulo busca la configuración óptima de Diámetro, Longitud y Cantidad.")
        
        # --- PARÁMETROS FIJOS ---
        DIAMETROS_COM = {0.100: 1.00, 0.150: 0.90, 0.200: 0.85} # D: Eficiencia
        LISTA_D = sorted(list(DIAMETROS_COM.keys()))
        MIN_MICROS = 3; MAX_MICROS = 15
        RANGO_L = range(6, 25)
        COSTO_BASE = 100
        # Factores CO2
        F_CO2_CEM = 0.9; F_CO2_ACE = 1.85; F_CO2_PERF = 15.0
        
        c_opt1, c_opt2 = st.columns(2)
        with c_opt1:
            Carga_Total_Ton = st.number_input("Carga Total Grupo (Ton)", 50.0, 500.0, 120.0)
            FS_target = st.number_input("FS Objetivo", 1.5, 3.0, 2.0)
        
        if st.button("Ejecutar Optimización"):
            Carga_kN = Carga_Total_Ton * 9.81
            resultados = []
            
            # Usamos los estratos definidos en Tab 1 para el cálculo
            # Convertir dataframe a lista de dicts para el loop
            estratos_opt = []
            z_curr = 0
            for _, r in df_geo.iterrows():
                estratos_opt.append({
                    "z_fin": r['z_fin'], "qs": r['Alpha_Bond_kPa'], "f_exp": 1.2 # Asumido
                })
            
            bar_prog = st.progress(0)
            
            for idx_d, D in enumerate(LISTA_D):
                bar_prog.progress((idx_d+1)/len(LISTA_D))
                for N in range(MIN_MICROS, MAX_MICROS+1):
                    Q_req_indiv = (Carga_kN / N) * FS_target
                    
                    for L in RANGO_L:
                        # Calc Capacidad
                        Q_cap = 0; z_now = 0
                        vol_grout = 0
                        for e in estratos_opt:
                            if z_now >= L: break
                            z_bot = min(e["z_fin"], L)
                            thick = z_bot - z_now
                            if thick > 0:
                                Q_cap += (np.pi*D*thick) * e["qs"]
                                vol_grout += (np.pi*(D/2)**2 * thick) * e["f_exp"]
                            z_now = z_bot
                        
                        if Q_cap >= Q_req_indiv:
                            # Calcular KPIs
                            eficiencia = DIAMETROS_COM[D]
                            costo = (L * N * COSTO_BASE) / eficiencia
                            
                            # CO2 Simplificado
                            kg_acero = (np.pi*(0.04/2)**2 * 7850 * L * N) # Asumiendo barra 40mm
                            kg_cem = vol_grout * N * 1000 # Asumiendo 1000kg cem/m3
                            co2 = (kg_acero*F_CO2_ACE + kg_cem*F_CO2_CEM + (L*N)*F_CO2_PERF)/1000
                            
                            resultados.append({
                                "D(mm)": int(D*1000), "N": N, "L(m)": L,
                                "Vol_Grout": vol_grout*N, "Costo_Idx": int(costo), "CO2_Ton": co2,
                                "FS_Real": Q_cap / (Carga_kN/N)
                            })
                            break # Encontró L minima para este D y N
            
            bar_prog.empty()
            
            if resultados:
                df_res = pd.DataFrame(resultados).sort_values("Costo_Idx")
                best = df_res.iloc[0]
                
                st.success("✅ Optimización Completada")
                
                k1, k2, k3 = st.columns(3)
                k1.metric("Mejor Config.", f"{best['N']} x Ø{best['D(mm)']}mm")
                k2.metric("Longitud", f"{best['L(m)']} m")
                k3.metric("Huella CO2", f"{best['CO2_Ton']:.1f} Ton")
                
                st.subheader("Top 5 Soluciones")
                st.dataframe(df_res.head(5).style.background_gradient(subset=['Costo_Idx', 'CO2_Ton'], cmap='Greens_r'), use_container_width=True)
                
                # Grafica Dispersión
                fig_opt, ax_opt = plt.subplots()
                sc = ax_opt.scatter(df_res['Costo_Idx'], df_res['CO2_Ton'], c=df_res['L(m)'], cmap='viridis')
                plt.colorbar(sc, label='Longitud (m)')
                ax_opt.set_xlabel('Índice Costo'); ax_opt.set_ylabel('Huella CO2 (Ton)')
                ax_opt.set_title("Espacio de Soluciones")
                st.pyplot(fig_opt)
                
            else:
                st.error("No se encontraron soluciones factibles dentro de los rangos.")

# ==============================================================================
# CONTROL DE FLUJO
# ==============================================================================
if st.session_state['usuario_registrado']:
    app_principal()
else:
    mostrar_registro()
