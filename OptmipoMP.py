import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass
import requests
import time
from io import BytesIO

# --- IMPORTACIONES PDF ---
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except ImportError:
    pass

# ==============================================================================
# CONFIGURACIÓN GLOBAL
# ==============================================================================
st.set_page_config(page_title="Micropile Pro Optimizer", layout="wide", page_icon="🏗️")

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; background-color: #f1f3f6; border-radius: 5px 5px 0px 0px;
        padding: 10px; font-weight: 600;
    }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-top: 3px solid #0047AB; }
    h1, h2, h3 { color: #2C3E50; }
</style>
""", unsafe_allow_html=True)

# VARIABLES DE ESTADO
if 'usuario_registrado' not in st.session_state: st.session_state['usuario_registrado'] = False
if 'datos_usuario' not in st.session_state: st.session_state['datos_usuario'] = {}
if 'df_geo' not in st.session_state: st.session_state['df_geo'] = pd.DataFrame()
if 'opt_result' not in st.session_state: st.session_state['opt_result'] = {} 

# ==============================================================================
# 1. BASE DE DATOS Y CLASES
# ==============================================================================

def get_dywidag_db():
    """Catálogo Dywidag (Solo R y T)."""
    data = {
        "Sistema": ["R32-280", "R32-360", "R38-500", "R51-660", "R51-800", "T76-1200", "T76-1600"],
        "D_ext_mm": [32, 32, 38, 51, 51, 76, 76],
        "As_mm2": [410, 510, 750, 970, 1150, 1610, 1990],
        "fy_MPa": [535, 550, 533, 555, 556, 620, 600]
    }
    return pd.DataFrame(data)

@dataclass
class SoilLayer:
    z_top: float; z_bot: float; tipo: str; alpha: float; kh: float; f_exp: float
    def contains(self, z): return self.z_top <= z <= self.z_bot

def procesar_geotecnia(df):
    """Calcula parámetros correlacionados si no se ingresan manualmente."""
    res = []
    z_acum = 0
    for _, row in df.iterrows():
        try:
            esp = float(row.get('Espesor_m', 0)); tipo = row.get('Tipo', 'Arcilla')
            n = float(row.get('N_SPT', 0)); su = float(row.get('Su_kPa', 0))
            kh = float(row.get('Kh_kNm3', 0)); a_man = float(row.get('Alpha_Manual', 0))
            f_exp = float(row.get('F_Exp', 1.2))
        except: continue

        phi = 0; E_MPa = 0; alpha = 0
        
        # Correlaciones (Wolff, Kulhawy, FHWA)
        if tipo == "Arena":
            phi = ((np.sqrt(20*n)+20) + (27.1+0.3*n-0.00054*n**2))/2
            E_MPa = 1.0 * n
            alpha = min(3.8 * n, 250)
        elif tipo == "Arcilla":
            E_MPa = 0.3 * su
            if su < 25: alpha = 20
            elif su < 50: alpha = 40
            elif su < 100: alpha = 70
            else: alpha = 100
        else: # Roca
            E_MPa = 5000; alpha = 300; f_exp = 1.0
            
        a_fin = a_man if a_man > 0 else alpha
        z_fin = z_acum + esp
        
        res.append({
            "z_ini": z_acum, "z_fin": z_fin, "Espesor_m": esp, "Tipo": tipo,
            "N_SPT": n, "Kh_kNm3": kh, "Alpha_Design": a_fin, 
            "Phi_Deg": phi, "E_MPa": E_MPa, "f_exp": f_exp
        })
        z_acum = z_fin
    return pd.DataFrame(res)

def calc_winkler(L, D, EI, V, M, layers_list):
    """Solución analítica Winkler (Beam on Elastic Foundation)."""
    if not layers_list: return np.array([]), np.array([]), np.array([]), np.array([]), 0
    
    # Tomamos Kh del primer estrato representativo
    kh = layers_list[0]['Kh_kNm3'] if layers_list[0]['Kh_kNm3'] > 0 else 5000
    
    beta = ((kh * D) / (4 * EI))**0.25
    z = np.linspace(0, L, 200)
    y, m_res, v_res = [], [], []
    
    for x in z:
        bz = beta * x
        if bz > 20: 
            y.append(0); m_res.append(0); v_res.append(0); continue
        
        exp = np.exp(-bz); sin = np.sin(bz); cos = np.cos(bz)
        A = exp*(cos+sin); B = exp*sin; C = exp*(cos-sin); D_f = exp*cos
        
        y_val = (2*V*beta/(kh*D))*D_f + (2*M*beta**2/(kh*D))*C
        m_val = (V/beta)*B + M*A
        v_val = V*C - 2*M*beta*D_f
        
        y.append(y_val); m_res.append(m_val); v_res.append(v_val)
        
    return z, np.array(y), np.array(m_res), np.array(v_res), beta

# ==============================================================================
# 3. INTERFAZ PRINCIPAL
# ==============================================================================
def app_principal():
    with st.sidebar:
        st.success(f"Ingeniero: **{st.session_state['datos_usuario'].get('nombre')}**")
        if st.button("Cerrar Sesión"):
            st.session_state['usuario_registrado'] = False; st.rerun()

    st.title("🏗️ Micropile Pro Optimizer")
    
    tab1, tab2, tab3 = st.tabs([
        "🌍 1. Geotecnia Detallada", 
        "🚀 2. Optimizador de Diseño", 
        "📐 3. Diseño Detallado & Reporte"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: GEOTECNIA
    # --------------------------------------------------------------------------
    with tab1:
        c1, c2 = st.columns([1.3, 1])
        with c1:
            st.subheader("1.1 Perfil Estratigráfico")
            st.info("Defina las capas. Deje 'Alpha Manual' en 0 para usar correlación automática.")
            
            df_def = pd.DataFrame([
                {"Espesor_m": 4.0, "Tipo": "Arcilla", "N_SPT": 5, "Su_kPa": 40, "Kh_kNm3": 8000, "Alpha_Manual": 0.0, "F_Exp": 1.1},
                {"Espesor_m": 8.0, "Tipo": "Arena", "N_SPT": 25, "Su_kPa": 0, "Kh_kNm3": 25000, "Alpha_Manual": 0.0, "F_Exp": 1.2},
                {"Espesor_m": 6.0, "Tipo": "Roca", "N_SPT": 50, "Su_kPa": 0, "Kh_kNm3": 90000, "Alpha_Manual": 450.0, "F_Exp": 1.0},
            ])
            
            edited_df = st.data_editor(
                df_def,
                column_config={
                    "Tipo": st.column_config.SelectboxColumn(options=["Arcilla", "Arena", "Roca", "Relleno"]),
                    "F_Exp": st.column_config.NumberColumn("Factor Expansión", min_value=1.0, step=0.1)
                },
                num_rows="dynamic", use_container_width=True
            )
            
            df_geo = procesar_geotecnia(edited_df)
            st.session_state['df_geo'] = df_geo # Guardar para Optimizador
            
            st.markdown("#### Parámetros Calculados")
            st.dataframe(df_geo[["z_ini", "z_fin", "Tipo", "Alpha_Design", "Kh_kNm3"]].style.format("{:.1f}"), use_container_width=True)
            
            st.markdown("#### Ecuaciones de Correlación")
            st.latex(r"\phi'_{arena} = [(\sqrt{20 N} + 20) + (27.1 + 0.3N - 0.00054N^2)] / 2")
            st.latex(r"\alpha_{bond} = \min(3.8 N, 250) \text{ (Arena)} \quad | \quad \alpha_{bond} = f(S_u) \text{ (Arcilla, FHWA)}")

        with c2:
            st.subheader("1.2 Visualización")
            if not df_geo.empty:
                fig, axs = plt.subplots(1, 3, figsize=(10, 6), sharey=True)
                z_max = df_geo['z_fin'].max() + 1
                
                # Arrays para Step Plot
                z_plt, n_plt, a_plt = [], [], []
                for _, r in df_geo.iterrows():
                    z_plt.extend([r['z_ini'], r['z_fin']])
                    n_plt.extend([r['N_SPT'], r['N_SPT']])
                    a_plt.extend([r['Alpha_Design'], r['Alpha_Design']])
                
                # Plot N-SPT
                axs[0].plot(n_plt, z_plt, 'b'); axs[0].set_title("N-SPT"); axs[0].invert_yaxis(); axs[0].grid(True, ls=':')
                # Plot Alpha
                axs[1].plot(a_plt, z_plt, 'r'); axs[1].set_title("Alpha (kPa)"); axs[1].grid(True, ls=':')
                # Plot Perfil Visual
                cols = {"Arcilla": "#D7BDE2", "Arena": "#F9E79F", "Roca": "#AED6F1", "Relleno": "#D3D3D3"}
                for _, r in df_geo.iterrows():
                    rect = patches.Rectangle((0, r['z_ini']), 1, r['Espesor_m'], fc=cols.get(r['Tipo'], 'white'), ec='k')
                    axs[2].add_patch(rect)
                    axs[2].text(0.5, (r['z_ini']+r['z_fin'])/2, r['Tipo'], ha='center', va='center', rotation=90)
                axs[2].set_xlim(0, 1); axs[2].axis('off'); axs[2].set_title("Perfil")
                
                plt.ylim(z_max, 0)
                st.pyplot(fig)

    # --------------------------------------------------------------------------
    # TAB 2: OPTIMIZACIÓN (TU ALGORITMO ORIGINAL)
    # --------------------------------------------------------------------------
    with tab2:
        st.subheader("🚀 Optimizador Multi-Variable")
        
        c_opt1, c_opt2 = st.columns(2)
        with c_opt1:
            CARGA_TON = st.number_input("Carga Total Grupo (Ton)", 50.0, 5000.0, 200.0)
            FS_REQ = st.number_input("FS Geotécnico Objetivo", 1.5, 3.0, 2.0)
        with c_opt2:
            WC_RATIO = st.slider("Relación A/C Lechada", 0.4, 0.6, 0.5)
        
        if st.button("EJECUTAR OPTIMIZACIÓN"):
            if st.session_state['df_geo'].empty:
                st.error("⚠️ Defina estratigrafía en Pestaña 1")
            else:
                # Datos del suelo como lista de dicts para velocidad
                ESTRATOS = []
                for _, r in st.session_state['df_geo'].iterrows():
                    ESTRATOS.append({"z_fin": r['z_fin'], "qs": r['Alpha_Design'], "f_exp": r['f_exp']})
                
                # --- ALGORITMO ORIGINAL ---
                DIAMETROS_COM = {0.100: 1.00, 0.115: 0.95, 0.150: 0.90, 0.200: 0.85} # D: Eficiencia
                LISTA_D = sorted(list(DIAMETROS_COM.keys()))
                MIN_MICROS = 3; MAX_MICROS = 15
                RANGO_L = range(6, 36) # 6 a 35m
                COSTO_PERF_BASE = 100
                
                # Factores Ambientales
                F_CO2_CEM = 0.90; F_CO2_PERF = 15.0; F_CO2_ACERO = 1.85
                DEN_ACERO = 7850.0; DEN_CEM = 3150.0; FY_KPA = 500000.0
                
                CARGA_KN = CARGA_TON * 9.81
                resultados_raw = []
                
                bar = st.progress(0)
                
                for idx, D in enumerate(LISTA_D):
                    bar.progress((idx+1)/len(LISTA_D))
                    for N in range(MIN_MICROS, MAX_MICROS + 1):
                        Q_act_pilote = CARGA_KN / N
                        Q_req_geo = Q_act_pilote * FS_REQ
                        
                        for L in RANGO_L:
                            # 1. Capacidad Geo
                            Q_ult = 0; z_curr = 0
                            for e in ESTRATOS:
                                if z_curr >= L: break
                                z_bot = min(e["z_fin"], L)
                                th = z_bot - z_curr
                                if th > 0: Q_ult += (np.pi * D * th) * e["qs"]
                                z_curr = z_bot
                            
                            if Q_ult >= Q_req_geo:
                                # Solución factible
                                FS_calc = Q_ult / Q_act_pilote
                                
                                # 2. Volúmenes (Grout Expansion)
                                v_exp_tot = 0; z_curr = 0
                                for e in ESTRATOS:
                                    if z_curr >= L: break
                                    z_bot = min(e["z_fin"], L)
                                    th = z_bot - z_curr
                                    if th > 0: v_exp_tot += (np.pi*(D/2)**2 * th) * e["f_exp"]
                                    z_curr = z_bot
                                v_exp_tot *= N
                                
                                # 3. Metricas
                                costo = (L * N * COSTO_PERF_BASE) / DIAMETROS_COM[D]
                                area_acero = Q_act_pilote / FY_KPA
                                peso_acero = area_acero * L * N * DEN_ACERO
                                peso_cem = v_exp_tot * (1.0 / (WC_RATIO/1000.0 + 1.0/DEN_CEM))
                                co2 = (peso_acero*F_CO2_ACERO + peso_cem*F_CO2_CEM + (L*N)*F_CO2_PERF)/1000
                                
                                resultados_raw.append({
                                    "D_mm": int(D*1000), "N": N, "L_m": L, "L_Tot_m": L*N,
                                    "FS": FS_calc, "Vol_Exp": v_exp_tot, "Costo_Idx": int(costo),
                                    "CO2_ton": co2, "Q_act": Q_act_pilote/9.81
                                })
                                break # L minima encontrada para este N y D
                
                bar.empty()
                
                if resultados_raw:
                    df_res = pd.DataFrame(resultados_raw).sort_values("Costo_Idx")
                    best = df_res.iloc[0]
                    st.session_state['opt_result'] = best.to_dict() # Guardar para Tab 3
                    
                    st.success("✅ Optimización Exitosa")
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Configuración", f"{int(best['N'])} x Ø{int(best['D_mm'])}mm")
                    k2.metric("Longitud", f"{int(best['L_m'])} m")
                    k3.metric("Huella CO2", f"{best['CO2_ton']:.1f} Ton")
                    
                    st.dataframe(df_res.head(10).style.background_gradient(subset=['Costo_Idx'], cmap='Greens_r'), use_container_width=True)
                    
                    # Grafica Dispersion
                    fig_sc, ax_sc = plt.subplots(figsize=(8, 4))
                    sc = ax_sc.scatter(df_res['Costo_Idx'], df_res['CO2_ton'], c=df_res['L_m'], cmap='viridis', alpha=0.7)
                    plt.colorbar(sc, label='Longitud (m)')
                    ax_sc.set_xlabel("Índice Costo"); ax_sc.set_ylabel("Huella CO2 (Ton)")
                    ax_sc.grid(True, ls=':')
                    st.pyplot(fig_sc)
                else:
                    st.error("No se encontraron soluciones factibles.")

    # --------------------------------------------------------------------------
    # TAB 3: DISEÑO DETALLADO
    # --------------------------------------------------------------------------
    with tab3:
        st.subheader("📐 Diseño Detallado & Verificaciones")
        
        # Cargar valores optimos o defaults
        opt = st.session_state.get('opt_result', {})
        def_L = float(opt.get('L_m', 12.0)); def_D = float(opt.get('D_mm', 200.0))/1000
        
        c_in, c_out = st.columns([1, 1.5])
        with c_in:
            st.markdown("##### Geometría")
            L = st.number_input("Longitud (m)", 1.0, 50.0, def_L)
            D = st.number_input("Diámetro (m)", 0.1, 0.6, def_D)
            
            st.markdown("##### Refuerzo")
            db = get_dywidag_db()
            sys = st.selectbox("Barra Dywidag", db['Sistema'], index=2)
            row_s = db[db['Sistema'] == sys].iloc[0]
            st.caption(f"As: {row_s['As_mm2']} mm2 | Fy: {row_s['fy_MPa']} MPa")
            fc = st.number_input("f'c Grout (MPa)", 20.0, 60.0, 30.0)
            
            st.markdown("##### Cargas Individuales")
            # Carga por pilote sugerida del optimizador o default
            def_P = float(opt.get('Q_act', 50.0) * 9.81) if opt else 500.0
            P_u = st.number_input("Compresión Pu (kN)", value=def_P)
            V_u = st.number_input("Cortante Vu (kN)", value=30.0)
            M_u = st.number_input("Momento Mu (kNm)", value=15.0)
            
        with c_out:
            if st.session_state['df_geo'].empty:
                st.warning("⚠️ Sin datos de suelo.")
            else:
                # --- CÁLCULOS DETALLADOS ---
                layers_list = st.session_state['df_geo'].to_dict('records')
                
                # 1. Axial (Integración visual)
                z_ax = np.linspace(0, L, 100)
                q_ult = []; curr = 0; perim = np.pi * D
                for z in z_ax:
                    alpha = 0
                    if z > 0:
                        for l in layers_list:
                            if l['z_ini'] <= z <= l['z_fin']: alpha = l['Alpha_Design']; break
                        if z > layers_list[-1]['z_fin']: alpha = layers_list[-1]['Alpha_Design']
                    curr += alpha * perim * (L/100)
                    q_ult.append(curr)
                q_adm = np.array(q_ult) / 2.0 # FS Visual
                
                # 2. Estructural
                As = row_s['As_mm2']; fy = row_s['fy_MPa']
                Ag = (np.pi*(D*1000/2)**2) - As
                P_est = (0.40*fc*Ag + 0.47*fy*As)/1000
                
                # 3. Lateral
                I_bar = (np.pi*(row_s['D_ext_mm']/1000)**4)/64
                EI = 200e6 * I_bar + (4700*np.sqrt(fc)*1000 * ((np.pi*D**4)/64 - I_bar))
                z_lat, y_lat, m_lat, v_lat, beta = calc_winkler(L, D, EI, V_u, M_u, layers_list)
                
                # --- RESULTADOS ---
                k1, k2, k3 = st.columns(3)
                q_geo_val = q_adm[-1]
                k1.metric("Q Admisible Geo", f"{q_geo_val:.1f} kN", delta="OK" if q_geo_val>P_u else "FALLA")
                k2.metric("P Estructural", f"{P_est:.1f} kN")
                k3.metric("Deflexión Máx", f"{max(abs(y_lat))*1000:.1f} mm")
                
                # --- GRÁFICAS ---
                fig_res, (ax_geo, ax_def, ax_mom) = plt.subplots(1, 3, figsize=(12, 6), sharey=True)
                
                # Axial + Suelo
                max_q = max(q_ult)*1.2
                cols = {"Arcilla": "#D7BDE2", "Arena": "#F9E79F", "Roca": "#AED6F1", "Relleno": "#D3D3D3"}
                for l in layers_list:
                    rect = patches.Rectangle((0, l['z_ini']), max_q, l['z_fin']-l['z_ini'], fc=cols.get(l['Tipo'], 'white'), alpha=0.3)
                    ax_geo.add_patch(rect)
                    ax_geo.text(max_q*0.05, (l['z_ini']+l['z_fin'])/2, f"{l['Tipo']}", fontsize=8)
                
                ax_geo.plot(q_adm, z_ax, 'b-', label='Q Adm')
                ax_geo.plot(q_ult, z_ax, 'k--', label='Q Ult')
                ax_geo.axvline(P_u, c='r', ls=':', label='Pu')
                
                rect_mp = patches.Rectangle((max_q*0.8, 0), max_q*0.05, L, fc='gray', ec='k')
                ax_geo.add_patch(rect_mp)
                ax_geo.invert_yaxis(); ax_geo.legend(); ax_geo.set_title("Capacidad Axial"); ax_geo.grid(True, ls=':')
                ax_geo.set_ylabel("Profundidad (m)")
                
                # Deflexión
                ax_def.plot(y_lat*1000, z_lat, 'm'); ax_def.set_title("Deflexión (mm)"); ax_def.grid(True, ls=':')
                
                # Momento/Cortante
                ax_mom.plot(m_lat, z_lat, 'g', label='Momento'); ax_mom.set_title("Momento (kNm)"); ax_mom.grid(True, ls=':')
                ax_mom2 = ax_mom.twiny()
                ax_mom2.plot(v_lat, z_lat, 'orange', ls='--', label='Cortante')
                
                st.pyplot(fig_res)
                
                st.markdown("---")
                st.markdown("#### Ecuaciones de Diseño")
                st.latex(r"Q_{ult} = \sum \alpha_{bond} \cdot \pi D \cdot \Delta L")
                st.latex(r"EI \frac{d^4y}{dz^4} + k_h D y = 0 \quad (\text{Winkler})")
                
                # Botón PDF
                if st.button("Generar PDF Reporte"):
                    try:
                        buffer = BytesIO()
                        p = canvas.Canvas(buffer, pagesize=letter)
                        p.drawString(100, 750, "REPORTE DETALLADO - MICROPILE PRO")
                        p.drawString(100, 730, f"Resultados para Configuración: L={L}m, D={D}m")
                        p.drawString(100, 710, f"Carga Pu: {P_u} kN | Capacidad Geo: {q_geo_val:.1f} kN")
                        p.drawString(100, 690, f"Deflexión Máxima: {max(abs(y_lat))*1000:.1f} mm")
                        p.showPage(); p.save()
                        st.download_button("Descargar", buffer, "Calculo_MP.pdf", "application/pdf")
                    except: st.error("Error generando PDF")

# ==============================================================================
# MAIN (LOGIN)
# ==============================================================================
def mostrar_registro():
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("## 🔒 Ingreso")
        with st.form("reg"):
            nombre = st.text_input("Usuario")
            if st.form_submit_button("Entrar"):
                st.session_state['usuario_registrado'] = True
                st.session_state['datos_usuario'] = {'nombre': nombre}
                st.rerun()

if st.session_state['usuario_registrado']:
    app_principal()
else:
    mostrar_registro()

# Ejecutar
if __name__ == "__main__":
    AppInterface.run()

