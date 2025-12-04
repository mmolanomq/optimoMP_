import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass
import requests
import time

# ==============================================================================
# CONFIGURACIÓN INICIAL (GLOBAL)
# ==============================================================================
st.set_page_config(page_title="Diseño Micropilotes Pro", layout="wide", page_icon="🏗️")

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

# ------------------------------------------------------------------
# ⚠️ IMPORTANTE: PEGA AQUÍ TU URL DE GOOGLE APPS SCRIPT
# ------------------------------------------------------------------
GOOGLE_SCRIPT_URL = "PEGAR_AQUI_TU_URL_DE_GOOGLE_APPS_SCRIPT" 

# ==============================================================================
# 1. SISTEMA DE GESTIÓN DE USUARIOS
# ==============================================================================
if 'usuario_registrado' not in st.session_state:
    st.session_state['usuario_registrado'] = False
if 'datos_usuario' not in st.session_state:
    st.session_state['datos_usuario'] = {}

def enviar_a_google_sheets(datos):
    """Envía los datos al Webhook de Google Apps Script"""
    if "PEGAR_AQUI" in GOOGLE_SCRIPT_URL:
        return True # Modo simulación si no hay URL configurada
    try:
        response = requests.post(GOOGLE_SCRIPT_URL, json=datos)
        return response.status_code == 200
    except:
        return False

def mostrar_registro():
    """Pantalla de Login/Registro"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("## 🔒 Acceso a Herramienta de Ingeniería")
        st.info("Bienvenido. Para acceder a la suite de cálculo FHWA, por favor identifíquese.")
        
        with st.form("login_form"):
            nombre = st.text_input("Nombre Completo")
            c1, c2 = st.columns(2)
            empresa = c1.text_input("Empresa / Organización")
            cargo = c2.selectbox("Rol", ["Ingeniero Geotecnista", "Ingeniero Estructural", "Estudiante", "Otro"])
            email = st.text_input("Email Corporativo")
            
            if st.form_submit_button("🚀 INGRESAR AL SISTEMA", type="primary"):
                if nombre and empresa and email:
                    datos = {
                        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "nombre": nombre, "empresa": empresa, 
                        "cargo": cargo, "email": email
                    }
                    
                    with st.spinner("Validando credenciales..."):
                        enviar_a_google_sheets(datos)
                        st.session_state['datos_usuario'] = datos
                        st.session_state['usuario_registrado'] = True
                        st.rerun()
                else:
                    st.error("Por favor complete todos los campos.")

# ==============================================================================
# 2. LÓGICA DE INGENIERÍA (BASE DE DATOS Y CLASES)
# ==============================================================================

def get_dywidag_db():
    """Base de datos corregida Dywidag/Ischebeck (R51-800 Area=1150mm2)."""
    data = {
        "Sistema": [
            "R32-210 (R32L)", "R32-280 (R32N)", "R32-320", "R32-360 (R32S)",
            "R38-420", "R38-500 (R38N)", "R38-550",
            "R51-550 (R51L)", "R51-660", "R51-800 (R51N)",
            "T76-1200", "T76-1600", "T76-1900",
            "Titan 30/11", "Titan 40/16", "Titan 52/26"
        ],
        "D_ext_bar_mm": [
            32, 32, 32, 32, 38, 38, 38, 51, 51, 51, 76, 76, 76, 30, 40, 52
        ],
        "As_mm2": [
            340, 410, 470, 510, 660, 750, 800, 890, 970, 1150, 
            1610, 1990, 2360, 532, 960, 1590
        ],
        "fy_MPa": [
            470, 535, 530, 550, 530, 533, 560, 505, 555, 556, 
            620, 600, 635, 540, 560, 580
        ]
    }
    return pd.DataFrame(data)

@dataclass
class SoilLayer:
    z_top: float; z_bot: float; tipo: str; alpha_bond: float; kh: float
    def contains(self, z): return self.z_top <= z <= self.z_bot

class MicropileCore:
    def __init__(self, L, D, layers, fs_geo):
        self.L = L; self.D = D; self.layers = layers; self.fs_geo = fs_geo
        self.perimeter = np.pi * D

    def get_layer_prop(self, z, prop):
        if z < 0: return 0.0
        if z > self.layers[-1].z_bot: return getattr(self.layers[-1], prop)
        for l in self.layers:
            if l.contains(z): return getattr(l, prop)
        return getattr(self.layers[-1], prop)

    def calc_axial_capacity(self):
        dz = 0.05; z_arr = np.arange(0, self.L + dz, dz)
        q_ult_accum = []; current_q = 0.0; alphas = []
        for z in z_arr:
            alpha = self.get_layer_prop(z, 'alpha_bond')
            alphas.append(alpha)
            current_q += alpha * self.perimeter * dz
            q_ult_accum.append(current_q)
        q_ult = np.array(q_ult_accum); q_adm = q_ult / self.fs_geo
        return {
            'z': z_arr, 'alpha': np.array(alphas),
            'Q_ult_profile': q_ult, 'Q_adm_profile': q_adm,
            'Q_ult_total': q_ult[-1], 'Q_adm_total': q_adm[-1]
        }

def calc_winkler_lateral(L, D, EI, kh_ref, V_load, M_load):
    beta = ((kh_ref * D) / (4 * EI))**0.25
    z_nodes = np.linspace(0, L, 200)
    y_list, M_list, V_list = [], [], []
    for z in z_nodes:
        bz = beta * z
        if bz > 15:
            y_list.append(0); M_list.append(0); V_list.append(0); continue
        exp_beta = np.exp(-bz); sin_b, cos_b = np.sin(bz), np.cos(bz)
        A = exp_beta * (cos_b + sin_b); B = exp_beta * sin_b
        C = exp_beta * (cos_b - sin_b); D_fact = exp_beta * cos_b
        y = (2 * V_load * beta / (kh_ref * D)) * D_fact + (2 * M_load * beta**2 / (kh_ref * D)) * C
        M = (V_load / beta) * B + M_load * A
        V = V_load * C - 2 * M_load * beta * D_fact
        y_list.append(y); M_list.append(M); V_list.append(V)
    return z_nodes, np.array(y_list), np.array(M_list), np.array(V_list), beta

# ==============================================================================
# 3. APLICACIÓN PRINCIPAL (SE EJECUTA TRAS EL LOGIN)
# ==============================================================================
def app_principal():
    # --- BARRA LATERAL ---
    with st.sidebar:
        st.success(f"👤 **{st.session_state['datos_usuario'].get('nombre')}**")
        if st.button("Cerrar Sesión"):
            st.session_state['usuario_registrado'] = False; st.rerun()
        st.markdown("---")
        
        st.header("1. Materiales")
        df_sys = get_dywidag_db()
        sys_sel = st.selectbox("Sistema Refuerzo:", df_sys['Sistema'], index=9)
        row_sys = df_sys[df_sys['Sistema'] == sys_sel].iloc[0]
        fy_bar = row_sys['fy_MPa']; As_bar = row_sys['As_mm2']
        st.info(f"As: {As_bar} mm² | Fy: {fy_bar} MPa")
        
        fc_grout = st.number_input("f'c Grout (MPa)", 20.0, 60.0, 30.0)
        st.divider()
        L_tot = st.number_input("Longitud (m)", 1.0, 50.0, 12.0)
        D_perf = st.number_input("Ø Perforación (m)", 0.1, 0.6, 0.20)
        
        st.header("2. Estratigrafía")
        default_soil = pd.DataFrame([
            {"z_top": 0.0, "z_bot": 5.0, "tipo": "Arcilla", "alpha_bond": 60.0, "kh_kN_m3": 8000.0},
            {"z_top": 5.0, "z_bot": 10.0, "tipo": "Arena", "alpha_bond": 120.0, "kh_kN_m3": 18000.0},
            {"z_top": 10.0, "z_bot": 15.0, "tipo": "Roca", "alpha_bond": 350.0, "kh_kN_m3": 50000.0},
        ])
        edited_soil = st.data_editor(default_soil, num_rows="dynamic")
        layers_objs = [SoilLayer(r['z_top'], r['z_bot'], r['tipo'], r['alpha_bond'], r['kh_kN_m3']) for _, r in edited_soil.iterrows()]

    st.title("🛡️ Diseño de Micropilotes - FHWA NHI-05-039")
    tab1, tab2, tab3 = st.tabs(["🏗️ Capacidad Axial", "📉 Lateral & Casing", "🧱 Losa Cabezal"])

    # --- TAB 1: AXIAL ---
    with tab1:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Cargas Axiales")
            FS_geo = st.number_input("FS Geotécnico", 1.0, 4.0, 2.0)
            P_act = st.number_input("Carga Compresión (kN)", 0.0, 5000.0, 450.0)
            
            # Cálculos
            mp_core = MicropileCore(L_tot, D_perf, layers_objs, FS_geo)
            res_geo = mp_core.calc_axial_capacity()
            
            A_grout = (np.pi*(D_perf/2)**2) - (As_bar*1e-6)
            P_c_all = (0.40*(fc_grout*1000)*A_grout) + (0.47*(fy_bar*1000)*(As_bar*1e-6))
            P_t_all = 0.55*(fy_bar*1000)*(As_bar*1e-6)
            
            k1, k2, k3 = st.columns(3)
            delta = "OK" if res_geo['Q_adm_total'] >= P_act else "FALLA" if P_act > 0 else None
            k1.metric("Q Admisible Geo", f"{res_geo['Q_adm_total']:.1f} kN", delta=delta)
            k2.metric("P Compresión Est", f"{P_c_all:.1f} kN")
            k3.metric("P Tensión Est", f"{P_t_all:.1f} kN")

        with c2:
            st.subheader("Perfil y Esquema")
            fig, ax = plt.subplots(figsize=(4, 6))
            colors = {"Arcilla": "#D2B48C", "Arena": "#F4A460", "Roca": "#808080"}
            max_d = max(L_tot+2, layers_objs[-1].z_bot)
            
            for l in layers_objs:
                h = l.z_bot - l.z_top
                rect = patches.Rectangle((0, l.z_top), 4, h, facecolor=colors.get(l.tipo, "#eee"), alpha=0.5)
                ax.add_patch(rect)
                ax.text(3.5, l.z_top+h/2, f"{l.tipo}\n$\\alpha$={l.alpha_bond}", ha='center', va='center', rotation=90, fontsize=7)
                
            rect_m = patches.Rectangle((1.8, 0), 0.4, L_tot, facecolor='#708090', edgecolor='k')
            ax.add_patch(rect_m)
            ax.plot([2, 2], [0, L_tot], 'r--', lw=1)
            ax.set_ylim(max_d, 0); ax.axis('off')
            st.pyplot(fig)

        st.divider()
        st.subheader("Gráficas Profundidad")
        f2, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        ax_a.step(res_geo['alpha'], res_geo['z'], 'brown', where='post'); ax_a.set_title("Alpha Bond (kPa)"); ax_a.invert_yaxis(); ax_a.grid(True, ls=':')
        ax_b.plot(res_geo['Q_adm_profile'], res_geo['z'], 'b-', label='Q adm'); ax_b.set_title("Capacidad (kN)"); ax_b.legend(); ax_b.grid(True, ls=':')
        st.pyplot(f2)

    # --- TAB 2: LATERAL ---
    with tab2:
        c_l1, c_l2 = st.columns(2)
        with c_l1:
            usar_casing = st.checkbox("Incluir Casing Permanente", False)
            D_cas, t_cas, fy_cas, L_cas = 0, 0, 0, 0
            if usar_casing:
                cc1, cc2 = st.columns(2)
                D_cas = cc1.number_input("Ø Ext (mm)", value=178.0)
                t_cas = cc2.number_input("Espesor (mm)", value=12.7)
                cc3, cc4 = st.columns(2)
                fy_cas = cc3.number_input("Fy Casing (MPa)", value=240.0)
                L_cas = cc4.number_input("Longitud Casing (m)", value=3.0)
        with c_l2:
            V_lat = st.number_input("Carga Lateral Vu (kN)", value=30.0)
            M_top = st.number_input("Momento Mu (kNm)", value=10.0)
            
        # Cálculos Lateral
        I_bar = (np.pi*(row_sys['D_ext_bar_mm']/1000)**4)/64
        I_grout = (np.pi*D_perf**4)/64 - I_bar
        E_s = 200e6; E_g = 4700*np.sqrt(fc_grout)*1000
        EI_eff = (E_s*I_bar) + (E_g*I_grout)
        
        Mn_tot, Vn_tot = 0, 0
        if usar_casing:
            D_ext_m, t_m = D_cas/1000, t_cas/1000
            D_int_m = D_ext_m - 2*t_m
            I_cas = (np.pi*(D_ext_m**4 - D_int_m**4))/64
            EI_eff += E_s * I_cas
            
            Z_cas = (D_ext_m**3 - D_int_m**3)/6
            Mn_casing = Z_cas * fy_cas * 1000
            Vn_casing = 0.6 * fy_cas * 1000 * (np.pi/4)*(D_ext_m**2 - D_int_m**2) * 0.5
            Mn_tot += 0.9 * Mn_casing
            Vn_tot += 0.9 * Vn_casing
            
        # Aporte Barra
        d_b = np.sqrt(4*As_bar/np.pi)/1000
        Mn_bar = (d_b**3/6) * fy_bar * 1000
        Vn_bar = 0.6 * fy_bar * 1000 * As_bar * 1e-6
        Mn_tot += 0.9 * Mn_bar; Vn_tot += 0.9 * Vn_bar
        
        kh = layers_objs[0].kh
        z_lat, y_lat, M_lat, V_lat_arr, beta = calc_winkler_lateral(L_tot, D_perf, EI_eff, kh, V_lat, M_top)
        
        # Resultados
        st.divider()
        st.subheader("Verificación Estructural")
        
        if usar_casing:
             L_crit = 4/beta
             if L_cas < L_crit: st.warning(f"⚠️ Casing corto ({L_cas}m < {L_crit:.1f}m). Carga se transfiere a zona débil.")
        
        col_res, col_eq = st.columns(2)
        with col_res:
            M_max = np.max(np.abs(M_lat)); V_max = np.max(np.abs(V_lat_arr))
            df_r = pd.DataFrame({
                "Tipo": ["Momento", "Cortante"],
                "Actuante": [f"{M_max:.1f} kNm", f"{V_max:.1f} kN"],
                "Capacidad": [f"{Mn_tot:.1f} kNm", f"{Vn_tot:.1f} kN"],
                "Estado": ["✅" if M_max < Mn_tot else "❌", "✅" if V_max < Vn_tot else "❌"]
            })
            st.dataframe(df_r, hide_index=True)
            st.metric("Deflexión Máx", f"{np.max(np.abs(y_lat))*1000:.1f} mm")
            
        with col_eq:
            st.markdown("##### Memoria de Cálculo")
            st.latex(r"M_n = F_y \cdot Z_{plas}"); st.latex(r"V_n = 0.6 F_y A_{eff}")
            st.latex(r"y(z) = \frac{2 V \beta}{k_h D} D_{\beta z} + \frac{2 M \beta^2}{k_h D} C_{\beta z}")
            st.caption("Referencias: AISC 360-16 (Caps F, G) y FHWA NHI-05-039.")
            
        # Gráficas Lateral
        f3, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
        ax1.plot(y_lat*1000, z_lat, 'm'); ax1.set_title("Deflexión (mm)"); ax1.invert_yaxis(); ax1.grid(True, ls=':')
        ax2.plot(M_lat, z_lat, 'g'); ax2.set_title("Momento (kNm)"); ax2.axvline(Mn_tot, c='r', ls='--'); ax2.grid(True, ls=':')
        ax3.plot(V_lat_arr, z_lat, 'b'); ax3.set_title("Cortante (kN)"); ax3.axvline(Vn_tot, c='orange', ls='--'); ax3.grid(True, ls=':')
        if usar_casing: 
            for ax in [ax1, ax2, ax3]: ax.axhline(L_cas, c='k', ls='--', label='Casing')
        st.pyplot(f3)

    # --- TAB 3: LOSA ---
    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            Bx = st.number_input("Ancho Losa (m)", 1.0, 50.0, 10.0)
            By = st.number_input("Largo Losa (m)", 1.0, 50.0, 10.0)
            H = st.number_input("Espesor (m)", 0.2, 2.0, 0.5)
            fcl = st.number_input("f'c Losa (MPa)", 21.0, 35.0, 28.0)
        with c2:
            q_sc = st.number_input("Sobrecarga (kPa)", 0.0, 50.0, 20.0)
        
        W_pp = H * 24; Q_tot = (q_sc + W_pp) * Bx * By
        Q_adm = res_geo['Q_adm_total']
        
        st.subheader("1. Distribución")
        if Q_adm > 0:
            N = int(np.ceil(Q_tot / Q_adm))
            st.write(f"Carga Total: **{Q_tot:.1f} kN**")
            st.metric("Micropilotes Requeridos", N)
            nx = int(np.sqrt(N*By/Bx)); ny = int(np.ceil(N/nx)) if nx else 1
            st.info(f"Grid Sugerido: {nx} x {ny}")
            
        st.subheader("2. Punzonamiento")
        Pu = st.number_input("Pu Crítica (kN)", value=Q_adm * 1.4)
        d = H - 0.075; D_act = (D_cas/1000) if usar_casing else D_perf
        bo = np.pi * (D_act + d)
        phi_Vc = 0.75 * 0.33 * np.sqrt(fcl) * bo * d * 1000
        
        r1, r2 = st.columns(2)
        r1.metric("Capacidad phi*Vc", f"{phi_Vc:.1f} kN")
        r2.metric("Demanda Pu", f"{Pu:.1f} kN")
        if Pu < phi_Vc: st.success("✅ Pasa Punzonamiento")
        else: st.error("❌ Falla Punzonamiento")

    # FOOTER
    st.markdown("---")
    st.caption("Referencias: FHWA NHI-05-039 | AISC 360-16 | Catálogos Dywidag/Ischebeck")

# ==============================================================================
# CONTROL DE FLUJO
# ==============================================================================
if st.session_state['usuario_registrado']:
    app_principal()
else:
    mostrar_registro()
