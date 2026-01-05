import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO

# ==============================================================================
# 0. CONFIGURACIÓN E INICIALIZACIÓN
# ==============================================================================
st.set_page_config(
    page_title="MicroPile Opt V3",
    layout="wide",
    page_icon="🏗️",
    initial_sidebar_state="expanded"
)

# Estilos CSS para mejorar la UI
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; }
    h1 { color: #1e3a8a; font-weight: 800; }
    h2 { color: #1e40af; font-size: 1.5rem; border-bottom: 2px solid #e5e7eb; padding-bottom: 0.5rem; }
    h3 { color: #374151; font-size: 1.2rem; font-weight: 600; }
    .stButton>button { width: 100%; border-radius: 6px; font-weight: bold; height: 3rem; }
    .stButton>button[kind="primary"] { background-color: #2563eb; border: none; }
    .stButton>button[kind="primary"]:hover { background-color: #1d4ed8; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; color: #0f172a; }
    div[data-testid="stMetricLabel"] { font-size: 0.9rem; color: #64748b; }
    .info-box { background-color: #eff6ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #3b82f6; margin-bottom: 1rem; }
    a { text-decoration: none; font-weight: bold; color: #2563eb; }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTES GLOBALES ---
DIAMETROS_COM = {100: 1.00, 115: 0.95, 130: 0.93, 150: 0.90, 200: 0.85}
LISTA_D = sorted(list(DIAMETROS_COM.keys()))
COSTO_PERF_BASE = 100
FACTOR_CO2_CEMENTO = 0.90
FACTOR_CO2_PERF = 15.0
FACTOR_CO2_ACERO = 1.85
DENSIDAD_ACERO = 7850.0
DENSIDAD_CEMENTO = 3150.0
FY_ACERO_KPA = 500000.0

# --- ESTADO DE SESIÓN ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if 'layers' not in st.session_state:
    st.session_state['layers'] = [
        {"name": "Relleno / Arcilla Blanda", "thickness": 3.0, "qs": 40.0, "f_exp": 1.1, "color": "#dbeafe"},
        {"name": "Arcilla Firme / Limo", "thickness": 5.0, "qs": 80.0, "f_exp": 1.2, "color": "#fef3c7"},
        {"name": "Estrato Resistente", "thickness": 10.0, "qs": 150.0, "f_exp": 1.3, "color": "#fee2e2"}
    ]

if 'global_results' not in st.session_state:
    st.session_state['global_results'] = None

if 'selected_indices' not in st.session_state:
    st.session_state['selected_indices'] = []

# ==============================================================================
# 1. PANTALLA DE LOGIN
# ==============================================================================
def login_screen():
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<div style='text-align: center;'><h1>🔒 Acceso de Ingeniería</h1><p>Sistema de Optimización de Micropilotes</p></div>", unsafe_allow_html=True)
            
            with st.form("login_form"):
                nombre = st.text_input("Nombre Completo", placeholder="Ej. Juan Pérez")
                email = st.text_input("Correo Electrónico", placeholder="nombre@empresa.com")
                empresa = st.text_input("Empresa / Proyecto")
                cargo = st.selectbox("Cargo", ["Ingeniero Geotecnista", "Ingeniero Estructural", "Constructor/Residente", "Estudiante"])
                acepto = st.checkbox("Acepto los términos de uso técnico y registro.")
                
                st.markdown("<br>", unsafe_allow_html=True)
                submitted = st.form_submit_button("🚀 INGRESAR AL SISTEMA", type="primary")
                
                if submitted:
                    if nombre and email and empresa and acepto:
                        st.session_state['logged_in'] = True
                        st.session_state['user_info'] = {'nombre': nombre, 'email': email, 'cargo': cargo}
                        st.rerun()
                    else:
                        st.error("Por favor complete todos los campos y acepte los términos.")

# ==============================================================================
# 2. FUNCIONES GRÁFICAS
# ==============================================================================
def draw_integrated_model(layers, spt_data, k_factor, water_table):
    total_depth = sum(l['thickness'] for l in layers)
    max_depth = max(total_depth, 15) * 1.1
    
    z_spt = [d['z'] for d in spt_data]
    n_spt = [d['n'] for d in spt_data]
    qs_est = [min(d['n'] * k_factor, 300) for d in spt_data]

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(10, 8), gridspec_kw={'width_ratios': [1, 1, 1]}, sharey=True)
    plt.subplots_adjust(wspace=0.15)

    # Estratigrafía
    current_depth = 0
    for layer in layers:
        rect = patches.Rectangle((0, current_depth), 1, layer['thickness'], linewidth=0.5, edgecolor='gray', facecolor=layer['color'])
        ax0.add_patch(rect)
        mid_y = current_depth + layer['thickness']/2
        ax0.text(0.5, mid_y, f"{layer['name']}\nH={layer['thickness']}m\nQs={int(layer['qs'])}", 
                ha='center', va='center', fontsize=8, color='#1e293b', fontweight='bold', wrap=True)
        current_depth += layer['thickness']
        ax0.axhline(y=current_depth, color='gray', linestyle=':', linewidth=0.5)
        ax0.text(1.05, current_depth, f"{current_depth:.1f}m", va='center', fontsize=7)

    if water_table > 0:
        ax0.axhline(y=water_table, color='blue', linestyle='--', linewidth=2)
        ax0.text(0.9, water_table - 0.2, "N.F.", color='blue', fontsize=8, fontweight='bold', ha='right')

    ax0.set_ylim(max_depth, 0)
    ax0.set_xlim(0, 1)
    ax0.set_xticks([])
    ax0.set_title("Estratigrafía", fontsize=10, fontweight='bold')
    ax0.set_ylabel("Profundidad (m)")

    # N-SPT
    ax1.plot(n_spt, z_spt, 'o-', color='#2563eb', linewidth=2, markersize=5)
    ax1.set_xlabel("N (golpes/pie)")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_title("N-SPT", fontsize=10, fontweight='bold', color='#1e40af')
    ax1.set_xlim(0, 60)

    # Qs
    ax2.plot(qs_est, z_spt, 's-', color='#dc2626', linewidth=2, markersize=5)
    ax2.set_xlabel("Adherencia (kPa)")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_title(f"Qs Est. (K={k_factor})", fontsize=10, fontweight='bold', color='#991b1b')
    ax2.set_xlim(0, 350)

    return fig

def draw_load_transfer(layers, results, fs_req):
    if results is None or results.empty: return None
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#2563eb', '#16a34a', '#dc2626', '#d97706', '#9333ea']
    max_depth, max_q = 0, 0

    # Fondo estratos
    current_depth = 0
    total_strat_depth = sum(l['thickness'] for l in layers)
    for layer in layers:
        ax.axhspan(current_depth, current_depth + layer['thickness'], color=layer['color'], alpha=0.3)
        current_depth += layer['thickness']

    # Curvas
    for i, (_, row) in enumerate(results.iterrows()):
        L, D_mm, N = row['L'], row['D_mm'], int(row['N'])
        D_m = D_mm / 1000.0
        z_points, q_points = [0], [0]
        curr_z, curr_q, acc_depth = 0, 0, 0

        for layer in layers:
            if curr_z >= L: break
            layer_bot = acc_depth + layer['thickness']
            start = max(0, acc_depth)
            end = min(L, layer_bot)
            seg_len = max(0, end - start)
            acc_depth += layer['thickness']
            
            if seg_len > 0:
                d_eff = D_m * layer['f_exp']
                q_seg = (np.pi * d_eff * seg_len * layer['qs']) / fs_req
                curr_q += q_seg
                curr_z += seg_len
                z_points.append(curr_z)
                q_points.append(curr_q)
        
        ax.plot(q_points, z_points, marker='o', markersize=3, label=f"{N}xØ{int(D_mm)} (L={L}m)", color=colors[i % len(colors)], linewidth=2)
        max_depth = max(max_depth, L)
        max_q = max(max_q, curr_q)

    ax.set_ylim(max(max_depth + 2, total_strat_depth), 0)
    ax.set_xlim(0, max_q * 1.1)
    ax.set_xlabel("Capacidad Admisible Acumulada (Ton)")
    ax.set_ylabel("Profundidad (m)")
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.7)
    return fig

def draw_pile_cap(config, rank):
    N, D = int(config['N']), config['D_mm'] / 1000.0
    S, Edge = max(0.75, 3 * D), max(0.30, 1.5 * D)
    
    if N == 1: cols, rows = 1, 1
    elif N == 2: cols, rows = 2, 1
    elif N == 3: cols, rows = 3, 1
    elif N == 4: cols, rows = 2, 2
    elif N <= 6: cols, rows = 3, 2
    elif N <= 9: cols, rows = 3, 3
    else: cols = int(np.ceil(np.sqrt(N))); rows = int(np.ceil(N/cols))
    
    W = (cols - 1) * S + 2 * Edge
    L = (rows - 1) * S + 2 * Edge
    H, Vol = 0.50 + (0.1 * (rows-1)), W * L * (0.50 + (0.1 * (rows-1)))
    
    fig, ax = plt.subplots(figsize=(3, 3))
    rect = patches.Rectangle((0, 0), W, L, facecolor='#e2e8f0', edgecolor='#475569', linewidth=2)
    ax.add_patch(rect)
    
    for i in range(N):
        r_idx, c_idx = i // cols, i % cols
        cx, cy = Edge + c_idx * S, Edge + r_idx * S
        circle = patches.Circle((cx, cy), D/2, facecolor='#1e293b', edgecolor='black')
        ax.add_patch(circle)
    
    ax.text(W/2, -0.2, f"{W:.2f}m", ha='center', fontsize=9, color='#64748b')
    ax.text(-0.2, L/2, f"{L:.2f}m", va='center', rotation=90, fontsize=9, color='#64748b')

    ax.set_xlim(-0.5, W + 0.5)
    ax.set_ylim(-0.5, L + 0.5)
    ax.axis('off')
    ax.set_aspect('equal')
    return fig, Vol, W, L, H, S

# ==============================================================================
# 3. LÓGICA DE OPTIMIZACIÓN
# ==============================================================================
def run_optimization(load_ton, fs_req, wc_ratio, min_n, max_n, min_d, max_d, layers):
    carga_req_kn = load_ton * 9.81
    solutions = []
    valid_diameters = [d for d in LISTA_D if min_d <= d <= max_d]
    
    for D_mm in valid_diameters:
        D_m = D_mm / 1000.0
        eficiencia = DIAMETROS_COM.get(D_mm, 0.85)
        for N in range(min_n, max_n + 1):
            q_act_pilote = carga_req_kn / N
            q_req_geo = q_act_pilote * fs_req
            for L in np.arange(5.0, 40.5, 0.5):
                q_ult, vol_exp, acc_depth, area_perf = 0, 0, 0, np.pi * (D_m/2)**2
                
                for layer in layers:
                    top, bot = acc_depth, acc_depth + layer['thickness']
                    acc_depth += layer['thickness']
                    start, end = max(0, top), min(L, bot)
                    seg = max(0, end - start)
                    if seg > 0:
                        d_eff = D_m * layer['f_exp']
                        q_ult += (np.pi * d_eff * seg) * layer['qs']
                        vol_exp += (area_perf * seg) * layer['f_exp']
                
                if L > acc_depth:
                    extra = L - acc_depth
                    last = layers[-1]
                    d_eff = D_m * last['f_exp']
                    q_ult += (np.pi * d_eff * extra) * last['qs']
                    vol_exp += (area_perf * extra) * last['f_exp']
                
                if q_ult >= q_req_geo:
                    vol_total = vol_exp * N
                    costo = (L * N * COSTO_PERF_BASE) / eficiencia
                    vol_acero = (q_act_pilote / FY_ACERO_KPA) * L * N
                    peso_acero = vol_acero * DENSIDAD_ACERO
                    peso_cemento = max(0, vol_total - vol_acero) * (1000 / (wc_ratio + 1/3.15))
                    co2 = (peso_acero * FACTOR_CO2_ACERO + peso_cemento * FACTOR_CO2_CEMENTO + (L*N) * FACTOR_CO2_PERF) / 1000
                    
                    solutions.append({
                        "D_mm": D_mm, "N": N, "L": L, "Perf_Total": L*N, "FS": q_ult / q_act_pilote,
                        "Q_adm": q_ult / fs_req / 9.81, "Q_act": q_act_pilote / 9.81, "Vol_Grout": vol_total,
                        "CO2": co2, "Costo_Idx": costo
                    })
                    break 
    if not solutions: return pd.DataFrame()
    return pd.DataFrame(solutions).sort_values("Costo_Idx")

# ==============================================================================
# 4. APP PRINCIPAL
# ==============================================================================
def main():
    if not st.session_state['logged_in']:
        login_screen()
        return

    with st.sidebar:
        st.info(f"👤 **{st.session_state['user_info']['nombre']}**\n\n{st.session_state['user_info']['cargo']}")
        if st.button("Cerrar Sesión"):
            st.session_state['logged_in'] = False
            st.rerun()

    st.title("🏗️ Optimizador de Micropilotes")
    tab_geo, tab_design, tab_caps = st.tabs(["1. Info Geotécnica", "2. Diseño & Cálculo", "3. Dados / Cabezales"])

    with tab_geo:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("### 📥 Datos de Entrada")
            if 'spt_df' not in st.session_state:
                st.session_state['spt_df'] = pd.DataFrame([{"z": 1.5, "n": 4}, {"z": 3.0, "n": 7}, {"z": 4.5, "n": 12}, {"z": 6.0, "n": 15}, {"z": 7.5, "n": 22}, {"z": 9.0, "n": 28}, {"z": 10.5, "n": 35}, {"z": 12.0, "n": 42}, {"z": 15.0, "n": 50}])
            st.session_state['spt_df'] = st.data_editor(st.session_state['spt_df'], num_rows="dynamic", hide_index=True)
            k_val = st.slider("Factor K (Correlación)", 1.0, 10.0, 3.5, 0.5)
            nf_val = st.number_input("Nivel Freático (m)", 0.0, 50.0, 2.0, 0.5)
            
            st.divider()
            st.markdown("**2. Definición de Estratos**")
            
            # --- CORRECCIÓN ROBUSTA PARA COLUMN CONFIG ---
            # Construimos la configuración de columnas dinámicamente
            # Si st.column_config no existe, usamos None para que st.data_editor use defaults
            
            cols_cfg = None
            if hasattr(st, "column_config"):
                cols_cfg = {
                    "name": "Nombre",
                    "thickness": st.column_config.NumberColumn("H (m)", min_value=0.1, format="%.1f"),
                    "qs": st.column_config.NumberColumn("Qs (kPa)", min_value=0),
                    "f_exp": st.column_config.NumberColumn("F.Exp", min_value=1.0, max_value=3.0, step=0.1)
                }
                # Intentamos añadir ColorColumn con fallback seguro
                if hasattr(st.column_config, "ColorColumn"):
                    cols_cfg["color"] = st.column_config.ColorColumn("Color")
                else:
                    cols_cfg["color"] = st.column_config.TextColumn("Color")

            layers_df = pd.DataFrame(st.session_state['layers'])
            edited_layers = st.data_editor(
                layers_df, 
                num_rows="dynamic", 
                hide_index=True,
                column_config=cols_cfg,
                use_container_width=True
            )
            st.session_state['layers'] = edited_layers.to_dict('records')

        with c2:
            st.markdown("### 📊 Modelo Geotécnico Integrado")
            fig = draw_integrated_model(st.session_state['layers'], st.session_state['spt_df'].to_dict('records'), k_val, nf_val)
            st.pyplot(fig)
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
            st.download_button("📸 Descargar Gráfico (PNG)", buf.getvalue(), "modelo_geotecnico.png", "image/png")

    with tab_design:
        st.markdown(r"""<div class="info-box" style="text-align: center;"><strong>Ecuación:</strong> $Q_{ult} = \pi \cdot \sum ( D_{nom} \cdot f_{exp,i} \cdot L_i \cdot q_{s,i} )$ &nbsp;|&nbsp; $FS = Q_{ult} / Q_{act} \ge FS_{req}$</div>""", unsafe_allow_html=True)
        c_conf, c_res = st.columns([1, 3])
        with c_conf:
            st.subheader("Parámetros")
            load = st.number_input("Carga (Ton)", value=120.0)
            fs = st.number_input("FS Req", value=2.0)
            wc = st.number_input("Rel. A/C", value=0.5)
            d_min = st.selectbox("Min Ø", LISTA_D, index=0)
            d_max = st.selectbox("Max Ø", LISTA_D, index=len(LISTA_D)-1)
            n_min = st.number_input("Min N", 1, 20, 1)
            n_max = st.number_input("Max N", 1, 20, 10)
            run_calc = st.button("🚀 CALCULAR", type="primary")

        with c_res:
            if run_calc:
                with st.spinner("Optimizando..."):
                    df_res = run_optimization(load, fs, wc, n_min, n_max, d_min, d_max, st.session_state['layers'])
                    if df_res.empty:
                        st.error("No se encontraron soluciones viables.")
                    else:
                        st.session_state['global_results'] = df_res
                        best = df_res.iloc[0]
                        k1, k2, k3, k4 = st.columns(4)
                        k1.metric("Mejor Config", f"{int(best['N'])} x Ø{int(best['D_mm'])}")
                        k2.metric("Longitud", f"{best['L']} m")
                        k3.metric("Grout", f"{best['Vol_Grout']:.1f} m³")
                        k4.metric("Huella CO2", f"{best['CO2']:.1f} Ton")
                        
                        st.subheader("Curvas de Transferencia")
                        fig_trans = draw_load_transfer(st.session_state['layers'], df_res.head(5), fs)
                        st.pyplot(fig_trans)
                        
                        st.subheader("Resultados")
                        df_show = df_res.copy()
                        df_show['Seleccionar'] = False
                        
                        # Configuración segura para tabla de resultados
                        res_cols_cfg = None
                        if hasattr(st, "column_config"):
                             res_cols_cfg = {
                                "Seleccionar": st.column_config.CheckboxColumn("Ver", default=False),
                                "D_mm": st.column_config.NumberColumn("Ø (mm)", format="%d"),
                                "L": st.column_config.NumberColumn("L (m)", format="%.1f"),
                                "FS": st.column_config.NumberColumn("FS", format="%.2f")
                             }

                        edited_res = st.data_editor(
                            df_show[["Seleccionar", "D_mm", "N", "L", "Perf_Total", "FS", "Q_adm", "Q_act", "Vol_Grout", "CO2"]].head(15),
                            hide_index=True, use_container_width=True,
                            column_config=res_cols_cfg
                        )
                        sel_rows = edited_res[edited_res['Seleccionar']]
                        st.session_state['selected_indices'] = sel_rows.index.tolist() if not sel_rows.empty else list(df_res.head(3).index)
                        
                        csv = df_res.to_csv(sep=';', decimal=',', index=False).encode('utf-8-sig')
                        st.download_button("📥 Descargar CSV", csv, "resultados.csv", "text/csv")

    with tab_caps:
        if st.session_state['global_results'] is None:
            st.info("⚠️ Ejecute el cálculo primero.")
        else:
            df = st.session_state['global_results']
            indices = st.session_state['selected_indices']
            st.subheader(f"Diseño de Dados ({len(indices)} seleccionados)")
            cols_grid = st.columns(3)
            for i, idx in enumerate(indices):
                if idx in df.index:
                    row = df.loc[idx]
                    fig_cap, vol, w, l, h, s = draw_pile_cap(row, i+1)
                    with cols_grid[i % 3]:
                        with st.container(border=True):
                            st.markdown(f"**Opción #{i+1}: {int(row['N'])} x Ø{int(row['D_mm'])}**")
                            st.pyplot(fig_cap)
                            st.caption(f"**Vol:** {vol:.2f} m³ | **Dim:** {w:.2f}x{l:.2f}x{h:.2f}m")

if __name__ == "__main__":
    main()
