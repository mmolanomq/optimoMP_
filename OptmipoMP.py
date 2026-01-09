import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO

# ==============================================================================
# 0. CONFIGURACIÓN E INICIALIZACIÓN
# ==============================================================================
try:
    st.set_page_config(
        page_title="MicroPile Opt V3",
        layout="wide",
        page_icon="🏗️",
        initial_sidebar_state="expanded"
    )
except Exception:
    pass # Ignorar si ya fue configurada

# Estilos CSS
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; }
    h1 { color: #1e3a8a; font-weight: 800; }
    h2 { color: #1e40af; font-size: 1.5rem; border-bottom: 2px solid #e5e7eb; padding-bottom: 0.5rem; }
    .stButton>button { width: 100%; border-radius: 6px; font-weight: bold; height: 3rem; }
    .stButton>button[kind="primary"] { background-color: #2563eb; border: none; }
    .stButton>button[kind="primary"]:hover { background-color: #1d4ed8; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; color: #0f172a; }
    .info-box { background-color: #eff6ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #3b82f6; text-align: center; }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTES ---
CONSTANTS = {
    'DIAMETROS_COM': {100: 1.00, 115: 0.95, 130: 0.93, 150: 0.90, 200: 0.85},
    'COSTO_PERF_BASE': 100,
    'FACTOR_CO2': {'CEMENTO': 0.90, 'PERF': 15.0, 'ACERO': 1.85},
    'DENSIDAD': {'ACERO': 7850.0, 'CEMENTO': 3150.0},
    'FY_ACERO_KPA': 500000.0
}
LISTA_D = sorted(list(CONSTANTS['DIAMETROS_COM'].keys()))

# ==============================================================================
# BLOQUE 1: GESTIÓN DE ESTADO
# ==============================================================================
def init_session_state():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        
    if 'layers' not in st.session_state:
        st.session_state['layers'] = [
            {"Nombre": "Relleno", "H (m)": 3.0, "Qs (kPa)": 40.0, "F.Exp": 1.1, "Color": "#dbeafe"},
            {"Nombre": "Arcilla", "H (m)": 5.0, "Qs (kPa)": 80.0, "F.Exp": 1.2, "Color": "#fef3c7"},
            {"Nombre": "Roca", "H (m)": 10.0, "Qs (kPa)": 150.0, "F.Exp": 1.3, "Color": "#fee2e2"}
        ]
        
    if 'spt_df' not in st.session_state:
        st.session_state['spt_df'] = pd.DataFrame([
            {"z": 1.5, "n": 4}, {"z": 3.0, "n": 7}, {"z": 4.5, "n": 12},
            {"z": 6.0, "n": 15}, {"z": 7.5, "n": 22}, {"z": 9.0, "n": 28},
            {"z": 10.5, "n": 35}, {"z": 12.0, "n": 42}, {"z": 15.0, "n": 50}
        ])

    if 'global_results' not in st.session_state:
        st.session_state['global_results'] = None

    if 'selected_indices' not in st.session_state:
        st.session_state['selected_indices'] = []

# ==============================================================================
# BLOQUE 2: MOTOR DE CÁLCULO
# ==============================================================================
def run_optimization(load_ton, fs_req, wc_ratio, min_n, max_n, min_d, max_d, layers_data):
    carga_req_kn = load_ton * 9.81
    solutions = []
    
    valid_diameters = [d for d in LISTA_D if min_d <= d <= max_d]
    
    for D_mm in valid_diameters:
        D_m = D_mm / 1000.0
        eficiencia = CONSTANTS['DIAMETROS_COM'].get(D_mm, 0.85)
        
        for N in range(min_n, max_n + 1):
            q_act_pilote = carga_req_kn / N
            q_req_geo = q_act_pilote * fs_req
            
            for L in np.arange(5.0, 40.5, 0.5):
                q_ult = 0
                vol_exp = 0
                acc_depth = 0
                area_perf = np.pi * (D_m/2)**2
                
                # Iterar sobre datos de capas (formato diccionario)
                for layer in layers_data:
                    h_layer = layer["H (m)"]
                    qs_layer = layer["Qs (kPa)"]
                    fexp_layer = layer["F.Exp"]
                    
                    top = acc_depth
                    bot = acc_depth + h_layer
                    acc_depth += h_layer
                    
                    start = max(0, top)
                    end = min(L, bot)
                    seg = max(0, end - start)
                    
                    if seg > 0:
                        d_eff = D_m * fexp_layer # Ecuación lineal
                        q_ult += (np.pi * d_eff * seg) * qs_layer
                        vol_exp += (area_perf * seg) * fexp_layer
                
                # Extensión final
                if L > acc_depth:
                    extra = L - acc_depth
                    last = layers_data[-1]
                    d_eff = D_m * last["F.Exp"]
                    q_ult += (np.pi * d_eff * extra) * last["Qs (kPa)"]
                    vol_exp += (area_perf * extra) * last["F.Exp"]
                
                if q_ult >= q_req_geo:
                    vol_total = vol_exp * N
                    costo = (L * N * COSTO_PERF_BASE) / eficiencia
                    
                    vol_acero = (q_act_pilote / CONSTANTS['FY_ACERO_KPA']) * L * N
                    peso_acero = vol_acero * CONSTANTS['DENSIDAD']['ACERO']
                    peso_cemento = max(0, vol_total - vol_acero) * (1000 / (wc_ratio + 1/3.15))
                    
                    co2 = (peso_acero * CONSTANTS['FACTOR_CO2']['ACERO'] + 
                           peso_cemento * CONSTANTS['FACTOR_CO2']['CEMENTO'] + 
                           (L*N) * CONSTANTS['FACTOR_CO2']['PERF']) / 1000
                    
                    solutions.append({
                        "D_mm": D_mm, "N": N, "L": L, "Perf_Total": L*N,
                        "FS": q_ult / q_act_pilote, "Q_adm": q_ult / fs_req / 9.81,
                        "Q_act": q_act_pilote / 9.81, "Vol_Grout": vol_total,
                        "CO2": co2, "Costo_Idx": costo
                    })
                    break 
    
    if not solutions: return pd.DataFrame()
    return pd.DataFrame(solutions).sort_values("Costo_Idx")

# ==============================================================================
# BLOQUE 3: VISUALIZACIÓN
# ==============================================================================
def draw_integrated_model(layers_data, spt_data, k_factor, water_table):
    total_depth = sum(l["H (m)"] for l in layers_data)
    max_depth = max(total_depth, 15) * 1.1
    
    z_spt = [d['z'] for d in spt_data]
    n_spt = [d['n'] for d in spt_data]
    qs_est = [min(d['n'] * k_factor, 300) for d in spt_data]

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(10, 6), gridspec_kw={'width_ratios': [1, 1, 1]}, sharey=True)
    plt.subplots_adjust(wspace=0.2)

    # Estratigrafía
    current_depth = 0
    for layer in layers_data:
        h = layer["H (m)"]
        color = layer.get("Color", "#e2e8f0")
        if not color.startswith("#"): color = "#e2e8f0" # Fallback color
        
        rect = patches.Rectangle((0, current_depth), 1, h, linewidth=0.5, edgecolor='gray', facecolor=color)
        ax0.add_patch(rect)
        
        ax0.text(0.5, current_depth + h/2, f"{layer['Nombre']}\nQs={int(layer['Qs (kPa)'])}", 
                ha='center', va='center', fontsize=8, color='#1e293b', fontweight='bold', wrap=True)
        
        current_depth += h
        ax0.axhline(y=current_depth, color='gray', linestyle=':', linewidth=0.5)

    if water_table > 0:
        ax0.axhline(y=water_table, color='blue', linestyle='--', linewidth=2)
        ax0.text(0.9, water_table - 0.2, "N.F.", color='blue', fontsize=8, fontweight='bold', ha='right')

    ax0.set_ylim(max_depth, 0)
    ax0.set_xlim(0, 1)
    ax0.axis('off')
    ax0.set_title("Estratigrafía", fontsize=10, fontweight='bold')

    # N-SPT
    ax1.plot(n_spt, z_spt, 'o-', color='#2563eb', linewidth=2, markersize=4)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_title("N-SPT", fontsize=10, fontweight='bold', color='#1e40af')
    ax1.set_xlabel("N")
    ax1.set_xlim(0, 60)
    ax1.set_ylabel("Profundidad (m)")

    # Qs
    ax2.plot(qs_est, z_spt, 's-', color='#dc2626', linewidth=2, markersize=4)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_title(f"Qs (K={k_factor})", fontsize=10, fontweight='bold', color='#991b1b')
    ax2.set_xlabel("kPa")
    ax2.set_xlim(0, 350)

    return fig

def draw_load_transfer(layers_data, results, fs_req):
    if results is None or results.empty: return None
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#2563eb', '#16a34a', '#dc2626', '#d97706', '#9333ea']
    
    current_depth = 0
    for layer in layers_data:
        h = layer["H (m)"]
        color = layer.get("Color", "#e2e8f0")
        if not color.startswith("#"): color = "#e2e8f0"
        ax.axhspan(current_depth, current_depth + h, color=color, alpha=0.3)
        current_depth += h

    for i, (_, row) in enumerate(results.iterrows()):
        L = row['L']
        Q_adm = row['Q_adm']
        ax.plot([0, Q_adm*9.81], [0, L], marker='o', label=f"{int(row['N'])}xØ{int(row['D_mm'])}", color=colors[i % 5], linewidth=2)

    ax.set_ylim(current_depth + 2, 0)
    ax.set_xlabel("Carga (kN)")
    ax.set_ylabel("Profundidad (m)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.5)
    return fig

def draw_pile_cap(config, rank):
    N, D = int(config['N']), config['D_mm'] / 1000.0
    S, Edge = max(0.75, 3 * D), max(0.30, 1.5 * D)
    
    cols = int(np.ceil(np.sqrt(N)))
    rows = int(np.ceil(N/cols))
    
    W, L = (cols - 1) * S + 2 * Edge, (rows - 1) * S + 2 * Edge
    H, Vol = 0.50 + (0.1 * (rows-1)), W * L * (0.50 + (0.1 * (rows-1)))
    
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.add_patch(patches.Rectangle((0, 0), W, L, facecolor='#e2e8f0', edgecolor='#475569', linewidth=2))
    
    for i in range(N):
        cx, cy = Edge + (i % cols) * S, Edge + (i // cols) * S
        ax.add_patch(patches.Circle((cx, cy), D/2, facecolor='#1e293b', edgecolor='black'))
    
    ax.set_xlim(-0.5, W + 0.5); ax.set_ylim(-0.5, L + 0.5)
    ax.axis('off'); ax.set_aspect('equal')
    return fig, Vol, W, L, H

# ==============================================================================
# 4. APP PRINCIPAL
# ==============================================================================
def main():
    init_session_state()

    if not st.session_state['logged_in']:
        login_screen()
        return

    with st.sidebar:
        st.info(f"👤 **{st.session_state['user_info']['nombre']}**\n\n{st.session_state['user_info']['cargo']}")
        if st.button("Cerrar Sesión"):
            st.session_state['logged_in'] = False
            st.rerun()

    st.title("🏗️ Optimizador de Micropilotes")
    tab1, tab2, tab3 = st.tabs(["1. Info Geotécnica", "2. Diseño", "3. Dados"])

    # --- TAB 1 ---
    with tab1:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("Datos de Campo")
            
            # Editor SPT (Simple)
            if hasattr(st, "data_editor"):
                st.session_state['spt_df'] = st.data_editor(st.session_state['spt_df'], num_rows="dynamic", hide_index=True, key="editor_spt")
            else:
                st.dataframe(st.session_state['spt_df'])
            
            k_val = st.slider("Factor K", 1.0, 10.0, 3.5)
            nf_val = st.number_input("Nivel Freático (m)", 0.0, 50.0, 2.0)
            
            st.divider()
            st.subheader("Estratos")
            
            # Editor Capas (Simplificado para evitar errores de ColorColumn)
            layers_df = pd.DataFrame(st.session_state['layers'])
            
            if hasattr(st, "data_editor"):
                # Configuración básica sin tipos complejos
                edited_layers = st.data_editor(
                    layers_df, 
                    num_rows="dynamic", 
                    hide_index=True, 
                    key="editor_layers",
                    use_container_width=True
                )
                st.session_state['layers'] = edited_layers.to_dict('records')
            else:
                st.info("Modo lectura (versión antigua de Streamlit)")
                st.dataframe(layers_df)

        with c2:
            st.subheader("Modelo Integrado")
            fig = draw_integrated_model(st.session_state['layers'], st.session_state['spt_df'].to_dict('records'), k_val, nf_val)
            st.pyplot(fig)

    # --- TAB 2 ---
    with tab2:
        st.markdown(r"""<div class="info-box"><strong>Ecuación:</strong> $Q_{ult} = \pi \cdot \sum ( D_{nom} \cdot f_{exp} \cdot L \cdot q_s )$</div>""", unsafe_allow_html=True)
        col_inp, col_out = st.columns([1, 3])
        
        with col_inp:
            load = st.number_input("Carga (Ton)", value=120.0)
            fs = st.number_input("FS", value=2.0)
            wc = st.number_input("Rel A/C", value=0.5)
            d_min = st.selectbox("Min Ø", LISTA_D, index=0)
            d_max = st.selectbox("Max Ø", LISTA_D, index=len(LISTA_D)-1)
            n_min = st.number_input("Min N", 1, 20, 1)
            n_max = st.number_input("Max N", 1, 20, 10)
            calc = st.button("CALCULAR", type="primary")

        with col_out:
            if calc:
                with st.spinner("Optimizando..."):
                    res = run_optimization(load, fs, wc, n_min, n_max, d_min, d_max, st.session_state['layers'])
                    if res.empty:
                        st.error("Sin soluciones.")
                    else:
                        st.session_state['global_results'] = res
                        best = res.iloc[0]
                        k1, k2, k3 = st.columns(3)
                        k1.metric("Mejor", f"{int(best['N'])}xØ{int(best['D_mm'])}")
                        k2.metric("Longitud", f"{best['L']}m")
                        k3.metric("CO2", f"{best['CO2']:.1f}T")
                        
                        st.pyplot(draw_load_transfer(st.session_state['layers'], res.head(5), fs))
                        
                        # Tabla Resultados Simplificada
                        df_show = res.copy()
                        df_show['Sel'] = False
                        
                        # Intentar usar checkbox si es posible, sino tabla normal
                        try:
                            edited_res = st.data_editor(
                                df_show[["Sel", "D_mm", "N", "L", "Perf_Total", "FS", "Q_adm", "Q_act", "Vol_Grout", "CO2"]].head(10),
                                hide_index=True,
                                column_config={"Sel": st.column_config.CheckboxColumn(default=False)} if hasattr(st, 'column_config') else None
                            )
                            # Guardar selección
                            sel_rows = edited_res[edited_res['Sel']]
                            st.session_state['selected_indices'] = sel_rows.index.tolist() if not sel_rows.empty else list(res.head(3).index)
                        except:
                            st.dataframe(df_show.head(10))
                            st.session_state['selected_indices'] = list(res.head(3).index)

                        csv = res.to_csv(sep=';', decimal=',', index=False).encode('utf-8-sig')
                        st.download_button("📥 Descargar CSV", csv, "resultados.csv", "text/csv")

    # --- TAB 3 ---
    with tab3:
        if st.session_state['global_results'] is None:
            st.info("Calcule primero.")
        else:
            df = st.session_state['global_results']
            indices = st.session_state['selected_indices']
            cols = st.columns(3)
            for i, idx in enumerate(indices[:3]): 
                if idx in df.index:
                    row = df.loc[idx]
                    fig, vol, w, l, h = draw_pile_cap(row, i+1)
                    with cols[i]:
                        st.markdown(f"**Opción {i+1}**")
                        st.pyplot(fig)
                        st.caption(f"Vol: {vol:.1f}m³ | {w:.1f}x{l:.1f}m")

if __name__ == "__main__":
    main()
