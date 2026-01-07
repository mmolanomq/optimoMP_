import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO

# ==============================================================================
# BLOQUE 1: CONFIGURACIÓN GLOBAL Y ESTILOS
# ==============================================================================
def setup_page():
    st.set_page_config(
        page_title="MicroPile Opt V3",
        layout="wide",
        page_icon="🏗️",
        initial_sidebar_state="expanded"
    )
    # CSS para mejorar la interfaz visual
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
# BLOQUE 2: GESTIÓN DE ESTADO (SESSION STATE)
# ==============================================================================
def init_session_state():
    defaults = {
        'logged_in': False,
        'layers': [
            {"name": "Relleno / Arcilla Blanda", "thickness": 3.0, "qs": 40.0, "f_exp": 1.1, "color": "#dbeafe"},
            {"name": "Arcilla Firme / Limo", "thickness": 5.0, "qs": 80.0, "f_exp": 1.2, "color": "#fef3c7"},
            {"name": "Estrato Resistente", "thickness": 10.0, "qs": 150.0, "f_exp": 1.3, "color": "#fee2e2"}
        ],
        'spt_df': pd.DataFrame([
            {"z": 1.5, "n": 4}, {"z": 3.0, "n": 7}, {"z": 4.5, "n": 12},
            {"z": 6.0, "n": 15}, {"z": 7.5, "n": 22}, {"z": 9.0, "n": 28},
            {"z": 10.5, "n": 35}, {"z": 12.0, "n": 42}, {"z": 15.0, "n": 50}
        ]),
        'global_results': None,
        'selected_indices': []
    }
    
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

# ==============================================================================
# BLOQUE 3: UTILIDADES SEGURAS (CORRECCIÓN DE ERRORES)
# ==============================================================================
def get_safe_column_config():
    """
    Crea la configuración de columnas verificando la versión de Streamlit.
    Corrige el error: AttributeError: 'ColorColumn'
    """
    # Configuración base numérica (compatible con versiones recientes)
    cols = {
        "name": "Nombre",
        "thickness": st.column_config.NumberColumn("H (m)", min_value=0.1, format="%.1f"),
        "qs": st.column_config.NumberColumn("Qs (kPa)", min_value=0),
        "f_exp": st.column_config.NumberColumn("F.Exp", min_value=1.0, max_value=3.0, step=0.1)
    }
    
    # Intento seguro para la columna de color
    try:
        if hasattr(st.column_config, 'ColorColumn'):
            cols["color"] = st.column_config.ColorColumn("Color")
        else:
            # Fallback a texto si la versión es antigua
            cols["color"] = st.column_config.TextColumn("Color (Hex)")
    except Exception:
        # Fallback de emergencia
        cols["color"] = "Color"
        
    return cols

# ==============================================================================
# BLOQUE 4: MOTOR DE CÁLCULO
# ==============================================================================
def run_optimization(load_ton, fs_req, wc_ratio, min_n, max_n, min_d, max_d, layers):
    carga_req_kn = load_ton * 9.81
    solutions = []
    
    # Filtrar diámetros seleccionados
    valid_diameters = [d for d in LISTA_D if min_d <= d <= max_d]
    
    for D_mm in valid_diameters:
        D_m = D_mm / 1000.0
        eficiencia = CONSTANTS['DIAMETROS_COM'].get(D_mm, 0.85)
        
        for N in range(min_n, max_n + 1):
            q_act_pilote = carga_req_kn / N
            q_req_geo = q_act_pilote * fs_req
            
            # Iterar longitudes
            for L in np.arange(5.0, 40.5, 0.5):
                q_ult = 0
                vol_exp = 0
                acc_depth = 0
                area_perf = np.pi * (D_m/2)**2
                
                # Integración por estratos
                for layer in layers:
                    top = acc_depth
                    bot = acc_depth + layer['thickness']
                    acc_depth += layer['thickness']
                    
                    start = max(0, top)
                    end = min(L, bot)
                    seg = max(0, end - start)
                    
                    if seg > 0:
                        # ECUACIÓN FÍSICA: D_eff = D_nom * F_exp
                        d_eff = D_m * layer['f_exp']
                        q_ult += (np.pi * d_eff * seg) * layer['qs']
                        vol_exp += (area_perf * seg) * layer['f_exp']
                
                # Extensión en último estrato si L > prof. suelos
                if L > acc_depth:
                    extra = L - acc_depth
                    last = layers[-1]
                    d_eff = D_m * last['f_exp']
                    q_ult += (np.pi * d_eff * extra) * last['qs']
                    vol_exp += (area_perf * extra) * last['f_exp']
                
                # Validar seguridad
                if q_ult >= q_req_geo:
                    vol_total = vol_exp * N
                    costo = (L * N * COSTO_PERF_BASE) / eficiencia
                    
                    # Cálculo Ambiental
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
# BLOQUE 5: MOTOR DE VISUALIZACIÓN (GRÁFICOS)
# ==============================================================================
def draw_integrated_model(layers, spt_data, k_factor, water_table):
    total_depth = sum(l['thickness'] for l in layers)
    max_depth = max(total_depth, 15) * 1.1
    
    z_spt = [d['z'] for d in spt_data]
    n_spt = [d['n'] for d in spt_data]
    qs_est = [min(d['n'] * k_factor, 300) for d in spt_data]

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(10, 8), gridspec_kw={'width_ratios': [1, 1, 1]}, sharey=True)
    plt.subplots_adjust(wspace=0.15)

    # 1. Estratigrafía
    current_depth = 0
    for layer in layers:
        rect = patches.Rectangle((0, current_depth), 1, layer['thickness'], linewidth=0.5, edgecolor='gray', facecolor=layer['color'])
        ax0.add_patch(rect)
        ax0.text(0.5, current_depth + layer['thickness']/2, f"{layer['name']}\nQs={int(layer['qs'])}", 
                ha='center', va='center', fontsize=8, color='#1e293b', fontweight='bold', wrap=True)
        current_depth += layer['thickness']
        ax0.axhline(y=current_depth, color='gray', linestyle=':', linewidth=0.5)

    if water_table > 0:
        ax0.axhline(y=water_table, color='blue', linestyle='--', linewidth=2)
        ax0.text(0.9, water_table - 0.2, "N.F.", color='blue', fontsize=8, fontweight='bold', ha='right')

    ax0.set_ylim(max_depth, 0)
    ax0.set_xlim(0, 1)
    ax0.axis('off') # Limpiar eje estratigrafía
    ax0.set_title("Estratigrafía", fontsize=10, fontweight='bold')

    # 2. N-SPT y 3. Qs (Estándar)
    for ax, data, title, color, x_lim in [(ax1, n_spt, "N-SPT", '#2563eb', 60), (ax2, qs_est, f"Qs (K={k_factor})", '#dc2626', 350)]:
        ax.plot(data, z_spt, 'o-', color=color, linewidth=2, markersize=4)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_title(title, fontsize=10, fontweight='bold', color=color)
        ax.set_xlim(0, x_lim)
        if ax == ax1: ax.set_ylabel("Profundidad (m)")

    return fig

def draw_load_transfer(layers, results, fs_req):
    if results is None or results.empty: return None
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#2563eb', '#16a34a', '#dc2626', '#d97706', '#9333ea']
    
    current_depth = 0
    for layer in layers:
        ax.axhspan(current_depth, current_depth + layer['thickness'], color=layer['color'], alpha=0.3)
        current_depth += layer['thickness']
    
    # Solo las 5 mejores
    for i, (_, row) in enumerate(results.head(5).iterrows()):
        # Reconstrucción simplificada de la curva para visualización rápida
        q_points = [0, row['Q_adm']*9.81] # Ton a kN aprox para visual
        z_points = [0, row['L']] 
        # Nota: En una implementación real, aquí iría la integración paso a paso como en la func. de optimización
        # Para mantener el código limpio en este bloque, simplificamos la visualización lineal vs prof.
        
        ax.plot([0, row['Q_adm']], [0, row['L']], marker='o', label=f"{int(row['N'])}xØ{int(row['D_mm'])}", color=colors[i % 5], linewidth=2)

    ax.set_ylim(current_depth + 2, 0)
    ax.set_xlabel("Capacidad Admisible (Ton)")
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
# BLOQUE 6: INTERFAZ DE USUARIO (MAIN)
# ==============================================================================
def main():
    setup_page()
    init_session_state()

    # --- PANTALLA DE LOGIN ---
    if not st.session_state['logged_in']:
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown("<h2 style='text-align: center;'>🔒 Acceso Ingeniería</h2>", unsafe_allow_html=True)
                with st.form("login"):
                    st.text_input("Nombre")
                    st.text_input("Email")
                    if st.form_submit_button("Ingresar", type="primary"):
                        st.session_state['logged_in'] = True
                        st.rerun()
        return

    # --- APP PRINCIPAL ---
    with st.sidebar:
        st.success("Sesión Activa")
        if st.button("Salir"):
            st.session_state['logged_in'] = False
            st.rerun()

    st.title("🏗️ Optimizador de Micropilotes")
    tab1, tab2, tab3 = st.tabs(["1. Info Geotécnica", "2. Diseño", "3. Dados"])

    # TAB 1: GEOTECNIA
    with tab1:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("### Datos de Entrada")
            st.session_state['spt_df'] = st.data_editor(st.session_state['spt_df'], num_rows="dynamic", hide_index=True)
            k_val = st.slider("Factor K", 1.0, 10.0, 3.5)
            nf_val = st.number_input("Nivel Freático (m)", 0.0, 50.0, 2.0)
            
            st.markdown("### Estratos")
            # USANDO LA FUNCIÓN SEGURA PARA EVITAR EL ERROR
            safe_cols = get_safe_column_config()
            edited_layers = st.data_editor(
                pd.DataFrame(st.session_state['layers']), 
                num_rows="dynamic", 
                hide_index=True, 
                column_config=safe_cols,
                use_container_width=True
            )
            st.session_state['layers'] = edited_layers.to_dict('records')

        with c2:
            fig = draw_integrated_model(st.session_state['layers'], st.session_state['spt_df'].to_dict('records'), k_val, nf_val)
            st.pyplot(fig)

    # TAB 2: DISEÑO
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
                with st.spinner("Calculando..."):
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
                        
                        st.pyplot(draw_load_transfer(st.session_state['layers'], res, fs))
                        
                        # Tabla de resultados
                        st.dataframe(res[["D_mm", "N", "L", "Perf_Total", "FS", "Q_adm", "CO2"]].head(10), use_container_width=True)
                        st.session_state['selected_indices'] = list(res.head(3).index) # Auto-select top 3

    # TAB 3: DADOS
    with tab3:
        if st.session_state['global_results'] is None:
            st.info("Calcule primero.")
        else:
            df = st.session_state['global_results']
            indices = st.session_state['selected_indices']
            cols = st.columns(3)
            for i, idx in enumerate(indices[:3]): # Max 3
                if idx in df.index:
                    row = df.loc[idx]
                    fig, vol, w, l, h = draw_pile_cap(row, i+1)
                    with cols[i]:
                        st.markdown(f"**Opción {i+1}**")
                        st.pyplot(fig)
                        st.caption(f"Vol: {vol:.1f}m³ | {w:.1f}x{l:.1f}m")

if __name__ == "__main__":
    main()
