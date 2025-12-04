from flask import Flask, render_template, request
import pandas as pd
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from batteryPredictor.pipeline.prediction import PredictionPipeline

app = Flask(__name__)

# --- THEME CONFIG ---
THEMES = {
    'dark': {
        'bg': '#0a0a0a',
        'grid': '#222222',
        'text': '#888888',
        'title': '#ffffff',
        'cyan': '#00f0ff',
        'green': '#ccff00',
        'spine': '#333333',
        'fill_alpha': 0.15
    },
    'light': {
        'bg': '#ffffff',
        'grid': '#e0e0e0',
        'text': '#333333',
        'title': '#000000',
        'cyan': '#0056b3',
        'green': '#008f28',
        'spine': '#cccccc',
        'fill_alpha': 0.2
    }
}

def get_theme_colors(theme_name):
    return THEMES.get(theme_name, THEMES['dark'])

def style_plot(ax, title, xlabel, ylabel, colors):
    ax.set_facecolor(colors['bg'])
    ax.grid(True, color=colors['grid'], linestyle='-', linewidth=0.5, alpha=0.6)
    
    ax.set_title(title, color=colors['title'], fontsize=12, fontweight='bold', pad=12, fontname='monospace')
    ax.set_xlabel(xlabel, color=colors['text'], fontsize=9, fontname='monospace')
    ax.set_ylabel(ylabel, color=colors['text'], fontsize=9, fontname='monospace')
    
    ax.tick_params(axis='both', colors=colors['text'], labelsize=8, width=0.5)
    
    for spine in ax.spines.values():
        spine.set_edgecolor(colors['spine'])
        spine.set_linewidth(1)

def plot_to_base64(fig, bg_color):
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', facecolor=bg_color, dpi=300)
    img.seek(0)
    plt.close(fig)
    return base64.b64encode(img.getvalue()).decode()

@app.route('/', methods=['GET'])
def homePage():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_analyze():
    try:
        file = request.files['file']
        
        # 1. GET THEME FROM FORM
        theme_mode = request.form.get('theme', 'dark') 
        colors = get_theme_colors(theme_mode)
        
        data = pd.read_csv(file)
        pipeline = PredictionPipeline()
        soh, res_df, proj_df, total_cap = pipeline.predict_bulk(data)
        
        # 2. APPLY THEME TO PLOT
        fig, ax = plt.subplots(figsize=(10, 5))
        style_plot(ax, "DISCHARGE TRAJECTORY PROJECTION", "CAPACITY (Ah)", "VOLTAGE (V)", colors)
        
        ax.plot(res_df['capacity'], res_df['avg_voltage'], color=colors['cyan'], label='LIVE TELEMETRY', linewidth=2)
        
        if not proj_df.empty:
            ax.plot(proj_df['capacity'], proj_df['avg_voltage'], color=colors['green'], 
                    linestyle=(0, (3, 1)), label='AI PREDICTION', linewidth=2)
            ax.fill_between(proj_df['capacity'], proj_df['lower_bound'], proj_df['upper_bound'], 
                            color=colors['green'], alpha=colors['fill_alpha'], linewidth=0)
        
        ax.legend(facecolor=colors['bg'], labelcolor=colors['text'], fontsize=8, loc='upper right', framealpha=1, edgecolor=colors['spine'])
        
        # 3. RENDER WITH CORRECT BG
        plot_url = plot_to_base64(fig, colors['bg'])
        
        return render_template("dashboard.html", 
                               soh=f"{soh:.2f}%", 
                               total_cap=f"{total_cap:.2f} Ah",
                               filename=file.filename,
                               plot=plot_url)
    except Exception as e:
        return f"Error: {e}"

@app.route('/compare', methods=['POST'])
def compare_files():
    try:
        f1, f2 = request.files['file1'], request.files['file2']
        theme_mode = request.form.get('theme', 'dark')
        colors = get_theme_colors(theme_mode)
        
        df1, df2 = pd.read_csv(f1), pd.read_csv(f2)
        pipeline = PredictionPipeline()
        result = pipeline.compare_files(df1, df2)
        d = result['data']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        plt.subplots_adjust(wspace=0.15)
        
        style_plot(ax1, "SOURCE A", "Ah", "V", colors)
        ax1.plot(d['f1_hist']['capacity'], d['f1_hist']['avg_voltage'], color=colors['cyan'], linewidth=1.5)
        ax1.plot(d['f1_proj']['capacity'], d['f1_proj']['avg_voltage'], color=colors['cyan'], linestyle=':', alpha=0.5)
        
        style_plot(ax2, "SOURCE B", "Ah", "V", colors)
        ax2.plot(d['f2_hist']['capacity'], d['f2_hist']['avg_voltage'], color=colors['green'], linewidth=1.5)
        ax2.plot(d['f2_proj']['capacity'], d['f2_proj']['avg_voltage'], color=colors['green'], linestyle=':', alpha=0.5)
        
        side_by_side_plot = plot_to_base64(fig, colors['bg'])
        
        overlay_plots = {}
        
        def draw_overlay(alpha_a, alpha_b):
            fig_ov, ax_ov = plt.subplots(figsize=(12, 5))
            style_plot(ax_ov, f"DIFFERENTIAL OVERLAY (MATCH: {result['similarity']}%)", "CAPACITY (Ah)", "VOLTAGE (V)", colors)
            
            ax_ov.plot(d['f1_hist']['capacity'], d['f1_hist']['avg_voltage'], color=colors['cyan'], alpha=alpha_a, linewidth=2, label='SRC A')
            ax_ov.plot(d['f1_proj']['capacity'], d['f1_proj']['avg_voltage'], color=colors['cyan'], alpha=alpha_a*0.5, linestyle=':')
            
            ax_ov.plot(d['f2_hist']['capacity'], d['f2_hist']['avg_voltage'], color=colors['green'], alpha=alpha_b, linewidth=2, label='SRC B')
            ax_ov.plot(d['f2_proj']['capacity'], d['f2_proj']['avg_voltage'], color=colors['green'], alpha=alpha_b*0.5, linestyle=':')
            
            if alpha_a == 1 and alpha_b == 1:
                ax_ov.legend(facecolor=colors['bg'], labelcolor=colors['text'], fontsize=8, framealpha=1)
            return plot_to_base64(fig_ov, colors['bg'])

        overlay_plots['default'] = draw_overlay(1.0, 1.0)
        overlay_plots['focus_a'] = draw_overlay(1.0, 0.1)
        overlay_plots['focus_b'] = draw_overlay(0.1, 1.0)

        return render_template("comparison_result.html", 
                               stats=result, 
                               f1_name=f1.filename, 
                               f2_name=f2.filename,
                               side_plot=side_by_side_plot,
                               overlays=overlay_plots)

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)