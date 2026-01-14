"""
EvaLex Gradio Interface

A web interface for evaluating lexical competence in language models.
"""

import os
import gradio as gr
import pandas as pd
from pathlib import Path

from evalex.config import EvaLexConfig, AVAILABLE_MODELS, DEFAULT_WORD_LISTS
from evalex.pipeline import EvaLexPipeline, WordEvaluator, load_results_for_ranking


# Custom CSS for a modern, premium look
CUSTOM_CSS = """
:root {
    --primary-color: #6366f1;
    --primary-hover: #4f46e5;
    --secondary-color: #8b5cf6;
    --success-color: #10b981;
    --warning-color: #f59e0b;
    --error-color: #ef4444;
    --bg-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --card-bg: rgba(255, 255, 255, 0.95);
    --text-primary: #1f2937;
    --text-secondary: #6b7280;
}

.gradio-container {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%) !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

.main-header {
    background: var(--bg-gradient);
    color: white;
    padding: 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
}

.main-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}

.main-header p {
    font-size: 1.1rem;
    opacity: 0.9;
}

.metric-card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    border: 1px solid rgba(99, 102, 241, 0.1);
    transition: transform 0.2s, box-shadow 0.2s;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.15);
}

.score-display {
    font-size: 3rem;
    font-weight: 800;
    background: var(--bg-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.ranking-table {
    background: white;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}

button.primary {
    background: var(--bg-gradient) !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
    transition: all 0.3s !important;
}

button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5) !important;
}

.tab-nav button {
    font-weight: 600 !important;
    transition: all 0.2s !important;
}

.tab-nav button.selected {
    background: var(--bg-gradient) !important;
    color: white !important;
}
"""


def get_all_models() -> list:
    """Get flat list of all available models."""
    models = []
    for category, model_list in AVAILABLE_MODELS.items():
        models.extend(model_list)
    return models


def parse_word_list(word_text: str) -> pd.DataFrame:
    """Parse word list from text input."""
    lines = [line.strip() for line in word_text.strip().split("\n") if line.strip()]
    
    # Check if first line is a header
    if lines and lines[0].lower() in ["word", "palabra", "words", "palabras"]:
        lines = lines[1:]
    
    # Check if lines contain tabs (TSV format)
    if lines and "\t" in lines[0]:
        words = [line.split("\t")[0] for line in lines]
    else:
        words = lines
    
    return pd.DataFrame({"word": words})


def load_default_word_list(filepath: str) -> pd.DataFrame:
    """Load a default word list from file."""
    try:
        df = pd.read_csv(filepath, sep="\t")
        if "word" not in df.columns:
            # Assume first column is words
            df.columns = ["word"] + list(df.columns[1:])
        return df
    except Exception as e:
        print(f"Error loading word list: {e}")
        return pd.DataFrame({"word": []})


def evaluate_model(
    word_input: str,
    word_file,
    model_name: str,
    backend_type: str,
    api_key: str,
    api_base_url: str,
    progress=gr.Progress(),
) -> tuple:
    """
    Evaluate a model's lexical competence.
    
    Returns:
        Tuple of (score_html, results_df, status_message)
    """
    try:
        # Parse words
        if word_file is not None:
            words_df = pd.read_csv(word_file.name, sep="\t")
            if "word" not in words_df.columns:
                words_df.columns = ["word"] + list(words_df.columns[1:])
        elif word_input.strip():
            words_df = parse_word_list(word_input)
        else:
            return (
                create_score_html(0, 0),
                pd.DataFrame(),
                "⚠️ Please enter words or upload a file."
            )
        
        if len(words_df) == 0:
            return (
                create_score_html(0, 0),
                pd.DataFrame(),
                "⚠️ No valid words found in input."
            )
        
        # Create config
        config = EvaLexConfig(
            model_name=model_name,
            backend="openai" if backend_type == "OpenAI API" else "local",
            openai_api_key=api_key if api_key else None,
            openai_base_url=api_base_url if api_base_url else None,
            num_words=len(words_df),
        )
        
        # Create and run pipeline
        def progress_callback(pct, msg):
            progress(pct, desc=msg)
        
        pipeline = EvaLexPipeline(config)
        
        try:
            results_df, metrics = pipeline.run(
                words_df,
                progress_callback=progress_callback,
                save_intermediate=False,
            )
        finally:
            pipeline.cleanup()
        
        # Create result displays
        score_html = create_score_html(
            metrics["known_count"],
            metrics["total_count"],
        )
        
        status = f"✅ Evaluation complete! {metrics['accuracy_percentage']} lexical competence"
        
        return score_html, results_df, status
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}\n{traceback.format_exc()}"
        return create_score_html(0, 0), pd.DataFrame(), error_msg


def create_score_html(known: int, total: int) -> str:
    """Create HTML for score display."""
    accuracy = (known / total * 100) if total > 0 else 0
    
    # Color based on score
    if accuracy >= 80:
        color = "#10b981"  # Green
    elif accuracy >= 60:
        color = "#6366f1"  # Purple
    elif accuracy >= 40:
        color = "#f59e0b"  # Amber
    else:
        color = "#ef4444"  # Red
    
    return f"""
    <div style="text-align: center; padding: 2rem;">
        <div style="font-size: 4rem; font-weight: 800; color: {color}; margin-bottom: 0.5rem;">
            {accuracy:.1f}%
        </div>
        <div style="font-size: 1.2rem; color: #6b7280;">
            {known} / {total} words known
        </div>
    </div>
    """


def get_ranking_data(word_list: str) -> pd.DataFrame:
    """Get ranking data for a specific word list."""
    # Map display names to file patterns
    list_patterns = {
        "CREA 10k (POS filtered)": "CREA_10k_pos_filter",
        "CREA 10k (Lemmas)": "CREA_10k_lemmas",
        "DEA 100": "DEA",
    }
    
    pattern = list_patterns.get(word_list, word_list)
    
    df = load_results_for_ranking(
        results_dir="results",
        word_list_filter=pattern,
    )
    
    if df.empty:
        return pd.DataFrame({
            "Model": [],
            "Known Words": [],
            "Total": [],
            "Accuracy": [],
        })
    
    # Format for display
    display_df = pd.DataFrame({
        "🏆 Rank": range(1, len(df) + 1),
        "Model": df["model"].values,
        "Known Words": df["known_words"].values,
        "Total": df["total_words"].values,
        "Accuracy": df["accuracy_pct"].values,
    })
    
    return display_df


def create_ranking_chart(word_list: str):
    """Create a bar chart for ranking visualization."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    
    df = load_results_for_ranking(
        results_dir="results",
        word_list_filter=word_list.split("(")[0].strip().replace(" ", "_"),
    )
    
    if df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig
    
    # Sort by accuracy
    df = df.sort_values("accuracy", ascending=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.5)))
    
    # Create gradient colors
    colors = plt.cm.viridis(df["accuracy"].values)
    
    # Create horizontal bar chart
    bars = ax.barh(df["model"], df["accuracy"] * 100, color=colors, edgecolor="white", linewidth=0.5)
    
    # Add value labels
    for bar, acc in zip(bars, df["accuracy"]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{acc*100:.1f}%', va='center', fontsize=10, fontweight='bold')
    
    # Styling
    ax.set_xlabel("Lexical Competence (%)", fontsize=12, fontweight='bold')
    ax.set_xlim(0, 105)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    
    plt.title(f"Model Ranking - {word_list}", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    return fig


def build_interface():
    """Build the Gradio interface."""
    
    with gr.Blocks(css=CUSTOM_CSS, title="EvaLex - Lexical Competence Evaluation") as app:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🧠 EvaLex</h1>
            <p>Evaluate Lexical Competence in Language Models</p>
        </div>
        """)
        
        with gr.Tabs() as tabs:
            
            # Tab 1: Evaluate Model
            with gr.TabItem("📊 Evaluate Model", id="evaluate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📝 Word List")
                        
                        word_input = gr.Textbox(
                            label="Enter words (one per line)",
                            placeholder="casa\nperro\namor\nlibro\n...",
                            lines=10,
                        )
                        
                        word_file = gr.File(
                            label="Or upload a TSV file",
                            file_types=[".tsv", ".txt", ".csv"],
                        )
                        
                        gr.Markdown("### 🤖 Model Selection")
                        
                        model_dropdown = gr.Dropdown(
                            choices=get_all_models(),
                            label="Select Model",
                            value="gpt-4o-mini",
                        )
                        
                        backend_radio = gr.Radio(
                            choices=["Local", "OpenAI API"],
                            label="Backend Type",
                            value="OpenAI API",
                        )
                        
                        with gr.Accordion("🔑 API Settings", open=False):
                            api_key_input = gr.Textbox(
                                label="API Key",
                                placeholder="Enter API key or leave empty to use OPENAI_API_KEY env var",
                                type="password",
                            )
                            api_base_input = gr.Textbox(
                                label="API Base URL",
                                placeholder="https://api.openai.com/v1",
                            )
                        
                        evaluate_btn = gr.Button(
                            "🚀 Evaluate Model",
                            variant="primary",
                            size="lg",
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 📈 Results")
                        
                        score_display = gr.HTML(
                            value=create_score_html(0, 0),
                        )
                        
                        status_text = gr.Textbox(
                            label="Status",
                            interactive=False,
                        )
                        
                        results_table = gr.DataFrame(
                            label="Detailed Results",
                            headers=["word", "group_known"],
                        )
                
                # Event handler
                evaluate_btn.click(
                    fn=evaluate_model,
                    inputs=[
                        word_input,
                        word_file,
                        model_dropdown,
                        backend_radio,
                        api_key_input,
                        api_base_input,
                    ],
                    outputs=[score_display, results_table, status_text],
                )
            
            # Tab 2: Ranking
            with gr.TabItem("🏆 Ranking", id="ranking"):
                gr.Markdown("""
                ### Model Leaderboard
                Compare lexical competence scores across different language models.
                """)
                
                word_list_selector = gr.Dropdown(
                    choices=[
                        "CREA 10k (POS filtered)",
                        "CREA 10k (Lemmas)",
                        "DEA 100",
                    ],
                    label="Select Word List",
                    value="CREA 10k (POS filtered)",
                )
                
                refresh_btn = gr.Button("🔄 Refresh Rankings", variant="secondary")
                
                ranking_table = gr.DataFrame(
                    value=get_ranking_data("CREA 10k (POS filtered)"),
                    label="Model Rankings",
                )
                
                ranking_chart = gr.Plot(
                    value=create_ranking_chart("CREA 10k (POS filtered)"),
                    label="Performance Comparison",
                )
                
                # Event handlers
                word_list_selector.change(
                    fn=lambda wl: (get_ranking_data(wl), create_ranking_chart(wl)),
                    inputs=[word_list_selector],
                    outputs=[ranking_table, ranking_chart],
                )
                
                refresh_btn.click(
                    fn=lambda wl: (get_ranking_data(wl), create_ranking_chart(wl)),
                    inputs=[word_list_selector],
                    outputs=[ranking_table, ranking_chart],
                )
        
        # Footer
        gr.Markdown("""
        ---
        <div style="text-align: center; color: #6b7280; font-size: 0.9rem;">
            <p><strong>EvaLex</strong> - Lexical Competence Evaluation for Language Models</p>
            <p>For questions or support, contact: jcollado@ujaen.es</p>
        </div>
        """)
    
    return app


def main():
    """Run the Gradio application."""
    app = build_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
