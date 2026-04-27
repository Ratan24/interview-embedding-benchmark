import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
base_dir = Path(__file__).resolve().parent
results_file = base_dir / "results" / "confusion_matrices.json"
figures_dir = base_dir / "figures"
figures_dir.mkdir(exist_ok=True)

# Define labels
labels = ['Not Exp (0)', 'Junior (1)', 'Mid (2)', 'Senior (3)']

with open(results_file, 'r') as f:
    matrices = json.load(f)

# Priority 1: Voyage. Priority 2: KaLM-12b
models_to_check = ['voyage', 'kalm-12b']
skills = ['react', 'javascript', 'html_css']
condition = 'C2a'
classifier = 'ordinal'

def plot_matrix(data_dict, model_name, skill_name):
    plt.figure(figsize=(8, 6))
    
    # Extract the actual 2D array from the dict saved by benchmark.py
    matrix = data_dict['confusion_matrix']
    
    # Generate heatmap
    ax = sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                     xticklabels=labels, yticklabels=labels,
                     annot_kws={'size': 14})
    
    # Typography and styling
    plt.title(f'Confusion Matrix: {model_name} / {condition} / {skill_name}\n({classifier.capitalize()} Classifier)', 
              fontsize=14, pad=15)
    plt.xlabel('Predicted Grade', fontsize=12, labelpad=10)
    plt.ylabel('True Grade', fontsize=12, labelpad=10)
    
    # Save figure
    out_path = figures_dir / f"confusion_{model_name}_{condition}_{skill_name}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path

generated = []
for skill in skills:
    # Try voyage first, fallback to kalm
    for model in models_to_check:
        key_ord = f"{model}_{condition}_{skill}_ordinal"
        key_nom = f"{model}_{condition}_{skill}_nominal"
        
        if key_ord in matrices:
            out_file = plot_matrix(matrices[key_ord], model, skill)
            generated.append(out_file)
            print(f"Generated: {out_file.name}")
            break
        elif key_nom in matrices:
            out_file = plot_matrix(matrices[key_nom], model, skill)
            generated.append(out_file)
            print(f"Generated (fallback to nominal): {out_file.name}")
            break

if not generated:
    print("Warning: No matching matrices found for voyage or kalm-12b on C2a ordinal.")
else:
    print(f"\nSuccessfully saved {len(generated)} heatmaps to {figures_dir}")
