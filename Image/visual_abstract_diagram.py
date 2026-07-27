import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_visual_abstract():
    # Compact width, slightly increased height
    fig, ax = plt.subplots(figsize=(11, 8), dpi=600)
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(-0.2, 9)

    c_rna = '#E3F2FD'
    c_r_edge = '#64B5F6'
    
    c_tcr = '#E8F5E9'
    c_t_edge = '#81C784'
    
    c_int = '#F3E5F5'
    c_i_edge = '#BA68C8'
    
    c_pc6 = '#FFF3E0'
    c_p_edge = '#FFB74D'
    
    c_base = '#F5F5F5'
    c_b_edge = '#BDBDBD'

    def draw_box(center, text, width, height, facecolor, edgecolor, fontsize=14, bold=False):
        x = center[0] - width / 2
        y = center[1] - height / 2
        box = patches.FancyBboxPatch((x, y), width, height, 
                                     boxstyle="round,pad=0.2,rounding_size=0.3",
                                     linewidth=2.0, edgecolor=edgecolor, facecolor=facecolor, zorder=3)
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        ax.text(center[0], center[1], text, ha='center', va='center', fontsize=fontsize, 
                color='#1F2937', fontweight=weight, zorder=4, family='sans-serif', linespacing=1.3)
        return (x, x+width, y, y+height)

    def draw_arrow(start, end, connectionstyle="arc3,rad=0", linestyle='-'):
        ax.annotate("", xy=end, xycoords='data', xytext=start, textcoords='data',
                    arrowprops=dict(arrowstyle="->,head_length=0.8,head_width=0.6", 
                                    color="#6B7280", lw=2.5, shrinkA=8, shrinkB=8, 
                                    connectionstyle=connectionstyle, linestyle=linestyle), zorder=2)

    w, h = 3.2, 1.8
    p_in = (2.0, 4)
    p_rna = (6.5, 6.2)
    p_tcr = (6.5, 2.0)
    p_pc6 = (6.5, 8.4)
    p_int = (11.0, 4)
    
    draw_box(p_in, "Peripheral Blood\n\nSingle-cell\nRNA + TCR", 3.0, 2.0, c_base, c_b_edge, bold=True)
    draw_box(p_rna, "RNA Pipeline\n\nPCA\n(PC1–PC50)", w, h, c_rna, c_r_edge, bold=True)
    draw_box(p_tcr, "TCR Pipeline\n\nSequence\nEncoding", w, h, c_tcr, c_t_edge, bold=True)
    draw_box(p_int, "Deep Learning\n(MLP)\n\nUnsupervised Learning\n(UMAP)\n\n Response vs\nNon-response", 3.4, 3, c_int, c_i_edge, bold=True)

    draw_box(p_pc6, "PC6 → Mitochondrial activity", 3.8, 0.8, c_pc6, c_p_edge, fontsize=14, bold=True)

    draw_arrow((p_in[0]+1.5, p_in[1]), (p_rna[0]-1.75, p_rna[1]), connectionstyle="arc3,rad=-0.2")
    draw_arrow((p_in[0]+1.5, p_in[1]), (p_tcr[0]-1.75, p_tcr[1]), connectionstyle="arc3,rad=0.2")
    
    draw_arrow((p_rna[0]+1.6, p_rna[1]), (p_int[0]-1.85, p_int[1]), connectionstyle="arc3,rad=0.15")
    draw_arrow((p_tcr[0]+1.6, p_tcr[1]), (p_int[0]-1.85, p_int[1]), connectionstyle="arc3,rad=-0.15")
    
    draw_arrow((p_pc6[0], p_pc6[1]-0.4), (p_rna[0], p_rna[1]+1), linestyle='--')

    phases = ["1. Input", "2. Feature Extraction", "3. Integration & Prediction"]
    x_pos = [2.0, 6.5, 11.0]
    for x, text in zip(x_pos, phases):
        ax.text(x, -0.2, text.upper(), ha='center', va='center', fontsize=14, 
                color='#6B7280', fontweight='bold', family='sans-serif')

    plt.tight_layout()
    plt.savefig('visual_abstract.png', bbox_inches='tight', dpi=1200)
    plt.savefig('visual_abstract.pdf', bbox_inches='tight', dpi=1200)
    print("Visual abstract saved as visual_abstract.png and visual_abstract.pdf")

if __name__ == "__main__":
    create_visual_abstract()
