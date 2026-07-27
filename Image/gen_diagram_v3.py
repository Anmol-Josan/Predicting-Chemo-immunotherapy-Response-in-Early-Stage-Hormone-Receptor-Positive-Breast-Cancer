import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

# Use a standard font
mpl.rcParams['font.family'] = 'sans-serif'

def create_visual_abstract():
    # Wider figure to fit the linear/branched layout seamlessly
    fig, ax = plt.subplots(figsize=(15, 7), dpi=600)
    ax.axis('off')
    ax.set_xlim(0, 16.5)
    ax.set_ylim(-0.8, 9.2)

    c_base = '#F5F5F5'     # Gray
    c_b_edge = '#BDBDBD'
    c_seq = '#E8EAF6'      # Light Indigo
    c_s_edge = '#9FA8DA'
    c_rna = '#E3F2FD'      # Blue
    c_r_edge = '#64B5F6'
    c_tcr = '#E8F5E9'      # Green
    c_t_edge = '#81C784'
    c_int = '#F3E5F5'      # Purple
    c_i_edge = '#BA68C8'
    c_out = '#FFF9C4'      # Yellow theme for outcome
    c_o_edge = '#FFF176'
    c_pc6 = '#FFF3E0'      # Orange
    c_p_edge = '#FFB74D'

    def draw_box(center, text, width, height, facecolor, edgecolor, fontsize=14, bold=False):
        x = center[0] - width / 2
        y = center[1] - height / 2
        box = patches.FancyBboxPatch((x, y), width, height, 
                                     boxstyle='round,pad=0.2,rounding_size=0.3',
                                     linewidth=2.5, edgecolor=edgecolor, facecolor=facecolor, zorder=3)
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        ax.text(center[0], center[1], text, ha='center', va='center', fontsize=fontsize, 
                color='#1F2937', fontweight=weight, zorder=4, family='sans-serif', linespacing=1.4)
        return (x, x+width, y, y+height)

    def draw_arrow(start, end, connectionstyle='arc3,rad=0', linestyle='-'):
        ax.annotate('', xy=end, xycoords='data', xytext=start, textcoords='data',
                    arrowprops=dict(arrowstyle='->,head_length=0.9,head_width=0.7', 
                                    color='#6B7280', lw=3.0, shrinkA=8, shrinkB=8, 
                                    connectionstyle=connectionstyle, linestyle=linestyle), zorder=2)

    # Coords for internal boxes
    w_box, h_box = 2.6, 1.8
    
    p_blood = (1.5, 4.0)
    p_seq = (4.5, 4.0)
    p_unsup = (8.5, 6.2)
    p_feat = (8.5, 1.8)
    p_pred = (12.5, 4.0)
    p_out = (15.5, 4.0)
    
    p_pc6 = (8.5, 8.4)
    
    # Draw Internal Label Boxes
    draw_box(p_blood, 'Peripheral\nblood', 2.2, h_box, c_base, c_b_edge, bold=True)
    draw_box(p_seq, 'Single-cell\nRNA + TCR', 2.4, h_box, c_seq, c_s_edge, bold=True)
    draw_box(p_unsup, 'Unsupervised\ndiscovery', w_box, h_box, c_rna, c_r_edge, bold=True)
    draw_box(p_feat, 'Feature\nextraction', w_box, h_box, c_tcr, c_t_edge, bold=True)
    draw_box(p_pred, 'Supervised\nprediction', w_box, h_box, c_int, c_i_edge, bold=True)
    draw_box(p_out, 'Response\noutcome', 2.2, h_box, c_out, c_o_edge, bold=True)
    
    # Draw PC6 Highlight Box
    draw_box(p_pc6, '★ PC6 → Mitochondrial activity', 3.8, 0.8, c_pc6, c_p_edge, fontsize=13, bold=True)

    # Draw Arrows
    draw_arrow((p_blood[0]+1.1, p_blood[1]), (p_seq[0]-1.2, p_seq[1]))
    
    # Splitting to RNA / TCR paths
    draw_arrow((p_seq[0]+1.2, p_seq[1]), (p_unsup[0]-1.3, p_unsup[1]), connectionstyle='arc3,rad=-0.15')
    draw_arrow((p_seq[0]+1.2, p_seq[1]), (p_feat[0]-1.3, p_feat[1]), connectionstyle='arc3,rad=0.15')
    
    # Merging back into prediction
    draw_arrow((p_unsup[0]+1.3, p_unsup[1]), (p_pred[0]-1.3, p_pred[1]), connectionstyle='arc3,rad=0.15')
    draw_arrow((p_feat[0]+1.3, p_feat[1]), (p_pred[0]-1.3, p_pred[1]), connectionstyle='arc3,rad=-0.15')
    
    # Final step
    draw_arrow((p_pred[0]+1.3, p_pred[1]), (p_out[0]-1.1, p_out[1]))
    
    # PC6 to Unsupervised discovery
    draw_arrow((p_pc6[0], p_pc6[1]-0.4), (p_unsup[0], p_unsup[1]+0.9), linestyle='--')

    # Draw the exactly 4 bottom Phase labels mapping the steps
    phases = [
        (1.5, 'Peripheral blood'),
        (4.5, 'scRNA-seq\n+ TCR-seq'),
        (8.5, 'Feature engineering\nand clustering'),
        (13.6, 'ML prediction\nof response') # covers both modeling and output phases roughly
    ]
    
    for x, text in phases:
        ax.text(x, -0.4, text, ha='center', va='center', fontsize=14, 
                color='#6B7280', fontweight='bold', family='sans-serif', linespacing=1.3)
        ax.plot([x-1.0, x+1.0], [0.3, 0.3], color='#E0E0E0', lw=2)

    plt.tight_layout()
    plt.savefig('visual_abstract.png', bbox_inches='tight', dpi=600)
    plt.savefig('visual_abstract.pdf', bbox_inches='tight', dpi=600)
    print('Visual abstract correctly structured with exactly 4 base steps!')

if __name__ == '__main__':
    create_visual_abstract()
