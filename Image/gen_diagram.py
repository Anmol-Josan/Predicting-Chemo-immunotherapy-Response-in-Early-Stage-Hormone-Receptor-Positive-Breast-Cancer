import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

# Use a standard font
mpl.rcParams['font.family'] = 'sans-serif'

def create_visual_abstract():
    # Slightly wider limits to accommodate the separated steps laterally, maintaining aspect ratio
    fig, ax = plt.subplots(figsize=(11.5, 8), dpi=600)
    ax.axis('off')
    ax.set_xlim(-0.2, 14.8)
    ax.set_ylim(-0.8, 9.2)

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

    def draw_box(center, text, width, height, facecolor, edgecolor, fontsize=15, bold=False):
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

    # Coordinates & Dimensions
    w_in, h_in = 2.8, 2.0
    p_in = (1.4, 4)
    
    w_mid, h_mid = 3.4, 2.0
    p_rna = (5.6, 6.4)
    p_tcr = (5.6, 1.8)
    
    p_pc6 = (5.6, 8.6)
    
    w_pred, h_pred = 3.2, 2.0
    p_pred = (9.8, 4)
    
    w_out, h_out = 2.8, 2.0
    p_out = (13.4, 4)
    
    # Draw Boxes using specifically requested labels
    draw_box(p_in, 'Peripheral blood\n\n[+] Single-cell\nRNA + TCR', w_in, h_in, c_base, c_b_edge, bold=True, fontsize=14)
    
    draw_box(p_rna, 'Unsupervised discovery\n\n[+] RNA PCA\n(PC1-PC50)', w_mid, h_mid, c_rna, c_r_edge, bold=True, fontsize=14)
    draw_box(p_tcr, 'Feature extraction\n\n[+] TCR Sequence\nEncoding', w_mid, h_mid, c_tcr, c_t_edge, bold=True, fontsize=14)
    
    draw_box(p_pred, 'Supervised prediction\n\n[+] Deep Learning\n(MLP)', w_pred, h_pred, c_int, c_i_edge, bold=True, fontsize=14)
    draw_box(p_out, 'Response outcome\n\n[+] Response vs\nNon-response', w_out, h_out, c_int, c_i_edge, bold=True, fontsize=14)
    
    # PC6 Bubble annotation
    draw_box(p_pc6, '★ PC6 → Mitochondrial activity', 3.8, 0.8, c_pc6, c_p_edge, fontsize=13, bold=True)

    # Draw Arrows
    draw_arrow((p_in[0]+w_in/2, p_in[1]), (p_rna[0]-w_mid/2, p_rna[1]), connectionstyle='arc3,rad=-0.2')
    draw_arrow((p_in[0]+w_in/2, p_in[1]), (p_tcr[0]-w_mid/2, p_tcr[1]), connectionstyle='arc3,rad=0.2')
    
    draw_arrow((p_rna[0]+w_mid/2, p_rna[1]), (p_pred[0]-w_pred/2, p_pred[1]), connectionstyle='arc3,rad=0.15')
    draw_arrow((p_tcr[0]+w_mid/2, p_tcr[1]), (p_pred[0]-w_pred/2, p_pred[1]), connectionstyle='arc3,rad=-0.15')
    
    draw_arrow((p_pred[0]+w_pred/2, p_pred[1]), (p_out[0]-w_out/2, p_out[1]))
    
    draw_arrow((p_pc6[0], p_pc6[1]-0.4), (p_rna[0], p_rna[1]+h_mid/2+0.05), linestyle='--')

    # Phase Labels at the bottom
    phases = ['1. Input', '2. Extraction & Discovery', '3. Modeling', '4. Outcome']
    x_pos = [1.4, 5.6, 9.8, 13.4]
    for x, text in zip(x_pos, phases):
        ax.text(x, -0.4, text.upper(), ha='center', va='center', fontsize=13, 
                color='#6B7280', fontweight='bold', family='sans-serif', linespacing=1.4)

    plt.tight_layout()
    plt.savefig('visual_abstract.png', bbox_inches='tight', dpi=600)
    plt.savefig('visual_abstract.pdf', bbox_inches='tight', dpi=600)
    print('Visual abstract successfully updated and saved.')

if __name__ == '__main__':
    create_visual_abstract()
