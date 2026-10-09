import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Diagram Generation
fig, ax = plt.subplots(figsize=(10, 8))
ax.axis('off')

def draw_box(ax, x, y, width, height, text, facecolor='lightblue'):
    rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='black', facecolor=facecolor)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=10, weight='bold')

# Input
draw_box(ax, 0.35, 0.9, 0.3, 0.08, "Input\n[B, 240, 12]", facecolor='#e0e0e0')
ax.annotate('', xy=(0.5, 0.88), xytext=(0.5, 0.9), arrowprops=dict(arrowstyle="->"))

# TCN Blocks
y_pos = 0.78
for i, d in enumerate([1, 2, 4, 8, 16]):
    draw_box(ax, 0.25, y_pos, 0.5, 0.08, f"TCN Block {i+1}\n(Conv1D, F=32, K=3, Dilation={d})\n+ BatchNorm & ReLU + Residual", facecolor='#ffcc99')
    ax.annotate('', xy=(0.5, y_pos-0.02), xytext=(0.5, y_pos), arrowprops=dict(arrowstyle="->"))
    y_pos -= 0.12

# Shared Representation
draw_box(ax, 0.3, y_pos, 0.4, 0.08, "Shared Temporal Features\n[B, 240, 32]", facecolor='#ccffcc')

# Arrows splitting to heads
ax.annotate('', xy=(0.2, y_pos-0.02), xytext=(0.5, y_pos), arrowprops=dict(arrowstyle="->"))
ax.annotate('', xy=(0.5, y_pos-0.02), xytext=(0.5, y_pos), arrowprops=dict(arrowstyle="->"))
ax.annotate('', xy=(0.8, y_pos-0.02), xytext=(0.5, y_pos), arrowprops=dict(arrowstyle="->"))

y_pos -= 0.12

# Heads
draw_box(ax, 0.05, y_pos, 0.25, 0.08, "Elbow Regression Head\nConv1D (K=1, F=1)", facecolor='#ffb3e6')
draw_box(ax, 0.375, y_pos, 0.25, 0.08, "Forearm Regression Head\nConv1D (K=1, F=2)", facecolor='#ffb3e6')
draw_box(ax, 0.7, y_pos, 0.25, 0.08, "Classification Head\nConv1D(K=1, F=20) + GlobalMaxPool1D", facecolor='#cce5ff')

# Outputs
ax.annotate('', xy=(0.175, y_pos-0.02), xytext=(0.175, y_pos), arrowprops=dict(arrowstyle="->"))
ax.annotate('', xy=(0.5, y_pos-0.02), xytext=(0.5, y_pos), arrowprops=dict(arrowstyle="->"))
ax.annotate('', xy=(0.825, y_pos-0.02), xytext=(0.825, y_pos), arrowprops=dict(arrowstyle="->"))

y_pos -= 0.1

draw_box(ax, 0.05, y_pos, 0.25, 0.08, "Elbow Angle\n[B, 240, 1]", facecolor='#e0e0e0')
draw_box(ax, 0.375, y_pos, 0.25, 0.08, "Forearm Sin/Cos\n[B, 240, 2]", facecolor='#e0e0e0')
draw_box(ax, 0.7, y_pos, 0.25, 0.08, "Exercise Logits\n[B, 20]", facecolor='#e0e0e0')

plt.title("Edge-AI Multi-Task TCN Architecture", fontsize=14, weight='bold')
plt.savefig(r"D:\me\project\data_audit_reports\tcn_architecture_diagram.png", bbox_inches='tight')

# Parameter calculation
# Conv1D: parameters = (kernel_size * input_channels + 1) * output_channels
# TCN Block 1: F_in=12, F_out=32. Layer1: (3*12+1)*32 = 1184. Layer2: (3*32+1)*32 = 3104. 
# Projection (residual): (1*12+1)*32 = 384. BN parameters: ~128. Total ~4800.
# Block 2..5: F_in=32, F_out=32. Layer1: (3*32+1)*32=3104. Layer2: 3104. BN: 128. No projection. Total ~6336 per block.
# 4 blocks * 6336 = 25344. Total shared = 4800 + 25344 = 30144.
# Elbow Head: (1*32+1)*1 = 33 params.
# Forearm Head: (1*32+1)*2 = 66 params.
# Classification Head: (1*32+1)*20 = 660 params.
# Total params = ~30,903. Very small! Perfectly suitable for Edge AI.
