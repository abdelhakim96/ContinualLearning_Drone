import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# Configuration
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
#model_path = 'models/dnn_drone_dynamics_128x128.pth'
#properties_path = 'models/properties_dnn_drone_dynamics_128x128.csv'
model_path = 'models/dnn_drone_dynamics_256x256x256.pth'
properties_path = 'models/properties_dnn_drone_dynamics_256x256x256.csv'


# -----------------------------
# Load model properties
# -----------------------------
props = pd.read_csv(properties_path, index_col=0)

# Extract header (column names)
header = props.columns.tolist()

# Extract statistics
mu = props.loc['mu'].values.astype(float)
sigma = props.loc['sigma'].values.astype(float)
type_scaling = int(props.loc['type_scaling'].values[0])
num_inputs = int(props.loc['num_inputs'].values[0])
num_outputs = int(props.loc['num_outputs'].values[0])

# Split header into input and output columns
input_columns = header[:num_inputs]
output_columns = header[num_inputs:]

print(f"\nInput columns: {input_columns}")
print(f"Output columns: {output_columns}")

# -----------------------------
# Load model
# -----------------------------
from models.model import MLP

# Match the architecture from your training script
hidden_sizes = [128, 128]
model = MLP(num_inputs, hidden_sizes, num_outputs).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print(f"\nModel loaded successfully with architecture: {num_inputs} -> {hidden_sizes} -> {num_outputs}")

# -----------------------------
# Helper functions
# -----------------------------
def normalize_input(input_dict):
    """Convert raw input dict to normalized numpy array"""
    input_vec = np.zeros(num_inputs)
    
    for i, col_name in enumerate(input_columns):
        if col_name in input_dict:
            input_vec[i] = input_dict[col_name]
    
    # Apply same scaling as training
    if type_scaling == 1:
        input_normalized = (input_vec - mu[:num_inputs]) / sigma[:num_inputs]
    elif type_scaling == 2:
        data_min = mu[:num_inputs]  # Assuming you saved min/max properly
        data_max = sigma[:num_inputs]
        input_normalized = 2 * (input_vec - data_min) / (data_max - data_min) - 1
    else:
        input_normalized = input_vec
    
    return input_normalized

def denormalize_output(output_normalized):
    """Convert normalized network output to raw values"""
    if type_scaling == 1:
        output_raw = output_normalized * sigma[num_inputs:] + mu[num_inputs:]
    elif type_scaling == 2:
        data_min = mu[num_inputs:]
        data_max = sigma[num_inputs:]
        output_raw = (output_normalized + 1) * (data_max - data_min) / 2 + data_min
    else:
        output_raw = output_normalized
    
    return output_raw

def predict(input_dict):
    """Predict outputs given input dictionary"""
    with torch.no_grad():
        input_normalized = normalize_input(input_dict)
        input_tensor = torch.tensor(input_normalized, dtype=torch.float32).unsqueeze(0).to(device)
        output_normalized = model(input_tensor).cpu().numpy().flatten()
        output_raw = denormalize_output(output_normalized)
    
    # Map to command outputs
    return {
        'cmd_thrust': output_raw[0],
        'cmd_roll': output_raw[1],
        'cmd_pitch': output_raw[2],
        'cmd_yaw': output_raw[3]
    }

# -----------------------------
# Create ZERO-STATE base input (hovering drone)
# -----------------------------
base_input = {
    'cos_yaw': 1.0,      # yaw = 0
    'sin_yaw': 0.0,
    'vx': 0.0,           # hovering (no velocity)
    'vy': 0.0,
    'diff_x': 0.0,       # no position change desired
    'diff_y': 0.0,
    'diff_z': 0.0,
    'diff_yaw': 0.0      # no yaw change desired
}

print("\n" + "="*60)
print("BASE INPUT (Hovering drone at origin with zero yaw)")
print("="*60)
for k, v in base_input.items():
    print(f"  {k:12s} = {v:8.4f}")

# -----------------------------
# Define test cases
# -----------------------------
test_delta = 0.1  # 1 meter/radian change

test_cases = []

# Test altitude changes (diff_z)
test_cases.append(("climb_1m", {"diff_z": test_delta}))
test_cases.append(("descend_1m", {"diff_z": -test_delta}))

# Test forward/backward (diff_x in body frame)
test_cases.append(("forward_1m", {"diff_x": test_delta}))
test_cases.append(("backward_1m", {"diff_x": -test_delta}))

# Test left/right (diff_y in body frame)
test_cases.append(("right_1m", {"diff_y": test_delta}))
test_cases.append(("left_1m", {"diff_y": -test_delta}))

# Test yaw changes
test_cases.append(("yaw_cw_1rad", {"diff_yaw": test_delta}))
test_cases.append(("yaw_ccw_1rad", {"diff_yaw": -test_delta}))

# Test with initial velocities
test_cases.append(("moving_forward", {"vx": 2.0}))
test_cases.append(("moving_right", {"vy": 2.0}))

# Combined maneuvers
test_cases.append(("climb_forward", {"diff_z": test_delta, "diff_x": test_delta}))
test_cases.append(("yaw_right", {"diff_yaw": test_delta, "diff_y": test_delta}))

# -----------------------------
# Run tests
# -----------------------------
results = []

print("\n" + "="*60)
print("RUNNING TEST CASES")
print("="*60)

for name, modifications in test_cases:
    # Create test input
    test_input = base_input.copy()
    test_input.update(modifications)
    
    # Predict
    output = predict(test_input)
    
    results.append({
        "test_case": name,
        "modifications": modifications,
        "cmd_thrust": output['cmd_thrust'],
        "cmd_roll": output['cmd_roll'],
        "cmd_pitch": output['cmd_pitch'],
        "cmd_yaw": output['cmd_yaw']
    })
    
    print(f"\n{name}:")
    print(f"  Modifications: {modifications}")
    print(f"  Commands -> thrust: {output['cmd_thrust']:.4f}, roll: {output['cmd_roll']:.4f}, "
          f"pitch: {output['cmd_pitch']:.4f}, yaw: {output['cmd_yaw']:.4f}")

# -----------------------------
# Physics-based sanity checks
# -----------------------------
TOLERANCE = 0.1  # Tolerance for near-zero checks

def check_physics(results):
    """Verify predictions follow physical laws for COMMAND outputs"""
    checks = []
    
    for r in results:
        test_name = r["test_case"]
        
        # ---- Altitude control (diff_z) ----
        if test_name == "climb_1m":
            # To climb, need higher thrust command
            passed = r["cmd_thrust"] > TOLERANCE
            checks.append({
                "test": test_name,
                "check": "cmd_thrust > 0 (climb)",
                "value": r["cmd_thrust"],
                "passed": passed
            })
        
        elif test_name == "descend_1m":
            # To descend, need lower thrust command
            passed = r["cmd_thrust"] < -TOLERANCE
            checks.append({
                "test": test_name,
                "check": "cmd_thrust < 0 (descend)",
                "value": r["cmd_thrust"],
                "passed": passed
            })
        
        # ---- Forward/backward motion (diff_x) ----
        elif test_name == "forward_1m":
            # Forward motion requires negative pitch command (nose down)
            passed = r["cmd_pitch"] < -TOLERANCE
            checks.append({
                "test": test_name,
                "check": "cmd_pitch < 0 (nose down)",
                "value": r["cmd_pitch"],
                "passed": passed
            })
        
        elif test_name == "backward_1m":
            # Backward motion requires positive pitch command (nose up)
            passed = r["cmd_pitch"] > TOLERANCE
            checks.append({
                "test": test_name,
                "check": "cmd_pitch > 0 (nose up)",
                "value": r["cmd_pitch"],
                "passed": passed
            })
        
        # ---- Left/right motion (diff_y) ----
        elif test_name == "right_1m":
            # Right motion requires positive roll command (right wing down)
            passed = r["cmd_roll"] > TOLERANCE
            checks.append({
                "test": test_name,
                "check": "cmd_roll > 0 (right)",
                "value": r["cmd_roll"],
                "passed": passed
            })
        
        elif test_name == "left_1m":
            # Left motion requires negative roll command (left wing down)
            passed = r["cmd_roll"] < -TOLERANCE
            checks.append({
                "test": test_name,
                "check": "cmd_roll < 0 (left)",
                "value": r["cmd_roll"],
                "passed": passed
            })
        
        # ---- Yaw control (diff_yaw) ----
        elif test_name == "yaw_cw_1rad":
            # Clockwise yaw requires positive yaw command
            passed = r["cmd_yaw"] > TOLERANCE
            checks.append({
                "test": test_name,
                "check": "cmd_yaw > 0 (CW)",
                "value": r["cmd_yaw"],
                "passed": passed
            })
        
        elif test_name == "yaw_ccw_1rad":
            # Counter-clockwise yaw requires negative yaw command
            passed = r["cmd_yaw"] < -TOLERANCE
            checks.append({
                "test": test_name,
                "check": "cmd_yaw < 0 (CCW)",
                "value": r["cmd_yaw"],
                "passed": passed
            })
    
    return checks

# Run physics checks
physics_checks = check_physics(results)

# -----------------------------
# Create PASS/FAIL table
# -----------------------------
print("\n" + "="*60)
print("PHYSICS SANITY CHECKS")
print("="*60)

table_rows = []
cell_colors = []

for check in physics_checks:
    status = "PASS" if check["passed"] else "FAIL"
    color = "lightgreen" if check["passed"] else "lightcoral"
    
    table_rows.append([
        check["test"],
        check["check"],
        f"{check['value']:.4f}",
        status
    ])
    
    cell_colors.append(["white", "white", "white", color])
    
    print(f"{check['test']:20s} | {check['check']:25s} | "
          f"Value: {check['value']:7.4f} | {status}")

# Calculate pass rate
total_checks = len(physics_checks)
passed_checks = sum(1 for c in physics_checks if c["passed"])
pass_rate = (passed_checks / total_checks * 100) if total_checks > 0 else 0

print(f"\n{'='*60}")
print(f"PASS RATE: {passed_checks}/{total_checks} ({pass_rate:.1f}%)")
print(f"{'='*60}\n")

# -----------------------------
# Visualization 1: PASS/FAIL Table
# -----------------------------
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.axis("off")

table = ax1.table(
    cellText=table_rows,
    colLabels=["Test Case", "Physics Check", "Value", "Result"],
    cellColours=cell_colors,
    loc="center",
    cellLoc="left"
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.0)

# Make header bold
for i in range(4):
    table[(0, i)].set_facecolor("lightgray")
    table[(0, i)].set_text_props(weight='bold')

plt.title(
    f"Forward Dynamics Sanity Check – Pass Rate: {passed_checks}/{total_checks} ({pass_rate:.1f}%)",
    fontsize=14,
    fontweight='bold',
    pad=20
)

plt.tight_layout()

# -----------------------------
# Visualization 2: Output values by test case
# -----------------------------
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

test_names = [r["test_case"] for r in results]
x = np.arange(len(results))

# cmd_thrust
axes[0, 0].bar(x, [r["cmd_thrust"] for r in results], color='skyblue', edgecolor='black')
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
axes[0, 0].set_ylabel('Thrust Command', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Thrust Command (cmd_thrust)', fontsize=12, fontweight='bold')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(test_names, rotation=45, ha='right', fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

# cmd_roll
axes[0, 1].bar(x, [r["cmd_roll"] for r in results], color='lightgreen', edgecolor='black')
axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
axes[0, 1].set_ylabel('Roll Command (rad)', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Roll Command (cmd_roll)', fontsize=12, fontweight='bold')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(test_names, rotation=45, ha='right', fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# cmd_pitch
axes[1, 0].bar(x, [r["cmd_pitch"] for r in results], color='lightcoral', edgecolor='black')
axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
axes[1, 0].set_ylabel('Pitch Command (rad)', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Pitch Command (cmd_pitch)', fontsize=12, fontweight='bold')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(test_names, rotation=45, ha='right', fontsize=9)
axes[1, 0].grid(True, alpha=0.3)

# cmd_yaw
axes[1, 1].bar(x, [r["cmd_yaw"] for r in results], color='plum', edgecolor='black')
axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
axes[1, 1].set_ylabel('Yaw Command (rad/s)', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Yaw Command (cmd_yaw)', fontsize=12, fontweight='bold')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(test_names, rotation=45, ha='right', fontsize=9)
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Forward Dynamics Network Command Outputs Across Test Cases', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()

# -----------------------------
# Save figures
# -----------------------------
fig1.savefig('forward_dynamics_pass_fail_table.png', dpi=150, bbox_inches='tight')
fig2.savefig('forward_dynamics_outputs_visualization.png', dpi=150, bbox_inches='tight')

print("Figures saved:")
print("  - forward_dynamics_pass_fail_table.png")
print("  - forward_dynamics_outputs_visualization.png")

plt.show()
