"""Convert KernelBench Level 2/3 to DirecTune format.

Extracts Model weights at conversion time, saves to .pt file, and generates
an EXPANDED reference that shows the computation with named weight tensors.
This lets the LLM see weight names/shapes and generate correct Triton.
"""

import argparse
import ast
import csv
import importlib.util
import json
import os
import re
import sys
import tempfile
from pathlib import Path


def extract_constants(source: str) -> dict:
    """Extract module-level constants including tuple-unpacking assignments.

    Handles both simple assignments (``batch_size = 16``) and tuple unpacking
    (``depth, height, width = 24, 48, 48``).
    """
    tree = ast.parse(source)
    constants = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        val = ast.literal_eval(node.value)
                        constants[target.id] = val
                    except (ValueError, TypeError):
                        pass
                elif isinstance(target, ast.Tuple):
                    # Handle tuple unpacking: depth, height, width = 24, 48, 48
                    try:
                        val = ast.literal_eval(node.value)
                        if isinstance(val, (tuple, list)):
                            for name_el, val_el in zip(target.elts, val):
                                if isinstance(name_el, ast.Name):
                                    constants[name_el.id] = val_el
                    except (ValueError, TypeError):
                        pass
    return constants


def _get_input_shape_dynamic(source: str) -> list | None:
    """Execute ``get_inputs()`` in a subprocess to obtain concrete tensor shapes.

    Used as fallback when static analysis leaves symbolic dimension names unresolved.
    Returns None on any failure.
    """
    import subprocess as _sp
    script = (
        "import sys\n"
        + source
        + "\ninputs = get_inputs()\n"
          "shapes = [list(t.shape) for t in inputs]\n"
          "print(repr(shapes))\n"
    )
    try:
        proc = _sp.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode == 0:
            shapes = eval(proc.stdout.strip())
            if isinstance(shapes, list) and len(shapes) > 0:
                return list(shapes[0])
    except Exception:
        pass
    return None


def get_input_shape(source: str) -> list:
    inputs_match = re.search(r'def get_inputs\(\):\s*return\s*\[(.*?)\]', source, re.DOTALL)
    if inputs_match:
        body = inputs_match.group(1)
        shape_match = re.search(r'torch\.rand(?:n)?\((.*?)\)', body)
        if shape_match:
            args = shape_match.group(1)
            constants = extract_constants(source)
            parts = [p.strip() for p in args.split(',')]
            resolved = []
            for p in parts:
                p = p.strip()
                if p in constants:
                    resolved.append(constants[p])
                else:
                    try:
                        resolved.append(int(p))
                    except ValueError:
                        resolved.append(p)
            # If any dimension is still a string, try dynamic execution as fallback
            if any(isinstance(d, str) for d in resolved):
                dyn = _get_input_shape_dynamic(source)
                if dyn is not None:
                    return dyn
            return resolved
    return [1, 3, 224, 224]


def load_model(source: str) -> tuple:
    """Import Module from source, instantiate Model with seed=42. Returns (model, state_dict)."""
    tmpdir = tempfile.mkdtemp()
    tmpfile = os.path.join(tmpdir, "_kb_model.py")
    with open(tmpfile, "w") as f:
        f.write(source)

    try:
        import torch
        spec = importlib.util.spec_from_file_location("_kb_model", tmpfile)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_kb_model"] = mod
        torch.manual_seed(42)
        spec.loader.exec_module(mod)
        init_inputs = mod.get_init_inputs()
        model = mod.Model(*init_inputs)
        state = {k: v.clone() for k, v in model.state_dict().items()}
        return model, state
    finally:
        try:
            os.unlink(tmpfile)
            os.rmdir(tmpdir)
        except OSError:
            pass


def expand_sequential(state: dict, source: str) -> str:
    """For nn.Sequential models, generate explicit torch code with named weights."""
    # Parse forward method for layer structure
    fwd_match = re.search(r'def forward\(self,\s*\w+\s*\):(.*?)(?=\n\w|\n# Test|\Z)', source, re.DOTALL)
    fwd_body = fwd_match.group(1).strip() if fwd_match else ""

    weight_vars = []
    op_lines = []
    layer_idx = 0

    # Map state_dict keys to variable names
    for key in sorted(state.keys()):
        name = key.replace('.', '_')
        weight_vars.append(f"{name} = _weights['{key}'].to(_device)")
        op_lines.append(f"# {name}: {list(state[key].shape)}")

    # Build explicit forward based on nn.Sequential patterns
    if 'Sequential' in source or 'self.network' in source:
        expanded = f'''
# --- EXPANDED REFERENCE (shows actual computation) ---
# Use this as the reference for kernel generation.
# Triton kernel should implement the same operations using these weight tensors.

import torch as _torch
_weights_path = "{_weights_path_marker}"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _device = device
    _weights = {{k: v.to(device) for k, v in _torch.load(_weights_path, map_location='cpu', weights_only=True).items()}}

def run(x):
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)
'''
        # Add weight variable declarations
        for key in sorted(state.keys()):
            var = key.replace('.', '_')
            shape = list(state[key].shape)
            expanded += f"    {var} = _weights['{key}']  # {shape}\n"

        # Add operations based on Sequential analysis
        expanded += '\n'
        # For MLP: network.0 (Linear) → network.1 (ReLU) → network.2 (Linear) → ...
        layers = re.findall(r'network\.(\d+)', source) or re.findall(r'layers\.(\d+)', source)
        seen_relu = False
        state_keys = sorted(state.keys())
        for sk in state_keys:
            if 'weight' in sk:
                base = sk.replace('.weight', '')
                bias_key = sk.replace('weight', 'bias')
                has_bias = bias_key in state
                w_var = sk.replace('.', '_')
                b_var = bias_key.replace('.', '_') if has_bias else None
                expanded += f"    x = _torch.nn.functional.linear(x, {w_var}"
                if has_bias:
                    expanded += f", {b_var}"
                expanded += ")\n"
                # Check if next layer after this Linear is ReLU
                layer_num = int(re.search(r'network\.(\d+)', sk).group(1))
                next_layer = f'network.{layer_num + 1}'
                # Check source for ReLU after this layer
                # Check if this is NOT the last Linear layer
                is_last = (sk == state_keys[-2] or sk == state_keys[-1])  # last weight or bias
                if not is_last:
                    expanded += "    x = _torch.relu(x)\n"

        expanded += "    return x\n"
        return expanded

    # Generic fallback: show weight access + Model-based forward
    return generate_generic_expanded(state, source)


def _extract_layer_types(source: str) -> dict[str, str]:
    """Parse __init__ to map self.attr_name → layer class name.

    Returns dict like ``{'conv_transpose': 'ConvTranspose3d', 'conv': 'Conv2d'}``.
    """
    layer_map: dict[str, str] = {}
    init_match = re.search(
        r'def __init__\(self,.*?\):(.*?)(?=\n    def |\n\w)', source, re.DOTALL
    )
    if not init_match:
        return layer_map
    # Known nn layer classes whose weights can be used in F.* calls
    KNOWN_LAYERS = (
        r'Conv(?:Transpose)?[123]d|Linear|BatchNorm[123]d|LayerNorm|GroupNorm|'
        r'MaxPool[123]d|AvgPool[123]d|'
        r'LeakyReLU|ReLU|GELU|Sigmoid|Tanh|Mish|Softmax|'
        r'Hardtanh|HardSwish|Swish|LogSumExp|Dropout'
    )
    for line in init_match.group(1).split('\n'):
        m = re.search(rf'self\.(\w+)\s*=\s*nn\.({KNOWN_LAYERS})\s*\(', line)
        if m:
            layer_map[m.group(1)] = m.group(2)
    return layer_map


def _layer_op_call(layer_class: str, var: str, bias_var: str | None) -> str | None:
    """Return the appropriate ``_torch.nn.functional.*`` call for a layer type.

    Returns None if the layer type is unknown (caller should fall back).
    """
    lc = layer_class.lower()
    # Linear / Matmul
    if 'linear' in lc or 'gemm' in lc:
        if bias_var:
            return f"    x = _torch.nn.functional.linear(x, {var}, {bias_var})"
        return f"    x = _torch.nn.functional.linear(x, {var})"
    # ConvTranspose{1,2,3}d
    if 'convtranspose' in lc:
        # Determine 1d/2d/3d from class name
        for dim in ('3d', '2d', '1d'):
            if dim in lc:
                break
        else:
            dim = '3d'
        return f"    x = _torch.nn.functional.conv_transpose{dim}(x, {var}" + (f", {bias_var}" if bias_var else "") + ")"
    # Conv{1,2,3}d
    if 'conv' in lc:
        for dim in ('3d', '2d', '1d'):
            if dim in lc:
                break
        else:
            dim = '2d'
        return f"    x = _torch.nn.functional.conv{dim}(x, {var}" + (f", {bias_var}" if bias_var else "") + ")"
    # BatchNorm
    if 'batchnorm' in lc:
        # Use training=True so batch_norm computes statistics from input
        # (running_mean/running_var buffers aren't always available)
        return f"    x = _torch.nn.functional.batch_norm(x, running_mean=None, running_var=None, weight={var}, bias={bias_var}, training=True)"
    # GroupNorm
    if 'groupnorm' in lc:
        return f"    x = _torch.nn.functional.group_norm(x, num_groups=1, weight={var}, bias={bias_var})"
    # LayerNorm
    if 'layernorm' in lc:
        return f"    x = _torch.nn.functional.layer_norm(x, normalized_shape=x.shape[1:], weight={var}, bias={bias_var})"
    # MaxPool
    if 'maxpool' in lc:
        for dim in ('3d', '2d', '1d'):
            if dim in lc:
                break
        else:
            dim = '3d'
        return f"    x = _torch.nn.functional.max_pool{dim}(x, kernel_size=2)"
    # AvgPool
    if 'avgpool' in lc:
        for dim in ('3d', '2d', '1d'):
            if dim in lc:
                break
        else:
            dim = '3d'
        return f"    x = _torch.nn.functional.avg_pool{dim}(x, kernel_size=2)"
    return None


def _replace_self_call(line: str, attr: str, replacement: str, extra_args: str = '') -> str | None:
    """Replace ``self.attr(expr)`` with ``replacement(expr + extra_args)`` handling nested parens.

    Returns the modified line, or None if ``self.attr`` is not found.
    """
    prefix = f'self.{attr}('
    start = line.find(prefix)
    if start < 0:
        return None
    # Count parens from the opening paren to find the matching close
    paren_start = start + len(prefix) - 1  # position of '('
    depth = 0
    for i in range(paren_start, len(line)):
        ch = line[i]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                # Found matching close
                inner = line[paren_start + 1:i]
                return line[:start] + replacement + '(' + inner + extra_args + ')' + line[i + 1:]
    return None


def _safe_replace_torch(line: str) -> str:
    """Replace ``torch.`` with ``_torch.``, but not ``_torch.`` (no double-replace)."""
    # Only replace standalone torch., not torch. that's already _torch.
    return re.sub(r'(?<!_)torch\.', '_torch.', line)


def generate_generic_expanded(state: dict, source: str, model=None) -> str:
    """Generic expanded reference: load weights, show explicit torch ops."""

    # Build layer-type mapping from __init__
    layer_types = _extract_layer_types(source)

    # Extract constants from __init__ source (nn.Module.__dir__ filters non-registered attrs)
    constants = {}
    init_match = re.search(r'def __init__\(self,.*?\):(.*?)(?=\n    def |\n\w)', source, re.DOTALL)
    if init_match:
        for line in init_match.group(1).split('\n'):
            m = re.search(r'self\.(\w+)\s*=\s*([^#\n]+)', line)
            if m:
                name, val_str = m.group(1), m.group(2).strip()
                try:
                    val = ast.literal_eval(val_str)
                    if isinstance(val, (int, float, bool)):
                        constants[name] = val
                except (ValueError, SyntaxError):
                    pass
    # Also get from loaded model as fallback
    if model is not None:
        for name in dir(model):
            if name in constants:
                continue
            if not name.startswith('_') and name not in ('forward', 'training', 'dump_patches', 'T_destination'):
                try:
                    val = getattr(model, name)
                    if isinstance(val, (int, float, bool)):
                        constants[name] = val
                except Exception:
                    pass

    fwd_match = re.search(r'def forward\(self,\s*\w+\s*\):(.*?)(?=\n\w|\n# Test|\n\w+ = |\ndef |\Z)', source, re.DOTALL)
    fwd_lines = []
    if fwd_match:
        body = fwd_match.group(1)
        in_docstring = False
        for line in body.strip().split('\n'):
            stripped = line.strip()
            if not stripped:
                continue
            # Track multi-line docstrings
            if stripped.startswith('"""') or stripped.startswith("'''"):
                in_docstring = not in_docstring
                continue
            if in_docstring:
                continue
            # Skip docstring-ish lines (type annotations, param docs, etc.)
            if stripped.startswith('Args:') or stripped.startswith('Returns:') or stripped.startswith('Raises:'):
                continue
            if stripped.startswith('(') or '): ' in stripped:
                # Likely a docstring continuation line like "x (_torch.Tensor): ..."
                continue
            code = re.sub(r'^\s{8}', '', line) if line.startswith('        ') else stripped
            fwd_lines.append(code)

    weight_var_map = {key: key.replace('.', '_') for key in state.keys()}
    seen_layers = set()

    # Pre-process: for each state key, determine if it's a raw param (no dot) or module param
    param_entries = []
    for key in sorted(state.keys()):
        parts = key.split('.')
        if len(parts) == 1:
            # Raw nn.Parameter: "weight" → just use the tensor directly
            param_entries.append({'key': key, 'var': weight_var_map[key], 'type': 'raw_param'})
        elif parts[-1] == 'weight':
            base = '.'.join(parts[:-1])
            w_var = weight_var_map[key]
            b_key = f'{base}.bias'
            b_var = weight_var_map.get(b_key, None)
            if b_key in state:
                param_entries.append({'key': key, 'var': w_var, 'bias_var': b_var, 'base': base, 'type': 'linear'})
            else:
                param_entries.append({'key': key, 'var': w_var, 'base': base, 'type': 'weight_only'})

    ops_code = []
    for line in fwd_lines:
        matched = False

        # Handle torch.matmul(x, self.X.T) for raw params
        if 'torch.matmul' in line and 'self.' in line:
            for entry in param_entries:
                if entry.get('type') == 'raw_param':
                    nm = entry['base'] if 'base' in entry else entry['key']
                    if f'self.{nm}' in line or f'self.{nm}.' in line:
                        ops_code.append(f"    x = _torch.matmul(x, {entry['var']}.T)")
                        matched = True
                        break
            if matched:
                continue

        if 'self.' in line:
            for entry in param_entries:
                base = entry.get('base', entry['key'])
                if f'self.{base}' in line and base not in seen_layers:
                    if entry.get('type') == 'raw_param':
                        # Raw nn.Parameter: replace self.X with the local weight variable
                        fixed = line.replace(f'self.{base}', entry['var'])
                        ops_code.append(f"    {fixed}")
                        seen_layers.add(base)
                        matched = True
                        break
                    else:
                        seen_layers.add(base)
                        # Use layer-type mapping to emit the correct F.* call
                        lc = layer_types.get(base, '')
                        call_code = _layer_op_call(lc, entry['var'], entry.get('bias_var'))
                        if call_code is not None:
                            ops_code.append(call_code)
                        elif entry['type'] == 'linear':
                            ops_code.append(f"    x = _torch.nn.functional.linear(x, {entry['var']}, {entry.get('bias_var')})")
                        else:
                            ops_code.append(f"    x = _torch.nn.functional.linear(x, {entry['var']})")
                        matched = True
                        break
            if not matched:
                # Try to rewrite self.X(...) using layer type mapping + constants
                fixed = line
                replaced_any = False
                # First try to replace self.X args (raw params used as scalars/multipliers)
                for entry in param_entries:
                    base = entry.get('base', entry['key'])
                    if entry.get('type') == 'raw_param' and f'self.{base}' in fixed:
                        fixed = fixed.replace(f'self.{base}', entry['var'])
                        replaced_any = True
                # Replace self.X() calls with F.* equivalents for activation/pooling modules
                for attr, lc in layer_types.items():
                    if f'self.{attr}' not in fixed:
                        continue
                    new_fixed = None
                    # For activation layers: self.relu(x) → F.relu(x)
                    if lc == 'LeakyReLU':
                        new_fixed = _replace_self_call(fixed, attr, '_torch.nn.functional.leaky_relu')
                    elif lc == 'ReLU':
                        new_fixed = _replace_self_call(fixed, attr, '_torch.relu')
                    elif lc == 'Sigmoid':
                        new_fixed = _replace_self_call(fixed, attr, '_torch.sigmoid')
                    elif lc == 'Tanh':
                        new_fixed = _replace_self_call(fixed, attr, '_torch.tanh')
                    elif lc == 'GELU':
                        new_fixed = _replace_self_call(fixed, attr, '_torch.nn.functional.gelu')
                    elif lc == 'Mish':
                        new_fixed = _replace_self_call(fixed, attr, '_torch.nn.functional.mish')
                    elif lc == 'Softmax':
                        new_fixed = _replace_self_call(fixed, attr, '_torch.softmax')
                    elif 'MaxPool' in lc:
                        for d in ('3d', '2d', '1d'):
                            if d in lc:
                                break
                        else:
                            d = '3d'
                        new_fixed = _replace_self_call(fixed, attr, f'_torch.nn.functional.max_pool{d}', extra_args=', kernel_size=2')
                    elif 'AvgPool' in lc:
                        for d in ('3d', '2d', '1d'):
                            if d in lc:
                                break
                        else:
                            d = '3d'
                        new_fixed = _replace_self_call(fixed, attr, f'_torch.nn.functional.avg_pool{d}', extra_args=', kernel_size=2')
                    if new_fixed is not None:
                        fixed = new_fixed
                        replaced_any = True
                # Replace remaining self.constant with literal values
                for cname, cval in sorted(constants.items(), key=lambda x: -len(x[0])):
                    if f'self.{cname}' in fixed:
                        fixed = fixed.replace(f'self.{cname}', str(cval))
                        replaced_any = True
                # Replace any remaining torch. → _torch. (must happen after self.* resolution)
                if 'torch.' in fixed:
                    fixed = _safe_replace_torch(fixed)
                    replaced_any = True
                if replaced_any:
                    ops_code.append(f"    {fixed}")
                    matched = True
        if matched:
            continue
        if 'torch.relu' in line:
            ops_code.append("    x = _torch.relu(x)")
        elif 'torch.sigmoid' in line:
            ops_code.append("    x = _torch.sigmoid(x)")
        elif 'torch.tanh' in line:
            ops_code.append("    x = _torch.tanh(x)")
        elif 'torch.softmax' in line:
            ops_code.append("    x = _torch.softmax(x, dim=-1)")
        elif 'torch.clamp' in line:
            ops_code.append(f"    {line.strip().replace('torch.', '_torch.')}")
        elif 'torch.logsumexp' in line:
            ops_code.append(f"    {line.strip().replace('torch.', '_torch.')}")
        elif 'torch.sum' in line:
            ops_code.append(f"    {line.strip().replace('torch.', '_torch.')}")
        elif 'mish' in line.lower() or 'nn.functional.mish' in line:
            ops_code.append("    x = _torch.nn.functional.mish(x)")
        elif 'torch.' in line:
            ops_code.append(f"    {line.strip().replace('torch.', '_torch.')}")
        elif 'x = x -' in line or 'x = x +' in line or 'x = x *' in line or 'x = x /' in line:
            ops_code.append(f"    {line.strip()}")
        elif 'return x' in line:
            ops_code.append("    return x")

    expanded = f'''
# --- EXPANDED REFERENCE (函数式，给 LLM 看) ---
# Model.forward() 的函数式展开（_weights + _torch.nn.functional.*）。
# LLM 写 Triton 时参考此函数式写法（torch.load(_weights_path) + 缓存 + 函数式计算）。
# 验证用下方的 run()（通用，调 Model），不要照抄 nn.Module。

import torch as _torch
_weights_path = "{_weights_path_marker}"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {{k: v.to(device) for k, v in _torch.load(_weights_path, map_location='cpu', weights_only=True).items()}}

def ref_expanded(x):
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)
'''
    for key in sorted(state.keys()):
        var = key.replace('.', '_')
        shape = list(state[key].shape)
        expanded += f"    {var} = _weights['{key}']  # {shape}\n"

    if ops_code:
        expanded += '\n' + '\n'.join(ops_code) + '\n'
    else:
        expanded += '\n    return x\n'

    # 通用 run()（验证用，调 Model.forward()，PyTorch 算正确，任何 op 组合都对）
    expanded += f'''
# --- UNIVERSAL RUN (验证用，调 Model.forward) ---
_model_cache = None
_model_device = None

def run(x):
    global _model_cache, _model_device
    if _model_cache is None or _model_device != str(x.device):
        _model_cache = Model(*get_init_inputs())
        _model_cache.load_state_dict(_torch.load(_weights_path, map_location='cpu', weights_only=True))
        _model_cache = _model_cache.to(x.device).eval()
        _model_device = str(x.device)
    return _model_cache(x)
'''
    return expanded


# Writable global for path injection
_weights_path_marker = ""


def process_one(filepath: str, out_dir: str) -> dict | None:
    global _weights_path_marker

    filename = os.path.basename(filepath)
    name = filename.replace('.py', '')

    with open(filepath) as f:
        source = f.read()

    inp_shape = get_input_shape(source)

    # Load model and extract weights
    try:
        import torch
        model, state_dict = load_model(source)
    except Exception as e:
        print(f"    WARNING: Could not load model: {e}")
        model, state_dict = None, None

    os.makedirs(out_dir, exist_ok=True)

    # Save weights to .pt file
    weights_path = os.path.abspath(os.path.join(out_dir, f"{name}_weights.pt"))
    _weights_path_marker = weights_path
    if state_dict:
        torch.save(state_dict, weights_path)

    # Generate expanded reference with explicit weight access
    if state_dict:
        if 'Sequential' in source and 'self.network' in source:
            expanded = expand_sequential(state_dict, source)
        else:
            expanded = generate_generic_expanded(state_dict, source, model)
    else:
        expanded = source

    # Build weight shapes comment
    weight_info = ""
    if state_dict:
        weight_info = "\n# Frozen weights (loaded from .pt at module init):\n"
        for k, v in state_dict.items():
            weight_info += f"#   {k}: {list(v.shape)} ({str(v.dtype).split('.')[-1]})\n"

    ref_source = source + weight_info + expanded

    # Write problem JSON
    problem = {
        "name": f"l2_{name}" if "level2" in str(filepath) else f"l3_{name}",
        "op_type": "fusion",
        "inputs": [{"name": "x", "shape": inp_shape, "dtype": "float32"}],
        "outputs": [{"name": "output", "shape": None, "dtype": "float32"}],
        "reference": ref_source,
    }
    json_path = os.path.join(out_dir, f"{name}.json")
    with open(json_path, "w") as f:
        json.dump(problem, f, indent=2, ensure_ascii=False)

    # Write initial solution
    init_path = os.path.join(out_dir, f"{name}_initial.py")
    with open(init_path, "w") as f:
        f.write(ref_source)

    return {"name": name, "problem": f"problems/{out_dir}/{name}.json",
            "initial_solution": f"problems/{out_dir}/{name}_initial.py"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb-level2", default=os.environ.get("KB_L2", "../../KernelBench/KernelBench/level2"))
    parser.add_argument("--out", default="problems/kb_level2/")
    parser.add_argument("--first", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    py_files = sorted(
        [f for f in os.listdir(args.kb_level2) if f.endswith('.py')],
        key=lambda f: int(re.match(r'(\d+)', f).group(1)) if re.match(r'(\d+)', f) else 9999
    )

    if args.first:
        py_files = py_files[:args.first]

    print(f"Converting {len(py_files)} problems...")
    batch = []
    for i, f in enumerate(py_files):
        try:
            entry = process_one(os.path.join(args.kb_level2, f), args.out)
            batch.append(entry)
            print(f"  [{i+1}/{len(py_files)}] {f}")
        except Exception as e:
            print(f"  [{i+1}/{len(py_files)}] ERROR {f}: {e}")
            import traceback; traceback.print_exc()

    csv_path = os.path.join(args.out, "_batch.csv")
    with open(csv_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=["name", "problem", "initial_solution"])
        w.writeheader()
        w.writerows(batch)

    print(f"\nDone: {len(batch)}/{len(py_files)} problems → {args.out}")


if __name__ == "__main__":
    main()
