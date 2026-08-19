"""重新生成 level1 _initial.py：从原始 KernelBench .py 生成通用 run()=Model(x)。
有权重的题（conv 等）生成 .pt + run() 加载；无权重的纯计算题不加载。"""
import os, json, tempfile, importlib.util
from pathlib import Path
import torch

KB = Path(os.environ.get('KB_L1', '../../KernelBench/KernelBench/level1'))
OUT = Path('problems/kb_level1')

kb_files = {}
for f in KB.glob('*.py'):
    kb_files[f.stem.lower()] = f

fixed = skip = 0
for init_file in sorted(OUT.glob('*_initial.py')):
    name = init_file.name.replace('_initial.py', '')
    kb_path = kb_files.get(name.lower())
    if kb_path is None:
        candidates = [k for k in kb_files if k.startswith(name.lower()[:20])]
        if candidates:
            kb_path = kb_files[candidates[0]]
    if kb_path is None:
        print(f'{name}: no original .py, skip'); skip += 1; continue

    source = kb_path.read_text()
    json_path = OUT / f'{name}.json'
    problem = json.load(open(json_path))
    inputs = problem.get('inputs', [])
    input_names = [i['name'] for i in inputs] if inputs else ['x']
    run_sig = ', '.join(input_names)
    first = input_names[0]

    # 检测 Model 是否有权重（load Model + state_dict）
    has_weights = False
    weights_path = OUT / f'{name}_weights.pt'
    try:
        tmpdir = tempfile.mkdtemp()
        tmpfile = Path(tmpdir) / '_m.py'
        tmpfile.write_text(source)
        spec = importlib.util.spec_from_file_location('_m', tmpfile)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        torch.manual_seed(42)
        model = mod.Model(*mod.get_init_inputs())
        sd = model.state_dict()
        if sd:
            has_weights = True
            torch.save(sd, weights_path)
    except Exception as e:
        print(f'{name}: model load fail ({e}), no .pt')

    # 通用 run()
    load_line = ""
    if has_weights:
        load_line = f"        _model_cache.load_state_dict(_torch.load(_weights_path, map_location='cpu', weights_only=True))\n"
    universal = f'''

# --- EXPANDED REFERENCE ---
# run() 调用原始 Model.forward()（PyTorch 算，正确）。
# 写 Triton kernel 时：不要照抄 nn.Module，用 torch 实现（有权重则从 _weights_path 加载并缓存）。
import torch as _torch
_weights_path = "{weights_path}"
_model_cache = None
_model_device = None

def run({run_sig}):
    global _model_cache, _model_device
    if _model_cache is None or _model_device != str({first}.device):
        _model_cache = Model(*get_init_inputs())
{load_line}        _model_cache = _model_cache.to({first}.device).eval()
        _model_device = str({first}.device)
    return _model_cache({run_sig})
'''
    new_initial = source + universal
    init_file.write_text(new_initial)
    problem['reference'] = new_initial
    json.dump(problem, open(json_path, 'w'), indent=2, ensure_ascii=False)
    fixed += 1

print(f'\n=== fixed={fixed} skip={skip} ===')
