"""验证 convert 逻辑：re-convert 全部 level2，每题对比 reference run() vs 原始 Model.forward()。"""
import sys, os, importlib.util, torch
sys.path.insert(0, 'scripts')
from convert_kb_level2 import process_one, load_model

KB = os.environ.get('KB_L2', '../../KernelBench/KernelBench/level2')
OUT = 'problems/kb_level2'
device = 'cuda'

files = sorted([f for f in os.listdir(KB) if f.endswith('.py')])
ok = mismatch = fail = 0
mismatch_list = []
for f in files:
    name = f.replace('.py', '')
    src_path = os.path.join(KB, f)
    # 1. re-convert
    try:
        process_one(src_path, OUT)
    except Exception as e:
        print(f'{name}: CONVERT FAIL {type(e).__name__}: {e}'); fail += 1; continue
    # 2. 对比 run() vs Model
    try:
        spec = importlib.util.spec_from_file_location('ref', f'{OUT}/{name}_initial.py')
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        src = open(src_path).read()
        model, _ = load_model(src)
        model = model.to(device).eval()
        torch.manual_seed(0)
        inputs = [t.to(device) for t in m.get_inputs()]
        ref_out = m.run(*inputs)
        model_out = model(*inputs)
        if not isinstance(ref_out, tuple): ref_out = (ref_out,)
        if not isinstance(model_out, tuple): model_out = (model_out,)
        allclose = len(ref_out) == len(model_out) and all(
            torch.allclose(r, mo, rtol=1e-3, atol=1e-2) for r, mo in zip(ref_out, model_out))
        if allclose:
            ok += 1
        else:
            mismatch += 1
            md = max((r-mo).abs().max().item() for r, mo in zip(ref_out, model_out))
            mismatch_list.append((name, md))
            print(f'{name}: MISMATCH max_diff={md:.4f}')
    except Exception as e:
        fail += 1
        print(f'{name}: VERIFY FAIL {type(e).__name__}: {str(e)[:80]}')

print(f'\n=== {len(files)} 题: OK={ok} mismatch={mismatch} fail={fail} ===')
if mismatch_list:
    print('mismatch 题:', [n for n, _ in mismatch_list])
