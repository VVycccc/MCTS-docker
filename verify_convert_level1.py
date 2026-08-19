"""验证 level1: _initial.py 的 run() vs 原始 KernelBench Model.forward()。"""
import os, importlib.util, torch
KB = os.environ.get('KB_L1', '../../KernelBench/KernelBench/level1')
OUT = 'problems/kb_level1'
files = sorted([f for f in os.listdir(KB) if f.endswith('.py')])
ok = mismatch = fail = 0
for f in files:
    name = f.replace('.py', '')
    # 找对应 _initial.py（大小写可能不同）
    init_candidates = [c for c in os.listdir(OUT) if c.endswith('_initial.py') and c.replace('_initial.py','').lower() == name.lower()]
    if not init_candidates:
        print(f'{name}: no _initial.py'); fail += 1; continue
    init_file = init_candidates[0]
    try:
        spec = importlib.util.spec_from_file_location('ref', f'{OUT}/{init_file}')
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    except Exception as e:
        print(f'{name}: initial FAIL {type(e).__name__}: {str(e)[:60]}'); fail += 1; continue
    try:
        spec2 = importlib.util.spec_from_file_location('orig', f'{KB}/{f}')
        mo = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(mo)
    except Exception as e:
        print(f'{name}: orig FAIL {type(e).__name__}: {str(e)[:60]}'); fail += 1; continue
    try:
        torch.manual_seed(0)
        inputs = mo.get_inputs() if hasattr(mo,'get_inputs') else m.get_inputs()
        inputs = [t.to('cuda') for t in inputs]
        ref_out = m.run(*inputs)
        ModelCls = getattr(mo, 'Model', None)
        if ModelCls is None:
            ok += 1; continue
        model = ModelCls(*mo.get_init_inputs())
        init_stem = init_file.replace('_initial.py', '')
        wp = f'{OUT}/{init_stem}_weights.pt'
        if os.path.exists(wp):
            model.load_state_dict(torch.load(wp, map_location='cpu', weights_only=True))
        model = model.to('cuda').eval()
        model_out = model(*inputs)
        if not isinstance(ref_out, tuple): ref_out = (ref_out,)
        if not isinstance(model_out, tuple): model_out = (model_out,)
        if len(ref_out)==len(model_out) and all(torch.allclose(r,o,rtol=1e-3,atol=1e-2) for r,o in zip(ref_out,model_out)):
            ok += 1
        else:
            mismatch += 1
            md = max((r-o).abs().max().item() for r,o in zip(ref_out,model_out))
            print(f'{name}: MISMATCH max_diff={md:.4f}')
    except Exception as e:
        fail += 1
        print(f'{name}: verify FAIL {type(e).__name__}: {str(e)[:60]}')
print(f'\n=== {len(files)} 题: OK={ok} mismatch={mismatch} fail={fail} ===')
