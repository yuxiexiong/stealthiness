"""Minimal SIGFPE reproduction matrix for the host cuBLASLt divide-error.

Each candidate op runs in its own subprocess so a native trap doesn't stop the
sweep. Shapes mirror LLaVA-1.5-7B LoRA training GEMMs.
"""
import subprocess
import sys

CASES = {
    "fwd_linear_bf16":      "x=torch.randn(8,1024,4096,device='cuda',dtype=torch.bfloat16);"
                            "w=torch.randn(4096,4096,device='cuda',dtype=torch.bfloat16);"
                            "y=x@w.t();torch.cuda.synchronize()",
    "bwd_linear_bf16":      "x=torch.randn(8,1024,4096,device='cuda',dtype=torch.bfloat16,requires_grad=True);"
                            "w=torch.randn(4096,4096,device='cuda',dtype=torch.bfloat16,requires_grad=True);"
                            "(x@w.t()).sum().backward();torch.cuda.synchronize()",
    "bwd_lmhead_bf16":      "x=torch.randn(8,64,4096,device='cuda',dtype=torch.bfloat16,requires_grad=True);"
                            "w=torch.randn(32064,4096,device='cuda',dtype=torch.bfloat16,requires_grad=True);"
                            "(x@w.t()).float().sum().backward();torch.cuda.synchronize()",
    "bmm_attn_bf16":        "a=torch.randn(32,1024,128,device='cuda',dtype=torch.bfloat16);"
                            "b=torch.randn(32,128,1024,device='cuda',dtype=torch.bfloat16);"
                            "c=torch.bmm(a,b);torch.cuda.synchronize()",
    "bwd_linear_fp16":      "x=torch.randn(8,1024,4096,device='cuda',dtype=torch.float16,requires_grad=True);"
                            "w=torch.randn(4096,4096,device='cuda',dtype=torch.float16,requires_grad=True);"
                            "(x@w.t()).sum().backward();torch.cuda.synchronize()",
    "bwd_linear_fp32":      "x=torch.randn(4,512,4096,device='cuda',requires_grad=True);"
                            "w=torch.randn(4096,4096,device='cuda',requires_grad=True);"
                            "(x@w.t()).sum().backward();torch.cuda.synchronize()",
    "bwd_bias_addmm_bf16":  "x=torch.randn(1024,4096,device='cuda',dtype=torch.bfloat16,requires_grad=True);"
                            "m=torch.nn.Linear(4096,4096,device='cuda',dtype=torch.bfloat16);"
                            "m(x).sum().backward();torch.cuda.synchronize()",
}

python = sys.argv[1] if len(sys.argv) > 1 else sys.executable
for name, body in CASES.items():
    code = "import torch;" + body + ";print('OK')"
    r = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=180)
    verdict = "OK" if r.returncode == 0 else f"CRASH rc={r.returncode}"
    detail = (r.stderr.strip().splitlines() or [""])[-1][:90] if r.returncode else ""
    print(f"{name:22s} {verdict}  {detail}", flush=True)
