"""Minimal OpenAI-compatible /v1/chat/completions server for the frozen T2I reader.

Serves one pinned Qwen2.5-VL snapshot with greedy decoding (temperature 0 requests are
honored exactly; any other temperature is rejected so the reader stays deterministic).
Single-threaded on purpose: calls serialize on the GPU and each sealed read records
usage from real token counts. Stdlib HTTP only; no web framework.

Usage:
  python reader_server.py --snapshot /path/to/snapshots/<revision> --device cuda:0 --port 8009
"""

import argparse
import base64
import io
import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer


def build(snapshot, device, dtype_name):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(snapshot)
    print("cuda available:", torch.cuda.is_available(), "devices:", torch.cuda.device_count(), flush=True)
    # This host's torch-2.3 cuBLASLt SIGFPEs on the bf16 GEMM path (kernel trap in
    # libcublasLt); fp16/fp32 are selectable and the served dtype is printed and logged.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        snapshot, torch_dtype=getattr(torch, dtype_name), device_map={"": device}).eval()
    parameter_device = next(model.parameters()).device
    print("model device:", parameter_device,
          "| allocated GiB:", round(torch.cuda.memory_allocated() / 2**30, 1), flush=True)
    if parameter_device.type != "cuda":
        raise RuntimeError("reader model must live on the GPU; refusing to serve from CPU")

    def complete(body):
        if body.get("temperature", 0) not in (0, 0.0):
            raise ValueError("this reader is frozen to greedy decoding; send temperature 0")
        message = body["messages"][0]
        images, content = [], []
        for item in message["content"]:
            if item["type"] == "text":
                content.append({"type": "text", "text": item["text"]})
            elif item["type"] == "image_url":
                url = item["image_url"]["url"]
                match = re.match(r"data:image/[^;]+;base64,(.*)", url, re.S)
                if not match:
                    raise ValueError("only data: image URLs are accepted")
                images.append(Image.open(io.BytesIO(base64.b64decode(match.group(1)))).convert("RGB"))
                content.append({"type": "image"})
            else:
                raise ValueError("unsupported content item type")
        text = processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=images or None,
                           return_tensors="pt", padding=True).to(device)
        prompt_tokens = int(inputs.input_ids.shape[1])
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=False,
                                    max_new_tokens=int(body.get("max_tokens", 2400)))
        generated = output[0, prompt_tokens:]
        reply = processor.tokenizer.decode(generated, skip_special_tokens=True)
        return {"object": "chat.completion", "model": body.get("model"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": reply},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": prompt_tokens,
                          "completion_tokens": int(generated.shape[0]),
                          "total_tokens": prompt_tokens + int(generated.shape[0])}}

    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=8009)
    parser.add_argument("--dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    args = parser.parse_args()
    print("serving dtype:", args.dtype, flush=True)
    complete = build(args.snapshot, args.device, args.dtype)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
                self.send_error(404)
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                result = complete(body)
                payload = json.dumps(result).encode()
                self.send_response(200)
            except Exception as error:  # surfaced to the client and the log
                payload = json.dumps({"error": {"message": repr(error)}}).encode()
                self.send_response(500)
                print("ERROR:", repr(error), flush=True)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt, *values):
            print(self.address_string(), fmt % values, flush=True)

    print(f"reader ready on port {args.port}", flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    import torch  # noqa: F401  (fail fast if the environment is wrong)
    main()
