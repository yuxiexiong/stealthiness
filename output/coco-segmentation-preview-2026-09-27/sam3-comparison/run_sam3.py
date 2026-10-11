"""Five-image, fixed-text-prompt SAM 3 inference; preserves every raw mask."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((args.input / 'protocol.json').read_text())
    assert len(protocol['samples']) == 5
    torch.set_num_threads(4)
    start = time.perf_counter()
    model = build_sam3_image_model(checkpoint_path=str(args.checkpoint), load_from_HF=False)
    processor = Sam3Processor(model, confidence_threshold=protocol['confidence_threshold'])
    info = {'model': 'facebook/sam3', 'mode': 'fixed_text_only', 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
            'confidence_threshold': protocol['confidence_threshold'], 'mask_threshold': .5,
            'model_load_seconds': time.perf_counter() - start, 'samples': []}
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for sample in protocol['samples']:
            sample_start = time.perf_counter()
            image_path = args.input / sample['image']
            image = Image.open(image_path).convert('RGB')
            state = processor.set_image(image)
            masks, records, prompts = {}, [], []
            for prompt in sample['prompts']:
                processor.reset_all_prompts(state)
                prompt_start = time.perf_counter()
                result = processor.set_text_prompt(prompt=prompt['text'], state=state)
                torch.cuda.synchronize()
                predicted = result['masks'].detach().cpu().numpy().reshape(-1, image.height, image.width)
                scores = result['scores'].detach().float().cpu().numpy().reshape(-1)
                boxes = result['boxes'].detach().float().cpu().numpy().reshape(-1, 4)
                assert len(predicted) == len(scores) == len(boxes)
                for mask, score, box in zip(predicted, scores, boxes):
                    candidate_id = f"sam_{sample['idx']:03d}_{len(records):03d}"
                    assert mask.dtype == np.bool_ and mask.shape == (image.height, image.width)
                    masks[candidate_id] = mask
                    records.append({'id': candidate_id, **prompt, 'score': float(score),
                                    'box_xyxy': box.tolist(), 'area': int(mask.sum()), 'source': 'sam3_text'})
                prompts.append({**prompt, 'instances': len(predicted), 'seconds': time.perf_counter() - prompt_start})
                print(json.dumps({'idx': sample['idx'], 'prompt': prompt['text'], 'instances': len(predicted)}), flush=True)
            stem = f"raw_{sample['idx']:03d}"
            np.savez_compressed(args.output / f'{stem}.npz', **masks)
            (args.output / f'{stem}.json').write_text(json.dumps(records, indent=2))
            info['samples'].append({'idx': sample['idx'], 'image_id': sample['image_id'],
                                    'image_sha256': hashlib.sha256(image_path.read_bytes()).hexdigest(),
                                    'prompts': prompts, 'seconds': time.perf_counter() - sample_start,
                                    'candidates': len(records)})
            (args.output / 'run_info.json').write_text(json.dumps(info, indent=2))
    info['status'] = 'complete'
    info['peak_cuda_gb'] = torch.cuda.max_memory_allocated() / 1e9
    info['total_seconds'] = time.perf_counter() - start
    (args.output / 'run_info.json').write_text(json.dumps(info, indent=2))
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
