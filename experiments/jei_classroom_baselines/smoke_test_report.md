# Classroom baseline smoke-test report

> These are paper-faithful reimplementations based on the published method descriptions and are not claimed to be the authors' official source code.

## Environment

- Python: 3.10.16
- PyTorch: 2.3.1+cu118
- CUDA: 11.8
- GPU: NVIDIA GeForce RTX 3090
- Ultralytics: 8.4.38
- Git commit: f8b724ae93e05d4deae1095d5bb1a7d72d254f70
- Formal training executed: NO
- Paper modified: NO

## PLA-YOLO11n

- Instantiation: PASS (CPU: True)
- Forward: PASS
- Input: [1, 3, 640, 640]
- Params: 3,474,412 (3.474412 M)
- Trainable Params: 3,474,396 (3.474396 M)
- GFLOPs @640: 10.299341
- Detect scales: 4
- Detect feature sizes: [[160, 160], [80, 80], [40, 40], [20, 20]]
- Detect channels: [32, 64, 128, 256]
- Detect strides: [4, 8, 16, 32]
- Module counts: {'C3k2PConv': 10, 'AIFI': 1, 'LSKA': 4, 'C2fWADCA': 0, 'TwoDPEMHA': 0, 'DySample': 0}
- Constructor validation: {'C3k2PConv': {'actual': {'c1': 32, 'c2': 64, 'n': 1, 'c3k': False, 'e': 0.25, 'n_div': 4}, 'correct': True}, 'LSKA': {'actual': {'c': 32, 'k': 7, 'dilation': 2}, 'correct': True}}
- TwoDPEMHA input shape: N/A
- Output type: tuple
- Output shapes: [{'path': 'output[0]', 'shape': [1, 7, 34000], 'dtype': 'torch.float32'}, {'path': 'output[1].boxes', 'shape': [1, 64, 34000], 'dtype': 'torch.float32'}, {'path': 'output[1].scores', 'shape': [1, 3, 34000], 'dtype': 'torch.float32'}, {'path': 'output[1].feats[0]', 'shape': [1, 32, 160, 160], 'dtype': 'torch.float32'}, {'path': 'output[1].feats[1]', 'shape': [1, 64, 80, 80], 'dtype': 'torch.float32'}, {'path': 'output[1].feats[2]', 'shape': [1, 128, 40, 40], 'dtype': 'torch.float32'}, {'path': 'output[1].feats[3]', 'shape': [1, 256, 20, 20], 'dtype': 'torch.float32'}]
- NaN: NO
- Inf: NO
- Peak GPU memory: 210.17529296875 MiB
- Smoke test: PASS
- Paper-reported Params: approximately 3.08 M
- Reimplementation difference: +0.394412 M (+12.81%)
- Difference may reflect LSKA/PConv details, four-scale neck reconstruction, and Ultralytics version.

## WAD-YOLOv8n

- Instantiation: PASS (CPU: True)
- Forward: PASS
- Input: [1, 3, 640, 640]
- Params: 3,909,769 (3.909769 M)
- Trainable Params: 3,909,753 (3.909753 M)
- GFLOPs @640: 9.623923
- Detect scales: 3
- Detect feature sizes: [[80, 80], [40, 40], [20, 20]]
- Detect channels: [64, 128, 256]
- Detect strides: [8, 16, 32]
- Module counts: {'C3k2PConv': 0, 'AIFI': 0, 'LSKA': 0, 'C2fWADCA': 4, 'TwoDPEMHA': 1, 'DySample': 2}
- Constructor validation: {'C2fWADCA': {'actual': {'c1': 32, 'c2': 32, 'n': 1, 'shortcut': True, 'g': 1, 'e': 0.5, 'reduction': 16}, 'correct': True}, 'TwoDPEMHA': {'actual': {'c': 256, 'num_heads': 8, 'dropout': 0.0}, 'correct': True}}
- TwoDPEMHA input shape: [[1, 256, 20, 20]]
- Output type: tuple
- Output shapes: [{'path': 'output[0]', 'shape': [1, 7, 8400], 'dtype': 'torch.float32'}, {'path': 'output[1].boxes', 'shape': [1, 64, 8400], 'dtype': 'torch.float32'}, {'path': 'output[1].scores', 'shape': [1, 3, 8400], 'dtype': 'torch.float32'}, {'path': 'output[1].feats[0]', 'shape': [1, 64, 80, 80], 'dtype': 'torch.float32'}, {'path': 'output[1].feats[1]', 'shape': [1, 128, 40, 40], 'dtype': 'torch.float32'}, {'path': 'output[1].feats[2]', 'shape': [1, 256, 20, 20], 'dtype': 'torch.float32'}]
- NaN: NO
- Inf: NO
- Peak GPU memory: 194.21533203125 MiB
- Smoke test: PASS

## Final status

- PLA smoke test passed: YES
- WAD smoke test passed: YES
- Ready for training review: YES
- Formal training executed: NO
- Paper modified: NO
