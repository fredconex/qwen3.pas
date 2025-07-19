## Qwen3 Inference for FreePascal

This do inference using CPU only, I've implemented some speedup improves with AVX2 but I bet there's still lot of room for improvements.

Performance compared to LM Studio (CPU runtime)  
```
## 0.6B
FreePascal - 20 tk/s | LM Studio - 43 tk/s  

## 1.7B
FreePascal - 8 tk/s | LM studio - 20 tk/s

## 4B
FreePascal - 3 tk/s | LM Studio - 9 tk/s

## 8B
FreePascal - 2 tk/s | LM Studio - 5 tk/s
```

Be aware that it does not take advantage of batching or parallel processing, so long prompts can really take a while to process !

### Clone repo and install python requirements
```
https://github.com/fredconex/qwen3.pas.git
cd qwen3.pas
pip install -r requirements.txt
```

### Convert the HF model to bin using python
```
python export.py Qwen3-0.6B.bin Qwen/Qwen3-0.6B
```
Available models:  
```
Qwen3-0.6B  
Qwen3-1.7B  
Qwen3-4B  
Qwen3-8B (This one requires lot of ram to convert the model!)
```

### Run it
```
.\qwen3.exe .\Qwen3-0.6B.bin
```
```
Usage:   qwen3 <checkpoint> [options]
Example: qwen3 Qwen3-4B.bin -r 1
Options:
  -t <float>  temperature in [0,inf], default 1.0
  -p <float>  p value in top-p (nucleus) sampling in [0,1], default 0.9
  -s <int>    random seed, default time(NULL)
  -c <int>    context window size, 0 (default) = max_seq_len
  -m <string> mode: generate|chat, default: chat
  -i <string> input prompt
  -y <string> system prompt in chat mode, default is none
  -r <int>    reasoning mode, 0 (default) = no thinking, 1 = thinking
```

### Credits to Adrian Cable:  
https://github.com/adriancable/qwen3.c
