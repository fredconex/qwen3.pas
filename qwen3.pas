{ Inference for Qwen-3 Transformer model in FreePascal, int8 quantized forward pass }
program Qwen3;

{$mode objfpc}{$H+}
{$modeswitch advancedrecords}

uses
  SysUtils,
  Classes,
  Math,
  DateUtils,
  CTypes,
  Windows;

const
  PROMPT_BUFFER_SIZE = 32768;

type
  { Quantized tensor structure }
  PQuantizedTensor = ^TInt8QuantizedTensor;

  TInt8QuantizedTensor = record
    q: PInt8;    // quantized values (int8)
    s: PSingle;  // scaling factors

    procedure Dequantize(x: PSingle; n: longint);
    procedure Quantize(x: PSingle; n: longint);
  end;

  { Array of quantized tensors with utility methods }
  TInt8QuantizedTensorArray = record
    Data: array of TInt8QuantizedTensor;
    
    procedure Initialize(Count: integer; var DataPtr: Pointer; ElementsPerTensor: integer);
    function GetTensor(Index: integer): TInt8QuantizedTensor;
    function GetTensorPtr(Index: integer): PQuantizedTensor;
    function Count: integer;
    function IsValidIndex(Index: integer): boolean;
    procedure Validate; // Validates that all tensors have valid pointers
  end;

  { Configuration structure }
  TConfig = record
    magic_number: longint;
    version: longint;
    dim: longint;
    hidden_dim: longint;
    n_layers: longint;
    n_heads: longint;
    n_kv_heads: longint;
    vocab_size: longint;
    seq_len: longint;
    head_dim: longint;
    shared_classifier: longint;
    group_size: longint;
  end;

  { Transformer weights }
  TTransformerWeights = record
    q_tokens: PQuantizedTensor;
    token_embedding_table: PSingle;
    rms_att_weight: PSingle;
    rms_ffn_weight: PSingle;
    wq: TInt8QuantizedTensorArray;
    wk: TInt8QuantizedTensorArray;
    wv: TInt8QuantizedTensorArray;
    wo: TInt8QuantizedTensorArray;
    q_norm_weights: PSingle;
    k_norm_weights: PSingle;
    w1: TInt8QuantizedTensorArray;
    w2: TInt8QuantizedTensorArray;
    w3: TInt8QuantizedTensorArray;
    rms_final_weight: PSingle;
    wcls: PQuantizedTensor;
  end;

  { Run state }
  TRunState = record
    x: PSingle;
    xb: PSingle;
    hb: PSingle;
    hb2: PSingle;
    xq: TInt8QuantizedTensor;
    hq: TInt8QuantizedTensor;
    q: PSingle;
    k: PSingle;
    v: PSingle;
    att: PSingle;
    logits: PSingle;
    key_cache: PSingle;
    value_cache: PSingle;
  end;

  { Transformer structure }
  TTransformer = record
    config: TConfig;
    weights: TTransformerWeights;
    state: TRunState;
    Data: Pointer;
    file_size: int64;
  end;

  { Tokenizer }
  PPChar = ^pchar;

  TTokenizer = record
    vocab: PPChar;
    merge_scores: PSingle;
    vocab_size: longint;
    max_token_length: longword;
    bos_token_id: longword;
    eos_token_id: longword;
    prompt_template: array[0..1023] of char;
    system_prompt_template: array[0..1023] of char;
  end;

  { Probability index for sampling }
  TProbIndex = record
    prob: single;
    index: longint;
  end;
  PProbIndex = ^TProbIndex;

  { Sampler }
  TSampler = record
    vocab_size: longint;
    probindex: PProbIndex;
    temperature: single;
    topp: single;
    rng_state: QWord;
  end;

var
  GS: longint = 0; // Global group size for quantization

  { TQuantizedTensor method implementations }
  procedure TInt8QuantizedTensor.Dequantize(x: PSingle; n: longint);
  var
    i: longint;
  begin
    for i := 0 to n - 1 do
      (x + i)^ := (self.q + i)^ * (self.s + (i div GS))^;
  end;

  procedure TInt8QuantizedTensor.Quantize(x: PSingle; n: longint);
  var
    group, i: longint;
    wmax, val, scale, quant_value: single;
    quantized: shortint;
  begin
    for group := 0 to (n div GS) - 1 do
    begin
      // Find max absolute value in current group
      wmax := 0;
      for i := 0 to GS - 1 do
      begin
        val := Abs((x + group * GS + i)^);
        if val > wmax then
          wmax := val;
      end;

      // Calculate scaling factor
      scale := wmax / 127.0;
      (self.s + group)^ := scale;

      // Quantize values
      for i := 0 to GS - 1 do
      begin
        if scale > 0 then
          quant_value := (x + group * GS + i)^ / scale
        else
          quant_value := 0;
        quantized := Round(quant_value);
        (self.q + group * GS + i)^ := quantized;
      end;
    end;
  end;

  { TQuantizedTensorArray method implementations }
  procedure TInt8QuantizedTensorArray.Initialize(Count: integer; var DataPtr: Pointer; ElementsPerTensor: integer);
  var
    CurrentPtr: pbyte;
    i: integer;
    QuantizedDataSize: integer;
    ScaleDataSize: integer;
  begin
    SetLength(Data, Count);
    CurrentPtr := pbyte(DataPtr);
    QuantizedDataSize := ElementsPerTensor * SizeOf(shortint);
    ScaleDataSize := (ElementsPerTensor div GS) * SizeOf(single);

    // Initialize each tensor in the array
    for i := 0 to Count - 1 do
    begin
      // Set quantized data pointer for tensor i
      Data[i].q := PShortInt(CurrentPtr);
      Inc(CurrentPtr, QuantizedDataSize);

      // Set scale data pointer for tensor i
      Data[i].s := PSingle(CurrentPtr);
      Inc(CurrentPtr, ScaleDataSize);
    end;

    // Update the input pointer to point past all processed data
    DataPtr := CurrentPtr;
  end;

  function TInt8QuantizedTensorArray.GetTensor(Index: integer): TInt8QuantizedTensor;
  begin
    if IsValidIndex(Index) then
      Result := Data[Index]
    else
    begin
      WriteLn(StdErr, 'Error: Tensor index out of bounds: ', Index);
      Halt(1);
    end;
  end;

  function TInt8QuantizedTensorArray.Count: integer;
  begin
    Result := Length(Data);
  end;

  function TInt8QuantizedTensorArray.GetTensorPtr(Index: integer): PQuantizedTensor;
  begin
    if IsValidIndex(Index) then
      Result := @Data[Index]
    else
    begin
      WriteLn(StdErr, 'Error: Tensor index out of bounds: ', Index);
      Halt(1);
    end;
  end;

  function TInt8QuantizedTensorArray.IsValidIndex(Index: integer): boolean;
  begin
    Result := (Index >= 0) and (Index < Length(Data));
  end;

  procedure TInt8QuantizedTensorArray.Validate;
  var
    i: integer;
  begin
    for i := 0 to Length(Data) - 1 do
    begin
      if (Data[i].q = nil) or (Data[i].s = nil) then
      begin
        WriteLn(StdErr, 'Error: Invalid tensor at index ', i, ' - null pointers detected');
        Halt(1);
      end;
    end;
  end;

  { Configure console for UTF-8 output }
  procedure ConfigureConsoleForUTF8;
  begin
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
  end;

  { Memory allocation helpers }
  function SafeGetMem(Size: PtrUInt): Pointer;
  begin
    GetMem(Result, Size);
    if Result = nil then
    begin
      WriteLn(StdErr, 'Memory allocation failed!');
      Halt(1);
    end;
    FillChar(Result^, Size, 0);
  end;





  { Memory map weights with improved structure and readability }
  procedure MapWeightsToMemory(var Weights: TTransformerWeights; const Config: TConfig; var DataPtr: Pointer);
  var
    FloatPtr: PSingle;
    BytePtr: pbyte;

    procedure AllocateFloatWeights(var WeightPtr: PSingle; ElementCount: integer);
    begin
      WeightPtr := FloatPtr;
      Inc(FloatPtr, ElementCount);
    end;

    procedure AllocateAndDequantizeTokens;
    var
      TokenTableSize: integer;
    begin
      TokenTableSize := Config.vocab_size * Config.dim;

      // Initialize quantized token embeddings (single tensor)
      GetMem(Weights.q_tokens, SizeOf(TInt8QuantizedTensor));
      Weights.q_tokens^.q := PShortInt(BytePtr);
      Inc(BytePtr, TokenTableSize * SizeOf(shortint));
      Weights.q_tokens^.s := PSingle(BytePtr);
      Inc(BytePtr, (TokenTableSize div GS) * SizeOf(single));

      // Allocate and dequantize token embedding table
      GetMem(Weights.token_embedding_table, TokenTableSize * SizeOf(single));
      Weights.q_tokens^.Dequantize(Weights.token_embedding_table, TokenTableSize);
    end;

  begin
    FloatPtr := PSingle(DataPtr);

    // Map float weights in order
    AllocateFloatWeights(Weights.rms_att_weight, Config.n_layers * Config.dim);
    AllocateFloatWeights(Weights.rms_ffn_weight, Config.n_layers * Config.dim);
    AllocateFloatWeights(Weights.rms_final_weight, Config.dim);
    AllocateFloatWeights(Weights.q_norm_weights, Config.n_layers * Config.head_dim);
    AllocateFloatWeights(Weights.k_norm_weights, Config.n_layers * Config.head_dim);

    // Switch to byte pointer for quantized data
    BytePtr := pbyte(FloatPtr);

    // Process token embeddings
    AllocateAndDequantizeTokens;

    // Map quantized weight matrices
    with Config do
    begin
      // Initialize arrays using the new type methods
      Weights.wq.Initialize(n_layers, Pointer(BytePtr), dim * (n_heads * head_dim));
      Weights.wk.Initialize(n_layers, Pointer(BytePtr), dim * (n_kv_heads * head_dim));
      Weights.wv.Initialize(n_layers, Pointer(BytePtr), dim * (n_kv_heads * head_dim));
      Weights.wo.Initialize(n_layers, Pointer(BytePtr), (n_heads * head_dim) * dim);

      // Feed-forward network weights
      Weights.w1.Initialize(n_layers, Pointer(BytePtr), dim * hidden_dim);
      Weights.w2.Initialize(n_layers, Pointer(BytePtr), hidden_dim * dim);
      Weights.w3.Initialize(n_layers, Pointer(BytePtr), dim * hidden_dim);

      // Validate all tensor arrays
      Weights.wq.Validate;
      Weights.wk.Validate;
      Weights.wv.Validate;
      Weights.wo.Validate;
      Weights.w1.Validate;
      Weights.w2.Validate;
      Weights.w3.Validate;

      // Classifier weights (shared or separate)
      if shared_classifier = 1 then
        Weights.wcls := Weights.q_tokens
      else
      begin
        GetMem(Weights.wcls, SizeOf(TInt8QuantizedTensor));
        Weights.wcls^.q := PShortInt(BytePtr);
        Inc(BytePtr, dim * vocab_size * SizeOf(shortint));
        Weights.wcls^.s := PSingle(BytePtr);
        Inc(BytePtr, (dim * vocab_size div GS) * SizeOf(single));
      end;
    end;

    // Update the data pointer
    DataPtr := BytePtr;
  end;

  { Allocate run state }
  procedure MallocRunState(var s: TRunState; var p: TConfig);
  var
    all_heads_dim, kv_dim: longint;
  begin
    all_heads_dim := p.n_heads * p.head_dim;
    kv_dim := p.n_kv_heads * p.head_dim;

    s.x := SafeGetMem(p.dim * SizeOf(single));
    s.xb := SafeGetMem(all_heads_dim * SizeOf(single));
    s.hb := SafeGetMem(p.hidden_dim * SizeOf(single));
    s.hb2 := SafeGetMem(p.hidden_dim * SizeOf(single));

    s.xq.q := SafeGetMem(all_heads_dim * SizeOf(shortint));
    s.xq.s := SafeGetMem((all_heads_dim div GS) * SizeOf(single));
    s.hq.q := SafeGetMem(p.hidden_dim * SizeOf(shortint));
    s.hq.s := SafeGetMem((p.hidden_dim div GS) * SizeOf(single));

    s.q := SafeGetMem(all_heads_dim * SizeOf(single));
    s.att := SafeGetMem(p.n_heads * p.seq_len * SizeOf(single));
    s.logits := SafeGetMem(p.vocab_size * SizeOf(single));
    s.key_cache := SafeGetMem(p.n_layers * QWord(p.seq_len) * kv_dim * SizeOf(single));
    s.value_cache := SafeGetMem(p.n_layers * QWord(p.seq_len) * kv_dim * SizeOf(single));
  end;

  { Free run state }
  procedure FreeRunState(var s: TRunState);
  begin
    FreeMem(s.x);
    FreeMem(s.xb);
    FreeMem(s.hb);
    FreeMem(s.hb2);
    FreeMem(s.xq.q);
    FreeMem(s.xq.s);
    FreeMem(s.hq.q);
    FreeMem(s.hq.s);
    FreeMem(s.q);
    FreeMem(s.att);
    FreeMem(s.logits);
    FreeMem(s.key_cache);
    FreeMem(s.value_cache);
  end;

  { Read checkpoint }
  procedure ReadCheckpoint(checkpoint: string; var config: TConfig; var weights: TTransformerWeights; var Data: Pointer; var file_size: int64; ctx_length: longint);
  var
    fs: TFileStream;
    weights_ptr: Pointer;
  begin
    fs := TFileStream.Create(checkpoint, fmOpenRead);
    try
      WriteLn('checkpoint: ' + checkpoint);

      file_size := fs.Size;
      GetMem(Data, file_size);
      fs.ReadBuffer(Data^, file_size);

      // Read config from first 256 bytes
      Move(Data^, config, SizeOf(TConfig));
      if config.magic_number <> $616a6331 then
      begin
        WriteLn(StdErr, 'File ', checkpoint, ' is not a qwen3.c checkpoint');
        Halt(1);
      end;

      if config.version <> 1 then
      begin
        WriteLn(StdErr, 'Checkpoint ', checkpoint, ' is version ', config.version, ', need version 1');
        Halt(1);
      end;

      if (ctx_length <> 0) and (ctx_length <= config.seq_len) then
        config.seq_len := ctx_length;

      GS := config.group_size;
      weights_ptr := PChar(Data) + 256;
      MapWeightsToMemory(weights, config, weights_ptr);
    finally
      fs.Free;
    end;
  end;

  { RMS Normalization }
  procedure RMSNorm(const o, x, weight: PSingle; size: longint);
  var
    ss: single;
    j: longint;
  begin
    ss := 0;
    for j := 0 to size - 1 do
      ss += (x + j)^ ** 2;

    ss := 1.0 / Sqrt((ss / size) + 1e-6);

    for j := 0 to size - 1 do
      (o + j)^ := (weight + j)^ * (ss * (x + j)^);
  end;

  { Softmax }
  procedure Softmax(const x: PSingle; size: longint);
  var
    max_val, sum: single;
    i: longint;
  begin
    max_val := 0;
    for i := 0 to size - 1 do
      if (x + i)^ > max_val then
        max_val := (x + i)^;

    sum := 0;
    for i := 0 to size - 1 do
    begin
      (x + i)^ := Exp((x + i)^ - max_val);
      sum := sum + (x + i)^;
    end;

    for i := 0 to size - 1 do
      (x + i)^ := (x + i)^ / sum;
  end;

  { Matrix multiplication }
  procedure MatMul(xout: PSingle; const x, w: TInt8QuantizedTensor; n, d: longint);
  var
    i, j, k, groups: longint;
    val: single;
    ival: longint;
    x_base, w_base: ^shortint;
    x_scales, w_scales: PSingle;
    group_scale: single;
  begin
    groups := n div GS;
    for i := 0 to d - 1 do
    begin
      val := 0;
      w_base := w.q + (i * n);
      w_scales := w.s + (i * groups);
      x_base := x.q;
      x_scales := x.s;
      for j := 0 to groups - 1 do
      begin
        ival := 0;
        for k := 0 to (GS div 8) - 1 do
        begin
          ival += x_base[0] * w_base[0];
          ival += x_base[1] * w_base[1];
          ival += x_base[2] * w_base[2];
          ival += x_base[3] * w_base[3];
          ival += x_base[4] * w_base[4];
          ival += x_base[5] * w_base[5];
          ival += x_base[6] * w_base[6];
          ival += x_base[7] * w_base[7];
          Inc(x_base, 8);
          Inc(w_base, 8);
        end;
        group_scale := x_scales^ * w_scales^;
        val += ival * group_scale;
        Inc(x_scales);
        Inc(w_scales);
      end;
      xout^ := val;
      Inc(xout);
    end;
  end;


  { Forward pass }
  function Forward(var transformer: TTransformer; token: longint; pos: longint): PSingle;
  var
    p: ^TConfig;
    w: ^TTransformerWeights;
    s: ^TRunState;
    kv_dim, kv_mul, all_heads_dim: longint;
    l, h, i, j, t: longint;
    loff: QWord;
    q_ptr, k_ptr, v_ptr, xb_ptr: PSingle;
    freq, cos_freq, sin_freq, x_val, y_val: single;
    score: single;
    sigmoid_val: single;
  begin
    p := @transformer.config;
    w := @transformer.weights;
    s := @transformer.state;
    kv_dim := p^.n_kv_heads * p^.head_dim;
    kv_mul := p^.n_heads div p^.n_kv_heads;
    all_heads_dim := p^.n_heads * p^.head_dim;

    // Copy token embedding
    Move((w^.token_embedding_table + token * p^.dim)^, s^.x^, p^.dim * SizeOf(single));

    // Forward through all layers
    for l := 0 to p^.n_layers - 1 do
    begin
      loff := l * QWord(p^.seq_len) * kv_dim;

      s^.k := s^.key_cache + loff + pos * kv_dim;
      s^.v := s^.value_cache + loff + pos * kv_dim;

      // Attention RMS norm
      RMSNorm(s^.xb, s^.x, w^.rms_att_weight + l * p^.dim, p^.dim);

      // QKV matmuls
      s^.xq.Quantize(s^.xb, p^.dim);
      MatMul(s^.q, s^.xq, w^.wq.GetTensor(l), p^.dim, all_heads_dim);
      MatMul(s^.k, s^.xq, w^.wk.GetTensor(l), p^.dim, kv_dim);
      MatMul(s^.v, s^.xq, w^.wv.GetTensor(l), p^.dim, kv_dim);

      // Q-RMSNorm + rotate each query head
      for h := 0 to p^.n_heads - 1 do
      begin
        q_ptr := s^.q + h * p^.head_dim;

        RMSNorm(q_ptr, q_ptr, w^.q_norm_weights + l * p^.head_dim, p^.head_dim);
        for j := 0 to (p^.head_dim div 2) - 1 do
        begin
          freq := Power(1e6, -j / (p^.head_dim / 2));
          cos_freq := Cos(pos * freq);
          sin_freq := Sin(pos * freq);

          x_val := (q_ptr + j)^;
          y_val := (q_ptr + j + p^.head_dim div 2)^;

          (q_ptr + j)^ := x_val * cos_freq - y_val * sin_freq;
          (q_ptr + j + p^.head_dim div 2)^ := x_val * sin_freq + y_val * cos_freq;
        end;
      end;

      // K-RMSNorm + rotate each key head
      for h := 0 to p^.n_kv_heads - 1 do
      begin
        k_ptr := s^.k + h * p^.head_dim;

        RMSNorm(k_ptr, k_ptr, w^.k_norm_weights + l * p^.head_dim, p^.head_dim);
        for j := 0 to (p^.head_dim div 2) - 1 do
        begin
          freq := Power(1e6, -j / (p^.head_dim / 2));
          cos_freq := Cos(pos * freq);
          sin_freq := Sin(pos * freq);

          x_val := (k_ptr + j)^;
          y_val := (k_ptr + j + p^.head_dim div 2)^;

          (k_ptr + j)^ := x_val * cos_freq - y_val * sin_freq;
          (k_ptr + j + p^.head_dim div 2)^ := x_val * sin_freq + y_val * cos_freq;
        end;
      end;

      // Multihead attention
      for h := 0 to p^.n_heads - 1 do
      begin
        q_ptr := s^.q + h * p^.head_dim;

        for t := 0 to pos do
        begin
          k_ptr := s^.key_cache + loff + t * kv_dim + (h div kv_mul) * p^.head_dim;

          score := 0;
          for i := 0 to p^.head_dim - 1 do
            score := score + (q_ptr + i)^ * (k_ptr + i)^;

          (s^.att + h * p^.seq_len + t)^ := score / Sqrt(p^.head_dim);
        end;

        Softmax(s^.att + h * p^.seq_len, pos + 1);

        xb_ptr := s^.xb + h * p^.head_dim;
        FillChar(xb_ptr^, p^.head_dim * SizeOf(single), 0);

        for t := 0 to pos do
        begin
          v_ptr := s^.value_cache + loff + t * kv_dim + (h div kv_mul) * p^.head_dim;
          for i := 0 to p^.head_dim - 1 do
            (xb_ptr + i)^ := (xb_ptr + i)^ + (s^.att + h * p^.seq_len + t)^ * (v_ptr + i)^;
        end;
      end;

      // Final attention matmul
      s^.xq.Quantize(s^.xb, all_heads_dim);
      MatMul(s^.xb, s^.xq, w^.wo.GetTensor(l), all_heads_dim, p^.dim);

      // Residual connection
      for i := 0 to p^.dim - 1 do
        (s^.x + i)^ := (s^.x + i)^ + (s^.xb + i)^;

      // FFN RMS norm
      RMSNorm(s^.xb, s^.x, w^.rms_ffn_weight + l * p^.dim, p^.dim);

      // FFN
      s^.xq.Quantize(s^.xb, p^.dim);
      MatMul(s^.hb, s^.xq, w^.w1.GetTensor(l), p^.dim, p^.hidden_dim);
      MatMul(s^.hb2, s^.xq, w^.w3.GetTensor(l), p^.dim, p^.hidden_dim);

      // SwiGLU
      for i := 0 to p^.hidden_dim - 1 do
      begin
        sigmoid_val := 1.0 / (1.0 + Exp(-(s^.hb + i)^));
        (s^.hb + i)^ := (s^.hb + i)^ * sigmoid_val * (s^.hb2 + i)^;
      end;

      // Final FFN matmul
      s^.hq.Quantize(s^.hb, p^.hidden_dim);
      MatMul(s^.xb, s^.hq, w^.w2.GetTensor(l), p^.hidden_dim, p^.dim);

      // Residual connection
      for i := 0 to p^.dim - 1 do
        (s^.x + i)^ := (s^.x + i)^ + (s^.xb + i)^;
    end;

    // Final RMS norm
    RMSNorm(s^.x, s^.x, w^.rms_final_weight, p^.dim);

    // Classifier
    s^.xq.Quantize(s^.x, p^.dim);
    MatMul(s^.logits, s^.xq, w^.wcls^, p^.dim, p^.vocab_size);

    Result := s^.logits;
  end;

  { Build transformer }
  procedure BuildTransformer(var t: TTransformer; checkpoint_path: string; ctx_length: longint);
  begin
    ReadCheckpoint(checkpoint_path, t.config, t.weights, t.Data, t.file_size, ctx_length);
    MallocRunState(t.state, t.config);
  end;

  { Free transformer }
  procedure FreeTransformer(var t: TTransformer);
  begin
    FreeMem(t.weights.q_tokens);
    FreeMem(t.weights.token_embedding_table);
    // Arrays are automatically freed by Pascal
    if t.weights.wcls <> t.weights.q_tokens then
      FreeMem(t.weights.wcls);
    FreeMem(t.Data);
    FreeRunState(t.state);
  end;

  { Tokenizer functions }
  procedure LoadPromptTemplate(var out_template: array of char; with_system_prompt: boolean; enable_thinking: boolean);
  var
    template_content: string;
    i: integer;
  begin
    FillChar(out_template, SizeOf(out_template), 0);

    // Hard-coded templates based on settings
    if with_system_prompt then
    begin
      if enable_thinking then
        // Template: with-system-and-thinking
        template_content := '<|im_start|>system' + #10 + '%s' + #10 + '<|im_end|>' + #10 + '<|im_start|>user' + #10 + '%s' + #10 + '<|im_end|>' + #10 + '<|im_start|>assistant' + #10
      else
        // Template: with-system
        template_content := '<|im_start|>system' + #10 + '%s' + #10 + '<|im_end|>' + #10 + '<|im_start|>user' + #10 + '%s' + #10 + '<|im_end|>' + #10 + '<|im_start|>assistant' + #10 + '<think>' + #10 + #10 + '</think>' + #10 + #10;
    end
    else
    begin
      if enable_thinking then
        // Template: with-thinking
        template_content := '<|im_start|>user' + #10 + '%s' + #10 + '<|im_end|>' + #10 + '<|im_start|>assistant' + #10
      else
        // Template: basic
        template_content := '<|im_start|>user' + #10 + '%s' + #10 + '<|im_end|>' + #10 + '<|im_start|>assistant' + #10 + '<think>' + #10 + #10 + '</think>' + #10 + #10;
    end;

    // Copy content to template array
    for i := 1 to Min(Length(template_content), High(out_template)) do
      out_template[i - 1] := template_content[i];
  end;

  procedure BuildTokenizer(var t: TTokenizer; checkpoint_path: string; vocab_size: longint; enable_thinking: boolean);
  var
    tokenizer_path: string;
    f: file;
    x: longint = 0;
    len: longint = 0;
    i: longint;
  begin
    tokenizer_path := checkpoint_path + '.tokenizer';
    t.vocab_size := vocab_size;

    GetMem(t.vocab, vocab_size * SizeOf(PChar));
    GetMem(t.merge_scores, vocab_size * SizeOf(single));

    AssignFile(f, tokenizer_path);
    Reset(f, 1);

    BlockRead(f, t.max_token_length, SizeOf(longint));
    BlockRead(f, t.bos_token_id, SizeOf(longint));
    BlockRead(f, t.eos_token_id, SizeOf(longint));

    for i := 0 to vocab_size - 1 do
    begin
      BlockRead(f, (t.merge_scores + i)^, SizeOf(single), x);
      if x = 0 then
      begin
        GetMem((t.vocab + i)^, 1);
        (t.vocab + i)^[0] := #0;
      end
      else
      begin
        BlockRead(f, len, SizeOf(longint));
        GetMem((t.vocab + i)^, len + 1);
        BlockRead(f, (t.vocab + i)^[0], len);
        (t.vocab + i)^[len] := #0;
      end;
    end;

    CloseFile(f);

    LoadPromptTemplate(t.prompt_template, False, enable_thinking);
    LoadPromptTemplate(t.system_prompt_template, True, enable_thinking);
  end;

  { QuickSort partition function for top-p sampling }
  function QuickSortPartition(probindex: PProbIndex; low, high: longint): longint;
  var
    pivot: single;
    i, j: longint;
    temp: TProbIndex;
  begin
    pivot := (probindex + high)^.prob;
    i := low - 1;

    for j := low to high - 1 do
    begin
      if (probindex + j)^.prob >= pivot then
      begin
        Inc(i);
        temp := (probindex + i)^;
        (probindex + i)^ := (probindex + j)^;
        (probindex + j)^ := temp;
      end;
    end;

    temp := (probindex + i + 1)^;
    (probindex + i + 1)^ := (probindex + high)^;
    (probindex + high)^ := temp;

    Result := i + 1;
  end;

  { QuickSort for top-p sampling }
  procedure QuickSort(probindex: PProbIndex; low, high: longint);
  var
    pi: longint;
  begin
    if low < high then
    begin
      pi := QuickSortPartition(probindex, low, high);
      QuickSort(probindex, low, pi - 1);
      QuickSort(probindex, pi + 1, high);
    end;
  end;

  { Sampling functions }
  function RandomU32(var state: QWord): longword;
  begin
    state := state xor (state shr 12);
    state := state xor (state shl 25);
    state := state xor (state shr 27);
    Result := (state * $2545F4914F6CDD1D) shr 32;
  end;

  function RandomF32(var state: QWord): single;
  begin
    Result := (RandomU32(state) shr 8) / 16777216.0;
  end;

  function SampleArgmax(const probabilities: PSingle; n: longint): longint;
  var
    max_i, i: longint;
    max_p: single;
  begin
    max_i := 0;
    max_p := probabilities^;
    for i := 1 to n - 1 do
    begin
      if (probabilities + i)^ > max_p then
      begin
        max_i := i;
        max_p := (probabilities + i)^;
      end;
    end;
    Result := max_i;
  end;

  function SampleMult(probabilities: PSingle; n: longint; coin: single): longint;
  var
    cdf: single;
    i: longint;
  begin
    cdf := 0;
    for i := 0 to n - 1 do
    begin
      cdf := cdf + (probabilities + i)^;
      if coin < cdf then
        Exit(i);
    end;
    Result := n - 1;
  end;

  function SampleTopP(probabilities: PSingle; n: longint; topp: single; probindex: PProbIndex; coin: single): longint;
  var
    n0, i, j, last_idx: longint;
    cutoff, cumulative_prob, r, cdf: single;
    temp: TProbIndex;
  begin
    n0 := 0;
    cutoff := (1.0 - topp) / (n - 1);

    for i := 0 to n - 1 do
    begin
      if (probabilities + i)^ >= cutoff then
      begin
        (probindex + n0)^.index := i;
        (probindex + n0)^.prob := (probabilities + i)^;
        Inc(n0);
      end;
    end;

    // Sort probindex array by probability in descending order using QuickSort
    if n0 > 1 then
      QuickSort(probindex, 0, n0 - 1);

    cumulative_prob := 0;
    last_idx := n0 - 1;

    for i := 0 to n0 - 1 do
    begin
      cumulative_prob := cumulative_prob + (probindex + i)^.prob;
      if cumulative_prob > topp then
      begin
        last_idx := i;
        Break;
      end;
    end;

    r := coin * cumulative_prob;
    cdf := 0;

    for i := 0 to last_idx do
    begin
      cdf := cdf + (probindex + i)^.prob;
      if r < cdf then
        Exit((probindex + i)^.index);
    end;

    Result := (probindex + last_idx)^.index;
  end;

  { Build sampler }
  procedure BuildSampler(var sampler: TSampler; vocab_size: longint; temperature: single; topp: single; rng_seed: QWord);
  begin
    sampler.vocab_size := vocab_size;
    sampler.temperature := temperature;
    sampler.topp := topp;
    sampler.rng_state := rng_seed;
    GetMem(sampler.probindex, vocab_size * SizeOf(TProbIndex));
  end;

  { Free sampler }
  procedure FreeSampler(var sampler: TSampler);
  begin
    FreeMem(sampler.probindex);
  end;

  { Sample function }
  function Sample(var sampler: TSampler; logits: PSingle): longint;
  var
    i: longint;
    coin: single;
  begin
    if sampler.temperature = 0 then
    begin
      // Greedy argmax sampling
      Result := SampleArgmax(logits, sampler.vocab_size);
    end
    else
    begin
      // Apply temperature
      for i := 0 to sampler.vocab_size - 1 do
        (logits + i)^ := (logits + i)^ / sampler.temperature;

      // Apply softmax
      Softmax(logits, sampler.vocab_size);

      // Sample
      coin := RandomF32(sampler.rng_state);

      if (sampler.topp <= 0) or (sampler.topp >= 1) then
        Result := SampleMult(logits, sampler.vocab_size, coin)
      else
        Result := SampleTopP(logits, sampler.vocab_size, sampler.topp, sampler.probindex, coin);
    end;
  end;

  procedure FreeTokenizer(var t: TTokenizer);
  var
    i: longint;
  begin
    for i := 0 to t.vocab_size - 1 do
      FreeMem(t.vocab[i]);
    FreeMem(t.vocab);
    FreeMem(t.merge_scores);
  end;

  function Decode(var t: TTokenizer; token: longint): pchar;
  begin
    Result := t.vocab[token];
  end;

  function StrLookup(str: pchar; vocab: PPChar; vocab_size: longint): longint;
  var
    i: longint;
  begin
    for i := 0 to vocab_size - 1 do
    begin
      if StrComp(str, vocab[i]) = 0 then
        Exit(i);
    end;
    Result := -1;
  end;

  procedure Encode(var t: TTokenizer; Text: pchar; tokens: PLongInt; var n_tokens: longint);
  var
    str_buffer: pchar;
    special_token: array[0..64] of char = '';
    c: pchar;
    id: longint = 0;
    found_special_token: longint;
    end_of_token_pos, k: longint;
    best_score: single;
    best_id, best_idx: longint;
    i: longint;
    merged_str: pchar;
    len1, len2: longint;
  begin
    // Allocate buffer for merge candidates
    GetMem(str_buffer, (t.max_token_length * 2 + 1 + 2) * SizeOf(char));

    n_tokens := 0;

    // Process raw UTF-8 byte sequence
    c := Text;
    while c^ <> #0 do
    begin
      found_special_token := 0;

      // Set buffer to current byte
      str_buffer[0] := c^;
      str_buffer[1] := #0;

      // Handle special tokens
      if c^ = '<' then
      begin
        end_of_token_pos := -1;
        found_special_token := 0;

        for k := 0 to 63 do
        begin
          if (c + k)^ = #0 then
            Break;
          if (c + k)^ = '>' then
          begin
            end_of_token_pos := k;
            Break;
          end;
        end;

        if end_of_token_pos <> -1 then
        begin
          Move(c^, special_token[0], end_of_token_pos + 1);
          special_token[end_of_token_pos + 1] := #0;

          id := StrLookup(@special_token[0], t.vocab, t.vocab_size);
          if id <> -1 then
          begin
            Inc(c, end_of_token_pos);
            found_special_token := 1;
          end;
        end;
      end;

      // Not a special token, look up single character
      if found_special_token = 0 then
        id := StrLookup(str_buffer, t.vocab, t.vocab_size);

      if id <> -1 then
      begin
        // Found in vocab, add as token
        (tokens + n_tokens)^ := id;
        Inc(n_tokens);
      end
      else
      begin
        WriteLn(StdErr, 'Warning: unknown character code point ', Ord(str_buffer[0]), ' in input, skipping.');
        Inc(n_tokens);
      end;

      Inc(c);
    end;

    // Merge best consecutive pairs
    while True do
    begin
      best_score := -1e10;
      best_id := -1;
      best_idx := -1;

      for i := 0 to n_tokens - 2 do
      begin
        // Check if we can merge the pair (tokens[i], tokens[i+1])
        GetMem(merged_str, (t.max_token_length * 2 + 1) * SizeOf(char));

        // Copy first token
        StrCopy(merged_str, t.vocab[(tokens + i)^]);
        len1 := StrLen(t.vocab[(tokens + i)^]);
        len2 := StrLen(t.vocab[(tokens + i + 1)^]);

        // Check if concatenation would exceed max token length
        if len1 + len2 <= t.max_token_length then
        begin
          // Concatenate second token
          Move(t.vocab[(tokens + i + 1)^]^, (merged_str + len1)^, len2 * SizeOf(char));
          (merged_str + len1 + len2)^ := #0;

          id := StrLookup(merged_str, t.vocab, t.vocab_size);

          if (id <> -1) and ((t.merge_scores + id)^ > best_score) then
          begin
            best_score := (t.merge_scores + id)^;
            best_id := id;
            best_idx := i;
          end;
        end;

        FreeMem(merged_str);
      end;

      if best_idx = -1 then
        Break; // No more pairs to merge

      // Merge the consecutive pair
      (tokens + best_idx)^ := best_id;

      // Delete token at position best_idx+1, shift sequence back
      for i := best_idx + 1 to n_tokens - 2 do
        (tokens + i)^ := (tokens + i + 1)^;

      Dec(n_tokens);
    end;

    FreeMem(str_buffer);
  end;

  { Generation functions }
  procedure Generate(var transformer: TTransformer; var tokenizer: TTokenizer; var sampler: TSampler; prompt: pchar);
  var
    empty_prompt: pchar;
    num_prompt_tokens: longint;
    prompt_tokens: PLongInt;
    Next, token, pos: longint;
    logits: PSingle;
    start_time, first_token_time, end_time: TDateTime;
    tokens_generated: longint;
    time_to_first_token, total_time: double;
    tokens_per_second: double;
  begin
    empty_prompt := '';
    if prompt = nil then
      prompt := empty_prompt;

    // Encode prompt into tokens
    num_prompt_tokens := 0;
    GetMem(prompt_tokens, (StrLen(prompt) + 3) * SizeOf(longint));
    Encode(tokenizer, prompt, prompt_tokens, num_prompt_tokens);

    if num_prompt_tokens < 1 then
    begin
      WriteLn(StdErr, 'Please provide a prompt using -i <string> on the command line.');
      Halt(1);
    end;

    // Start main loop
    token := prompt_tokens^;
    pos := 0;
    tokens_generated := 0;
    start_time := Now;
    first_token_time := 0;

    while pos < transformer.config.seq_len do
    begin
      // Forward transformer to get logits
      logits := Forward(transformer, token, pos);

      // Advance state machine
      if pos < num_prompt_tokens - 1 then
        Next := (prompt_tokens + pos + 1)^
      else
        Next := Sample(sampler, logits);

      Inc(pos);

      // Print token
      Write(Decode(tokenizer, token));
      Flush(Output);

      // Track first token time
      if (pos >= num_prompt_tokens) and (first_token_time = 0) then
        first_token_time := Now;

      // Count generated tokens (excluding prompt tokens)
      if pos >= num_prompt_tokens then
        Inc(tokens_generated);

      token := Next;

      // Check termination condition
      if (pos >= num_prompt_tokens) and ((Next = tokenizer.bos_token_id) or (Next = tokenizer.eos_token_id)) then
        Break;
    end;

    WriteLn;

    // Calculate and display statistics
    end_time := Now;
    total_time := MilliSecondsBetween(start_time, end_time) / 1000.0;
    time_to_first_token := MilliSecondsBetween(start_time, first_token_time) / 1000.0;
    tokens_per_second := tokens_generated / total_time;

    WriteLn('--- Response Statistics ---');
    WriteLn('Prompt tokens:', num_prompt_tokens,
      ' | Generated tokens: ', tokens_generated,
      ' | Total tokens: ', num_prompt_tokens + tokens_generated,
      ' | Time to first token: ', time_to_first_token: 0: 3,
      's | Total response time: ', total_time: 0: 3,
      's | Tokens per second: ', tokens_per_second: 0: 2, ' tk/s');

    FreeMem(prompt_tokens);
  end;

  procedure Chat(var transformer: TTransformer; var tokenizer: TTokenizer; var sampler: TSampler; cli_user_prompt: pchar; system_prompt: pchar);
  var
    user_prompt: array[0..PROMPT_BUFFER_SIZE - 1] of char;
    rendered_prompt: array[0..PROMPT_BUFFER_SIZE - 1] of char;
    num_prompt_tokens: longint = 0;
    prompt_tokens: PLongInt;
    user_turn, Next, token, pos: longint;
    logits: PSingle;
    temp_str: string;
    start_time, first_token_time, end_time: TDateTime;
    tokens_generated: longint;
    time_to_first_token, total_time: double;
    tokens_per_second: double;
    response_started: boolean;
  begin
    GetMem(prompt_tokens, PROMPT_BUFFER_SIZE * SizeOf(longint));

    user_turn := 1;
    pos := 0;
    tokens_generated := 0;
    response_started := False;

    while True do
    begin
      // Check context window
      if pos >= transformer.config.seq_len then
      begin
        WriteLn;
        WriteLn('(context window full, clearing)');
        user_turn := 1;
        pos := 0;
      end;

      // User's turn
      if user_turn <> 0 then
      begin
        if cli_user_prompt <> nil then
        begin
          if pos > 0 then
            Break;
          StrCopy(@user_prompt[0], cli_user_prompt);
        end
        else
        begin
          Write('> ');
          ReadLn(temp_str);
          if Length(temp_str) = 0 then
            Break;
          StrCopy(@user_prompt[0], PChar(temp_str));
        end;

        // Render prompts into template
        if (pos = 0) and (system_prompt <> nil) then
        begin
          // Replace %s placeholders in system prompt template
          temp_str := string(tokenizer.system_prompt_template);
          temp_str := StringReplace(temp_str, '%s', system_prompt, [rfReplaceAll]);
          temp_str := StringReplace(temp_str, '%s', PChar(@user_prompt[0]), [rfReplaceAll]);
          StrCopy(@rendered_prompt[0], PChar(temp_str));
        end
        else
        begin
          // Replace %s placeholder in prompt template
          temp_str := string(tokenizer.prompt_template);
          temp_str := StringReplace(temp_str, '%s', PChar(@user_prompt[0]), [rfReplaceAll]);
          StrCopy(@rendered_prompt[0], PChar(temp_str));
        end;

        //writeLn('Prompt: '+temp_str);

        // Encode prompt
        Encode(tokenizer, @rendered_prompt[0], prompt_tokens, num_prompt_tokens);
        pos := 0;
        user_turn := 0;
        start_time := Now;
        first_token_time := 0;
      end;

      // Determine token to pass to transformer
      if pos < num_prompt_tokens then
        token := (prompt_tokens + pos)^
      else
        token := Next;

      // Forward transformer
      logits := Forward(transformer, token, pos);
      Inc(pos);
      Next := Sample(sampler, logits);

      // Assistant responding
      if pos >= num_prompt_tokens then
      begin
        if (Next = tokenizer.bos_token_id) or (Next = tokenizer.eos_token_id) then
        begin
          WriteLn;

          // Calculate and display statistics for this response
          if response_started then
          begin
            end_time := Now;
            total_time := MilliSecondsBetween(start_time, end_time) / 1000.0;
            time_to_first_token := MilliSecondsBetween(start_time, first_token_time) / 1000.0;
            tokens_per_second := tokens_generated / total_time;

            WriteLn('--- Response Statistics ---');
            WriteLn('Prompt tokens:', num_prompt_tokens,
              ' | Generated tokens: ', tokens_generated,
              ' | Total tokens: ', num_prompt_tokens + tokens_generated,
              ' | Time to first token: ', time_to_first_token: 0: 3,
              's | Total response time: ', total_time: 0: 3,
              's | Tokens per second: ', tokens_per_second: 0: 2, ' tk/s');
          end;

          user_turn := 1;
          tokens_generated := 0;
          response_started := False;
        end
        else
        begin
          // Track first token time
          if not response_started then
          begin
            first_token_time := Now;
            response_started := True;
          end;

          Write(Decode(tokenizer, Next));
          Flush(Output);
          Inc(tokens_generated);
        end;
      end;
    end;

    FreeMem(prompt_tokens);
  end;

  procedure ErrorUsage;
  begin
    WriteLn(StdErr, 'Usage:   qwen3 <checkpoint> [options]');
    WriteLn(StdErr, 'Example: qwen3 Qwen3-4B.bin -r 1');
    WriteLn(StdErr, 'Options:');
    WriteLn(StdErr, '  -t <float>  temperature in [0,inf], default 1.0');
    WriteLn(StdErr, '  -p <float>  p value in top-p (nucleus) sampling in [0,1], default 0.9');
    WriteLn(StdErr, '  -s <int>    random seed, default time(NULL)');
    WriteLn(StdErr, '  -c <int>    context window size, 0 (default) = max_seq_len');
    WriteLn(StdErr, '  -m <string> mode: generate|chat, default: chat');
    WriteLn(StdErr, '  -i <string> input prompt');
    WriteLn(StdErr, '  -y <string> system prompt in chat mode, default is none');
    WriteLn(StdErr, '  -r <int>    reasoning mode, 0 (default) = no thinking, 1 = thinking');
    Halt(1);
  end;

  { Main program }
var
  checkpoint_path: pchar = nil;
  temperature: single = 0.6;
  topp: single = 0.95;
  prompt: pchar = nil;
  rng_seed: QWord = 0;
  mode: pchar = 'chat';
  system_prompt: pchar = nil;
  enable_thinking: longint = 0;
  ctx_length: longint = 4096;
  transformer: TTransformer;
  tokenizer: TTokenizer;
  sampler: TSampler;
  i: longint;
  arg: pchar;
begin
  // Configure console for UTF-8 output
  ConfigureConsoleForUTF8;

  // Parse command line arguments
  if ParamCount >= 1 then
    checkpoint_path := PChar(ParamStr(1))
  else
    ErrorUsage;

  i := 2;
  while i <= ParamCount do
  begin
    if i + 1 > ParamCount then
      ErrorUsage;

    arg := PChar(ParamStr(i));
    if (arg[0] <> '-') or (StrLen(arg) <> 2) then
      ErrorUsage;

    case arg[1] of
      't': temperature := StrToFloat(ParamStr(i + 1));
      'p': topp := StrToFloat(ParamStr(i + 1));
      's': rng_seed := StrToInt64(ParamStr(i + 1));
      'c': ctx_length := StrToInt(ParamStr(i + 1));
      'i': prompt := PChar(ParamStr(i + 1));
      'm': mode := PChar(ParamStr(i + 1));
      'y': system_prompt := PChar(ParamStr(i + 1));
      'r': enable_thinking := StrToInt(ParamStr(i + 1));
      else
        ErrorUsage;
    end;

    Inc(i, 2);
  end;

  // Parameter validation
  if rng_seed <= 0 then
    rng_seed := QWord(DateTimeToUnix(Now));
  if temperature < 0 then
    temperature := 0;
  if (topp < 0) or (topp > 1.0) then
    topp := 0.9;

  // Build transformer
  BuildTransformer(transformer, checkpoint_path, ctx_length);

  writeln('BuildTransformer.');

  // Build tokenizer
  BuildTokenizer(tokenizer, checkpoint_path, transformer.config.vocab_size, enable_thinking = 1);

  writeln('BuildTokenizer.');

  // Build sampler
  BuildSampler(sampler, transformer.config.vocab_size, temperature, topp, rng_seed);
  writeln('BuildSampler.');


  // Print model info if no prompt
  if prompt = nil then
  begin
    WriteLn('hidden_size=', transformer.config.dim,
      ', intermediate_size=', transformer.config.hidden_dim,
      ', num_hidden_layers=', transformer.config.n_layers,
      ', num_attention_heads=', transformer.config.n_heads,
      ', num_kv_heads=', transformer.config.n_kv_heads,
      ', head_dim=', transformer.config.head_dim,
      ', ctx_length=', transformer.config.seq_len,
      ', vocab_size=', transformer.config.vocab_size,
      ', shared_classifier=', transformer.config.shared_classifier,
      ', quantization_block_size=', transformer.config.group_size);
  end;

  // Run
  if StrComp(mode, 'generate') = 0 then
    Generate(transformer, tokenizer, sampler, prompt)
  else if StrComp(mode, 'chat') = 0 then
    Chat(transformer, tokenizer, sampler, prompt, system_prompt)
  else
  begin
    WriteLn(StdErr, 'Unknown mode: ', mode);
    ErrorUsage;
  end;

  // Cleanup
  FreeSampler(sampler);
  FreeTokenizer(tokenizer);
  FreeTransformer(transformer);
end.
