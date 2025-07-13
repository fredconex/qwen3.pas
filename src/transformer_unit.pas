{ Transformer unit for Qwen-3 model inference in FreePascal }
unit Transformer_Unit;

{$mode objfpc}{$H+}
{$modeswitch advancedrecords}

interface

uses
  SysUtils,
  Classes,
  Math,
  DateUtils,
  Windows,
  Tokenizer_Unit;

const
  PROMPT_BUFFER_SIZE = 32768;

type
  { Quantized tensor structure }
  PInt8QuantizedTensor = ^TInt8QuantizedTensor;

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
    function GetTensorPtr(Index: integer): PInt8QuantizedTensor;
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
    q_tokens: PInt8QuantizedTensor;
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
    wcls: PInt8QuantizedTensor;
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
    
    // Methods
    procedure Build(checkpoint_path: string; ctx_length: longint);
    procedure Free;
    function Forward(token: longint; pos: longint): PSingle;
    procedure Generate(var tokenizer: TTokenizer; var sampler: TSampler; prompt: pchar);
    procedure Chat(var tokenizer: TTokenizer; var sampler: TSampler; cli_user_prompt: pchar; system_prompt: pchar);
  end;

var
  GS: longint = 0; // Global group size for quantization

implementation

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

function TInt8QuantizedTensorArray.GetTensorPtr(Index: integer): PInt8QuantizedTensor;
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

{ TTransformer method implementations }
procedure TTransformer.Build(checkpoint_path: string; ctx_length: longint);
begin
  ReadCheckpoint(checkpoint_path, self.config, self.weights, self.Data, self.file_size, ctx_length);
  MallocRunState(self.state, self.config);
end;

procedure TTransformer.Free;
begin
  FreeMem(self.weights.q_tokens);
  FreeMem(self.weights.token_embedding_table);
  // Arrays are automatically freed by Pascal
  if self.weights.wcls <> self.weights.q_tokens then
    FreeMem(self.weights.wcls);
  FreeMem(self.Data);
  FreeRunState(self.state);
end;

function TTransformer.Forward(token: longint; pos: longint): PSingle;
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
  p := @self.config;
  w := @self.weights;
  s := @self.state;
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

procedure TTransformer.Generate(var tokenizer: TTokenizer; var sampler: TSampler; prompt: pchar);
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
  Tokenizer.Encode(prompt, prompt_tokens, num_prompt_tokens);

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

  while pos < self.config.seq_len do
  begin
    // Forward transformer to get logits
    logits := self.Forward(token, pos);

    // Advance state machine
    if pos < num_prompt_tokens - 1 then
      Next := (prompt_tokens + pos + 1)^
    else
      Next := Sampler.Sample(logits);

    Inc(pos);

    // Print token
    Write(Tokenizer.Decode(token));
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

procedure TTransformer.Chat(var tokenizer: TTokenizer; var sampler: TSampler; cli_user_prompt: pchar; system_prompt: pchar);
var
  user_prompt: array[0..PROMPT_BUFFER_SIZE - 1] of char;
  rendered_prompt: array[0..PROMPT_BUFFER_SIZE - 1] of char;
  num_prompt_tokens: longint = 0;
  prompt_tokens: PLongInt;
  Next : longint = 0;
  user_turn, token, pos: longint;
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
    if pos >= self.config.seq_len then
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
      Tokenizer.Encode(@rendered_prompt[0], prompt_tokens, num_prompt_tokens);
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
    logits := self.Forward(token, pos);
    Inc(pos);
    Next := Sampler.Sample(logits);

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

        Write(Tokenizer.Decode(Next));
        Flush(Output);
        Inc(tokens_generated);
      end;
    end;
  end;

  FreeMem(prompt_tokens);
end;

end. 
