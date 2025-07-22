{ Transformer unit for Qwen-3 model inference in FreePascal }
unit Transformer_Unit;

{$mode objfpc}{$H+}
{$modeswitch advancedrecords}
{$modeswitch nestedprocvars}
{$modeswitch anonymousfunctions}

interface

uses
  SysUtils,
  Classes,
  Math,
  DateUtils,
  mtprocs,
  Windows,
  Tokenizer_Unit,
  Tensor_Unit;

type
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
    q_tokens: PInt8QuantizedTensor;           // Quantized token embedding tensor (vocab_size x dim)
    token_embedding_table: PSingle;           // Dequantized token embedding table (vocab_size x dim)
    rms_att_weight: PSingle;                  // RMSNorm weights for attention layers (n_layers x dim)
    rms_ffn_weight: PSingle;                  // RMSNorm weights for FFN layers (n_layers x dim)
    wq: TInt8QuantizedTensorArray;            // Quantized weight matrices for Q projection (n_layers x dim x (n_heads x head_dim))
    wk: TInt8QuantizedTensorArray;            // Quantized weight matrices for K projection (n_layers x dim x (n_kv_heads x head_dim))
    wv: TInt8QuantizedTensorArray;            // Quantized weight matrices for V projection (n_layers x dim x (n_kv_heads x head_dim))
    wo: TInt8QuantizedTensorArray;            // Quantized weight matrices for output projection (n_layers x (n_heads x head_dim) x dim)
    q_norm_weights: PSingle;                  // RMSNorm weights for Q heads (n_layers x head_dim)
    k_norm_weights: PSingle;                  // RMSNorm weights for K heads (n_layers x head_dim)
    w1: TInt8QuantizedTensorArray;            // Quantized weight matrices for FFN first layer (n_layers x dim x hidden_dim)
    w2: TInt8QuantizedTensorArray;            // Quantized weight matrices for FFN second layer (n_layers x hidden_dim x dim)
    w3: TInt8QuantizedTensorArray;            // Quantized weight matrices for FFN gate layer (n_layers x dim x hidden_dim)
    rms_final_weight: PSingle;                // RMSNorm weights for final normalization (dim)
    wcls: PInt8QuantizedTensor;               // Quantized classifier weights (dim x vocab_size) or shared with q_tokens
  end;

  { Run state }
  TRunState = record
    batch_size: integer;
    x: array of PSingle;      // [batch_size][dim]
    xb: array of PSingle;     // [batch_size][all_heads_dim]
    hb: array of PSingle;     // [batch_size][hidden_dim]
    hb2: array of PSingle;    // [batch_size][hidden_dim]
    xq: array of TInt8QuantizedTensor; // [batch_size]
    hq: array of TInt8QuantizedTensor; // [batch_size]
    q: array of PSingle;      // [batch_size][all_heads_dim]
    k: array of PSingle;      // [batch_size][kv_dim]
    v: array of PSingle;      // [batch_size][kv_dim]
    att: array of PSingle;    // [batch_size][n_heads * seq_len]
    logits: array of PSingle; // [batch_size][vocab_size]
    key_cache: PSingle;   // Cached keys for all layers (n_layers x seq_len x n_kv_heads x head_dim)
    value_cache: PSingle; // Cached values for all layers (n_layers x seq_len x n_kv_heads x head_dim)
  end;

  PConfig = ^TConfig;
  PTransformerWeights = ^TTransformerWeights;
  PRunState = ^TRunState;

  { Transformer structure }
  TTransformer = class
  private
    procedure MapWeightsToMemory(var DataPtr: Pointer);
    procedure MallocRunState(var s: TRunState; var p: TConfig);
    procedure FreeRunState(var s: TRunState);
    procedure LoadFromFile(checkpoint: string; ctx_length: longint);
    function GenerateFromTokens(var tokenizer: TTokenizer; var sampler: TSampler; prompt_tokens: PLongInt; num_prompt_tokens: longint; start_pos: longint; output_prompt: boolean): longint;

    procedure ApplyRotaryEmbeddings(const ptr: PSingle; const head_dim, pos: longint);
    procedure ProcessAttentionLayer(var s: TRunState; const w: TTransformerWeights; const p: TConfig; const l, pos, batch_idx: longint);
    procedure ProcessFFNLayer(var s: TRunState; const w: TTransformerWeights; const p: TConfig; const l, batch_idx: longint);
    procedure ApplyResidualConnection(const x, xb: PSingle; const dim: longint);
    procedure ProcessFinalLayer(var s: TRunState; const w: TTransformerWeights; const p: TConfig; batch_idx: longint);
    procedure RMSNorm(const o, x, weight: PSingle; Size: longint);
    function Sigmoid(const x: single): single;
    procedure SwiGLU(var hb, hb2: PSingle; hidden_dim: longint);
    // --- Begin deduplication helpers ---
    procedure DoEmbeddingCopy(s: PRunState; w: PTransformerWeights; p: PConfig; token, batch_idx: integer);
    procedure DoLayerPass(s: PRunState; w: PTransformerWeights; p: PConfig; l, pos, batch_idx: integer);
    procedure DoFinalLayer(s: PRunState; w: PTransformerWeights; p: PConfig; batch_idx: integer);
    // --- End deduplication helpers ---
  public
    config: TConfig;
    weights: TTransformerWeights;
    state: TRunState;
    Data: Pointer;
    file_size: int64;

    constructor Create(checkpoint_path: string; ctx_length: longint);
    destructor Destroy; override;
    function Forward(token: longint; pos: longint): PSingle;
    procedure ForwardBatchPrompt(tokens, positions: PLongInt; batch_count: integer); // NEW
    procedure Generate(var tokenizer: TTokenizer; var sampler: TSampler; prompt: PChar);
    procedure Chat(var tokenizer: TTokenizer; var sampler: TSampler; cli_user_prompt: PChar; system_prompt: PChar);
  end;

implementation

{ Memory map weights with improved structure and readability }
procedure TTransformer.MapWeightsToMemory(var DataPtr: Pointer);
var
  FloatPtr: PSingle;
  BytePtr: pbyte;

  procedure AllocateFloatWeights(var WeightPtr: PSingle; ElementCount: integer);
  begin
    WeightPtr := FloatPtr;
    Inc(FloatPtr, ElementCount);
  end;

  procedure AllocateAndDequantizeTokens(NUM_PARTS: integer = 1);
  var
    TokenTableSize: integer;
    BatchSize: integer;
    NumBatches: integer;

    // Parallel dequantization in batches
{$HINTS OFF}
    procedure DequantProc(BatchIdx: PtrInt; Data: Pointer; Item: TMultiThreadProcItem);
    var
      local_start, local_end, j: integer;
    begin
      local_start := BatchIdx * BatchSize;
      local_end := local_start + BatchSize - 1;
      if local_end >= TokenTableSize then
        local_end := TokenTableSize - 1;
      for j := local_start to local_end do
        (weights.token_embedding_table + j)^ := (weights.q_tokens^.q + j)^ * (weights.q_tokens^.s + (j div weights.q_tokens^.group_size))^;
    end;

  begin
    TokenTableSize := config.vocab_size * config.dim;

    // Initialize quantized token embeddings (single tensor)
    GetMem(weights.q_tokens, SizeOf(TInt8QuantizedTensor));
    weights.q_tokens^.q := PShortInt(BytePtr);
    Inc(BytePtr, TokenTableSize * SizeOf(shortint));
    weights.q_tokens^.s := PSingle(BytePtr);
    Inc(BytePtr, (TokenTableSize div config.group_size) * SizeOf(single));
    weights.q_tokens^.group_size := config.group_size;

    // Allocate token embedding table
    GetMem(weights.token_embedding_table, TokenTableSize * SizeOf(single));

    // Batching
    NumBatches := Min(NUM_PARTS, TokenTableSize);  // don't create more batches than elements
    BatchSize := (TokenTableSize + NumBatches - 1) div NumBatches;
    ProcThreadPool.DoParallelNested(@DequantProc, 0, NumBatches - 1, nil);
  end;

begin
  FloatPtr := PSingle(DataPtr);

  // Map float weights in order
  AllocateFloatWeights(weights.rms_att_weight, config.n_layers * config.dim);
  AllocateFloatWeights(weights.rms_ffn_weight, config.n_layers * config.dim);
  AllocateFloatWeights(weights.rms_final_weight, config.dim);
  AllocateFloatWeights(weights.q_norm_weights, config.n_layers * config.head_dim);
  AllocateFloatWeights(weights.k_norm_weights, config.n_layers * config.head_dim);

  // Switch to byte pointer for quantized data
  BytePtr := pbyte(FloatPtr);

  AllocateAndDequantizeTokens(128);

  // Map quantized weight matrices
  with config do
  begin
    // Initialize arrays using the new type methods
    weights.wq.Initialize(n_layers, Pointer(BytePtr), dim * (n_heads * head_dim), group_size);
    weights.wk.Initialize(n_layers, Pointer(BytePtr), dim * (n_kv_heads * head_dim), group_size);
    weights.wv.Initialize(n_layers, Pointer(BytePtr), dim * (n_kv_heads * head_dim), group_size);
    weights.wo.Initialize(n_layers, Pointer(BytePtr), (n_heads * head_dim) * dim, group_size);

    // Feed-forward network weights
    weights.w1.Initialize(n_layers, Pointer(BytePtr), dim * hidden_dim, group_size);
    weights.w2.Initialize(n_layers, Pointer(BytePtr), hidden_dim * dim, group_size);
    weights.w3.Initialize(n_layers, Pointer(BytePtr), dim * hidden_dim, group_size);

    // Classifier weights (shared or separate)
    if shared_classifier = 1 then
      weights.wcls := weights.q_tokens
    else
    begin
      GetMem(weights.wcls, SizeOf(TInt8QuantizedTensor));
      weights.wcls^.q := PShortInt(BytePtr);
      Inc(BytePtr, dim * vocab_size * SizeOf(shortint));
      weights.wcls^.s := PSingle(BytePtr);
      Inc(BytePtr, (dim * vocab_size div group_size) * SizeOf(single));
    end;
  end;

  // Update the data pointer
  DataPtr := BytePtr;
end;

{ Read checkpoint }
procedure TTransformer.LoadFromFile(checkpoint: string; ctx_length: longint);
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

    weights_ptr := PChar(Data) + 256;
    MapWeightsToMemory(weights_ptr);
  finally
    fs.Free;
  end;
end;

{ TTransformer method implementations }
constructor TTransformer.Create(checkpoint_path: string; ctx_length: longint);
begin
  LoadFromFile(checkpoint_path, ctx_length);
  MallocRunState(self.state, self.config);
end;

destructor TTransformer.Destroy;
begin
  FreeMem(self.weights.q_tokens);
  FreeMem(self.weights.token_embedding_table);
  // Arrays are automatically freed by Pascal
  if self.weights.wcls <> self.weights.q_tokens then
    FreeMem(self.weights.wcls);
  FreeMem(self.Data);
  FreeRunState(self.state);
  inherited Destroy;
end;

function TTransformer.Forward(token: longint; pos: longint): PSingle;
var
  p: ^TConfig;
  w: ^TTransformerWeights;
  s: ^TRunState;
  l: longint;
begin
  p := @self.config;
  w := @self.weights;
  s := @self.state;

  // Copy token embedding
  self.DoEmbeddingCopy(s, w, p, token, 0);

  // Forward through all layers
  for l := 0 to p^.n_layers - 1 do
    self.DoLayerPass(s, w, p, l, pos, 0);

  // Process final layer
  self.DoFinalLayer(s, w, p, 0);

  Result := s^.logits[0];
end;

function TTransformer.GenerateFromTokens(var tokenizer: TTokenizer; var sampler: TSampler; prompt_tokens: PLongInt; num_prompt_tokens: longint; start_pos: longint; output_prompt: boolean): longint;
var
  Next, token, pos: longint;
  logits: PSingle;
  start_counter, first_token_counter, end_counter, freq: int64;
  tokens_generated: longint;
  time_to_first_token, total_time: double;
  tokens_per_second: double;
  response_started: boolean;
  batch_count, i: longint;
  batch_tokens: array of longint = ();
  batch_positions: array of longint = ();
begin
  // Initialize state
  pos := start_pos;
  tokens_generated := 0;
  QueryPerformanceFrequency(@freq);
  QueryPerformanceCounter(@start_counter);
  first_token_counter := 0;
  response_started := False;

  // Process prompt tokens in a single batch using two-pass method
  if num_prompt_tokens > 0 then
  begin
    pos := 0;
    while pos < num_prompt_tokens do
    begin
      batch_count := Min(self.state.batch_size, num_prompt_tokens - pos);
      SetLength(batch_tokens, batch_count);
      SetLength(batch_positions, batch_count);
      for i := 0 to batch_count - 1 do
      begin
        batch_tokens[i] := (prompt_tokens + pos + i)^;
        batch_positions[i] := pos + i;
      end;
      ForwardBatchPrompt(@batch_tokens[0], @batch_positions[0], batch_count);
      if output_prompt then
        for i := 0 to batch_count - 1 do
          Write(tokenizer.Decode(batch_tokens[i]));
      Inc(pos, batch_count);
    end;
    if output_prompt then
      Flush(Output);
  end;
  if output_prompt then
    Flush(Output);

  // After prompt, process generation one token at a time
  if num_prompt_tokens > 0 then
    token := (prompt_tokens + num_prompt_tokens - 1)^
  else
    token := 0;

  while pos < self.config.seq_len do
  begin
    logits := self.Forward(token, pos);
    Next := sampler.Sample(logits);
    Inc(pos);
    // Check for termination
    if (Next = tokenizer.bos_token_id) or (Next = tokenizer.eos_token_id) then
      Break;
    // Track first token time
    if not response_started then
    begin
      QueryPerformanceCounter(@first_token_counter);
      response_started := True;
    end;
    // Output token and count it
    Write(tokenizer.Decode(Next));
    Flush(Output);
    Inc(tokens_generated);
    token := Next;
  end;

  // Calculate and display statistics
  if response_started then
  begin
    QueryPerformanceCounter(@end_counter);
    total_time := (end_counter - start_counter) / freq;
    time_to_first_token := (first_token_counter - start_counter) / freq;
    if tokens_generated > 0 then
      tokens_per_second := tokens_generated / ((end_counter - first_token_counter) / freq)
    else
      tokens_per_second := 0;
    WriteLn;
    WriteLn('--- Response Statistics ---');
    WriteLn('Prompt tokens:', num_prompt_tokens,
      ' | Generated tokens: ', tokens_generated,
      ' | Total tokens: ', num_prompt_tokens + tokens_generated,
      ' | Time to first token: ', time_to_first_token: 0: 3, 's',
      ' | Total response time: ', total_time: 0: 3, 's',
      ' | Tokens per second: ', tokens_per_second: 0: 2, ' tk/s');
  end;
  Result := pos; // Return final position
end;


const
  PROMPT_BUFFER_SIZE = 32768;

// Simplified Generate method using GenerateFromTokens
procedure TTransformer.Generate(var tokenizer: TTokenizer; var sampler: TSampler; prompt: PChar);
var
  user_prompt: array[0..PROMPT_BUFFER_SIZE - 1] of char;
  num_prompt_tokens: longint;
  prompt_tokens: PLongInt;
  temp_str: string;
  input_prompt: PChar;
begin
  GetMem(prompt_tokens, PROMPT_BUFFER_SIZE * SizeOf(longint));
  num_prompt_tokens := 0;

  // Determine user prompt: CLI or interactive
  if (prompt <> nil) and (StrLen(prompt) > 0) then
  begin
    StrCopy(@user_prompt[0], prompt);
    input_prompt := prompt;
  end
  else
  begin
    Write('> ');
    ReadLn(temp_str);
    if Length(temp_str) = 0 then
    begin
      FreeMem(prompt_tokens);
      Exit;
    end;
    StrCopy(@user_prompt[0], PChar(temp_str));
    input_prompt := @user_prompt[0];
  end;

  // Encode prompt as-is (no template)
  tokenizer.Encode(input_prompt, prompt_tokens, num_prompt_tokens);

  if num_prompt_tokens < 1 then
  begin
    WriteLn(StdErr, 'Please provide a prompt using -i <string> on the command line or enter text interactively.');
    FreeMem(prompt_tokens);
    Halt(1);
  end;

  // Generate response using shared function
  GenerateFromTokens(tokenizer, sampler, prompt_tokens, num_prompt_tokens, 0, True);

  FreeMem(prompt_tokens);
end;

// Simplified Chat method using GenerateFromTokens
procedure TTransformer.Chat(var tokenizer: TTokenizer; var sampler: TSampler; cli_user_prompt: PChar; system_prompt: PChar);
var
  user_prompt: array[0..PROMPT_BUFFER_SIZE - 1] of char;
  rendered_prompt: array[0..PROMPT_BUFFER_SIZE - 1] of char;
  num_prompt_tokens: longint = 0;
  prompt_tokens: PLongInt;
  user_turn, pos: longint;
  temp_str: string;
begin
  GetMem(prompt_tokens, PROMPT_BUFFER_SIZE * SizeOf(longint));

  user_turn := 1;
  pos := 0;

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
        temp_str := StringReplace(tokenizer.prompt_template, '%s', PChar(@user_prompt[0]), [rfReplaceAll]);
        StrCopy(@rendered_prompt[0], PChar(temp_str));
      end;

      // Encode prompt
      tokenizer.Encode(@rendered_prompt[0], prompt_tokens, num_prompt_tokens);
      user_turn := 0;

      // Generate response using shared function
      pos := GenerateFromTokens(tokenizer, sampler, prompt_tokens, num_prompt_tokens, 0, False);

      user_turn := 1;
    end;
  end;

  FreeMem(prompt_tokens);
end;

// --- Begin TTransformer helper method implementations ---
procedure TTransformer.RMSNorm(const o, x, weight: PSingle; Size: longint);
var
  ss: single;
  j: longint;
begin
  ss := 0;
  for j := 0 to Size - 1 do
    ss += (x + j)^ ** 2;
  ss := 1.0 / Sqrt((ss / Size) + 1e-6);
  for j := 0 to Size - 1 do
    (o + j)^ := (weight + j)^ * (ss * (x + j)^);
end;

function TTransformer.Sigmoid(const x: single): single;
begin
  Result := 1.0 / (1.0 + Exp(-x));
end;

procedure TTransformer.SwiGLU(var hb, hb2: PSingle; hidden_dim: longint);
var
  i: longint;
begin
  for i := 0 to hidden_dim - 1 do
    (hb + i)^ := (hb + i)^ * self.Sigmoid((hb + i)^) * (hb2 + i)^;
end;

procedure TTransformer.ApplyRotaryEmbeddings(const ptr: PSingle; const head_dim, pos: longint);
var
  j: longint;
  pfreq, cos_freq, sin_freq, x_val, y_val: single;
  half_dim: longint;
  freq_base: single;
  x_ptr, y_ptr: PSingle;
begin
  half_dim := head_dim shr 1; // Fast division by 2
  freq_base := 1.0 / Power(1e6, 1.0 / half_dim); // Precompute base

  x_ptr := ptr;
  y_ptr := ptr + half_dim;

  for j := 0 to half_dim - 1 do
  begin
    // Use incremental frequency computation
    if j = 0 then
      pfreq := pos * Power(1e6, -j / half_dim)
    else
      pfreq := pfreq * freq_base; // Incremental computation

    cos_freq := Cos(pfreq);
    sin_freq := Sin(pfreq);

    x_val := x_ptr^;
    y_val := y_ptr^;

    x_ptr^ := x_val * cos_freq - y_val * sin_freq;
    y_ptr^ := x_val * sin_freq + y_val * cos_freq;

    Inc(x_ptr);
    Inc(y_ptr);
  end;
end;

procedure TTransformer.ProcessAttentionLayer(var s: TRunState; const w: TTransformerWeights; const p: TConfig; const l, pos, batch_idx: longint);
var
  kv_dim, kv_mul, all_heads_dim: longint;
  loff: QWord;
  i, h, t: longint;
  att_ptr, q_ptr, k_ptr, v_ptr, xb_ptr: PSingle;
  scale: single;
begin
  scale := 1.0 / Sqrt(p.head_dim);
  kv_dim := p.n_kv_heads * p.head_dim;
  kv_mul := p.n_heads div p.n_kv_heads;
  all_heads_dim := p.n_heads * p.head_dim;
  loff := l * QWord(p.seq_len) * kv_dim;
  // Attention RMS norm
  self.RMSNorm(s.xb[batch_idx], s.x[batch_idx], w.rms_att_weight + l * p.dim, p.dim);
  // QKV matmuls
  s.xq[batch_idx].Quantize(s.xb[batch_idx], p.dim);
  s.xq[batch_idx].MatMul(s.q[batch_idx], w.wq.GetTensor(l), p.dim, all_heads_dim);
  s.xq[batch_idx].MatMul(s.k[batch_idx], w.wk.GetTensor(l), p.dim, kv_dim);
  s.xq[batch_idx].MatMul(s.v[batch_idx], w.wv.GetTensor(l), p.dim, kv_dim);
  // Q-RMSNorm + rotate each query head
  for h := 0 to p.n_heads - 1 do
  begin
    q_ptr := s.q[batch_idx] + h * p.head_dim;
    self.RMSNorm(q_ptr, q_ptr, w.q_norm_weights + l * p.head_dim, p.head_dim);
    self.ApplyRotaryEmbeddings(q_ptr, p.head_dim, pos);
  end;
  // K-RMSNorm + rotate each key head
  for h := 0 to p.n_kv_heads - 1 do
  begin
    k_ptr := s.k[batch_idx] + h * p.head_dim;
    self.RMSNorm(k_ptr, k_ptr, w.k_norm_weights + l * p.head_dim, p.head_dim);
    self.ApplyRotaryEmbeddings(k_ptr, p.head_dim, pos);
  end;
  // NOW write K/V to cache for this position (after K RMSNorm+RoPE)
  Move(s.k[batch_idx]^, (s.key_cache + loff + pos * kv_dim)^, kv_dim * SizeOf(single));
  Move(s.v[batch_idx]^, (s.value_cache + loff + pos * kv_dim)^, kv_dim * SizeOf(single));
  // Multihead attention - optimized
  for h := 0 to p.n_heads - 1 do
  begin
    q_ptr := s.q[batch_idx] + h * p.head_dim;
    att_ptr := s.att[batch_idx] + h * p.seq_len;
    for t := 0 to pos do
    begin
      k_ptr := s.key_cache + loff + t * kv_dim + (h div kv_mul) * p.head_dim;
      (att_ptr + t)^ := DotProduct_Hybrid(q_ptr, k_ptr, p.head_dim) * scale;
    end;
    Softmax(att_ptr, pos + 1);
    xb_ptr := s.xb[batch_idx] + h * p.head_dim;
    FillDWord(xb_ptr^, p.head_dim, 0);
    for t := 0 to pos do
    begin
      v_ptr := s.value_cache + loff + t * kv_dim + (h div kv_mul) * p.head_dim;

      // Apply Scalar
      for i := 0 to p.head_dim - 1 do
        (xb_ptr + i)^ := (xb_ptr + i)^ + (att_ptr + t)^ * (v_ptr + i)^;
    end;
  end;
  // Final attention matmul
  s.xq[batch_idx].Quantize(s.xb[batch_idx], all_heads_dim);
  s.xq[batch_idx].MatMul(s.xb[batch_idx], w.wo.GetTensor(l), all_heads_dim, p.dim);
end;

procedure TTransformer.ProcessFFNLayer(var s: TRunState; const w: TTransformerWeights; const p: TConfig; const l, batch_idx: longint);
begin
  self.RMSNorm(s.xb[batch_idx], s.x[batch_idx], w.rms_ffn_weight + l * p.dim, p.dim);
  s.xq[batch_idx].Quantize(s.xb[batch_idx], p.dim);
  s.xq[batch_idx].MatMul(s.hb[batch_idx], w.w1.GetTensor(l), p.dim, p.hidden_dim);
  s.xq[batch_idx].MatMul(s.hb2[batch_idx], w.w3.GetTensor(l), p.dim, p.hidden_dim);
  self.SwiGLU(s.hb[batch_idx], s.hb2[batch_idx], p.hidden_dim);
  s.hq[batch_idx].Quantize(s.hb[batch_idx], p.hidden_dim);
  s.hq[batch_idx].MatMul(s.xb[batch_idx], w.w2.GetTensor(l), p.hidden_dim, p.dim);
end;

procedure TTransformer.ApplyResidualConnection(const x, xb: PSingle; const dim: longint);
var
  i: longint;
begin
  for i := 0 to dim - 1 do
    (x + i)^ := (x + i)^ + (xb + i)^;
end;

procedure TTransformer.ProcessFinalLayer(var s: TRunState; const w: TTransformerWeights; const p: TConfig; batch_idx: longint);
begin
  self.RMSNorm(s.x[batch_idx], s.x[batch_idx], w.rms_final_weight, p.dim);
  s.xq[batch_idx].Quantize(s.x[batch_idx], p.dim);
  s.xq[batch_idx].MatMul(s.logits[batch_idx], w.wcls^, p.dim, p.vocab_size);
end;

procedure TTransformer.MallocRunState(var s: TRunState; var p: TConfig);
var
  all_heads_dim, kv_dim, i: longint;
begin
  // Batch size for prompt evaluation
  s.batch_size := 512;

  all_heads_dim := p.n_heads * p.head_dim;
  kv_dim := p.n_kv_heads * p.head_dim;
  SetLength(s.x, s.batch_size);
  SetLength(s.xb, s.batch_size);
  SetLength(s.hb, s.batch_size);
  SetLength(s.hb2, s.batch_size);
  SetLength(s.xq, s.batch_size);
  SetLength(s.hq, s.batch_size);
  SetLength(s.q, s.batch_size);
  SetLength(s.k, s.batch_size);
  SetLength(s.v, s.batch_size);
  SetLength(s.att, s.batch_size);
  SetLength(s.logits, s.batch_size);
  for i := 0 to s.batch_size - 1 do
  begin
    s.x[i] := AllocMem(p.dim * SizeOf(single));
    s.xb[i] := AllocMem(all_heads_dim * SizeOf(single));
    s.hb[i] := AllocMem(p.hidden_dim * SizeOf(single));
    s.hb2[i] := AllocMem(p.hidden_dim * SizeOf(single));
    s.xq[i].q := AllocMem(all_heads_dim * SizeOf(shortint));
    s.xq[i].s := AllocMem((all_heads_dim div p.group_size) * SizeOf(single));
    s.xq[i].group_size := p.group_size;
    s.hq[i].q := AllocMem(p.hidden_dim * SizeOf(shortint));
    s.hq[i].s := AllocMem((p.hidden_dim div p.group_size) * SizeOf(single));
    s.hq[i].group_size := p.group_size;
    s.q[i] := AllocMem(all_heads_dim * SizeOf(single));
    s.k[i] := AllocMem(kv_dim * SizeOf(single));
    s.v[i] := AllocMem(kv_dim * SizeOf(single));
    s.att[i] := AllocMem(p.n_heads * p.seq_len * SizeOf(single));
    s.logits[i] := AllocMem(p.vocab_size * SizeOf(single));
  end;
  s.key_cache := AllocMem(p.n_layers * QWord(p.seq_len) * kv_dim * SizeOf(single));
  s.value_cache := AllocMem(p.n_layers * QWord(p.seq_len) * kv_dim * SizeOf(single));
end;

procedure TTransformer.FreeRunState(var s: TRunState);
var
  i: integer;
begin
  for i := 0 to High(s.x) do
    if Assigned(s.x[i]) then FreeMem(s.x[i]);
  for i := 0 to High(s.xb) do
    if Assigned(s.xb[i]) then FreeMem(s.xb[i]);
  for i := 0 to High(s.hb) do
    if Assigned(s.hb[i]) then FreeMem(s.hb[i]);
  for i := 0 to High(s.hb2) do
    if Assigned(s.hb2[i]) then FreeMem(s.hb2[i]);
  for i := 0 to High(s.xq) do
  begin
    if Assigned(s.xq[i].q) then FreeMem(s.xq[i].q);
    if Assigned(s.xq[i].s) then FreeMem(s.xq[i].s);
  end;
  for i := 0 to High(s.hq) do
  begin
    if Assigned(s.hq[i].q) then FreeMem(s.hq[i].q);
    if Assigned(s.hq[i].s) then FreeMem(s.hq[i].s);
  end;
  for i := 0 to High(s.q) do
    if Assigned(s.q[i]) then FreeMem(s.q[i]);
  for i := 0 to High(s.k) do
    if Assigned(s.k[i]) then FreeMem(s.k[i]);
  for i := 0 to High(s.v) do
    if Assigned(s.v[i]) then FreeMem(s.v[i]);
  for i := 0 to High(s.att) do
    if Assigned(s.att[i]) then FreeMem(s.att[i]);
  for i := 0 to High(s.logits) do
    if Assigned(s.logits[i]) then FreeMem(s.logits[i]);
  if Assigned(s.key_cache) then FreeMem(s.key_cache);
  if Assigned(s.value_cache) then FreeMem(s.value_cache);
end;

procedure TTransformer.ForwardBatchPrompt(tokens, positions: PLongInt; batch_count: integer);
var
  p: ^TConfig;
  w: ^TTransformerWeights;
  s: ^TRunState;
  l, h, i: longint;
  k_ptr: PSingle;
  loff: integer;
  kv_dim: integer;
  // For first pass
  pos: longint;
  // For second pass
{$HINTS OFF}
  procedure BatchEmbeddingCopy(Index: PtrInt; Data: Pointer; Item: TMultiThreadProcItem);
  var
    token: longint;
  begin
    token := (tokens + Index)^;
    self.DoEmbeddingCopy(s, w, p, token, Index);
  end;

  procedure BatchAttentionAndFFN(Index: PtrInt; Data: Pointer; Item: TMultiThreadProcItem);
  var
    pos: longint;
  begin
    pos := (positions + Index)^;
    self.DoLayerPass(s, w, p, l, pos, Index);
  end;

  procedure BatchFinalLayerProc(Index: PtrInt; Data: Pointer; Item: TMultiThreadProcItem);
  begin
    self.DoFinalLayer(s, w, p, Index);
  end;

begin
  p := @self.config;
  w := @self.weights;
  s := @self.state;

  // Embedding copy for each batch element (can be parallelized)
  ProcThreadPool.DoParallelNested(@BatchEmbeddingCopy, 0, batch_count - 1, nil);

  // For each layer, do two passes:
  for l := 0 to p^.n_layers - 1 do
  begin
    // First pass: compute and store K/V for all positions (no attention yet)
    for i := 0 to batch_count - 1 do
    begin
      pos := (positions + i)^;
      // RMSNorm and QKV matmuls, K/V cache update only
      self.RMSNorm(s^.xb[i], s^.x[i], w^.rms_att_weight + l * p^.dim, p^.dim);
      s^.xq[i].Quantize(s^.xb[i], p^.dim);
      s^.xq[i].MatMul(s^.q[i], w^.wq.GetTensor(l), p^.dim, p^.n_heads * p^.head_dim);
      s^.xq[i].MatMul(s^.k[i], w^.wk.GetTensor(l), p^.dim, p^.n_kv_heads * p^.head_dim);
      s^.xq[i].MatMul(s^.v[i], w^.wv.GetTensor(l), p^.dim, p^.n_kv_heads * p^.head_dim);
      // Q/K RMSNorm + RoPE
      // Q not needed for cache, but K is
      for  h := 0 to p^.n_kv_heads - 1 do
      begin
        k_ptr := s^.k[i] + h * p^.head_dim;
        self.RMSNorm(k_ptr, k_ptr, w^.k_norm_weights + l * p^.head_dim, p^.head_dim);
        self.ApplyRotaryEmbeddings(k_ptr, p^.head_dim, pos);
      end;
      // Write K/V to cache for this position
      kv_dim := p^.n_kv_heads * p^.head_dim;
      loff := l * QWord(p^.seq_len) * kv_dim;
      Move(s^.k[i]^, (s^.key_cache + loff + pos * kv_dim)^, kv_dim * SizeOf(single));
      Move(s^.v[i]^, (s^.value_cache + loff + pos * kv_dim)^, kv_dim * SizeOf(single));
    end;
    // Second pass: attention and FFN (now cache is fully populated)
    ProcThreadPool.DoParallelNested(@BatchAttentionAndFFN, 0, batch_count - 1, nil);
  end;
  // Final layer for all batch elements in parallel
  ProcThreadPool.DoParallelNested(@BatchFinalLayerProc, 0, batch_count - 1, nil);
end;

procedure TTransformer.DoEmbeddingCopy(s: PRunState; w: PTransformerWeights; p: PConfig; token, batch_idx: integer);
begin
  Move((w^.token_embedding_table + token * p^.dim)^, s^.x[batch_idx]^, p^.dim * SizeOf(single));
end;

procedure TTransformer.DoLayerPass(s: PRunState; w: PTransformerWeights; p: PConfig; l, pos, batch_idx: integer);
begin
  // Attention
  self.ProcessAttentionLayer(s^, w^, p^, l, pos, batch_idx);
  self.ApplyResidualConnection(s^.x[batch_idx], s^.xb[batch_idx], p^.dim);
  // FFN
  self.ProcessFFNLayer(s^, w^, p^, l, batch_idx);
  self.ApplyResidualConnection(s^.x[batch_idx], s^.xb[batch_idx], p^.dim);
end;

procedure TTransformer.DoFinalLayer(s: PRunState; w: PTransformerWeights; p: PConfig; batch_idx: integer);
begin
  self.ProcessFinalLayer(s^, w^, p^, batch_idx);
end;

end.
