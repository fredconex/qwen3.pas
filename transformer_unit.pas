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
  TBatchState = record
    x: PSingle;      // [dim]
    xb: PSingle;     // [all_heads_dim]
    hb: PSingle;     // [hidden_dim]
    hb2: PSingle;    // [hidden_dim]
    xq: TInt8QuantizedTensor;
    hq: TInt8QuantizedTensor;
    q: PSingle;      // [all_heads_dim]
    k: PSingle;      // [kv_dim]
    v: PSingle;      // [kv_dim]
    att: PSingle;    // [n_heads * seq_len]
    logits: PSingle; // [vocab_size]
  end;

  TRunState = record
    batch: array of TBatchState;
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
        (weights.token_embedding_table + j)^ := weights.q_tokens^.q[j] * weights.q_tokens^.s[j div weights.q_tokens^.group_size];
    end;

  begin
    TokenTableSize := config.vocab_size * config.dim;

    // Initialize quantized token embeddings (single tensor)
    New(weights.q_tokens);
    SetLength(weights.q_tokens^.q, TokenTableSize);
    Move(BytePtr^, weights.q_tokens^.q[0], TokenTableSize * SizeOf(shortint));
    Inc(BytePtr, TokenTableSize * SizeOf(shortint));
    SetLength(weights.q_tokens^.s, TokenTableSize div config.group_size);
    Move(BytePtr^, weights.q_tokens^.s[0], (TokenTableSize div config.group_size) * SizeOf(single));
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
      New(weights.wcls);
      SetLength(weights.wcls^.q, dim * vocab_size);
      Move(BytePtr^, weights.wcls^.q[0], dim * vocab_size * SizeOf(shortint));
      Inc(BytePtr, dim * vocab_size * SizeOf(shortint));
      SetLength(weights.wcls^.s, dim * vocab_size div group_size);
      Move(BytePtr^, weights.wcls^.s[0], (dim * vocab_size div group_size) * SizeOf(single));
      Inc(BytePtr, (dim * vocab_size div group_size) * SizeOf(single));
      weights.wcls^.group_size := group_size;
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
  Dispose(self.weights.q_tokens);
  FreeMem(self.weights.token_embedding_table);
  // Arrays are automatically freed by Pascal
  if self.weights.wcls <> self.weights.q_tokens then
    Dispose(self.weights.wcls);
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

  Result := s^.batch[0].logits;
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
      batch_count := Min(Length(self.state.batch), num_prompt_tokens - pos);
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
  self.RMSNorm(s.batch[batch_idx].xb, s.batch[batch_idx].x, w.rms_att_weight + l * p.dim, p.dim);
  // QKV matmuls
  s.batch[batch_idx].xq.Quantize(s.batch[batch_idx].xb, p.dim, all_heads_dim);
  s.batch[batch_idx].xq.MatMul(s.batch[batch_idx].q, w.wq.GetTensor(l), p.dim, all_heads_dim);
  s.batch[batch_idx].xq.MatMul(s.batch[batch_idx].k, w.wk.GetTensor(l), p.dim, kv_dim);
  s.batch[batch_idx].xq.MatMul(s.batch[batch_idx].v, w.wv.GetTensor(l), p.dim, kv_dim);
  // Q-RMSNorm + rotate each query head
  for h := 0 to p.n_heads - 1 do
  begin
    q_ptr := s.batch[batch_idx].q + h * p.head_dim;
    self.RMSNorm(q_ptr, q_ptr, w.q_norm_weights + l * p.head_dim, p.head_dim);
    self.ApplyRotaryEmbeddings(q_ptr, p.head_dim, pos);
  end;
  // K-RMSNorm + rotate each key head
  for h := 0 to p.n_kv_heads - 1 do
  begin
    k_ptr := s.batch[batch_idx].k + h * p.head_dim;
    self.RMSNorm(k_ptr, k_ptr, w.k_norm_weights + l * p.head_dim, p.head_dim);
    self.ApplyRotaryEmbeddings(k_ptr, p.head_dim, pos);
  end;
  // NOW write K/V to cache for this position (after K RMSNorm+RoPE)
  Move(s.batch[batch_idx].k^, (s.key_cache + loff + pos * kv_dim)^, kv_dim * SizeOf(single));
  Move(s.batch[batch_idx].v^, (s.value_cache + loff + pos * kv_dim)^, kv_dim * SizeOf(single));
  // Multihead attention - optimized
  for h := 0 to p.n_heads - 1 do
  begin
    q_ptr := s.batch[batch_idx].q + h * p.head_dim;
    att_ptr := s.batch[batch_idx].att + h * p.seq_len;
    for t := 0 to pos do
    begin
      k_ptr := s.key_cache + loff + t * kv_dim + (h div kv_mul) * p.head_dim;
      (att_ptr + t)^ := DotProduct_Hybrid(q_ptr, k_ptr, p.head_dim) * scale;
    end;
    Softmax(att_ptr, pos + 1);
    xb_ptr := s.batch[batch_idx].xb + h * p.head_dim;
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
  s.batch[batch_idx].xq.Quantize(s.batch[batch_idx].xb, all_heads_dim, all_heads_dim);
  s.batch[batch_idx].xq.MatMul(s.batch[batch_idx].xb, w.wo.GetTensor(l), all_heads_dim, p.dim);
end;

procedure TTransformer.ProcessFFNLayer(var s: TRunState; const w: TTransformerWeights; const p: TConfig; const l, batch_idx: longint);
begin
  self.RMSNorm(s.batch[batch_idx].xb, s.batch[batch_idx].x, w.rms_ffn_weight + l * p.dim, p.dim);
  s.batch[batch_idx].xq.Quantize(s.batch[batch_idx].xb, p.dim, p.n_heads * p.head_dim);
  s.batch[batch_idx].xq.MatMul(s.batch[batch_idx].hb, w.w1.GetTensor(l), p.dim, p.hidden_dim);
  s.batch[batch_idx].xq.MatMul(s.batch[batch_idx].hb2, w.w3.GetTensor(l), p.dim, p.hidden_dim);
  self.SwiGLU(s.batch[batch_idx].hb, s.batch[batch_idx].hb2, p.hidden_dim);
  s.batch[batch_idx].hq.Quantize(s.batch[batch_idx].hb, p.hidden_dim, p.hidden_dim);
  s.batch[batch_idx].hq.MatMul(s.batch[batch_idx].xb, w.w2.GetTensor(l), p.hidden_dim, p.dim);
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
  self.RMSNorm(s.batch[batch_idx].x, s.batch[batch_idx].x, w.rms_final_weight, p.dim);
  s.batch[batch_idx].xq.Quantize(s.batch[batch_idx].x, p.dim, p.n_heads * p.head_dim);
  s.batch[batch_idx].xq.MatMul(s.batch[batch_idx].logits, w.wcls^, p.dim, p.vocab_size);
end;

procedure TTransformer.MallocRunState(var s: TRunState; var p: TConfig);
var
  all_heads_dim, kv_dim, i: longint;
begin
  // Batch size for prompt evaluation
  all_heads_dim := p.n_heads * p.head_dim;
  kv_dim := p.n_kv_heads * p.head_dim;
  SetLength(s.batch, 512);
  for i := 0 to Length(s.batch) - 1 do
  begin
    s.batch[i].x := AllocMem(p.dim * SizeOf(single));
    s.batch[i].xb := AllocMem(all_heads_dim * SizeOf(single));
    s.batch[i].hb := AllocMem(p.hidden_dim * SizeOf(single));
    s.batch[i].hb2 := AllocMem(p.hidden_dim * SizeOf(single));
    SetLength(s.batch[i].xq.q, all_heads_dim);
    SetLength(s.batch[i].xq.s, all_heads_dim div p.group_size);
    s.batch[i].xq.group_size := p.group_size;
    SetLength(s.batch[i].hq.q, p.hidden_dim);
    SetLength(s.batch[i].hq.s, p.hidden_dim div p.group_size);
    s.batch[i].hq.group_size := p.group_size;
    s.batch[i].q := AllocMem(all_heads_dim * SizeOf(single));
    s.batch[i].k := AllocMem(kv_dim * SizeOf(single));
    s.batch[i].v := AllocMem(kv_dim * SizeOf(single));
    s.batch[i].att := AllocMem(p.n_heads * p.seq_len * SizeOf(single));
    s.batch[i].logits := AllocMem(p.vocab_size * SizeOf(single));
  end;
  s.key_cache := AllocMem(p.n_layers * QWord(p.seq_len) * kv_dim * SizeOf(single));
  s.value_cache := AllocMem(p.n_layers * QWord(p.seq_len) * kv_dim * SizeOf(single));
end;

procedure TTransformer.FreeRunState(var s: TRunState);
var
  i: integer;
begin
  for i := 0 to High(s.batch) do
  begin
    if Assigned(s.batch[i].x) then FreeMem(s.batch[i].x);
    if Assigned(s.batch[i].xb) then FreeMem(s.batch[i].xb);
    if Assigned(s.batch[i].hb) then FreeMem(s.batch[i].hb);
    if Assigned(s.batch[i].hb2) then FreeMem(s.batch[i].hb2);
    // Arrays are automatically freed by Pascal - no need to free s arrays
    // Arrays are automatically freed by Pascal - no need to free s arrays
    if Assigned(s.batch[i].q) then FreeMem(s.batch[i].q);
    if Assigned(s.batch[i].k) then FreeMem(s.batch[i].k);
    if Assigned(s.batch[i].v) then FreeMem(s.batch[i].v);
    if Assigned(s.batch[i].att) then FreeMem(s.batch[i].att);
    if Assigned(s.batch[i].logits) then FreeMem(s.batch[i].logits);
  end;
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
      self.RMSNorm(s^.batch[i].xb, s^.batch[i].x, w^.rms_att_weight + l * p^.dim, p^.dim);
      s^.batch[i].xq.Quantize(s^.batch[i].xb, p^.dim, p^.n_heads * p^.head_dim);
      s^.batch[i].xq.MatMul(s^.batch[i].q, w^.wq.GetTensor(l), p^.dim, p^.n_heads * p^.head_dim);
      s^.batch[i].xq.MatMul(s^.batch[i].k, w^.wk.GetTensor(l), p^.dim, p^.n_kv_heads * p^.head_dim);
      s^.batch[i].xq.MatMul(s^.batch[i].v, w^.wv.GetTensor(l), p^.dim, p^.n_kv_heads * p^.head_dim);
      // Q/K RMSNorm + RoPE
      // Q not needed for cache, but K is
      for  h := 0 to p^.n_kv_heads - 1 do
      begin
        k_ptr := s^.batch[i].k + h * p^.head_dim;
        self.RMSNorm(k_ptr, k_ptr, w^.k_norm_weights + l * p^.head_dim, p^.head_dim);
        self.ApplyRotaryEmbeddings(k_ptr, p^.head_dim, pos);
      end;
      // Write K/V to cache for this position
      kv_dim := p^.n_kv_heads * p^.head_dim;
      loff := l * QWord(p^.seq_len) * kv_dim;
      Move(s^.batch[i].k^, (s^.key_cache + loff + pos * kv_dim)^, kv_dim * SizeOf(single));
      Move(s^.batch[i].v^, (s^.value_cache + loff + pos * kv_dim)^, kv_dim * SizeOf(single));
    end;
    // Second pass: attention and FFN (now cache is fully populated)
    ProcThreadPool.DoParallelNested(@BatchAttentionAndFFN, 0, batch_count - 1, nil);
  end;
  // Final layer for all batch elements in parallel
  ProcThreadPool.DoParallelNested(@BatchFinalLayerProc, 0, batch_count - 1, nil);
end;

procedure TTransformer.DoEmbeddingCopy(s: PRunState; w: PTransformerWeights; p: PConfig; token, batch_idx: integer);
begin
  Move((w^.token_embedding_table + token * p^.dim)^, s^.batch[batch_idx].x^, p^.dim * SizeOf(single));
end;

procedure TTransformer.DoLayerPass(s: PRunState; w: PTransformerWeights; p: PConfig; l, pos, batch_idx: integer);
begin
  // Attention
  self.ProcessAttentionLayer(s^, w^, p^, l, pos, batch_idx);
  self.ApplyResidualConnection(s^.batch[batch_idx].x, s^.batch[batch_idx].xb, p^.dim);
  // FFN
  self.ProcessFFNLayer(s^, w^, p^, l, batch_idx);
  self.ApplyResidualConnection(s^.batch[batch_idx].x, s^.batch[batch_idx].xb, p^.dim);
end;

procedure TTransformer.DoFinalLayer(s: PRunState; w: PTransformerWeights; p: PConfig; batch_idx: integer);
begin
  self.ProcessFinalLayer(s^, w^, p^, batch_idx);
end;

end.
