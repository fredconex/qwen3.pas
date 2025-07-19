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
    x: PSingle;           // Current hidden state (dim)
    xb: PSingle;          // Buffer for intermediate hidden state (varies)
    hb: PSingle;          // Buffer for FFN hidden state (hidden_dim)
    hb2: PSingle;         // Buffer for FFN gate (hidden_dim)
    xq: TInt8QuantizedTensor; // Quantized buffer for hidden state (for matmuls)
    hq: TInt8QuantizedTensor; // Quantized buffer for FFN hidden state
    q: PSingle;           // Query vector (n_heads x head_dim)
    k: PSingle;           // Key vector (n_kv_heads x head_dim)
    v: PSingle;           // Value vector (n_kv_heads x head_dim)
    att: PSingle;         // Attention scores (n_heads x seq_len)
    logits: PSingle;      // Output logits (vocab_size)
    key_cache: PSingle;   // Cached keys for all layers (n_layers x seq_len x n_kv_heads x head_dim)
    value_cache: PSingle; // Cached values for all layers (n_layers x seq_len x n_kv_heads x head_dim)
  end;

  { Transformer structure }
  TTransformer = class
  private
    procedure MapWeightsToMemory(var DataPtr: Pointer);
    procedure MallocRunState(var s: TRunState; var p: TConfig);
    procedure FreeRunState(var s: TRunState);
    procedure LoadFromFile(checkpoint: string; ctx_length: longint);
    function GenerateFromTokens(var tokenizer: TTokenizer; var sampler: TSampler; prompt_tokens: PLongInt; num_prompt_tokens: longint; start_pos: longint; output_prompt: boolean): longint;

    procedure ApplyRotaryEmbeddings(const ptr: PSingle; const head_dim, pos: longint);
    procedure ProcessAttentionLayer(var s: TRunState; const w: TTransformerWeights; const p: TConfig; const l, pos: longint);
    procedure ProcessFFNLayer(var s: TRunState; const w: TTransformerWeights; const p: TConfig; const l: longint);
    procedure ApplyResidualConnection(const x, xb: PSingle; const dim: longint);
    procedure ProcessFinalLayer(var s: TRunState; const w: TTransformerWeights; const p: TConfig);
    procedure RMSNorm(const o, x, weight: PSingle; Size: longint);
    function Sigmoid(const x: single): single;
    procedure SwiGLU(var hb, hb2: PSingle; hidden_dim: longint);
  public
    config: TConfig;
    weights: TTransformerWeights;
    state: TRunState;
    Data: Pointer;
    file_size: int64;

    constructor Create(checkpoint_path: string; ctx_length: longint);
    destructor Destroy; override;
    function Forward(token: longint; pos: longint): PSingle;
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

  procedure AllocateAndDequantizeTokens;
  var
    TokenTableSize: integer;
  begin
    TokenTableSize := config.vocab_size * config.dim;

    // Initialize quantized token embeddings (single tensor)
    GetMem(weights.q_tokens, SizeOf(TInt8QuantizedTensor));
    weights.q_tokens^.q := PShortInt(BytePtr);
    Inc(BytePtr, TokenTableSize * SizeOf(shortint));
    weights.q_tokens^.s := PSingle(BytePtr);
    Inc(BytePtr, (TokenTableSize div config.group_size) * SizeOf(single));
    weights.q_tokens^.group_size := config.group_size;

    // Allocate and dequantize token embedding table
    GetMem(weights.token_embedding_table, TokenTableSize * SizeOf(single));
    weights.q_tokens^.Dequantize(weights.token_embedding_table, TokenTableSize);
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

  // Process token embeddings
  AllocateAndDequantizeTokens;

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
  Move((w^.token_embedding_table + token * p^.dim)^, s^.x^, p^.dim * SizeOf(single));

  // Forward through all layers
  for l := 0 to p^.n_layers - 1 do
  begin
    // Process attention layer
    ProcessAttentionLayer(s^, w^, p^, l, pos);

    // Apply residual connection after attention
    ApplyResidualConnection(s^.x, s^.xb, p^.dim);

    // Process feed-forward network layer
    ProcessFFNLayer(s^, w^, p^, l);

    // Apply residual connection after FFN
    ApplyResidualConnection(s^.x, s^.xb, p^.dim);
  end;

  // Process final layer
  ProcessFinalLayer(s^, w^, p^);

  Result := s^.logits;
end;

function TTransformer.GenerateFromTokens(var tokenizer: TTokenizer; var sampler: TSampler; prompt_tokens: PLongInt; num_prompt_tokens: longint; start_pos: longint; output_prompt: boolean): longint;
var
  Next, token, pos: longint;
  logits: PSingle;
  start_time, first_token_time, end_time: TDateTime;
  tokens_generated: longint;
  time_to_first_token, total_time: double;
  tokens_per_second: double;
  response_started: boolean;
begin
  // Initialize state
  pos := start_pos;
  tokens_generated := 0;
  start_time := Now;
  first_token_time := 0;
  response_started := False;

  // Determine initial token
  if pos < num_prompt_tokens then
    token := (prompt_tokens + pos)^
  else
    token := 0; // This shouldn't happen in normal usage

  while pos < self.config.seq_len do
  begin
    // Forward transformer to get logits
    logits := self.Forward(token, pos);

    // Advance state machine
    if pos < num_prompt_tokens - 1 then
      Next := (prompt_tokens + pos + 1)^
    else
      Next := sampler.Sample(logits);

    Inc(pos);

    // Handle response generation (after prompt tokens)
    if pos >= num_prompt_tokens then
    begin
      // Check for termination
      if (Next = tokenizer.bos_token_id) or (Next = tokenizer.eos_token_id) then
        Break;

      // Track first token time
      if not response_started then
      begin
        first_token_time := Now;
        response_started := True;
      end;

      // Output token and count it
      Write(tokenizer.Decode(Next));
      Flush(Output);
      Inc(tokens_generated);
    end
    else
    begin
      // Still processing prompt tokens - output them only if requested
      if output_prompt then
      begin
        Write(tokenizer.Decode(token));
        Flush(Output);
      end;
    end;

    token := Next;
  end;

  // Calculate and display statistics
  if response_started then
  begin
    end_time := Now;
    total_time := MilliSecondsBetween(start_time, end_time) / 1000.0;
    time_to_first_token := MilliSecondsBetween(start_time, first_token_time) / 1000.0;
    if tokens_generated > 0 then
      tokens_per_second := tokens_generated / ((MilliSecondsBetween(first_token_time, end_time)) / 1000.0)
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
  head_dim_half: single;
  head_dim_half_int: longint;
begin
  head_dim_half := head_dim / 2;
  head_dim_half_int := head_dim div 2;
  for j := 0 to head_dim_half_int - 1 do
  begin
    pfreq := pos * Power(1e6, -j / head_dim_half);
    cos_freq := Cos(pfreq);
    sin_freq := Sin(pfreq);
    x_val := (ptr + j)^;
    y_val := (ptr + j + head_dim_half_int)^;
    (ptr + j)^ := x_val * cos_freq - y_val * sin_freq;
    (ptr + j + head_dim_half_int)^ := x_val * sin_freq + y_val * cos_freq;
  end;
end;

procedure TTransformer.ProcessAttentionLayer(var s: TRunState; const w: TTransformerWeights; const p: TConfig; const l, pos: longint);
var
  kv_dim, kv_mul, all_heads_dim: longint;
  loff: QWord;
  i, h, t: longint;
  att_ptr, q_ptr, k_ptr, v_ptr, xb_ptr: PSingle;
begin
  kv_dim := p.n_kv_heads * p.head_dim;
  kv_mul := p.n_heads div p.n_kv_heads;
  all_heads_dim := p.n_heads * p.head_dim;
  loff := l * QWord(p.seq_len) * kv_dim;
  s.k := s.key_cache + loff + pos * kv_dim;
  s.v := s.value_cache + loff + pos * kv_dim;
  // Attention RMS norm
  self.RMSNorm(s.xb, s.x, w.rms_att_weight + l * p.dim, p.dim);
  // QKV matmuls
  s.xq.Quantize(s.xb, p.dim);
  s.xq.MatMul(s.q, w.wq.GetTensor(l), p.dim, all_heads_dim);
  s.xq.MatMul(s.k, w.wk.GetTensor(l), p.dim, kv_dim);
  s.xq.MatMul(s.v, w.wv.GetTensor(l), p.dim, kv_dim);
  // Q-RMSNorm + rotate each query head
  for h := 0 to p.n_heads - 1 do
  begin
    q_ptr := s.q + h * p.head_dim;
    self.RMSNorm(q_ptr, q_ptr, w.q_norm_weights + l * p.head_dim, p.head_dim);
    self.ApplyRotaryEmbeddings(q_ptr, p.head_dim, pos);
  end;
  // K-RMSNorm + rotate each key head
  for h := 0 to p.n_kv_heads - 1 do
  begin
    k_ptr := s.k + h * p.head_dim;
    self.RMSNorm(k_ptr, k_ptr, w.k_norm_weights + l * p.head_dim, p.head_dim);
    self.ApplyRotaryEmbeddings(k_ptr, p.head_dim, pos);
  end;
  // Multihead attention - optimized
  for h := 0 to p.n_heads - 1 do
  begin
    q_ptr := s.q + h * p.head_dim;
    att_ptr := s.att + h * p.seq_len;
    for t := 0 to pos do
    begin
      k_ptr := s.key_cache + loff + t * kv_dim + (h div kv_mul) * p.head_dim;
      (att_ptr + t)^ := DotProduct_Hybrid(q_ptr, k_ptr, p.head_dim) / Sqrt(p.head_dim);
    end;
    Softmax(att_ptr, pos + 1);
    xb_ptr := s.xb + h * p.head_dim;
    FillChar(xb_ptr^, p.head_dim * SizeOf(single), 0);
    for t := 0 to pos do
    begin
      v_ptr := s.value_cache + loff + t * kv_dim + (h div kv_mul) * p.head_dim;

      // Apply Scalar
      for i := 0 to p.head_dim - 1 do
          (xb_ptr + i)^ := (xb_ptr + i)^ + (att_ptr + t)^ * (v_ptr + i)^;
    end;
  end;
  // Final attention matmul
  s.xq.Quantize(s.xb, all_heads_dim);
  s.xq.MatMul(s.xb, w.wo.GetTensor(l), all_heads_dim, p.dim);
end;

procedure TTransformer.ProcessFFNLayer(var s: TRunState; const w: TTransformerWeights; const p: TConfig; const l: longint);
begin
  self.RMSNorm(s.xb, s.x, w.rms_ffn_weight + l * p.dim, p.dim);
  s.xq.Quantize(s.xb, p.dim);
  s.xq.MatMul(s.hb, w.w1.GetTensor(l), p.dim, p.hidden_dim);
  s.xq.MatMul(s.hb2, w.w3.GetTensor(l), p.dim, p.hidden_dim);
  self.SwiGLU(s.hb, s.hb2, p.hidden_dim);
  s.hq.Quantize(s.hb, p.hidden_dim);
  s.hq.MatMul(s.xb, w.w2.GetTensor(l), p.hidden_dim, p.dim);
end;

procedure TTransformer.ApplyResidualConnection(const x, xb: PSingle; const dim: longint);
var
  i: longint;
begin
  for i := 0 to dim - 1 do
    (x + i)^ := (x + i)^ + (xb + i)^;
end;

procedure TTransformer.ProcessFinalLayer(var s: TRunState; const w: TTransformerWeights; const p: TConfig);
begin
  self.RMSNorm(s.x, s.x, w.rms_final_weight, p.dim);
  s.xq.Quantize(s.x, p.dim);
  s.xq.MatMul(s.logits, w.wcls^, p.dim, p.vocab_size);
end;

procedure TTransformer.MallocRunState(var s: TRunState; var p: TConfig);
var
  all_heads_dim, kv_dim: longint;
begin
  all_heads_dim := p.n_heads * p.head_dim;
  kv_dim := p.n_kv_heads * p.head_dim;
  s.x := AllocMem(p.dim * SizeOf(single));
  s.xb := AllocMem(all_heads_dim * SizeOf(single));
  s.hb := AllocMem(p.hidden_dim * SizeOf(single));
  s.hb2 := AllocMem(p.hidden_dim * SizeOf(single));
  s.xq.q := AllocMem(all_heads_dim * SizeOf(shortint));
  s.xq.s := AllocMem((all_heads_dim div p.group_size) * SizeOf(single));
  s.xq.group_size := p.group_size;
  s.hq.q := AllocMem(p.hidden_dim * SizeOf(shortint));
  s.hq.s := AllocMem((p.hidden_dim div p.group_size) * SizeOf(single));
  s.hq.group_size := p.group_size;
  s.q := AllocMem(all_heads_dim * SizeOf(single));
  s.att := AllocMem(p.n_heads * p.seq_len * SizeOf(single));
  s.logits := AllocMem(p.vocab_size * SizeOf(single));
  s.key_cache := AllocMem(p.n_layers * QWord(p.seq_len) * kv_dim * SizeOf(single));
  s.value_cache := AllocMem(p.n_layers * QWord(p.seq_len) * kv_dim * SizeOf(single));
end;

procedure TTransformer.FreeRunState(var s: TRunState);
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
// --- End TTransformer helper method implementations ---

end.

