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

const
  PROMPT_BUFFER_SIZE = 32768;

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

implementation

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

{ Helper function to apply rotary positional embeddings }
procedure ApplyRotaryEmbeddings(const ptr: PSingle; const head_dim: longint; const pos: longint);
var
  j: longint;
  freq, cos_freq, sin_freq, x_val, y_val: single;
begin
  for j := 0 to (head_dim div 2) - 1 do
  begin
    freq := Power(1e6, -j / (head_dim / 2));
    cos_freq := Cos(pos * freq);
    sin_freq := Sin(pos * freq);

    x_val := (ptr + j)^;
    y_val := (ptr + j + head_dim div 2)^;

    (ptr + j)^ := x_val * cos_freq - y_val * sin_freq;
    (ptr + j + head_dim div 2)^ := x_val * sin_freq + y_val * cos_freq;
  end;
end;

{ Helper function to process attention layer }
procedure ProcessAttentionLayer(var s: TRunState; const w: TTransformerWeights; const p: TConfig; const l, pos: longint);
var
  kv_dim, kv_mul, all_heads_dim: longint;
  loff: QWord;
  h, t, i: longint;
  q_ptr, k_ptr, v_ptr, xb_ptr: PSingle;
  score: single;
begin
  kv_dim := p.n_kv_heads * p.head_dim;
  kv_mul := p.n_heads div p.n_kv_heads;
  all_heads_dim := p.n_heads * p.head_dim;
  loff := l * QWord(p.seq_len) * kv_dim;

  s.k := s.key_cache + loff + pos * kv_dim;
  s.v := s.value_cache + loff + pos * kv_dim;

  // Attention RMS norm
  RMSNorm(s.xb, s.x, w.rms_att_weight + l * p.dim, p.dim);

  // QKV matmuls
  s.xq.Quantize(s.xb, p.dim);
  s.xq.MatMul(s.q, w.wq.GetTensor(l), p.dim, all_heads_dim);
  s.xq.MatMul(s.k, w.wk.GetTensor(l), p.dim, kv_dim);
  s.xq.MatMul(s.v, w.wv.GetTensor(l), p.dim, kv_dim);

  // Q-RMSNorm + rotate each query head
  for h := 0 to p.n_heads - 1 do
  begin
    q_ptr := s.q + h * p.head_dim;
    RMSNorm(q_ptr, q_ptr, w.q_norm_weights + l * p.head_dim, p.head_dim);
    ApplyRotaryEmbeddings(q_ptr, p.head_dim, pos);
  end;

  // K-RMSNorm + rotate each key head
  for h := 0 to p.n_kv_heads - 1 do
  begin
    k_ptr := s.k + h * p.head_dim;
    RMSNorm(k_ptr, k_ptr, w.k_norm_weights + l * p.head_dim, p.head_dim);
    ApplyRotaryEmbeddings(k_ptr, p.head_dim, pos);
  end;

  // Multihead attention
  for h := 0 to p.n_heads - 1 do
  begin
    q_ptr := s.q + h * p.head_dim;

    for t := 0 to pos do
    begin
      k_ptr := s.key_cache + loff + t * kv_dim + (h div kv_mul) * p.head_dim;

      score := 0;
      for i := 0 to p.head_dim - 1 do
        score := score + (q_ptr + i)^ * (k_ptr + i)^;

      (s.att + h * p.seq_len + t)^ := score / Sqrt(p.head_dim);
    end;

    Softmax(s.att + h * p.seq_len, pos + 1);

    xb_ptr := s.xb + h * p.head_dim;
    FillChar(xb_ptr^, p.head_dim * SizeOf(single), 0);

    for t := 0 to pos do
    begin
      v_ptr := s.value_cache + loff + t * kv_dim + (h div kv_mul) * p.head_dim;
      for i := 0 to p.head_dim - 1 do
        (xb_ptr + i)^ := (xb_ptr + i)^ + (s.att + h * p.seq_len + t)^ * (v_ptr + i)^;
    end;
  end;

  // Final attention matmul
  s.xq.Quantize(s.xb, all_heads_dim);
  s.xq.MatMul(s.xb, w.wo.GetTensor(l), all_heads_dim, p.dim);
end;

{ Helper function to process feed-forward network layer }
procedure ProcessFFNLayer(var s: TRunState; const w: TTransformerWeights; const p: TConfig; const l: longint);
var
  i: longint;
  sigmoid_val: single;
begin
  // FFN RMS norm
  RMSNorm(s.xb, s.x, w.rms_ffn_weight + l * p.dim, p.dim);

  // FFN
  s.xq.Quantize(s.xb, p.dim);
  s.xq.MatMul(s.hb, w.w1.GetTensor(l), p.dim, p.hidden_dim);
  s.xq.MatMul(s.hb2, w.w3.GetTensor(l), p.dim, p.hidden_dim);

  // SwiGLU
  for i := 0 to p.hidden_dim - 1 do
  begin
    sigmoid_val := 1.0 / (1.0 + Exp(-(s.hb + i)^));
    (s.hb + i)^ := (s.hb + i)^ * sigmoid_val * (s.hb2 + i)^;
  end;

  // Final FFN matmul
  s.hq.Quantize(s.hb, p.hidden_dim);
  s.hq.MatMul(s.xb, w.w2.GetTensor(l), p.hidden_dim, p.dim);
end;

{ Helper function to apply residual connections }
procedure ApplyResidualConnection(const x, xb: PSingle; const dim: longint);
var
  i: longint;
begin
  for i := 0 to dim - 1 do
    (x + i)^ := (x + i)^ + (xb + i)^;
end;

{ Helper function to process final layer }
procedure ProcessFinalLayer(var s: TRunState; const w: TTransformerWeights; const p: TConfig);
begin
  // Final RMS norm
  RMSNorm(s.x, s.x, w.rms_final_weight, p.dim);

  // Classifier
  s.xq.Quantize(s.x, p.dim);
  s.xq.MatMul(s.logits, w.wcls^, p.dim, p.vocab_size);
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
  if tokens_generated > 0 then
    tokens_per_second := tokens_generated / ((MilliSecondsBetween(first_token_time, end_time)) / 1000.0)
  else
    tokens_per_second := 0;

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
        temp_str := StringReplace(tokenizer.prompt_template, '%s', PChar(@user_prompt[0]), [rfReplaceAll]);
        StrCopy(@rendered_prompt[0], PChar(temp_str));
      end;

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
          if tokens_generated > 0 then
            tokens_per_second := tokens_generated / ((MilliSecondsBetween(first_token_time, end_time)) / 1000.0)
          else
            tokens_per_second := 0;

          WriteLn('--- Response Statistics ---');
          WriteLn('Prompt tokens:', num_prompt_tokens,
            ' | Generated tokens: ', tokens_generated,
            ' | Total tokens: ', num_prompt_tokens + tokens_generated,
            ' | Time to first token: ', time_to_first_token: 0: 3, 's',
            ' | Total response time: ', total_time: 0: 3, 's',
            ' | Tokens per second: ', tokens_per_second: 0: 2, ' tk/s');
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
