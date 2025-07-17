unit Tokenizer_Unit;

{$mode objfpc}{$H+}
{$modeswitch advancedrecords}
{$HINTS OFF}
{$NOTES OFF}

interface

uses
  SysUtils,
  Math,
  Classes,
  fgl;

type
  { Hashmap for vocabulary lookup }
  TVocabMap = specialize TFPGMap<string, longint>;

  { Probability index for sampling }
  TProbIndex = record
    prob: single;
    index: longint;
  end;
  PProbIndex = ^TProbIndex;

  { Tokenizer }
  TTokenizer = class
  public
    vocab: PPChar;
    vocab_map: TVocabMap;
    merge_scores: PSingle;
    vocab_size: longint;
    max_token_length: longword;
    bos_token_id: longword;
    eos_token_id: longword;
    prompt_template: array[0..1023] of char;
    system_prompt_template: array[0..1023] of char;

    constructor Create(checkpoint_path: string; _vocab_size: longint; enable_thinking: boolean);
    destructor Destroy; override;
    procedure LoadPromptTemplate(var out_template: array of char; with_system_prompt: boolean; enable_thinking: boolean);
    function Decode(token: longint): pchar;
    procedure Encode(Text: pchar; tokens: PLongInt; var n_tokens: longint);
    function LookupToken(str: pchar): longint;
  end;

  { Sampler }
  TSampler = class
  public
    vocab_size: longint;
    probindex: PProbIndex;
    temperature: single;
    topp: single;
    rng_state: QWord;

    constructor Create(_vocab_size: longint; _temperature: single; _topp: single; rng_seed: QWord);
    destructor Destroy; override;
    function Sample(logits: PSingle): longint;
  end;

{ Tokenizer functions }

{ Sampling functions }
function RandomU32(var state: QWord): longword;
function RandomF32(var state: QWord): single;
function SampleArgmax(const probabilities: PSingle; n: longint): longint;
function SampleMult(probabilities: PSingle; n: longint; coin: single): longint;
function SampleTopP(probabilities: PSingle; n: longint; topp: single; probindex: PProbIndex; coin: single): longint;

{ QuickSort functions for top-p sampling }
function QuickSortPartition(probindex: PProbIndex; low, high: longint): longint;
procedure QuickSort(probindex: PProbIndex; low, high: longint);

{ Softmax function }
procedure Softmax(const x: PSingle; size: longint);

implementation

{ Tokenizer functions }
procedure TTokenizer.LoadPromptTemplate(var out_template: array of char; with_system_prompt: boolean; enable_thinking: boolean);
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

constructor TTokenizer.Create(checkpoint_path: string; _vocab_size: longint; enable_thinking: boolean);
var
  tokenizer_path: string;
  f: file;
  x: longint = 0;
  len: longint = 0;
  i: longint;
  token_str: string;
begin
  tokenizer_path := checkpoint_path + '.tokenizer';
  Self.vocab_size := _vocab_size;

  GetMem(Self.vocab, vocab_size * SizeOf(PChar));
  GetMem(Self.merge_scores, vocab_size * SizeOf(single));
  
  // Initialize hashmap
  Self.vocab_map := TVocabMap.Create;
  Self.vocab_map.Sorted := True;

  AssignFile(f, tokenizer_path);
  Reset(f, 1);

  BlockRead(f, Self.max_token_length, SizeOf(longint));
  BlockRead(f, Self.bos_token_id, SizeOf(longint));
  BlockRead(f, Self.eos_token_id, SizeOf(longint));

  for i := 0 to vocab_size - 1 do
  begin
    BlockRead(f, (Self.merge_scores + i)^, SizeOf(single), x);
    if x = 0 then
    begin
      GetMem((Self.vocab + i)^, 1);
      (Self.vocab + i)^[0] := #0;
    end
    else
    begin
      BlockRead(f, len, SizeOf(longint));
      GetMem((Self.vocab + i)^, len + 1);
      BlockRead(f, (Self.vocab + i)^[0], len);
      (Self.vocab + i)^[len] := #0;
    end;
    
    // Add to hashmap for fast lookup
    token_str := string((Self.vocab + i)^);
    Self.vocab_map.Add(token_str, i);
  end;

  CloseFile(f);

  Self.LoadPromptTemplate(Self.prompt_template, False, enable_thinking);
  Self.LoadPromptTemplate(Self.system_prompt_template, True, enable_thinking);
end;

destructor TTokenizer.Destroy;
var
  i: longint;
begin
  for i := 0 to Self.vocab_size - 1 do
    FreeMem(Self.vocab[i]);
  FreeMem(Self.vocab);
  FreeMem(Self.merge_scores);
  
  // Free hashmap
  if Assigned(Self.vocab_map) then
    Self.vocab_map.Free;
  inherited Destroy;
end;

function TTokenizer.Decode(token: longint): pchar;
begin
  Result := Self.vocab[token];
end;

function TTokenizer.LookupToken(str: pchar): longint;
var
  token_str: string;
  idx: longint;
begin
  token_str := string(str);
  if Self.vocab_map.Find(token_str, idx) then
    Result := Self.vocab_map.Data[idx]
  else
    Result := -1;
end;

procedure TTokenizer.Encode(Text: pchar; tokens: PLongInt; var n_tokens: longint);
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
  GetMem(str_buffer, (Self.max_token_length * 2 + 1 + 2) * SizeOf(char));

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

        id := Self.LookupToken(@special_token[0]);
        if id <> -1 then
        begin
          Inc(c, end_of_token_pos);
          found_special_token := 1;
        end;
      end;
    end;

    // Not a special token, look up single character
    if found_special_token = 0 then
      id := Self.LookupToken(str_buffer);

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
      GetMem(merged_str, (Self.max_token_length * 2 + 1) * SizeOf(char));

      // Copy first token
      StrCopy(merged_str, Self.vocab[(tokens + i)^]);
      len1 := StrLen(Self.vocab[(tokens + i)^]);
      len2 := StrLen(Self.vocab[(tokens + i + 1)^]);

      // Check if concatenation would exceed max token length
      if len1 + len2 <= Self.max_token_length then
      begin
        // Concatenate second token
        Move(Self.vocab[(tokens + i + 1)^]^, (merged_str + len1)^, len2 * SizeOf(char));
        (merged_str + len1 + len2)^ := #0;

        id := Self.LookupToken(merged_str);

        if (id <> -1) and ((Self.merge_scores + id)^ > best_score) then
        begin
          best_score := (Self.merge_scores + id)^;
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
  //writeln('Prompt Encoded.');
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
  n0, i, last_idx: longint;
  cutoff, cumulative_prob, r, cdf: single;
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
constructor TSampler.Create(_vocab_size: longint; _temperature: single; _topp: single; rng_seed: QWord);
begin
  self.vocab_size := _vocab_size;
  self.temperature := _temperature;
  self.topp := _topp;
  self.rng_state := rng_seed;
  GetMem(self.probindex, _vocab_size * SizeOf(TProbIndex));
end;

destructor TSampler.Destroy;
begin
  FreeMem(self.probindex);
  inherited Destroy;
end;

{ Sample function }
function TSampler.Sample(logits: PSingle): longint;
var
  i: longint;
  coin: single;
begin
  if self.temperature = 0 then
  begin
    // Greedy argmax sampling
    Result := SampleArgmax(logits, self.vocab_size);
  end
  else
  begin
    // Apply temperature
    for i := 0 to self.vocab_size - 1 do
      (logits + i)^ := (logits + i)^ / self.temperature;

    // Apply softmax
    Softmax(logits, self.vocab_size);

    // Sample
    coin := RandomF32(self.rng_state);

    if (self.topp <= 0) or (self.topp >= 1) then
      Result := SampleMult(logits, self.vocab_size, coin)
    else
      Result := SampleTopP(logits, self.vocab_size, self.topp, self.probindex, coin);
  end;
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

end. 
