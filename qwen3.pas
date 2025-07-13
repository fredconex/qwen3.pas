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
  Windows,
  Tokenizer_Unit in 'src/tokenizer_unit.pas',
  Transformer_Unit in 'src/transformer_unit.pas';

  { Configure console for UTF-8 output }
  procedure ConfigureConsoleForUTF8;
  begin
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
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
  transformer.Build(checkpoint_path, ctx_length);

  writeln('BuildTransformer.');

  // Build tokenizer
  Tokenizer.Build(checkpoint_path, transformer.config.vocab_size, enable_thinking = 1);

  writeln('BuildTokenizer.');

  // Build sampler
  Sampler.build(transformer.config.vocab_size, temperature, topp, rng_seed);
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
    transformer.Generate(tokenizer, sampler, prompt)
  else if StrComp(mode, 'chat') = 0 then
    transformer.Chat(tokenizer, sampler, prompt, system_prompt)
  else
  begin
    WriteLn(StdErr, 'Unknown mode: ', mode);
    ErrorUsage;
  end;

  // Cleanup
  sampler.Free;
  tokenizer.Free;
  transformer.Free;
end.
