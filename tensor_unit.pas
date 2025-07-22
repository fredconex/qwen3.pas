{ Tensor unit for Qwen-3 model inference in FreePascal }
unit Tensor_Unit;

{$mode objfpc}{$H+}
{$modeswitch advancedrecords}

interface

uses
  SysUtils,
  Math,
  Classes;

type
  { Quantized tensor structure }
  PInt8QuantizedTensor = ^TInt8QuantizedTensor;

  TInt8QuantizedTensor = record
    q: PInt8;    // quantized values (int8)
    s: PSingle;  // scaling factors
    group_size: longint; // Added for quantization/dequantization

   // procedure Dequantize(x: PSingle; n: longint);
    procedure Quantize(x: PSingle; n: longint);
    procedure MatMul(xout: PSingle; const w: TInt8QuantizedTensor; n, d: longint);
  end;

  { Array of quantized tensors with utility methods }
  TInt8QuantizedTensorArray = record
    Data: array of TInt8QuantizedTensor;

    procedure Initialize(Count: integer; var DataPtr: Pointer; ElementsPerTensor: integer; GroupSize: longint);
    function GetTensor(Index: integer): TInt8QuantizedTensor;
    function GetTensorPtr(Index: integer): PInt8QuantizedTensor;
    function Count: integer;
    function IsValidIndex(Index: integer): boolean;
    procedure Validate; // Validates that all tensors have valid pointers
  end;

{ Matrix multiplication for quantized tensors }
function Int8_DotProduct64_AVX2(const x_base, w_base: PShortInt): longint; assembler;
function DotProduct(a, b: PSingle; dim: integer): single;
function DotProduct_Hybrid(a, b: PSingle; dim: integer): single;
function Int8_DotProduct256_AVX2(const x_base, w_base: PShortInt): longint;

implementation

{ TQuantizedTensor method implementations }
{procedure TInt8QuantizedTensor.Dequantize(x: PSingle; n: longint);
var
  i: longint;
begin
  for i := 0 to n - 1 do
    (x + i)^ := (self.q + i)^ * (self.s + (i div self.group_size))^;
end; }

procedure TInt8QuantizedTensor.Quantize(x: PSingle; n: longint);
var
  group, i: longint;
  wmax, scale, quant_value: single;
  quantized: shortint;
  current_ptr: PSingle;
begin
  for group := 0 to (n div self.group_size) - 1 do
  begin
    current_ptr := x + group * self.group_size;

    // Find max absolute value
    wmax := 0;
    for i := 0 to self.group_size - 1 do
      wmax := Max(wmax, Abs((current_ptr + i)^));

    // Calculate scaling factor
    scale := wmax / 127.0;
    (self.s + group)^ := scale;

    // Quantize values
    for i := 0 to self.group_size - 1 do
    begin
      quant_value := (current_ptr + i)^ / scale;
      quantized := Round(quant_value);
      (self.q + group * self.group_size + i)^ := quantized;
    end;
  end;
end;

{ TQuantizedTensorArray method implementations }
procedure TInt8QuantizedTensorArray.Initialize(Count: integer; var DataPtr: Pointer; ElementsPerTensor: integer; GroupSize: longint);
var
  CurrentPtr: pbyte;
  i: integer;
  QuantizedDataSize: integer;
  ScaleDataSize: integer;
begin
  SetLength(Data, Count);
  CurrentPtr := pbyte(DataPtr);
  QuantizedDataSize := ElementsPerTensor * SizeOf(shortint);
  ScaleDataSize := (ElementsPerTensor div GroupSize) * SizeOf(single);

  // Initialize each tensor in the array
  for i := 0 to Count - 1 do
  begin
    // Set quantized data pointer for tensor i
    Data[i].q := PShortInt(CurrentPtr);
    Inc(CurrentPtr, QuantizedDataSize);

    // Set scale data pointer for tensor i
    Data[i].s := PSingle(CurrentPtr);
    Inc(CurrentPtr, ScaleDataSize);
    Data[i].group_size := GroupSize; // Set group size for each tensor
  end;

  // Update the input pointer to point past all processed data
  DataPtr := CurrentPtr;

  Validate;
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

{ Matrix multiplication }
{$ASMMODE INTEL}
function Float_DotProduct32_AVX2(const x_base, w_base: PSingle): single; assembler;
asm
         // Load base pointers into general-purpose registers to use as iterators.
         MOV     RAX, x_base
         MOV     RDX, w_base

         // --- Unrolled Loop ---

         // Block 1 (offset 0)
         VMOVUPS YMM0, YMMWORD PTR [RAX]
         VMULPS  YMM0, YMM0, YMMWORD PTR [RDX]

         // Block 2 (offset 32)
         // Manually advance the pointers instead of using an offset in the instruction.
         ADD     RAX, 32
         ADD     RDX, 32
         VMOVUPS YMM1, YMMWORD PTR [RAX]
         VMULPS  YMM1, YMM1, YMMWORD PTR [RDX]

         // Block 3 (offset 64)
         ADD     RAX, 32
         ADD     RDX, 32
         VMOVUPS YMM2, YMMWORD PTR [RAX]
         VMULPS  YMM2, YMM2, YMMWORD PTR [RDX]

         // Block 4 (offset 96)
         ADD     RAX, 32
         ADD     RDX, 32
         VMOVUPS YMM3, YMMWORD PTR [RAX]
         VMULPS  YMM3, YMM3, YMMWORD PTR [RDX]

         // --- Summation Tree ---
         VADDPS  YMM0, YMM0, YMM1
         VADDPS  YMM2, YMM2, YMM3
         VADDPS  YMM0, YMM0, YMM2

         // --- Horizontal Sum ---
         VEXTRACTF128 XMM1, YMM0, 1
         VADDPS  XMM0, XMM0, XMM1
         VHADDPS XMM0, XMM0, XMM0
         VHADDPS XMM0, XMM0, XMM0

         VZEROUPPER
end;

// Standard dot product implementation
function DotProduct(a, b: PSingle; dim: integer): single;
var
  i: integer;
begin
  Result := 0;
  for i := 0 to dim - 1 do
    Result := Result + (a + i)^ * (b + i)^;
end;

function DotProduct_Hybrid(a, b: PSingle; dim: integer): single;
const
  BLOCK = 32;                       // elements processed per AVX2 call
var
  blocks, tail: integer;
begin
  Result := 0;

  for blocks := 0 to (dim div BLOCK) - 1 do
  begin
    Result := Result + Float_DotProduct32_AVX2(a, b);
    Inc(a, BLOCK);
    Inc(b, BLOCK);
  end;

  // scalar clean-up for the last <32 elements
  for tail := 0 to (dim mod BLOCK) - 1 do
  begin
    Result := Result + a^ * b^;
    Inc(a);
    Inc(b);
  end;
end;

{$ASMMODE INTEL}
function Int8_DotProduct64_AVX2(const x_base, w_base: PShortInt): longint; assembler;
asm
         // Use two accumulators to improve instruction-level parallelism
         VPXOR   YMM4, YMM4, YMM4       // First accumulator
         VPXOR   YMM5, YMM5, YMM5       // Second accumulator

         MOV     RAX, x_base
         MOV     RDX, w_base

         // Process in pairs for better pipeline utilization
         // Block 1 & 2 (0-31)
         VPMOVSXBW YMM0, [RAX]
         VPMOVSXBW YMM1, [RDX]
         VPMOVSXBW YMM2, [RAX + 16]
         VPMOVSXBW YMM3, [RDX + 16]

         VPMADDWD YMM0, YMM0, YMM1
         VPMADDWD YMM2, YMM2, YMM3
         VPADDD  YMM4, YMM4, YMM0
         VPADDD  YMM5, YMM5, YMM2

         // Block 3 & 4 (32-63)
         ADD     RAX, 32
         ADD     RDX, 32
         VPMOVSXBW YMM0, [RAX]
         VPMOVSXBW YMM1, [RDX]
         VPMOVSXBW YMM2, [RAX + 16]
         VPMOVSXBW YMM3, [RDX + 16]

         VPMADDWD YMM0, YMM0, YMM1
         VPMADDWD YMM2, YMM2, YMM3
         VPADDD  YMM4, YMM4, YMM0
         VPADDD  YMM5, YMM5, YMM2

         // Combine accumulators
         VPADDD  YMM4, YMM4, YMM5

         // Optimized horizontal sum
         VEXTRACTI128 XMM0, YMM4, 1
         VPADDD  XMM4, XMM4, XMM0
         VPHADDD XMM4, XMM4, XMM4
         VPHADDD XMM4, XMM4, XMM4
         VMOVD   EAX, XMM4
         VZEROUPPER
end;

{$ASMMODE INTEL}
function Int8_DotProduct64_AVX2_Aligned(const x_base, w_base: PShortInt): longint; assembler;
asm
        // Zero accumulators using efficient VPXOR
        VPXOR   YMM4, YMM4, YMM4            // YMM4 = accumulator low  (blocks 0-31)
        VPXOR   YMM5, YMM5, YMM5            // YMM5 = accumulator high (blocks 32-63)

        MOV     RAX, x_base
        MOV     RDX, w_base

        // ┌─────────────────────────────────────────────────────┐
        // │ Process first 32 int8s (offset 0–31)                │
        // └─────────────────────────────────────────────────────┘
        VPMOVSXBW YMM0, [RAX +  0]           // Load 32 int8 → 32 int16 (sign-extended)
        VPMOVSXBW YMM1, [RDX +  0]
        VPMOVSXBW YMM2, [RAX + 16]           // Second half of first 32 elements
        VPMOVSXBW YMM3, [RDX + 16]

        VPMADDWD YMM0, YMM0, YMM1            // Multiply-add: 16 signed word pairs → 8 dwords
        VPMADDWD YMM2, YMM2, YMM3
        VPADDD   YMM4, YMM4, YMM0            // Accumulate into YMM4
        VPADDD   YMM5, YMM5, YMM2            // Keep separate for ILP

        // ┌─────────────────────────────────────────────────────┐
        // │ Process next 32 int8s (offset 32–63)                │
        // └─────────────────────────────────────────────────────┘
        VPMOVSXBW YMM0, [RAX + 32]           // Second block: 32–63
        VPMOVSXBW YMM1, [RDX + 32]
        VPMOVSXBW YMM2, [RAX + 48]
        VPMOVSXBW YMM3, [RDX + 48]

        VPMADDWD YMM0, YMM0, YMM1
        VPMADDWD YMM2, YMM2, YMM3
        VPADDD   YMM4, YMM4, YMM0            // Add to lower accumulator
        VPADDD   YMM5, YMM5, YMM2            // Finish upper

        // ┌─────────────────────────────────────────────────────┐
        // │ Combine final results                               │
        // └─────────────────────────────────────────────────────┘
        VPADDD  YMM4, YMM4, YMM5              // Merge both accumulators → YMM4[8 dwords]

        // Horizontal sum: reduce 8 dwords to scalar
        VEXTRACTI128 XMM0, YMM4, 1             // Get upper 128 bits
        VPADDD  XMM4, XMM4, XMM0               // Sum all 8 → 4 dwords in XMM4
        VPHADDD XMM4, XMM4, XMM4               // → [a+b, c+d, e+f, g+h] → [a+b+c+d, e+f+g+h, ...]
        VPHADDD XMM4, XMM4, XMM4               // → [total, ?, ?, ?]

        VMOVD   EAX, XMM4                      // Extract lowest dword = total sum

        VZEROUPPER                             // Avoid AVX/SSE transition penalty
end;

function Int8_DotProduct128_AVX2(const x_base, w_base: PShortInt): longint; assembler;
asm
         // Computes the dot product of two vectors of 128 signed 8-bit integers using AVX2.
         VPXOR   YMM4, YMM4, YMM4       // Zero out the accumulator register ymm4

         MOV     RAX, x_base
         MOV     RDX, w_base

         // Process 8 blocks of 16 bytes (128 elements)
         // Block 1 (0-15)
         VPMOVSXBW YMM0, [RAX]
         VPMOVSXBW YMM1, [RDX]
         VPMADDWD YMM2, YMM0, YMM1
         VPADDD  YMM4, YMM4, YMM2
         // Block 2 (16-31)
         ADD     RAX, 16
         ADD     RDX, 16
         VPMOVSXBW YMM0, [RAX]
         VPMOVSXBW YMM1, [RDX]
         VPMADDWD YMM2, YMM0, YMM1
         VPADDD  YMM4, YMM4, YMM2
         // Block 3 (32-47)
         ADD     RAX, 16
         ADD     RDX, 16
         VPMOVSXBW YMM0, [RAX]
         VPMOVSXBW YMM1, [RDX]
         VPMADDWD YMM2, YMM0, YMM1
         VPADDD  YMM4, YMM4, YMM2
         // Block 4 (48-63)
         ADD     RAX, 16
         ADD     RDX, 16
         VPMOVSXBW YMM0, [RAX]
         VPMOVSXBW YMM1, [RDX]
         VPMADDWD YMM2, YMM0, YMM1
         VPADDD  YMM4, YMM4, YMM2
         // Block 5 (64-79)
         ADD     RAX, 16
         ADD     RDX, 16
         VPMOVSXBW YMM0, [RAX]
         VPMOVSXBW YMM1, [RDX]
         VPMADDWD YMM2, YMM0, YMM1
         VPADDD  YMM4, YMM4, YMM2
         // Block 6 (80-95)
         ADD     RAX, 16
         ADD     RDX, 16
         VPMOVSXBW YMM0, [RAX]
         VPMOVSXBW YMM1, [RDX]
         VPMADDWD YMM2, YMM0, YMM1
         VPADDD  YMM4, YMM4, YMM2
         // Block 7 (96-111)
         ADD     RAX, 16
         ADD     RDX, 16
         VPMOVSXBW YMM0, [RAX]
         VPMOVSXBW YMM1, [RDX]
         VPMADDWD YMM2, YMM0, YMM1
         VPADDD  YMM4, YMM4, YMM2
         // Block 8 (112-127)
         ADD     RAX, 16
         ADD     RDX, 16
         VPMOVSXBW YMM0, [RAX]
         VPMOVSXBW YMM1, [RDX]
         VPMADDWD YMM2, YMM0, YMM1
         VPADDD  YMM4, YMM4, YMM2

         // Horizontal sum as before
         VEXTRACTI128 XMM0, YMM4, 1
         VPADDD  XMM4, XMM4, XMM0
         VPHADDD XMM4, XMM4, XMM4
         VPHADDD XMM4, XMM4, XMM4
         VMOVD   EAX, XMM4
         VZEROUPPER
end;

function Int8_DotProduct256_AVX2(const x_base, w_base: PShortInt): longint; assembler;
asm
         // Computes the dot product of two vectors of 128 signed 8-bit integers using AVX2.
         VPXOR   YMM4, YMM4, YMM4       // Zero out the accumulator register ymm4

         MOV     RAX, x_base
         MOV     RDX, w_base

         // Process 8 blocks of 16 bytes (128 elements)
         // Block 1 (0-15)
         VPMOVSXBW YMM0, [RAX]
         VPMOVSXBW YMM1, [RDX]
         VPMADDWD YMM2, YMM0, YMM1
         VPADDD  YMM4, YMM4, YMM2
         // Block 2 (16-31)
         ADD     RAX, 16
         ADD     RDX, 16
         VPMOVSXBW YMM0, [RAX]
         VPMOVSXBW YMM1, [RDX]
         VPMADDWD YMM2, YMM0, YMM1
         VPADDD  YMM4, YMM4, YMM2
         // Block 3 (32-47)
         ADD     RAX, 16
         ADD     RDX, 16
         VPMOVSXBW YMM0, [RAX]
         VPMOVSXBW YMM1, [RDX]
         VPMADDWD YMM2, YMM0, YMM1
         VPADDD  YMM4, YMM4, YMM2
         // Block 4 (48-63)
         ADD     RAX, 16
         ADD     RDX, 16
         VPMOVSXBW YMM0, [RAX]
         VPMOVSXBW YMM1, [RDX]
         VPMADDWD YMM2, YMM0, YMM1
         VPADDD  YMM4, YMM4, YMM2
         // Block 5 (64-79)
         ADD     RAX, 16
         ADD     RDX, 16
         VPMOVSXBW YMM0, [RAX]
         VPMOVSXBW YMM1, [RDX]
         VPMADDWD YMM2, YMM0, YMM1
         VPADDD  YMM4, YMM4, YMM2
         // Block 6 (80-95)
         ADD     RAX, 16
         ADD     RDX, 16
         VPMOVSXBW YMM0, [RAX]
         VPMOVSXBW YMM1, [RDX]
         VPMADDWD YMM2, YMM0, YMM1
         VPADDD  YMM4, YMM4, YMM2
         // Block 7 (96-111)
         ADD     RAX, 16
         ADD     RDX, 16
         VPMOVSXBW YMM0, [RAX]
         VPMOVSXBW YMM1, [RDX]
         VPMADDWD YMM2, YMM0, YMM1
         VPADDD  YMM4, YMM4, YMM2
         // Block 8 (112-127)
         ADD     RAX, 16
         ADD     RDX, 16
         VPMOVSXBW YMM0, [RAX]
         VPMOVSXBW YMM1, [RDX]
         VPMADDWD YMM2, YMM0, YMM1
         VPADDD  YMM4, YMM4, YMM2

         // Block 1 (0-15)
         ADD     RAX, 16
         ADD     RDX, 16
         VPMOVSXBW YMM0, [RAX]
         VPMOVSXBW YMM1, [RDX]
         VPMADDWD YMM2, YMM0, YMM1
         VPADDD  YMM4, YMM4, YMM2
         // Block 2 (16-31)
         ADD     RAX, 16
         ADD     RDX, 16
         VPMOVSXBW YMM0, [RAX]
         VPMOVSXBW YMM1, [RDX]
         VPMADDWD YMM2, YMM0, YMM1
         VPADDD  YMM4, YMM4, YMM2
         // Block 3 (32-47)
         ADD     RAX, 16
         ADD     RDX, 16
         VPMOVSXBW YMM0, [RAX]
         VPMOVSXBW YMM1, [RDX]
         VPMADDWD YMM2, YMM0, YMM1
         VPADDD  YMM4, YMM4, YMM2
         // Block 4 (48-63)
         ADD     RAX, 16
         ADD     RDX, 16
         VPMOVSXBW YMM0, [RAX]
         VPMOVSXBW YMM1, [RDX]
         VPMADDWD YMM2, YMM0, YMM1
         VPADDD  YMM4, YMM4, YMM2
         // Block 5 (64-79)
         ADD     RAX, 16
         ADD     RDX, 16
         VPMOVSXBW YMM0, [RAX]
         VPMOVSXBW YMM1, [RDX]
         VPMADDWD YMM2, YMM0, YMM1
         VPADDD  YMM4, YMM4, YMM2
         // Block 6 (80-95)
         ADD     RAX, 16
         ADD     RDX, 16
         VPMOVSXBW YMM0, [RAX]
         VPMOVSXBW YMM1, [RDX]
         VPMADDWD YMM2, YMM0, YMM1
         VPADDD  YMM4, YMM4, YMM2
         // Block 7 (96-111)
         ADD     RAX, 16
         ADD     RDX, 16
         VPMOVSXBW YMM0, [RAX]
         VPMOVSXBW YMM1, [RDX]
         VPMADDWD YMM2, YMM0, YMM1
         VPADDD  YMM4, YMM4, YMM2
         // Block 8 (112-127)
         ADD     RAX, 16
         ADD     RDX, 16
         VPMOVSXBW YMM0, [RAX]
         VPMOVSXBW YMM1, [RDX]
         VPMADDWD YMM2, YMM0, YMM1
         VPADDD  YMM4, YMM4, YMM2

         // Horizontal sum as before
         VEXTRACTI128 XMM0, YMM4, 1
         VPADDD  XMM4, XMM4, XMM0
         VPHADDD XMM4, XMM4, XMM4
         VPHADDD XMM4, XMM4, XMM4
         VMOVD   EAX, XMM4
         VZEROUPPER
end;


procedure TInt8QuantizedTensor.MatMul(xout: PSingle; const w: TInt8QuantizedTensor; n, d: longint);
var
  i, j: longint;
  val: single;
  x_base, w_base: PInt8;
  x_scales, w_scales: PSingle;
  groups: integer;
  dot_func: function(const x_base, w_base: PShortInt): longint;
  x_qdata: PInt8;
  x_sdata: PSingle;
  w_qdata: PInt8;
  w_sdata: PSingle;
begin
  // Cache object fields locally
  x_qdata := self.q;
  x_sdata := self.s;
  w_qdata := w.q;
  w_sdata := w.s;

  groups := n div self.group_size;

  // Dispatch dot function
  case self.group_size of
    64:  dot_func := @Int8_DotProduct64_AVX2_Aligned;
    128: dot_func := @Int8_DotProduct128_AVX2;
    256: dot_func := @Int8_DotProduct256_AVX2;
  else
    WriteLn(StdErr, 'Error: Unsupported group size in MatMul: ', self.group_size);
    Halt(1);
  end;

  // Ensure inputs are compatible
  assert((n mod self.group_size) = 0);

  // Main matrix multiplication: d outputs
  for i := 0 to d - 1 do
  begin
    val := 0.0;

    // Set up weight pointers
    w_base := @w_qdata[i * n];
    w_scales := @w_sdata[i * groups];

    // Set up input quantized data and scales
    x_base := x_qdata;
    x_scales := x_sdata;

    // Fused loop: compute each group's dot product and apply scale immediately
    for j := 0 to groups - 1 do
    begin
      // Compute dot product
      val += dot_func(x_base, w_base) * (x_scales^ * w_scales^);

      // Advance pointers
      Inc(x_base, self.group_size);
      Inc(w_base, self.group_size);
      Inc(x_scales);
      Inc(w_scales);
    end;

    // Store result
    xout^ := val;
    Inc(xout);
  end;
end;

end.
