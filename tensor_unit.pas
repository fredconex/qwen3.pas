{ Tensor unit for Qwen-3 model inference in FreePascal }
unit Tensor_Unit;

{$mode objfpc}{$H+}
{$modeswitch advancedrecords}

interface

uses
  SysUtils,
  Classes;

type
  { Quantized tensor structure }
  PInt8QuantizedTensor = ^TInt8QuantizedTensor;

  TInt8QuantizedTensor = record
    q: PInt8;    // quantized values (int8)
    s: PSingle;  // scaling factors
    group_size: longint; // Added for quantization/dequantization

    procedure Dequantize(x: PSingle; n: longint);
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
procedure TInt8QuantizedTensor.Dequantize(x: PSingle; n: longint);
var
  i: longint;
begin
  for i := 0 to n - 1 do
    (x + i)^ := (self.q + i)^ * (self.s + (i div self.group_size))^;
end;

procedure TInt8QuantizedTensor.Quantize(x: PSingle; n: longint);
var
  group, i: longint;
  wmax, val, scale, quant_value: single;
  quantized: shortint;
begin
  for group := 0 to (n div self.group_size) - 1 do
  begin
    // Find max absolute value in current group
    wmax := 0;
    for i := 0 to self.group_size - 1 do
    begin
      val := Abs((x + group * self.group_size + i)^);
      if val > wmax then
        wmax := val;
    end;

    // Calculate scaling factor
    scale := wmax / 127.0;
    (self.s + group)^ := scale;

    // Quantize values
    for i := 0 to self.group_size - 1 do
    begin
      if scale > 0 then
        quant_value := (x + group * self.group_size + i)^ / scale
      else
        quant_value := 0;
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
         // This function computes the dot product of two vectors of 64 signed 8-bit integers.
         // It requires AVX2 support.
         // We use ymm4 as our 256-bit accumulator, which holds 8 x 32-bit integer sums.
         VPXOR   YMM4, YMM4, YMM4       // Zero out the accumulator register ymm4

         // Load base pointers into registers
         MOV     RAX, x_base
         MOV     RDX, w_base

         // --- Process first 16 bytes (elements 0-15) ---
         VPMOVSXBW YMM0, [RAX]      // Load 16 bytes from x and sign-extend to 16-bit words in ymm0
         VPMOVSXBW YMM1, [RDX]      // Load 16 bytes from w and sign-extend to 16-bit words in ymm1
         VPMADDWD YMM2, YMM0, YMM1  // Multiply 16-bit words, then add adjacent 32-bit products.
         VPADDD  YMM4, YMM4, YMM2    // Add the partial products to our main accumulator.

         // --- Process second 16 bytes (elements 16-31) ---
         ADD     RAX, 16
         ADD     RDX, 16
         VPMOVSXBW YMM0, [RAX]
         VPMOVSXBW YMM1, [RDX]
         VPMADDWD YMM2, YMM0, YMM1
         VPADDD  YMM4, YMM4, YMM2

         // --- Process third 16 bytes (elements 32-47) ---
         ADD     RAX, 16
         ADD     RDX, 16
         VPMOVSXBW YMM0, [RAX]
         VPMOVSXBW YMM1, [RDX]
         VPMADDWD YMM2, YMM0, YMM1
         VPADDD  YMM4, YMM4, YMM2

         // --- Process fourth 16 bytes (elements 48-63) ---
         ADD     RAX, 16
         ADD     RDX, 16
         VPMOVSXBW YMM0, [RAX]
         VPMOVSXBW YMM1, [RDX]
         VPMADDWD YMM2, YMM0, YMM1
         VPADDD  YMM4, YMM4, YMM2

         // At this point, ymm4 contains 8 partial sums (each a sum of 8 products).
         // Now we need to sum these 8 dword values together to get the final result.

         // --- Horizontal Summation of the accumulator (ymm4) ---
         // Add the upper 128 bits of ymm4 to the lower 128 bits.
         VEXTRACTI128 XMM0, YMM4, 1    // xmm0 = upper 128 bits of ymm4
         VPADDD  XMM4, XMM4, XMM0       // xmm4 = lower 128 bits + upper 128 bits.
         // Result is in the lower 128 bits (xmm4). We now have 4 sums.

         // Horizontal add the remaining 4 dwords in xmm4
         VPHADDD XMM4, XMM4, XMM4      // [d0,d1,d2,d3] -> [d0+d1, d2+d3, d0+d1, d2+d3]
         VPHADDD XMM4, XMM4, XMM4      // [s0,s1,s0,s1] -> [s0+s1, s0+s1, s0+s1, s0+s1]

         // The final sum is now in the lowest 32 bits of xmm4.
         VMOVD   EAX, XMM4               // Move the 32-bit result into eax

         // It's good practice to clear the upper state of YMM registers after use
         // to avoid performance penalties when mixing AVX and legacy SSE code.
         VZEROUPPER
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


// Updated MatMul procedure
procedure TInt8QuantizedTensor.MatMul(xout: PSingle; const w: TInt8QuantizedTensor; n, d: longint);
var
  i, j: longint;
  val: single;
  ival: longint;
  x_base, w_base: PInt8;
  x_scales, w_scales: PSingle;
  group_scale: single;
  groups: integer;
  dot_func: function(const x_base, w_base: PShortInt): longint;
begin
  groups := n div self.group_size;
  case self.group_size of
    64: dot_func := @Int8_DotProduct64_AVX2;
    128: dot_func := @Int8_DotProduct128_AVX2;
    256: dot_func := @Int8_DotProduct256_AVX2;
    else
    begin
      WriteLn(StdErr, 'Error: Unsupported group size in MatMul: ', self.group_size);
      Halt(1);
    end;

    end;
  groups := n div self.group_size;
  for i := 0 to d - 1 do
  begin
    val := 0;
    w_base := w.q + (i * n);
    w_scales := w.s + (i * groups);
    x_base := self.q;
    x_scales := self.s;

    for j := 0 to groups - 1 do
    begin
      // Select dot product function based on group size
      ival := dot_func(x_base, w_base);

      group_scale := x_scales^ * w_scales^;
      val += ival * group_scale;

      Inc(x_base, self.group_size);
      Inc(w_base, self.group_size);
      Inc(x_scales);
      Inc(w_scales);
    end;

    xout^ := val;
    Inc(xout);
  end;
end;

end.
