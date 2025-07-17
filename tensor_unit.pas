{ Tensor unit for Qwen-3 model inference in FreePascal }
unit Tensor_Unit;

{$mode objfpc}{$H+}
{$modeswitch advancedrecords}

interface

uses
  SysUtils,
  Classes;

var
  GS: longint = 0; // Global group size for quantization

type
  { Quantized tensor structure }
  PInt8QuantizedTensor = ^TInt8QuantizedTensor;

  TInt8QuantizedTensor = record
    q: PInt8;    // quantized values (int8)
    s: PSingle;  // scaling factors

    procedure Dequantize(x: PSingle; n: longint);
    procedure Quantize(x: PSingle; n: longint);
    procedure MatMul(xout: PSingle; const w: TInt8QuantizedTensor; n, d: longint);
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

{ Matrix multiplication for quantized tensors }
function Int8_DotProduct64_AVX2(const x_base, w_base: PShortInt): LongInt; assembler;
function Int8_DotProduct(const x_base, w_base: PShortInt): LongInt;

function DotProduct(a, b: PSingle; dim: Integer): Single;
function DotProduct_Hybrid(a, b: PSingle; dim: Integer): Single;

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

{ Matrix multiplication }
function Int8_DotProduct(const x_base, w_base: PShortInt): LongInt;
var
  i: longint;
  x_ptr, w_ptr: PShortInt;
begin
  Result := 0;
  x_ptr := x_base;
  w_ptr := w_base;

  // Process 8 elements at a time (matching the unrolled loop in MatMul)
  for i := 0 to (GS div 8) - 1 do
  begin
    Result += x_ptr[0] * w_ptr[0];
    Result += x_ptr[1] * w_ptr[1];
    Result += x_ptr[2] * w_ptr[2];
    Result += x_ptr[3] * w_ptr[3];
    Result += x_ptr[4] * w_ptr[4];
    Result += x_ptr[5] * w_ptr[5];
    Result += x_ptr[6] * w_ptr[6];
    Result += x_ptr[7] * w_ptr[7];
    Inc(x_ptr, 8);
    Inc(w_ptr, 8);
  end;
end;

{$ASMMODE INTEL}
function Float_DotProduct32_AVX2(const x_base, w_base: PSingle): Single; assembler;
asm
  // Load base pointers into general-purpose registers to use as iterators.
  mov rax, x_base
  mov rdx, w_base

  // --- Unrolled Loop ---

  // Block 1 (offset 0)
  vmovups ymm0, YMMWORD PTR [rax]
  vmulps  ymm0, ymm0, YMMWORD PTR [rdx]

  // Block 2 (offset 32)
  // Manually advance the pointers instead of using an offset in the instruction.
  add rax, 32
  add rdx, 32
  vmovups ymm1, YMMWORD PTR [rax]
  vmulps  ymm1, ymm1, YMMWORD PTR [rdx]

  // Block 3 (offset 64)
  add rax, 32
  add rdx, 32
  vmovups ymm2, YMMWORD PTR [rax]
  vmulps  ymm2, ymm2, YMMWORD PTR [rdx]

  // Block 4 (offset 96)
  add rax, 32
  add rdx, 32
  vmovups ymm3, YMMWORD PTR [rax]
  vmulps  ymm3, ymm3, YMMWORD PTR [rdx]

  // --- Summation Tree ---
  vaddps ymm0, ymm0, ymm1
  vaddps ymm2, ymm2, ymm3
  vaddps ymm0, ymm0, ymm2

  // --- Horizontal Sum ---
  vextractf128 xmm1, ymm0, 1
  vaddps xmm0, xmm0, xmm1
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0

  vzeroupper
end;

// Standard dot product implementation
function DotProduct(a, b: PSingle; dim: Integer): Single;
var
  i: Integer;
begin
  Result := 0;
  for i := 0 to dim - 1 do
    Result := Result + (a + i)^ * (b + i)^;
end;

function DotProduct_Hybrid(a, b: PSingle; dim: Integer): Single;
const
  BLOCK = 32;                       // elements processed per AVX2 call
var
  blocks, tail : Integer;
begin
  Result := 0;

  for blocks := 0 to (dim div block)-1 do
  begin
    Result := Result + Float_DotProduct32_AVX2(a, b);
    Inc(a, BLOCK);
    Inc(b, BLOCK);
  end;

  // scalar clean-up for the last <32 elements
  for tail := 0 to (dim mod BLOCK)-1 do
  begin
    Result := Result + a^ * b^;
    Inc(a);  Inc(b);
  end;
end;

{$ASMMODE INTEL}
function Int8_DotProduct64_AVX2(const x_base, w_base: PShortInt): LongInt; assembler;
asm
  // This function computes the dot product of two vectors of 64 signed 8-bit integers.
  // It requires AVX2 support.
  // We use ymm4 as our 256-bit accumulator, which holds 8 x 32-bit integer sums.
  vpxor ymm4, ymm4, ymm4       // Zero out the accumulator register ymm4

  // Load base pointers into registers
  mov rax, x_base
  mov rdx, w_base

  // --- Process first 16 bytes (elements 0-15) ---
  vpmovsxbw ymm0, [rax]      // Load 16 bytes from x and sign-extend to 16-bit words in ymm0
  vpmovsxbw ymm1, [rdx]      // Load 16 bytes from w and sign-extend to 16-bit words in ymm1
  vpmaddwd ymm2, ymm0, ymm1  // Multiply 16-bit words, then add adjacent 32-bit products.
  vpaddd ymm4, ymm4, ymm2    // Add the partial products to our main accumulator.

  // --- Process second 16 bytes (elements 16-31) ---
  add rax, 16
  add rdx, 16
  vpmovsxbw ymm0, [rax]
  vpmovsxbw ymm1, [rdx]
  vpmaddwd ymm2, ymm0, ymm1
  vpaddd ymm4, ymm4, ymm2

  // --- Process third 16 bytes (elements 32-47) ---
  add rax, 16
  add rdx, 16
  vpmovsxbw ymm0, [rax]
  vpmovsxbw ymm1, [rdx]
  vpmaddwd ymm2, ymm0, ymm1
  vpaddd ymm4, ymm4, ymm2

  // --- Process fourth 16 bytes (elements 48-63) ---
  add rax, 16
  add rdx, 16
  vpmovsxbw ymm0, [rax]
  vpmovsxbw ymm1, [rdx]
  vpmaddwd ymm2, ymm0, ymm1
  vpaddd ymm4, ymm4, ymm2

  // At this point, ymm4 contains 8 partial sums (each a sum of 8 products).
  // Now we need to sum these 8 dword values together to get the final result.

  // --- Horizontal Summation of the accumulator (ymm4) ---
  // Add the upper 128 bits of ymm4 to the lower 128 bits.
  vextracti128 xmm0, ymm4, 1    // xmm0 = upper 128 bits of ymm4
  vpaddd xmm4, xmm4, xmm0       // xmm4 = lower 128 bits + upper 128 bits.
                                // Result is in the lower 128 bits (xmm4). We now have 4 sums.

  // Horizontal add the remaining 4 dwords in xmm4
  vphaddd xmm4, xmm4, xmm4      // [d0,d1,d2,d3] -> [d0+d1, d2+d3, d0+d1, d2+d3]
  vphaddd xmm4, xmm4, xmm4      // [s0,s1,s0,s1] -> [s0+s1, s0+s1, s0+s1, s0+s1]

  // The final sum is now in the lowest 32 bits of xmm4.
  vmovd eax, xmm4               // Move the 32-bit result into eax

  // It's good practice to clear the upper state of YMM registers after use
  // to avoid performance penalties when mixing AVX and legacy SSE code.
  vzeroupper
end;


// Updated MatMul procedure
procedure TInt8QuantizedTensor.MatMul(xout: PSingle; const w: TInt8QuantizedTensor; n, d: longint);
var
  i, j: longint; // k is no longer needed
  val: single;
  ival: longint;
  x_base, w_base: PShortInt;
  x_scales, w_scales: PSingle;
  group_scale: single;
  groups: integer;
begin
  // GS is a constant, assumed to be 64
  groups := n div GS;
  for i := 0 to d - 1 do
  begin
    val := 0;
    w_base := w.q + (i * n);
    w_scales := w.s + (i * groups);
    x_base := self.q;
    x_scales := self.s;

    for j := 0 to groups - 1 do
    begin
      // The inner loop over k is replaced by a single call to the 64-element dot product function.
      //ival := Int8_DotProduct(x_base, w_base);
      ival := Int8_DotProduct64_AVX2(x_base, w_base);

      group_scale := x_scales^ * w_scales^;
      val += ival * group_scale;

      // Advance pointers to the next group of 64 elements for the next iteration.
      Inc(x_base, GS);
      Inc(w_base, GS);
      Inc(x_scales);
      Inc(w_scales);
    end;

    xout^ := val;
    Inc(xout);
  end;
end;

end.
