{ Tensor unit for Qwen-3 model inference in FreePascal }
unit Tensor_Unit;

{$mode objfpc}{$H+}
{$modeswitch advancedrecords}

interface

uses
  SysUtils,
  Classes,
  Math;

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

{ Memory allocation helpers }
function SafeGetMem(Size: PtrUInt): Pointer;

{ Matrix multiplication for quantized tensors }

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

{ Matrix multiplication }
{procedure TInt8QuantizedTensor.MatMul(xout: PSingle; const w: TInt8QuantizedTensor; n, d: longint);
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
    x_base := self.q;
    x_scales := self.s;
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
end;  }

function Int8_DotProduct8(const x_base, w_base: PShortInt): LongInt; assembler; inline;
asm
  // Load 8 bytes from x_base
  movq xmm0, [x_base]

  // Load 8 bytes from w_base
  movq xmm1, [w_base]

  // Convert signed 8-bit integers to 16-bit integers
  pmovsxbw xmm0, xmm0     // Sign-extend x_base bytes to words
  pmovsxbw xmm1, xmm1     // Sign-extend w_base bytes to words

  // Multiply corresponding 16-bit elements
  pmullw xmm0, xmm1

  // Horizontal add to sum all 8 products
  // First, add adjacent pairs (8 elements -> 4 elements)
  phaddw xmm0, xmm0

  // Add adjacent pairs again (4 elements -> 2 elements)
  phaddw xmm0, xmm0

  // Add the final pair and extract to 32-bit result
  phaddw xmm0, xmm0

  // Extract the lower 16 bits and sign-extend to 32-bit
  pextrw eax, xmm0, 0
  movsx eax, ax
end;

function Int8_DotProduct64(const x_base, w_base: PShortInt): LongInt; assembler; inline;
asm
  // This function computes the dot product of two vectors of 64 signed 8-bit integers.
  // It requires AVX2 support.
  // We use ymm4 as our 256-bit accumulator, which holds 8 x 32-bit integer sums.
  vpxor ymm4, ymm4, ymm4       // Zero out the accumulator register ymm4

  // --- Process first 16 bytes (elements 0-15) ---
  vpmovsxbw ymm0, [x_base]      // Load 16 bytes from x and sign-extend to 16-bit words in ymm0
  vpmovsxbw ymm1, [w_base]      // Load 16 bytes from w and sign-extend to 16-bit words in ymm1
  vpmaddwd ymm2, ymm0, ymm1     // Multiply 16-bit words, then add adjacent 32-bit products.
                                // ymm2 now contains 8 partial dot products of 2 elements each.
  vpaddd ymm4, ymm4, ymm2       // Add the partial products to our main accumulator.

  // --- Process second 16 bytes (elements 16-31) ---
  vpmovsxbw ymm0, [x_base+16]
  vpmovsxbw ymm1, [w_base+16]
  vpmaddwd ymm2, ymm0, ymm1
  vpaddd ymm4, ymm4, ymm2

  // --- Process third 16 bytes (elements 32-47) ---
  vpmovsxbw ymm0, [x_base+32]
  vpmovsxbw ymm1, [w_base+32]
  vpmaddwd ymm2, ymm0, ymm1
  vpaddd ymm4, ymm4, ymm2

  // --- Process fourth 16 bytes (elements 48-63) ---
  vpmovsxbw ymm0, [x_base+48]
  vpmovsxbw ymm1, [w_base+48]
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
      ival := Int8_DotProduct64(x_base, w_base);

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
