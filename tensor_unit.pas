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
procedure MatMul(xout: PSingle; const x, w: TInt8QuantizedTensor; n, d: longint);

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
procedure MatMul(xout: PSingle; const x, w: TInt8QuantizedTensor; n, d: longint);
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
    x_base := x.q;
    x_scales := x.s;
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
end;

end. 